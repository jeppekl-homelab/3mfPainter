"""GPU viewer window (moderngl + glfw).

Renders the mesh with flat per-face shading. The vertex layout already carries
a per-face id, so the next step (offscreen ID-buffer picking for painting)
plugs straight in.
"""

from __future__ import annotations

import ctypes

import moderngl
import glfw
import glm
import numpy as np
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd

from . import mesh as mesh_io
from .camera import OrbitCamera
from .mesh import PaintMesh
from .painting import PALETTE, PaintState, Picker


def mat_bytes(m: glm.mat4) -> bytes:
    """pyGLM 2.8 ligger row-major i hukommelsen; GLSL forventer column-major."""
    return bytes(glm.transpose(m))

VERTEX_SHADER = """
#version 410
uniform mat4 u_mvp;
uniform mat4 u_model;

in vec3 in_position;
in vec3 in_normal;
in uint in_face_id;

out vec3 v_normal;
out vec3 v_world_pos;
flat out uint v_face_id;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_normal = mat3(u_model) * in_normal;
    v_world_pos = (u_model * vec4(in_position, 1.0)).xyz;
    v_face_id = in_face_id;
}
"""

FRAGMENT_SHADER = """
#version 410
uniform vec3 u_camera_pos;
uniform vec3 u_palette[9];
uniform usampler2D u_face_colors;

in vec3 v_normal;
in vec3 v_world_pos;
flat in uint v_face_id;

out vec4 frag_color;

void main() {
    int fid = int(v_face_id);
    uint cidx = texelFetch(u_face_colors, ivec2(fid % 4096, fid / 4096), 0).r;
    vec3 base = u_palette[min(cidx, 8u)];

    vec3 n = normalize(v_normal);
    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
    // headlight + soft fill from above
    float diffuse = max(dot(n, view_dir), 0.0) * 0.75
                  + max(n.y, 0.0) * 0.15;
    vec3 color = base * (0.15 + diffuse);
    float spec = pow(max(dot(n, view_dir), 0.0), 48.0) * 0.25;
    frag_color = vec4(color + spec, 1.0);
}
"""


class Viewer:
    def __init__(self, mesh: PaintMesh, title: str = "3mf Painter"):
        if not glfw.init():
            raise RuntimeError("glfw could not be initialized")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)

        self.window = glfw.create_window(1600, 1000, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )

        self.current_color = 1
        self.brush_radius = 20  # px
        self._mvp_bytes = bytes(glm.mat4(1.0))
        self._vbo: moderngl.Buffer | None = None
        self._open_dialog: pfd.open_file | None = None
        self.set_mesh(mesh)

        self._last_cursor = None
        self._rotating = False
        self._panning = False
        self._painting = False
        # Vores callbacks sættes FØR imgui-backenden initialiseres, så den
        # kæder videre til dem når imgui ikke selv bruger inputtet.
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

        imgui.create_context()
        imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        window_addr = ctypes.cast(self.window, ctypes.c_void_p).value
        imgui.backends.glfw_init_for_opengl(window_addr, True)
        imgui.backends.opengl3_init("#version 410")

    def set_mesh(self, mesh: PaintMesh) -> None:
        """(Re)build all GPU buffers for a newly loaded mesh."""
        if self._vbo is not None:
            self.vao.release()
            self.picker.vao.release()
            self.paint.texture.release()
            self._vbo.release()

        self.mesh = mesh
        self.camera = OrbitCamera(mesh.center, mesh.radius)
        self._vbo = self.ctx.buffer(
            np.hstack(
                [
                    mesh.positions,
                    mesh.normals,
                    mesh.face_ids.reshape(-1, 1).view("f4"),
                ]
            ).astype("f4").tobytes()
        )
        # GLSL-compileren fjerner in_face_id indtil den faktisk bruges —
        # bind den kun hvis den findes i programmet.
        if "in_face_id" in self.prog:
            fmt = [(self._vbo, "3f 3f 1u", "in_position", "in_normal", "in_face_id")]
        else:
            fmt = [(self._vbo, "3f 3f 4x", "in_position", "in_normal")]
        self.vao = self.ctx.vertex_array(self.prog, fmt)
        self.paint = PaintState(self.ctx, mesh.n_faces)
        self.picker = Picker(self.ctx, self._vbo)

    # --- input callbacks ---------------------------------------------------
    def _on_mouse_button(self, window, button, action, mods):
        if imgui.get_io().want_capture_mouse:
            return
        pressed = action == glfw.PRESS
        shift = bool(mods & glfw.MOD_SHIFT)
        if button == glfw.MOUSE_BUTTON_LEFT:
            if shift:
                self._panning = pressed
            else:
                self._painting = pressed
                if pressed:
                    self._paint_at(*glfw.get_cursor_pos(window))
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._rotating = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._panning = pressed
        if pressed:
            self._last_cursor = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, x, y):
        if imgui.get_io().want_capture_mouse:
            self._painting = self._rotating = self._panning = False
            return
        if self._last_cursor is None:
            self._last_cursor = (x, y)
            return
        dx, dy = x - self._last_cursor[0], y - self._last_cursor[1]
        self._last_cursor = (x, y)
        if self._rotating:
            self.camera.rotate(dx, dy)
        elif self._panning:
            self.camera.pan(dx, dy)
        elif self._painting:
            self._paint_at(x, y)

    def _on_scroll(self, window, dx, dy):
        if imgui.get_io().want_capture_mouse:
            return
        if glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            self.brush_radius = int(max(2, min(200, self.brush_radius * 1.15 ** dy)))
            self._update_title()
        else:
            self.camera.zoom(dy)

    def _on_key(self, window, key, scancode, action, mods):
        if imgui.get_io().want_capture_keyboard or action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif glfw.KEY_0 <= key <= glfw.KEY_8:
            self.current_color = key - glfw.KEY_0  # 0 = viskelæder/umalet
            self._update_title()
        elif key == glfw.KEY_C:
            self.paint.clear()

    def _paint_at(self, x: float, y: float) -> None:
        w, h = glfw.get_framebuffer_size(self.window)
        ww, wh = glfw.get_window_size(self.window)
        if h == 0 or wh == 0:
            return
        scale = h / wh  # DPI-skalering: window-koordinater -> framebuffer-pixels
        face_ids = self.picker.pick(
            self._mvp_bytes, (w, h), (x * scale, y * scale),
            int(self.brush_radius * scale),
        )
        self.paint.set_faces(face_ids, self.current_color)

    def _update_title(self) -> None:
        glfw.set_window_title(
            self.window,
            f"3mf Painter — farve {self.current_color}, pensel {self.brush_radius}px",
        )

    # --- main loop -----------------------------------------------------------
    def run(self) -> None:
        from .compute import get_profile

        print(f"Renderer: {self.ctx.info['GL_RENDERER']}")
        print(f"Faces:    {self.mesh.n_faces:,}")
        get_profile()  # detektér og log compute-device ved opstart
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self._poll_file_dialog()
            self._render()
            self._render_ui()
            glfw.swap_buffers(self.window)
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        glfw.terminate()

    # --- UI -------------------------------------------------------------
    def _render_ui(self) -> None:
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                clicked, _ = imgui.menu_item("Import STL...", "", False)
                if clicked and self._open_dialog is None:
                    self._open_dialog = pfd.open_file(
                        "Import STL",
                        "",
                        ["STL-filer", "*.stl", "Alle mesh-filer", "*.stl *.3mf *.obj"],
                    )
                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def _poll_file_dialog(self) -> None:
        if self._open_dialog is None or not self._open_dialog.ready():
            return
        paths = self._open_dialog.result()
        self._open_dialog = None
        if not paths:
            return
        path = paths[0]
        print(f"Importerer {path} ...")
        try:
            self.set_mesh(mesh_io.load(path))
        except Exception as exc:  # ødelagt/ukendt fil må ikke crashe appen
            print(f"Kunne ikke indlæse {path}: {exc}")
            return
        print(f"Faces:    {self.mesh.n_faces:,}")

    def _render(self) -> None:
        w, h = glfw.get_framebuffer_size(self.window)
        if h == 0:
            return
        self.ctx.screen.use()
        # ImGui's backend ændrer GL-state direkte (depth test/culling slås
        # fra) uden om moderngl's cache — gen-håndhæv den hver frame.
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.12, 0.13, 0.15)

        model = glm.mat4(1.0)
        mvp = self.camera.proj_matrix(w / h) * self.camera.view_matrix() * model
        self._mvp_bytes = mat_bytes(mvp)
        self.prog["u_mvp"].write(self._mvp_bytes)
        self.prog["u_model"].write(mat_bytes(model))
        self.prog["u_camera_pos"].value = tuple(self.camera.position())
        self.prog["u_palette"].write(PALETTE.tobytes())
        self.paint.texture.use(location=0)
        self.prog["u_face_colors"].value = 0
        self.vao.render(moderngl.TRIANGLES)
