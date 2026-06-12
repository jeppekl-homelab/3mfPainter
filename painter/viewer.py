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

from . import export3mf
from . import mesh as mesh_io
from .camera import OrbitCamera
from .mesh import PaintMesh
from .painting import Palette, PaintState, Picker, HoverHighlight
from .segmentation import Segmenter
from .symmetry import MirrorMap, find_mirror


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
uniform vec3 u_palette[6];
uniform usampler2D u_face_colors;
uniform usampler2D u_hover;

in vec3 v_normal;
in vec3 v_world_pos;
flat in uint v_face_id;

out vec4 frag_color;

void main() {
    int fid = int(v_face_id);
    ivec2 texel = ivec2(fid % 4096, fid / 4096);
    uint cidx = texelFetch(u_face_colors, texel, 0).r;
    vec3 base = u_palette[min(cidx, 5u)];

    vec3 n = normalize(v_normal);
    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
    // headlight + soft fill from above
    float diffuse = max(dot(n, view_dir), 0.0) * 0.75
                  + max(n.y, 0.0) * 0.15;
    vec3 color = base * (0.15 + diffuse);
    float spec = pow(max(dot(n, view_dir), 0.0), 48.0) * 0.25;
    color += spec;

    // fill-mode hover: lys den region op som et klik ville farve
    if (texelFetch(u_hover, texel, 0).r != 0u) {
        color = mix(color, vec3(1.0, 1.0, 0.6), 0.35);
    }
    frag_color = vec4(color, 1.0);
}
"""

# Debug-overlay: tegner det detekterede spejlplan som en gennemsigtig flade
PLANE_VERTEX_SHADER = """
#version 410
uniform mat4 u_mvp;
in vec3 in_position;
void main() { gl_Position = u_mvp * vec4(in_position, 1.0); }
"""

PLANE_FRAGMENT_SHADER = """
#version 410
uniform vec4 u_color;
out vec4 frag_color;
void main() { frag_color = u_color; }
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
        self.plane_prog = self.ctx.program(
            vertex_shader=PLANE_VERTEX_SHADER, fragment_shader=PLANE_FRAGMENT_SHADER
        )
        self._plane_vbo: moderngl.Buffer | None = None
        self._plane_vao: moderngl.VertexArray | None = None
        self.show_plane = True  # vis spejlplanet når mirror er aktivt

        self.palette = Palette()
        self.current_color = 1
        self.brush_radius = 20  # px
        self.fill_mode = False
        self.seg_angle = 25.0  # max dihedral-vinkel inden for en region
        self.mirror_mode = False
        self._mvp_bytes = bytes(glm.mat4(1.0))
        self._vbo: moderngl.Buffer | None = None
        self._open_dialog: pfd.open_file | None = None
        self._save_dialog: pfd.save_file | None = None
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
            self.hover.texture.release()
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
        self.hover = HoverHighlight(self.ctx, mesh.n_faces)
        self._segmenter: Segmenter | None = None  # bygges dovent ved første fill
        self._mirror: MirrorMap | None = None     # findes dovent ved første brug
        self._mirror_searched = False
        if self._plane_vao is not None:           # spejlplan-overlay bygges på ny
            self._plane_vao.release()
            self._plane_vbo.release()
            self._plane_vao = None

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
                if pressed and not self._painting:
                    self._painting = True
                    self.hover.clear()
                    self.paint.begin_stroke()
                    self._paint_at(*glfw.get_cursor_pos(window))
                elif not pressed and self._painting:
                    self._painting = False
                    self.paint.end_stroke()
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._rotating = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._panning = pressed
        if pressed:
            self._last_cursor = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, x, y):
        if imgui.get_io().want_capture_mouse:
            if self._painting:
                self._painting = False
                self.paint.end_stroke()
            self._rotating = self._panning = False
            self.hover.clear()
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
        elif self.fill_mode:
            self._hover_at(x, y)
        else:
            self.hover.clear()

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
        ctrl = bool(mods & glfw.MOD_CONTROL)
        shift = bool(mods & glfw.MOD_SHIFT)
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif ctrl and key == glfw.KEY_Z and shift:
            self.paint.redo()
        elif ctrl and key == glfw.KEY_Z:
            self.paint.undo()
        elif ctrl and key == glfw.KEY_Y:
            self.paint.redo()
        elif glfw.KEY_0 <= key <= glfw.KEY_5:
            self.current_color = key - glfw.KEY_0  # 0 = viskelæder/umalet
            self._update_title()
        elif key == glfw.KEY_F:
            self.fill_mode = not self.fill_mode
            if not self.fill_mode:
                self.hover.clear()
            self._update_title()
        elif key == glfw.KEY_M:
            self.mirror_mode = not self.mirror_mode
        elif key == glfw.KEY_C:
            self.paint.clear()

    def _paint_at(self, x: float, y: float) -> None:
        w, h = glfw.get_framebuffer_size(self.window)
        ww, wh = glfw.get_window_size(self.window)
        if h == 0 or wh == 0:
            return
        scale = h / wh  # DPI-skalering: window-koordinater -> framebuffer-pixels
        radius = 2 if self.fill_mode else int(self.brush_radius * scale)
        face_ids = self._dab((w, h), (x * scale, y * scale), radius)
        self.paint.set_faces(face_ids, self.current_color)

    def _dab(
        self, fb_size: tuple[int, int], cursor: tuple[float, float], radius: int
    ) -> np.ndarray:
        """One brush dab: pick (+ segment expand) and, in mirror mode, the
        IDENTICAL pick through the plane-reflected camera. Mirroring the brush
        instead of the painted faces gives both sides the same first-order
        discretization of the dab — no correspondence, no sawtooth."""
        face_ids = self.picker.pick(self._mvp_bytes, fb_size, cursor, radius)
        if self.fill_mode:
            face_ids = self._expand_to_segments(face_ids)
        if self.mirror_mode:
            mirror = self._get_mirror()
            if mirror is not None:
                mids = self.picker.pick(
                    self._mirrored_mvp_bytes(mirror), fb_size, cursor, radius,
                    flip_winding=True,
                )
                if self.fill_mode:
                    mids = self._expand_to_segments(mids)
                face_ids = np.union1d(face_ids, mids)
        return face_ids

    def _mirrored_mvp_bytes(self, mirror: MirrorMap) -> bytes:
        """MVP composed with the world-space reflection (column-major bytes)."""
        m = np.frombuffer(self._mvp_bytes, dtype="f4").reshape(4, 4, order="F")
        mr = m.astype("f8") @ mirror.reflection_matrix()
        return mr.astype("f4").T.tobytes()

    def _hover_at(self, x: float, y: float) -> None:
        """Highlight the region a fill-mode click would paint (incl. mirror)."""
        w, h = glfw.get_framebuffer_size(self.window)
        ww, wh = glfw.get_window_size(self.window)
        if h == 0 or wh == 0:
            return
        scale = h / wh
        face_ids = self._dab((w, h), (x * scale, y * scale), 2)
        self.hover.set(face_ids)

    def _get_mirror(self) -> MirrorMap | None:
        if not self._mirror_searched:
            self._mirror_searched = True
            self._mirror = find_mirror(self.mesh)
            if self._mirror is None:
                self.mirror_mode = False
        return self._mirror

    def _build_plane(self) -> None:
        """Build a translucent quad lying in the detected mirror plane."""
        m = self._mirror
        n = np.asarray(m.normal, dtype="f8")
        n = n / max(np.linalg.norm(n), 1e-12)
        o = np.asarray(m.origin, dtype="f8")
        # to akser i planet (vilkårligt orienteret, vinkelret på normalen)
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        r = self.mesh.radius * 1.5
        c = [o + (-u - v) * r, o + (u - v) * r, o + (u + v) * r, o + (-u + v) * r]
        quad = np.array([c[0], c[1], c[2], c[0], c[2], c[3]], dtype="f4")
        if self._plane_vao is not None:
            self._plane_vao.release()
            self._plane_vbo.release()
        self._plane_vbo = self.ctx.buffer(quad.tobytes())
        self._plane_vao = self.ctx.vertex_array(
            self.plane_prog, [(self._plane_vbo, "3f", "in_position")]
        )

    def _expand_to_segments(self, face_ids: np.ndarray) -> np.ndarray:
        if len(face_ids) == 0:
            return face_ids
        if self._segmenter is None:
            self._segmenter = Segmenter(self.mesh)
        labels = self._segmenter.labels(self.seg_angle)
        hit_labels = np.unique(labels[face_ids])
        return np.flatnonzero(np.isin(labels, hit_labels))

    def _update_title(self) -> None:
        mode = "fill" if self.fill_mode else f"pensel {self.brush_radius}px"
        glfw.set_window_title(
            self.window,
            f"3mf Painter — farve {self.current_color}, {mode}",
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
                clicked, _ = imgui.menu_item("Open... (STL/OBJ/3MF)", "", False)
                if clicked and self._open_dialog is None:
                    self._open_dialog = pfd.open_file(
                        "Open model or project",
                        "",
                        ["Mesh & projekter", "*.stl *.3mf *.obj", "Alle filer", "*"],
                    )
                clicked, _ = imgui.menu_item("Save project (3MF)...", "", False)
                if clicked and self._save_dialog is None:
                    self._save_dialog = pfd.save_file(
                        "Save project (3MF)", "model.3mf", ["3MF-filer", "*.3mf"]
                    )
                imgui.end_menu()
            if imgui.begin_menu("Edit"):
                clicked, _ = imgui.menu_item("Undo", "Ctrl+Z", False)
                if clicked:
                    self.paint.undo()
                clicked, _ = imgui.menu_item("Redo", "Ctrl+Y", False)
                if clicked:
                    self.paint.redo()
                imgui.end_menu()
            imgui.end_main_menu_bar()

        self._tool_window()

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def _tool_window(self) -> None:
        # NB: ImGui's standardfont har ikke æ/ø/å — hold labels på engelsk/ASCII
        imgui.set_next_window_pos((10, 35), imgui.Cond_.first_use_ever)
        imgui.begin("Tools", None, imgui.WindowFlags_.always_auto_resize)

        imgui.text("Colors")
        n = self.palette.size
        for i in range(n):
            imgui.push_id(i)
            r, g, b = self.palette.colors[i]
            flags = imgui.ColorEditFlags_.no_tooltip
            if imgui.color_button("##color", imgui.ImVec4(r, g, b, 1.0), flags, (28, 28)):
                self.current_color = i
                self._update_title()
            if i == self.current_color:
                pmin = imgui.get_item_rect_min()
                pmax = imgui.get_item_rect_max()
                imgui.get_window_draw_list().add_rect(
                    pmin, pmax,
                    imgui.color_convert_float4_to_u32(imgui.ImVec4(1, 1, 1, 1)),
                    rounding=0.0, thickness=2.0,
                )
            imgui.pop_id()
            if i != n - 1:
                imgui.same_line()
        imgui.text("(0 = eraser/unpainted)")
        if self.current_color != 0:
            col = self.palette.colors[self.current_color]
            imgui.set_next_item_width(200)
            changed, new = imgui.color_edit3(
                f"Slot {self.current_color}",
                (float(col[0]), float(col[1]), float(col[2])),
            )
            if changed:
                self.palette.colors[self.current_color] = new
        imgui.separator()

        changed, self.fill_mode = imgui.checkbox("Fill mode (F)", self.fill_mode)
        if changed:
            if not self.fill_mode:
                self.hover.clear()
            self._update_title()
        imgui.set_next_item_width(160)
        _, self.seg_angle = imgui.slider_float(
            "Edge angle", self.seg_angle, 5.0, 90.0, "%.0f deg"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Max dihedral angle inside a region.\n"
                "Lower = more, smaller regions."
            )
        imgui.set_next_item_width(160)
        _, self.brush_radius = imgui.slider_int(
            "Brush size", self.brush_radius, 2, 200, "%d px"
        )
        imgui.separator()

        changed, self.mirror_mode = imgui.checkbox("Mirror (M)", self.mirror_mode)
        if changed and self.mirror_mode:
            self._get_mirror()  # detektér med det samme ved aktivering
        if self.mirror_mode and self._mirror is not None:
            imgui.same_line()
            imgui.text_disabled(
                f"plane {self._mirror.axis_label}, {self._mirror.match:.0%}"
            )
        elif self._mirror_searched and self._mirror is None:
            imgui.same_line()
            imgui.text_disabled("no symmetry found")
        if self.mirror_mode and self._mirror is not None:
            _, self.show_plane = imgui.checkbox("Show mirror plane", self.show_plane)
        imgui.end()

    def _poll_file_dialog(self) -> None:
        if self._open_dialog is not None and self._open_dialog.ready():
            paths = self._open_dialog.result()
            self._open_dialog = None
            if paths:
                self._open(paths[0])

        if self._save_dialog is not None and self._save_dialog.ready():
            path = self._save_dialog.result()
            self._save_dialog = None
            if path:
                if not path.lower().endswith(".3mf"):
                    path += ".3mf"
                try:
                    export3mf.export(
                        self.mesh, self.paint.face_colors, self.palette.colors, path
                    )
                except Exception as exc:
                    print(f"Eksport fejlede: {exc}")

    def _open(self, path: str) -> None:
        """Load a mesh; a .3mf with paint data restores colors + palette too."""
        print(f"Åbner {path} ...")
        if path.lower().endswith(".3mf"):
            try:
                mesh, face_colors, palette_cols = export3mf.load_project(path)
                self.set_mesh(mesh)
                if palette_cols is not None:
                    self.palette.set_slots(palette_cols)
                self.paint.load_colors(face_colors)
                return
            except Exception as exc:
                # fremmed/ukurant 3MF: fald tilbage til ren geometri
                print(f"Kunne ikke læse maledata ({exc}); indlæser kun geometri")
        try:
            self.set_mesh(mesh_io.load(path))
            print(f"Faces:    {self.mesh.n_faces:,}")
        except Exception as exc:  # ødelagt fil må ikke crashe appen
            print(f"Kunne ikke indlæse {path}: {exc}")

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
        self.prog["u_palette"].write(self.palette.colors.tobytes())
        self.paint.texture.use(location=0)
        self.prog["u_face_colors"].value = 0
        self.hover.texture.use(location=1)
        self.prog["u_hover"].value = 1
        self.vao.render(moderngl.TRIANGLES)

        # debug: tegn det detekterede spejlplan som en gennemsigtig blå flade
        if self.show_plane and self.mirror_mode and self._mirror is not None:
            if self._plane_vao is None:
                self._build_plane()
            self.plane_prog["u_mvp"].write(self._mvp_bytes)
            self.plane_prog["u_color"].value = (0.25, 0.6, 1.0, 0.28)
            self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND)  # dobbeltsidet
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._plane_vao.render(moderngl.TRIANGLES)
