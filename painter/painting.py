"""Per-face paint state + GPU picking.

Picking works by rendering every face's ID into an offscreen R32UI buffer.
The brush then reads a small square around the cursor and assigns the current
color to every face-ID found inside the brush circle — pixel-precise and
independent of mesh size.
"""

from __future__ import annotations

import numpy as np
import moderngl

# Filament-agtig palette; index 0 er "umalet" (grundfarve)
PALETTE = np.array(
    [
        [0.78, 0.80, 0.85],  # 0: umalet
        [0.90, 0.12, 0.14],  # 1: rød
        [0.13, 0.45, 0.85],  # 2: blå
        [0.16, 0.65, 0.27],  # 3: grøn
        [0.95, 0.77, 0.06],  # 4: gul
        [0.93, 0.45, 0.13],  # 5: orange
        [0.55, 0.20, 0.70],  # 6: lilla
        [0.10, 0.10, 0.10],  # 7: sort
        [0.97, 0.97, 0.97],  # 8: hvid
    ],
    dtype="f4",
)

ID_VERTEX_SHADER = """
#version 410
uniform mat4 u_mvp;
in vec3 in_position;
in uint in_face_id;
flat out uint v_face_id;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_face_id = in_face_id;
}
"""

ID_FRAGMENT_SHADER = """
#version 410
flat in uint v_face_id;
out uint frag_id;
void main() {
    frag_id = v_face_id + 1u;  // 0 = baggrund
}
"""

# Teksturbredde til face->farve opslag (face_id -> (id % W, id / W))
COLOR_TEX_WIDTH = 4096


MAX_UNDO = 100


class PaintState:
    """Holds per-face color indices and mirrors them into a GPU texture.

    Undo/redo: a stroke (mouse-down to mouse-up) is recorded as a diff of the
    faces whose color actually changed, with their previous color.
    """

    def __init__(self, ctx: moderngl.Context, n_faces: int):
        self.ctx = ctx
        self.n_faces = n_faces
        self.face_colors = np.zeros(n_faces, dtype="u1")

        self._undo: list[tuple[np.ndarray, np.ndarray]] = []
        self._redo: list[tuple[np.ndarray, np.ndarray]] = []
        # -1 = ikke berørt i aktuel stroke; ellers farven før strokens start
        self._stroke_old = np.full(n_faces, -1, dtype="i2")

        height = (n_faces + COLOR_TEX_WIDTH - 1) // COLOR_TEX_WIDTH
        self._tex_shape = (COLOR_TEX_WIDTH, max(height, 1))
        self.texture = ctx.texture(self._tex_shape, 1, dtype="u1")
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._upload()

    # --- strokes ----------------------------------------------------------
    def begin_stroke(self) -> None:
        self._stroke_old.fill(-1)

    def end_stroke(self) -> None:
        changed = np.flatnonzero(self._stroke_old >= 0)
        if len(changed) == 0:
            return
        self._undo.append((changed, self._stroke_old[changed].astype("u1")))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def set_faces(self, face_ids: np.ndarray, color_idx: int) -> None:
        if len(face_ids) == 0:
            return
        # registrér gammel farve for faces der ændres første gang i denne stroke
        changing = face_ids[self.face_colors[face_ids] != color_idx]
        unrecorded = changing[self._stroke_old[changing] < 0]
        self._stroke_old[unrecorded] = self.face_colors[unrecorded]

        self.face_colors[face_ids] = color_idx
        self._upload()

    def clear(self) -> None:
        self.begin_stroke()
        self.set_faces(np.arange(self.n_faces), 0)
        self.end_stroke()

    # --- undo/redo --------------------------------------------------------
    def undo(self) -> None:
        self._swap(self._undo, self._redo)

    def redo(self) -> None:
        self._swap(self._redo, self._undo)

    def _swap(self, source: list, target: list) -> None:
        if not source:
            return
        faces, colors = source.pop()
        target.append((faces, self.face_colors[faces].copy()))
        self.face_colors[faces] = colors
        self._upload()

    def _upload(self) -> None:
        w, h = self._tex_shape
        padded = np.zeros(w * h, dtype="u1")
        padded[: self.n_faces] = self.face_colors
        self.texture.write(padded.tobytes())


class Picker:
    """Offscreen face-ID buffer for pixel-precise brush picking."""

    def __init__(self, ctx: moderngl.Context, vbo: moderngl.Buffer):
        self.ctx = ctx
        self.prog = ctx.program(
            vertex_shader=ID_VERTEX_SHADER, fragment_shader=ID_FRAGMENT_SHADER
        )
        # samme VBO som hovedrenderingen; normalen springes over med 12x padding
        self.vao = ctx.vertex_array(
            self.prog, [(vbo, "3f 12x 1u", "in_position", "in_face_id")]
        )
        self._size = (0, 0)
        self._fbo: moderngl.Framebuffer | None = None

    def _ensure_fbo(self, size: tuple[int, int]) -> moderngl.Framebuffer:
        if self._fbo is None or self._size != size:
            if self._fbo is not None:
                self._fbo.release()
            color = self.ctx.texture(size, 1, dtype="u4")
            depth = self.ctx.depth_renderbuffer(size)
            self._fbo = self.ctx.framebuffer(color, depth)
            self._size = size
        return self._fbo

    def pick(
        self,
        mvp_bytes: bytes,
        fb_size: tuple[int, int],
        cursor: tuple[float, float],
        radius_px: int,
    ) -> np.ndarray:
        """Render ID-pass and return unique face ids inside the brush circle."""
        w, h = fb_size
        fbo = self._ensure_fbo(fb_size)
        fbo.use()
        # depth test kan være slået fra af ImGui-backenden — håndhæv den
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear()
        self.prog["u_mvp"].write(mvp_bytes)
        self.vao.render(moderngl.TRIANGLES)

        cx, cy = int(cursor[0]), int(cursor[1])
        r = radius_px
        x0, y0 = max(cx - r, 0), max(cy - r, 0)
        x1, y1 = min(cx + r + 1, w), min(cy + r + 1, h)
        if x0 >= x1 or y0 >= y1:
            return np.empty(0, dtype="i8")

        # fbo.read har origin nederst-venstre; cursor er øverst-venstre
        gl_y0 = h - y1
        data = fbo.read(
            viewport=(x0, gl_y0, x1 - x0, y1 - y0), components=1, dtype="u4"
        )
        ids = np.frombuffer(data, dtype="u4").reshape(y1 - y0, x1 - x0)
        ids = ids[::-1]  # flip til skærm-koordinater (top-venstre origin)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        hits = np.unique(ids[circle])
        hits = hits[hits > 0].astype("i8") - 1
        return hits
