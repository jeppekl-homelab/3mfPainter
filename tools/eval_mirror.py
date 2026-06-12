"""Mirrored-painting quality harness (run from repo root).

Drives the REAL app paint path offscreen: Picker ID-buffer picks along a
simulated mouse stroke beside the symmetry plane, with the mirrored pick
(reflected camera, flipped winding) — exactly what the viewer does in mirror
mode. Quality metric: the mirrored patch should have the same boundary
complexity (adjacency edges crossing the patch border) as the source patch.
Correspondence-based mirroring inflated it 25-100%; brush mirroring keeps
them equal by construction.

Usage:
    python tools/eval_mirror.py [path/to/model.stl] [out.png]
Defaults to the chiral worst case: a stretched uv-sphere where reflection
flips every quad diagonal.
"""

import sys

import numpy as np
import moderngl
import glm
import trimesh
from PIL import Image

sys.path.insert(0, ".")

from painter import mesh as mesh_io                       # noqa: E402
from painter.camera import OrbitCamera                    # noqa: E402
from painter.mesh import _prepare                         # noqa: E402
from painter.painting import Palette, PaintState, Picker  # noqa: E402
from painter.symmetry import find_mirror                  # noqa: E402
from painter.viewer import (                              # noqa: E402
    VERTEX_SHADER, FRAGMENT_SHADER, mat_bytes,
)

W, H = 1100, 850


def boundary_edges(mesh, member_mask) -> int:
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    adj = np.asarray(tm.face_adjacency)
    return int((member_mask[adj[:, 0]] != member_mask[adj[:, 1]]).sum())


def main() -> None:
    if len(sys.argv) > 1:
        mesh = mesh_io.load(sys.argv[1])
    else:
        ell = trimesh.creation.uv_sphere(radius=1.0, count=[48, 40])
        ell.apply_scale([20.0, 20.0, 40.0])
        mesh = _prepare(ell)
    out = sys.argv[2] if len(sys.argv) > 2 else "eval_mirror.png"

    ctx = moderngl.create_context(standalone=True, require=410)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    vbo = ctx.buffer(
        np.hstack(
            [mesh.positions, mesh.normals, mesh.face_ids.reshape(-1, 1).view("f4")]
        ).astype("f4").tobytes()
    )
    vao = ctx.vertex_array(
        prog, [(vbo, "3f 3f 1u", "in_position", "in_normal", "in_face_id")]
    )
    paint = PaintState(ctx, mesh.n_faces)
    picker = Picker(ctx, vbo)
    mirror = find_mirror(mesh)
    assert mirror is not None, "modellen skal have et spejlplan"

    camera = OrbitCamera(mesh.center, mesh.radius)
    mvp = camera.proj_matrix(W / H) * camera.view_matrix()
    mvp_bytes = mat_bytes(mvp)
    m_np = np.frombuffer(mvp_bytes, dtype="f4").reshape(4, 4, order="F").astype("f8")
    mir_bytes = (m_np @ mirror.reflection_matrix()).astype("f4").T.tobytes()

    def project(p3):
        clip = mvp * glm.vec4(*p3.tolist(), 1.0)
        return ((clip.x / clip.w + 1) / 2 * W, (1 - (clip.y / clip.w + 1) / 2) * H)

    # strøg parallelt med planet, en anelse til den ene side, på den del af
    # overfladen der vender mod kameraet
    n, o = mirror.normal, mirror.origin
    cam = np.array(list(camera.position()), dtype="f8")
    view_dir = cam - o
    view_dir -= (view_dir @ n) * n
    if np.linalg.norm(view_dir) < 1e-6:
        view_dir = np.array([1.0, 0, 0])
    view_dir /= np.linalg.norm(view_dir)
    axis = np.cross(n, view_dir)

    r_model = mesh.radius
    src_total = np.zeros(mesh.n_faces, dtype=bool)
    mir_total = np.zeros(mesh.n_faces, dtype=bool)
    for t in np.linspace(-0.35, 0.35, 16):
        p = o + (view_dir * 0.85 + n * 0.45 + axis * t) * r_model * 0.7
        sx, sy = project(p)
        s_ids = picker.pick(mvp_bytes, (W, H), (sx, sy), 22)
        m_ids = picker.pick(mir_bytes, (W, H), (sx, sy), 22, flip_winding=True)
        src_total[s_ids] = True
        mir_total[m_ids] = True
        paint.set_faces(np.union1d(s_ids, m_ids), 1)

    only_mir = mir_total & ~src_total
    b_src = boundary_edges(mesh, src_total)
    b_mir = boundary_edges(mesh, only_mir)
    print(f"kilde: {src_total.sum()} faces, boundary {b_src} kanter")
    print(f"spejl: {only_mir.sum()} faces, boundary {b_mir} kanter")
    ratio = b_mir / max(b_src, 1)
    print(
        f"raggedness-ratio (spejl/kilde): {ratio:.2f}  "
        "(1.0 = perfekt; korrespondance-metoderne lå på 1.3-2.0)"
    )

    # render fra begge sider af planet, så kilde- og spejlstrøg kan sammenlignes
    fbo = ctx.framebuffer(ctx.texture((W, H), 4), ctx.depth_renderbuffer((W, H)))
    views = []
    import math
    for dyaw in (0.0, math.pi):
        camera.yaw += dyaw
        view_mvp = camera.proj_matrix(W / H) * camera.view_matrix()
        camera.yaw -= dyaw
        fbo.use()
        ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.clear(0.12, 0.13, 0.15)
        prog["u_mvp"].write(mat_bytes(view_mvp))
        prog["u_model"].write(mat_bytes(glm.mat4(1.0)))
        prog["u_camera_pos"].value = tuple(camera.position())
        prog["u_palette"].write(Palette().colors.tobytes())
        paint.texture.use(location=0)
        prog["u_face_colors"].value = 0
        vao.render(moderngl.TRIANGLES)
        views.append(
            Image.frombytes("RGBA", (W, H), fbo.read(components=4)).transpose(
                Image.FLIP_TOP_BOTTOM
            )
        )
    combo = Image.new("RGBA", (W * 2, H))
    combo.paste(views[0], (0, 0))
    combo.paste(views[1], (W, 0))
    combo.save(out)
    print(f"gemt {out} (venstre: kilde-side, højre: spejl-side)")


if __name__ == "__main__":
    main()
