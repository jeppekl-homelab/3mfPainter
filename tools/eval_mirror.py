"""Quantitative evaluation of mirrored-painting quality.

Builds a coarse capsule (tall thin side triangles, like the bug screenshot),
paints a ball-shaped patch on one side of the mirror plane, mirrors it, and
compares against a ground truth computed by dense area sampling: a face on the
far side SHOULD be painted iff >=50% of its area, reflected through the plane,
lands on painted source faces.

Outputs the symmetric difference vs ground truth and a render of the result.
"""

import sys

import numpy as np
import moderngl
import glm
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from painter.mesh import _prepare
from painter import symmetry
from painter.symmetry import find_mirror, _containing_face
from painter.viewer import mat_bytes

TAG = sys.argv[1] if len(sys.argv) > 1 else "old"


def dense_bary(n: int = 9) -> np.ndarray:
    """Interior barycentric grid points (i+j+k = n, all >= 1)."""
    pts = [
        (i / n, j / n, (n - i - j) / n)
        for i in range(1, n)
        for j in range(1, n - i)
    ]
    return np.array(pts)


def ground_truth(mesh, mm, painted_mask):
    """Faces whose reflected area majority-overlaps the painted set."""
    tri = mesh.vertices[mesh.faces]                     # (F,3,3)
    bary = dense_bary(9)                                # (B,3)
    samples = np.einsum("bk,fkx->fbx", bary, tri)       # (F,B,3)
    flat = samples.reshape(-1, 3)
    refl = flat - 2.0 * (((flat - mm.origin) @ mm.normal)[:, None] * mm.normal)

    centroids = tri.mean(axis=1)
    tree = cKDTree(centroids)
    hits, _ = _containing_face(refl, tree, tri)
    votes = painted_mask[hits].reshape(len(tri), -1).mean(axis=1)
    return votes >= 0.5


def boundary_len(mesh, member_mask, adjacency):
    """Number of adjacency edges crossing the set boundary (raggedness proxy)."""
    a, b = adjacency[:, 0], adjacency[:, 1]
    return int((member_mask[a] != member_mask[b]).sum())


def render(mesh, colors_u1, path, eye, target):
    ctx = moderngl.create_context(standalone=True, require=410)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    prog = ctx.program(
        vertex_shader="""
            #version 410
            uniform mat4 u_mvp;
            in vec3 in_position; in vec3 in_normal; in vec3 in_color;
            out vec3 v_n; out vec3 v_c;
            void main(){ gl_Position=u_mvp*vec4(in_position,1.0); v_n=in_normal; v_c=in_color; }
        """,
        fragment_shader="""
            #version 410
            uniform vec3 u_eye_dir;
            in vec3 v_n; in vec3 v_c; out vec4 f;
            void main(){
                float d = max(dot(normalize(v_n), u_eye_dir), 0.0)*0.8+0.2;
                f = vec4(v_c*d, 1.0);
            }
        """,
    )
    palette = np.array(
        [[0.8, 0.8, 0.85], [0.9, 0.15, 0.15], [0.2, 0.8, 0.3], [0.2, 0.4, 1.0]], dtype="f4"
    )
    vcols = np.repeat(palette[colors_u1], 3, axis=0)
    data = np.hstack([mesh.positions, mesh.normals, vcols]).astype("f4")
    vbo = ctx.buffer(data.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")])
    W, H = 900, 700
    fbo = ctx.framebuffer(ctx.texture((W, H), 4), ctx.depth_renderbuffer((W, H)))
    fbo.use()
    ctx.clear(0.12, 0.13, 0.15)
    view = glm.lookAt(glm.vec3(*eye), glm.vec3(*target), glm.vec3(0, 0, 1))
    proj = glm.perspective(glm.radians(40), W / H, 0.1, 100.0)
    prog["u_mvp"].write(mat_bytes(proj * view))
    d = glm.normalize(glm.vec3(*eye) - glm.vec3(*target))
    prog["u_eye_dir"].value = (d.x, d.y, d.z)
    vao.render(moderngl.TRIANGLES)
    img = Image.frombytes("RGBA", (W, H), fbo.read(components=4)).transpose(
        Image.FLIP_TOP_BOTTOM
    )
    img.save(path)


def main():
    # kapsel: lav opløsning -> høje tynde trekanter på siderne, som i bug-billedet
    # strakt uv-kugle: kvadratgitter med ensrettede diagonaler — reflektionen
    # flipper hver diagonal, så det er værst tænkelige tilfælde for spejling
    cap = trimesh.creation.uv_sphere(radius=1.0, count=[48, 40])
    cap.apply_scale([1.0, 1.0, 2.0])
    mesh = _prepare(cap)
    mm = find_mirror(mesh)
    assert mm is not None, "kapsel skal have symmetri"
    print(f"mesh: {mesh.n_faces} faces; plan: {mm.axis_label}, origin {np.round(mm.origin, 3)}")

    tri = mesh.vertices[mesh.faces]
    centroids = tri.mean(axis=1)

    # "pensel-dab": klat på den ene side af planet, tæt på planet
    side = (centroids - mm.origin) @ mm.normal
    perp = np.array([1.0, 0, 0]) if abs(mm.normal[0]) < 0.9 else np.array([0, 1.0, 0])
    perp = perp - (perp @ mm.normal) * mm.normal
    perp /= np.linalg.norm(perp)
    # pensel-agtig dab: enhver face der RØRER cirklen males (som ID-buffer-
    # picking), dvs. min-afstand over hjørner+centroid — giver samme slags
    # flossede kildekant som den rigtige pensel på tynde trekanter
    seed = mm.origin + perp * 1.05 + mm.normal * 0.45
    pts = np.concatenate([tri, centroids[:, None, :]], axis=1)   # (F,4,3)
    dmin = np.linalg.norm(pts - seed, axis=2).min(axis=1)
    src = np.flatnonzero((dmin < 0.55) & (side > 0.02))
    print(f"kilde-patch: {len(src)} faces")

    painted = np.zeros(mesh.n_faces, dtype=bool)
    painted[src] = True

    result = mm.mirror(src)
    res_mask = np.zeros(mesh.n_faces, dtype=bool)
    res_mask[result] = True

    gt_far = ground_truth(mesh, mm, painted)        # ideal spejlside
    far = ~painted                                   # alt uden for kilden
    ideal = painted | (gt_far & far)

    diff = res_mask != ideal
    print(f"resultat: {res_mask.sum()} faces, ideal: {ideal.sum()}")
    print(f"symmetrisk difference vs ideal: {diff.sum()} faces")
    print(
        f"boundary-kanter resultat: {boundary_len(mesh, res_mask, mm.adjacency)}, "
        f"ideal: {boundary_len(mesh, ideal, mm.adjacency)}"
    )

    # farver: rød = kilde, grøn = spejlet, blå = afvigelse fra ideal
    cols = np.zeros(mesh.n_faces, dtype="i8")
    cols[result] = 2
    cols[src] = 1
    cols[diff] = 3

    seed_r = seed - 2 * ((seed - mm.origin) @ mm.normal) * mm.normal
    eye = seed_r * 1.2 + np.array([0, 0, 0.4]) + (seed_r - mm.origin) * 2.2
    render(mesh, cols, f"eval_mirror_{TAG}.png", eye[:3], seed_r[:3])
    print(f"gemt eval_mirror_{TAG}.png")


if __name__ == "__main__":
    main()
