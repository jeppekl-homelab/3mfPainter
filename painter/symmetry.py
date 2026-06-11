"""Mirror-symmetry detection.

Strategy: candidate mirror planes pass through the area-weighted centroid with
normals along the principal axes (PCA) plus the global X/Y/Z axes. Each
candidate is scored by reflecting all face centroids through the plane and
measuring how many land on top of an existing face (KD-tree nearest neighbor,
O(n log n)). The best plane above the acceptance threshold yields a per-face
mirror map used to mirror paint strokes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .mesh import PaintMesh

# andel af faces der skal have en spejl-makker før planet accepteres
ACCEPT_FRACTION = 0.85
# NN-tolerance i enheder af makker-facens egen størrelse (sqrt af areal).
# Trianguleringen er aldrig spejlsymmetrisk i praksis, så den reflekterede
# centroid lander et tilfældigt sted på den spejlede face — ikke i dens centroid.
TOLERANCE_FACE_SCALE = 1.0
# spejlet normal skal pege samme vej som makkerens normal
NORMAL_AGREEMENT = 0.8


@dataclass
class MirrorMap:
    face_map: np.ndarray   # (F,) int64; spejl-face pr. face, -1 = ingen makker
    adjacency: np.ndarray  # (E,2) int64; face-nabopar, til hul-lukning
    normal: np.ndarray     # (3,) planets normal
    origin: np.ndarray     # (3,) punkt i planet
    match: float           # andel faces med makker (0..1)

    @property
    def axis_label(self) -> str:
        ax = np.argmax(np.abs(self.normal))
        names = ["X", "Y", "Z"]
        if np.abs(self.normal[ax]) > 0.99:
            return names[ax]
        return f"({self.normal[0]:.2f}, {self.normal[1]:.2f}, {self.normal[2]:.2f})"

    def _fill_holes(self, faces: np.ndarray) -> np.ndarray:
        """Fill fully-enclosed single-face gaps (all edge-neighbors painted).

        The ~few % of faces with no mirror partner can leave a pinhole inside
        the mirrored region. Filling only faces whose every neighbor is already
        painted closes those without growing the region's outer boundary, so
        the result stays the shape of the stroke.
        """
        adj = self.adjacency
        if len(adj) == 0 or len(faces) == 0:
            return faces
        n = len(self.face_map)
        painted = np.zeros(n, dtype=bool)
        painted[faces] = True
        a, b = adj[:, 0], adj[:, 1]
        deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
        pnb = np.bincount(a, painted[b], n) + np.bincount(b, painted[a], n)
        holes = np.flatnonzero((~painted) & (deg > 0) & (pnb >= deg))
        if len(holes) == 0:
            return faces
        return np.union1d(faces, holes)

    def mirror(self, face_ids: np.ndarray) -> np.ndarray:
        """The painted faces plus their mirror image, as a clean region.

        Uses the map in the INVERSE direction: a target face is mirrored-in
        when its own reflection lands on a painted face. Evaluating it per
        target face fills the reflected region completely (no speckle holes)
        and — unlike the forward direction — never drags in a lone outlier
        triangle whose partner sits just outside the painted set. A final
        hole-fill closes the rare pinholes left by partnerless faces.
        """
        inverse = np.flatnonzero(np.isin(self.face_map, face_ids))
        result = np.union1d(face_ids, inverse)
        return self._fill_holes(result)


def _face_adjacency(mesh: PaintMesh) -> np.ndarray:
    """(E, 2) array of neighboring face-id pairs (shared-edge adjacency)."""
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    return np.asarray(tm.face_adjacency, dtype=np.int64)


def _containing_face(
    points: np.ndarray, tree: cKDTree, tri: np.ndarray, k: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """For each point, the triangle that contains it (projected), and its distance.

    Nearest-centroid matching fails near a quad's diagonal: the reflected point
    lands inside one triangle, but the other triangle's centroid may be closer,
    so contiguous regions mirror to ragged ones. Instead, among the k nearest
    centroids, test which triangle actually contains the point (barycentric,
    projected onto the triangle plane) and pick the one it sits closest to the
    plane of. Faces that contain no candidate fall back to the nearest centroid.
    """
    k = min(k, tri.shape[0])
    _, idx = tree.query(points, k=k, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]      # (N, k, 3)
    p = points[:, None, :]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    eps = 1e-6
    inside = (u >= -eps) & (v >= -eps) & (w >= -eps)     # (N, k)
    nf = np.cross(v0, v1)
    nf /= np.maximum(np.linalg.norm(nf, axis=-1, keepdims=True), 1e-12)
    perp = np.abs((v2 * nf).sum(-1))                     # afstand til trekantplanet
    score = np.where(inside, perp, np.inf)
    best = np.argmin(score, axis=1)
    rows = np.arange(len(points))
    chosen = idx[rows, best].copy()
    dist = score[rows, best].copy()
    # punkter uden nogen indeholdende kandidat: fald tilbage til nærmeste centroid
    none = ~inside.any(axis=1)
    chosen[none] = idx[none, 0]
    dist[none] = np.linalg.norm(points[none] - tri[idx[none, 0]].mean(1), axis=1)
    return chosen, dist


def find_mirror(mesh: PaintMesh) -> MirrorMap | None:
    t0 = time.perf_counter()
    verts = mesh.vertices[mesh.faces]              # (F, 3, 3)
    centroids = verts.mean(axis=1)                 # (F, 3)
    cross = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2.0
    weights = areas / areas.sum()

    center = (centroids * weights[:, None]).sum(axis=0)
    d = centroids - center
    cov = (d * weights[:, None]).T @ d
    _, eigvecs = np.linalg.eigh(cov)

    candidates = [eigvecs[:, i] for i in range(3)] + [np.eye(3)[i] for i in range(3)]
    # dedupe næsten-parallelle normaler
    unique: list[np.ndarray] = []
    for n in candidates:
        if not any(abs(n @ u) > 0.99 for u in unique):
            unique.append(n)

    tree = cKDTree(centroids)
    face_scale = np.sqrt(np.maximum(areas, 1e-12))
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    adjacency = _face_adjacency(mesh)

    # 1) hurtig scoring med centroid-NN for at vælge det bedste kandidatplan
    best_n: np.ndarray | None = None
    best_match = 0.0
    for n in unique:
        reflected = centroids - 2.0 * ((d @ n)[:, None] * n)
        dist, idx = tree.query(reflected, workers=-1)
        mirrored_n = normals - 2.0 * ((normals @ n)[:, None] * n)
        normal_ok = (mirrored_n * normals[idx]).sum(axis=1) >= NORMAL_AGREEMENT
        ok = (dist <= TOLERANCE_FACE_SCALE * face_scale[idx]) & normal_ok
        match = float(ok.mean())
        if match >= ACCEPT_FRACTION and match > best_match:
            best_match, best_n = match, n.copy()

    elapsed = time.perf_counter() - t0
    if best_n is None:
        print(f"Symmetri: intet spejlplan fundet ({elapsed:.2f}s)")
        return None

    # 2) nøjagtig partner pr. face for det valgte plan: hvilken trekant INDEHOLDER
    #    det spejlede punkt (ikke blot nærmeste centroid) — fjerner hakkede kanter
    reflected = centroids - 2.0 * ((d @ best_n)[:, None] * best_n)
    tri = mesh.vertices[mesh.faces]
    idx, sdist = _containing_face(reflected, tree, tri)
    mirrored_n = normals - 2.0 * ((normals @ best_n)[:, None] * best_n)
    normal_ok = (mirrored_n * normals[idx]).sum(axis=1) >= NORMAL_AGREEMENT
    ok = (sdist <= TOLERANCE_FACE_SCALE * face_scale[idx]) & normal_ok
    face_map = np.where(ok, idx, -1).astype(np.int64)
    best = MirrorMap(face_map, adjacency, best_n, center, best_match)

    print(
        f"Symmetri: spejlplan {best.axis_label}, "
        f"{best.match:.1%} match ({time.perf_counter() - t0:.2f}s)"
    )
    return best
