"""Mirror-symmetry detection and mirrored painting.

Detection: candidate planes pass through the area-weighted centroid with normals
along the principal axes (PCA) plus global X/Y/Z. Each is scored by reflecting
all face centroids and counting how many land on an existing face (KD-tree NN).

Painting: a chiral triangulation (e.g. a torus, where each quad's splitting
diagonal flips under reflection) admits NO exact triangle-to-triangle mirror
map — any per-triangle correspondence leaves notches or speckle holes. So we
mirror geometrically instead: reflect the painted faces' centroids and paint
every face whose reflection lands within a local radius of a painted one. This
is a solid area fill — no correspondence, no chirality assumption, so it cannot
speckle, and on real (irregular) meshes the two sides come out symmetric.
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
# NN-tolerance i enheder af makker-facens egen størrelse (sqrt af areal)
TOLERANCE_FACE_SCALE = 1.0
# spejlet normal skal pege samme vej som makkerens normal
NORMAL_AGREEMENT = 0.8
# splat-radius i enheder af en faces lokale centroid-afstand (fylder huller
# uden at overmale; >1 sikrer solidt fyld på tværs af triangulerings-skift)
MIRROR_RADIUS_SCALE = 1.3


@dataclass
class MirrorMap:
    centroids: np.ndarray  # (F,3) face-centroider
    reflected: np.ndarray  # (F,3) centroider spejlet i planet
    radius: np.ndarray     # (F,) splat-radius pr. face (lokal centroid-afstand)
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
        """Fill fully-enclosed single-face gaps (all edge-neighbors painted)."""
        adj = self.adjacency
        if len(adj) == 0 or len(faces) == 0:
            return faces
        n = len(self.radius)
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
        """The painted faces plus a solid mirror image across the plane.

        A face is mirrored-in when ITS reflection lands within the face's local
        radius of a painted face's centroid. Evaluated per face, this fills the
        reflected region solidly regardless of how the two sides are triangulated
        — the cause of the notches and speckles that any triangle-correspondence
        map produces on chiral or imperfectly-symmetric meshes.
        """
        if len(face_ids) == 0:
            return face_ids
        dab = cKDTree(self.centroids[face_ids])
        dist, _ = dab.query(self.reflected, workers=-1)
        mirrored = np.flatnonzero(dist <= self.radius)
        return self._fill_holes(np.union1d(face_ids, mirrored))


def _face_adjacency(mesh: PaintMesh) -> np.ndarray:
    """(E, 2) array of neighboring face-id pairs (shared-edge adjacency)."""
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    return np.asarray(tm.face_adjacency, dtype=np.int64)


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

    # vælg det bedste kandidatplan via centroid-NN-scoring
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

    # forbered geometrisk spejling: reflekterede centroider + lokal splat-radius
    reflected = centroids - 2.0 * ((d @ best_n)[:, None] * best_n)
    nn_dist, _ = tree.query(centroids, k=2, workers=-1)   # [:,1] = nærmeste nabo
    radius = MIRROR_RADIUS_SCALE * nn_dist[:, 1]
    best = MirrorMap(centroids, reflected, radius, adjacency, best_n, center, best_match)

    print(
        f"Symmetri: spejlplan {best.axis_label}, "
        f"{best.match:.1%} match ({time.perf_counter() - t0:.2f}s)"
    )
    return best
