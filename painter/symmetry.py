"""Mirror-symmetry detection and mirrored painting.

Detection: candidate planes pass through the area-weighted centroid with normals
along the principal axes (PCA) plus global X/Y/Z. Each is scored by reflecting
all face centroids and counting how many land on an existing face (KD-tree NN).

Painting: there is no exact triangle-to-triangle mirror map on a real mesh — a
chiral triangulation (e.g. a torus) flips each quad's diagonal under reflection,
and a sculpted model is only ~98% symmetric — so any per-triangle correspondence
produces notches, speckles or ragged edges. We mirror PURELY GEOMETRICALLY: the
brush dab is a small ball of faces; we reflect its centre through the plane and
paint the faces within the same radius on the far side. Source and mirror use the
identical rule, so the two sides look the same by construction, on any mesh.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .mesh import PaintMesh

# andel af faces der skal have en spejl-makker før planet accepteres
ACCEPT_FRACTION = 0.85
# NN-tolerance i enheder af makker-facens egen størrelse (sqrt af areal)
TOLERANCE_FACE_SCALE = 1.0
# spejlet normal skal pege samme vej som makkerens normal
NORMAL_AGREEMENT = 0.8


@dataclass
class MirrorMap:
    centroids: np.ndarray  # (F,3) face-centroider
    tree: cKDTree          # KD-træ over centroider, til radius-opslag
    spacing: float         # median centroid-afstand (radius-gulv)
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

    def mirror(self, face_ids: np.ndarray) -> np.ndarray:
        """The painted faces plus a geometric mirror of the brush dab.

        Treat the dab as a ball: its centre is the mean of the painted faces'
        centroids and its radius their spread (floored at the local spacing).
        Reflect the centre through the plane and paint every face within that
        radius. No per-triangle correspondence, so nothing to speckle or notch —
        the mirror is the same ball, just reflected, and matches the source.
        """
        if len(face_ids) == 0:
            return face_ids
        pts = self.centroids[face_ids]
        p = pts.mean(axis=0)
        radius = max(float(np.linalg.norm(pts - p, axis=1).max()), self.spacing)
        n = self.normal
        p_ref = p - 2.0 * ((p - self.origin) @ n) * n
        mirrored = self.tree.query_ball_point(p_ref, radius)
        return np.unique(np.concatenate([face_ids, np.asarray(mirrored, dtype=np.int64)]))


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

    nn_dist, _ = tree.query(centroids, k=2, workers=-1)   # [:,1] = nærmeste nabo
    spacing = float(np.median(nn_dist[:, 1]))
    best = MirrorMap(centroids, tree, spacing, best_n, center, best_match)

    print(
        f"Symmetri: spejlplan {best.axis_label}, "
        f"{best.match:.1%} match ({time.perf_counter() - t0:.2f}s)"
    )
    return best
