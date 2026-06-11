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
import trimesh
from scipy.spatial import cKDTree

from .mesh import PaintMesh

# andel af faces der skal have en spejl-makker før planet accepteres
ACCEPT_FRACTION = 0.85
# NN-tolerance i enheder af makker-facens egen størrelse (sqrt af areal)
TOLERANCE_FACE_SCALE = 1.0
# spejlet normal skal pege samme vej som makkerens normal
NORMAL_AGREEMENT = 0.8
# søm-bro: bånd-bredde (i centroid-afstande) og maks. iterationer
SEAM_BAND = 3.0
SEAM_ITERS = 6


@dataclass
class MirrorMap:
    centroids: np.ndarray  # (F,3) face-centroider
    tree: cKDTree          # KD-træ over centroider, til radius-opslag
    spacing: float         # median centroid-afstand (radius-gulv)
    adjacency: np.ndarray  # (E,2) face-nabopar, til søm-bro
    dside: np.ndarray      # (F,) signeret afstand til planet, pr. face
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

    def _bridge_seam(self, faces: np.ndarray) -> np.ndarray:
        """Fill faces straddling the plane that connect the two painted halves.

        The ball mirror can leave an unpainted strip right on the symmetry plane
        when a stroke runs near it: the source reaches the plane from one side
        and the mirror from the other, but the faces the plane cuts through stay
        blank. We grow paint INTO that band only through faces that have painted
        neighbors on BOTH sides of the plane — so the gap is bridged without
        spreading paint along the plane past the stroke.
        """
        adj = self.adjacency
        if len(adj) == 0:
            return faces
        n = len(self.dside)
        painted = np.zeros(n, dtype=bool)
        painted[faces] = True
        band = np.abs(self.dside) < SEAM_BAND * self.spacing
        a, b = adj[:, 0], adj[:, 1]
        da, db = self.dside[a], self.dside[b]
        for _ in range(SEAM_ITERS):
            pos = (np.bincount(b, (painted[a] & (da > 0)).astype(float), n)
                   + np.bincount(a, (painted[b] & (db > 0)).astype(float), n)) > 0
            neg = (np.bincount(b, (painted[a] & (da < 0)).astype(float), n)
                   + np.bincount(a, (painted[b] & (db < 0)).astype(float), n)) > 0
            bridge = (~painted) & band & pos & neg
            if not bridge.any():
                break
            painted[bridge] = True
        return np.flatnonzero(painted)

    def mirror(self, face_ids: np.ndarray) -> np.ndarray:
        """The painted faces plus a geometric mirror of the brush dab.

        Treat the dab as a ball: its centre is the mean of the painted faces'
        centroids and its radius their spread (floored at the local spacing).
        Paint that ball on BOTH sides — once around the centre, once around its
        reflection through the plane — so the result is symmetric by construction
        (no triangle correspondence to speckle or notch). Then bridge any
        unpainted strip the plane cuts through (see _bridge_seam) so the two
        halves connect when the stroke runs along the symmetry plane.
        """
        if len(face_ids) == 0:
            return face_ids
        pts = self.centroids[face_ids]
        p = pts.mean(axis=0)
        radius = max(float(np.linalg.norm(pts - p, axis=1).max()), self.spacing)
        n = self.normal
        p_ref = p - 2.0 * ((p - self.origin) @ n) * n
        src = self.tree.query_ball_point(p, radius)
        mir = self.tree.query_ball_point(p_ref, radius)
        result = np.unique(
            np.concatenate([
                face_ids,
                np.asarray(src, dtype=np.int64),
                np.asarray(mir, dtype=np.int64),
            ])
        )
        return self._bridge_seam(result)


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
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    adjacency = np.asarray(tm.face_adjacency, dtype=np.int64)
    dside = (centroids - center) @ best_n
    best = MirrorMap(
        centroids, tree, spacing, adjacency, dside, best_n, center, best_match
    )

    print(
        f"Symmetri: spejlplan {best.axis_label}, "
        f"{best.match:.1%} match ({time.perf_counter() - t0:.2f}s)"
    )
    return best
