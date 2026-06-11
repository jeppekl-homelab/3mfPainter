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
    face_map: np.ndarray   # (F,) face på hver faces spejl-position, -1 = ingen
    adjacency: np.ndarray  # (E,2) face-nabopar, til hul-lukning og søm-bro
    dside: np.ndarray      # (F,) signeret afstand til planet, pr. face
    spacing: float         # median centroid-afstand (bånd-skala)
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

    def _fill_holes(self, faces: np.ndarray, passes: int = 2) -> np.ndarray:
        """Fill fully-enclosed gaps (faces whose every edge-neighbor is painted).

        Forward correspondence maps several source faces onto one mirror face,
        leaving the odd interior face blank; filling only fully-surrounded faces
        closes those speckles without growing the region's outer boundary.
        """
        adj = self.adjacency
        if len(adj) == 0 or len(faces) == 0:
            return faces
        n = len(self.dside)
        a, b = adj[:, 0], adj[:, 1]
        deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
        for _ in range(passes):
            painted = np.zeros(n, dtype=bool)
            painted[faces] = True
            pnb = np.bincount(a, painted[b], n) + np.bincount(b, painted[a], n)
            holes = np.flatnonzero((~painted) & (deg > 0) & (pnb >= deg))
            if len(holes) == 0:
                break
            faces = np.union1d(faces, holes)
        return faces

    def _bridge_seam(self, faces: np.ndarray) -> np.ndarray:
        """Fill faces straddling the plane that connect the two painted halves.

        A stroke running near the plane leaves the faces the plane cuts through
        blank — the source reaches it from one side, the mirror from the other.
        Grow paint into that band only through faces that have painted neighbors
        on BOTH sides of the plane, so the gap is bridged without spreading paint
        along the plane past the stroke.
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
        """The painted faces plus their mirror, by face correspondence.

        The source stays exactly the brushed faces (no radius ball — a 3D ball
        engulfs whole tall triangles and spikes the edge). Each painted face is
        mirrored to the face that sits at its reflected position; a hole-fill
        closes the speckles that the many-to-one mapping leaves, and a seam
        bridge connects the two halves across the plane.
        """
        if len(face_ids) == 0:
            return face_ids
        forward = self.face_map[face_ids]
        forward = forward[forward >= 0]
        result = np.union1d(face_ids, forward)
        result = self._fill_holes(result)
        return self._bridge_seam(result)


def _containing_face(
    points: np.ndarray, tree: cKDTree, tri: np.ndarray, k: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """For each point, the triangle that contains it (projected), and its distance.

    Among the k nearest face centroids, test which triangle actually contains
    the point (barycentric, projected onto the triangle plane) and pick the one
    it sits closest to the plane of. This is robust where nearest-centroid fails
    (e.g. a point near a quad diagonal). Points contained by no candidate fall
    back to the nearest centroid.
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

    # spejl-partner pr. face: hvilken trekant INDEHOLDER den reflekterede centroid
    reflected = centroids - 2.0 * ((d @ best_n)[:, None] * best_n)
    tri = mesh.vertices[mesh.faces]
    idx, sdist = _containing_face(reflected, tree, tri)
    mirrored_n = normals - 2.0 * ((normals @ best_n)[:, None] * best_n)
    normal_ok = (mirrored_n * normals[idx]).sum(axis=1) >= NORMAL_AGREEMENT
    ok = (sdist <= TOLERANCE_FACE_SCALE * face_scale[idx]) & normal_ok
    face_map = np.where(ok, idx, -1).astype(np.int64)

    best = MirrorMap(
        face_map, adjacency, dside, spacing, best_n, center, best_match
    )

    print(
        f"Symmetri: spejlplan {best.axis_label}, "
        f"{best.match:.1%} match ({time.perf_counter() - t0:.2f}s)"
    )
    return best
