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
    coquad: np.ndarray     # (F,) int64; co-quad-nabo (længste kant), -1 = ingen
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

    def _quad_close(self, face_ids: np.ndarray) -> np.ndarray:
        """Include each face's co-quad neighbor so a stroke covers whole quads.

        A grid triangulation splits every quad along a diagonal, and reflection
        flips that diagonal — so a contiguous run of triangles maps to a ragged
        set of lone triangles on the mirrored side. Pairing each triangle with
        its neighbor across the longest shared edge (the diagonal) makes both
        sides whole quads, which mirror onto whole quads cleanly.
        """
        mates = self.coquad[face_ids]
        return np.union1d(face_ids, mates[mates >= 0])

    def mirror(self, face_ids: np.ndarray) -> np.ndarray:
        """Union of the faces and their mirror partners.

        Strokes are first closed to whole quads (see _quad_close) so the
        mirrored side has no lone, flipped triangles. The map is then used in
        both directions: forward (the painted face's partner) AND inverse
        (every face whose own reflection lands in the painted set). Forward
        alone is not surjective — faces on the mirrored side that are nobody's
        nearest neighbor would be skipped, leaving unpainted speckles.
        """
        src = self._quad_close(face_ids)
        forward = self.face_map[src]
        forward = forward[forward >= 0]
        inverse = np.flatnonzero(np.isin(self.face_map, src))
        return np.union1d(np.union1d(src, forward), inverse)


def _coquad_map(mesh: PaintMesh) -> np.ndarray:
    """For each face, the neighbor across its longest shared edge (the diagonal).

    Vectorized: both directions of every adjacency are sorted by shared-edge
    length so the longest edge wins per face. Boundary faces stay -1.
    """
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    adj = np.asarray(tm.face_adjacency)
    edges = np.asarray(tm.face_adjacency_edges)
    coquad = np.full(mesh.n_faces, -1, dtype=np.int64)
    if len(adj) == 0:
        return coquad
    length = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
    )
    src = np.concatenate([adj[:, 0], adj[:, 1]])
    dst = np.concatenate([adj[:, 1], adj[:, 0]])
    length = np.concatenate([length, length])
    order = np.argsort(length, kind="stable")  # stigende → længste kant vinder
    coquad[src[order]] = dst[order]
    return coquad


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
    coquad = _coquad_map(mesh)

    best: MirrorMap | None = None
    for n in unique:
        reflected = centroids - 2.0 * ((d @ n)[:, None] * n)
        dist, idx = tree.query(reflected, workers=-1)
        # spejlet normal: reflektér normalvektoren i planet
        mirrored_n = normals - 2.0 * ((normals @ n)[:, None] * n)
        normal_ok = (mirrored_n * normals[idx]).sum(axis=1) >= NORMAL_AGREEMENT
        ok = (dist <= TOLERANCE_FACE_SCALE * face_scale[idx]) & normal_ok
        match = float(ok.mean())
        if match >= ACCEPT_FRACTION and (best is None or match > best.match):
            face_map = np.where(ok, idx, -1).astype(np.int64)
            best = MirrorMap(face_map, coquad, n.copy(), center, match)

    elapsed = time.perf_counter() - t0
    if best is None:
        print(f"Symmetri: intet spejlplan fundet ({elapsed:.2f}s)")
    else:
        print(
            f"Symmetri: spejlplan {best.axis_label}, "
            f"{best.match:.1%} match ({elapsed:.2f}s)"
        )
    return best
