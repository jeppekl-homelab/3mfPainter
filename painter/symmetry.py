"""Mirror-symmetry detection.

Detection: candidate planes pass through the area-weighted centroid with
normals along the principal axes (PCA) plus global X/Y/Z. Each is scored by
reflecting all face centroids and counting how many land on an existing face
(KD-tree nearest neighbor with a face-size tolerance and mirrored-normal
agreement — real triangulations are never exactly mirror-symmetric).

Painting does NOT use any face-to-face correspondence. Every correspondence/
voting scheme discretizes twice (the source boundary is already a triangle
zigzag around the smooth brush circle; re-discretizing that zigzag onto the
differently-triangulated far side compounds the aliasing into sawtooth). The
viewer instead mirrors the BRUSH: it renders a second ID-pass with the camera
reflected through the plane and runs the identical screen-circle pick there,
so both sides are first-order discretizations of the same smooth dab and look
the same by construction. This module only supplies the plane (and the
reflection matrix for that pass).
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
    normal: np.ndarray   # (3,) planets normal
    origin: np.ndarray   # (3,) punkt i planet
    match: float         # andel faces med spejl-makker (0..1)

    @property
    def axis_label(self) -> str:
        ax = np.argmax(np.abs(self.normal))
        names = ["X", "Y", "Z"]
        if np.abs(self.normal[ax]) > 0.99:
            return names[ax]
        return f"({self.normal[0]:.2f}, {self.normal[1]:.2f}, {self.normal[2]:.2f})"

    def reflection_matrix(self) -> np.ndarray:
        """4x4 world-space reflection through the plane (column-vector math).

        p' = p - 2((p-o)·n)n  ⇒  M = [I-2nnᵀ | 2(o·n)n]. Determinant is -1:
        the caller must flip triangle winding/culling when rendering with it.
        """
        n = self.normal / np.linalg.norm(self.normal)
        m = np.eye(4)
        m[:3, :3] -= 2.0 * np.outer(n, n)
        m[:3, 3] = 2.0 * float(self.origin @ n) * n
        return m


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

    best: MirrorMap | None = None
    for n in unique:
        reflected = centroids - 2.0 * ((d @ n)[:, None] * n)
        dist, idx = tree.query(reflected, workers=-1)
        mirrored_n = normals - 2.0 * ((normals @ n)[:, None] * n)
        normal_ok = (mirrored_n * normals[idx]).sum(axis=1) >= NORMAL_AGREEMENT
        ok = (dist <= TOLERANCE_FACE_SCALE * face_scale[idx]) & normal_ok
        match = float(ok.mean())
        if match >= ACCEPT_FRACTION and (best is None or match > best.match):
            best = MirrorMap(n.copy(), center, match)

    elapsed = time.perf_counter() - t0
    if best is None:
        print(f"Symmetri: intet spejlplan fundet ({elapsed:.2f}s)")
    else:
        print(
            f"Symmetri: spejlplan {best.axis_label}, "
            f"{best.match:.1%} match ({elapsed:.2f}s)"
        )
    return best
