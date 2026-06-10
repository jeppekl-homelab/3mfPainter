"""Edge-based mesh segmentation (torch, device-agnostic).

Faces are nodes in a graph; two adjacent faces are connected when the dihedral
angle between them is below a threshold. Sharp edges therefore become segment
boundaries, and a connected-components pass labels each region. Everything is
vectorized torch, so it runs on CUDA when available and CPU otherwise.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import trimesh

from .compute import get_device
from .mesh import PaintMesh


class Segmenter:
    def __init__(self, mesh: PaintMesh):
        t0 = time.perf_counter()
        tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        # kopi: trimesh's cachede arrays er read-only, hvilket torch ikke kan lide
        adjacency = np.array(tm.face_adjacency)  # (E, 2) nabo-face-par

        self.device = get_device()
        self.n_faces = mesh.n_faces
        self._adj = torch.as_tensor(adjacency, dtype=torch.long, device=self.device)

        normals = torch.as_tensor(
            np.array(tm.face_normals), dtype=torch.float32, device=self.device
        )
        n0 = normals[self._adj[:, 0]]
        n1 = normals[self._adj[:, 1]]
        cos = (n0 * n1).sum(dim=1).clamp(-1.0, 1.0)
        self._angles_deg = torch.rad2deg(torch.arccos(cos))

        self._cache: dict[float, np.ndarray] = {}
        print(
            f"Segmenter: {len(adjacency):,} kanter analyseret "
            f"på {time.perf_counter() - t0:.2f}s [{self.device}]"
        )

    def labels(self, max_angle_deg: float) -> np.ndarray:
        """Per-face segment label. Faces in the same smooth region share label."""
        key = round(max_angle_deg, 1)
        if key not in self._cache:
            t0 = time.perf_counter()
            self._cache[key] = self._connected_components(key)
            n_segments = len(np.unique(self._cache[key]))
            print(
                f"Segmentering @ {key}°: {n_segments:,} regioner "
                f"på {time.perf_counter() - t0:.2f}s"
            )
        return self._cache[key]

    def _connected_components(self, max_angle_deg: float) -> np.ndarray:
        smooth = self._angles_deg <= max_angle_deg
        e = self._adj[smooth]
        edges = torch.cat([e, e.flip(1)])  # begge retninger
        src, dst = edges[:, 0], edges[:, 1]

        # Min-label-propagation + pointer jumping: hver face starter med sit
        # eget label; naboer overtager det mindste, og labels[labels] kortslutter
        # kæder så det konvergerer i ~O(log n) iterationer i stedet for O(diameter).
        labels = torch.arange(self.n_faces, device=self.device)
        while True:
            new = labels.scatter_reduce(
                0, dst, labels[src], reduce="amin", include_self=True
            )
            new = new[new]  # pointer jumping
            if torch.equal(new, labels):
                break
            labels = new
        return labels.cpu().numpy()
