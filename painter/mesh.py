"""Mesh loading and GPU-friendly preparation.

Vertices are duplicated per face ("unwelded") so every face can carry its own
flat normal, color and face-ID. That layout is what makes GPU picking and
per-face painting trivial later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class PaintMesh:
    """Unwelded triangle soup ready for the GPU."""

    positions: np.ndarray   # (F*3, 3) float32
    normals: np.ndarray     # (F*3, 3) float32, flat per-face normal
    face_ids: np.ndarray    # (F*3,)   uint32, same id for the 3 verts of a face
    faces: np.ndarray       # (F, 3)   int64, original indexed faces
    vertices: np.ndarray    # (V, 3)   float64, original welded vertices
    center: np.ndarray      # (3,) bounding-sphere-ish center
    radius: float           # bounding radius, for camera framing

    @property
    def n_faces(self) -> int:
        return len(self.faces)


def load(path: str) -> PaintMesh:
    mesh = trimesh.load_mesh(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    return _prepare(mesh)


def demo() -> PaintMesh:
    """Fallback mesh so the viewer always has something to show."""
    mesh = trimesh.creation.torus(major_radius=2.0, minor_radius=0.8)
    return _prepare(mesh)


def _prepare(mesh: trimesh.Trimesh) -> PaintMesh:
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)

    positions = verts[faces].reshape(-1, 3).astype("f4")
    normals = np.repeat(np.asarray(mesh.face_normals), 3, axis=0).astype("f4")
    face_ids = np.repeat(np.arange(len(faces), dtype="u4"), 3)

    lo, hi = positions.min(axis=0), positions.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0) or 1.0

    return PaintMesh(
        positions=positions,
        normals=normals,
        face_ids=face_ids,
        faces=faces,
        vertices=verts,
        center=center,
        radius=radius,
    )
