"""Orbit camera: drag to rotate, scroll to zoom, middle/shift-drag to pan."""

from __future__ import annotations

import math

import glm
import numpy as np


class OrbitCamera:
    def __init__(self, target: np.ndarray, radius: float):
        self.target = glm.vec3(*target.tolist())
        self.distance = radius * 2.5
        self._min_distance = radius * 0.05
        self.yaw = math.radians(45.0)
        self.pitch = math.radians(25.0)
        self.fov_deg = 45.0

    # --- input -----------------------------------------------------------
    def rotate(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.008
        self.pitch += dy * 0.008
        limit = math.radians(89.0)
        self.pitch = max(-limit, min(limit, self.pitch))

    def pan(self, dx: float, dy: float) -> None:
        view = self.view_matrix()
        right = glm.vec3(view[0][0], view[1][0], view[2][0])
        up = glm.vec3(view[0][1], view[1][1], view[2][1])
        scale = self.distance * 0.0015
        self.target += (-right * dx + up * dy) * scale

    def zoom(self, scroll: float) -> None:
        self.distance *= 0.9 ** scroll
        self.distance = max(self._min_distance, self.distance)

    # --- matrices --------------------------------------------------------
    def position(self) -> glm.vec3:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        offset = glm.vec3(cp * cy, sp, cp * sy) * self.distance
        return self.target + offset

    def view_matrix(self) -> glm.mat4:
        return glm.lookAt(self.position(), self.target, glm.vec3(0, 1, 0))

    def proj_matrix(self, aspect: float) -> glm.mat4:
        near = self.distance * 0.01
        far = self.distance * 100.0
        return glm.perspective(glm.radians(self.fov_deg), aspect, near, far)
