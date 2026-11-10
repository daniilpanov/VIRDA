from dataclasses import dataclass

import numpy as np

Normals = np.ndarray
"""Per-vertex ``(N, 3)`` unit normals aligned with the mesh vertices."""


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be (N, 3) array, got shape {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must be (M, 3) array, got shape {self.faces.shape}")

        n_vertices = self.vertices.shape[0]
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= n_vertices):
            raise ValueError(
                f"Faces indices must be in [0, {n_vertices}), "
                f"got [{self.faces.min()}, {self.faces.max()}]"
            )
