from dataclasses import dataclass

import numpy as np

Normals = np.ndarray
"""Per-vertex ``(N, 3)`` unit normals aligned with the ESE vertices."""


@dataclass(frozen=True)
class ESEMesh:
    vertices: np.ndarray
    faces: np.ndarray
    scalp_vertices: np.ndarray
    quality: np.ndarray

    def __post_init__(self) -> None:
        for name in ("vertices", "scalp_vertices"):
            array = getattr(self, name)
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(f"{name} must be (N, 3) array, got shape {array.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3) array, got shape {self.faces.shape}")
        if self.quality.ndim != 1:
            raise ValueError(f"quality must be (N,) array, got shape {self.quality.shape}")

        n_vertices = self.vertices.shape[0]
        for name in ("scalp_vertices", "quality"):
            array = getattr(self, name)
            if array.shape[0] != n_vertices:
                raise ValueError(
                    f"{name} must have {n_vertices} rows to match vertices, got {array.shape[0]}"
                )
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= n_vertices):
            raise ValueError(
                f"Faces indices must be in [0, {n_vertices}), "
                f"got [{self.faces.min()}, {self.faces.max()}]"
            )
