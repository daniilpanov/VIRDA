from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MRIVolume:
    data: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.affine.shape != (4, 4):
            raise ValueError(f"Affine must be 4x4, got {self.affine.shape}")
        if self.data.ndim != 3:
            raise ValueError(f"Data must be 3D, got {self.data.ndim}D")
        if len(self.spacing) != 3:
            raise ValueError(f"Spacing must have 3 elements, got {len(self.spacing)}")
        if any(s <= 0 for s in self.spacing):
            raise ValueError(f"Spacing must be strictly positive, got {self.spacing}")
        expected_bottom_row = np.array([0.0, 0.0, 0.0, 1.0])
        if not np.allclose(self.affine[3, :], expected_bottom_row, atol=1e-6):
            raise ValueError(
                f"Invalid affine matrix: bottom row must be [0, 0, 0, 1], got {self.affine[3, :]}"
            )
