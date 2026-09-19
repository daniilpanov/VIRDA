from dataclasses import dataclass
from typing import Literal


SmootherKind = Literal["laplacian", "taubin", "none"]


@dataclass(frozen=True)
class SealingOptions:
    seal_enabled: bool = True
    seal_radius: int = 4


@dataclass(frozen=True)
class SmoothOptions:
    smoother: SmootherKind = "laplacian"
    iterations: int = 5
    lamb: float = 0.5
    nu: float = -0.53


@dataclass(frozen=True)
class DecimateOptions:
    density_percent: float = 100.0


@dataclass(frozen=True)
class CleanOptions:
    min_component_vertices: int = 100
    merge_digits: int = 7


@dataclass(frozen=True)
class EseOptions:
    ese_offset_mm: float
    neighborhood_radius_mm: float = 10.0
    k_neighbors: int | None = None
    use_weighted_pca: bool = False
    pca_sigma_mm: float = 5.0
    min_neighbors: int = 5


@dataclass(frozen=True)
class LocalizeOptions:
    calibrate_ese_offset: bool = True
    residual_threshold_mm: float = 10.0