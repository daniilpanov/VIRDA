from dataclasses import dataclass


@dataclass(frozen=True)
class Stage2Config:
    neighborhood_radius_mm: float = 10.0
    k_neighbors: int | None = None
    use_weighted_pca: bool = False
    pca_sigma_mm: float = 5.0
    min_neighbors: int = 5

    def __post_init__(self) -> None:
        if self.neighborhood_radius_mm <= 0:
            raise ValueError(
                f"neighborhood_radius_mm must be positive, got {self.neighborhood_radius_mm}"
            )
        if self.k_neighbors is not None and self.k_neighbors < 2:
            raise ValueError(f"k_neighbors must be >= 2 when set, got {self.k_neighbors}")
        if self.pca_sigma_mm <= 0:
            raise ValueError(f"pca_sigma_mm must be positive, got {self.pca_sigma_mm}")
        if self.min_neighbors < 1:
            raise ValueError(f"min_neighbors must be at least 1, got {self.min_neighbors}")
