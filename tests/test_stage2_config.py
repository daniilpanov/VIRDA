import pytest

from virda.models.stage2_config import Stage2Config


class TestStage2Config:
    @pytest.mark.parametrize("radius", [0.0, -1.0])
    def test_rejects_non_positive_radius(self, radius: float) -> None:
        with pytest.raises(ValueError, match="neighborhood_radius_mm must be positive"):
            Stage2Config(neighborhood_radius_mm=radius)

    @pytest.mark.parametrize("k", [0, 1])
    def test_rejects_small_k(self, k: int) -> None:
        with pytest.raises(ValueError, match="k_neighbors must be >= 2"):
            Stage2Config(k_neighbors=k)

    @pytest.mark.parametrize("sigma", [0.0, -0.5])
    def test_rejects_non_positive_sigma(self, sigma: float) -> None:
        with pytest.raises(ValueError, match="pca_sigma_mm must be positive"):
            Stage2Config(pca_sigma_mm=sigma)

    @pytest.mark.parametrize("min_neighbors", [0, -3])
    def test_rejects_invalid_min_neighbors(self, min_neighbors: int) -> None:
        with pytest.raises(ValueError, match="min_neighbors must be at least 1"):
            Stage2Config(min_neighbors=min_neighbors)
