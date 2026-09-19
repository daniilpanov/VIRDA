"""Environment-backed settings for the VIRDA pipeline."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from virda.segmentation.head_segmenter import OtsuScope


class VirdaSettings(BaseSettings):
    """Base settings loaded from the environment and the ``.env`` dotenv file.

    This is the lowest-priority settings source: input config files and CLI
    arguments override it (see :func:`virda.config.build_config`).
    """

    nifti_path: str | None = None
    project_dir: str | None = None
    fiducials_path: str | None = None
    auto_detect_fiducials: bool = False
    measurements_path: str | None = None

    closing_radius: int = 5

    otsu_scope: OtsuScope = "all"
    otsu_threshold_scale: float = Field(default=0.6, gt=0)

    seal_enabled: bool = True
    seal_radius: int = 4

    mesh_voxel_size_mm: float | None = Field(default=None, gt=0)
    mesh_density_percent: float = Field(default=100.0, ge=1, le=100)

    cleaner_min_vertices: int = 100
    cleaner_merge_digits: int = 7

    smoother_type: str = "laplacian"
    smoother_iterations: int = 5
    smoother_lamb: float = 0.5
    smoother_nu: float = -0.53

    ese_offset_mm: float | None = None

    neighborhood_radius_mm: float = Field(default=10.0, gt=0)
    k_neighbors: int | None = None
    use_weighted_pca: bool = False
    pca_sigma_mm: float = Field(default=5.0, gt=0)
    min_neighbors: int = Field(default=5, ge=1)

    residual_threshold_mm: float = Field(default=10.0, gt=0)
    calibrate_ese_offset: bool = True

    model_config = SettingsConfigDict(env_file=".env")