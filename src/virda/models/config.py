"""Final merged pipeline configuration."""

from pydantic import BaseModel, Field

from virda.models.coordsystem import Coordsystem
from virda.models.ese_config import ESEConfig
from virda.models.stage2_config import Stage2Config
from virda.segmentation.head_segmenter import OtsuScope


class Config(BaseModel):
    """Merged settings for the whole VIRDA pipeline.

    Built from ``VirdaSettings`` (env/dotenv), the input config files
    (e.g. ``coordsystem.json``) and the CLI arguments, in that order of
    priority. Stored in the pipeline context so every step can read it.
    """

    nifti_path: str | None = None
    project_dir: str | None = None
    fiducials_path: str | None = None
    auto_detect_fiducials: bool = False

    closing_radius: int = 5

    otsu_scope: OtsuScope = "all"
    otsu_threshold_scale: float = Field(default=0.6, gt=0)

    seal_enabled: bool = True
    seal_radius: int = 4

    cleaner_min_vertices: int = 100
    cleaner_merge_digits: int = 7

    smoother_type: str = "laplacian"
    smoother_iterations: int = 5
    smoother_lamb: float = 0.5
    smoother_nu: float = -0.53

    n_electrodes: int | None = None
    ese_offset_mm: float | None = None
    ese_reference: str | None = None

    neighborhood_radius_mm: float = Field(default=10.0, gt=0)
    k_neighbors: int | None = None
    use_weighted_pca: bool = False
    pca_sigma_mm: float = Field(default=5.0, gt=0)
    min_neighbors: int = Field(default=5, ge=1)

    coordsystem: Coordsystem | None = None

    def to_ese_config(self) -> ESEConfig | None:
        """Build the ESE config when fully configured, otherwise return None."""
        if self.n_electrodes is None or self.ese_offset_mm is None or self.ese_reference is None:
            return None
        return ESEConfig(
            n_electrodes=self.n_electrodes,
            ese_offset_mm=self.ese_offset_mm,
            ese_reference=self.ese_reference,
        )

    def to_stage2_config(self) -> Stage2Config:
        """Build the stage 2 (ESE) neighborhood config."""
        return Stage2Config(
            neighborhood_radius_mm=self.neighborhood_radius_mm,
            k_neighbors=self.k_neighbors,
            use_weighted_pca=self.use_weighted_pca,
            pca_sigma_mm=self.pca_sigma_mm,
            min_neighbors=self.min_neighbors,
        )
