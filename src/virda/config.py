import os
from functools import cache

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from virda.models.ese_config import ESEConfig
from virda.models.stage2_config import Stage2Config
from virda.segmentation.head_segmenter import OtsuScope


def resolve_config_file(default: str = ".env") -> str:
    """Return the per-project settings file.

    Defaults to ``.env`` in the current directory. Set the
    ``VIRDA_CONFIG_FILE`` environment variable to load settings from a file in
    the processed dataset instead, e.g.::

        VIRDA_CONFIG_FILE=/data/CTRL_1277/.env.json python -m virda ...
    """
    return os.getenv("VIRDA_CONFIG_FILE", default)


class VirdaSettings(BaseSettings):
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

    neighborhood_radius_mm: float = 10.0
    k_neighbors: int | None = None
    use_weighted_pca: bool = False
    pca_sigma_mm: float = 5.0
    min_neighbors: int = 5

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_file=resolve_config_file(),
        json_file=resolve_config_file(".env.json"),
        yaml_file=resolve_config_file(".env.yaml"),
    )

    # TODO: In the future, avoid this method and find a replacement
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        ``pydantic-settings`` does not register the JSON/YAML file sources automatically
        wire them in explicitly so per-dataset config files actually take effect.
        CLI and env still override them because of source ordering.
        @see ``resolve_config_file``
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            JsonConfigSettingsSource(settings_cls, json_file=resolve_config_file()),
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def resolve_ese_config(settings: VirdaSettings) -> ESEConfig | None:
    """Build the ESE config when fully configured, otherwise return None."""
    if (
        settings.n_electrodes is None
        or settings.ese_offset_mm is None
        or settings.ese_reference is None
    ):
        return None
    return ESEConfig(
        n_electrodes=settings.n_electrodes,
        ese_offset_mm=settings.ese_offset_mm,
        ese_reference=settings.ese_reference,
    )


def resolve_stage2_config(settings: VirdaSettings) -> Stage2Config | None:
    """Build the Stage 2 (ESE) config when ESE is enabled, otherwise None."""
    if settings.ese_offset_mm is None:
        return None
    return Stage2Config(
        neighborhood_radius_mm=settings.neighborhood_radius_mm,
        k_neighbors=settings.k_neighbors,
        use_weighted_pca=settings.use_weighted_pca,
        pca_sigma_mm=settings.pca_sigma_mm,
        min_neighbors=settings.min_neighbors,
    )


@cache
def get_virda_settings() -> VirdaSettings:
    return VirdaSettings()
