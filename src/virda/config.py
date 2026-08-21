import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from virda.models.config import Config
from virda.models.coordsystem import Coordsystem
from virda.models.stage3_config import Stage3Config
from virda.segmentation.head_segmenter import OtsuScope


class VirdaSettings(BaseSettings):
    """Base settings loaded from the environment and the ``.env`` dotenv file.

    This is the lowest-priority settings source: input config files and CLI
    arguments override it (see :func:`build_config`).
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

    residual_threshold_mm: float = Field(default=10.0, gt=0)
    calibrate_ese_offset: bool = False

    model_config = SettingsConfigDict(env_file=".env")


def resolve_config_files(cli_files: list[str] | None = None) -> list[Path]:
    """Collect the input config files in priority order (last one wins).

    Sources, from lowest to highest priority:

    1. ``VIRDA_CONFIG_FILE`` — legacy single dataset config (e.g. ``.env.json``);
    2. ``VIRDA_CONFIG_FILES`` — :data:`os.pathsep`-separated list of config files;
    3. ``--config-file`` CLI arguments, in the order given on the command line.
    """
    paths: list[str] = []
    legacy = os.getenv("VIRDA_CONFIG_FILE")
    if legacy:
        paths.append(legacy)
    env_files = os.getenv("VIRDA_CONFIG_FILES")
    if env_files:
        paths.extend(path for path in env_files.split(os.pathsep) if path)
    if cli_files:
        paths.extend(cli_files)
    return [Path(path) for path in paths]


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load one input config file into a flat settings dict.

    An MNE ``coordsystem.json`` (recognized by its ``CoordinateSystem`` or
    ``FiducialsCoordinates`` key) contributes ``n_electrodes`` and the parsed
    ``coordsystem`` value; any other JSON file is merged as-is.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a JSON object: {path}")
    if "CoordinateSystem" in data or "FiducialsCoordinates" in data:
        coordsystem = Coordsystem.model_validate(data)
        flat: dict[str, Any] = {}
        if coordsystem.electrode_count is not None:
            flat["n_electrodes"] = coordsystem.electrode_count
        if coordsystem.electrode_offset_mm is not None:
            flat["ese_offset_mm"] = coordsystem.electrode_offset_mm
        if coordsystem.electrode_reference is not None:
            flat["ese_reference"] = coordsystem.electrode_reference
        flat["coordsystem"] = coordsystem
        return flat
    return data


def build_config(
    settings: VirdaSettings,
    config_files: Sequence[Path | str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Merge the settings sources into the final :class:`Config`.

    Priority (lowest to highest): ``settings``, then each config file in order,
    then ``overrides`` (CLI arguments).
    """
    data: dict[str, Any] = settings.model_dump()
    for config_file in config_files or []:
        data.update(load_config_file(config_file))
    if overrides:
        data.update(overrides)
    return Config.model_validate(data)


def resolve_stage3_config(config: Config) -> Stage3Config:
    """Build the Stage 3 (localization) config."""
    return Stage3Config(
        residual_threshold_mm=config.residual_threshold_mm,
        calibrate_ese_offset=config.calibrate_ese_offset,
    )
