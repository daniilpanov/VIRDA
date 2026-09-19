"""Merge the settings sources into the final pipeline :class:`Config`."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from virda.config.files import load_config_file
from virda.config.settings import VirdaSettings
from virda.models.config import Config
from virda.models.stage3_config import Stage3Config


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