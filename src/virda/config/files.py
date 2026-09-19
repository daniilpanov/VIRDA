"""Input config file discovery and loading."""

import json
import os
from pathlib import Path
from typing import Any

from virda.models.coordsystem import Coordsystem


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

    An MNE ``coordsystem.json`` (detected via
    :meth:`Coordsystem.is_coordsystem_dict`) contributes
    the parsed ``coordsystem`` value; any other JSON file is merged as-is.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a JSON object: {path}")
    if Coordsystem.is_coordsystem_dict(data):
        coordsystem = Coordsystem.model_validate(data)
        flat: dict[str, Any] = {}
        if coordsystem.electrode_offset_mm is not None:
            flat["ese_offset_mm"] = coordsystem.electrode_offset_mm
        flat["coordsystem"] = coordsystem
        return flat
    return data