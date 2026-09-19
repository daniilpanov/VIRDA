"""Import a pipeline config file (JSON or MNE coordsystem.json) into a flat dict."""

import json
from pathlib import Path
from typing import Any, cast

from virda.models.coordsystem import Coordsystem


def import_pipeline_config(path: str | Path) -> dict[str, Any]:
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
    return cast(dict[str, Any], data)