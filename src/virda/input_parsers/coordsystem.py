"""Parser for MNE-style ``coordsystem.json`` files."""

import json
from pathlib import Path
from typing import Any

from virda.models.coordsystem import Coordsystem


def parse_coordsystem(path: str | Path) -> Coordsystem:
    """Parse an MNE ``coordsystem.json`` file into a :class:`Coordsystem`.

    Delegates to :meth:`Coordsystem.model_validate` once the file is
    confirmed to be a JSON object.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"coordsystem file must contain a JSON object: {path}")
    return Coordsystem.model_validate(data)
