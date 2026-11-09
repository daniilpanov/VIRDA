"""Export the merged pipeline ``Config`` as a JSON file."""

import json
from pathlib import Path

from virda.models.config import Config


def export_pipeline_config(path: str | Path, config: Config) -> Path:
    """Write *config*'s full ``model_dump(mode="json")`` dump to ``path``.

    Returns the written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8")
    return target
