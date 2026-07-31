import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic_settings import BaseSettings

from virda.models.fiducial import Fiducial


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def save_fiducials(path: Path, fiducials: list[Fiducial]) -> None:
    data = {
        "fiducials": [
            {
                "fiducial_id": f.fiducial_id,
                "name": f.name,
                "coordinates": f.coordinates.tolist(),
                "coordinate_system": f.coordinate_system,
                "definition_method": f.definition_method,
            }
            for f in fiducials
        ]
    }
    save_json(path, data)


def load_fiducials(path: Path) -> list[Fiducial]:
    data = load_json(path)
    return [
        Fiducial(
            fiducial_id=item["fiducial_id"],
            name=item["name"],
            coordinates=np.asarray(item["coordinates"], dtype=np.float64),
            coordinate_system=item["coordinate_system"],
            definition_method=item["definition_method"],
        )
        for item in data["fiducials"]
    ]


def save_config(path: Path, config: BaseSettings | dict[str, Any]) -> None:
    data = config.model_dump() if isinstance(config, BaseSettings) else config
    save_json(path, data)


def load_config[T: BaseSettings](path: Path, model_cls: type[T]) -> T:
    return model_cls(**load_json(path))
