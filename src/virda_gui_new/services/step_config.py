"""Saving and loading the per-step configuration of a project.

Each pipeline step has settings of its own (the sealing options, the
mesh postprocessing parameters, the ESE offset). The GUI keeps them in
``config/<step>.json`` next to the project artifacts, so reopening a
project restores the settings the step was last run with.
"""

import json
from pathlib import Path
from typing import Any

from virda_gui_new.dto.steps import Step

#: Project subdirectory that holds the per-step configuration files.
_CONFIG_DIR = "config"


def config_path(project: Path, step: Step) -> Path:
    """Path of the configuration file of ``step``."""
    name = step.name.lower().replace(" ", "_")
    return project / _CONFIG_DIR / f"{name}.json"


def save_step_config(project: Path, step: Step, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as the JSON configuration of ``step``.

    The file is written eagerly, so the settings survive a crash. The
    written path is returned.
    """
    target = config_path(project, step)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_step_config(project: Path, step: Step) -> dict[str, Any]:
    """Read the configuration of ``step``.

    An empty dict is returned when the project has no configuration file
    for the step, or when the file cannot be read or parsed: the step
    then starts from its defaults.
    """
    target = config_path(project, step)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}
    return document if isinstance(document, dict) else {}
