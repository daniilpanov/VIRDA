from pathlib import Path
from typing import Any

import numpy as np

from virda.io.fiducial_helpers import save_fiducials
from virda.models.fiducial import Fiducial, Fiducials
from virda.pipeline_context import PipelineContext


def build_context(**stores: Any) -> PipelineContext:
    return PipelineContext({type(value): value for value in stores.values()})


def make_fiducials() -> Fiducials:
    return Fiducials(
        items=[
            Fiducial(
                fiducial_id="NAS",
                name="Nasion",
                coordinates=np.array([0.0, 88.0, -10.0]),
                coordinate_system="world",
                definition_method="manual",
            ),
            Fiducial(
                fiducial_id="LPA",
                name="Left pre-auricular",
                coordinates=np.array([-75.0, -1.0, -14.0]),
                coordinate_system="world",
            ),
        ]
    )


def make_auto_fiducials() -> Fiducials:
    return Fiducials(
        items=[
            Fiducial(
                fiducial_id="NAS",
                name="Nasion",
                coordinates=np.array([0.0, 85.0, -12.0]),
                coordinate_system="world",
                definition_method="auto",
            ),
            Fiducial(
                fiducial_id="LPA",
                name="Left pre-auricular",
                coordinates=np.array([-74.0, -2.0, -13.0]),
                coordinate_system="world",
                definition_method="auto",
            ),
        ]
    )


def save_test_fiducials(path: Path) -> Path:
    save_fiducials(path, make_fiducials())
    return path
