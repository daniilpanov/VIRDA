import json
from pathlib import Path
from typing import Any

import numpy as np

from tests.helpers.meshes import make_sphere
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducial, Fiducials, FiducialType
from virda.models.mesh import Mesh, Normals
from virda.models.stage2_config import Stage2Config
from virda.ops.build_ese import build_ese


def make_fiducials() -> Fiducials:
    return Fiducials(
        items=[
            Fiducial(
                fiducial_id=FiducialType.NAS,
                name="Nasion",
                point=Point(x=0.0, y=88.0, z=-10.0, system=CoordinateSystem.WORLD),
            ),
            Fiducial(
                fiducial_id=FiducialType.LPA,
                name="Left pre-auricular",
                point=Point(x=-75.0, y=-1.0, z=-14.0, system=CoordinateSystem.WORLD),
            ),
            Fiducial(
                fiducial_id=FiducialType.RPA,
                name="Right pre-auricular",
                point=Point(x=75.0, y=-1.0, z=-14.0, system=CoordinateSystem.WORLD),
            ),
        ]
    )


def make_ese(
    scalp_mesh: Mesh | None = None, ese_offset_mm: float = 2.0
) -> tuple[Mesh, Normals, Mesh]:
    mesh = scalp_mesh or make_sphere()
    ese, normals = build_ese(mesh, config=Stage2Config(k_neighbors=20), ese_offset_mm=ese_offset_mm)
    return ese, normals, mesh


def make_electrodes(points: np.ndarray, fiducials: Fiducials) -> Electrodes:
    """Build electrodes whose measured distances are exact to the given points."""
    items: list[Electrode] = []
    for i, point in enumerate(points):
        distances = {
            fiducial.fiducial_id: float(np.linalg.norm(point - fiducial.point.as_array()))
            for fiducial in fiducials.items
        }
        items.append(Electrode(electrode_id=f"E{i}", measured_distances=distances))
    return Electrodes(items=items)


def make_measurements_file(
    path: Path,
    points: np.ndarray,
    fiducials: Fiducials | None = None,
    weights: dict[str, float] | None = None,
) -> Path:
    """Write a Stage 3 measurements JSON with exact distances to ``points``.

    Optionally includes per-fiducial ``fiducial_weights``.
    """
    fiducials = fiducials or make_fiducials()
    data: dict[str, Any] = {
        "electrodes": [
            {
                "electrode_id": f"E{i}",
                "measured_distances": {
                    fiducial.fiducial_id: float(np.linalg.norm(point - fiducial.point.as_array()))
                    for fiducial in fiducials.items
                },
            }
            for i, point in enumerate(points)
        ]
    }
    if weights:
        data["fiducial_weights"] = weights
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path
