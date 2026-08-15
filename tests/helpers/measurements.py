import json
from pathlib import Path
from typing import Any

import numpy as np

from tests.helpers.meshes import make_sphere
from tests.helpers.pipelines import build_context
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.models.electrode import Electrode, Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage2_config import Stage2Config


def make_fiducials() -> Fiducials:
    return Fiducials(
        items=[
            Fiducial(
                fiducial_id="NAS",
                name="Nasion",
                coordinates=np.array([0.0, 88.0, -10.0]),
                coordinate_system="world",
            ),
            Fiducial(
                fiducial_id="LPA",
                name="Left pre-auricular",
                coordinates=np.array([-75.0, -1.0, -14.0]),
                coordinate_system="world",
            ),
            Fiducial(
                fiducial_id="RPA",
                name="Right pre-auricular",
                coordinates=np.array([75.0, -1.0, -14.0]),
                coordinate_system="world",
            ),
        ]
    )


def make_ese(scalp_mesh: ScalpMesh | None = None, ese_offset_mm: float = 2.0) -> ESEMesh:
    mesh = scalp_mesh or make_sphere()
    builder = PCAESEBuilder(config=Stage2Config(k_neighbors=20), ese_offset_mm=ese_offset_mm)
    return builder.run(build_context(scalp_mesh=mesh))


def make_electrodes(points: np.ndarray, fiducials: Fiducials) -> Electrodes:
    """Build electrodes whose measured distances are exact to the given points."""
    items: list[Electrode] = []
    for i, point in enumerate(points):
        distances = {
            fiducial.fiducial_id: float(np.linalg.norm(point - fiducial.coordinates))
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
                    fiducial.fiducial_id: float(np.linalg.norm(point - fiducial.coordinates))
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
