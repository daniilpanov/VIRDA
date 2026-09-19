import numpy as np

from virda.models.fiducial import Fiducial, Fiducials


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