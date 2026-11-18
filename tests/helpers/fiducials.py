from virda.models.coordinates import CoordinateSystem, Point
from virda.models.fiducial import Fiducial, Fiducials, FiducialType


def make_fiducials() -> Fiducials:
    return Fiducials(
        items=[
            Fiducial(
                fiducial_id=FiducialType.NAS,
                name="Nasion",
                point=Point(x=0.0, y=88.0, z=-10.0, system=CoordinateSystem.WORLD),
                definition_method="manual",
            ),
            Fiducial(
                fiducial_id=FiducialType.LPA,
                name="Left pre-auricular",
                point=Point(x=-75.0, y=-1.0, z=-14.0, system=CoordinateSystem.WORLD),
            ),
        ]
    )
