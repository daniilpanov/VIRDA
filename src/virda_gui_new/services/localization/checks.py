"""Check the localization inputs against the scalp surface.

The fiducials are marked on the patient's skin, so they belong to the
scalp surface -- not to the ESE surface the electrodes sit on. The
distance of every fiducial from the scalp mesh tells the user whether
the marked point is where it should be.
"""

from dataclasses import dataclass

from virda.geometry import Geometry
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.mesh import Mesh
from virda_gui_new.services.localization.build import FiducialRows


@dataclass(frozen=True)
class SurfaceStatus:
    """How far a fiducial is from the scalp surface, and whether it is on it."""

    distance_mm: float
    on_surface: bool


def fiducial_surface_status(
    rows: FiducialRows,
    mesh: Mesh,
    geometry: Geometry,
    tolerance_mm: float,
) -> dict[str, SurfaceStatus]:
    """The surface status of every fiducial, by name.

    The keys are the row names, so the table can show the status next
    to the fiducial it belongs to. Whether a fiducial is on the surface
    is the core service's rule: the table only renders it.
    """
    status: dict[str, SurfaceStatus] = {}
    for name, xyz in rows:
        point = Point(
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]),
            system=CoordinateSystem.WORLD,
        )
        status[str(name).strip()] = SurfaceStatus(
            distance_mm=geometry.distance_to_surface(point, mesh),
            on_surface=geometry.point_on_surface(point, mesh, tolerance_mm=tolerance_mm),
        )
    return status
