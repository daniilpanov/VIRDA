"""Step: auto-detect anatomical fiducials from a scalp mesh."""

from virda.fiducials.detect import find_fiducials, to_fiducials
from virda.models.fiducial import AutoDetectedFiducials, Fiducials
from virda.models.scalp_mesh import ScalpMesh


def auto_detect_fiducials(mesh: ScalpMesh) -> AutoDetectedFiducials:
    """Detect the NAS/LPA/RPA fiducial points on the mesh surface."""
    points = find_fiducials(mesh.vertices)
    return AutoDetectedFiducials(fiducials=Fiducials(items=to_fiducials(points)))
