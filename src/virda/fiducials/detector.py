from virda.fiducials.detect import find_fiducials, to_fiducials
from virda.models.fiducial import AutoDetectedFiducials, Fiducials
from virda.models.scalp_mesh import ScalpMesh
from virda.pipeline_context import PipelineContext


class AutoFiducialsDetector:
    def run(self, context: PipelineContext) -> AutoDetectedFiducials:
        mesh = context.get_store_notnull(ScalpMesh)
        points = find_fiducials(mesh.vertices)
        return AutoDetectedFiducials(fiducials=Fiducials(items=to_fiducials(points)))
