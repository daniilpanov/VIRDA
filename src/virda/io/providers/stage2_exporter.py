import json
from pathlib import Path

import numpy as np
import trimesh

from virda.models.ese_mesh import ESEMesh
from virda.pipeline import Provider
from virda.pipeline_context import PipelineContext


class Stage2Exporter(Provider[ESEMesh]):
    """Export Stage 2 (ESE) artifacts: mesh, arrays, point pairs."""

    def __init__(self, project_dir: Path) -> None:
        self.project = Path(project_dir)
        (self.project / "ese").mkdir(parents=True, exist_ok=True)

    def provide(self, result: ESEMesh | None, context: PipelineContext) -> None:
        if not result:
            raise ValueError("There is no result of Stage#2")

        ese_dir = self.project / "ese"

        mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces)
        mesh.export(str(ese_dir / "ese_mesh.ply"))
        np.save(str(ese_dir / "ese_vertices.npy"), result.vertices)
        np.save(str(ese_dir / "ese_faces.npy"), result.faces)
        np.save(str(ese_dir / "normals.npy"), result.normals)
        np.save(str(ese_dir / "quality.npy"), result.quality)

        (ese_dir / "point_pairs.json").write_text(
            json.dumps(
                {
                    "n_points": int(result.vertices.shape[0]),
                    "scalp_vertices": result.scalp_vertices.tolist(),
                    "ese_vertices": result.vertices.tolist(),
                    "normals": result.normals.tolist(),
                    "quality": result.quality.tolist(),
                },
                indent=2,
            )
        )
