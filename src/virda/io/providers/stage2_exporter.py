import json
from dataclasses import asdict
from logging import Logger
from pathlib import Path

import numpy as np
import trimesh

from virda.models.ese_mesh import ESEMesh
from virda.models.stage2_config import Stage2Config
from virda.pipeline import Provider


class Stage2Exporter(Provider[ESEMesh]):
    """Export Stage 2 (ESE) artifacts: mesh, arrays, point pairs, config."""

    def __init__(
        self,
        project_dir: Path,
        stage2_config: Stage2Config,
        logger: Logger | None = None,
    ) -> None:
        self.project = Path(project_dir)
        (self.project / "stage2").mkdir(parents=True, exist_ok=True)
        self._stage2_config = stage2_config
        self._logger = logger

    def provide(self, result: ESEMesh | None) -> None:
        if not result:
            raise ValueError("There is no result of Stage#2")

        stage2_dir = self.project / "stage2"

        mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces)
        mesh.export(str(stage2_dir / "ese_mesh.ply"))
        np.save(str(stage2_dir / "ese_vertices.npy"), result.vertices)
        np.save(str(stage2_dir / "ese_faces.npy"), result.faces)
        np.save(str(stage2_dir / "normals.npy"), result.normals)
        np.save(str(stage2_dir / "quality.npy"), result.quality)

        (stage2_dir / "point_pairs.json").write_text(
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

        (stage2_dir / "stage2_config.json").write_text(
            json.dumps({"stage2": asdict(self._stage2_config)}, indent=2)
        )
