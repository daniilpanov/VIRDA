from pathlib import Path

import trimesh

from virda.models.scalp_mesh import ScalpMesh
from virda.pipeline import Provider
from virda.pipeline_context import PipelineContext


class ScalpMeshVersioningProvider(Provider[ScalpMesh]):
    """Save every ScalpMesh update to mesh/versions/mesh-{n}.ply."""

    def __init__(self, versions_dir: Path) -> None:
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def provide(self, store: ScalpMesh | None, context: PipelineContext) -> None:
        if not store:
            return

        self._counter += 1
        path = self.versions_dir / f"mesh-{self._counter}.ply"
        tm = trimesh.Trimesh(vertices=store.vertices, faces=store.faces)
        tm.export(str(path))
