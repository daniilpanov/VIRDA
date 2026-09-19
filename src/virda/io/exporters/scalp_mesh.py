"""Export a triangular scalp surface mesh to a PLY file."""

from pathlib import Path

import trimesh

from virda.models.scalp_mesh import ScalpMesh


def export_scalp_mesh(path: str | Path, mesh: ScalpMesh) -> Path:
    """Write *mesh* as a triangular PLY file; returns the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces).export(str(target))
    return target