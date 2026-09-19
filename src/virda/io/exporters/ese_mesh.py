"""Export an :class:`ESEMesh` to a PLY file."""

from pathlib import Path

import trimesh

from virda.models.ese_mesh import ESEMesh


def export_ese_mesh(path: str | Path, ese_mesh: ESEMesh) -> Path:
    """Write *ese_mesh* as a triangular PLY file; returns the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=ese_mesh.vertices, faces=ese_mesh.faces)
    mesh.export(str(target))
    return target