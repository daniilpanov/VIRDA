from pathlib import Path

import trimesh
from trimesh.exchange.ply import export_ply as _export_ply_bytes

from virda.models.scalp_mesh import ScalpMesh


def export_ply(path: Path, mesh: ScalpMesh, binary: bool = False) -> None:
    trimesh_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
    )
    encoding = "binary" if binary else "ascii"
    data = _export_ply_bytes(trimesh_mesh, encoding=encoding)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
