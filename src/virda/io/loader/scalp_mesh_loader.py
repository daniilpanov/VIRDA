"""Read and write triangular surface meshes as :class:`ScalpMesh` objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from virda.models.scalp_mesh import ScalpMesh


def load_scalp_mesh(path: str | Path) -> ScalpMesh:
    """Load a triangular surface mesh (PLY/OBJ/STL/...) into a :class:`ScalpMesh`.

    Raises :class:`ValueError` when the file is not a single triangular
    :class:`trimesh.Trimesh`, e.g. a point cloud or a scene.
    """
    source = trimesh.load(Path(path))
    if not isinstance(source, trimesh.Trimesh):
        raise ValueError(f"Not a triangular surface mesh: {path}")
    return ScalpMesh(
        vertices=np.asarray(source.vertices, dtype=np.float64),
        faces=np.asarray(source.faces, dtype=np.int64),
        face_adjacency=np.asarray(source.face_adjacency, dtype=np.int64),
    )


def save_scalp_mesh(path: str | Path, mesh: ScalpMesh) -> Path:
    """Write *mesh* as a triangular PLY file; returns the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces).export(str(target))
    return target