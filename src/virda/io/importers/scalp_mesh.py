"""Import a triangular scalp surface mesh from a PLY/OBJ/STL file."""

from pathlib import Path

import numpy as np
import trimesh

from virda.models.mesh import Mesh


def import_scalp_mesh(path: str | Path) -> Mesh:
    """Read a triangular surface mesh (PLY/OBJ/STL/...) into a :class:`Mesh`.

    Raises :class:`ValueError` when the file is not a single triangular
    :class:`trimesh.Trimesh`, e.g. a point cloud or a scene.
    """
    source = trimesh.load(Path(path))
    if not isinstance(source, trimesh.Trimesh):
        raise ValueError(f"Not a triangular surface mesh: {path}")
    return Mesh(
        vertices=np.asarray(source.vertices, dtype=np.float64),
        faces=np.asarray(source.faces, dtype=np.int64),
    )
