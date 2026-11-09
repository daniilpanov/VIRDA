"""Export an :class:`ESEMesh` to a PLY file plus project companions."""

from pathlib import Path

import numpy as np
import trimesh

from virda.models.ese_mesh import ESEMesh


def export_ese_mesh(path: str | Path, ese_mesh: ESEMesh) -> Path:
    """Write *ese_mesh* as a triangular PLY file; returns the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=ese_mesh.vertices, faces=ese_mesh.faces)
    mesh.export(str(target))
    return target


def export_ese_companions(path: str | Path, ese_mesh: ESEMesh) -> Path:
    """Write the ``ese/`` companion files next to the PLY at *path*.

    Writes ``ese_vertices.npy`` and ``ese_faces.npy`` so
    :func:`virda.io.importers.ese_mesh.import_ese_mesh` can rebuild the mesh
    later.  Returns the directory holding the companions.
    """
    root = Path(path).parent
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "ese_vertices.npy", np.asarray(ese_mesh.vertices, dtype=np.float64))
    np.save(root / "ese_faces.npy", np.asarray(ese_mesh.faces, dtype=np.int64))
    return root
