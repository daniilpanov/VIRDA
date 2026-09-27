"""Export an :class:`ESEMesh` to a PLY file plus project companions."""

import json
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

    Writes ``ese_vertices.npy``, ``ese_faces.npy``, ``normals.npy``,
    ``quality.npy`` and ``point_pairs.json`` so :func:`virda.io.importers
    .ese_mesh.import_ese_mesh` can rebuild the mesh later.  Returns the
    directory holding the companions.
    """
    root = Path(path).parent
    root.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(ese_mesh.vertices, dtype=np.float64)
    faces = np.asarray(ese_mesh.faces, dtype=np.int64)
    normals = np.asarray(ese_mesh.normals, dtype=np.float64)
    quality = np.asarray(ese_mesh.quality, dtype=np.float64)
    scalp_vertices = np.asarray(ese_mesh.scalp_vertices, dtype=np.float64)
    np.save(root / "ese_vertices.npy", vertices)
    np.save(root / "ese_faces.npy", faces)
    np.save(root / "normals.npy", normals)
    np.save(root / "quality.npy", quality)
    pairs = {
        "n_points": int(vertices.shape[0]),
        "scalp_vertices": scalp_vertices.tolist(),
        "ese_vertices": vertices.tolist(),
        "normals": normals.tolist(),
        "quality": quality.tolist(),
    }
    (root / "point_pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    return root
