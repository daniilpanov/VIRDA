"""Import an ESE surface mesh from a project ``ese/`` directory."""

import json
from pathlib import Path

import numpy as np

from virda.models.ese_mesh import ESEMesh


def import_ese_mesh(path: str | Path) -> ESEMesh:
    """Rebuild an :class:`ESEMesh` from ``ese_mesh.ply`` companion files.

    Reads ``ese_vertices.npy``, ``ese_faces.npy``, ``normals.npy``,
    ``quality.npy`` and the ``scalp_vertices`` dump of ``point_pairs.json``
    next to *path* (the project ``ese/`` layout); the PLY itself only carries
    the surface geometry and is not parsed.

    Raises :class:`ValueError` when a companion file is missing or malformed.
    """
    ply_path = Path(path)
    root = ply_path.parent
    if not ply_path.is_file():
        raise ValueError(f"ESE mesh file not found: {ply_path}")
    try:
        vertices = np.asarray(np.load(root / "ese_vertices.npy"), dtype=np.float64)
        faces = np.asarray(np.load(root / "ese_faces.npy"), dtype=np.int64)
        normals = np.asarray(np.load(root / "normals.npy"), dtype=np.float64)
        quality = np.asarray(np.load(root / "quality.npy"), dtype=np.float64)
        with open(root / "point_pairs.json", encoding="utf-8") as handle:
            pairs = json.load(handle)
        scalp_vertices = np.asarray(pairs["scalp_vertices"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Cannot import ESE mesh from {ply_path}: {exc}") from exc
    try:
        return ESEMesh(
            vertices=vertices,
            faces=faces,
            scalp_vertices=scalp_vertices,
            normals=normals,
            quality=quality,
        )
    except ValueError as exc:
        raise ValueError(f"Cannot import ESE mesh from {ply_path}: {exc}") from exc
