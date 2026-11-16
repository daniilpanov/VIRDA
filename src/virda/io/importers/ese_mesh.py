"""Import an ESE surface mesh from a project ``ese/`` directory."""

from pathlib import Path

import numpy as np

from virda.models.mesh import Mesh, Normals


def import_ese_normals(path: str | Path) -> Normals:
    """Read the per-vertex normals from the .npy file at *path*.

    Raises :class:`ValueError` when the file is missing or malformed.
    """
    source = Path(path)
    try:
        normals = np.asarray(np.load(source), dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot import the ESE normals from {source}: {exc}") from exc
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"ESE normals must be (N, 3) arrays, got shape {normals.shape}")
    return normals


def import_ese_mesh(path: str | Path) -> Mesh:
    """Rebuild an :class:`Mesh` from ``mesh.ply`` companion files.

    Reads ``ese_vertices.npy`` and ``ese_faces.npy`` next to *path* (the
    project ``ese/`` layout); the PLY itself only carries the surface geometry
    and is not parsed.

    Raises :class:`ValueError` when a companion file is missing or malformed.
    """
    ply_path = Path(path)
    root = ply_path.parent
    if not ply_path.is_file():
        raise ValueError(f"ESE mesh file not found: {ply_path}")
    try:
        vertices = np.asarray(np.load(root / "ese_vertices.npy"), dtype=np.float64)
        faces = np.asarray(np.load(root / "ese_faces.npy"), dtype=np.int64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Cannot import ESE mesh from {ply_path}: {exc}") from exc
    try:
        return Mesh(
            vertices=vertices,
            faces=faces,
        )
    except ValueError as exc:
        raise ValueError(f"Cannot import ESE mesh from {ply_path}: {exc}") from exc
