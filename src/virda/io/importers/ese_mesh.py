"""Import an ESE surface mesh from a project ``ese/`` directory."""

from pathlib import Path

import numpy as np

from virda.models.ese_mesh import ESEMesh


def import_ese_mesh(path: str | Path) -> ESEMesh:
    """Rebuild an :class:`ESEMesh` from ``mesh.ply`` companion files.

    Reads ``ese_vertices.npy``, ``ese_faces.npy``, ``quality.npy`` and
    ``scalp_vertices.npy`` next to *path* (the project ``ese/`` layout); the
    PLY itself only carries the surface geometry and is not parsed.

    Raises :class:`ValueError` when a companion file is missing or malformed.
    """
    ply_path = Path(path)
    root = ply_path.parent
    if not ply_path.is_file():
        raise ValueError(f"ESE mesh file not found: {ply_path}")
    try:
        vertices = np.asarray(np.load(root / "ese_vertices.npy"), dtype=np.float64)
        faces = np.asarray(np.load(root / "ese_faces.npy"), dtype=np.int64)
        quality = np.asarray(np.load(root / "quality.npy"), dtype=np.float64)
        scalp_vertices = np.asarray(np.load(root / "scalp_vertices.npy"), dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Cannot import ESE mesh from {ply_path}: {exc}") from exc
    try:
        return ESEMesh(
            vertices=vertices,
            faces=faces,
            scalp_vertices=scalp_vertices,
            quality=quality,
        )
    except ValueError as exc:
        raise ValueError(f"Cannot import ESE mesh from {ply_path}: {exc}") from exc
