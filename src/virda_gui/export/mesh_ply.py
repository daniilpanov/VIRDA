"""Export an in-memory triangle mesh to a PLY file anywhere on disk.

Unlike the canonical project exporters in :mod:`virda.io.exporters`, this
helper writes an explicit user-picked path (a Save dialog default), so both
the mesh-processing and the ESE tabs share it instead of each vendoring
the same ``trimesh`` snippet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_mesh_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Write *vertices*/*faces* as a triangular PLY mesh; return the target."""
    import trimesh

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    ).export(str(target))
    return target
