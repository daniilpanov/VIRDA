"""Export the Stage 2 (ESE) artifacts into a patient project directory."""

from pathlib import Path

import numpy as np
import trimesh

from virda.models.ese_mesh import ESEMesh


def export_ese(result: ESEMesh, project_dir: Path) -> None:
    """Write the ESE mesh and its vertex/face/normal arrays below ``project_dir``."""
    ese_dir = Path(project_dir) / "ese"
    ese_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces)
    mesh.export(str(ese_dir / "ese_mesh.ply"))
    np.save(str(ese_dir / "ese_vertices.npy"), result.vertices)
    np.save(str(ese_dir / "ese_faces.npy"), result.faces)
    np.save(str(ese_dir / "normals.npy"), result.normals)
