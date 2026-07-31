from pathlib import Path
from typing import cast

import numpy as np
import pytest
import trimesh

from virda.io.exporter.ply_exporter import export_ply
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.scalp_mesh import ScalpMesh


@pytest.fixture
def sphere_mesh() -> ScalpMesh:
    grid = np.indices((20, 20, 20))
    center = np.array([10, 10, 10])
    squared_distance = np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    mask = squared_distance <= 8**2
    return MarchingCubesExtractor().extract(mask, np.eye(4))


class TestPlyExporter:
    def test_export_round_trip(self, tmp_path: Path, sphere_mesh: ScalpMesh) -> None:
        path = tmp_path / "mesh.ply"

        export_ply(path, sphere_mesh)

        assert path.exists()
        loaded = cast(trimesh.Trimesh, trimesh.load(path))
        np.testing.assert_allclose(loaded.vertices, sphere_mesh.vertices)
        np.testing.assert_allclose(loaded.faces, sphere_mesh.faces)

    def test_export_binary(self, tmp_path: Path, sphere_mesh: ScalpMesh) -> None:
        path = tmp_path / "mesh_binary.ply"

        export_ply(path, sphere_mesh, binary=True)

        assert path.exists()
        assert path.read_bytes().startswith(b"ply\nformat binary")
        loaded = cast(trimesh.Trimesh, trimesh.load(path))
        np.testing.assert_allclose(loaded.vertices, sphere_mesh.vertices, atol=1e-4)
        np.testing.assert_allclose(loaded.faces, sphere_mesh.faces)

    def test_export_creates_parent_directories(
        self, tmp_path: Path, sphere_mesh: ScalpMesh
    ) -> None:
        path = tmp_path / "nested" / "dirs" / "mesh.ply"

        export_ply(path, sphere_mesh)

        assert path.exists()
