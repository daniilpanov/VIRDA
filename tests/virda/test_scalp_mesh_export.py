from pathlib import Path

import numpy as np
import trimesh

from tests.helpers.meshes import make_sphere
from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.models.scalp_mesh import ScalpMesh


class TestExportScalpMesh:
    def test_writes_then_reads_back(self, tmp_path: Path) -> None:
        mesh = make_sphere()

        path = export_scalp_mesh(tmp_path / "out" / "scalp.ply", mesh)
        loaded = import_scalp_mesh(path)

        assert isinstance(loaded, ScalpMesh)
        np.testing.assert_array_almost_equal(loaded.vertices, mesh.vertices)
        np.testing.assert_array_equal(loaded.faces, mesh.faces)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "mesh.ply"

        path = export_scalp_mesh(nested, make_sphere())

        assert path == nested
        assert nested.is_file()

    def test_ply_is_triangular_geometry(self, tmp_path: Path) -> None:
        path = export_scalp_mesh(tmp_path / "mesh.ply", make_sphere())

        raw = trimesh.load(path)
        assert isinstance(raw, trimesh.Trimesh)
        assert raw.faces.shape[1] == 3

    def test_empty_mesh_writes_but_is_not_a_surface(self, tmp_path: Path) -> None:
        empty = ScalpMesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
        )

        path = export_scalp_mesh(tmp_path / "empty.ply", empty)

        assert path.is_file()
        assert not isinstance(trimesh.load(path), trimesh.Trimesh)
