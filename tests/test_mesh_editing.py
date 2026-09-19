"""Unit tests for the atomic mesh-editing ops and the PLY loader."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.mesh_decimator import QuadraticDecimator
from virda.models.scalp_mesh import ScalpMesh


def _tiny_mesh() -> ScalpMesh:
    icosphere = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    return ScalpMesh(
        vertices=np.asarray(icosphere.vertices, dtype=np.float64),
        faces=np.asarray(icosphere.faces, dtype=np.int64),
        face_adjacency=np.asarray(icosphere.face_adjacency, dtype=np.int64),
    )


class TestEditMesh:
    def test_full_density_decimation_returns_input_instance(self) -> None:
        source = _tiny_mesh()
        result = QuadraticDecimator(density_percent=100.0).process(source)
        assert result is source

    def test_smoothing_returns_new_mesh_and_leaves_input_unchanged(self) -> None:
        source = _tiny_mesh()
        original_vertices = source.vertices.copy()
        result = LaplacianSmoother(iterations=3, lamb=0.5).process(source)
        assert result is not source
        assert result.vertices.shape == source.vertices.shape
        assert not np.allclose(result.vertices, source.vertices, atol=1e-7)
        np.testing.assert_allclose(source.vertices, original_vertices, atol=0.0)

    def test_decimation_reduces_vertex_count(self) -> None:
        source = _tiny_mesh()
        result = QuadraticDecimator(density_percent=25.0).process(source)
        assert len(result.vertices) < len(source.vertices)

    def test_skips_decimation_at_full_density(self) -> None:
        source = _tiny_mesh()
        result = QuadraticDecimator(density_percent=100.0).process(source)
        assert len(result.vertices) == len(source.vertices)


class TestScalpMeshImporter:
    def test_ply_round_trip_preserves_geometry(self, tmp_path) -> None:
        source = _tiny_mesh()
        target = export_scalp_mesh(tmp_path / "mesh.ply", source)
        loaded = import_scalp_mesh(target)

        # Binary PLY stores vertices as float32, so allow ~float32 roundoff.
        assert np.allclose(loaded.vertices, source.vertices, atol=1e-5)
        assert np.allclose(loaded.faces, source.faces, atol=1e-9)
        assert loaded.faces.shape == source.faces.shape

    def test_load_rejects_non_surface(self, tmp_path) -> None:
        points = trimesh.PointCloud(np.zeros((3, 3)))
        path = tmp_path / "points.ply"
        points.export(str(path))
        with pytest.raises(ValueError, match="Not a triangular surface mesh"):
            import_scalp_mesh(path)