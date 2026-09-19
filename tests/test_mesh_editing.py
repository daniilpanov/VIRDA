"""Unit tests for the atomic mesh-editing pipeline and its PLY loader."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from virda.io.loader.scalp_mesh_loader import load_scalp_mesh, save_scalp_mesh
from virda.models.scalp_mesh import ScalpMesh
from virda.pipelines.contracts import ContractValidationError
from virda.pipelines.mesh_editing import (
    MeshEditingPipeline,
    MeshEditingPipelineContract,
    edit_mesh,
)


def _tiny_mesh() -> ScalpMesh:
    icosphere = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    return ScalpMesh(
        vertices=np.asarray(icosphere.vertices, dtype=np.float64),
        faces=np.asarray(icosphere.faces, dtype=np.int64),
        face_adjacency=np.asarray(icosphere.face_adjacency, dtype=np.int64),
    )


class TestMeshEditingPipelineContract:
    def test_requires_scalp_mesh(self) -> None:
        contract = MeshEditingPipelineContract()
        with pytest.raises(ContractValidationError, match="scalp_mesh"):
            contract.validate_for_run()

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MeshEditingPipelineContract(scalp_mesh=_tiny_mesh(), density_percent=50)

    def test_accepts_none_smoother_type(self) -> None:
        contract = MeshEditingPipelineContract(scalp_mesh=_tiny_mesh(), smoother_type="none")
        contract.validate_for_run()
        assert contract.smoother_type == "none"

    def test_accepts_taubin_parameters(self) -> None:
        contract = MeshEditingPipelineContract(
            scalp_mesh=_tiny_mesh(),
            smoother_type="taubin",
            smoother_iterations=3,
            smoother_lamb=0.5,
            smoother_nu=-0.53,
        )
        contract.validate_for_run()
        assert contract.smoother_iterations == 3


class TestEditMesh:
    def test_identity_returns_input_instance_when_no_postprocessing(self) -> None:
        source = _tiny_mesh()
        contract = MeshEditingPipelineContract(
            scalp_mesh=source, smoother_type="none", mesh_density_percent=100.0
        )
        result = edit_mesh(contract)
        assert result is source

    def test_smoothing_returns_new_mesh_and_leaves_input_unchanged(self) -> None:
        source = _tiny_mesh()
        original_vertices = source.vertices.copy()
        contract = MeshEditingPipelineContract(
            scalp_mesh=source,
            smoother_type="laplacian",
            smoother_iterations=3,
            smoother_lamb=0.5,
            mesh_density_percent=100.0,
        )
        result = edit_mesh(contract)
        assert result is not source
        assert result.vertices.shape == source.vertices.shape
        assert not np.allclose(result.vertices, source.vertices, atol=1e-7)
        np.testing.assert_allclose(source.vertices, original_vertices, atol=0.0)

    def test_decimation_reduces_vertex_count(self) -> None:
        source = _tiny_mesh()
        contract = MeshEditingPipelineContract(
            scalp_mesh=source,
            smoother_type="laplacian",
            smoother_iterations=1,
            smoother_lamb=0.3,
            mesh_density_percent=25.0,
        )
        result = edit_mesh(contract)
        assert len(result.vertices) < len(source.vertices)

    def test_skips_decimation_at_full_density(self) -> None:
        source = _tiny_mesh()
        contract = MeshEditingPipelineContract(
            scalp_mesh=source,
            smoother_type="none",
            mesh_density_percent=100.0,
        )
        result = edit_mesh(contract)
        assert len(result.vertices) == len(source.vertices)


class TestScalpMeshLoader:
    def test_ply_round_trip_preserves_geometry(self, tmp_path) -> None:
        source = _tiny_mesh()
        target = save_scalp_mesh(tmp_path / "mesh.ply", source)
        loaded = load_scalp_mesh(target)

        assert np.allclose(loaded.vertices, source.vertices, atol=1e-9)
        assert np.allclose(loaded.faces, source.faces, atol=1e-9)
        assert loaded.faces.shape == source.faces.shape

    def test_load_rejects_non_surface(self, tmp_path) -> None:
        points = trimesh.PointCloud(np.zeros((3, 3)))
        path = tmp_path / "points.ply"
        points.export(str(path))
        with pytest.raises(ValueError, match="Not a triangular surface mesh"):
            load_scalp_mesh(path)

    def test_edit_mesh_runs_stage1_functions(self) -> None:
        source = _tiny_mesh()
        contract = MeshEditingPipelineContract(scalp_mesh=source, smoother_type="none")
        pipeline = MeshEditingPipeline(contract)
        context = pipeline.run_stage0()
        assert context.get_store_notnull(ScalpMesh) is source