import numpy as np
import pytest

from tests.helpers.meshes import make_sphere
from virda.models.ese_mesh import ESEMesh
from virda.pipelines.contracts import ContractValidationError
from virda.pipelines.ese import ESEPipeline, ESEPipelineContract


@pytest.fixture
def contract() -> ESEPipelineContract:
    return ESEPipelineContract(scalp_mesh=make_sphere(), ese_offset_mm=2.0, k_neighbors=30)


class TestESEPipelineContract:
    def test_requires_scalp_mesh(self) -> None:
        with pytest.raises(ContractValidationError, match="scalp_mesh"):
            ESEPipelineContract(ese_offset_mm=2.0).validate_for_run()

    def test_requires_ese_offset(self) -> None:
        with pytest.raises(ContractValidationError, match="ese_offset_mm"):
            ESEPipelineContract(scalp_mesh=make_sphere()).validate_for_run()

    def test_rejects_bad_k_neighbors(self) -> None:
        contract = ESEPipelineContract(scalp_mesh=make_sphere(), ese_offset_mm=2.0, k_neighbors=1)
        with pytest.raises(ContractValidationError, match="k_neighbors"):
            contract.validate_for_run()

    def test_rejects_non_positive_offset(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            ESEPipelineContract(scalp_mesh=make_sphere(), ese_offset_mm=0.0)

    def test_uses_stage2_defaults_when_options_not_set(self) -> None:
        contract = ESEPipelineContract(scalp_mesh=make_sphere(), ese_offset_mm=2.0)
        stage2 = contract.to_stage2_config()
        assert stage2.neighborhood_radius_mm == 10.0
        assert stage2.k_neighbors is None
        assert stage2.use_weighted_pca is False
        assert stage2.pca_sigma_mm == 5.0
        assert stage2.min_neighbors == 5


class TestESEPipeline:
    def test_run_stage0_produces_ese_mesh(self, contract) -> None:
        pipeline = ESEPipeline(contract)
        context = pipeline.run_stage0()

        ese = context.get_store_notnull(ESEMesh)
        scalp = contract.scalp_mesh
        assert ese.vertices.shape == scalp.vertices.shape
        assert ese.vertices.shape[0] == scalp.vertices.shape[0]
        assert np.array_equal(ese.scalp_vertices, scalp.vertices)

    def test_ese_mesh_is_offset_outward(self, contract) -> None:
        pipeline = ESEPipeline(contract)
        context = pipeline.run_stage0()

        ese = context.get_store_notnull(ESEMesh)
        scalp = contract.scalp_mesh
        assert np.all(ese.vertices.shape == scalp.vertices.shape)
        assert ese.vertices.shape[0] > 0

    def test_stores_contract_in_context(self, contract) -> None:
        pipeline = ESEPipeline(contract)
        context = pipeline.run_stage0()

        stored = context.get_store(ESEPipelineContract)
        assert stored is contract

    def test_unknown_stage1_function(self, contract) -> None:
        pipeline = ESEPipeline(contract)
        context = pipeline.run_stage0()
        with pytest.raises(ValueError, match="store_normals"):
            pipeline.run_stage1("nope", context)

    def test_stage1_stores_vertices(self, contract, tmp_path) -> None:
        contract = contract.model_copy(update={"project_dir": tmp_path})
        pipeline = ESEPipeline(contract)
        context = pipeline.run_stage0()

        target = pipeline.run_stage1("store_vertices", context)
        assert target == tmp_path / "ese" / "ese_vertices.npy"
        assert np.array_equal(
            np.load(tmp_path / "ese" / "ese_vertices.npy"),
            context.get_store_notnull(ESEMesh).vertices,
        )

    def test_stage1_stores_faces_and_normals(self, contract, tmp_path) -> None:
        contract = contract.model_copy(update={"project_dir": tmp_path})
        pipeline = ESEPipeline(contract)
        context = pipeline.run_stage0()

        assert pipeline.run_stage1("store_faces", context) == tmp_path / "ese" / "ese_faces.npy"
        assert pipeline.run_stage1("store_normals", context) == tmp_path / "ese" / "normals.npy"

        ese = context.get_store_notnull(ESEMesh)
        assert np.array_equal(np.load(tmp_path / "ese" / "ese_faces.npy"), ese.faces)
        assert np.array_equal(np.load(tmp_path / "ese" / "normals.npy"), ese.normals)

    def test_stage1_store_requires_project_dir(self, contract) -> None:
        pipeline = ESEPipeline(contract)
        context = pipeline.run_stage0()
        with pytest.raises(ValueError, match="project_dir"):
            pipeline.run_stage1("store_vertices", context)
