from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.pipelines.contracts import ContractValidationError
from virda.pipelines.mesh_generate import MeshPipeline, MeshPipelineContract


@pytest.fixture
def synthetic_nifti_path(tmp_path: Path) -> Path:
    volume_shape = (20, 20, 20)
    center = np.array([10, 10, 10])
    sphere_radius = 8
    grid_indices = np.indices(volume_shape)
    squared_distance = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    is_inside_sphere = squared_distance <= sphere_radius**2

    image_data = np.zeros(volume_shape, dtype=np.float32)
    image_data[is_inside_sphere] = 100.0

    nifti_image = nib.Nifti1Image(image_data, np.eye(4))
    nifti_file_path = tmp_path / "synthetic.nii.gz"
    nib.save(nifti_image, nifti_file_path)
    return nifti_file_path


def build_pipeline(nifti_path: Path, **contract_overrides) -> MeshPipeline:
    contract = MeshPipelineContract(nifti_path=nifti_path, **contract_overrides)
    return MeshPipeline(contract=contract)


def assert_valid_mesh(mesh: ScalpMesh) -> None:
    assert mesh.vertices.ndim == 2 and mesh.vertices.shape[1] == 3
    assert mesh.faces.ndim == 2 and mesh.faces.shape[1] == 3
    n_faces = mesh.faces.shape[0]
    if n_faces:
        assert mesh.faces.min() >= 0 and mesh.faces.max() < len(mesh.vertices)
    assert mesh.face_adjacency.shape == (0, 2) or (
        mesh.face_adjacency.shape[1] == 2 and mesh.face_adjacency.max() < n_faces
    )


def test_contract_rejects_missing_nifti() -> None:
    contract = MeshPipelineContract()

    with pytest.raises(ContractValidationError) as excinfo:
        contract.validate_for_run()

    assert "nifti_path" in str(excinfo.value)


def test_stage0_produces_mesh_stores(synthetic_nifti_path: Path) -> None:
    context = build_pipeline(synthetic_nifti_path).run_stage0()

    assert isinstance(context.get_store_notnull(MRIVolume), MRIVolume)
    assert isinstance(context.get_store_notnull(SegmentationMask), SegmentationMask)
    mesh = context.get_store_notnull(ScalpMesh)
    assert isinstance(mesh, ScalpMesh)
    assert_valid_mesh(mesh)

    stored_contract = context.stores[MeshPipelineContract]
    assert stored_contract == build_pipeline(synthetic_nifti_path).contract


def test_stage0_alone_does_not_postprocess(synthetic_nifti_path: Path) -> None:
    context = build_pipeline(synthetic_nifti_path).run_stage0()
    mesh = context.get_store_notnull(ScalpMesh)

    assert "clean" in build_pipeline(synthetic_nifti_path).stage1_names
    assert mesh.face_adjacency.shape[1] == 2


def test_smooth_stage1_replaces_mesh_store(synthetic_nifti_path: Path) -> None:
    pipeline = build_pipeline(synthetic_nifti_path)
    context = pipeline.run_stage0()
    raw_mesh = context.get_store_notnull(ScalpMesh)

    smoothed = pipeline.run_stage1("smooth", context)

    assert isinstance(smoothed, ScalpMesh)
    assert_valid_mesh(smoothed)
    assert context.get_store_notnull(ScalpMesh) is smoothed
    assert smoothed.vertices.shape == raw_mesh.vertices.shape


def test_clean_stage1_replaces_mesh_store(synthetic_nifti_path: Path) -> None:
    pipeline = build_pipeline(synthetic_nifti_path)
    context = pipeline.run_stage0()

    cleaned = pipeline.run_stage1("clean", context)

    assert isinstance(cleaned, ScalpMesh)
    assert_valid_mesh(cleaned)
    assert context.get_store_notnull(ScalpMesh) is cleaned


def test_unknown_stage1_function_raises(synthetic_nifti_path: Path) -> None:
    pipeline = build_pipeline(synthetic_nifti_path)
    context = pipeline.run_stage0()

    with pytest.raises(ValueError) as excinfo:
        pipeline.run_stage1("no_such_function", context)

    assert "no_such_function" in str(excinfo.value)


def test_stage1_names_list_available_functions(synthetic_nifti_path: Path) -> None:
    pipeline = build_pipeline(synthetic_nifti_path)

    assert set(pipeline.stage1_names) == {"clean", "smooth"}
