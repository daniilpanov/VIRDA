import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tests.helpers.meshes import make_sphere
from tests.helpers.pipelines import save_test_fiducials
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.models.config import Config
from virda.models.ese_mesh import ESEMesh
from virda.models.stage1_result import Stage1Result
from virda.models.stage2_config import Stage2Config
from virda.pipelines.stage2 import Stage2PipelineBuilder

ESE_OFFSET_MM = 2.0


def build_pipeline(tmp_path, config: Stage2Config):
    builder = PCAESEBuilder(config=config, ese_offset_mm=ESE_OFFSET_MM)
    return Stage2PipelineBuilder(
        ese_builder=builder,
        scalp_mesh=make_sphere(),
        project_dir=tmp_path,
    ).build()


class TestStage2Pipeline:
    def test_run_builds_ese_mesh(self, tmp_path) -> None:
        config = Stage2Config(k_neighbors=30)
        pipeline = build_pipeline(tmp_path, config)

        context = pipeline.run()
        ese_mesh = context.get_store_notnull(ESEMesh)

        assert ese_mesh.vertices.shape == ese_mesh.scalp_vertices.shape
        assert ese_mesh.vertices.shape[0] > 0

    def test_run_exports_artifacts(self, tmp_path) -> None:
        config = Stage2Config(k_neighbors=30)
        pipeline = build_pipeline(tmp_path, config)

        context = pipeline.run()
        ese_mesh = context.get_store_notnull(ESEMesh)

        ese_dir = tmp_path / "ese"
        assert (ese_dir / "ese_mesh.ply").exists()
        assert np.array_equal(np.load(ese_dir / "ese_vertices.npy"), ese_mesh.vertices)
        assert np.array_equal(np.load(ese_dir / "ese_faces.npy"), ese_mesh.faces)
        assert np.array_equal(np.load(ese_dir / "normals.npy"), ese_mesh.normals)
        assert np.array_equal(np.load(ese_dir / "quality.npy"), ese_mesh.quality)

        pairs = json.loads((ese_dir / "point_pairs.json").read_text())
        assert pairs["n_points"] == ese_mesh.vertices.shape[0]
        assert len(pairs["scalp_vertices"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["ese_vertices"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["normals"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["quality"]) == ese_mesh.vertices.shape[0]

    def test_from_config_builds_pipeline(self, tmp_path) -> None:
        config = Config(
            nifti_path="/dev/null",
            project_dir=str(tmp_path),
            n_electrodes=32,
            ese_offset_mm=ESE_OFFSET_MM,
            ese_reference="electrode_body_center",
            k_neighbors=30,
        )
        scalp_mesh = make_sphere()
        builder = Stage2PipelineBuilder.from_config(config=config, scalp_mesh=scalp_mesh)
        pipeline = builder.build()

        context = pipeline.run()
        ese_mesh = context.get_store_notnull(ESEMesh)

        assert ese_mesh.vertices.shape[0] > 0
        assert (tmp_path / "ese" / "ese_mesh.ply").exists()

    def test_from_config_raises_without_ese(self, tmp_path) -> None:
        config = Config(nifti_path="/dev/null", project_dir=str(tmp_path))
        scalp_mesh = make_sphere()

        with pytest.raises(ValueError, match="ESE is not configured"):
            Stage2PipelineBuilder.from_config(config=config, scalp_mesh=scalp_mesh)

    def test_from_config_raises_without_project_dir(self, tmp_path) -> None:
        config = Config(
            n_electrodes=32,
            ese_offset_mm=ESE_OFFSET_MM,
            ese_reference="electrode_body_center",
            k_neighbors=30,
        )
        scalp_mesh = make_sphere()

        with pytest.raises(ValueError, match="Project directory"):
            Stage2PipelineBuilder.from_config(config=config, scalp_mesh=scalp_mesh)


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


@pytest.fixture
def fiducials_file(tmp_path: Path) -> Path:
    return save_test_fiducials(tmp_path / "fiducials.json")


class TestMainRunIntegration:
    def test_run_with_ese_returns_ese_mesh(
        self,
        synthetic_nifti_path: Path,
        fiducials_file: Path,
        tmp_path: Path,
    ) -> None:
        config = Config(
            nifti_path=str(synthetic_nifti_path),
            project_dir=str(tmp_path),
            fiducials_path=str(fiducials_file),
            closing_radius=0,
            seal_enabled=False,
            n_electrodes=32,
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
            k_neighbors=10,
        )

        from virda.main import run

        stage1_result, ese_mesh = run(config)

        assert ese_mesh is not None
        assert ese_mesh.vertices.shape[0] == stage1_result.mesh.vertices.shape[0]
        assert (tmp_path / "ese" / "ese_mesh.ply").exists()

    def test_run_without_ese_returns_none(
        self,
        synthetic_nifti_path: Path,
        fiducials_file: Path,
        tmp_path: Path,
    ) -> None:
        config = Config(
            nifti_path=str(synthetic_nifti_path),
            project_dir=str(tmp_path),
            fiducials_path=str(fiducials_file),
            closing_radius=0,
            seal_enabled=False,
        )

        from virda.main import run

        stage1_result, ese_mesh = run(config)

        assert isinstance(stage1_result, Stage1Result)
        assert ese_mesh is None
        assert not (tmp_path / "ese").exists()
