import json
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import trimesh
from pydantic_settings import BaseSettings

from virda.fiducials.provider import ManualFiducialProvider
from virda.io.exporter.json_io import save_fiducials
from virda.io.exporter.stage1_exporter import Stage1Exporter
from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.models.ese_config import ESEConfig
from virda.models.fiducial import Fiducial
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1Pipeline
from virda.segmentation.head_segmenter import OtsuHeadSegmenter


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

    voxel_to_world_affine = np.eye(4)

    nifti_image = nib.Nifti1Image(image_data, voxel_to_world_affine)
    nifti_file_path = tmp_path / "synthetic.nii.gz"
    nib.save(nifti_image, nifti_file_path)
    return nifti_file_path


class TestStage1Pipeline:
    def test_run_returns_stage1_result(self, synthetic_nifti_path: Path) -> None:
        loader = NiftiLoader()
        segmenter = OtsuHeadSegmenter()
        cleaner = TrimeshCleaner()

        pipeline = Stage1Pipeline(
            loader=loader, segmenter=segmenter, cleaner=cleaner, smoother=None
        )
        result = pipeline.run(synthetic_nifti_path, closing_radius=0)

        assert isinstance(result, Stage1Result)
        assert result.mri_volume.data.shape == (20, 20, 20)
        assert result.segmentation_mask.shape == (20, 20, 20)
        assert result.segmentation_mask.dtype == bool
        assert result.mesh.vertices.shape[1] == 3
        assert result.mesh.faces.shape[1] == 3
        assert result.mesh.faces.min() >= 0
        assert result.mesh.faces.max() < result.mesh.vertices.shape[0]

    def test_run_with_smoother(self, synthetic_nifti_path: Path) -> None:
        from virda.mesh.laplacian_smoother import LaplacianSmoother

        loader = NiftiLoader()
        segmenter = OtsuHeadSegmenter()
        cleaner = TrimeshCleaner()
        smoother = LaplacianSmoother(iterations=2, lamb=0.5)

        pipeline = Stage1Pipeline(
            loader=loader, segmenter=segmenter, cleaner=cleaner, smoother=smoother
        )
        result = pipeline.run(synthetic_nifti_path, closing_radius=0)

        assert isinstance(result, Stage1Result)
        assert result.mesh.vertices.shape[0] > 0

    def test_run_with_output_dir_exports_artifacts(
        self, synthetic_nifti_path: Path, tmp_path: Path
    ) -> None:
        class DummySettings(BaseSettings):
            closing_radius: int = 5
            smoother_iterations: int = 10

        fiducial = Fiducial(
            fiducial_id="NAS",
            name="nasion",
            coordinates=np.array([10.0, 10.0, 10.0]),
            coordinate_system="world",
        )
        fiducials_path = tmp_path / "fiducials.json"
        save_fiducials(fiducials_path, [fiducial])
        fiducial_provider = ManualFiducialProvider(fiducials_path)
        exporter = Stage1Exporter(
            settings=DummySettings(),
            ese_config=ESEConfig(ese_offset_mm=4.0),
        )
        pipeline = Stage1Pipeline(
            loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(),
            cleaner=TrimeshCleaner(),
            exporter=exporter,
            fiducial_provider=fiducial_provider,
        )

        result = pipeline.run(synthetic_nifti_path, output_dir=tmp_path, closing_radius=0)

        project_dir = tmp_path / "patient_project"
        assert project_dir.is_dir()

        mesh_path = project_dir / "mesh.ply"
        assert mesh_path.is_file()
        loaded_mesh = cast(trimesh.Trimesh, trimesh.load(mesh_path))
        np.testing.assert_allclose(loaded_mesh.vertices, result.mesh.vertices)
        np.testing.assert_allclose(loaded_mesh.faces, result.mesh.faces)

        stage1_payload = json.loads(
            (project_dir / "stage1_result.json").read_text(encoding="utf-8")
        )
        np.testing.assert_allclose(np.asarray(stage1_payload["mri_volume"]["affine"]), np.eye(4))
        assert stage1_payload["mri_volume"]["spacing"] == [1.0, 1.0, 1.0]
        assert stage1_payload["mri_volume"]["shape"] == [20, 20, 20]

        config_payload = json.loads(
            (project_dir / "pipeline_config.json").read_text(encoding="utf-8")
        )
        assert config_payload["ese"]["ese_offset_mm"] == 4.0
        assert config_payload["closing_radius"] == 5

        fiducials_payload = json.loads((project_dir / "fiducials.json").read_text(encoding="utf-8"))
        assert fiducials_payload["fiducials"][0]["fiducial_id"] == "NAS"

    def test_run_without_fiducials_raises(self, synthetic_nifti_path: Path) -> None:
        pipeline = Stage1Pipeline(
            loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(),
            cleaner=TrimeshCleaner(),
            fiducial_provider=ManualFiducialProvider(None),
        )

        with pytest.raises(ValueError, match="fiducials"):
            pipeline.run(synthetic_nifti_path, closing_radius=0)

    def test_run_with_skip_fiducials_returns_empty(self, synthetic_nifti_path: Path) -> None:
        pipeline = Stage1Pipeline(
            loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(),
            cleaner=TrimeshCleaner(),
            fiducial_provider=ManualFiducialProvider(None, skip=True),
        )

        result = pipeline.run(synthetic_nifti_path, closing_radius=0)

        assert result.fiducials == []

    def test_run_with_output_dir_without_exporter_raises(
        self, synthetic_nifti_path: Path, tmp_path: Path
    ) -> None:
        pipeline = Stage1Pipeline(
            loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(),
        )

        with pytest.raises(ValueError, match="exporter"):
            pipeline.run(synthetic_nifti_path, output_dir=tmp_path, closing_radius=0)
