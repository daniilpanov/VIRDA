import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tests.helpers.measurements import make_measurements_file
from tests.helpers.meshes import make_sphere
from tests.helpers.pipelines import make_fiducials, save_test_fiducials
from virda.config import VirdaSettings
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.main import run
from virda.models.ese_mesh import ESEMesh
from virda.models.stage1_result import Stage1Result
from virda.models.stage2_config import Stage2Config
from virda.pipelines.stage2 import Stage2PipelineBuilder

ESE_OFFSET_MM = 2.0


def build_pipeline(tmp_path, config: Stage2Config):
    builder = PCAESEBuilder(config=config, ese_offset_mm=ESE_OFFSET_MM)
    return Stage2PipelineBuilder(
        ese_builder=builder,
        stage2_config=config,
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

        stage2_dir = tmp_path / "stage2"
        assert (stage2_dir / "ese_mesh.ply").exists()
        assert np.array_equal(np.load(stage2_dir / "ese_vertices.npy"), ese_mesh.vertices)
        assert np.array_equal(np.load(stage2_dir / "ese_faces.npy"), ese_mesh.faces)
        assert np.array_equal(np.load(stage2_dir / "normals.npy"), ese_mesh.normals)
        assert np.array_equal(np.load(stage2_dir / "quality.npy"), ese_mesh.quality)

        pairs = json.loads((stage2_dir / "point_pairs.json").read_text())
        assert pairs["n_points"] == ese_mesh.vertices.shape[0]
        assert len(pairs["scalp_vertices"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["ese_vertices"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["normals"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["quality"]) == ese_mesh.vertices.shape[0]

        written_config = json.loads((stage2_dir / "stage2_config.json").read_text())
        assert written_config["stage2"]["k_neighbors"] == 30


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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = VirdaSettings(
            n_electrodes=32,
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
            k_neighbors=10,
            closing_radius=0,
            seal_enabled=False,
            _cli_parse_args=False,  # type: ignore[call-arg]
        )
        monkeypatch.setattr("virda.main.get_virda_settings", lambda: settings)

        stage1_result, ese_mesh, electrodes = run(
            nifti_path=synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
        )

        assert ese_mesh is not None
        assert electrodes is None
        assert ese_mesh.vertices.shape[0] == stage1_result.mesh.vertices.shape[0]
        assert (tmp_path / "stage2" / "ese_mesh.ply").exists()
        assert (tmp_path / "stage2" / "stage2_config.json").exists()

    def test_run_without_ese_returns_none(
        self,
        synthetic_nifti_path: Path,
        fiducials_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = VirdaSettings(
            closing_radius=0,
            seal_enabled=False,
            _cli_parse_args=False,  # type: ignore[call-arg]
        )
        monkeypatch.setattr("virda.main.get_virda_settings", lambda: settings)

        stage1_result, ese_mesh, electrodes = run(
            nifti_path=synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
        )

        assert isinstance(stage1_result, Stage1Result)
        assert ese_mesh is None
        assert electrodes is None
        assert not (tmp_path / "stage2").exists()
        assert not (tmp_path / "stage3").exists()

    def test_run_with_ese_and_measurements_returns_electrodes(
        self,
        synthetic_nifti_path: Path,
        fiducials_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        measurements_file = make_measurements_file(
            tmp_path / "measurements.json",
            points=np.array([[10.0, 10.0, 18.0]]),
            fiducials=make_fiducials(),
        )
        settings = VirdaSettings(
            n_electrodes=32,
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
            k_neighbors=10,
            closing_radius=0,
            seal_enabled=False,
            _cli_parse_args=False,  # type: ignore[call-arg]
        )
        monkeypatch.setattr("virda.main.get_virda_settings", lambda: settings)

        stage1_result, ese_mesh, electrodes = run(
            nifti_path=synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
            measurements_path=measurements_file,
        )

        assert ese_mesh is not None
        assert electrodes is not None
        assert len(electrodes.items) == 1
        assert electrodes.items[0].is_localized
        assert stage1_result.fiducials.get("NAS") is not None
        assert (tmp_path / "stage3" / "electrodes.json").exists()
        assert (tmp_path / "stage3" / "electrode_coords.csv").exists()
        assert (tmp_path / "stage3" / "localization_summary.json").exists()
