import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tests.helpers.pipelines import save_test_fiducials
from virda.config import VirdaSettings
from virda.io.fiducial_helpers import load_fiducials
from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.contracts import MeshPostprocessor
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.ese_config import ESEConfig
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1PipelineBuilder
from virda.segmentation.head_segmenter import OtsuHeadSegmenter
from virda.segmentation.seal import MaskSealer


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


@pytest.fixture
def fiducials_file(tmp_path: Path) -> Path:
    return save_test_fiducials(tmp_path / "fiducials.json")


def build_pipeline(
    nifti_path: Path,
    postprocessors: list[MeshPostprocessor] | None = None,
    fiducials_path: Path | None = None,
    project_dir: Path | None = None,
    ese_config: ESEConfig | None = None,
    settings: VirdaSettings | None = None,
):
    builder = Stage1PipelineBuilder(
        nifti_path=nifti_path,
        mri_loader=NiftiLoader(),
        segmenter=OtsuHeadSegmenter(closing_radius=0),
        extractor=MarchingCubesExtractor(),
        project_dir=project_dir,
        fiducials_path=fiducials_path,
        ese_config=ese_config,
        settings=settings,
    )
    if postprocessors:
        builder.setup_mesh_postprocessors(postprocessors)
    return builder.build()


class TestStage1Pipeline:
    def test_run_returns_stage1_result(
        self, synthetic_nifti_path: Path, fiducials_file: Path
    ) -> None:
        pipeline = build_pipeline(synthetic_nifti_path, fiducials_path=fiducials_file)
        result = pipeline.run().get_store_notnull(Stage1Result)

        assert isinstance(result, Stage1Result)
        assert result.mri_volume.data.shape == (20, 20, 20)
        assert result.segmentation_mask.mask.shape == (20, 20, 20)
        assert result.segmentation_mask.mask.dtype == bool
        assert result.mesh.vertices.shape[1] == 3
        assert result.mesh.faces.shape[1] == 3
        assert result.mesh.faces.min() >= 0
        assert result.mesh.faces.max() < result.mesh.vertices.shape[0]

    def test_run_with_smoother(self, synthetic_nifti_path: Path, fiducials_file: Path) -> None:
        from virda.mesh.laplacian_smoother import LaplacianSmoother

        pipeline = build_pipeline(
            synthetic_nifti_path,
            postprocessors=[TrimeshCleaner(), LaplacianSmoother(iterations=2, lamb=0.5)],
            fiducials_path=fiducials_file,
        )
        result = pipeline.run().get_store_notnull(Stage1Result)

        assert isinstance(result, Stage1Result)
        assert result.mesh.vertices.shape[0] > 0

    def test_run_exports_mesh_arrays(
        self, synthetic_nifti_path: Path, tmp_path: Path, fiducials_file: Path
    ) -> None:
        pipeline = build_pipeline(
            synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
        )

        result = pipeline.run().get_store_notnull(Stage1Result)

        vertices = np.load(tmp_path / "mesh" / "scalp_vertices.npy")
        faces = np.load(tmp_path / "mesh" / "scalp_faces.npy")
        face_adjacency = np.load(tmp_path / "mesh" / "scalp_face_adjacency.npy")

        assert np.array_equal(vertices, result.mesh.vertices)
        assert np.array_equal(faces, result.mesh.faces)
        assert np.array_equal(face_adjacency, result.mesh.face_adjacency)
        assert (tmp_path / "mesh" / "final_mesh.ply").exists()

        n_adjacency_edges = (tmp_path / "mesh" / "n_adjacency_edges.json").read_text()
        assert (
            n_adjacency_edges == f'{{"n_adjacency_edges": {result.mesh.face_adjacency.shape[0]}}}'
        )

    def test_run_populates_fiducials(
        self, synthetic_nifti_path: Path, fiducials_file: Path
    ) -> None:
        pipeline = build_pipeline(synthetic_nifti_path, fiducials_path=fiducials_file)
        result = pipeline.run().get_store_notnull(Stage1Result)

        assert result.fiducials.ids == ["NAS", "LPA"]
        nas = result.fiducials.get("NAS")
        assert nas is not None
        assert nas.coordinate_system == "world"
        assert nas.definition_method == "manual"

    def test_run_exports_fiducials_json(
        self, synthetic_nifti_path: Path, tmp_path: Path, fiducials_file: Path
    ) -> None:
        pipeline = build_pipeline(
            synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
        )

        result = pipeline.run().get_store_notnull(Stage1Result)

        exported_path = tmp_path / "fiducials" / "fiducials.json"
        assert exported_path.exists()
        restored = load_fiducials(exported_path)
        assert restored.ids == result.fiducials.ids

    def test_run_exports_ese_config(
        self, synthetic_nifti_path: Path, tmp_path: Path, fiducials_file: Path
    ) -> None:
        ese_config = ESEConfig(
            n_electrodes=32, ese_offset_mm=2.5, ese_reference="electrode_body_center"
        )
        pipeline = build_pipeline(
            synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
            ese_config=ese_config,
        )

        pipeline.run().get_store_notnull(Stage1Result)

        config = json.loads((tmp_path / "config" / "ese.json").read_text())
        assert config == {
            "ese": {
                "n_electrodes": 32,
                "ese_offset_mm": 2.5,
                "ese_reference": "electrode_body_center",
            }
        }

    def test_run_without_ese_config_skips_pipeline_config(
        self, synthetic_nifti_path: Path, tmp_path: Path, fiducials_file: Path
    ) -> None:
        pipeline = build_pipeline(
            synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
        )

        pipeline.run().get_store_notnull(Stage1Result)

        assert not (tmp_path / "config" / "ese.json").exists()

    def test_run_exports_settings(
        self, synthetic_nifti_path: Path, tmp_path: Path, fiducials_file: Path
    ) -> None:
        settings = VirdaSettings(
            n_electrodes=32,
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
            _cli_parse_args=False,  # type: ignore[call-arg]
        )
        pipeline = build_pipeline(
            synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
            settings=settings,
        )

        pipeline.run().get_store_notnull(Stage1Result)

        config = json.loads((tmp_path / "input" / "pipeline_config.json").read_text())
        assert config["n_electrodes"] == 32
        assert config["ese_offset_mm"] == 2.5
        assert config["ese_reference"] == "electrode_body_center"
        assert config["auto_detect_fiducials"] is False

    def test_run_without_settings_skips_pipeline_config(
        self, synthetic_nifti_path: Path, tmp_path: Path, fiducials_file: Path
    ) -> None:
        pipeline = build_pipeline(
            synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
        )

        pipeline.run().get_store_notnull(Stage1Result)

        assert not (tmp_path / "input" / "pipeline_config.json").exists()

    def test_run_copies_source_nifti(
        self, synthetic_nifti_path: Path, tmp_path: Path, fiducials_file: Path
    ) -> None:
        pipeline = build_pipeline(
            synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
        )

        pipeline.run().get_store_notnull(Stage1Result)

        copied = tmp_path / "input" / synthetic_nifti_path.name
        assert copied.read_bytes() == synthetic_nifti_path.read_bytes()

    def test_run_copy_nifti_failure_logs_warning(
        self,
        synthetic_nifti_path: Path,
        tmp_path: Path,
        fiducials_file: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil

        def raise_os_error(*args, **kwargs) -> None:
            raise OSError("copy failed")

        monkeypatch.setattr(shutil, "copy2", raise_os_error)
        pipeline = build_pipeline(
            synthetic_nifti_path,
            project_dir=tmp_path,
            fiducials_path=fiducials_file,
        )

        with caplog.at_level("WARNING"):
            pipeline.run().get_store_notnull(Stage1Result)

        assert not (tmp_path / "input" / synthetic_nifti_path.name).exists()
        assert any("Failed to copy source NIfTI" in record.message for record in caplog.records)

    def test_run_auto_detects_fiducials_when_enabled(
        self, synthetic_nifti_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from virda.fiducials import detector

        points = {
            "NAS": np.array([0.0, 88.0, -10.0]),
            "LPA": np.array([-75.0, -1.0, -14.0]),
            "RPA": np.array([75.0, -1.0, -14.0]),
            "INI": np.array([0.0, -70.0, 10.0]),
        }
        monkeypatch.setattr(detector, "find_fiducials", lambda vertices: points)

        builder = Stage1PipelineBuilder(
            nifti_path=synthetic_nifti_path,
            mri_loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(closing_radius=0),
            extractor=MarchingCubesExtractor(),
            auto_detect_fiducials=True,
        )

        result = builder.build().run().get_store_notnull(Stage1Result)

        assert set(result.fiducials.ids) == {"NAS", "LPA", "RPA", "INI"}
        assert all(fiducial.definition_method == "auto" for fiducial in result.fiducials.items)

    def test_run_raises_without_fiducial_source(self, synthetic_nifti_path: Path) -> None:
        pipeline = build_pipeline(synthetic_nifti_path)

        with pytest.raises(ValueError, match="No fiducials available"):
            pipeline.run()

    def test_from_settings_matches_manual_assembly(
        self, synthetic_nifti_path: Path, fiducials_file: Path, tmp_path: Path
    ) -> None:
        settings = VirdaSettings(  # type: ignore[call-arg]
            closing_radius=0,
            seal_enabled=False,
            smoother_type="taubin",
            _cli_parse_args=False,
        )

        from_settings_result = (
            Stage1PipelineBuilder.from_settings(
                settings=settings,
                nifti_path=synthetic_nifti_path,
                project_dir=tmp_path / "from_settings",
                fiducials_path=fiducials_file,
            )
            .build()
            .run()
            .get_store_notnull(Stage1Result)
        )

        expected_result = (
            Stage1PipelineBuilder(
                nifti_path=synthetic_nifti_path,
                mri_loader=NiftiLoader(),
                segmenter=OtsuHeadSegmenter(
                    closing_radius=settings.closing_radius,
                    otsu_scope=settings.otsu_scope,
                    threshold_scale=settings.otsu_threshold_scale,
                ),
                extractor=MarchingCubesExtractor(),
                project_dir=tmp_path / "expected",
                fiducials_path=fiducials_file,
                auto_detect_fiducials=settings.auto_detect_fiducials,
                ese_config=None,
                settings=settings,
            )
            .setup_mesh_postprocessors(
                [
                    TrimeshCleaner(
                        min_component_vertices=settings.cleaner_min_vertices,
                        merge_digits=settings.cleaner_merge_digits,
                    ),
                    TaubinSmoother(
                        iterations=settings.smoother_iterations,
                        lamb=settings.smoother_lamb,
                        nu=settings.smoother_nu,
                    ),
                ]
            )
            .build()
            .run()
            .get_store_notnull(Stage1Result)
        )

        assert np.array_equal(from_settings_result.mesh.vertices, expected_result.mesh.vertices)
        assert np.array_equal(from_settings_result.mesh.faces, expected_result.mesh.faces)
        assert np.array_equal(
            from_settings_result.segmentation_mask.mask,
            expected_result.segmentation_mask.mask,
        )
        assert from_settings_result.fiducials.ids == expected_result.fiducials.ids

    def test_run_with_mask_sealer(self, synthetic_nifti_path: Path, fiducials_file: Path) -> None:
        builder = Stage1PipelineBuilder(
            nifti_path=synthetic_nifti_path,
            mri_loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(closing_radius=0),
            extractor=MarchingCubesExtractor(),
            fiducials_path=fiducials_file,
        )
        builder.setup_mask_postprocessors([MaskSealer(radius=2)])
        pipeline = builder.build()

        result = pipeline.run().get_store_notnull(Stage1Result)

        assert isinstance(result, Stage1Result)
        assert result.segmentation_mask.mask.dtype == bool
        assert result.mesh.vertices.shape[0] > 0
