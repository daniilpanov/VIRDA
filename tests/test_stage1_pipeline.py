import json
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import trimesh

from virda.config import VirdaSettings
from virda.fiducials.provider import ManualFiducialProvider
from virda.io.exporter.json_io import save_fiducials
from virda.io.exporter.stage1_exporter import Stage1Exporter
from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.air_depth import AirDepthCleaner
from virda.mesh.cleaners import LargestComponentCleaner, MergeCleaner
from virda.mesh.contracts import MeshCleaner
from virda.mesh.hole_fill import HoleFillCleaner
from virda.models.ese_config import ESEConfig
from virda.models.fiducial import Fiducial
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1Pipeline
from virda.segmentation.head_segmenter import OtsuHeadSegmenter


class TestStage1Pipeline:
    def test_run_returns_stage1_result(self, synthetic_nifti_path: Path) -> None:
        loader = NiftiLoader()
        segmenter = OtsuHeadSegmenter()
        cleaners: list[MeshCleaner] = [
            MergeCleaner(),
            AirDepthCleaner(),
            HoleFillCleaner(),
            LargestComponentCleaner(),
        ]

        pipeline = Stage1Pipeline(
            loader=loader, segmenter=segmenter, cleaners=cleaners, smoother=None
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
        cleaners: list[MeshCleaner] = [MergeCleaner(), LargestComponentCleaner()]
        smoother = LaplacianSmoother(iterations=2, lamb=0.5)

        pipeline = Stage1Pipeline(
            loader=loader, segmenter=segmenter, cleaners=cleaners, smoother=smoother
        )
        result = pipeline.run(synthetic_nifti_path, closing_radius=0)

        assert isinstance(result, Stage1Result)
        assert result.mesh.vertices.shape[0] > 0

    def test_run_with_output_dir_exports_artifacts(
        self, synthetic_nifti_path: Path, tmp_path: Path
    ) -> None:
        class DummySettings(VirdaSettings):
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
            settings=DummySettings(_cli_parse_args=False),  # type: ignore[call-arg]
            ese_config=ESEConfig(ese_offset_mm=4.0),
        )
        pipeline = Stage1Pipeline(
            loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(),
            cleaners=[MergeCleaner(), LargestComponentCleaner()],
            exporter=exporter,
            fiducial_provider=fiducial_provider,
        )

        result = pipeline.run(synthetic_nifti_path, output_dir=tmp_path, closing_radius=0)

        project_dir = tmp_path / "patient_project"
        assert project_dir.is_dir()

        mesh_path = project_dir / "mesh" / "scalp.ply"
        assert mesh_path.is_file()
        loaded_mesh = cast(trimesh.Trimesh, trimesh.load(mesh_path))
        np.testing.assert_allclose(loaded_mesh.vertices, result.mesh.vertices)
        np.testing.assert_allclose(loaded_mesh.faces, result.mesh.faces)

        adjacency_path = project_dir / "mesh" / "scalp_face_adjacency.npy"
        assert adjacency_path.is_file()
        adjacency = np.load(adjacency_path)
        assert adjacency.ndim == 2 and adjacency.shape[1] == 2
        assert adjacency.max() < result.mesh.faces.shape[0]

        stage1_payload = json.loads(
            (project_dir / "stage1_result.json").read_text(encoding="utf-8")
        )
        np.testing.assert_allclose(np.asarray(stage1_payload["mri_volume"]["affine"]), np.eye(4))
        assert stage1_payload["mri_volume"]["spacing"] == [1.0, 1.0, 1.0]
        assert stage1_payload["mri_volume"]["shape"] == [20, 20, 20]
        assert stage1_payload["mesh"]["n_adjacency_edges"] == adjacency.shape[0]

        config_payload = json.loads(
            (project_dir / "config" / "pipeline_config.json").read_text(encoding="utf-8")
        )
        assert config_payload["ese"]["ese_offset_mm"] == 4.0
        assert config_payload["closing_radius"] == 5

        fiducials_payload = json.loads(
            (project_dir / "fiducials" / "fiducials.json").read_text(encoding="utf-8")
        )
        assert fiducials_payload["fiducials"][0]["fiducial_id"] == "NAS"

        mask_path = project_dir / "segmentation" / "head_mask.nii.gz"
        assert mask_path.is_file()
        loaded_mask = nib.load(mask_path)
        assert isinstance(loaded_mask, nib.Nifti1Image)
        np.testing.assert_allclose(loaded_mask.affine, np.eye(4))
        assert loaded_mask.shape == (20, 20, 20)
        stored_voxels = int(np.asanyarray(loaded_mask.dataobj).astype(bool).sum())
        assert stored_voxels == int(result.segmentation_mask.sum())

        provenance = json.loads(
            (project_dir / "input_mri" / "provenance.json").read_text(encoding="utf-8")
        )
        assert provenance["shape"] == [20, 20, 20]
        assert provenance["source"] is not None

        report = json.loads(
            (project_dir / "quality_control" / "report.json").read_text(encoding="utf-8")
        )
        assert "status" in report
        assert any(check["name"] == "mri_metadata" for check in report["checks"])
        assert (project_dir / "logs" / "stage1.log").is_file()
        assert (project_dir / "quality_control" / "qc_overlay_sagittal.png").is_file()
        assert (project_dir / "quality_control" / "qc_overlay_coronal.png").is_file()
        assert (project_dir / "quality_control" / "qc_overlay_axial.png").is_file()
        assert (project_dir / "quality_control" / "qc_3d_front.png").is_file()

    def test_run_without_fiducials_raises(self, synthetic_nifti_path: Path) -> None:
        pipeline = Stage1Pipeline(
            loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(),
            cleaners=[MergeCleaner(), LargestComponentCleaner()],
            fiducial_provider=ManualFiducialProvider(None),
        )

        with pytest.raises(ValueError, match="fiducials"):
            pipeline.run(synthetic_nifti_path, closing_radius=0)

    def test_run_with_skip_fiducials_returns_empty(self, synthetic_nifti_path: Path) -> None:
        pipeline = Stage1Pipeline(
            loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(),
            cleaners=[MergeCleaner(), LargestComponentCleaner()],
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


class TestStage1PipelineCutoff:
    def test_cutoff_skipped_gracefully_when_fiducials_missing(
        self, synthetic_nifti_path: Path, tmp_path: Path
    ) -> None:
        def run_mask(cutoff: bool) -> np.ndarray:
            pipeline = Stage1Pipeline(
                loader=NiftiLoader(),
                segmenter=OtsuHeadSegmenter(),
                fiducial_provider=ManualFiducialProvider(None, skip=True),
                seal=True,
                cutoff=cutoff,
                cutoff_below_nasion_mm=5.0,
            )
            result = pipeline.run(synthetic_nifti_path, closing_radius=0)
            assert result.fiducials == []
            return result.segmentation_mask

        np.testing.assert_array_equal(run_mask(cutoff=True), run_mask(cutoff=False))

    def test_cutoff_reuses_anchor_fiducials(
        self, synthetic_nifti_path: Path, tmp_path: Path
    ) -> None:
        fiducials = [
            Fiducial(
                fiducial_id="NAS",
                name="nasion",
                coordinates=np.array([10.0, 18.0, 10.0]),
                coordinate_system="world",
            ),
            Fiducial(
                fiducial_id="LPA",
                name="left",
                coordinates=np.array([2.0, 12.0, 10.0]),
                coordinate_system="world",
            ),
            Fiducial(
                fiducial_id="RPA",
                name="right",
                coordinates=np.array([18.0, 12.0, 10.0]),
                coordinate_system="world",
            ),
        ]
        fiducials_path = tmp_path / "fiducials.json"
        save_fiducials(fiducials_path, fiducials)
        provider = ManualFiducialProvider(fiducials_path)

        pipeline = Stage1Pipeline(
            loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(),
            cleaners=[MergeCleaner(), LargestComponentCleaner()],
            fiducial_provider=provider,
            seal=True,
            cutoff=True,
            cutoff_below_nasion_mm=5.0,
        )
        result = pipeline.run(synthetic_nifti_path, closing_radius=0)

        assert result.mesh.vertices.shape[0] > 0
        assert len(result.fiducials) == 3
        assert {f.fiducial_id for f in result.fiducials} == {"NAS", "LPA", "RPA"}

        kept_voxels = np.argwhere(result.segmentation_mask)
        assert kept_voxels[:, 2].min() >= 5  # material below the plane was cut
