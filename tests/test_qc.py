from pathlib import Path

import numpy as np
import pytest

from virda.fiducials.provider import ManualFiducialProvider
from virda.io.exporter.ply_exporter import export_ply
from virda.io.loader.nifti_loader import NiftiLoader
from virda.io.qc import overlay_slices, render_3d, run_qc, write_viewer_html
from virda.io.qc.geometry import fiducials_world_coordinates, mesh_voxel_coordinates
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.models.fiducial import Fiducial
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1Pipeline
from virda.segmentation.head_segmenter import OtsuHeadSegmenter

matplotlib = pytest.importorskip("matplotlib")
pyvista = pytest.importorskip("pyvista")


@pytest.fixture
def qc_result(synthetic_nifti_path: Path) -> Stage1Result:
    pipeline = Stage1Pipeline(
        loader=NiftiLoader(),
        segmenter=OtsuHeadSegmenter(),
        cleaner=TrimeshCleaner(),
        fiducial_provider=ManualFiducialProvider(None, skip=True),
    )
    return pipeline.run(synthetic_nifti_path, closing_radius=0)


@pytest.fixture
def mesh_dir(qc_result: Stage1Result, tmp_path: Path) -> Path:
    out = tmp_path / "qc"
    out.mkdir(parents=True)
    export_ply(out / "mesh.ply", qc_result.mesh)
    return out


class TestOverlaySlices:
    def test_writes_three_plane_images(self, qc_result: Stage1Result, tmp_path: Path) -> None:
        out = overlay_slices(qc_result, tmp_path)
        for name in ("sagittal", "coronal", "axial"):
            assert (out / f"qc_overlay_{name}.png").is_file()


class TestRender3d:
    def test_writes_view_screenshots(self, qc_result: Stage1Result, mesh_dir: Path) -> None:
        out = render_3d(qc_result, mesh_dir)
        assert (out / "qc_3d_front.png").is_file()
        assert (out / "qc_3d_three_quarter.png").is_file()


class TestViewerHtml:
    def test_writes_self_contained_html(self, qc_result: Stage1Result, mesh_dir: Path) -> None:
        out = write_viewer_html(qc_result, mesh_dir)
        html = (out / "head_viewer.html").read_text(encoding="utf-8")
        assert "three.min.js" in html
        assert "THREE.BufferGeometry" in html


class TestRunQc:
    def test_run_qc_writes_all_artifacts(self, qc_result: Stage1Result, tmp_path: Path) -> None:
        export_ply(tmp_path / "mesh.ply", qc_result.mesh)
        out = run_qc(qc_result, tmp_path, with_html=True)
        assert (out / "qc_overlay_sagittal.png").is_file()
        assert (out / "qc_3d_front.png").is_file()
        assert (out / "head_viewer.html").is_file()


class TestGeometry:
    def test_fiducials_world_coordinates_keeps_world(self) -> None:
        fiducial = Fiducial(
            fiducial_id="NAS",
            name="nasion",
            coordinates=np.array([1.0, 2.0, 3.0]),
            coordinate_system="world",
        )
        coords = fiducials_world_coordinates([fiducial], np.eye(4))
        np.testing.assert_allclose(coords, [[1.0, 2.0, 3.0]])

    def test_fiducials_world_coordinates_converts_voxel(self) -> None:
        affine = np.eye(4)
        affine[:3, :3] = np.diag([2.0, 2.0, 2.0])
        affine[:3, 3] = [10.0, 20.0, 30.0]
        fiducial = Fiducial(
            fiducial_id="NAS",
            name="nasion",
            coordinates=np.array([1.0, 1.0, 1.0]),
            coordinate_system="voxel",
        )
        coords = fiducials_world_coordinates([fiducial], affine)
        np.testing.assert_allclose(coords, [[12.0, 22.0, 32.0]])

    def test_mesh_voxel_coordinates_round_trip(self, qc_result: Stage1Result) -> None:
        affine = qc_result.mri_volume.affine
        vertices = qc_result.mesh.vertices
        vox = mesh_voxel_coordinates(vertices, affine)
        world_again = vox @ affine[:3, :3].T + affine[:3, 3]
        np.testing.assert_allclose(world_again, vertices, atol=1e-5)
