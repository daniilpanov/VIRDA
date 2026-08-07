from pathlib import Path
from types import SimpleNamespace

import numpy as np

from virda.io.exporter.nifti_exporter import export_segmentation
from virda.models.fiducial import Fiducial
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage1_result import Stage1Result
from virda.qc.checks import (
    check_components,
    check_fiducials,
    check_mesh,
    check_mri,
    check_nifti_mask,
    run_checks,
)


def _volume(affine: np.ndarray | None = None) -> MRIVolume:
    return MRIVolume(
        data=np.zeros((16, 16, 16), dtype=np.float32),
        affine=np.eye(4) if affine is None else affine,
        spacing=(1.0, 1.0, 1.0),
        orientation=("R", "A", "S"),
    )


def _triangle_mesh() -> ScalpMesh:
    return ScalpMesh(
        vertices=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )


def _result(
    mesh: ScalpMesh, mask: np.ndarray, fiducials: list[Fiducial] | None = None
) -> Stage1Result:
    return Stage1Result(
        mri_volume=_volume(),
        segmentation_mask=mask,
        mesh=mesh,
        fiducials=[] if fiducials is None else fiducials,
    )


def _ball_mask(shape: tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    grid = np.indices(shape)
    center = np.array(shape, dtype=float) / 2
    radius = 5.0
    return np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0) <= radius**2


class TestCheckMri:
    def test_valid_volume_passes(self) -> None:
        check = check_mri(_volume())
        assert check["status"] == "ok"

    def test_bad_affine_fails(self) -> None:
        mri = SimpleNamespace(
            data=np.zeros((16, 16, 16), dtype=np.float32),
            affine=np.zeros((3, 3)),
            spacing=(1.0, 1.0, 1.0),
            orientation=("R", "A", "S"),
        )
        assert check_mri(mri)["status"] == "fail"

    def test_bad_spacing_fails(self) -> None:
        mri = SimpleNamespace(
            data=np.zeros((16, 16, 16), dtype=np.float32),
            affine=np.eye(4),
            spacing=(1.0, -1.0, 1.0),
            orientation=("R", "A", "S"),
        )
        assert check_mri(mri)["status"] == "fail"


class TestCheckMesh:
    def test_valid_mesh_passes(self) -> None:
        check = check_mesh(_triangle_mesh(), min_vertices=3)
        assert check["status"] == "ok"
        assert check["n_vertices"] == 3

    def test_empty_mesh_fails(self) -> None:
        mesh = ScalpMesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
        )
        assert check_mesh(mesh)["status"] == "fail"

    def test_invalid_face_indices_fail(self) -> None:
        mesh = ScalpMesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 7]]),
        )
        assert check_mesh(mesh)["status"] == "fail"

    def test_sparse_mesh_warns(self) -> None:
        check = check_mesh(_triangle_mesh(), min_vertices=100)
        assert check["status"] == "warn"


class TestCheckComponents:
    def test_single_component_passes(self) -> None:
        mask = _ball_mask()
        assert check_components(mask)["status"] == "ok"

    def test_multiple_large_components_warn(self) -> None:
        mask = _ball_mask()
        second_center = np.array([2.0, 2.0, 2.0])
        grid = np.indices((16, 16, 16))
        second_ball = np.sum((grid - second_center.reshape(-1, 1, 1, 1)) ** 2, axis=0) <= 3**2
        mask |= second_ball
        assert check_components(mask)["status"] == "warn"


class TestCheckFiducials:
    def test_no_fiducials_warns(self) -> None:
        result = _result(_triangle_mesh(), _ball_mask())
        check = check_fiducials([], result)
        assert check["status"] == "warn"

    def test_on_surface_passes(self) -> None:
        fiducial = Fiducial(
            fiducial_id="NAS",
            name="nasion",
            coordinates=np.array([0.5, 0.5, 0.0]),
            coordinate_system="world",
        )
        result = _result(_triangle_mesh(), _ball_mask(), [fiducial])
        assert check_fiducials([fiducial], result)["status"] == "ok"

    def test_far_fiducial_warns(self) -> None:
        fiducial = Fiducial(
            fiducial_id="NAS",
            name="nasion",
            coordinates=np.array([0.0, 0.0, 50.0]),
            coordinate_system="world",
        )
        result = _result(_triangle_mesh(), _ball_mask(), [fiducial])
        assert check_fiducials([fiducial], result)["status"] == "warn"


class TestCheckNiftiMask:
    def test_matching_mask_passes(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        path = export_segmentation(tmp_path / "mask.nii.gz", mask, np.eye(4))
        check = check_nifti_mask(path, mask, np.eye(4))
        assert check["status"] == "ok"

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        assert check_nifti_mask(tmp_path / "nope.nii.gz", mask, np.eye(4))["status"] == "fail"

    def test_shape_mismatch_fails(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        path = export_segmentation(tmp_path / "mask.nii.gz", mask, np.eye(4))
        check = check_nifti_mask(path, np.zeros((8, 8, 8), dtype=bool), np.eye(4))
        assert check["status"] == "fail"


class TestRunChecks:
    def test_aggregates_report(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        result = _result(_triangle_mesh(), mask)
        mask_path = export_segmentation(tmp_path / "m.nii.gz", mask, np.eye(4))
        report = run_checks(result, nifti_mask_path=mask_path)
        assert report["status"] in {"ok", "warn", "fail"}
        assert any(check["name"] == "mri_metadata" for check in report["checks"])
        assert "fiducials" in report

    def test_fail_propagates(self) -> None:
        mesh = ScalpMesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
        )
        result = _result(mesh, np.zeros((16, 16, 16), dtype=bool))
        assert run_checks(result)["status"] == "fail"
