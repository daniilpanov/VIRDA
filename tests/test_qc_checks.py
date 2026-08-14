from pathlib import Path
from types import SimpleNamespace
from typing import cast

import nibabel as nib
import numpy as np

from virda.models.ese_config import ESEConfig
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.models.stage1_result import Stage1Result
from virda.qc.checks import (
    check_components,
    check_coordinates_mm,
    check_ese_config,
    check_fiducials,
    check_holes,
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
        face_adjacency=np.zeros((0, 2), dtype=np.int64),
    )


def _result(
    mesh: ScalpMesh, mask: np.ndarray, fiducials: list[Fiducial] | None = None
) -> Stage1Result:
    return Stage1Result(
        mri_volume=_volume(),
        segmentation_mask=SegmentationMask(mask=mask),
        mesh=mesh,
        fiducials=Fiducials([] if fiducials is None else fiducials),
    )


def _ball_mask(shape: tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    grid = np.indices(shape)
    center = np.array(shape, dtype=float) / 2
    radius = 5.0
    return np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0) <= radius**2


def _save_mask(path: Path, mask: np.ndarray) -> Path:
    image = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
    nib.save(image, str(path))
    return path


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
        assert check_mri(cast(MRIVolume, mri))["status"] == "fail"

    def test_bad_spacing_fails(self) -> None:
        mri = SimpleNamespace(
            data=np.zeros((16, 16, 16), dtype=np.float32),
            affine=np.eye(4),
            spacing=(1.0, -1.0, 1.0),
            orientation=("R", "A", "S"),
        )
        assert check_mri(cast(MRIVolume, mri))["status"] == "fail"

    def test_bad_orientation_fails(self) -> None:
        mri = SimpleNamespace(
            data=np.zeros((16, 16, 16), dtype=np.float32),
            affine=np.eye(4),
            spacing=(1.0, 1.0, 1.0),
            orientation=("R", "X", "S"),
        )
        assert check_mri(cast(MRIVolume, mri))["status"] == "fail"


class TestCheckCoordinatesMm:
    def test_inside_volume_passes(self) -> None:
        check = check_coordinates_mm(_triangle_mesh(), _volume())
        assert check["status"] == "ok"

    def test_empty_mesh_passes(self) -> None:
        mesh = ScalpMesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            face_adjacency=np.zeros((0, 2), dtype=np.int64),
        )
        assert check_coordinates_mm(mesh, _volume())["status"] == "ok"

    def test_voxel_coordinates_fail(self) -> None:
        affine = np.eye(4)
        affine[0, 3] = 100.0
        check = check_coordinates_mm(_triangle_mesh(), _volume(affine=affine))
        assert check["status"] == "fail"

    def test_scaled_coordinates_fail(self) -> None:
        mesh = ScalpMesh(
            vertices=np.array([[0.0, 0.0, 0.0], [4000.0, 0.0, 0.0], [0.0, 4000.0, 0.0]]),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            face_adjacency=np.zeros((0, 2), dtype=np.int64),
        )
        assert check_coordinates_mm(mesh, _volume())["status"] == "fail"


class TestCheckESEConfig:
    def test_missing_config_skips(self) -> None:
        assert check_ese_config(None)["status"] == "skip"

    def test_valid_config_passes(self) -> None:
        config = ESEConfig(
            n_electrodes=64,
            ese_offset_mm=15.0,
            ese_reference="electrode_capsule_center",
        )
        check = check_ese_config(config)
        assert check["status"] == "ok"
        assert check["ese_offset_mm"] == 15.0

    def test_non_positive_offset_fails(self) -> None:
        config = SimpleNamespace(
            n_electrodes=64,
            ese_offset_mm=-1.0,
            ese_reference="electrode_capsule_center",
        )
        assert check_ese_config(cast(ESEConfig, config))["status"] == "fail"


class TestCheckMesh:
    def test_valid_mesh_passes(self) -> None:
        check = check_mesh(_triangle_mesh(), min_vertices=3)
        assert check["status"] == "ok"
        assert check["n_vertices"] == 3

    def test_empty_mesh_fails(self) -> None:
        mesh = ScalpMesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            face_adjacency=np.zeros((0, 2), dtype=np.int64),
        )
        assert check_mesh(mesh)["status"] == "fail"

    def test_invalid_face_indices_fail(self) -> None:
        mesh = ScalpMesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 7]]),
            face_adjacency=np.zeros((0, 2), dtype=np.int64),
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

    def test_empty_mask_fails(self) -> None:
        mask = np.zeros((16, 16, 16), dtype=bool)
        check = check_components(mask)
        assert check["status"] == "fail"
        assert "empty" in check["message"]


class TestCheckFiducials:
    def test_no_fiducials_warns(self) -> None:
        result = _result(_triangle_mesh(), _ball_mask())
        check = check_fiducials(Fiducials([]), result)
        assert check["status"] == "warn"

    def test_on_surface_passes(self) -> None:
        fiducial = Fiducial(
            fiducial_id="NAS",
            name="nasion",
            coordinates=np.array([0.5, 0.5, 0.0]),
            coordinate_system="world",
        )
        result = _result(_triangle_mesh(), _ball_mask(), [fiducial])
        assert check_fiducials(Fiducials([fiducial]), result)["status"] == "ok"

    def test_far_fiducial_warns(self) -> None:
        fiducial = Fiducial(
            fiducial_id="NAS",
            name="nasion",
            coordinates=np.array([0.0, 0.0, 50.0]),
            coordinate_system="world",
        )
        result = _result(_triangle_mesh(), _ball_mask(), [fiducial])
        assert check_fiducials(Fiducials([fiducial]), result)["status"] == "warn"


class TestCheckHoles:
    def _mesh(self, faces: np.ndarray, vertices: np.ndarray) -> ScalpMesh:
        return ScalpMesh(
            vertices=vertices,
            faces=faces,
            face_adjacency=np.zeros((0, 2), dtype=np.int64),
        )

    def test_watertight_mesh_passes(self) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
        check = check_holes(self._mesh(faces, vertices))
        assert check["status"] == "ok"
        assert check["n_boundary_loops"] == 0

    def test_single_loop_is_neck_and_passes(self) -> None:
        vertices = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        check = check_holes(self._mesh(faces, vertices))
        assert check["status"] == "ok"
        assert check["n_boundary_loops"] == 1

    def test_small_non_neck_loop_passes(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
                [0.0, 40.0, 0.0],
                [0.0, 0.0, 100.0],
                [1.0, 0.0, 100.0],
                [0.0, 1.0, 100.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        check = check_holes(self._mesh(faces, vertices))
        assert check["status"] == "ok"
        assert check["n_boundary_loops"] == 2

    def test_large_non_neck_loop_warns(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
                [0.0, 40.0, 0.0],
                [100.0, 100.0, 0.0],
                [120.0, 100.0, 0.0],
                [100.0, 120.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        check = check_holes(self._mesh(faces, vertices))
        assert check["status"] == "warn"
        assert check["n_boundary_loops"] == 2
        assert len(check["diameters"]) == 2


class TestCheckNiftiMask:
    def test_matching_mask_passes(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        path = _save_mask(tmp_path / "mask.nii.gz", mask)
        check = check_nifti_mask(path, mask, np.eye(4))
        assert check["status"] == "ok"

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        assert check_nifti_mask(tmp_path / "nope.nii.gz", mask, np.eye(4))["status"] == "fail"

    def test_shape_mismatch_fails(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        path = _save_mask(tmp_path / "mask.nii.gz", mask)
        check = check_nifti_mask(path, np.zeros((8, 8, 8), dtype=bool), np.eye(4))
        assert check["status"] == "fail"

    def test_non_nifti_file_fails(self, tmp_path: Path) -> None:
        image = nib.AnalyzeImage(_ball_mask().astype(np.uint8), np.eye(4))
        path = tmp_path / "mask.img"
        nib.save(image, str(path))
        assert check_nifti_mask(path, _ball_mask(), np.eye(4))["status"] == "fail"


class TestRunChecks:
    def test_aggregates_report(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        result = _result(_triangle_mesh(), mask)
        mask_path = _save_mask(tmp_path / "m.nii.gz", mask)
        ese_config = ESEConfig(
            n_electrodes=64,
            ese_offset_mm=15.0,
            ese_reference="electrode_capsule_center",
        )
        report = run_checks(result, nifti_mask_path=mask_path, ese_config=ese_config)
        assert report["status"] in {"ok", "warn", "fail"}
        assert any(check["name"] == "mri_metadata" for check in report["checks"])
        assert any(check["name"] == "coordinates_mm" for check in report["checks"])
        assert any(check["name"] == "holes_over_scalp" for check in report["checks"])
        assert any(check["name"] == "ese_offset" for check in report["checks"])
        assert "fiducials" in report
        assert "warnings" in report

    def test_fail_propagates(self) -> None:
        mesh = ScalpMesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            face_adjacency=np.zeros((0, 2), dtype=np.int64),
        )
        result = _result(mesh, np.zeros((16, 16, 16), dtype=bool))
        assert run_checks(result)["status"] == "fail"

    def test_missing_ese_config_does_not_fail_overall(self, tmp_path: Path) -> None:
        mask = _ball_mask()
        result = _result(_triangle_mesh(), mask)
        mask_path = _save_mask(tmp_path / "m.nii.gz", mask)
        report = run_checks(result, nifti_mask_path=mask_path)
        ese = next(check for check in report["checks"] if check["name"] == "ese_offset")
        assert ese["status"] == "skip"
        assert report["status"] != "fail"
