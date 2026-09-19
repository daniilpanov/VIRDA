"""Unit tests for the viewer's coordinate-frame math and frame exports.

:mod:`virda_gui.viewer.frames` is Qt-free: the tests exercise the frame
matrices, the scene -> frame composition and the OBJ/TSV serializers with
plain NumPy data plus a lightweight PyVista scene, without any display.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from virda_gui.viewer.frames import (
    FRAME_CRAS,
    FRAME_SCANNER,
    FRAME_VOXEL,
    collect_electrodes_export,
    collect_fiducials_export,
    collect_mesh_export,
    frame_label,
    frame_to_world_matrix,
    natural_frame,
    scene_to_frame_matrix,
    scene_to_world_matrix,
    triangle_faces,
    world_to_frame_matrix,
    world_to_frame_points,
    write_mesh_obj,
    write_points_tsv,
)
from virda_gui.viewer.viewer_loaders import SceneData, _cras_to_scanner_ras_offset


def _non_degenerate_affine() -> np.ndarray:
    angle = 0.3
    affine = np.eye(4)
    affine[:3, :3] = 2.0 * np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    affine[:3, 3] = [10.0, 20.0, 30.0]
    return affine


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v":
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[0] == "f":
            faces.append([int(value) - 1 for value in parts[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _scene() -> SceneData:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cells = np.array([3, 0, 1, 2, 3, 0, 1, 3], dtype=np.int64)
    return SceneData(
        scene_mesh=pv.PolyData(vertices, cells),
        fiducial_points=np.array([[7.0, 8.0, 9.0]], dtype=np.float64),
        fiducial_labels=["Nz (Nasion)"],
        fiducial_id_to_point={"Nz": np.array([7.0, 8.0, 9.0])},
        electrode_groups=[
            {
                "points": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "flags": np.array([False, True]),
                "measured": [],
                "names": ["E1", "E2"],
                "label": "electrodes.tsv",
                "color": "yellow",
            }
        ],
    )


class TestWorldToFrameMatrix:
    def test_scanner_is_identity(self) -> None:
        np.testing.assert_array_equal(
            world_to_frame_matrix(FRAME_SCANNER, _non_degenerate_affine(), None),
            np.eye(4),
        )

    def test_voxel_is_inverse_affine(self) -> None:
        affine = _non_degenerate_affine()
        matrix = world_to_frame_matrix(FRAME_VOXEL, affine, None)
        np.testing.assert_allclose(matrix, np.linalg.inv(affine), atol=1e-9)

    def test_voxel_requires_affine(self) -> None:
        with pytest.raises(ValueError, match="requires the NIfTI affine"):
            world_to_frame_matrix(FRAME_VOXEL, None, None)

    def test_cras_shifts_by_offset(self) -> None:
        affine = _non_degenerate_affine()
        offset = _cras_to_scanner_ras_offset(affine, (32, 32, 32))
        world = np.array([[12.0, 34.0, 56.0], [1.0, 2.0, 3.0]])
        cras = world_to_frame_points(FRAME_CRAS, world, affine, offset)
        np.testing.assert_allclose(cras, world - offset, atol=1e-9)
        back = transform_4x4(cras, frame_to_world_matrix(FRAME_CRAS, affine, offset))
        np.testing.assert_allclose(back, world, atol=1e-9)

    def test_cras_requires_offset(self) -> None:
        with pytest.raises(ValueError, match="requires the NIfTI cRAS offset"):
            world_to_frame_matrix(FRAME_CRAS, _non_degenerate_affine(), None)

    def test_unknown_frame_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown coordinate frame"):
            world_to_frame_matrix("polar", None, None)


class TestVoxelRoundTrip:
    def test_world_voxel_world(self) -> None:
        affine = _non_degenerate_affine()
        world = np.array(
            [[10.0, 20.0, 30.0], [0.0, 0.0, 0.0], [41.5, 2.3, -7.9], [12.0, 12.0, 12.0]]
        )
        voxel = world_to_frame_points(FRAME_VOXEL, world, affine, None)
        back = transform_4x4(voxel, frame_to_world_matrix(FRAME_VOXEL, affine, None))
        np.testing.assert_allclose(back, world, atol=1e-8)


class TestSceneToFrameMatrix:
    def test_world_scene_equals_world_frames(self) -> None:
        affine = _non_degenerate_affine()
        mm_scene = True
        for frame in (FRAME_SCANNER, FRAME_VOXEL, FRAME_CRAS):
            offset = (
                _cras_to_scanner_ras_offset(affine, (32, 32, 32)) if frame == FRAME_CRAS else None
            )
            left = scene_to_frame_matrix(frame, affine, offset, mm_scene)
            right = world_to_frame_matrix(frame, affine, offset)
            np.testing.assert_allclose(left, right, atol=1e-9)

    def test_voxel_scene_scanner_is_affine(self) -> None:
        affine = _non_degenerate_affine()
        np.testing.assert_allclose(
            scene_to_frame_matrix(FRAME_SCANNER, affine, None, False), affine, atol=1e-9
        )

    def test_voxel_scene_voxel_is_identity(self) -> None:
        affine = _non_degenerate_affine()
        np.testing.assert_allclose(
            scene_to_frame_matrix(FRAME_VOXEL, affine, None, False), np.eye(4), atol=1e-9
        )

    def test_voxel_scene_composes_offset(self) -> None:
        affine = _non_degenerate_affine()
        offset = _cras_to_scanner_ras_offset(affine, (32, 32, 32))
        expected = world_to_frame_matrix(FRAME_CRAS, affine, offset) @ affine
        np.testing.assert_allclose(
            scene_to_frame_matrix(FRAME_CRAS, affine, offset, False), expected, atol=1e-9
        )

    def test_scene_to_frame_round_trip(self) -> None:
        affine = _non_degenerate_affine()
        scene_pts = np.array([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0]])
        world = transform_4x4(scene_pts, scene_to_world_matrix(affine, False))
        voxel = transform_4x4(scene_pts, scene_to_frame_matrix(FRAME_VOXEL, affine, None, False))
        np.testing.assert_allclose(voxel, scene_pts, atol=1e-9)
        back_to_scene = transform_4x4(world, frame_to_world_matrix(FRAME_VOXEL, affine, None))
        np.testing.assert_allclose(back_to_scene, scene_pts, atol=1e-8)

    def test_selector_helpers(self) -> None:
        assert natural_frame(True) == FRAME_SCANNER
        assert natural_frame(False) == FRAME_VOXEL
        assert frame_label(FRAME_CRAS) == "FreeSurfer cRAS"
        assert frame_label("scanner_ras") == "Scanner RAS (world mm)"
        assert frame_label("custom") == "custom"


class TestTriangleFaces:
    def test_raveled_cells_to_faces(self) -> None:
        cells = np.array([3, 0, 1, 2, 3, 2, 3, 4])
        np.testing.assert_array_equal(triangle_faces(cells), [[0, 1, 2], [2, 3, 4]])

    def test_empty_cells(self) -> None:
        assert triangle_faces(np.empty(0)).shape == (0, 3)

    def test_rejects_quad_cells(self) -> None:
        cells = np.array([4, 0, 1, 2, 3])
        with pytest.raises(ValueError, match="triangular"):
            triangle_faces(cells)


class TestExportSerialization:
    def test_mesh_obj_round_trip(self, tmp_path: Path) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        path = write_mesh_obj(tmp_path / "mesh.obj", points, faces, FRAME_SCANNER)
        read_points, read_faces = _read_obj(path)
        np.testing.assert_allclose(read_points, points, atol=1e-12)
        np.testing.assert_array_equal(read_faces, faces)
        assert "# coordinate frame: scanner_ras" in path.read_text(encoding="utf-8")

    def test_points_tsv_round_trip(self, tmp_path: Path) -> None:
        names = ["Fz", "Cz"]
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        path = write_points_tsv(tmp_path / "points.tsv", names, points)
        rows = _read_tsv(path)
        assert [row["name"] for row in rows] == names
        np.testing.assert_allclose(
            [[float(row[c]) for c in ("x", "y", "z")] for row in rows], points, atol=1e-12
        )

    def test_points_tsv_rejects_shape_mismatch(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="names for"):
            write_points_tsv(tmp_path / "bad.tsv", ["only"], np.ones((2, 3)))

    def test_points_tsv_rejects_bad_shape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            write_points_tsv(tmp_path / "bad.tsv", [], np.ones((2, 2)))


class TestCollectExports:
    def _cras_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        affine = _non_degenerate_affine()
        offset = _cras_to_scanner_ras_offset(affine, (32, 32, 32))
        return affine, offset

    def test_collect_mesh_export_transforms_points(self) -> None:
        affine, offset = self._cras_matrix()
        scene = _scene()
        matrix = scene_to_frame_matrix(FRAME_VOXEL, affine, offset, True)
        points, faces = collect_mesh_export(scene, matrix)
        assert scene.scene_mesh is not None
        inv_affine = np.linalg.inv(affine)
        expected = scene.scene_mesh.points @ inv_affine[:3, :3].T + inv_affine[:3, 3]
        np.testing.assert_allclose(points, expected, atol=1e-9)
        np.testing.assert_array_equal(faces.shape, (2, 3))

    def test_collect_electrodes_export(self) -> None:
        scene = _scene()
        names, points = collect_electrodes_export(scene, np.eye(4))
        assert names == ["E1", "E2"]
        np.testing.assert_allclose(points, [[1, 2, 3], [4, 5, 6]])

    def test_collect_electrodes_export_empty(self) -> None:
        names, points = collect_electrodes_export(SceneData(), np.eye(4))
        assert names == []
        assert points.shape == (0, 3)

    def test_collect_fiducials_export(self) -> None:
        affine, offset = self._cras_matrix()
        scene = _scene()
        names, points = collect_fiducials_export(
            scene, scene_to_frame_matrix(FRAME_CRAS, affine, offset, True)
        )
        assert names == ["Nz (Nasion)"]
        np.testing.assert_allclose(points, [[7, 8, 9]] - offset, atol=1e-9)

    def test_collect_fiducials_export_empty(self) -> None:
        names, points = collect_fiducials_export(SceneData(), np.eye(4))
        assert names == []
        assert points.shape == (0, 3)

    def test_mesh_export_requires_mesh(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no scalp mesh"):
            collect_mesh_export(SceneData(), np.eye(4))

    def test_full_export_writes_parseable_files(self, tmp_path: Path) -> None:
        affine, offset = self._cras_matrix()
        scene = _scene()
        matrix = scene_to_frame_matrix(FRAME_VOXEL, affine, offset, True)

        mesh_path = write_mesh_obj(
            tmp_path / "scalp_mesh_voxel.obj", *collect_mesh_export(scene, matrix), FRAME_VOXEL
        )
        read_points, read_faces = _read_obj(mesh_path)
        assert len(read_points) == 4
        assert read_faces.shape == (2, 3)

        names, points = collect_electrodes_export(scene, matrix)
        elec_path = write_points_tsv(tmp_path / "electrodes_voxel.tsv", names, points)
        rows = _read_tsv(elec_path)
        assert len(rows) == 2

        f_names, f_points = collect_fiducials_export(scene, matrix)
        fid_path = write_points_tsv(tmp_path / "fiducials_voxel.tsv", f_names, f_points)
        assert len(_read_tsv(fid_path)) == 1

        world_back = transform_4x4(points, frame_to_world_matrix(FRAME_VOXEL, affine, None))
        np.testing.assert_allclose(world_back, [[1, 2, 3], [4, 5, 6]], atol=1e-8)


def transform_4x4(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]