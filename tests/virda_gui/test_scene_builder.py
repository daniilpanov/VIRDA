"""Unit tests for :mod:`virda_gui_new.services.artifact_loading` and scene building."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

from virda.io.fiducial_helpers import save_fiducials
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.artifact_loading import (
    load_fiducial_points,
    load_mesh,
    load_point_rows,
    load_volume,
)
from virda_gui_new.services.project_scan import find_project_artifacts
from virda_gui_new.services.scene_builder import collect_scene_kwargs
from virda_gui_new.services.step_seeding import seed_steps_from_artifacts
from virda_gui_new.services.steps_manager import StepsManager


def _write_nifti(path: Path) -> None:
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), "float32"), np.eye(4)), path)


def _write_ply(path: Path) -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 1, 3], [1, 2, 3]]).export(path)


def _write_fiducials(path: Path) -> None:
    fiducials = Fiducials(
        items=[
            Fiducial("c1", "nas", np.array([0.0, 1.0, 2.0]), "world"),
            Fiducial("c2", "", np.array([3.0, 4.0, 5.0]), "world"),
        ]
    )
    save_fiducials(path, fiducials)


def _write_point_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _make_manager(project: Path) -> StepsManager:
    manager = StepsManager(tuple(step.required_keys for step in STEPS))
    seed_steps_from_artifacts(manager, find_project_artifacts(project))
    return manager


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "input").mkdir()
    (tmp_path / "mesh").mkdir()
    (tmp_path / "ese").mkdir()
    (tmp_path / "localization").mkdir()
    return tmp_path


def test_load_volume_reads_nifti(project: Path) -> None:
    nifti_path = project / "input" / "a.nii"
    _write_nifti(nifti_path)

    volume = load_volume(nifti_path)

    assert volume.data.size == 64


def test_load_mesh_reads_ply(project: Path) -> None:
    ply_path = project / "mesh" / "final_mesh.ply"
    _write_ply(ply_path)

    mesh = load_mesh(ply_path)

    assert isinstance(mesh, Mesh)
    assert len(mesh.vertices) == 4


def test_load_fiducial_points_uses_names_with_id_fallback(project: Path) -> None:
    fiducials_path = project / "input" / "fiducials.json"
    _write_fiducials(fiducials_path)

    points = load_fiducial_points(fiducials_path)

    assert points.points.shape == (2, 3)
    assert points.labels == ("nas", "c2")


def test_load_point_rows_skips_rows_with_null_coordinates(project: Path) -> None:
    rows_path = project / "localization" / "electrodes_scalp.json"
    _write_point_rows(
        rows_path,
        [
            {"name": "A1", "x": 1.0, "y": 2.0, "z": 3.0},
            {"name": "A2", "x": 1.0, "y": 2.0, "z": None},
            {"name": 4, "x": 4.0, "y": 5.0, "z": 6.0, "note": "extra"},
        ],
    )

    points = load_point_rows(rows_path)

    assert points.points.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert points.labels == ("A1", "4")


def test_collect_scene_kwargs_volume_only(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")

    kwargs = collect_scene_kwargs(_make_manager(project))

    assert list(kwargs) == ["nifti"]
    assert isinstance(kwargs["nifti"], MRIVolume)
    assert kwargs["nifti"].data.size == 64


def test_collect_scene_kwargs_with_scalp_and_points(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_point_rows(
        project / "localization" / "electrodes_scalp.json",
        [{"name": "A1", "x": 1.0, "y": 2.0, "z": 3.0}],
    )

    kwargs = collect_scene_kwargs(_make_manager(project))

    assert set(kwargs) == {"nifti", "scalp", "scalp_points"}
    assert isinstance(kwargs["scalp"], Mesh)
    assert len(kwargs["scalp"].vertices) == 4
    assert isinstance(kwargs["scalp_points"], LabeledPoints)
    assert kwargs["scalp_points"].labels == ("A1",)


def test_collect_scene_kwargs_with_ese_mesh(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_ply(project / "ese" / "ese_mesh.ply")

    kwargs = collect_scene_kwargs(_make_manager(project))

    assert set(kwargs) == {"nifti", "scalp", "ese"}
    assert kwargs["scalp"] is not kwargs["ese"]


def test_collect_scene_kwargs_with_localization_artifacts(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_fiducials(project / "input" / "fiducials.json")
    _write_point_rows(
        project / "localization" / "electrodes_scalp.json",
        [{"name": "A1", "x": 1.0, "y": 2.0, "z": 3.0}],
    )
    _write_point_rows(
        project / "localization" / "electrodes_ese.json",
        [{"name": "E1", "x": 9.0, "y": 8.0, "z": 7.0}],
    )

    kwargs = collect_scene_kwargs(_make_manager(project))

    assert set(kwargs) == {"nifti", "scalp", "fiducials", "scalp_points", "ese_points"}
    fiducials = kwargs["fiducials"]
    assert fiducials.points.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert fiducials.labels == ("nas", "c2")
    assert kwargs["ese_points"].labels == ("E1",)
    assert kwargs["scalp_points"].labels == ("A1",)


def test_collect_scene_kwargs_preserves_same_point_file_for_both_steps(project: Path) -> None:
    """The step-1 and step-4 scalp points are one file; it is loaded once."""
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_point_rows(
        project / "localization" / "electrodes_scalp.json",
        [{"name": "A1", "x": 1.0, "y": 2.0, "z": 3.0}],
    )

    kwargs = collect_scene_kwargs(_make_manager(project))

    assert set(kwargs) == {"scalp", "scalp_points"}
    assert kwargs["scalp_points"].points.tolist() == [[1.0, 2.0, 3.0]]


def test_collect_scene_kwargs_passes_ready_objects_through(project: Path) -> None:
    volume = MRIVolume(
        data=np.zeros((2, 2, 2), "float32"),
        affine=np.eye(4),
        spacing=(1.0, 1.0, 1.0),
        orientation=("L", "P", "S"),
    )
    points = LabeledPoints(np.array([[0.0, 0.0, 1.0]]), ("E1",))
    manager = StepsManager(tuple(step.required_keys for step in STEPS))
    manager.next_step()

    manager.set_step_context_item(0, "volume", volume)
    manager.set_step_context_item(1, "scalp_points", points)
    kwargs = collect_scene_kwargs(manager)

    assert kwargs["nifti"] is volume
    assert kwargs["scalp_points"] is points


def test_collect_scene_kwargs_empty_for_empty_project(project: Path) -> None:
    assert collect_scene_kwargs(_make_manager(project)) == {}


def test_collect_scene_kwargs_reports_unparsable_file(project: Path) -> None:
    (project / "localization" / "electrodes_scalp.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="electrodes_scalp.json"):
        collect_scene_kwargs(_make_manager(project))
