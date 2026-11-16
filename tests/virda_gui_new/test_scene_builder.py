"""Unit tests for :mod:`virda_gui_new.services.artifact_loading` and scene building."""

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
from virda_gui_new.dto.viewer3d import LabeledPoints, SceneSources
from virda_gui_new.services.artifact_loading import (
    load_electrode_points,
    load_fiducial_points,
    load_mesh,
    load_volume,
)
from virda_gui_new.services.project_scan import find_project_artifacts
from virda_gui_new.services.scene_builder import collect_scene_sources
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


def _write_points_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """Write the pipeline coords CSV header plus ``(id, x, y, z)`` rows."""
    lines = ["electrode_id,x,y,z,residual_mm,flagged"]
    lines += [f"{electrode_id},{x},{y},{z},,no" for electrode_id, x, y, z in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_manager(project: Path) -> StepsManager:
    manager = StepsManager(tuple(step.required_keys for step in STEPS))
    seed_steps_from_artifacts(manager, find_project_artifacts(project))
    return manager


def _write_full_chain(project: Path) -> None:
    """Write the artifacts of a complete chain: volume, scalp mesh, ESE mesh."""
    _write_nifti(project / "input" / "a.nii")
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_ply(project / "ese" / "ese_mesh.ply")


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


def test_load_electrode_points_skips_rows_without_coordinates(project: Path) -> None:
    rows_path = project / "localization" / "electrodes_scalp.csv"
    _write_points_csv(
        rows_path,
        [("A1", "1.0", "2.0", "3.0"), ("A2", "", "", "")],
    )

    points = load_electrode_points(rows_path)

    assert points.points.tolist() == [[1.0, 2.0, 3.0]]
    assert points.labels == ("A1",)


def _present(sources: SceneSources) -> set[str]:
    """The names of the sources the scene holds."""
    return {
        name
        for name in ("nifti", "fiducials", "scalp", "scalp_points", "ese", "ese_points")
        if getattr(sources, name) is not None
    }


def test_collect_scene_sources_volume_only(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")

    sources = collect_scene_sources(_make_manager(project))

    assert _present(sources) == {"nifti"}
    assert isinstance(sources.nifti, MRIVolume)
    assert sources.nifti.data.size == 64


def test_collect_scene_sources_with_scalp_and_points(project: Path) -> None:
    _write_full_chain(project)
    _write_points_csv(
        project / "localization" / "electrodes_scalp.csv",
        [("A1", "1.0", "2.0", "3.0")],
    )

    sources = collect_scene_sources(_make_manager(project))

    assert _present(sources) == {"nifti", "scalp", "ese", "scalp_points"}
    assert isinstance(sources.scalp, Mesh)
    assert len(sources.scalp.vertices) == 4
    assert isinstance(sources.scalp_points, LabeledPoints)
    assert sources.scalp_points.labels == ("A1",)


def test_collect_scene_sources_with_ese_mesh(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_ply(project / "ese" / "ese_mesh.ply")

    sources = collect_scene_sources(_make_manager(project))

    assert _present(sources) == {"nifti", "scalp", "ese"}
    assert sources.scalp is not sources.ese


def test_collect_scene_sources_with_localization_artifacts(project: Path) -> None:
    _write_full_chain(project)
    _write_points_csv(
        project / "localization" / "electrodes_scalp.csv",
        [("A1", "1.0", "2.0", "3.0")],
    )
    _write_points_csv(
        project / "localization" / "electrodes_ese.csv",
        [("E1", "9.0", "8.0", "7.0")],
    )

    sources = collect_scene_sources(_make_manager(project))

    assert _present(sources) == {
        "nifti",
        "scalp",
        "ese",
        "scalp_points",
        "ese_points",
    }
    assert sources.ese_points is not None
    assert sources.scalp_points is not None
    assert sources.ese_points.labels == ("E1",)
    assert sources.scalp_points.labels == ("A1",)


def test_collect_scene_sources_preserves_same_point_file_for_both_steps(project: Path) -> None:
    """The step-1 and step-4 scalp points are one file; it is loaded once."""
    _write_full_chain(project)
    _write_points_csv(
        project / "localization" / "electrodes_scalp.csv",
        [("A1", "1.0", "2.0", "3.0")],
    )

    sources = collect_scene_sources(_make_manager(project))

    assert _present(sources) == {"nifti", "scalp", "ese", "scalp_points"}
    assert sources.scalp_points is not None
    assert sources.scalp_points.points.tolist() == [[1.0, 2.0, 3.0]]


def test_collect_scene_sources_passes_ready_objects_through(project: Path) -> None:
    volume = MRIVolume(
        data=np.zeros((2, 2, 2), "float32"),
        affine=np.eye(4),
        spacing=(1.0, 1.0, 1.0),
        orientation=("L", "P", "S"),
    )
    mesh = Mesh(np.eye(4, 3), np.array([[0, 1, 2]]))
    points = LabeledPoints(np.array([[0.0, 0.0, 1.0]]), ("E1",))
    manager = StepsManager(tuple(step.required_keys for step in STEPS))
    manager.set_step_context_item(0, "volume", volume)
    manager.next_step()
    manager.set_step_context_item(1, "scalp_surface", mesh)
    manager.next_step()
    manager.set_step_context_item(2, "mesh", mesh)
    manager.next_step()
    manager.set_step_context_item(3, "ese_mesh", mesh)
    manager.set_step_context_item(3, "normals", np.ones((3, 3)))
    manager.next_step()

    manager.set_step_context_item(4, "scalp_points", points)
    sources = collect_scene_sources(manager)

    assert sources.nifti is volume
    assert sources.scalp_points is points


def test_collect_scene_sources_ignores_the_points_of_another_step(project: Path) -> None:
    """The electrode points belong to the localization step alone."""
    volume = MRIVolume(
        data=np.zeros((2, 2, 2), "float32"),
        affine=np.eye(4),
        spacing=(1.0, 1.0, 1.0),
        orientation=("L", "P", "S"),
    )
    points = LabeledPoints(np.array([[0.0, 0.0, 1.0]]), ("E1",))
    manager = StepsManager(tuple(step.required_keys for step in STEPS))
    manager.set_step_context_item(0, "volume", volume)
    manager.next_step()

    manager.set_step_context_item(1, "scalp_points", points)
    sources = collect_scene_sources(manager)

    assert sources.scalp_points is None


def test_collect_scene_sources_empty_for_empty_project(project: Path) -> None:
    assert collect_scene_sources(_make_manager(project)).is_empty()


def test_collect_scene_sources_reports_unparsable_file(project: Path) -> None:
    _write_full_chain(project)
    (project / "localization" / "electrodes_scalp.csv").write_text(
        "not a table\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="electrodes_scalp.csv"):
        collect_scene_sources(_make_manager(project))


def test_collect_scene_sources_collects_failures_instead_of_raising(project: Path) -> None:
    _write_full_chain(project)
    (project / "localization" / "electrodes_scalp.csv").write_text(
        "not a table\n", encoding="utf-8"
    )

    failures: list[str] = []
    sources = collect_scene_sources(_make_manager(project), failures)

    assert len(failures) == 1
    assert "electrodes_scalp.csv" in failures[0]
    assert isinstance(sources.nifti, MRIVolume)
    assert sources.scalp_points is None
