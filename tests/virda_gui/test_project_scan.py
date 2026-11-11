"""Unit tests for project scanning and step-context seeding."""

from pathlib import Path

import pytest

from virda_gui_new.dto.steps import STEPS
from virda_gui_new.services.project_scan import ProjectArtifacts, find_project_artifacts
from virda_gui_new.services.step_seeding import seed_steps_from_artifacts
from virda_gui_new.services.steps_manager import StepsManager


def make_project(
    tmp_path: Path,
    *,
    nifti: str | None = "volume.nii.gz",
    fiducials: bool = True,
    scalp_mesh: bool = True,
    scalp_points: bool = True,
    ese_mesh: bool = True,
    ese_points: bool = True,
    base_mesh: bool = False,
) -> Path:
    """Build a fake VIRDA project tree with the selected artifacts."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    files: list[Path] = []
    if nifti is not None:
        files.append(input_dir / nifti)
    if fiducials:
        files.append(input_dir / "fiducials.json")
    if scalp_mesh:
        files.append(tmp_path / "mesh" / "final_mesh.ply")
    if scalp_points:
        files.append(tmp_path / "localization" / "electrodes_scalp.json")
    if ese_mesh:
        files.append(tmp_path / "ese" / "ese_mesh.ply")
    if ese_points:
        files.append(tmp_path / "localization" / "electrodes_ese.json")
    if base_mesh:
        files.append(tmp_path / "scalp" / "base_mesh.ply")
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return tmp_path


def new_manager() -> StepsManager:
    return StepsManager([step.required_keys for step in STEPS])


# --- find_project_artifacts ---


def test_scan_reports_all_artifacts(tmp_path: Path) -> None:
    project = make_project(tmp_path)

    artifacts = find_project_artifacts(project)

    assert artifacts.nifti == project / "input" / "volume.nii.gz"
    assert artifacts.fiducials == project / "input" / "fiducials.json"
    assert artifacts.scalp_mesh == project / "mesh" / "final_mesh.ply"
    assert artifacts.scalp_points == project / "localization" / "electrodes_scalp.json"
    assert artifacts.ese_mesh == project / "ese" / "ese_mesh.ply"
    assert artifacts.ese_points == project / "localization" / "electrodes_ese.json"
    assert artifacts.measurements is None
    assert artifacts.coordsystem is None
    assert artifacts.pipeline_config is None
    assert artifacts.base_mesh is None


def test_scan_of_empty_directory_finds_nothing(tmp_path: Path) -> None:
    artifacts = find_project_artifacts(tmp_path)

    assert artifacts == ProjectArtifacts()


def test_scan_finds_whole_project_artifacts_in_any_folder(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    names = {
        "fiducials": "fiducials.json",
        "measurements": "measurements.json",
        "coordsystem": "coordsystem.json",
        "pipeline_config": "pipeline_config.json",
        "base_mesh": "base_mesh.ply",
        "scalp_mesh": "final_mesh.ply",
    }
    for _slot, name in names.items():
        (nested / name).write_text("")

    artifacts = find_project_artifacts(tmp_path)

    for slot, name in names.items():
        assert getattr(artifacts, slot) == nested / name


def test_scan_nifti_only_in_input(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "head.nii.gz").write_text("")

    assert find_project_artifacts(tmp_path).nifti is None


def test_scan_ese_mesh_from_new_name(tmp_path: Path) -> None:
    ese = tmp_path / "ese"
    ese.mkdir()
    (ese / "mesh.ply").write_text("")

    assert find_project_artifacts(tmp_path).ese_mesh == ese / "mesh.ply"


def test_scan_ese_mesh_keeps_legacy_name(tmp_path: Path) -> None:
    """Old projects store the ESE mesh as ``ese/ese_mesh.ply``; still found."""
    ese = tmp_path / "ese"
    ese.mkdir()
    (ese / "ese_mesh.ply").write_text("")

    assert find_project_artifacts(tmp_path).ese_mesh == ese / "ese_mesh.ply"


def test_scan_ese_mesh_prefers_new_name_over_legacy(tmp_path: Path) -> None:
    ese = tmp_path / "ese"
    ese.mkdir()
    (ese / "ese_mesh.ply").write_text("")
    (ese / "mesh.ply").write_text("")

    assert find_project_artifacts(tmp_path).ese_mesh == ese / "mesh.ply"


def test_scan_prefers_canonical_folder_over_stray_copy(tmp_path: Path) -> None:
    """A GUI-written canonical file wins over a stray or legacy copy."""
    stray = tmp_path / "mesh"
    stray.mkdir()
    (stray / "final_mesh.ply").write_text("")
    canonical = tmp_path / "scalp"
    canonical.mkdir()
    (canonical / "final_mesh.ply").write_text("")

    artifacts = find_project_artifacts(tmp_path)

    assert artifacts.scalp_mesh == canonical / "final_mesh.ply"


def test_scan_falls_back_to_any_folder(tmp_path: Path) -> None:
    """Without a canonical copy the file is still found anywhere."""
    stray = tmp_path / "mesh"
    stray.mkdir()
    (stray / "final_mesh.ply").write_text("")

    assert find_project_artifacts(tmp_path).scalp_mesh == stray / "final_mesh.ply"


def test_scan_mesh_ply_outside_ese_is_not_an_ese_mesh(tmp_path: Path) -> None:
    (tmp_path / "mesh").mkdir()
    (tmp_path / "mesh" / "mesh.ply").write_text("")

    artifacts = find_project_artifacts(tmp_path)

    assert artifacts.ese_mesh is None
    assert artifacts.scalp_mesh is None


def test_scan_prefers_nii_gz_over_nii(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "b.nii").write_text("")
    (input_dir / "a.nii.gz").write_text("")

    artifacts = find_project_artifacts(tmp_path)

    assert artifacts.nifti == input_dir / "a.nii.gz"


def test_scan_plain_nii_when_no_gz(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "b.nii").write_text("")
    (input_dir / "a.nii").write_text("")

    artifacts = find_project_artifacts(tmp_path)

    assert artifacts.nifti == input_dir / "a.nii"


def test_scan_reports_partial_artifacts(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        fiducials=False,
        scalp_points=False,
        ese_mesh=False,
        ese_points=False,
    )

    artifacts = find_project_artifacts(project)

    assert artifacts.nifti is not None
    assert artifacts.fiducials is None
    assert artifacts.scalp_mesh is not None
    assert artifacts.scalp_points is None
    assert artifacts.ese_mesh is None
    assert artifacts.ese_points is None


def test_scan_skips_dot_directories(tmp_path: Path) -> None:
    """Tool/venv caches never contribute artifacts, wherever they hide."""
    hidden = tmp_path / ".opencode" / "cache"
    hidden.mkdir(parents=True)
    (hidden / "fiducials.json").write_text("")
    (hidden / "final_mesh.ply").write_text("")

    artifacts = find_project_artifacts(tmp_path)

    assert artifacts.fiducials is None
    assert artifacts.scalp_mesh is None


# --- seed_steps_from_artifacts ---


def test_seed_only_nifti(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        fiducials=False,
        scalp_mesh=False,
        scalp_points=False,
        ese_mesh=False,
        ese_points=False,
    )
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 0
    assert manager.all_contexts() == [{"volume": project / "input" / "volume.nii.gz"}]
    assert manager.is_step_done(0) is True


def test_seed_nifti_and_scalp(tmp_path: Path) -> None:
    project = make_project(
        tmp_path, fiducials=False, ese_mesh=False, ese_points=False, scalp_points=False
    )
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 2
    contexts = manager.all_contexts()
    assert contexts[1] == {"scalp_surface": project / "mesh" / "final_mesh.ply"}
    assert contexts[2] == {"mesh": project / "mesh" / "final_mesh.ply"}
    assert manager.is_step_done(1) is True
    assert manager.is_step_done(2) is True


def test_seed_full_set_without_electrodes(tmp_path: Path) -> None:
    project = make_project(tmp_path, ese_points=False)
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 4
    contexts = manager.all_contexts()
    assert contexts[3] == {"ese_mesh": project / "ese" / "ese_mesh.ply"}
    assert contexts[4] == {"scalp_points": project / "localization" / "electrodes_scalp.json"}
    assert manager.get_global_item("fiducials") == project / "input" / "fiducials.json"
    # "normals" is in-memory only and not part of the done state.
    assert "normals" not in contexts[3]
    assert manager.is_step_done(3) is True


def test_seed_full_set_with_electrodes(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 4
    contexts = manager.all_contexts()
    assert contexts[4] == {
        "scalp_points": project / "localization" / "electrodes_scalp.json",
        "ese_points": project / "localization" / "electrodes_ese.json",
    }
    assert manager.get_global_item("fiducials") == project / "input" / "fiducials.json"


def test_seed_nothing_when_no_artifacts(tmp_path: Path) -> None:
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(tmp_path))

    assert manager.get_current_step() == 0
    assert manager.all_contexts() == [{}]


def test_seed_ese_points_without_ese_mesh(tmp_path: Path) -> None:
    project = make_project(tmp_path, ese_mesh=False, fiducials=False, scalp_points=False)
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 4
    contexts = manager.all_contexts()
    assert contexts[3] == {}
    assert contexts[4] == {"ese_points": project / "localization" / "electrodes_ese.json"}


def test_seed_fiducials_only_still_reaches_step_4(tmp_path: Path) -> None:
    """Fiducials are step-4 artifacts for navigation: the menu shows step 4."""
    project = make_project(
        tmp_path,
        nifti=None,
        scalp_mesh=False,
        scalp_points=False,
        ese_mesh=False,
        ese_points=False,
    )
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 4
    assert manager.all_contexts() == [{}, {}, {}, {}, {}]
    assert manager.get_global_item("fiducials") == project / "input" / "fiducials.json"


def test_invalidation_preserves_seeded_fiducials_and_wipes_step_4(tmp_path: Path) -> None:
    """The global store survives; step-4 context items are wiped."""
    project = make_project(tmp_path)
    manager = new_manager()
    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    manager.invalidate_steps_after(1)

    assert manager.get_current_step() == 1
    assert manager.get_global_item("fiducials") == project / "input" / "fiducials.json"
    with pytest.raises(ValueError, match="out of bounds"):
        manager.get_step_context_item(2, "mesh")


def test_seed_base_mesh_only_seeds_step_1(tmp_path: Path) -> None:
    """A lone raw base mesh is the step-1 scalp surface; step 2 stays empty."""
    project = make_project(
        tmp_path,
        fiducials=False,
        scalp_mesh=False,
        scalp_points=False,
        ese_mesh=False,
        ese_points=False,
        base_mesh=True,
    )
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 1
    assert manager.all_contexts()[1] == {"scalp_surface": project / "scalp" / "base_mesh.ply"}


def test_seed_final_mesh_wins_over_base_mesh(tmp_path: Path) -> None:
    """The postprocessed final mesh feeds steps 1 and 2; the base mesh is unused."""
    project = make_project(
        tmp_path,
        fiducials=False,
        scalp_points=False,
        ese_mesh=False,
        ese_points=False,
        base_mesh=True,
    )
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 2
    contexts = manager.all_contexts()
    assert contexts[1] == {"scalp_surface": project / "mesh" / "final_mesh.ply"}
    assert contexts[2] == {"mesh": project / "mesh" / "final_mesh.ply"}
