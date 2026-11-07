"""Unit tests for project scanning and step-context seeding."""

from pathlib import Path

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


def test_scan_of_empty_directory_finds_nothing(tmp_path: Path) -> None:
    artifacts = find_project_artifacts(tmp_path)

    assert artifacts == ProjectArtifacts()


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
    assert contexts[4] == {
        "fiducials": project / "input" / "fiducials.json",
        "scalp_points": project / "localization" / "electrodes_scalp.json",
    }

    assert manager.is_step_done(3) is True


def test_seed_full_set_with_electrodes(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 4
    contexts = manager.all_contexts()
    assert contexts[4] == {
        "fiducials": project / "input" / "fiducials.json",
        "scalp_points": project / "localization" / "electrodes_scalp.json",
        "ese_points": project / "localization" / "electrodes_ese.json",
    }


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
