"""Unit tests for project scanning and step-context seeding."""

from pathlib import Path

import pytest

from virda_gui_new.dto.steps import STEPS
from virda_gui_new.services.project_scan import ProjectArtifacts, find_project_artifacts
from virda_gui_new.services.step_seeding import StepChainIncompleteErrorError, seed_steps_from_artifacts
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
        files.append(tmp_path / "localization" / "electrodes_scalp.csv")
    if ese_mesh:
        files.append(tmp_path / "ese" / "ese_mesh.ply")
    if ese_points:
        files.append(tmp_path / "localization" / "electrodes_ese.csv")
    if base_mesh:
        files.append(tmp_path / "scalp" / "base_mesh.ply")
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return tmp_path


def new_manager() -> StepsManager:
    return StepsManager(tuple(step.required_keys for step in STEPS))


# --- find_project_artifacts ---


def test_scan_reports_all_artifacts(tmp_path: Path) -> None:
    project = make_project(tmp_path)

    artifacts = find_project_artifacts(project)

    assert artifacts.nifti == project / "input" / "volume.nii.gz"
    assert artifacts.fiducials == project / "input" / "fiducials.json"
    assert artifacts.scalp_mesh == project / "mesh" / "final_mesh.ply"
    assert artifacts.scalp_points == project / "localization" / "electrodes_scalp.csv"
    assert artifacts.ese_mesh == project / "ese" / "ese_mesh.ply"
    assert artifacts.ese_points == project / "localization" / "electrodes_ese.csv"
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


def test_scan_nifti_found_anywhere(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "head.nii.gz").write_text("")

    assert find_project_artifacts(tmp_path).nifti == outside / "head.nii.gz"


def test_scan_ese_mesh_new_name_only_in_ese(tmp_path: Path) -> None:
    ese = tmp_path / "ese"
    ese.mkdir()
    (ese / "mesh.ply").write_text("")

    assert find_project_artifacts(tmp_path).ese_mesh == ese / "mesh.ply"


def test_scan_ese_mesh_old_name_anywhere(tmp_path: Path) -> None:
    """``ese_mesh.ply`` is a first-class rule searched in every folder."""
    outside = tmp_path / "mesh"
    outside.mkdir()
    (outside / "ese_mesh.ply").write_text("")

    assert find_project_artifacts(tmp_path).ese_mesh == outside / "ese_mesh.ply"


def test_scan_ese_mesh_new_name_outranks_old_name(tmp_path: Path) -> None:
    """Rule order decides: ``ese/mesh.ply`` before an ``ese_mesh.ply``."""
    ese = tmp_path / "ese"
    ese.mkdir()
    (ese / "ese_mesh.ply").write_text("")
    (ese / "mesh.ply").write_text("")

    assert find_project_artifacts(tmp_path).ese_mesh == ese / "mesh.ply"


def test_scan_duplicate_names_resolve_to_first_sorted_match(tmp_path: Path) -> None:
    """Several matching files fill the slot deterministically (sorted)."""
    (tmp_path / "z").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "z" / "final_mesh.ply").write_text("")
    (tmp_path / "a" / "final_mesh.ply").write_text("")

    artifacts = find_project_artifacts(tmp_path)

    assert artifacts.scalp_mesh == tmp_path / "a" / "final_mesh.ply"


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


def test_scan_head_mask_is_not_a_nifti(tmp_path: Path) -> None:
    """The head mask is excluded from the NIfTI globs by name."""
    segmentation = tmp_path / "segmentation"
    segmentation.mkdir()
    (segmentation / "head_mask.nii.gz").write_text("")

    artifacts = find_project_artifacts(tmp_path)

    assert artifacts.nifti is None
    assert artifacts.head_mask == segmentation / "head_mask.nii.gz"


def test_scan_json_electrode_files_are_ignored(tmp_path: Path) -> None:
    """The GUI-invented JSON point-row format is gone: no legacy support."""
    localization = tmp_path / "localization"
    localization.mkdir()
    (localization / "electrodes_scalp.json").write_text("[]", encoding="utf-8")
    (localization / "electrodes_ese.json").write_text("[]", encoding="utf-8")

    artifacts = find_project_artifacts(tmp_path)

    assert artifacts.scalp_points is None
    assert artifacts.ese_points is None


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

    # The pipeline continues from the first step without a result.
    assert manager.get_current_step() == 1
    assert manager.all_contexts() == [
        {"volume": project / "input" / "volume.nii.gz"},
        {},
    ]
    assert manager.is_step_done(0) is True


def test_seed_nifti_and_scalp(tmp_path: Path) -> None:
    project = make_project(
        tmp_path, fiducials=False, ese_mesh=False, ese_points=False, scalp_points=False
    )
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 3
    contexts = manager.all_contexts()
    assert contexts[1] == {"scalp_surface": project / "mesh" / "final_mesh.ply"}
    assert contexts[2] == {"mesh": project / "mesh" / "final_mesh.ply"}
    assert manager.is_step_done(1) is True
    assert manager.is_step_done(2) is True


def test_seed_full_set_without_electrodes(tmp_path: Path) -> None:
    project = make_project(tmp_path, ese_points=False)
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    # Every step is finished: the pipeline sits past the last one.
    assert manager.get_current_step() == 5
    contexts = manager.all_contexts()
    assert contexts[3] == {"ese_mesh": project / "ese" / "ese_mesh.ply"}
    assert contexts[4] == {"scalp_points": project / "localization" / "electrodes_scalp.csv"}
    assert manager.get_global_item("fiducials") == project / "input" / "fiducials.json"
    # "normals" is in-memory only and not part of the done state.
    assert "normals" not in contexts[3]
    assert manager.is_step_done(3) is True


def test_seed_full_set_with_electrodes(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert manager.get_current_step() == 5
    contexts = manager.all_contexts()
    assert contexts[4] == {
        "scalp_points": project / "localization" / "electrodes_scalp.csv",
        "ese_points": project / "localization" / "electrodes_ese.csv",
    }
    assert manager.get_global_item("fiducials") == project / "input" / "fiducials.json"


def test_seed_nothing_when_no_artifacts(tmp_path: Path) -> None:
    manager = new_manager()

    seed_steps_from_artifacts(manager, find_project_artifacts(tmp_path))

    assert manager.get_current_step() == 0
    assert manager.all_contexts() == [{}]


def test_seed_ese_points_without_ese_mesh_stops_at_the_gap(tmp_path: Path) -> None:
    """A missing ESE mesh stops the chain: the electrode points stay unloaded."""
    project = make_project(tmp_path, ese_mesh=False, fiducials=False, scalp_points=False)
    manager = new_manager()

    with pytest.raises(StepChainIncompleteErrorError) as raised:
        seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert raised.value.step_n == 3
    assert raised.value.missing_keys == ("ese_mesh",)
    assert manager.get_current_step() == 3
    contexts = manager.all_contexts()
    assert contexts[3] == {}
    assert len(contexts) == 4


def test_seed_fiducials_only_stops_at_the_missing_nifti(tmp_path: Path) -> None:
    """Fiducials are a project input, but the chain needs the volume first."""
    project = make_project(
        tmp_path,
        nifti=None,
        scalp_mesh=False,
        scalp_points=False,
        ese_mesh=False,
        ese_points=False,
    )
    manager = new_manager()

    with pytest.raises(StepChainIncompleteError) as raised:
        seed_steps_from_artifacts(manager, find_project_artifacts(project))

    assert raised.value.step_n == 0
    assert raised.value.missing_keys == ("volume",)
    assert manager.get_current_step() == 0
    assert manager.all_contexts() == [{}]
    # The fiducials are a project input: they are stored anyway.
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
