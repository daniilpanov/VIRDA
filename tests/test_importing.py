"""Unit tests for role-based project artifact import."""

from pathlib import Path

import pytest

from virda_gui.importing import ROLE_REGISTRY, import_file, import_target
from virda_gui.project import create_project


def _role(key: str):
    for role in ROLE_REGISTRY:
        if role.key == key:
            return role
    raise AssertionError(f"unknown role {key!r}")


def test_import_target_fixed_names(tmp_path: Path) -> None:
    project = create_project(tmp_path / "p")
    source = tmp_path / "whatever.ply"
    assert import_target(_role("mesh"), project, source) == project / "mesh" / "final_mesh.ply"
    assert import_target(_role("electrodes"), project, source) == (
        project / "localization" / "electrodes.json"
    )


def test_import_target_nifti_keeps_source_name(tmp_path: Path) -> None:
    project = create_project(tmp_path / "p")
    source = tmp_path / "patient.nii.gz"
    assert import_target(_role("nifti"), project, source) == project / "input" / "patient.nii.gz"


def test_import_file_copies_contents(tmp_path: Path) -> None:
    project = create_project(tmp_path / "p")
    source = tmp_path / "final_mesh.ply"
    source.write_text("solid\n", encoding="utf-8")

    destination = import_file(project, source, _role("mesh"))

    assert destination == project / "mesh" / "final_mesh.ply"
    assert destination.read_text(encoding="utf-8") == "solid\n"


def test_import_file_raises_without_overwrite(tmp_path: Path) -> None:
    project = create_project(tmp_path / "p")
    (project / "input").mkdir()
    source = tmp_path / "scan.nii.gz"
    source.write_bytes(b"\x00")
    existing = project / "input" / "scan.nii.gz"
    existing.write_bytes(b"old")

    with pytest.raises(FileExistsError):
        import_file(project, source, _role("nifti"))

    assert existing.read_bytes() == b"old"


def test_import_file_overwrites_existing(tmp_path: Path) -> None:
    project = create_project(tmp_path / "p")
    (project / "input").mkdir()
    source = tmp_path / "scan.nii.gz"
    source.write_bytes(b"new")
    existing = project / "input" / "scan.nii.gz"
    existing.write_bytes(b"old")

    destination = import_file(project, source, _role("nifti"), overwrite=True)

    assert destination.read_bytes() == b"new"


def test_import_creates_target_subdirectories(tmp_path: Path) -> None:
    project = create_project(tmp_path / "p")
    source = tmp_path / "ese_mesh.ply"
    source.write_bytes(b"ply\n")
    import_file(project, source, _role("ese_mesh"))
    assert (project / "ese" / "ese_mesh.ply").is_file()
