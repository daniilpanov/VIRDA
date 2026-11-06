"""Tests for the pure patient-project helpers in :mod:`virda_gui.project`."""

from pathlib import Path

import pytest

from virda_gui.project import (
    classify_artifact,
    create_project,
    format_file_size,
    format_mtime,
    is_project,
    scan_project,
)


def test_create_project_makes_directory(tmp_path: Path) -> None:
    project = create_project(tmp_path / "new" / "CTRL_0001")
    assert project.is_dir()


def test_create_project_accepts_existing_directory(tmp_path: Path) -> None:
    project = tmp_path / "existing"
    project.mkdir()
    assert create_project(project) == project


def test_is_project_requires_an_existing_directory(tmp_path: Path) -> None:
    assert is_project(None) is False
    assert is_project("") is False
    assert is_project(tmp_path / "missing") is False
    assert is_project(tmp_path / "a.txt") is False
    project = create_project(tmp_path / "proj")
    assert is_project(project) is True


def test_scan_project_orders_artifact_dirs_and_files(tmp_path: Path) -> None:
    project = create_project(tmp_path / "proj")
    (project / "mesh").mkdir()
    (project / "ese").mkdir()
    (project / "localization").mkdir()
    (project / "zzz_extra").mkdir()
    (project / "aaa_extra").mkdir()
    (project / "notes.txt").write_text("x", encoding="utf-8")
    (project / "a.csv").write_text("x", encoding="utf-8")

    scan = scan_project(project)

    assert scan.root == project
    assert scan.groups == [project / "mesh", project / "ese", project / "localization"]
    assert scan.extra_dirs == [project / "aaa_extra", project / "zzz_extra"]
    assert scan.loose_files == [project / "a.csv", project / "notes.txt"]


def test_scan_project_respects_artifact_dirs_override(tmp_path: Path) -> None:
    project = create_project(tmp_path / "proj")
    (project / "input").mkdir()
    (project / "mesh").mkdir()

    scan = scan_project(project, artifact_dirs=["input", "esh", "mesh"])

    assert scan.groups == [project / "input", project / "mesh"]
    assert scan.extra_dirs == []


def test_scan_project_empty_project(tmp_path: Path) -> None:
    project = create_project(tmp_path / "proj")

    scan = scan_project(project)

    assert scan.groups == []
    assert scan.extra_dirs == []
    assert scan.loose_files == []


def test_scan_project_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_project(tmp_path / "missing")


def test_format_file_size_scales_units(tmp_path: Path) -> None:
    small = tmp_path / "small.bin"
    small.write_bytes(b"x" * 512)
    assert format_file_size(small) == "512 B"

    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (1500 * 1024))
    assert format_file_size(big) == "1.5 MB"


def test_format_mtime_reports_iso_datetime(tmp_path: Path) -> None:
    artifact = tmp_path / "note.txt"
    artifact.write_text("hi", encoding="utf-8")
    formatted = format_mtime(artifact)
    assert len(formatted) == 16
    assert formatted[4] == "-" and formatted[10] == " " and formatted[13] == ":"


def test_classify_artifact_recognises_visual_files(tmp_path: Path) -> None:
    mesh = tmp_path / "final_mesh.ply"
    mesh.write_bytes(b"ply\n")
    assert classify_artifact(mesh) == "mesh"

    nifti_gz = tmp_path / "head.nii.gz"
    nifti_gz.write_bytes(b"\x00")
    assert classify_artifact(nifti_gz) == "nifti"

    nifti = tmp_path / "head.nii"
    nifti.write_bytes(b"\x00")
    assert classify_artifact(nifti) == "nifti"

    assert classify_artifact(tmp_path / "large.PLY") == "mesh"


def test_classify_artifact_falls_back_to_text(tmp_path: Path) -> None:
    note = tmp_path / "README.txt"
    note.write_text("hi", encoding="utf-8")
    assert classify_artifact(note) == "text"
