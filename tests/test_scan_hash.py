"""Unit tests for the NIfTI source-hash sidecar (Qt-free)."""

from pathlib import Path

from virda_gui.scan_hash import (
    SOURCE_HASH_FILENAME,
    read_source_hash,
    sha256_file,
    write_source_hash,
)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "scan.nii.gz"
    target.write_bytes(bytes(range(256)) * 100)
    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_sidecar_round_trip(tmp_path: Path) -> None:
    mesh_dir = tmp_path / "mesh"
    path = write_source_hash(mesh_dir, tmp_path / "head.nii.gz", "a" * 64)
    assert path == mesh_dir / SOURCE_HASH_FILENAME
    assert read_source_hash(mesh_dir) == "a" * 64


def test_read_source_hash_missing_or_malformed(tmp_path: Path) -> None:
    assert read_source_hash(tmp_path / "mesh") is None
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    (mesh_dir / SOURCE_HASH_FILENAME).write_text("garbage\n", encoding="utf-8")
    assert read_source_hash(mesh_dir) is None
    (mesh_dir / SOURCE_HASH_FILENAME).write_text("", encoding="utf-8")
    assert read_source_hash(mesh_dir) is None


def test_write_source_hash_creates_mesh_dir(tmp_path: Path) -> None:
    mesh_dir = tmp_path / "proj" / "mesh"
    write_source_hash(mesh_dir, "other.nii", "b" * 64)
    assert read_source_hash(mesh_dir) == "b" * 64
