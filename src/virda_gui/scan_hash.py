"""SHA-256 tracking of the NIfTI scan a project's meshes were built from.

The mesh tab writes a small sidecar file next to the generated meshes so a
later import (or a regeneration from another file) can tell whether the scan
changed: same hash means the derived meshes are still valid, a different or
missing hash means they went stale.  Everything here is Qt-free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Sidecar next to the generated meshes holding the source scan hash.
SOURCE_HASH_FILENAME = "source_nifti.sha256"

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of the file at *path* (streamed in chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash_path(mesh_dir: str | Path) -> Path:
    """Return the sidecar path inside the project's ``mesh/`` directory."""
    return Path(mesh_dir) / SOURCE_HASH_FILENAME


def write_source_hash(mesh_dir: str | Path, source: str | Path, hash_hex: str) -> Path:
    """Record *hash_hex* of *source* in the ``mesh/`` sidecar; return its path."""
    target = source_hash_path(mesh_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{hash_hex}  {Path(source).name}\n", encoding="utf-8")
    return target


def read_source_hash(mesh_dir: str | Path) -> str | None:
    """Return the recorded source hash, or None when missing or malformed."""
    try:
        text = source_hash_path(mesh_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    digest, _, _name = text.partition("  ")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return digest
