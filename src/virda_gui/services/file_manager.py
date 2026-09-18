"""Open folders in the platform file manager."""

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path


def file_manager_opener() -> Callable[[Path], None] | None:
    """Return a callable that reveals a folder, or None if unsupported."""
    if hasattr(os, "startfile"):  # Windows
        return lambda path: os.startfile(str(path))  # noqa: S606
    for command in ("xdg-open", "open"):  # Linux, macOS
        if shutil.which(command) is not None:
            return lambda path, command=command: subprocess.run(
                [command, str(path)],
                check=True,
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    return None


def open_in_file_manager(path: Path) -> None:
    """Reveal ``path`` in the platform file manager.

    Raises :class:`OSError`, :class:`subprocess.SubprocessError` or
    :class:`RuntimeError` (no supported manager exists) when the folder cannot
    be opened.
    """
    opener = file_manager_opener()
    if opener is None:
        raise RuntimeError("no supported file manager found on this platform")
    opener(path)
