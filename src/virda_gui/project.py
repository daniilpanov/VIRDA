"""Pure helpers for VIRDA patient project directories (no Qt).

A patient project is a folder that stores the pipeline artifacts under a set
of well-known subdirectories (see ``PATIENT_PROJECT_FORMAT.md``).  These
helpers create, validate and scan project folders without touching Qt, so they
are unit-testable headless and can be shared by every part of the GUI
(sidebar, run pipeline tab, import).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from virda_gui.constants import PROJECT_ARTIFACT_DIRS


@dataclass(frozen=True)
class ProjectScan:
    """Snapshot of a project directory.

    ``groups`` lists the well-known artifact subdirectories that exist, in
    pipeline order; ``extra_dirs`` the remaining subdirectories sorted by name
    and ``loose_files`` the files directly inside the project root.
    """

    root: Path
    groups: list[Path]
    extra_dirs: list[Path]
    loose_files: list[Path]


def create_project(path: str | Path) -> Path:
    """Create a new empty project directory and return it.

    Creates *path* together with its parents; an existing directory is
    accepted as-is, so "create" doubles as "pick a folder" for a brand-new
    project.
    """
    project = Path(path)
    project.mkdir(parents=True, exist_ok=True)
    return project


def is_project(path: str | Path | None) -> bool:
    """Return whether *path* names an existing project directory."""
    if not path:
        return False
    return Path(path).is_dir()


def scan_project(path: str | Path, artifact_dirs: list[str] | None = None) -> ProjectScan:
    """Scan *path* and return its grouped artifact tree.

    Well-known artifact subdirectories are reported in pipeline order
    (``PROJECT_ARTIFACT_DIRS`` by default); unknown subdirectories and loose
    root files are appended sorted by name.  Raises :class:`FileNotFoundError`
    when the directory does not exist.
    """
    project = Path(path)
    if not project.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project}")

    order = artifact_dirs or PROJECT_ARTIFACT_DIRS
    known = [name for name in order if (project / name).is_dir()]
    extra = sorted(
        entry.name for entry in project.iterdir() if entry.is_dir() and entry.name not in order
    )
    return ProjectScan(
        root=project,
        groups=[project / name for name in known],
        extra_dirs=[project / name for name in extra],
        loose_files=sorted(entry for entry in project.iterdir() if entry.is_file()),
    )


def format_file_size(path: str | Path) -> str:
    """Human-readable file size for a project artifact file."""
    size = float(Path(path).stat().st_size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_mtime(path: str | Path) -> str:
    """Human-readable modification timestamp for a project artifact file."""
    return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def classify_artifact(path: str | Path) -> str:
    """Classify a project artifact for tab opening: ``mesh``, ``nifti`` or ``text``."""
    name = Path(path).name.lower()
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        return "nifti"
    if Path(path).suffix.lower() == ".ply":
        return "mesh"
    return "text"
