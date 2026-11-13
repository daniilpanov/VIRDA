"""Project directory scan: report which ready pipeline artifacts exist on disk."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectArtifacts:
    """Paths of ready project artifacts; ``None`` where the file is absent."""

    nifti: Path | None = None
    fiducials: Path | None = None
    scalp_mesh: Path | None = None
    scalp_points: Path | None = None
    ese_mesh: Path | None = None
    ese_points: Path | None = None


def _find_nifti(project: Path) -> Path | None:
    """First sorted ``input`` volume: ``*.nii.gz`` preferred over ``*.nii``."""
    input_dir = project / "input"
    for pattern in ("*.nii.gz", "*.nii"):
        found = sorted(input_dir.glob(pattern))
        if found:
            return found[0]
    return None


def find_project_artifacts(project: Path) -> ProjectArtifacts:
    """Scan a VIRDA project directory for ready artifact files.

    Scanning only: file contents are not read. See :class:`ProjectArtifacts`
    for the reported slots.
    """
    input_dir = project / "input"

    def existing(path: Path) -> Path | None:
        return path if path.is_file() else None

    return ProjectArtifacts(
        nifti=_find_nifti(project),
        fiducials=existing(input_dir / "fiducials.json"),
        scalp_mesh=existing(project / "mesh" / "final_mesh.ply"),
        scalp_points=existing(project / "localization" / "electrodes_scalp.json"),
        ese_mesh=existing(project / "ese" / "ese_mesh.ply"),
        ese_points=existing(project / "localization" / "electrodes_ese.json"),
    )
