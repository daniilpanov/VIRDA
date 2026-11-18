"""Canonical project-relative paths of the artifacts.

One place knows where every artifact of a project lives: the step
savers write them, the exporter offers them as defaults and the
scanner finds them by name.
"""

from pathlib import Path

#: The subdirectory that holds the localization step's inputs and outputs.
LOCALIZATION_DIR = "localization"
#: The subdirectory that holds the canonical copy of the source MRI.
INPUT_DIR = "input"
#: The subdirectory that holds the head segmentation.
SEGMENTATION_DIR = "segmentation"
#: The subdirectory that holds the scalp meshes.
SCALP_DIR = "scalp"
#: The subdirectory that holds the ESE mesh and its normals.
ESE_DIR = "ese"


def head_mask(project: Path) -> Path:
    """The path of the head segmentation mask."""
    return project / SEGMENTATION_DIR / "head_mask.nii.gz"


def base_mesh(project: Path) -> Path:
    """The path of the raw scalp mesh."""
    return project / SCALP_DIR / "base_mesh.ply"


def final_mesh(project: Path) -> Path:
    """The path of the postprocessed scalp mesh."""
    return project / SCALP_DIR / "final_mesh.ply"


def ese_mesh(project: Path) -> Path:
    """The path of the ESE mesh."""
    return project / ESE_DIR / "mesh.ply"


def ese_normals(project: Path) -> Path:
    """The path of the ESE mesh normals."""
    return project / ESE_DIR / "normals.npy"


def fiducials(project: Path) -> Path:
    """The path of the fiducials."""
    return project / LOCALIZATION_DIR / "fiducials.json"


def measurements(project: Path) -> Path:
    """The path of the measured distances."""
    return project / LOCALIZATION_DIR / "measurements.json"


def scalp_points(project: Path) -> Path:
    """The path of the electrode contact points on the scalp."""
    return project / LOCALIZATION_DIR / "electrodes_scalp.csv"


def ese_points(project: Path) -> Path:
    """The path of the electrode body centres on the ESE."""
    return project / LOCALIZATION_DIR / "electrodes_ese.csv"


def electrodes(project: Path) -> Path:
    """The path of the exported electrode table."""
    return project / LOCALIZATION_DIR / "electrodes.csv"
