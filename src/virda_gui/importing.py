"""Role-based project import: copy an artifact into its canonical location.

A *role* says what kind of artifact a source file is (e.g. the localized
electrodes or the usual mesh); importing copies the file into the project's
canonical location for that role, so the pipeline and viewer find it without
extra configuration.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportRole:
    """One importable artifact role."""

    key: str
    label: str
    group: str
    target: str | None


ROLE_REGISTRY: tuple[ImportRole, ...] = (
    ImportRole("mesh", "Usual mesh", "Mesh", "mesh/final_mesh.ply"),
    ImportRole("scalp_vertices", "Scalp vertices", "Mesh", "mesh/scalp_vertices.npy"),
    ImportRole("scalp_faces", "Scalp faces", "Mesh", "mesh/scalp_faces.npy"),
    ImportRole(
        "scalp_face_adjacency",
        "Scalp face adjacency",
        "Mesh",
        "mesh/scalp_face_adjacency.npy",
    ),
    ImportRole("ese_mesh", "ESE mesh", "ESE", "ese/ese_mesh.ply"),
    ImportRole("normals", "Normals", "ESE", "ese/normals.npy"),
    ImportRole(
        "electrodes",
        "Localized electrodes",
        "Localization",
        "localization/electrodes.json",
    ),
    ImportRole("nifti", "NIfTI scan", "Input", None),
    ImportRole("fiducials", "Fiducials", "Input", "input/fiducials.json"),
    ImportRole("config", "Pipeline config", "Input", "input/config.json"),
    ImportRole("measurements", "Measurements", "Input", "input/measurements.json"),
)


def import_target(role: ImportRole, project: Path, source: Path) -> Path:
    """Resolve where *source* would land for *role* (without copying)."""
    if role.target is None:
        return project / "input" / source.name
    return project / role.target


def import_file(
    project: Path,
    source: str | Path,
    role: ImportRole,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy *source* into the project as *role*; return the destination.

    Raises :class:`FileExistsError` when the destination already exists and
    ``overwrite`` is not set.
    """
    destination = import_target(role, project, Path(source))
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
