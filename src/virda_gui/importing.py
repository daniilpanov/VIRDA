"""Role-based project import: copy an artifact into its canonical location.

A *role* says what kind of artifact a source file is (e.g. the localized
electrodes or the usual mesh); importing copies the file into the project's
canonical location for that role, so the viewer finds it without extra
configuration.

The import flow is user-driven: a source file is picked first, its role is
auto-detected (:func:`detect_role`) and validated (:func:`validate_import_source`),
and only when the role is ambiguous does the UI ask the user to pick it from
:const:`IMPORT_FALLBACK_ROLES`.  Everything here is Qt-free so it is
unit-testable headless.
"""

import json
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
    ImportRole("ese_mesh", "ESE mesh", "ESE", "ese/ese_mesh.ply"),
    ImportRole("normals", "Normals", "ESE", "ese/normals.npy"),
    ImportRole(
        "electrodes",
        "Localized electrodes",
        "Localization",
        "localization/electrodes.json",
    ),
    ImportRole("nifti", "NIfTI scan", "Input", None),
    ImportRole("measurements", "Measurements", "Input", "input/measurements.json"),
    ImportRole("fiducials", "Fiducials", "Input", "input/fiducials.json"),
)

#: The roles the user is asked about when detection is ambiguous, in the order
#: shown in the fallback role picker.
IMPORT_FALLBACK_ROLES: tuple[ImportRole, ...] = (
    ROLE_REGISTRY[4],  # nifti
    ROLE_REGISTRY[0],  # mesh
    ROLE_REGISTRY[1],  # ese_mesh
    ROLE_REGISTRY[2],  # normals
    ROLE_REGISTRY[3],  # electrodes
    ROLE_REGISTRY[5],  # measurements
)


def _registry(key: str) -> ImportRole:
    return next(role for role in ROLE_REGISTRY if role.key == key)


def detect_role(path: str | Path) -> ImportRole | None:
    """Guess the import role of *path*, or None when it is ambiguous.

    Detection is based on the file name/extension and, for JSON files, on the
    document structure: NIfTI scans by extension, PLY meshes by name (an
    ``ese``-named file is an ESE mesh, otherwise the role is ambiguous),
    ``normals.npy`` by name, and JSON files by their top-level shape.  A mesh
    part stored in ``.npy`` is *not* recognisable as a mesh (a single array
    lacks the vertex/face geometry) so it reports None and fails validation.
    """
    source = Path(path)
    name = source.name.lower()
    suffix = source.suffix.lower()

    if suffix in {".nii", ".nifti"} or name.endswith(".nii.gz"):
        return _registry("nifti")
    if suffix == ".ply":
        if "ese" in name:
            return _registry("ese_mesh")
        return None
    if suffix == ".npy":
        if "normals" in name:
            return _registry("normals")
        return None
    if suffix == ".json":
        return detect_json_role(source)
    return None


def detect_json_role(path: Path) -> ImportRole | None:
    """Guess a JSON document's role from its structure."""
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        if data and all(isinstance(item, dict) for item in data):
            first = data[0]
            if any(key in first for key in ("ese_coords", "scalp_coords", "coords")):
                return _registry("electrodes")
        return None
    if isinstance(data, dict):
        if "electrodes" in data:
            return _registry("measurements")
        if "fiducials" in data:
            return _registry("fiducials")
    return None


def validate_import_source(role: ImportRole, path: str | Path) -> None:
    """Raise :class:`ValueError` when *path* is not a valid source for *role*."""
    source = Path(path)
    try:
        if role.key in {"mesh", "ese_mesh"}:
            _validate_mesh(source, role)
        elif role.key == "nifti":
            _validate_nifti(source)
        elif role.key == "normals":
            _validate_normals(source)
        elif role.key == "electrodes":
            _validate_electrodes(source)
        elif role.key == "measurements":
            from virda.io.importers.measurements import import_measurements

            import_measurements(source)
        elif role.key == "fiducials":
            from virda.io.importers.fiducials import import_fiducials

            import_fiducials(source)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - any loading failure is surfaced
        raise ValueError(f"{role.label}: could not read the file: {exc}") from exc


def _validate_mesh(source: Path, role: ImportRole) -> None:
    if source.suffix.lower() != ".ply":
        raise ValueError(
            f"{role.label}: only triangular PLY meshes can be imported, "
            f"got {source.name}. A single NPY array cannot be rebuilt into a mesh."
        )
    from virda.io.importers.scalp_mesh import import_scalp_mesh

    import_scalp_mesh(source)


def _validate_nifti(source: Path) -> None:
    if not source.exists():
        raise ValueError(f"NIfTI scan not found: {source}")
    import nibabel as nib

    try:
        image = nib.load(source)
    except Exception as exc:
        raise ValueError(f"NIfTI scan: not a valid NIfTI file: {exc}") from exc
    if not isinstance(image, nib.Nifti1Image):
        raise ValueError("NIfTI scan: expected a NIfTI-1 image.")


def _validate_normals(source: Path) -> None:
    import numpy as np

    array = np.load(source)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(
            f"Normals: expected an (N, 3) point array, got shape {array.shape}"
        )


def _validate_electrodes(source: Path) -> None:
    with source.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Localized electrodes: expected a JSON array of electrodes.")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Localized electrodes: item {index} is not an electrode object."
            )
        if not any(key in item for key in ("ese_coords", "scalp_coords", "coords")):
            raise ValueError(
                f"Localized electrodes: item {index} has no coordinate field."
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