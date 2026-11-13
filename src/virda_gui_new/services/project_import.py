"""Project import preparation: scan, load, and validate artifact files.

Every scanned artifact file is loaded into its ready in-memory object
through the same loaders the popups use, so unsupported or outdated
formats fail at open time, before any step context is seeded. Seeding
then writes ready objects: a step marked done (or opened) is
guaranteed to hold runnable inputs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_loading
from virda_gui_new.services.project_scan import find_project_artifacts

_T = TypeVar("_T")


@dataclass(frozen=True)
class LoadedArtifacts:
    """Ready in-memory pipeline artifacts; ``None`` where absent or invalid.

    Field names match :class:`ProjectArtifacts`, so the same seeding
    contract applies; the values are ready objects instead of paths.
    """

    nifti: MRIVolume | None = None
    scalp_mesh: Mesh | None = None
    ese_mesh: Mesh | None = None
    fiducials: LabeledPoints | None = None
    scalp_points: LabeledPoints | None = None
    ese_points: LabeledPoints | None = None


@dataclass(frozen=True)
class PreparedProject:
    """A scanned project with every loadable artifact in memory.

    ``failures`` reports dropped files as ``"<relative path>: <reason>"``;
    ``recoverable_step`` is the last step that still receives a seeding
    write from the loaded artifacts (``-1`` when nothing is left).
    """

    loaded: LoadedArtifacts
    failures: tuple[str, ...] = ()
    recoverable_step: int = -1


def prepare_project(project: Path) -> PreparedProject:
    """Scan ``project`` and load every artifact file it contains."""
    scanned = find_project_artifacts(project)
    failures: list[str] = []
    loaded = LoadedArtifacts(
        nifti=_load(scanned.nifti, artifact_loading.load_volume, project, failures),
        scalp_mesh=_load(scanned.scalp_mesh, artifact_loading.load_mesh, project, failures),
        ese_mesh=_load(scanned.ese_mesh, artifact_loading.load_mesh, project, failures),
        fiducials=_load(
            scanned.fiducials, artifact_loading.load_fiducial_points, project, failures
        ),
        scalp_points=_load(
            scanned.scalp_points, artifact_loading.load_electrode_points, project, failures
        ),
        ese_points=_load(
            scanned.ese_points, artifact_loading.load_electrode_points, project, failures
        ),
    )
    return PreparedProject(
        loaded=loaded,
        failures=tuple(failures),
        recoverable_step=_last_seeded_step(loaded),
    )


def _load[T](
    path: Path | None,
    loader: Callable[[Path], T],
    project: Path,
    failures: list[str],
) -> T | None:
    """Load one artifact file; a failure is reported and yields ``None``."""
    if path is None:
        return None
    try:
        return loader(path)
    except Exception as error:  # noqa: BLE001 - any load error is user-facing
        failures.append(f"{path.relative_to(project)}: {error}")
        return None


def _last_seeded_step(loaded: LoadedArtifacts) -> int:
    """Mirror the seeding contract: the last step that receives a write."""
    any_points = (
        loaded.fiducials is not None
        or loaded.scalp_points is not None
        or loaded.ese_points is not None
    )
    if any_points:
        return 4
    if loaded.ese_mesh is not None:
        return 3
    if loaded.scalp_mesh is not None:
        return 2
    if loaded.nifti is not None:
        return 0
    return -1
