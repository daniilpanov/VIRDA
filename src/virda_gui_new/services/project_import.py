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

    ``failures`` reports dropped files as ``"<relative path>: <reason>"``.
    The step the pipeline continues from is not kept here: that is the
    manager's current step after the seeding, which owns the rule.
    """

    loaded: LoadedArtifacts
    failures: tuple[str, ...] = ()


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
    return PreparedProject(loaded=loaded, failures=tuple(failures))


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
