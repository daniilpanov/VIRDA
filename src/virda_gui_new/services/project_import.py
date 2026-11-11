"""Project import preparation: scan, load, and validate artifact files.

Every scanned artifact file is loaded into its ready in-memory object
through the same loaders the popups use, so unsupported or outdated
formats fail at open time, before any step context is seeded. Seeding
then writes ready objects: a step marked done (or opened) is
guaranteed to hold runnable inputs.

Precedence rules:

- ``fiducials.json`` wins over ``coordsystem.json``; a ``coordsystem.json``
  alone is converted to fiducials through
  :meth:`virda.models.coordsystem.Coordsystem.to_fiducials` (millimetre
  MRI surface-RAS world coordinates). The winner fills the single
  ``fiducials`` input value that seeds the global store.
- ``final_mesh.ply`` (the postprocessed mesh) wins over ``base_mesh.ply``
  (the raw extraction result): the final mesh seeds both the step-1
  scalp surface and the step-2 mesh, a lone base mesh seeds only the
  step-1 scalp surface.
- ``measurements.json`` is loaded into an :class:`Electrodes` model and
  kept for the localization step: seeding writes it into the step-4
  context (``measurements``) so the export path can use it in memory.
- ``pipeline_config.json`` stays a scanned path: nothing in this GUI
  consumes it yet, so it is not loaded.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from virda.models.electrode import Electrodes
from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_loading
from virda_gui_new.services.project_scan import ProjectArtifacts, find_project_artifacts

_T = TypeVar("_T")


@dataclass(frozen=True)
class LoadedArtifacts:
    """Ready in-memory pipeline artifacts; ``None`` where absent or invalid.

    Field names follow :class:`ProjectArtifacts`, so the same seeding
    contract applies; the values are ready objects instead of paths --
    except ``pipeline_config``, which is scanned but deliberately not
    loaded (nothing in this GUI consumes it yet). ``fiducials`` is
    resolved at prepare time from ``fiducials.json`` or
    ``coordsystem.json`` (see the module precedence rules).
    """

    nifti: MRIVolume | None = None
    scalp_mesh: Mesh | None = None
    ese_mesh: Mesh | None = None
    fiducials: LabeledPoints | None = None
    scalp_points: LabeledPoints | None = None
    ese_points: LabeledPoints | None = None
    measurements: Electrodes | None = None
    base_mesh: Mesh | None = None
    pipeline_config: Path | None = None


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
        base_mesh=_load(scanned.base_mesh, artifact_loading.load_mesh, project, failures),
        ese_mesh=_load(scanned.ese_mesh, artifact_loading.load_mesh, project, failures),
        fiducials=_resolve_fiducials(scanned, project, failures),
        scalp_points=_load(
            scanned.scalp_points, artifact_loading.load_electrode_points, project, failures
        ),
        ese_points=_load(
            scanned.ese_points, artifact_loading.load_electrode_points, project, failures
        ),
        measurements=_load(
            scanned.measurements, artifact_loading.load_measurements, project, failures
        ),
        pipeline_config=scanned.pipeline_config,
    )
    return PreparedProject(
        loaded=loaded,
        failures=tuple(failures),
        recoverable_step=_last_seeded_step(loaded),
    )


def _resolve_fiducials(
    scanned: ProjectArtifacts, project: Path, failures: list[str]
) -> LabeledPoints | None:
    """Load the fiducial input: ``fiducials.json`` wins over the coordsystem.

    Both files are read through the same loader, which converts an MNE
    ``coordsystem.json`` into fiducials.
    """
    path = scanned.fiducials if scanned.fiducials is not None else scanned.coordsystem
    return _load(path, artifact_loading.load_fiducial_points, project, failures)


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
    any_localization_inputs = (
        loaded.fiducials is not None
        or loaded.scalp_points is not None
        or loaded.ese_points is not None
        or loaded.measurements is not None
    )
    if any_localization_inputs:
        return 4
    if loaded.ese_mesh is not None:
        return 3
    if loaded.scalp_mesh is not None:
        return 2
    if loaded.base_mesh is not None:
        return 1
    if loaded.nifti is not None:
        return 0
    return -1
