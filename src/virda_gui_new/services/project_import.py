"""Project import preparation: scan, load, and validate artifact files.

Every scanned artifact file is loaded into its ready in-memory object, so
unsupported or outdated formats fail at open time, before any step
context is seeded. Seeding then writes ready objects: a step marked done
(or opened) is guaranteed to hold runnable inputs.

Precedence rules: ``fiducials.json`` wins over ``coordsystem.json``
(converted to fiducials through
:meth:`virda.models.coordsystem.Coordsystem.to_fiducials`); the
postprocessed ``final_mesh.ply`` wins over ``base_mesh.ply`` and seeds
the step-1 scalp surface and the step-2 mesh, a lone base mesh seeds
only the step-1 scalp surface; ``measurements.json`` is loaded into an
:class:`Electrodes` model; ``pipeline_config.json`` stays a scanned path.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from virda.models.electrode import Electrodes
from virda.models.mesh import Mesh, Normals
from virda.models.mri_volume import MRIVolume
from virda.models.stage2_config import Stage2Config
from virda.ops.build_ese import estimate_normals
from virda_gui_new.dto.steps import ESE_MESH
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_loading
from virda_gui_new.services.project_scan import ProjectArtifacts, find_project_artifacts
from virda_gui_new.services.step_config import load_step_config

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
    normals: Normals | None = None
    fiducials: LabeledPoints | None = None
    scalp_points: LabeledPoints | None = None
    ese_points: LabeledPoints | None = None
    measurements: Electrodes | None = None
    base_mesh: Mesh | None = None
    pipeline_config: Path | None = None


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
    scalp_mesh = _load(scanned.scalp_mesh, artifact_loading.load_mesh, project, failures)
    base_mesh = _load(scanned.base_mesh, artifact_loading.load_mesh, project, failures)
    loaded = LoadedArtifacts(
        nifti=_load(scanned.nifti, artifact_loading.load_volume, project, failures),
        scalp_mesh=scalp_mesh,
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
        base_mesh=base_mesh,
        pipeline_config=scanned.pipeline_config,
    )
    loaded = _with_normals(loaded, scanned, project, failures)
    return PreparedProject(loaded=loaded, failures=tuple(failures))


def _with_normals(
    loaded: LoadedArtifacts,
    scanned: ProjectArtifacts,
    project: Path,
    failures: list[str],
) -> LoadedArtifacts:
    """The loaded artifacts together with the normals of the scalp mesh.

    The normals are a byproduct of the ESE step, and the localization
    needs them. A project that stored them is read as it is; an older
    one has them estimated from the scalp mesh, with the settings of
    the ESE step. Without an ESE mesh there is nothing to estimate:
    the localization cannot run without one.
    """
    if loaded.ese_mesh is None:
        return loaded
    if scanned.ese_normals is not None:
        normals = _load(scanned.ese_normals, artifact_loading.load_ese_normals, project, failures)
        if normals is not None:
            return replace(loaded, normals=normals)
    scalp_mesh = loaded.scalp_mesh if loaded.scalp_mesh is not None else loaded.base_mesh
    if scalp_mesh is None:
        return loaded
    return replace(loaded, normals=estimate_normals(scalp_mesh, _ese_config(project)))


def _ese_config(project: Path) -> Stage2Config:
    """The ESE step settings of the project (the defaults without them)."""
    stored = load_step_config(project, ESE_MESH)
    auto_k = stored.get("use_auto_k")
    k_neighbors = stored.get("k_neighbors")
    return Stage2Config(
        neighborhood_radius_mm=_number(stored, "neighborhood_radius_mm", 10.0),
        k_neighbors=None if auto_k is not False else int(k_neighbors or 10),
        use_weighted_pca=stored.get("use_weighted_pca") is True,
        pca_sigma_mm=_number(stored, "pca_sigma_mm", 5.0),
        min_neighbors=int(_number(stored, "min_neighbors", 5)),
    )


def _number(stored: dict[str, Any], key: str, default: float) -> float:
    """A numeric setting of the step configuration (the default without one)."""
    value = stored.get(key)
    return float(value) if isinstance(value, int | float) else default


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
