"""Project import preparation: scan artifacts and validate their files.

Each scanned artifact file is loaded once through the same loader the
scene builder uses, so unsupported or outdated formats fail here, before
any step context is seeded. The result carries the validated artifacts
and a per-file failure report for the user.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from virda_gui_new.services import artifact_loading
from virda_gui_new.services.project_scan import ProjectArtifacts, find_project_artifacts

# ProjectArtifacts field -> (step index, context key, file loader), matching
# the seeding contract and the scene builder's rules table.
_FIELDS: tuple[tuple[str, int, str, Callable[[Path], Any]], ...] = (
    ("nifti", 0, "volume", artifact_loading.load_volume),
    ("scalp_mesh", 1, "scalp_surface", artifact_loading.load_mesh),
    ("ese_mesh", 3, "ese_mesh", artifact_loading.load_mesh),
    ("fiducials", 4, "fiducials", artifact_loading.load_fiducial_points),
    ("scalp_points", 4, "scalp_points", artifact_loading.load_point_rows),
    ("ese_points", 4, "ese_points", artifact_loading.load_point_rows),
)


@dataclass(frozen=True)
class PreparedProject:
    """A scanned project with validated artifact files.

    ``artifacts`` only holds paths whose files loaded successfully;
    ``failures`` reports the dropped ones as ``"<relative path>: <reason>"``;
    ``recoverable_step`` is the last step that still receives a seeding
    write from the filtered artifacts (``-1`` when nothing is left).
    """

    artifacts: ProjectArtifacts
    failures: tuple[str, ...] = ()
    recoverable_step: int = -1


def prepare_project(project: Path) -> PreparedProject:
    """Scan ``project`` and validate every artifact file it contains."""
    scanned = find_project_artifacts(project)
    dropped: set[str] = set()
    failures: list[str] = []
    for name, _step_n, _key, loader in _FIELDS:
        path = getattr(scanned, name)
        if path is None:
            continue
        try:
            loader(path)
        except Exception as error:  # noqa: BLE001 - any load error is user-facing
            dropped.add(name)
            failures.append(f"{path.relative_to(project)}: {error}")

    artifacts = ProjectArtifacts(
        **{
            name: None if name in dropped else getattr(scanned, name)
            for name in ProjectArtifacts.__dataclass_fields__
        }
    )
    return PreparedProject(
        artifacts=artifacts,
        failures=tuple(failures),
        recoverable_step=_last_seeded_step(artifacts),
    )


def _last_seeded_step(artifacts: ProjectArtifacts) -> int:
    """Mirror the seeding contract: the last step that receives a write."""
    any_points = artifacts.fiducials is not None or artifacts.ese_points is not None
    if any_points or artifacts.scalp_points is not None:
        return 4
    if artifacts.ese_mesh is not None:
        return 3
    if artifacts.scalp_mesh is not None:
        return 2
    if artifacts.nifti is not None:
        return 0
    return -1
