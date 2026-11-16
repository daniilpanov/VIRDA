"""Scene sources resolution: manager contexts -> :class:`SceneSources`.

Step-context values are file :class:`~pathlib.Path` (loaded here) or
ready in-memory sources
(:class:`~virda.models.mri_volume.MRIVolume`,
:class:`~virda.models.mesh.Mesh`,
:class:`~virda_gui_new.dto.viewer3d.LabeledPoints`), passed through as-is.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.viewer3d import LabeledPoints, SceneSources
from virda_gui_new.services import artifact_loading
from virda_gui_new.services.steps_manager import StepsManager

# (step index, context key) -> (source name, file loader). Earlier
# entries win: the postprocessed step-2 mesh replaces the raw step-1
# scalp surface. The electrode points belong to the localization step:
# no other step holds them.
_RULES: tuple[tuple[int, str, str, Callable[[Path], Any]], ...] = (
    (0, "volume", "nifti", artifact_loading.load_volume),
    (2, "mesh", "scalp", artifact_loading.load_mesh),
    (1, "scalp_surface", "scalp", artifact_loading.load_mesh),
    (3, "ese_mesh", "ese", artifact_loading.load_mesh),
    (4, "scalp_points", "scalp_points", artifact_loading.load_point_rows),
    (4, "ese_points", "ese_points", artifact_loading.load_point_rows),
)

_READY_SOURCE_TYPES = (MRIVolume, Mesh, LabeledPoints)


def _load(name: str, path: Path, loader: Callable[[Path], Any]) -> Any:
    try:
        return loader(path)
    except Exception as error:
        raise ValueError(f"Failed to load {name} file {path}: {error}") from error


def collect_scene_sources(manager: StepsManager, failures: list[str] | None = None) -> SceneSources:
    """Resolve the scene sources of the current contexts.

    With ``failures`` given, unloadable file paths are skipped and
    reported there as ``"<message>"`` strings instead of raising; an
    empty scene means there is nothing to show.
    """
    sources: dict[str, Any] = {}
    attempted: dict[str, set[Path]] = {}

    contexts = manager.all_contexts()
    for step_n, key, name, loader in _RULES:
        if name in sources or step_n >= len(contexts):
            continue
        value = contexts[step_n].get(key)
        if isinstance(value, Path):
            # Several steps can share one file (the scalp points of
            # step 1 and step 4): it is loaded once, so a broken file
            # is reported once too.
            seen = attempted.setdefault(name, set())
            if value in seen:
                continue
            seen.add(value)
        _resolve_value(f"{step_n}.{key}", value, name, loader, sources, failures)

    return SceneSources(
        nifti=sources.get("nifti"),
        scalp=sources.get("scalp"),
        scalp_points=sources.get("scalp_points"),
        ese=sources.get("ese"),
        ese_points=sources.get("ese_points"),
    )


def _resolve_value(
    label: str,
    value: Any,
    name: str,
    loader: Callable[[Path], Any],
    sources: dict[str, Any],
    failures: list[str] | None,
) -> None:
    """Resolve one context/store value into ``sources``.

    Paths are loaded through ``loader`` (failures reported or raised),
    ready sources pass through unchanged, other values are ignored.
    """
    if value is None or name in sources:
        return
    if isinstance(value, Path):
        try:
            sources[name] = _load(name, value, loader)
        except ValueError as error:
            if failures is None:
                raise
            failures.append(str(error))
    elif isinstance(value, _READY_SOURCE_TYPES):
        sources[name] = value
