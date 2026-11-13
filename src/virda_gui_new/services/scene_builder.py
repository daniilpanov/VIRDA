"""Scene kwargs resolution: manager contexts -> :meth:`SceneViewer3D.set_scene` arguments.

Global-store and step-context values are file :class:`~pathlib.Path`
(loaded here) or ready in-memory sources
(:class:`~virda.models.mri_volume.MRIVolume`,
:class:`~virda.models.mesh.Mesh`,
:class:`~virda_gui_new.dto.viewer3d.LabeledPoints`), passed through as-is.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_loading
from virda_gui_new.services.steps_manager import StepsManager

# Global store rules, checked before the step contexts: the listed keys
# are read from :meth:`StepsManager.get_global_item`.
_GLOBAL_RULES: tuple[tuple[str, str, Callable[[Path], Any]], ...] = (
    ("fiducials", "fiducials", artifact_loading.load_fiducial_points),
)

# (step index, context key) -> (set_scene kwarg, file loader). Earlier
# entries win: the postprocessed step-2 mesh replaces the raw step-1
# scalp surface, and the localization step-4 point files beat the ones
# recorded at their collection step.
_RULES: tuple[tuple[int, str, str, Callable[[Path], Any]], ...] = (
    (0, "volume", "nifti", artifact_loading.load_volume),
    (2, "mesh", "scalp", artifact_loading.load_mesh),
    (1, "scalp_surface", "scalp", artifact_loading.load_mesh),
    (3, "ese_mesh", "ese", artifact_loading.load_mesh),
    (4, "scalp_points", "scalp_points", artifact_loading.load_electrode_points),
    (4, "ese_points", "ese_points", artifact_loading.load_electrode_points),
    (1, "scalp_points", "scalp_points", artifact_loading.load_electrode_points),
)

_READY_SOURCE_TYPES = (MRIVolume, Mesh, LabeledPoints)


def _load(kwarg: str, path: Path, loader: Callable[[Path], Any]) -> Any:
    try:
        return loader(path)
    except Exception as error:
        raise ValueError(f"Failed to load {kwarg} file {path}: {error}") from error


def collect_scene_kwargs(
    manager: StepsManager, failures: list[str] | None = None
) -> dict[str, Any]:
    """Resolve the :meth:`SceneViewer3D.set_scene` kwargs for the current contexts.

    Global-store rules are resolved first, then the step contexts. With
    ``failures`` given, unloadable file paths are skipped and reported
    there as ``"<message>"`` strings instead of raising; an empty dict
    means there is nothing to show.
    """
    kwargs: dict[str, Any] = {}

    for _kwarg_key, key, loader in _GLOBAL_RULES:
        _resolve_value(key, manager.get_global_item(key), key, loader, kwargs, failures)

    contexts = manager.all_contexts()
    for step_n, key, kwarg, loader in _RULES:
        if kwarg in kwargs or step_n >= len(contexts):
            continue
        _resolve_value(
            f"{step_n}.{key}", contexts[step_n].get(key), kwarg, loader, kwargs, failures
        )

    return kwargs


def _resolve_value(
    name: str,
    value: Any,
    kwarg: str,
    loader: Callable[[Path], Any],
    kwargs: dict[str, Any],
    failures: list[str] | None,
) -> None:
    """Resolve one context/store value into ``kwargs``.

    Paths are loaded through ``loader`` (failures reported or raised),
    ready sources pass through unchanged, other values are ignored.
    """
    if value is None or kwarg in kwargs:
        return
    if isinstance(value, Path):
        try:
            kwargs[kwarg] = _load(kwarg, value, loader)
        except ValueError as error:
            if failures is None:
                raise
            failures.append(str(error))
    elif isinstance(value, _READY_SOURCE_TYPES):
        kwargs[kwarg] = value
