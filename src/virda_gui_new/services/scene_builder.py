"""Scene kwargs resolution: step contexts -> :meth:`SceneViewer3D.set_scene` arguments.

Context values are file :class:`~pathlib.Path` (loaded here) or ready
in-memory sources (:class:`~virda.models.mri_volume.MRIVolume`,
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

# (step index, context key) -> (set_scene kwarg, file loader). Earlier
# entries win: the postprocessed step-2 mesh replaces the raw step-1
# scalp surface, and the localization step-4 point files beat the ones
# recorded at their collection step.
_RULES: tuple[tuple[int, str, str, Callable[[Path], Any]], ...] = (
    (0, "volume", "nifti", artifact_loading.load_volume),
    (1, "scalp_surface", "scalp", artifact_loading.load_mesh),
    (1, "scalp_points", "scalp_points", artifact_loading.load_point_rows),
    (2, "mesh", "scalp", artifact_loading.load_mesh),
    (3, "ese_mesh", "ese", artifact_loading.load_mesh),
    (4, "fiducials", "fiducials", artifact_loading.load_fiducial_points),
    (4, "scalp_points", "scalp_points", artifact_loading.load_point_rows),
    (4, "ese_points", "ese_points", artifact_loading.load_point_rows),
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

    With ``failures`` given, unloadable context paths are skipped and
    reported there as ``"<message>"`` strings instead of raising; an empty
    dict means there is nothing to show.
    """
    contexts = manager.all_contexts()
    kwargs: dict[str, Any] = {}

    for step_n, key, kwarg, loader in _RULES:
        if kwarg in kwargs or step_n >= len(contexts):
            continue
        value = contexts[step_n].get(key)
        if isinstance(value, Path):
            try:
                kwargs[kwarg] = _load(kwarg, value, loader)
            except ValueError as error:
                if failures is None:
                    raise
                failures.append(str(error))
        elif isinstance(value, _READY_SOURCE_TYPES):
            kwargs[kwarg] = value

    return kwargs
