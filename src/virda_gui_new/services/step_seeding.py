"""Seeding pipeline step contexts from project artifacts.

Contexts store whatever the given artifact set holds: file
:class:`~pathlib.Path` values for a scanned set
(:class:`ProjectArtifacts`) or ready in-memory objects for a loaded
set (:class:`LoadedArtifacts`). Loading paths stays with later
commits; loaded objects are runnable immediately. Keys written per
step:

- step 0 (NIfTI import): ``volume``
- step 1 (Base mesh): ``scalp_surface``; ``scalp_points`` when present
- step 2 (Mesh postprocessing): ``mesh`` (the same mesh artifact,
  which the pipeline overwrites in place)
- step 3 (ESE mesh): ``ese_mesh``; ``normals`` is never written
  because normals are in-memory only
- step 4 (Localization): ``fiducials``; ``scalp_points``; ``ese_points``,
  each when present

Contexts are created for steps 0 through the last step that received a
write; the working context of the following step is left uncreated.
"""

from pathlib import Path

from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.project_import import LoadedArtifacts
from virda_gui_new.services.project_scan import ProjectArtifacts
from virda_gui_new.services.steps_manager import StepsManager


def seed_steps_from_artifacts(
    manager: StepsManager,
    artifacts: ProjectArtifacts | LoadedArtifacts,
) -> None:
    """Write artifact values into matching :class:`StepsManager` contexts.

    Only present artifacts are written; missing (or dropped) ones leave
    their keys unset. The manager is advanced with ``next_step()`` up to
    the last step that received a write, leaving no trailing empty
    contexts.
    """
    writes: dict[int, dict[str, Path | MRIVolume | Mesh | LabeledPoints]] = {}

    if artifacts.nifti is not None:
        writes[0] = {"volume": artifacts.nifti}

    if artifacts.scalp_mesh is not None:
        writes[1] = {"scalp_surface": artifacts.scalp_mesh}
        if artifacts.scalp_points is not None:
            writes[1]["scalp_points"] = artifacts.scalp_points

        writes[2] = {"mesh": artifacts.scalp_mesh}

    if artifacts.ese_mesh is not None:
        writes[3] = {"ese_mesh": artifacts.ese_mesh}

    if (
        artifacts.fiducials is not None
        or artifacts.scalp_points is not None
        or (artifacts.ese_points is not None)
    ):
        writes[4] = {}
        if artifacts.fiducials is not None:
            writes[4]["fiducials"] = artifacts.fiducials
        if artifacts.scalp_points is not None:
            writes[4]["scalp_points"] = artifacts.scalp_points
        if artifacts.ese_points is not None:
            writes[4]["ese_points"] = artifacts.ese_points

    if not writes:
        return

    last_step = max(writes)
    for step_n in range(last_step + 1):
        if step_n > 0:
            manager.next_step()
        for key, value in writes.get(step_n, {}).items():
            manager.set_step_context_item(step_n, key, value)
