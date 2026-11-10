"""Seeding pipeline step contexts from scanned project artifacts.

Seeding contract: contexts store file :class:`~pathlib.Path` values; loading
them stays with later commits. Keys written per step:

- step 0 (NIfTI import): ``volume`` -> Path
- step 1 (Base mesh): ``scalp_surface`` -> Path; ``scalp_points`` -> Path
  when present
- step 2 (Mesh postprocessing): ``mesh`` -> Path (the same
  ``mesh/final_mesh.ply``, which the pipeline overwrites in place)
- step 3 (ESE mesh): ``ese_mesh`` -> Path; ``normals`` is never written
  because normals are in-memory only, so the step stays not-done
- step 4 (Localization): ``fiducials`` -> Path; ``scalp_points`` -> Path;
  ``ese_points`` -> Path, each when present

Contexts are created for steps 0 through the last step that received a
write; the working context of the following step is left uncreated.
"""

from pathlib import Path

from virda_gui_new.services.project_scan import ProjectArtifacts
from virda_gui_new.services.steps_manager import StepsManager


def seed_steps_from_artifacts(manager: StepsManager, artifacts: ProjectArtifacts) -> None:
    """Write ready artifact paths into matching :class:`StepsManager` contexts.

    Only ready artifacts are written; missing files leave their keys unset.
    The manager is advanced with ``next_step()`` up to the last step that
    received a write, leaving no trailing empty contexts.
    """
    writes: dict[int, dict[str, Path]] = {}

    if artifacts.nifti is not None:
        writes[0] = {"volume": artifacts.nifti}

    if artifacts.scalp_mesh is not None:
        writes[1] = {"scalp_surface": artifacts.scalp_mesh}
        if artifacts.scalp_points is not None:
            writes[1]["scalp_points"] = artifacts.scalp_points

    if artifacts.scalp_mesh is not None:
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
