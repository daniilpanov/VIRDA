"""Seeding pipeline step contexts and the global store from project artifacts.

Step contexts store whatever the given artifact set holds: file
:class:`~pathlib.Path` values for a scanned set (:class:`ProjectArtifacts`)
or ready in-memory objects for a loaded set (:class:`LoadedArtifacts`).
Loading paths stays with later commits; loaded objects are runnable
immediately. Keys written per step:

- step 0 (NIfTI import): ``volume``
- step 1 (Base mesh): ``scalp_surface`` (the postprocessed mesh, else
  the raw base mesh); ``scalp_points`` when present
- step 2 (Mesh postprocessing): ``mesh`` (the postprocessed mesh
  artifact, which the pipeline overwrites in place)
- step 3 (ESE mesh): ``ese_mesh``; ``normals`` is never written
  because normals are in-memory only
- step 4 (Localization): ``scalp_points``; ``ese_points``, each when
  present

Fiducials are project-level inputs, not step-derived results: they go
into the manager's global store (``set_global_item("fiducials", ...)``)
instead of the step-4 context. They are stored even when the chain
stops short of step 4, so a project that is not complete yet keeps its
fiducials.

Contexts are created for steps 0 through the last step that received a
write, but only while the chain stays complete: ``next_step`` refuses to
leave an unfinished step, so the seeding stops at the first gap and
raises :class:`StepChainIncompleteError`. The manager is left in the state the
pipeline can continue from: the gap step is current, the steps after it
are not created and their artifacts are not loaded. The global fiducials
are stored before the walk, so they survive a gap.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.steps_manager import StepsManager

if TYPE_CHECKING:
    # The artifact types are only needed for the annotations: the seeding
    # works on the fields they share, loaded or scanned.
    from virda_gui_new.services.project_import import LoadedArtifacts
    from virda_gui_new.services.project_scan import ProjectArtifacts


class StepChainIncompleteError(ValueError):
    """The pipeline stopped at a step whose required artifacts are missing.

    ``step_n`` is the step that could not be finished;
    ``missing_keys`` names the required keys it lacks.
    """

    def __init__(self, step_n: int, missing_keys: tuple[str, ...]) -> None:
        super().__init__(f"Step {step_n} is missing required keys: {', '.join(missing_keys)}")
        self.step_n = step_n
        self.missing_keys = missing_keys

    def explanation(self) -> str:
        """One sentence for the import dialog: why the chain stopped."""
        name = STEPS[self.step_n].name
        keys = ", ".join(self.missing_keys)
        return (
            f"Step {self.step_n} ({name}) has no {keys} result: the later steps were "
            f"not loaded and the pipeline continues from step {self.step_n}."
        )


def seed_steps_from_artifacts(
    manager: StepsManager,
    artifacts: ProjectArtifacts | LoadedArtifacts,
) -> None:
    """Write artifact values into matching :class:`StepsManager` contexts.

    Only present artifacts are written; missing (or dropped) ones leave
    their keys unset. Fiducials are written to the global store; every
    other value goes into the step context of its step. The manager is
    advanced with ``next_step()`` up to the last step that received a
    write, leaving no trailing empty contexts, and one step further to
    the first step without a result.

    Raises :class:`StepChainIncompleteError` when a step in between is not
    finished: the manager then holds the pipeline up to that gap, and
    the artifacts of the later steps stay unloaded.
    """
    writes: dict[int, dict[str, Path | MRIVolume | Mesh | LabeledPoints]] = {}

    if artifacts.nifti is not None:
        writes[0] = {"volume": artifacts.nifti}

    # Precedence: the postprocessed final mesh wins over the raw base
    # mesh. The final mesh feeds steps 1 and 2 (the step-2 mesh is what
    # the pipeline overwrites in place); a lone base mesh only seeds the
    # step-1 scalp surface, from which the scene rules pick it up.
    scalp_mesh = artifacts.scalp_mesh if artifacts.scalp_mesh is not None else artifacts.base_mesh
    if scalp_mesh is not None:
        writes[1] = {"scalp_surface": scalp_mesh}
        if artifacts.scalp_points is not None:
            writes[1]["scalp_points"] = artifacts.scalp_points
        if artifacts.scalp_mesh is not None:
            writes[2] = {"mesh": artifacts.scalp_mesh}

    if artifacts.ese_mesh is not None:
        writes[3] = {"ese_mesh": artifacts.ese_mesh}

    if artifacts.scalp_points is not None or artifacts.ese_points is not None:
        writes[4] = {}
        if artifacts.scalp_points is not None:
            writes[4]["scalp_points"] = artifacts.scalp_points
        if artifacts.ese_points is not None:
            writes[4]["ese_points"] = artifacts.ese_points

    if artifacts.fiducials is not None:
        # Fiducials are a step-4 artifact as well: they mark step 4 as
        # carrying something, even without electrode points.
        writes.setdefault(4, {})

    if artifacts.fiducials is not None:
        # The global store is filled before the walk: the fiducials are a
        # project input, so they survive a gap in the chain.
        manager.set_global_item("fiducials", artifacts.fiducials)

    if not writes:
        return

    last_step = max(writes)
    for step_n in range(last_step + 1):
        if step_n > 0:
            try:
                manager.next_step()
            except ValueError as error:
                # The step we cannot leave is unfinished: the pipeline
                # stops there and the later artifacts stay unloaded.
                gap = step_n - 1
                raise StepChainIncompleteError(gap, manager.missing_required_keys(gap)) from error
        for key, value in writes.get(step_n, {}).items():
            manager.set_step_context_item(step_n, key, value)

    if manager.can_advance():
        # The last seeded step is finished, so the pipeline continues from
        # the first step without a result.
        manager.next_step()
