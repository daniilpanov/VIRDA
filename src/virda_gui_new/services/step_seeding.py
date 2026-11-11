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
  present; ``measurements`` when the loaded artifact set carries the
  measurements JSON as an :class:`Electrodes` model

Fiducials are project-level inputs, not step-derived results: they go
into the manager's global store (``set_global_item("fiducials", ...)``)
instead of the step-4 context. They still count as a step-4 artifact for
navigation: a project with only a fiducials file gets a step-4 context
(created, possibly empty) and the manager is advanced to step 4, keeping
the steps-menu state of earlier versions.

Contexts are created for steps 0 through the last step that received a
write; the working context of the following step is left uncreated.
"""

from pathlib import Path

from virda.models.electrode import Electrodes
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
    their keys unset. Fiducials are written to the global store; every
    other value goes into the step context of its step. The manager is
    advanced with ``next_step()`` up to the last step that received a
    write, leaving no trailing empty contexts.
    """
    writes: dict[int, dict[str, Path | MRIVolume | Mesh | LabeledPoints | Electrodes]] = {}

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

    if artifacts.measurements is not None:
        # The measurements JSON is seeded only as a ready Electrodes
        # model (loaded projects): the export path works on in-memory
        # models, and a scanned path stays a scan-only artifact. Like
        # the fiducials, the measurements alone keep step 4 in the
        # navigation (the context is created, possibly empty).
        measurements = artifacts.measurements
        if isinstance(measurements, Electrodes):
            writes.setdefault(4, {})["measurements"] = measurements

    if artifacts.fiducials is not None:
        # Fiducials keep step 4 in the navigation even without electrode
        # points: the context is created (empty) and the menu shows it.
        writes.setdefault(4, {})

    if not writes:
        return

    last_step = max(writes)
    for step_n in range(last_step + 1):
        if step_n > 0:
            manager.next_step()
        for key, value in writes.get(step_n, {}).items():
            manager.set_step_context_item(step_n, key, value)

    if artifacts.fiducials is not None:
        manager.set_global_item("fiducials", artifacts.fiducials)
