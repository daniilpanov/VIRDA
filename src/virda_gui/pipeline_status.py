"""Pipeline step model for the guided workflow bar.

Qt-free: given what is saved in the project folder and what lives only in
memory, describe the five user-facing steps (brain scan, base mesh, skin
surface, sensor surface, points and measures) with plain-language titles,
next-action hints, locked flags and one of four states.  Unit-testable
headless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StepState = Literal["done", "unsaved", "active", "waiting"]


@dataclass(frozen=True)
class PipelineStep:
    """One user-facing pipeline step."""

    key: str
    title: str
    detail: str
    state: StepState
    enabled: bool


def pipeline_steps(
    *,
    nifti_saved: bool,
    base_saved: bool,
    mesh_saved: bool,
    ese_saved: bool,
    base_in_memory: bool,
    mesh_in_memory: bool,
    ese_in_memory: bool,
    fiducials_filled: int,
    measurement_rows: int,
) -> list[PipelineStep]:
    """Build the five pipeline steps from saved and in-memory progress.

    Saved files win over memory: a step is ``done`` when its artifact is on
    disk, ``unsaved`` when it exists only in memory, ``active`` when it is
    the next actionable step, and ``waiting`` otherwise.  A step is
    ``enabled`` only when its prerequisites are ready, enforcing the
    1 -> 2 -> 3 -> 4 -> 5 order: scan, base, mesh, ESE, points.  The points
    step additionally needs the sensor surface, so dropping the ESE mesh
    un-dones it while the tables themselves are kept.  Progress is a strict
    prefix without holes: a step is actionable only when the previous
    artifact exists, so orphaned files behind a gap (e.g. a final mesh
    without a base) stay waiting and locked instead of showing a stale [ok].
    """
    base_present = base_saved or base_in_memory
    mesh_present = mesh_saved or mesh_in_memory
    ese_present = ese_saved or ese_in_memory

    if base_saved:
        base_state: StepState = "done"
        base_detail = "Base surface saved."
    elif base_in_memory:
        base_state = "unsaved"
        base_detail = "Save the base surface to continue."
    else:
        base_state = "active" if nifti_saved else "waiting"
        base_detail = (
            "Generate the base surface from the scan."
            if nifti_saved
            else "Import a brain scan first."
        )
    if not base_present:
        mesh_state = "waiting"
        mesh_detail = "Generate the base surface first."
    elif mesh_saved:
        mesh_state = "done"
        mesh_detail = "Skin surface saved."
    elif mesh_in_memory:
        mesh_state = "unsaved"
        mesh_detail = "Save the skin surface to continue."
    else:
        mesh_state = "active"
        mesh_detail = "Postprocess the base surface."
    mesh_eff = base_present and mesh_present

    if not mesh_eff:
        ese_state = "waiting"
        ese_detail = "Needs the skin surface first."
    elif ese_saved:
        ese_state = "done"
        ese_detail = "Sensor surface saved."
    elif ese_in_memory:
        ese_state = "unsaved"
        ese_detail = "Save the sensor surface to continue."
    else:
        ese_state = "active"
        ese_detail = "Set the offset and generate the sensor surface."
    ese_eff = mesh_eff and ese_present

    tables_ready = fiducials_filled >= 3 and measurement_rows >= 1
    if not ese_eff:
        points_state = "waiting"
        points_detail = "Needs the sensor surface first."
    elif tables_ready:
        points_state = "done"
        points_detail = "Points and measures ready."
    else:
        points_state = "active"
        missing = []
        if fiducials_filled < 3:
            missing.append("fill NAS, LPA and RPA")
        if measurement_rows < 1:
            missing.append("add one measurement row")
        points_detail = "To continue: " + " and ".join(missing) + "."

    return [
        PipelineStep(
            key="scan",
            title="1. Brain scan",
            detail=("Scan imported." if nifti_saved else "Import a brain scan (NIfTI)."),
            state="done" if nifti_saved else "active",
            enabled=True,
        ),
        PipelineStep(
            key="base",
            title="2. Base mesh",
            detail=base_detail,
            state=base_state,
            enabled=nifti_saved,
        ),
        PipelineStep(
            key="mesh",
            title="3. Skin surface (mesh)",
            detail=mesh_detail,
            state=mesh_state,
            enabled=base_present,
        ),
        PipelineStep(
            key="ese",
            title="4. Sensor surface (ESE)",
            detail=ese_detail,
            state=ese_state,
            enabled=mesh_eff,
        ),
        PipelineStep(
            key="points",
            title="5. Points and measures",
            detail=points_detail,
            state=points_state,
            enabled=ese_eff,
        ),
    ]
