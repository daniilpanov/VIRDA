"""Pipeline step model for the guided workflow bar.

Qt-free: given what is saved in the project folder and what lives only in
memory, describe the five user-facing steps (brain scan, skin surface,
sensor surface, points and measures, located sensors) with plain-language
titles, next-action hints and one of four states.  Unit-testable headless.
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


def pipeline_steps(
    *,
    nifti_saved: bool,
    mesh_saved: bool,
    ese_saved: bool,
    mesh_in_memory: bool,
    ese_in_memory: bool,
    fiducials_filled: int,
    measurement_rows: int,
    localized_summary: str,
) -> list[PipelineStep]:
    """Build the five pipeline steps from saved and in-memory progress.

    Saved files win over memory: a step is ``done`` when its artifact is on
    disk, ``unsaved`` when it exists only in memory, ``active`` when it is
    the next actionable step, and ``waiting`` otherwise.
    """
    if mesh_saved:
        mesh_state: StepState = "done"
        mesh_detail = "Skin surface saved."
    elif mesh_in_memory:
        mesh_state = "unsaved"
        mesh_detail = "Save the skin surface to continue."
    else:
        mesh_state = "active" if nifti_saved else "waiting"
        mesh_detail = (
            "Generate the skin surface from the scan."
            if nifti_saved
            else "Import a brain scan first."
        )

    if ese_saved:
        ese_state: StepState = "done"
        ese_detail = "Sensor surface saved."
    elif ese_in_memory:
        ese_state = "unsaved"
        ese_detail = "Save the sensor surface to continue."
    else:
        ese_state = "active" if mesh_saved or mesh_in_memory else "waiting"
        ese_detail = (
            "Set the offset and generate the sensor surface."
            if mesh_saved or mesh_in_memory
            else "Needs the skin surface first."
        )

    points_done = fiducials_filled >= 3 and measurement_rows >= 1
    if points_done:
        points_state: StepState = "done"
        points_detail = "Points and measures ready."
    else:
        points_state = "active" if ese_saved or ese_in_memory else "waiting"
        missing = []
        if fiducials_filled < 3:
            missing.append("fill NAS, LPA and RPA")
        if measurement_rows < 1:
            missing.append("add one measurement row")
        points_detail = "To continue: " + " and ".join(missing) + "."

    electrodes_done = localized_summary not in ("", "none")
    if electrodes_done:
        electrodes_state: StepState = "done"
        electrodes_detail = f"Located {localized_summary} sensors."
    else:
        electrodes_state = "active" if points_done else "waiting"
        electrodes_detail = (
            "Run localization." if points_done else "Needs points and measures first."
        )

    return [
        PipelineStep(
            key="scan",
            title="1. Brain scan",
            detail=("Scan imported." if nifti_saved else "Import a brain scan (NIfTI)."),
            state="done" if nifti_saved else "active",
        ),
        PipelineStep(key="mesh", title="2. Skin surface", detail=mesh_detail, state=mesh_state),
        PipelineStep(
            key="ese",
            title="3. Sensor surface",
            detail=ese_detail,
            state=ese_state,
        ),
        PipelineStep(
            key="points",
            title="4. Points and measures",
            detail=points_detail,
            state=points_state,
        ),
        PipelineStep(
            key="electrodes",
            title="5. Sensors found",
            detail=electrodes_detail,
            state=electrodes_state,
        ),
    ]
