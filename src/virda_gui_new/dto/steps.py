"""Ordered definitions of the GUI pipeline steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """One pipeline step: a display name and the context keys marking it done."""

    index: int
    name: str
    required_keys: tuple[str, ...] = ()


# Steps of the GUI pipeline, in order. required_keys are the context keys
# that mark the step done.
NIFTI = Step(0, "NIfTI import", ("volume",))
BASE_MESH = Step(1, "Base mesh", ("scalp_surface",))
MESH_POSTPROCESSING = Step(2, "Mesh postprocessing", ("mesh",))
ESE_MESH = Step(3, "ESE mesh", ("ese_mesh",))
LOCALIZATION = Step(4, "Localization", ())  # live step: no required keys

STEPS: tuple[Step, ...] = (NIFTI, BASE_MESH, MESH_POSTPROCESSING, ESE_MESH, LOCALIZATION)
