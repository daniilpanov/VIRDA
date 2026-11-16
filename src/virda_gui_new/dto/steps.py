"""Ordered definitions of the GUI pipeline steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """One pipeline step: a display name and the context keys marking it done."""

    index: int
    name: str
    required_keys: tuple[str, ...] = ()


#: The keys of the manager's global store: the project-level inputs
#: that survive a step invalidation.
FIDUCIALS_KEY = "fiducials"
MEASUREMENTS_KEY = "measurements"

#: The context key of the localization step that holds the results.
ELECTRODES_KEY = "electrodes"

# Steps of the GUI pipeline, in order. required_keys are the context keys
# that mark the step done.
NIFTI = Step(0, "NIfTI import", ("volume",))
BASE_MESH = Step(1, "Base mesh", ("scalp_surface",))
MESH_POSTPROCESSING = Step(2, "Mesh postprocessing", ("mesh",))
ESE_MESH = Step(3, "ESE mesh", ("ese_mesh", "normals"))
LOCALIZATION = Step(4, "Localization", ())  # live step: no required keys

STEPS: tuple[Step, ...] = (NIFTI, BASE_MESH, MESH_POSTPROCESSING, ESE_MESH, LOCALIZATION)

#: One input of a step: the context keys that satisfy it. The first
#: entry is the step the key belongs to, the second one the key.
RunInput = tuple[tuple[int, str], ...]

#: The inputs every step needs to run, as alternatives: a requirement
#: is met when one of its context keys holds a value. The steps read
#: their input from the step that produced it.
RUN_INPUTS: dict[int, tuple[RunInput, ...]] = {
    BASE_MESH.index: (((NIFTI.index, "volume"),),),
    MESH_POSTPROCESSING.index: (((BASE_MESH.index, "scalp_surface"),),),
    ESE_MESH.index: (((MESH_POSTPROCESSING.index, "mesh"), (BASE_MESH.index, "scalp_surface")),),
    LOCALIZATION.index: (
        ((ESE_MESH.index, "ese_mesh"),),
        ((ESE_MESH.index, "normals"),),
        ((MESH_POSTPROCESSING.index, "mesh"), (BASE_MESH.index, "scalp_surface")),
    ),
}
