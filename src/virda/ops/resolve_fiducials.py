"""Step: pick the Stage 1 fiducials from manual, auto or coordsystem sources."""

from virda.models.coordsystem import Coordsystem
from virda.models.fiducial import AutoDetectedFiducials, Fiducials, ManualFiducials


def resolve_fiducials(
    manual: ManualFiducials | None,
    auto: AutoDetectedFiducials | None,
    coordsystem: Coordsystem | None,
) -> Fiducials:
    """Prefer manual fiducials, then auto-detected ones, then coordsystem defaults.

    Raises ``ValueError`` when none of the three sources provides at least one
    fiducial.
    """
    if manual is not None:
        return manual.fiducials

    if auto is not None:
        return auto.fiducials

    if coordsystem is not None:
        coordsystem_fiducials = coordsystem.to_fiducials()
        if coordsystem_fiducials.items:
            return coordsystem_fiducials

    raise ValueError(
        "No fiducials available: provide a manual fiducials file,"
        " enable auto_detect_fiducials, or supply a coordsystem.json config file"
    )
