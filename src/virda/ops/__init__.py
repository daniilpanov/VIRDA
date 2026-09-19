"""Pure orchestration layer: options, surface bundles and atoms.

This package is the public functional API of VIRDA. ``import virda.ops``
exposes the option dataclasses, the :class:`ScalpSurface` bundle and the
pure atoms operating directly on the domain models.
"""

from virda.ops.atoms import (
    clean,
    decimate,
    generate_ese,
    generate_scalp_surface,
    localize,
    smooth,
)
from virda.ops.options import (
    CleanOptions,
    DecimateOptions,
    EseOptions,
    LocalizeOptions,
    SealingOptions,
    SmoothOptions,
)
from virda.ops.scalp_surface import ScalpSurface

__all__ = [
    "CleanOptions",
    "DecimateOptions",
    "EseOptions",
    "LocalizeOptions",
    "ScalpSurface",
    "SealingOptions",
    "SmoothOptions",
    "clean",
    "decimate",
    "generate_ese",
    "generate_scalp_surface",
    "localize",
    "smooth",
]