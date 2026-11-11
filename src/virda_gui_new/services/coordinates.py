"""Coordinate-frame conversions between world, head and fiducials frames.

World is the canonical frame of the whole GUI: scanner surface-RAS
millimetres -- every stored value (the global fiducials store, the saved
artifact files, the localized electrode coordinates) lives in it. These
conversions exist only to change the *displayed* numbers; nothing
canonical is ever converted.

Frames:

- ``world``: scanner RAS millimetres (the identity frame).
- ``head``: world translated so the origin sits at the head centre of
  mass, the centroid of the active scalp mesh vertices (the step-2
  ``mesh`` context, falling back to the step-1 ``scalp_surface``, the
  same precedence the scene rules use). Axes stay world-aligned.
- ``fiducials``: an anatomical frame built from the three fiducials
  NAS, LPA and RPA (world coordinates): the origin is NAS, the right
  axis is ``normalize(RPA - LPA)``, the anterior axis is
  ``normalize(NAS - midpoint(LPA, RPA))`` Gram-Schmidt-orthogonalized
  against the right axis, and the up axis is ``right x anterior``
  (already unit because its inputs are orthonormal). With ``R`` the
  matrix whose columns are ``(right, anterior, up)``, a world point
  ``p`` maps to ``R.T @ (p - origin)``. The frame is available only
  when all three fiducials have coordinates.

Fiducial labels are matched case-insensitively onto the three canonical
slots: ``nasion``/``nas``/``nz`` map to NAS, ``lpa`` to LPA and ``rpa``
to RPA; any other label is ignored. The project artifacts name the NAS
point either way (``fiducial_id="NAS"`` with ``name="NASION"``), so the
alias table accepts both spellings.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import BASE_MESH, MESH_POSTPROCESSING
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.steps_manager import StepsManager

#: The three canonical fiducial slots, in fixed row order.
SLOT_IDS = ("NAS", "LPA", "RPA")

#: Lowercased label -> canonical slot. Checked case-insensitively.
_ALIASES: dict[str, str] = {
    "nas": "NAS",
    "nasion": "NAS",
    "nz": "NAS",
    "lpa": "LPA",
    "rpa": "RPA",
}

_FIDUCIALS_STORE_KEY = "fiducials"


class CoordinateFrame(Enum):
    """Display frame of a coordinate table; ``world`` is the canonical one."""

    WORLD = "world"
    HEAD = "head"
    FIDUCIALS = "fiducials"


def match_fiducial_name(name: str) -> str | None:
    """Return the canonical slot of a fiducial label, or ``None``."""
    return _ALIASES.get(name.strip().lower())


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalize ``vector``; zero vectors raise ``ValueError``."""
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise ValueError("cannot normalize a zero-length direction")
    return np.asarray(vector, dtype=np.float64) / length


def fiducials_frame(
    nas: np.ndarray, lpa: np.ndarray, rpa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fiducials frame: ``(origin, rotation)`` in world terms.

    ``origin`` is NAS; ``rotation`` columns are the unit right, anterior
    and up axes expressed in world coordinates (see the module docstring
    for the exact convention). Collinear fiducials raise ``ValueError``.
    """
    origin = np.asarray(nas, dtype=np.float64)
    right = _unit(np.asarray(rpa, dtype=np.float64) - np.asarray(lpa, dtype=np.float64))
    anterior = _unit(
        origin - (np.asarray(lpa, dtype=np.float64) + np.asarray(rpa, dtype=np.float64)) / 2.0
    )
    anterior = _unit(anterior - np.dot(anterior, right) * right)  # Gram-Schmidt
    up = np.cross(right, anterior)
    return origin, np.column_stack((right, anterior, up))


@dataclass(frozen=True)
class FrameContext:
    """World-frame anchors the frame conversions need.

    ``head_origin`` is the head centre of mass in world coordinates;
    ``fiducials`` maps canonical slots (``NAS``/``LPA``/``RPA``) to their
    world coordinates. A missing anchor makes the corresponding frame
    conversion unavailable (``ValueError`` on use).
    """

    head_origin: np.ndarray | None = None
    fiducials: Mapping[str, np.ndarray] | None = None


def _as_points(points: np.ndarray) -> tuple[np.ndarray, bool]:
    """Validate point input: ``(N, 3)`` or a single ``(3,)`` point.

    Returns the points as ``(N, 3)`` plus whether the input was a single
    point (the result is squeezed back to ``(3,)``).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1 and pts.shape == (3,):
        return pts.reshape(1, 3), True
    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts, False
    raise ValueError(f"expected an (N, 3) point array, got shape {pts.shape}")


def _fiducials_rotation(ctx: FrameContext) -> tuple[np.ndarray, np.ndarray]:
    """Origin and rotation of the fiducials frame; incomplete input raises."""
    fiducials = ctx.fiducials
    if fiducials is None or not all(slot in fiducials for slot in SLOT_IDS):
        raise ValueError("fiducials frame needs NAS, LPA and RPA coordinates")
    missing = [slot for slot in SLOT_IDS if np.shape(fiducials[slot]) != (3,)]
    if missing:
        raise ValueError(f"fiducials {missing} need (3,) coordinates")
    return fiducials_frame(fiducials["NAS"], fiducials["LPA"], fiducials["RPA"])


def from_world(points: np.ndarray, frame: CoordinateFrame, ctx: FrameContext) -> np.ndarray:
    """Convert world points into ``frame`` display coordinates.

    Accepts ``(N, 3)`` arrays or a single ``(3,)`` point (returned as
    ``(3,)``).
    """
    pts, single = _as_points(points)
    if frame is CoordinateFrame.WORLD:
        converted: np.ndarray = pts.copy()
    elif frame is CoordinateFrame.HEAD:
        if ctx.head_origin is None:
            raise ValueError("head frame needs the scalp mesh centroid")
        converted = np.asarray(pts - ctx.head_origin, dtype=np.float64)
    else:
        origin, rotation = _fiducials_rotation(ctx)
        converted = np.asarray((pts - origin) @ rotation, dtype=np.float64)
    return converted[0] if single else converted


def to_world(points: np.ndarray, frame: CoordinateFrame, ctx: FrameContext) -> np.ndarray:
    """Convert ``frame`` display coordinates back into world points.

    Accepts ``(N, 3)`` arrays or a single ``(3,)`` point (returned as
    ``(3,)``).
    """
    pts, single = _as_points(points)
    if frame is CoordinateFrame.WORLD:
        converted: np.ndarray = pts.copy()
    elif frame is CoordinateFrame.HEAD:
        if ctx.head_origin is None:
            raise ValueError("head frame needs the scalp mesh centroid")
        converted = np.asarray(pts + ctx.head_origin, dtype=np.float64)
    else:
        origin, rotation = _fiducials_rotation(ctx)
        converted = np.asarray(origin + pts @ rotation.T, dtype=np.float64)
    return converted[0] if single else converted


class FrameConverter(Protocol):
    """What a coordinate table needs to switch display frames."""

    def from_world(self, points: np.ndarray, frame: CoordinateFrame) -> np.ndarray:
        """World points -> ``frame`` display coordinates."""
        ...

    def to_world(self, points: np.ndarray, frame: CoordinateFrame) -> np.ndarray:
        """``frame`` display coordinates -> world points."""
        ...

    def is_available(self, frame: CoordinateFrame) -> bool:
        """Whether ``frame`` conversions can run with the current inputs."""
        ...


class CoordinateService:
    """Frame conversions over the live :class:`StepsManager` state.

    The head origin is computed once per mesh value and cached: the
    active mesh object is kept by identity, and a different object (the
    only way the context value changes) recomputes the centroid. The
    fiducials anchor is rebuilt on every use -- it is O(1) work and the
    store may change between calls, so caching it would only add
    invalidation bookkeeping. Both remain deterministic for unchanged
    inputs.
    """

    def __init__(self, manager: StepsManager) -> None:
        self._manager = manager
        self._mesh: Mesh | None = None
        self._head_origin: np.ndarray | None = None

    def context(self) -> FrameContext:
        """Current anchors for the pure conversion functions."""
        return FrameContext(head_origin=self.head_origin(), fiducials=self.fiducial_coordinates())

    def head_origin(self) -> np.ndarray | None:
        """Head centre of mass (world), or ``None`` without a scalp mesh."""
        mesh = self._active_mesh()
        if mesh is None:
            return None
        if mesh is not self._mesh:
            self._mesh = mesh
            self._head_origin = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
        return self._head_origin

    def fiducial_coordinates(self) -> dict[str, np.ndarray] | None:
        """Canonical-slot world coordinates from the global store.

        Labels are matched onto the fixed slots case-insensitively;
        unmatched labels are ignored, and ``None`` is returned when the
        store is empty or holds no known fiducial.
        """
        points = self._manager.get_global_item(_FIDUCIALS_STORE_KEY)
        if not isinstance(points, LabeledPoints) or points.labels is None:
            return None
        coords: dict[str, np.ndarray] = {}
        for label, xyz in zip(points.labels, points.points, strict=True):
            slot = match_fiducial_name(label)
            if slot is not None and slot not in coords:
                coords[slot] = np.asarray(xyz, dtype=np.float64)
        return coords or None

    def is_available(self, frame: CoordinateFrame) -> bool:
        """Whether ``frame`` conversions can run with the current inputs."""
        if frame is CoordinateFrame.WORLD:
            return True
        if frame is CoordinateFrame.HEAD:
            return self.head_origin() is not None
        coords = self.fiducial_coordinates()
        return coords is not None and all(slot in coords for slot in SLOT_IDS)

    def from_world(self, points: np.ndarray, frame: CoordinateFrame) -> np.ndarray:
        """World points -> ``frame`` display coordinates."""
        return from_world(points, frame, self.context())

    def to_world(self, points: np.ndarray, frame: CoordinateFrame) -> np.ndarray:
        """``frame`` display coordinates -> world points."""
        return to_world(points, frame, self.context())

    def _active_mesh(self) -> Mesh | None:
        """The step-2 mesh, else the step-1 scalp surface (scene precedence)."""
        rules = (
            (MESH_POSTPROCESSING.index, "mesh"),
            (BASE_MESH.index, "scalp_surface"),
        )
        for step_n, key in rules:
            try:
                value = self._manager.get_step_context_item(step_n, key)
            except ValueError:
                continue
            if isinstance(value, Mesh):
                return value
        return None
