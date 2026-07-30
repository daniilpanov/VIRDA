"""Anatomical fiducial management: CRUD, validation, persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .types import Fiducial

logger = logging.getLogger(__name__)

REQUIRED_FIDUCIALS = frozenset({"NAS", "LPA", "RPA"})
SURFACE_TOLERANCE_MM = 20.0
BOUNDS_TOLERANCE_MM = 100.0


class FiducialManager:
    """Manages anatomical fiducial landmarks.

    Parameters
    ----------
    head_centroid : np.ndarray
        Approximate centroid of the head (3,).
    surface_vertices : np.ndarray
        Scalp mesh vertices (N,3) used for proximity checks.
    """

    def __init__(
        self,
        head_centroid: np.ndarray,
        surface_vertices: np.ndarray,
    ) -> None:
        self.head_centroid = np.asarray(head_centroid, dtype=np.float64)
        self.surface_vertices = np.asarray(surface_vertices, dtype=np.float64)
        self._fiducials: dict[str, Fiducial] = {}

    def add_fiducial(
        self,
        fiducial_id: str,
        name: str,
        coordinates: np.ndarray,
        definition_method: str = "manual",
        confidence: float | None = None,
        notes: str = "",
    ) -> Fiducial:
        """Add or update a fiducial.

        Parameters
        ----------
        fiducial_id : str
            Unique identifier (e.g. 'NAS', 'LPA').
        name : str
            Human-readable name.
        coordinates : np.ndarray
            3D coordinates in MRI mm.

        Returns
        -------
        Fiducial
            The created fiducial.

        Raises
        ------
        ValueError
            If coordinates are not 3D.
        """
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (3,):
            raise ValueError(f"Coordinates must be shape (3,), got {coords.shape}")

        fiducial = Fiducial(
            fiducial_id=fiducial_id,
            name=name,
            coordinates=coords,
            definition_method=definition_method,
            confidence=confidence,
            notes=notes,
        )
        self._fiducials[fiducial_id] = fiducial
        logger.info("Added fiducial: %s at %s", fiducial_id, coords)
        return fiducial

    def remove_fiducial(self, fiducial_id: str) -> None:
        """Remove a fiducial by ID.

        Raises
        ------
        KeyError
            If fiducial_id does not exist.
        """
        if fiducial_id not in self._fiducials:
            raise KeyError(f"Fiducial '{fiducial_id}' not found")
        del self._fiducials[fiducial_id]
        logger.info("Removed fiducial: %s", fiducial_id)

    def get_fiducial(self, fiducial_id: str) -> Fiducial:
        """Get a fiducial by ID."""
        return self._fiducials[fiducial_id]

    def get_all_fiducials(self) -> dict[str, Fiducial]:
        """Return all fiducials."""
        return dict(self._fiducials)

    def get_coordinates_matrix(
        self,
        ids: list[str] | None = None,
    ) -> np.ndarray:
        """Return fiducial coordinates as a (K,3) matrix.

        Parameters
        ----------
        ids : list[str], optional
            Fiducial IDs to include. If None, returns all.

        Returns
        -------
        np.ndarray
            Coordinates matrix (K,3).
        """
        if ids is None:
            ids = list(self._fiducials.keys())
        return np.array(
            [self._fiducials[fid].coordinates for fid in ids],
            dtype=np.float64,
        )

    def validate(self) -> list[str]:
        """Validate fiducials: check count, bounds, and proximity to surface.

        Returns
        -------
        list[str]
            List of warning/error messages. Empty if all checks pass.
        """
        messages: list[str] = []

        if len(self._fiducials) < 3:
            messages.append(
                f"ERROR: Need at least 3 fiducials (NAS, LPA, RPA), "
                f"have {len(self._fiducials)}"
            )

        for fid in self._fiducials.values():
            dist_from_centroid = np.linalg.norm(
                fid.coordinates - self.head_centroid
            )
            if dist_from_centroid > BOUNDS_TOLERANCE_MM:
                messages.append(
                    f"WARNING: Fiducial {fid.fiducial_id} is "
                    f"{dist_from_centroid:.1f}mm from head centroid "
                    f"(>{BOUNDS_TOLERANCE_MM}mm)"
                )

        if len(self.surface_vertices) > 0:
            for fid in self._fiducials.values():
                distances = np.linalg.norm(
                    self.surface_vertices - fid.coordinates, axis=1
                )
                min_dist = float(distances.min())
                if min_dist > SURFACE_TOLERANCE_MM:
                    messages.append(
                        f"WARNING: Fiducial {fid.fiducial_id} is "
                        f"{min_dist:.1f}mm from nearest surface vertex "
                        f"(>{SURFACE_TOLERANCE_MM}mm)"
                    )

        return messages

    def save(self, path: Path) -> None:
        """Save fiducials to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for fid_id, fid in self._fiducials.items():
            data[fid_id] = {
                "name": fid.name,
                "coordinates": fid.coordinates.tolist(),
                "definition_method": fid.definition_method,
                "confidence": fid.confidence,
                "notes": fid.notes,
            }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Load fiducials from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self._fiducials.clear()
        for fid_id, info in data.items():
            self.add_fiducial(
                fiducial_id=fid_id,
                name=info["name"],
                coordinates=np.array(info["coordinates"], dtype=np.float64),
                definition_method=info.get("definition_method", "manual"),
                confidence=info.get("confidence"),
                notes=info.get("notes", ""),
            )
        logger.info("Loaded %d fiducials from %s", len(self._fiducials), path)
