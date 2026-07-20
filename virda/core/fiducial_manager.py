"""Fiducial manager — mark, store, validate anatomical landmarks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Fiducial:
    """Anatomical fiducial point."""

    fiducial_id: str
    name: str
    coordinates: np.ndarray
    definition_method: str = "manual"
    confidence: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["coordinates"] = self.coordinates.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Fiducial:
        return cls(
            fiducial_id=d["fiducial_id"],
            name=d["name"],
            coordinates=np.array(d["coordinates"], dtype=np.float64),
            definition_method=d.get("definition_method", "manual"),
            confidence=d.get("confidence"),
            notes=d.get("notes", ""),
        )


class FiducialManager:
    """Manage anatomical fiducials.

    Standard fiducials for VIRDA:
    - NAS: Nasion
    - LPA: Left Preauricular Point
    - RPA: Right Preauricular Point
    - INI: Inion (optional)
    """

    STANDARD_FIDUCIALS = {
        "NAS": "Nasion",
        "LPA": "Left Preauricular Point",
        "RPA": "Right Preauricular Point",
        "INI": "Inion",
    }

    def __init__(
        self,
        head_centroid: Optional[np.ndarray] = None,
        surface_vertices: Optional[np.ndarray] = None,
        max_distance_from_surface: float = 30.0,
    ):
        self.fiducials: dict[str, Fiducial] = {}
        self.head_centroid = head_centroid
        self.surface_vertices = surface_vertices
        self.max_distance_from_surface = max_distance_from_surface

    def add_fiducial(
        self,
        fiducial_id: str,
        name: str,
        coordinates: np.ndarray,
        definition_method: str = "manual",
        confidence: Optional[float] = None,
        notes: str = "",
    ) -> Fiducial:
        """Add or update a fiducial point."""
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (3,):
            raise ValueError(f"Coordinates must be shape (3,), got {coords.shape}")

        fid = Fiducial(
            fiducial_id=fiducial_id,
            name=name,
            coordinates=coords,
            definition_method=definition_method,
            confidence=confidence,
            notes=notes,
        )

        self.fiducials[fiducial_id] = fid
        logger.info("Added fiducial %s at (%.1f, %.1f, %.1f)", fiducial_id, *coords)
        return fid

    def remove_fiducial(self, fiducial_id: str) -> None:
        """Remove a fiducial by ID."""
        if fiducial_id in self.fiducials:
            del self.fiducials[fiducial_id]
            logger.info("Removed fiducial %s", fiducial_id)

    def get_fiducial(self, fiducial_id: str) -> Optional[Fiducial]:
        """Get a fiducial by ID."""
        return self.fiducials.get(fiducial_id)

    def get_all_fiducials(self) -> dict[str, Fiducial]:
        """Return all fiducials."""
        return dict(self.fiducials)

    def get_coordinates_matrix(self, fiducial_ids: Optional[list[str]] = None) -> np.ndarray:
        """Get fiducial coordinates as (N, 3) array."""
        if fiducial_ids is None:
            fiducial_ids = list(self.fiducials.keys())

        coords = []
        for fid in fiducial_ids:
            if fid in self.fiducials:
                coords.append(self.fiducials[fid].coordinates)

        if not coords:
            return np.empty((0, 3))

        return np.vstack(coords)

    def validate(self) -> list[str]:
        """Validate all fiducials. Returns list of warning messages."""
        warnings = []

        if len(self.fiducials) < 3:
            warnings.append(
                f"Only {len(self.fiducials)} fiducials defined; at least 3 required"
            )

        if self.surface_vertices is not None:
            for fid in self.fiducials.values():
                dists = np.linalg.norm(
                    self.surface_vertices - fid.coordinates, axis=1
                )
                min_dist = dists.min()
                if min_dist > self.max_distance_from_surface:
                    warnings.append(
                        f"Fiducial {fid.fiducial_id} is {min_dist:.1f} mm from "
                        f"nearest surface point (threshold: {self.max_distance_from_surface} mm)"
                    )

        if self.head_centroid is not None:
            for fid in self.fiducials.values():
                vec = fid.coordinates - self.head_centroid
                if np.linalg.norm(vec) < 10.0:
                    warnings.append(
                        f"Fiducial {fid.fiducial_id} is very close to head centroid"
                    )

        for std_id, std_name in self.STANDARD_FIDUCIALS.items():
            if std_id not in self.fiducials:
                warnings.append(f"Standard fiducial {std_id} ({std_name}) not defined")

        return warnings

    def save(self, path: str | Path) -> None:
        """Save fiducials to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "fiducials": {k: v.to_dict() for k, v in self.fiducials.items()},
            "metadata": {
                "max_distance_from_surface": self.max_distance_from_surface,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d fiducials to %s", len(self.fiducials), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> FiducialManager:
        """Load fiducials from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mgr = cls(**kwargs)

        for fid_dict in data.get("fiducials", {}).values():
            fid = Fiducial.from_dict(fid_dict)
            mgr.fiducials[fid.fiducial_id] = fid

        logger.info("Loaded %d fiducials from %s", len(mgr.fiducials), path)
        return mgr
