"""Shared data structures for the VIRDA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MRIData:
    """Loaded MRI volume with spatial metadata."""

    volume: np.ndarray
    affine: np.ndarray
    voxel_size: np.ndarray
    header_info: dict = field(default_factory=dict)


@dataclass
class SegmentationResult:
    """Binary head segmentation mask."""

    mask: np.ndarray
    affine: np.ndarray
    method: str = "threshold"


@dataclass
class MeshData:
    """Triangular surface mesh."""

    vertices: np.ndarray
    faces: np.ndarray
    adjacency: list[list[int]] = field(default_factory=list)
    coordinate_system: str = "MRI_world_mm"
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)


@dataclass
class Fiducial:
    """Anatomical fiducial landmark."""

    fiducial_id: str
    name: str
    coordinates: np.ndarray
    definition_method: str = "manual"
    confidence: float | None = None
    notes: str = ""


@dataclass
class ESEConfigData:
    """ESE configuration parameters."""

    offset_mm: float = 5.0
    reference_point: str = "center_of_external_surface"
    description: str = ""


@dataclass
class NormalResult:
    """Result of PCA-based surface normal estimation."""

    normals: np.ndarray
    quality: np.ndarray
    eigenvalues: np.ndarray


@dataclass
class ESEResult:
    """Result of ESE surface generation."""

    scalp_vertices: np.ndarray
    ese_vertices: np.ndarray
    normals: np.ndarray
    quality: np.ndarray
    head_centroid: np.ndarray
    num_points: int


@dataclass
class ElectrodeLocalization:
    """Localization result for a single electrode."""

    electrode_id: str
    measured_distances: dict[str, float]
    ese_coords: np.ndarray
    scalp_coords: np.ndarray
    residual_error: float
    confidence: float


@dataclass
class LocalizationResult:
    """Full electrode localization result."""

    electrodes: list[ElectrodeLocalization]
    num_electrodes: int
    method: str = "brute_force"

    def get_electrode_coords(self) -> tuple[list[str], np.ndarray]:
        """Return electrode IDs and their ESE coordinates as (N,) list and (N,3) array."""
        ids = [e.electrode_id for e in self.electrodes]
        coords = np.array([e.ese_coords for e in self.electrodes])
        return ids, coords
