"""Cavity-wall detection based on air depth through the tissue mask.

Port of the ``headmesh`` geodesic approach: a face bounds an internal cavity if
the air pocket it faces is deep inside the head. Air depth is the geodesic
distance through air to the deep exterior air (border-connected air at least
``wide_mm`` away from the scalp), found by isotropic layer-by-layer dilation.
The scalp itself hugs only the thin air layer right outside the skin, so its
depth stays near ``wide_mm``; narrow open cavities (nasal cavity, pharynx, ear
canals) and sealed pockets (sinuses) are deep along the air path and get large
depths that Euclidean-distance tests miss.

The volume is assumed to be approximately 1 mm isotropic: layer indices are
interpreted as millimetres.
"""

from typing import cast

import numpy as np
import trimesh
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph

_MAX_AIR_LAYERS = 150


def air_depth_score(mask: np.ndarray, wide_mm: float = 10.0) -> np.ndarray:
    """Per-voxel air depth in mm (0 = no air).

    The flood seed is the border-connected air at least ``wide_mm`` away from
    the mask (air EDT >= ``wide_mm``): the exterior air outside the thin layer
    hugging the scalp. The seed is dilated through air, so every air voxel gets
    its geodesic distance (in mm; the volume is ~1 mm isotropic, so the layer
    index is in mm) to that deep exterior air. Air pockets never reached by the
    flood (sealed cavities such as the paranasal sinuses) get
    ``_MAX_AIR_LAYERS`` so their walls are always treated as internal. Tissue
    voxels inherit the largest air depth of their air neighbours, so cavity
    walls read the depth of the pocket they bound.
    """
    air = ~mask
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, _ = ndi.label(air)
    border_mask = np.zeros(air.shape, dtype=bool)
    border_mask[0, :, :] = True
    border_mask[-1, :, :] = True
    border_mask[:, 0, :] = True
    border_mask[:, -1, :] = True
    border_mask[:, :, 0] = True
    border_mask[:, :, -1] = True
    exterior = np.isin(labels, np.unique(labels[border_mask]))
    wide = exterior & air & (ndi.distance_transform_edt(air) >= wide_mm)
    if not wide.any():
        wide = exterior & air

    layers = np.zeros(mask.shape, dtype=np.int32)
    front = wide
    assigned = wide
    for k in range(1, _MAX_AIR_LAYERS + 1):
        if not front.any():
            break
        grown = ndi.binary_dilation(front, structure=structure)
        grown &= air & ~assigned
        layers[grown] = k
        assigned |= grown
        front = grown
    sealed = air & ~assigned
    if sealed.any():
        layers[sealed] = _MAX_AIR_LAYERS

    score = layers.astype(np.float64)
    tissue = mask
    if tissue.any():
        neighbor_layers = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    shifted = np.roll(layers, (dz, dy, dx), axis=(0, 1, 2))
                    shifted[0, :, :] = 0
                    shifted[-1, :, :] = 0
                    shifted[:, 0, :] = 0
                    shifted[:, -1, :] = 0
                    shifted[:, :, 0] = 0
                    shifted[:, :, -1] = 0
                    neighbor_layers.append(shifted[tissue])
        if neighbor_layers:
            score[tissue] = np.maximum.reduce(neighbor_layers)
    return score


def face_air_depths(
    mesh: trimesh.Trimesh, affine: np.ndarray, score: np.ndarray
) -> np.ndarray:
    """Largest air depth (mm) among the voxels of each face's three vertices."""
    world = mesh.vertices.astype(np.float64)
    shape = np.asarray(score.shape, dtype=np.float64)
    vox = (world - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T
    idx = np.clip(np.round(vox).astype(np.int64), 0, shape.astype(np.int64) - 1)
    per_vertex = score[idx[:, 0], idx[:, 1], idx[:, 2]]
    return cast(np.ndarray, per_vertex[mesh.faces].max(axis=1))


def internal_face_mask(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    affine: np.ndarray,
    wide_mm: float = 10.0,
    seed_mm: float = 20.0,
    flood_mm: float = 12.0,
) -> np.ndarray:
    """Return a boolean mask of internal (cavity-wall) faces to remove.

    ``mask`` is the binary tissue mask the mesh was extracted from and
    ``affine`` its world-space affine. Faces with air depth >= ``seed_mm`` are
    guaranteed-internal seeds; a face is removed if it belongs to a face
    adjacency component whose members all have depth >= ``flood_mm`` and that
    contains a seed face. The flood stops near cavity openings (lips, nostrils,
    ear-canal mouth), preserving natural rims, the ear concha and all outer
    skin.
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=bool)
    score = air_depth_score(mask, wide_mm=wide_mm)
    depths = face_air_depths(mesh, affine, score)
    eligible = depths >= flood_mm
    seeds = depths >= seed_mm
    return connected_components_containing_seeds(mesh, eligible, seeds)


def connected_components_containing_seeds(
    mesh: trimesh.Trimesh, eligible: np.ndarray, seeds: np.ndarray
) -> np.ndarray:
    """Faces in a face-adjacency component that contains a seed face."""
    result = np.zeros(len(mesh.faces), dtype=bool)
    seed_idx = np.flatnonzero(seeds)
    if not len(seed_idx):
        return result
    adjacency = mesh.face_adjacency
    both_eligible = eligible[adjacency[:, 0]] & eligible[adjacency[:, 1]]
    if not both_eligible.any():
        return result
    pairs = adjacency[both_eligible]
    eligible_idx = np.flatnonzero(eligible)
    position = np.full(len(mesh.faces), -1)
    position[eligible_idx] = np.arange(len(eligible_idx))
    rows = position[pairs[:, 0]]
    cols = position[pairs[:, 1]]
    subgraph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(len(eligible_idx), len(eligible_idx)),
    ).tocsr()
    _, labels = csgraph.connected_components(subgraph, directed=False)
    seed_labels = np.unique(labels[position[seed_idx]])
    result[eligible_idx] = np.isin(labels, seed_labels)
    return cast(np.ndarray, result)
