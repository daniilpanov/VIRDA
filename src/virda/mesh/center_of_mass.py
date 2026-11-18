"""The centre of mass of a mesh."""

import numpy as np
import trimesh

from virda.models.mesh import Mesh


def center_of_mass(mesh: Mesh) -> np.ndarray:
    """The centre of mass of the mesh, in world millimeters.

    A closed mesh encloses a volume, and its centre of mass is the
    centroid of that volume; an open surface has no volume, so its
    centre of mass is the area-weighted centroid of its triangles. A
    mesh without area (an empty or a degenerate one) has no centre of
    mass at all; the mean of its vertices is returned instead.
    """
    solid = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    centre = np.asarray(solid.center_mass, dtype=np.float64)
    if not np.all(np.isfinite(centre)):
        return np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    return centre
