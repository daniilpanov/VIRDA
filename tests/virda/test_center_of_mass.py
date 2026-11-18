"""The centre of mass of a mesh."""

import numpy as np

from virda.mesh.center_of_mass import center_of_mass
from virda.models.mesh import Mesh

# A cube centred on the origin: every centre of mass lies at its middle.
_CUBE = Mesh(
    np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    ),
    np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ]
    ),
)


def test_center_of_mass_of_a_symmetric_mesh_is_its_middle() -> None:
    centre = center_of_mass(_CUBE)

    assert centre.shape == (3,)
    assert np.allclose(centre, np.zeros(3), atol=1e-6)


def test_center_of_mass_follows_the_mesh() -> None:
    """The centre of mass moves with the mesh it is computed from."""
    shifted = Mesh(_CUBE.vertices + np.array([10.0, 0.0, 0.0]), _CUBE.faces)

    centre = center_of_mass(shifted)

    assert np.allclose(centre, np.array([10.0, 0.0, 0.0]), atol=1e-6)


def test_center_of_mass_of_a_degenerate_mesh_is_the_vertex_mean() -> None:
    """A mesh without area has no centre of mass; the mean stands in."""
    degenerate = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))

    centre = center_of_mass(degenerate)

    assert np.allclose(centre, np.zeros(3))
