import numpy as np
import pytest
from skimage.measure import marching_cubes

from virda.fiducials.detect import (
    find_ears,
    find_fiducials,
    find_inion,
    find_nasion,
    find_nose_tip,
    to_fiducials,
)
from virda.models.scalp_mesh import ScalpMesh


def _head_mesh() -> ScalpMesh:
    """Ellipsoidal head with nose and inion bumps (world mm, RAS)."""
    spacing = np.array([2.0, 2.0, 2.0])
    origin = np.array([-70.0, -95.0, -100.0])
    xs = np.arange(0, 71)
    ys = np.arange(0, 96)
    zs = np.arange(0, 101)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    x = gx * spacing[0] + origin[0]
    y = gy * spacing[1] + origin[1]
    z = gz * spacing[2] + origin[2]

    a, b, c = 60.0, 80.0, 90.0
    r0 = np.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2)
    nose = (
        0.18
        * np.exp(-(((y - 40) / 35) ** 2))
        * np.exp(-(((z - 20) / 18) ** 2))
        / (1 + (x / 25) ** 2)
    )
    inion = (
        0.15
        * np.exp(-(((y + 30) / 30) ** 2))
        * np.exp(-(((z - 5) / 15) ** 2))
        / (1 + (x / 25) ** 2)
    )
    field = r0 - nose - inion

    verts, faces, _, _ = marching_cubes(field, level=1.0)
    return ScalpMesh(
        vertices=verts.astype(np.float64) * spacing + origin,
        faces=faces.astype(np.int64),
    )


class TestFiducialDetection:
    @pytest.fixture(scope="class")
    def head_mesh(self) -> ScalpMesh:
        return _head_mesh()

    def test_find_fiducials_returns_all_points(self, head_mesh: ScalpMesh) -> None:
        points = find_fiducials(head_mesh.vertices)
        assert set(points) == {"NAS", "LPA", "RPA", "INI"}

    def test_nasion_is_anterior_midline(self, head_mesh: ScalpMesh) -> None:
        nose_tip = find_nose_tip(head_mesh.vertices)
        nasion = find_nasion(head_mesh.vertices, nose_tip)
        assert abs(nasion[0]) < 5.0
        assert nasion[1] > 0

    def test_ears_are_lateral(self, head_mesh: ScalpMesh) -> None:
        nose_tip = find_nose_tip(head_mesh.vertices)
        nasion = find_nasion(head_mesh.vertices, nose_tip)
        left, right = find_ears(head_mesh.vertices, nasion)
        assert left[0] < 0
        assert right[0] > 0
        assert abs(left[1]) < nasion[1]
        assert abs(right[1]) < nasion[1]

    def test_inion_is_posterior_to_nasion(self, head_mesh: ScalpMesh) -> None:
        nose_tip = find_nose_tip(head_mesh.vertices)
        nasion = find_nasion(head_mesh.vertices, nose_tip)
        inion = find_inion(head_mesh.vertices, nasion)
        assert abs(inion[0]) < 5.0
        assert inion[1] < nasion[1]

    def test_to_fiducials_builds_world_coordinates(self, head_mesh: ScalpMesh) -> None:
        fiducials = to_fiducials(find_fiducials(head_mesh.vertices))
        ids = {fiducial.fiducial_id for fiducial in fiducials}
        assert ids == {"NAS", "LPA", "RPA", "INI"}
        for fiducial in fiducials:
            assert fiducial.coordinate_system == "world"
            assert fiducial.definition_method == "auto"
            assert fiducial.coordinates.shape == (3,)
