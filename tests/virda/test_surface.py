"""Unit tests for the mesh-surface proximity queries."""

import numpy as np
import pytest

from virda.geometry import Geometry, SurfaceProbe, distance_to_surface, nearest_vertex
from virda.geometry.frames import FrameContext
from virda.geometry.surface import _triangle_distances
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.mesh import Mesh


def _sphere(radius: float = 30.0, rings: int = 4, segments: int = 8) -> Mesh:
    """A triangulated sphere; the vertices lie on the sphere."""
    points = [np.array([0.0, 0.0, radius])]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        z = radius * np.cos(phi)
        circle = radius * np.sin(phi)
        for segment in range(segments):
            theta = 2 * np.pi * segment / segments
            points.append(np.array([circle * np.cos(theta), circle * np.sin(theta), z]))
    points.append(np.array([0.0, 0.0, -radius]))
    vertices = np.asarray(points)
    faces: list[tuple[int, int, int]] = []
    for segment in range(segments):
        faces.append((0, 1 + segment, 1 + (segment + 1) % segments))
    for ring in range(rings - 2):
        first = 1 + ring * segments
        second = first + segments
        for segment in range(segments):
            following = (segment + 1) % segments
            faces.append((first + segment, second + segment, second + following))
            faces.append((first + segment, second + following, first + following))
    last_ring = 1 + (rings - 2) * segments
    south = len(vertices) - 1
    for segment in range(segments):
        faces.append((south, last_ring + (segment + 1) % segments, last_ring + segment))
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _plane(size: float = 10.0) -> Mesh:
    """A flat square in the z = 0 plane."""
    vertices = np.array(
        [[-size, -size, 0.0], [size, -size, 0.0], [size, size, 0.0], [-size, size, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


class TestDistanceToSurface:
    def test_a_point_on_the_surface_is_at_distance_zero(self) -> None:
        mesh = _plane()

        assert distance_to_surface(np.array([0.0, 0.0, 0.0]), mesh) == pytest.approx(0.0, abs=1e-9)

    def test_the_distance_is_measured_above_the_plane(self) -> None:
        mesh = _plane()

        assert distance_to_surface(np.array([0.0, 0.0, 5.0]), mesh) == pytest.approx(5.0)
        assert distance_to_surface(np.array([0.0, 0.0, -3.0]), mesh) == pytest.approx(3.0)

    def test_the_distance_to_a_sphere_is_its_inradius(self) -> None:
        mesh = _sphere(radius=30.0)
        triangles = mesh.vertices[mesh.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        offsets = np.abs(np.einsum("ij,ij->i", normals, triangles[:, 0])) / np.maximum(
            lengths, 1e-12
        )

        # The centre of a triangulated sphere is its inradius away from
        # the surface, not the radius itself.
        assert distance_to_surface(np.zeros(3), mesh) == pytest.approx(
            float(offsets.min()), rel=1e-9
        )

    def test_a_point_beyond_the_surface_is_measured_from_it(self) -> None:
        mesh = _sphere(radius=30.0)
        direction = np.array([0.0, 0.0, 1.0])

        assert distance_to_surface(direction * 35.0, mesh) == pytest.approx(5.0, rel=1e-6)

    def test_a_mesh_without_faces_has_no_surface(self) -> None:
        mesh = Mesh(vertices=np.zeros((3, 3)), faces=np.zeros((0, 3), dtype=np.int64))

        assert distance_to_surface(np.array([1.0, 1.0, 1.0]), mesh) == float("inf")

    def test_the_probe_reuses_its_structures(self) -> None:
        mesh = _sphere()
        probe = SurfaceProbe(mesh)

        first = probe.distance(np.array([0.0, 0.0, 35.0]))
        second = probe.distance(np.array([0.0, 0.0, 40.0]))

        assert first == pytest.approx(5.0, rel=1e-6)
        assert second == pytest.approx(10.0, rel=1e-6)

    def test_the_candidate_prefilter_misses_nothing(self) -> None:
        """The search must find the same closest triangle as a full scan."""
        rng = np.random.default_rng(0)
        vertices = rng.uniform(-40.0, 40.0, (60, 3))
        faces = rng.integers(0, 60, (80, 3))
        faces = faces[faces[:, 0] != faces[:, 1]]
        mesh = Mesh(vertices=vertices, faces=faces)
        triangles = vertices[mesh.faces]

        for _ in range(10):
            point = rng.uniform(-45.0, 45.0, 3)
            brute_force = float(np.min(_triangle_distances(point, triangles)))
            assert distance_to_surface(point, mesh) == pytest.approx(brute_force, abs=1e-9)


class TestNearestVertex:
    def test_the_closest_vertex_is_found(self) -> None:
        mesh = _plane()
        point = np.array([1.0, 2.0, 5.0])

        index, coordinates = nearest_vertex(point, mesh)

        assert index in (0, 1, 2, 3)
        assert np.allclose(coordinates, mesh.vertices[index])

    def test_the_vertex_of_a_sphere_is_the_nearest_one(self) -> None:
        mesh = _sphere(radius=30.0)
        point = np.array([0.0, 0.0, 40.0])

        index, coordinates = nearest_vertex(point, mesh)

        assert np.allclose(coordinates, mesh.vertices[index])
        assert coordinates[2] > 29.0


class TestGeometrySurface:
    def test_the_point_may_come_in_any_frame(self) -> None:
        mesh = _plane()
        geometry = Geometry(
            FrameContext(head_origin=Point(x=0.0, y=0.0, z=0.0, system=CoordinateSystem.WORLD))
        )
        point = Point(x=0.0, y=0.0, z=5.0, system=CoordinateSystem.WORLD)

        assert geometry.distance_to_surface(point, mesh) == pytest.approx(5.0)
        assert geometry.point_on_surface(point, mesh) is False

    def test_the_tolerance_decides_what_counts_as_on_the_surface(self) -> None:
        mesh = _plane()
        geometry = Geometry(FrameContext())
        point = Point(x=0.0, y=0.0, z=0.5, system=CoordinateSystem.WORLD)

        assert geometry.point_on_surface(point, mesh) is True
        assert geometry.point_on_surface(point, mesh, tolerance_mm=0.1) is False

    def test_a_head_frame_point_is_converted_before_the_query(self) -> None:
        mesh = _plane()
        geometry = Geometry(
            FrameContext(head_origin=Point(x=0.0, y=0.0, z=10.0, system=CoordinateSystem.WORLD))
        )
        # In the head frame the origin sits 10 mm above the plane.
        point = Point(x=0.0, y=0.0, z=0.0, system=CoordinateSystem.HEAD)

        assert geometry.distance_to_surface(point, mesh) == pytest.approx(10.0)

    def test_the_probe_of_a_mesh_is_cached(self) -> None:
        mesh = _plane()
        geometry = Geometry(FrameContext())

        probe = geometry.probe(mesh)

        assert geometry.probe(mesh) is probe
