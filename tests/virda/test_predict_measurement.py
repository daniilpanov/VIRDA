"""Unit tests for the predicted third measurement."""

import numpy as np
import pytest

from virda.geometry import Geometry
from virda.geometry.frames import FrameContext
from virda.geometry.measurement import predict_measurement
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.fiducial import FiducialType
from virda.models.mesh import Mesh

_RADIUS = 90.0


def _head_mesh(rings: int = 30, segments: int = 60) -> Mesh:
    """A triangulated head-ish sphere with the poles on the z axis."""
    points = [np.array([0.0, 0.0, _RADIUS])]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        z = _RADIUS * np.cos(phi)
        circle = _RADIUS * np.sin(phi)
        for segment in range(segments):
            theta = 2 * np.pi * segment / segments
            points.append(np.array([circle * np.cos(theta), circle * np.sin(theta), z]))
    points.append(np.array([0.0, 0.0, -_RADIUS]))
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


def _on_sphere(direction: np.ndarray) -> np.ndarray:
    """The point of the sphere along a direction."""
    unit = direction / np.linalg.norm(direction)
    return np.asarray(unit * _RADIUS, dtype=np.float64)


def _fiducials() -> dict[FiducialType, Point]:
    """The three canonical fiducials on the sphere."""
    nas = _on_sphere(np.array([0.0, 0.89, 0.45]))
    lpa = _on_sphere(np.array([-0.92, -0.15, 0.35]))
    rpa = _on_sphere(np.array([0.92, -0.15, 0.35]))
    return {
        FiducialType.NAS: Point(
            x=float(nas[0]), y=float(nas[1]), z=float(nas[2]), system=CoordinateSystem.WORLD
        ),
        FiducialType.LPA: Point(
            x=float(lpa[0]), y=float(lpa[1]), z=float(lpa[2]), system=CoordinateSystem.WORLD
        ),
        FiducialType.RPA: Point(
            x=float(rpa[0]), y=float(rpa[1]), z=float(rpa[2]), system=CoordinateSystem.WORLD
        ),
    }


def _distances(
    point: np.ndarray, fiducials: dict[FiducialType, Point]
) -> dict[FiducialType, float]:
    """The exact distances from a point to the fiducials."""
    return {
        fiducial: float(np.linalg.norm(point - fiducial_point.as_array()))
        for fiducial, fiducial_point in fiducials.items()
    }


def _electrode() -> np.ndarray:
    """A point on the upper left of the sphere."""
    return _on_sphere(np.array([-0.5, 0.25, 0.83]))


class TestPredictMeasurement:
    def test_the_third_measurement_is_recovered(self) -> None:
        mesh = _head_mesh()
        fiducials = _fiducials()
        electrode = _electrode()
        distances = _distances(electrode, fiducials)

        prediction = predict_measurement(
            {
                FiducialType.NAS: distances[FiducialType.NAS],
                FiducialType.LPA: distances[FiducialType.LPA],
            },
            fiducials,
            mesh,
            FiducialType.RPA,
            FrameContext(),
            approximate=False,
        )

        assert prediction.point is not None
        assert prediction.distance_mm == pytest.approx(distances[FiducialType.RPA], abs=0.5)
        assert prediction.residual_mm == pytest.approx(0.0, abs=1e-6)
        assert prediction.ambiguous is False
        assert prediction.vertex_index is None

    def test_the_position_is_the_electrode(self) -> None:
        mesh = _head_mesh()
        fiducials = _fiducials()
        electrode = _electrode()
        distances = _distances(electrode, fiducials)

        prediction = predict_measurement(
            {
                FiducialType.NAS: distances[FiducialType.NAS],
                FiducialType.LPA: distances[FiducialType.LPA],
            },
            fiducials,
            mesh,
            FiducialType.RPA,
            FrameContext(),
            approximate=False,
        )

        assert prediction.point is not None
        assert np.allclose(prediction.point.as_array(), electrode, atol=0.5)

    def test_the_approximation_snaps_to_a_mesh_vertex(self) -> None:
        mesh = _head_mesh()
        fiducials = _fiducials()
        electrode = _electrode()
        distances = _distances(electrode, fiducials)

        prediction = predict_measurement(
            {
                FiducialType.NAS: distances[FiducialType.NAS],
                FiducialType.LPA: distances[FiducialType.LPA],
            },
            fiducials,
            mesh,
            FiducialType.RPA,
            FrameContext(),
            approximate=True,
        )

        assert prediction.vertex_index is not None
        assert prediction.point is not None
        assert np.allclose(prediction.point.as_array(), mesh.vertices[prediction.vertex_index])
        # The vertex is on the surface, but it is a few millimetres away
        # from the exact intersection: that is what the mesh density buys.
        assert prediction.residual_mm is not None
        assert prediction.residual_mm < 5.0

    def test_the_measurement_follows_the_approximated_position(self) -> None:
        mesh = _head_mesh()
        fiducials = _fiducials()
        electrode = _electrode()
        distances = _distances(electrode, fiducials)

        prediction = predict_measurement(
            {
                FiducialType.NAS: distances[FiducialType.NAS],
                FiducialType.LPA: distances[FiducialType.LPA],
            },
            fiducials,
            mesh,
            FiducialType.RPA,
            FrameContext(),
            approximate=True,
        )

        assert prediction.point is not None
        rpa = fiducials[FiducialType.RPA].as_array()
        assert prediction.distance_mm == pytest.approx(
            float(np.linalg.norm(prediction.point.as_array() - rpa))
        )

    def test_the_position_is_returned_in_the_requested_frame(self) -> None:
        mesh = _head_mesh()
        fiducials = _fiducials()
        electrode = _electrode()
        distances = _distances(electrode, fiducials)
        context = FrameContext(
            head_origin=Point(x=1.0, y=2.0, z=3.0, system=CoordinateSystem.WORLD)
        )

        prediction = predict_measurement(
            {
                FiducialType.NAS: distances[FiducialType.NAS],
                FiducialType.LPA: distances[FiducialType.LPA],
            },
            fiducials,
            mesh,
            FiducialType.RPA,
            context,
            approximate=False,
            system=CoordinateSystem.HEAD,
        )

        assert prediction.point is not None
        assert prediction.point.system is CoordinateSystem.HEAD
        assert np.allclose(
            prediction.point.as_array(),
            electrode - np.array([1.0, 2.0, 3.0]),
            atol=0.5,
        )

    def test_a_surface_without_the_circle_has_no_prediction(self) -> None:
        mesh = _head_mesh()
        fiducials = _fiducials()
        separation = float(
            np.linalg.norm(
                fiducials[FiducialType.NAS].as_array() - fiducials[FiducialType.LPA].as_array()
            )
        )

        # The circle collapses to the midpoint between the fiducials, which
        # is inside the head: the surface never meets it.
        prediction = predict_measurement(
            {FiducialType.NAS: separation / 2.0, FiducialType.LPA: separation / 2.0},
            fiducials,
            mesh,
            FiducialType.RPA,
            FrameContext(),
        )

        assert prediction.point is None
        assert prediction.distance_mm is None
        assert prediction.ambiguous is False


class TestPredictMeasurementAmbiguity:
    def test_two_candidates_on_the_plane_are_ambiguous(self) -> None:
        """A flat mesh leaves both candidates on the fiducial plane."""
        size = 200.0
        mesh = Mesh(
            vertices=np.array(
                [
                    [-size, -size, 0.0],
                    [size, -size, 0.0],
                    [size, size, 0.0],
                    [-size, size, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        fiducials = {
            FiducialType.NAS: Point(x=0.0, y=80.0, z=0.0, system=CoordinateSystem.WORLD),
            FiducialType.LPA: Point(x=-80.0, y=0.0, z=0.0, system=CoordinateSystem.WORLD),
            FiducialType.RPA: Point(x=80.0, y=0.0, z=0.0, system=CoordinateSystem.WORLD),
        }

        prediction = predict_measurement(
            {FiducialType.NAS: 80.0, FiducialType.LPA: 80.0},
            fiducials,
            mesh,
            FiducialType.RPA,
            FrameContext(),
            approximate=False,
        )

        assert prediction.point is not None
        assert prediction.ambiguous is True

    def test_an_ambiguous_prediction_still_reports_a_measurement(self) -> None:
        size = 200.0
        mesh = Mesh(
            vertices=np.array(
                [
                    [-size, -size, 0.0],
                    [size, -size, 0.0],
                    [size, size, 0.0],
                    [-size, size, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        fiducials = {
            FiducialType.NAS: Point(x=0.0, y=80.0, z=0.0, system=CoordinateSystem.WORLD),
            FiducialType.LPA: Point(x=-80.0, y=0.0, z=0.0, system=CoordinateSystem.WORLD),
            FiducialType.RPA: Point(x=80.0, y=0.0, z=0.0, system=CoordinateSystem.WORLD),
        }

        prediction = predict_measurement(
            {FiducialType.NAS: 80.0, FiducialType.LPA: 80.0},
            fiducials,
            mesh,
            FiducialType.RPA,
            FrameContext(),
            approximate=False,
        )

        assert prediction.distance_mm is not None
        assert prediction.distance_mm > 0.0


class TestPredictMeasurementErrors:
    def test_two_measurements_are_needed(self) -> None:
        fiducials = _fiducials()

        with pytest.raises(ValueError, match="exactly two"):
            predict_measurement(
                {FiducialType.NAS: 10.0},
                fiducials,
                _head_mesh(rings=4, segments=8),
                FiducialType.RPA,
                FrameContext(),
            )

    def test_the_measured_fiducials_need_coordinates(self) -> None:
        fiducials = _fiducials()

        with pytest.raises(ValueError, match="no coordinates"):
            predict_measurement(
                {FiducialType.NAS: 10.0, FiducialType.LPA: 10.0},
                {FiducialType.NAS: fiducials[FiducialType.NAS]},
                _head_mesh(rings=4, segments=8),
                FiducialType.RPA,
                FrameContext(),
            )

    def test_the_missing_fiducial_needs_coordinates(self) -> None:
        fiducials = _fiducials()
        without_rpa: dict[FiducialType, Point] = {
            fiducial: point
            for fiducial, point in fiducials.items()
            if fiducial is not FiducialType.RPA
        }

        with pytest.raises(ValueError, match="missing fiducial"):
            predict_measurement(
                {FiducialType.NAS: 10.0, FiducialType.LPA: 10.0},
                without_rpa,
                _head_mesh(rings=4, segments=8),
                FiducialType.RPA,
                FrameContext(),
            )

    def test_impossible_distances_are_rejected(self) -> None:
        fiducials = _fiducials()

        with pytest.raises(ValueError, match="cannot belong to one point"):
            predict_measurement(
                {FiducialType.NAS: 1.0, FiducialType.LPA: 2.0},
                fiducials,
                _head_mesh(rings=4, segments=8),
                FiducialType.RPA,
                FrameContext(),
            )


class TestElectrodeSide:
    """The side of the fiducial plane the electrode sits on."""

    @staticmethod
    def _asymmetric_head() -> Mesh:
        """A head that is not symmetric: the mirror candidates differ."""
        mesh = _head_mesh(rings=20, segments=40)
        vertices = mesh.vertices * np.array([0.85, 0.95, 1.1])
        return Mesh(vertices=vertices, faces=mesh.faces)

    @staticmethod
    def _vertex_towards(mesh: Mesh, direction: np.ndarray) -> np.ndarray:
        """The vertex of the mesh that lies furthest along the direction."""
        unit = direction / np.linalg.norm(direction)
        vertex = mesh.vertices[int(np.argmax(mesh.vertices @ unit))]
        return np.asarray(vertex, dtype=np.float64)

    @staticmethod
    def _fiducials_on(mesh: Mesh) -> dict[FiducialType, Point]:
        """The three canonical fiducials, on the vertices of the mesh."""
        directions = {
            FiducialType.NAS: np.array([0.0, 0.89, 0.45]),
            FiducialType.LPA: np.array([-0.92, -0.15, 0.35]),
            FiducialType.RPA: np.array([0.92, -0.15, 0.35]),
        }
        fiducials: dict[FiducialType, Point] = {}
        for slot, direction in directions.items():
            vertex = TestElectrodeSide._vertex_towards(mesh, direction)
            fiducials[slot] = Point(
                x=float(vertex[0]),
                y=float(vertex[1]),
                z=float(vertex[2]),
                system=CoordinateSystem.WORLD,
            )
        return fiducials

    def test_the_side_does_not_depend_on_the_measured_order(self) -> None:
        """Every missing fiducial is predicted on the same side of the head."""
        mesh = self._asymmetric_head()
        fiducials = self._fiducials_on(mesh)
        electrode = self._vertex_towards(mesh, np.array([0.1, 0.5, 0.9]))
        distances = _distances(electrode, fiducials)

        for missing in (FiducialType.NAS, FiducialType.LPA, FiducialType.RPA):
            measured = {slot: value for slot, value in distances.items() if slot is not missing}
            prediction = predict_measurement(
                measured,
                fiducials,
                mesh,
                missing,
                FrameContext(),
                approximate=False,
            )
            assert prediction.distance_mm == pytest.approx(distances[missing], abs=0.01)


class TestGeometryPrediction:
    def test_the_service_predicts_a_measurement(self) -> None:
        mesh = _head_mesh(rings=8, segments=16)
        fiducials = _fiducials()
        electrode = _electrode()
        distances = _distances(electrode, fiducials)
        geometry = Geometry(FrameContext())

        prediction = geometry.predict_measurement(
            {
                FiducialType.NAS: distances[FiducialType.NAS],
                FiducialType.LPA: distances[FiducialType.LPA],
            },
            fiducials,
            mesh,
            FiducialType.RPA,
            approximate=False,
        )

        assert prediction.distance_mm == pytest.approx(distances[FiducialType.RPA], abs=5.0)
