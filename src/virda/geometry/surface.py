"""Proximity queries against a triangulated mesh surface.

The distances are measured to the *surface*, not to the vertices: a point
above the middle of a triangle is closer to that triangle than to any of
its corners. The acceleration structures are built once per mesh (the
vertex KD-tree and the vertex-to-face map), so repeated queries over the
same mesh stay cheap.
"""

import numpy as np
from scipy.spatial import cKDTree

from virda.models.mesh import Mesh

#: How many nearest vertices seed the candidate set of a query.
_QUERY_NEIGHBORS = 8
#: How often the search radius may grow to cover the local edge length.
_RADIUS_PASSES = 3
#: Slack of the search radius, so a vertex on the boundary stays in.
_RADIUS_SLACK = 1e-6


class SurfaceProbe:
    """Proximity queries against one mesh surface."""

    def __init__(self, mesh: Mesh) -> None:
        self._vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self._faces = np.asarray(mesh.faces, dtype=np.int64)
        self._tree = cKDTree(self._vertices)
        self._incident, self._offsets = _vertex_faces(self._faces, self._vertices.shape[0])

    def distance(self, point: np.ndarray) -> float:
        """The distance from the point to the surface, in millimetres.

        A mesh without faces has no surface: the distance is infinite.
        """
        faces = self._candidate_faces(np.asarray(point, dtype=np.float64))
        if faces.size == 0:
            return float("inf")
        triangles = self._vertices[self._faces[faces]]
        return float(np.min(_triangle_distances(np.asarray(point, dtype=np.float64), triangles)))

    def nearest_vertex(self, point: np.ndarray) -> tuple[int, np.ndarray]:
        """The index and the coordinates of the vertex closest to the point."""
        _, index = self._tree.query(np.asarray(point, dtype=np.float64), k=1)
        vertex = int(index)
        return vertex, self._vertices[vertex]

    def _candidate_faces(self, point: np.ndarray) -> np.ndarray:
        """The faces that may hold the closest surface point.

        The closest point lies in a triangle whose vertices are all
        within ``distance + longest edge`` of the query, so the faces
        around the vertices inside that radius cover it. The edge length
        comes from the neighbourhood of the query, not from the whole
        mesh: one oversized edge elsewhere must not blow up the search.
        """
        if self._faces.size == 0:
            return np.zeros(0, dtype=np.int64)
        neighbors = min(_QUERY_NEIGHBORS, self._vertices.shape[0])
        seed = float(np.max(self._tree.query(point, k=neighbors)[0]))
        radius = seed + _RADIUS_SLACK
        faces = np.zeros(0, dtype=np.int64)
        for _ in range(_RADIUS_PASSES):
            faces = self._faces_of(self._tree.query_ball_point(point, radius))
            if faces.size == 0:
                return faces
            grown = seed + self._longest_edge_of(faces) + _RADIUS_SLACK
            if grown <= radius + 1e-9:
                break
            radius = grown
        return faces

    def _faces_of(self, vertices: np.ndarray) -> np.ndarray:
        """The faces the given vertices belong to."""
        if len(vertices) == 0:
            return np.zeros(0, dtype=np.int64)
        groups = [
            self._incident[self._offsets[vertex] : self._offsets[vertex + 1]] for vertex in vertices
        ]
        return np.unique(np.concatenate(groups))

    def _longest_edge_of(self, faces: np.ndarray) -> float:
        """The longest edge of the given faces."""
        corners = self._vertices[self._faces[faces]]
        lengths = [
            np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1),
            np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
            np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1),
        ]
        return float(max(length.max() for length in lengths))


def distance_to_surface(point: np.ndarray, mesh: Mesh) -> float:
    """The distance from a world-frame point to the mesh surface.

    A one-shot helper: it builds the probe and throws it away. Use
    :class:`SurfaceProbe` directly for repeated queries.
    """
    return SurfaceProbe(mesh).distance(point)


def nearest_vertex(point: np.ndarray, mesh: Mesh) -> tuple[int, np.ndarray]:
    """The index and the coordinates of the mesh vertex closest to the point."""
    return SurfaceProbe(mesh).nearest_vertex(point)


def _vertex_faces(faces: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """The incident faces of every vertex in a compressed layout.

    The faces of a vertex are
    ``incident[offsets[vertex]:offsets[vertex + 1]]``, so the lookup
    costs nothing but a slice.
    """
    if faces.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(n_vertices + 1, dtype=np.int64)
    corners = faces.reshape(-1)
    incident = np.argsort(corners, kind="stable") // 3
    counts = np.bincount(corners, minlength=n_vertices)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return incident.astype(np.int64), offsets


def _triangle_distances(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The distance from the point to each triangle.

    A point inside a triangle is closest to its projection onto the
    triangle's plane; a point outside is closest to one of the edges.
    """
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point - a
    bp = point - b
    cp = point - c
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denominator = va + vb + vc
    usable = denominator != 0
    safe = np.where(usable, denominator, 1.0)
    v = np.where(usable, vb / safe, 0.0)
    w = np.where(usable, vc / safe, 0.0)
    projected = a + v[:, None] * ab + w[:, None] * ac
    inside = usable & (va * denominator >= 0) & (vb * denominator >= 0) & (vc * denominator >= 0)
    plane_distance = np.linalg.norm(point - projected, axis=1)
    edge_distance = np.minimum(
        np.minimum(
            _point_segment_distance(point, a, b)[0],
            _point_segment_distance(point, b, c)[0],
        ),
        _point_segment_distance(point, c, a)[0],
    )
    return np.asarray(np.where(inside, plane_distance, edge_distance), dtype=np.float64)


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The distance from the point to each segment and the closest points."""
    direction = end - start
    length = np.einsum("ij,ij->i", direction, direction)
    t = np.einsum("ij,ij->i", point - start, direction) / np.maximum(length, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = start + t[:, None] * direction
    return np.linalg.norm(point - closest, axis=1), closest
