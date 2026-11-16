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


def outward_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The unit normals of the faces, oriented away from the mesh centre.

    The orientation does not rely on the winding of the faces, so a mesh
    stored the other way round still yields outward normals.
    """
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(lengths, 1e-12)[:, None]
    centre = vertices.mean(axis=0)
    flip = np.einsum("ij,ij->i", normals, a - centre) < 0
    normals[flip] *= -1
    return normals


def circle_surface_intersections(
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray]:
    """The points where a circle meets the mesh surface.

    The circle lies in the plane through ``center`` with the given
    ``normal``. Every triangle is clipped by that plane into a segment
    (a triangle lying in the plane keeps its edges), and the segments are
    intersected with the circle.

    Returns the points and the index of the face each of them lies on.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    segments, segment_faces = _plane_segments(vertices, faces, center, normal)
    points, point_faces = _circle_segment_points(center, radius, segments, segment_faces)
    return points, point_faces


def _plane_segments(
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The segments where the plane meets the triangles.

    Only the triangles the plane crosses are clipped, so the cost follows
    the neighbourhood of the circle instead of the size of the mesh.

    Returns the segment endpoints ``(S, 2, 3)`` and the index of the
    triangle each segment belongs to. A triangle lying in the plane
    keeps its three edges instead of a single segment.
    """
    scale = float(np.linalg.norm(normal)) or 1.0
    unit = normal / scale
    offsets = np.empty((faces.shape[0], 3))
    for corner in range(3):
        offsets[:, corner] = (vertices[faces[:, corner]] - center) @ unit
    tolerance = 1e-9 * max(1.0, float(np.abs(offsets).max()))
    on_plane = np.abs(offsets) <= tolerance
    # The plane meets a triangle only where the offsets change sign.
    near = np.flatnonzero((offsets.min(axis=1) <= tolerance) & (offsets.max(axis=1) >= -tolerance))
    if near.size == 0:
        return np.zeros((0, 2, 3)), np.zeros(0, dtype=np.int64)
    triangles = vertices[faces[near]]
    on_plane = on_plane[near]
    offsets = offsets[near]

    crossings = np.full((triangles.shape[0], 3, 3), np.nan)
    for edge in range(3):
        following = (edge + 1) % 3
        start = offsets[:, edge]
        end = offsets[:, following]
        crosses = (start * end) < 0
        denominator = end - start
        ratio = np.where(crosses, -start / np.where(denominator == 0, 1.0, denominator), 0.0)
        point = triangles[:, edge] + ratio[:, None] * (triangles[:, following] - triangles[:, edge])
        crossings[:, edge] = np.where(crosses[:, None], point, np.nan)
    zeros = np.where(on_plane[:, :, None], triangles, np.nan)

    candidates = np.concatenate([crossings, zeros], axis=1)
    valid = ~np.isnan(candidates[:, :, 0])
    order = np.argsort(~valid, axis=1, kind="stable")
    ordered = np.take_along_axis(candidates, order[:, :, None], axis=1)
    counts = valid.sum(axis=1)
    segments = np.stack([ordered[:, 0], ordered[:, 1]], axis=1)

    coplanar = on_plane.all(axis=1)
    crossed = (counts >= 2) & ~coplanar
    segments = segments[crossed]
    faces = near[crossed]
    if coplanar.any():
        corners = triangles[coplanar]
        edges = np.stack(
            [
                np.stack([corners[:, 0], corners[:, 1]], axis=1),
                np.stack([corners[:, 1], corners[:, 2]], axis=1),
                np.stack([corners[:, 2], corners[:, 0]], axis=1),
            ],
            axis=1,
        ).reshape(-1, 2, 3)
        coplanar_faces = np.repeat(near[coplanar], 3)
        segments = np.concatenate([segments, edges])
        faces = np.concatenate([faces, coplanar_faces])
    return segments, faces


def _circle_segment_points(
    center: np.ndarray,
    radius: float,
    segments: np.ndarray,
    segment_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The points where the circle meets the segments, and their faces."""
    if segments.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    start = segments[:, 0]
    direction = segments[:, 1] - start
    offset = start - center
    a = np.einsum("ij,ij->i", direction, direction)
    b = 2.0 * np.einsum("ij,ij->i", offset, direction)
    c = np.einsum("ij,ij->i", offset, offset) - radius**2
    discriminant = b * b - 4.0 * a * c
    usable = (a > 0) & (discriminant >= 0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    safe = np.where(a == 0, 1.0, a)
    hits: list[np.ndarray] = []
    hit_faces: list[np.ndarray] = []
    for ratio in ((-b - root) / (2.0 * safe), (-b + root) / (2.0 * safe)):
        inside = usable & (ratio >= -1e-12) & (ratio <= 1.0 + 1e-12)
        hits.append((start + ratio[:, None] * direction)[inside])
        hit_faces.append(segment_faces[inside])
    if not hits:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.concatenate(hits, axis=0), np.concatenate(hit_faces)


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
