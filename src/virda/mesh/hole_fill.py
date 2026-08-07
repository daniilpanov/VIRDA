"""Closing of small boundary rings left by internal-face removal."""

from collections import defaultdict

import numpy as np
import trimesh


def fill_small_boundary_holes(
    mesh: trimesh.Trimesh, max_perimeter_mm: float = 15.0
) -> int:
    """Close small boundary rings of ``mesh`` in place.

    A boundary ring is a closed loop of boundary edges whose vertices each lie
    on exactly two boundary edges. Rings with a perimeter at most
    ``max_perimeter_mm`` are closed with a fan of triangles from the ring
    centroid, wound to match the surrounding faces. Large open boundaries
    (neck cut, skull-base openings) are left untouched. Returns the number of
    rings filled.
    """
    if len(mesh.faces) < 3 or mesh.is_watertight:
        return 0
    edges = mesh.edges
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) < 3:
        return 0

    degree: dict[int, int] = defaultdict(int)
    for u, v in boundary:
        degree[u] += 1
        degree[v] += 1
    ring_verts = {k for k, d in degree.items() if d == 2}
    adjacency: dict[int, list[int]] = defaultdict(list)
    for u, v in boundary:
        if u in ring_verts and v in ring_verts:
            adjacency[u].append(v)
            adjacency[v].append(u)

    face_normals = mesh.face_normals
    vertex_faces: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, face in enumerate(mesh.faces):
        for q in face:
            vertex_faces[q].append(fi)

    added_vertices: list[np.ndarray] = []
    added_faces: list[np.ndarray] = []
    used: set[int] = set()
    filled = 0
    for start in ring_verts:
        if start in used:
            continue
        chain = [start]
        used.add(start)
        current = start
        while True:
            nxt = [x for x in adjacency[current] if x not in used]
            if not nxt:
                break
            current = nxt[0]
            used.add(current)
            chain.append(current)
            if current == start:
                break
        if len(chain) < 3:
            continue
        ring = mesh.vertices[chain]
        perimeter = float(np.linalg.norm(ring - np.roll(ring, -1, axis=0), axis=1).sum())
        if perimeter > max_perimeter_mm:
            continue

        center = ring.mean(axis=0)
        refs = []
        for q in chain:
            refs.append(face_normals[vertex_faces[q]])
        ref = np.mean(np.concatenate(refs), axis=0)
        norm = np.linalg.norm(ref)
        if norm == 0:
            continue
        ref /= norm
        a0 = mesh.vertices[chain[0]]
        b0 = mesh.vertices[chain[1]]
        orientation = float(np.dot(np.cross(a0 - center, b0 - center), ref))

        center_idx = len(mesh.vertices) + len(added_vertices)
        added_vertices.append(center)
        for i in range(len(chain)):
            a = chain[i]
            b = chain[(i + 1) % len(chain)]
            triangle = [center_idx, b, a] if orientation < 0 else [center_idx, a, b]
            added_faces.append(np.asarray(triangle, dtype=np.int64))
        filled += 1

    if not added_faces:
        return 0
    mesh.vertices = np.vstack([mesh.vertices, np.asarray(added_vertices)])
    mesh.faces = np.vstack([mesh.faces, np.asarray(added_faces)])
    return filled
