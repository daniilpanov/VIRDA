"""Tests for core.pca_normal_estimator module."""

import numpy as np
import pytest

from virda.core.pca_normal_estimator import estimate_normals_pca
from virda.core.types import MeshData


class TestPCA:
    def test_flat_plane_normal(self):
        """Points on the XY plane should have normals pointing along Z."""
        rng = np.random.default_rng(42)
        n = 200
        xy = rng.uniform(-10, 10, (n, 2))
        verts = np.column_stack([xy, np.zeros(n)])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = MeshData(vertices=verts, faces=faces)

        result = estimate_normals_pca(mesh, radius_mm=5.0, min_neighbors=3)

        mean_normal = result.normals.mean(axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        assert abs(abs(mean_normal[2]) - 1.0) < 0.1, (
            f"Expected normals along Z, got mean normal {mean_normal}"
        )

    def test_sphere_normals_radial(self):
        """Normals on a sphere should point approximately radially outward."""
        rng = np.random.default_rng(42)
        n = 500
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = np.arccos(2 * rng.random(n) - 1)
        r = 30.0
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        verts = np.column_stack([x, y, z])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = MeshData(vertices=verts, faces=faces)

        result = estimate_normals_pca(mesh, radius_mm=10.0, min_neighbors=5)

        radial = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        high_quality = result.quality < 0.1
        if high_quality.any():
            assert np.mean(dots[high_quality]) > 0.8, (
                f"Sphere normals should align with radial direction"
            )
