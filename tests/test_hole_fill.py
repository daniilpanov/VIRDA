from typing import cast

import numpy as np
import trimesh

from virda.mesh.contracts import MeshCleaner
from virda.mesh.hole_fill import HoleFillCleaner, fill_small_boundary_holes


def _box_mesh(extents: float = 2.0) -> trimesh.Trimesh:
    return cast(trimesh.Trimesh, trimesh.creation.box(extents=(extents, extents, extents)))


class TestFillSmallBoundaryHoles:
    def test_fills_small_hole(self) -> None:
        mesh = _box_mesh()
        assert mesh.is_watertight

        mesh.update_faces(np.arange(1, len(mesh.faces)))
        assert not mesh.is_watertight

        filled = fill_small_boundary_holes(mesh)

        assert filled >= 1
        assert mesh.is_watertight
        assert len(mesh.faces) > 11

    def test_skips_large_hole(self) -> None:
        mesh = _box_mesh(extents=10.0)
        mesh.update_faces(np.arange(1, len(mesh.faces)))

        filled = fill_small_boundary_holes(mesh)

        assert filled == 0
        assert not mesh.is_watertight

    def test_watertight_mesh_returns_zero(self) -> None:
        mesh = _box_mesh()

        assert fill_small_boundary_holes(mesh) == 0

    def test_tiny_mesh_returns_zero(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2]],
        )

        assert fill_small_boundary_holes(mesh) == 0


class TestHoleFillCleaner:
    def test_is_mesh_cleaner(self) -> None:
        assert isinstance(HoleFillCleaner(), MeshCleaner)
