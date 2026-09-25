import json
from pathlib import Path

import numpy as np
import pytest

from virda.io.importers.ese_mesh import import_ese_mesh
from virda.models.ese_mesh import ESEMesh


def make_ese_mesh(n_vertices: int = 8, n_faces: int = 6) -> ESEMesh:
    return ESEMesh(
        vertices=np.zeros((n_vertices, 3), dtype=np.float64),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [4, 5, 6],
                [4, 5, 7],
                [4, 6, 7],
            ][:n_faces],
            dtype=np.int64,
        ),
        scalp_vertices=np.zeros((n_vertices, 3), dtype=np.float64),
        normals=np.zeros((n_vertices, 3), dtype=np.float64),
        quality=np.zeros(n_vertices, dtype=np.float64),
    )


class TestESEMesh:
    def test_constructs_valid_mesh(self) -> None:
        mesh = make_ese_mesh()
        assert mesh.vertices.shape == (8, 3)
        assert mesh.faces.shape == (6, 3)
        assert mesh.scalp_vertices.shape == (8, 3)
        assert mesh.normals.shape == (8, 3)
        assert mesh.quality.shape == (8,)

    @pytest.mark.parametrize(
        "field",
        ["vertices", "scalp_vertices", "normals"],
    )
    def test_rejects_non_3d_column_mesh(self, field: str) -> None:
        mesh = make_ese_mesh()
        kwargs = {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "scalp_vertices": mesh.scalp_vertices,
            "normals": mesh.normals,
            "quality": mesh.quality,
        }
        kwargs[field] = np.zeros((8, 2), dtype=np.float64)
        with pytest.raises(ValueError, match=rf"{field} must be \(N, 3\) array"):
            ESEMesh(**kwargs)

    def test_rejects_wrong_faces_shape(self) -> None:
        mesh = make_ese_mesh()
        with pytest.raises(ValueError, match=r"faces must be \(M, 3\) array"):
            ESEMesh(
                vertices=mesh.vertices,
                faces=np.zeros((6, 2), dtype=np.int64),
                scalp_vertices=mesh.scalp_vertices,
                normals=mesh.normals,
                quality=mesh.quality,
            )

    def test_rejects_wrong_quality_shape(self) -> None:
        mesh = make_ese_mesh()
        with pytest.raises(ValueError, match=r"quality must be \(N,\) array"):
            ESEMesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                scalp_vertices=mesh.scalp_vertices,
                normals=mesh.normals,
                quality=np.zeros((8, 1), dtype=np.float64),
            )

    def test_rejects_row_count_mismatch(self) -> None:
        mesh = make_ese_mesh()
        with pytest.raises(ValueError, match="must have 8 rows to match vertices"):
            ESEMesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                scalp_vertices=np.zeros((7, 3), dtype=np.float64),
                normals=mesh.normals,
                quality=mesh.quality,
            )

    @pytest.mark.parametrize("faces", [[[0, 1, 8]], [[-1, 0, 1]]])
    def test_rejects_faces_indices_out_of_range(self, faces: list[list[int]]) -> None:
        mesh = make_ese_mesh()
        with pytest.raises(ValueError, match="Faces indices must be in"):
            ESEMesh(
                vertices=mesh.vertices,
                faces=np.array(faces, dtype=np.int64),
                scalp_vertices=mesh.scalp_vertices,
                normals=mesh.normals,
                quality=mesh.quality,
            )


def _write_ese_project_files(ese_dir: Path, mesh: ESEMesh) -> Path:
    """Write a project ``ese/`` directory for *mesh*; return the PLY path."""
    ese_dir.mkdir(parents=True, exist_ok=True)
    np.save(ese_dir / "ese_vertices.npy", mesh.vertices)
    np.save(ese_dir / "ese_faces.npy", mesh.faces)
    np.save(ese_dir / "normals.npy", mesh.normals)
    np.save(ese_dir / "quality.npy", mesh.quality)
    (ese_dir / "point_pairs.json").write_text(
        json.dumps(
            {
                "n_points": len(mesh.vertices),
                "scalp_vertices": mesh.scalp_vertices.tolist(),
                "ese_vertices": mesh.vertices.tolist(),
                "normals": mesh.normals.tolist(),
                "quality": mesh.quality.tolist(),
            }
        ),
        encoding="utf-8",
    )
    ply = ese_dir / "ese_mesh.ply"
    ply.write_text("placeholder", encoding="utf-8")
    return ply


class TestImportEseMesh:
    def test_round_trip_through_project_files(self, tmp_path: Path) -> None:
        mesh = make_ese_mesh()
        ply = _write_ese_project_files(tmp_path / "ese", mesh)

        restored = import_ese_mesh(ply)

        assert np.array_equal(restored.vertices, mesh.vertices)
        assert np.array_equal(restored.faces, mesh.faces)
        assert np.array_equal(restored.scalp_vertices, mesh.scalp_vertices)
        assert np.array_equal(restored.normals, mesh.normals)
        assert np.array_equal(restored.quality, mesh.quality)

    def test_missing_companion_raises(self, tmp_path: Path) -> None:
        ese_dir = tmp_path / "ese"
        ese_dir.mkdir()
        ply = ese_dir / "ese_mesh.ply"
        ply.write_text("placeholder", encoding="utf-8")

        with pytest.raises(ValueError, match="Cannot import ESE mesh"):
            import_ese_mesh(ply)
