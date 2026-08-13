import numpy as np
import pytest

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
