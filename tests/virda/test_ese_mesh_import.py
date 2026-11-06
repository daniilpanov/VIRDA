import json
from pathlib import Path

import numpy as np
import pytest

from virda.io.importers.ese_mesh import import_ese_mesh
from virda.models.ese_mesh import ESEMesh

VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64
)
FACES = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
SCALP = VERTICES - np.array([0.0, 0.0, 2.0])
QUALITY = np.array([1.0, 1.0, 1.0, 1.0])


def _ese() -> ESEMesh:
    return ESEMesh(
        vertices=VERTICES,
        faces=FACES,
        scalp_vertices=SCALP,
        quality=QUALITY,
    )


def _write_companions(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mesh.ply").touch()
    np.save(root / "ese_vertices.npy", VERTICES)
    np.save(root / "ese_faces.npy", FACES)
    np.save(root / "quality.npy", QUALITY)
    (root / "point_pairs.json").write_text(
        json.dumps({"n_points": 4, "scalp_vertices": SCALP.tolist()}), encoding="utf-8"
    )


class TestImportEseMesh:
    def test_round_trip_from_companions(self, tmp_path: Path) -> None:
        _write_companions(tmp_path / "ese")

        ese = import_ese_mesh(tmp_path / "ese" / "mesh.ply")

        assert isinstance(ese, ESEMesh)
        np.testing.assert_array_almost_equal(ese.vertices, VERTICES)
        np.testing.assert_array_equal(ese.faces, FACES)
        np.testing.assert_array_almost_equal(ese.scalp_vertices, SCALP)
        np.testing.assert_array_almost_equal(ese.quality, QUALITY)


class TestImportEseMeshErrors:
    def test_missing_ply(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            import_ese_mesh(tmp_path / "ese" / "mesh.ply")

    def test_missing_companion_file(self, tmp_path: Path) -> None:
        root = tmp_path / "ese"
        _write_companions(root)
        (root / "quality.npy").unlink()

        with pytest.raises(ValueError, match="Cannot import ESE mesh"):
            import_ese_mesh(root / "mesh.ply")

    def test_malformed_point_pairs(self, tmp_path: Path) -> None:
        root = tmp_path / "ese"
        _write_companions(root)
        (root / "point_pairs.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Cannot import ESE mesh"):
            import_ese_mesh(root / "mesh.ply")

    def test_inconsistent_shapes(self, tmp_path: Path) -> None:
        root = tmp_path / "ese"
        _write_companions(root)
        pairs_path = root / "point_pairs.json"
        pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
        pairs["scalp_vertices"] = pairs["scalp_vertices"][:2]
        pairs_path.write_text(json.dumps(pairs), encoding="utf-8")

        with pytest.raises(ValueError, match="Cannot import ESE mesh"):
            import_ese_mesh(root / "mesh.ply")
