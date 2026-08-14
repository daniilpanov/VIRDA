import json
from dataclasses import asdict

import numpy as np
import pytest

from tests.helpers.meshes import make_sphere
from tests.helpers.pipelines import build_context
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.io.providers.stage2_exporter import Stage2Exporter
from virda.models.ese_mesh import ESEMesh
from virda.models.stage2_config import Stage2Config

ESE_OFFSET_MM = 2.0


def build_ese_mesh(config: Stage2Config) -> ESEMesh:
    builder = PCAESEBuilder(config=config, ese_offset_mm=ESE_OFFSET_MM)
    return builder.run(build_context(mesh=make_sphere()))


class TestStage2Exporter:
    def test_exports_mesh_and_arrays(self, tmp_path) -> None:
        config = Stage2Config(k_neighbors=30)
        result = build_ese_mesh(config)
        exporter = Stage2Exporter(project_dir=tmp_path, stage2_config=config)
        exporter.provide(result)

        stage2_dir = tmp_path / "stage2"
        assert (stage2_dir / "ese_mesh.ply").exists()
        assert np.array_equal(np.load(stage2_dir / "ese_vertices.npy"), result.vertices)
        assert np.array_equal(np.load(stage2_dir / "ese_faces.npy"), result.faces)
        assert np.array_equal(np.load(stage2_dir / "normals.npy"), result.normals)
        assert np.array_equal(np.load(stage2_dir / "quality.npy"), result.quality)

    def test_point_pairs_match_by_index(self, tmp_path) -> None:
        config = Stage2Config(k_neighbors=30)
        result = build_ese_mesh(config)
        exporter = Stage2Exporter(project_dir=tmp_path, stage2_config=config)
        exporter.provide(result)

        pairs = json.loads((tmp_path / "stage2" / "point_pairs.json").read_text())
        assert pairs["n_points"] == result.vertices.shape[0]
        assert np.allclose(pairs["scalp_vertices"], result.scalp_vertices.tolist())
        assert np.allclose(pairs["ese_vertices"], result.vertices.tolist())
        assert np.allclose(pairs["normals"], result.normals.tolist())
        assert np.allclose(pairs["quality"], result.quality.tolist())

        vertex_index = 10
        expected_ese = np.array(pairs["scalp_vertices"][vertex_index]) + ESE_OFFSET_MM * np.array(
            pairs["normals"][vertex_index]
        )
        assert np.allclose(pairs["ese_vertices"][vertex_index], expected_ese)

    def test_writes_stage2_config(self, tmp_path) -> None:
        config = Stage2Config(k_neighbors=30, use_weighted_pca=True, pca_sigma_mm=4.0)
        result = build_ese_mesh(config)
        exporter = Stage2Exporter(project_dir=tmp_path, stage2_config=config)
        exporter.provide(result)

        written = json.loads((tmp_path / "stage2" / "stage2_config.json").read_text())
        assert written == {"stage2": asdict(config)}

    def test_raises_without_result(self, tmp_path) -> None:
        exporter = Stage2Exporter(project_dir=tmp_path, stage2_config=Stage2Config(k_neighbors=30))
        with pytest.raises(ValueError, match="no result of Stage#2"):
            exporter.provide(None)
