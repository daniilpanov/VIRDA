import json

import numpy as np

from tests.helpers.meshes import make_sphere
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.models.ese_mesh import ESEMesh
from virda.models.stage2_config import Stage2Config
from virda.pipelines.stage2 import Stage2PipelineBuilder

ESE_OFFSET_MM = 2.0


def build_pipeline(tmp_path, config: Stage2Config):
    builder = PCAESEBuilder(config=config, ese_offset_mm=ESE_OFFSET_MM)
    return Stage2PipelineBuilder(
        ese_builder=builder,
        stage2_config=config,
        scalp_mesh=make_sphere(),
        project_dir=tmp_path,
    ).build()


class TestStage2Pipeline:
    def test_run_builds_ese_mesh(self, tmp_path) -> None:
        config = Stage2Config(k_neighbors=30)
        pipeline = build_pipeline(tmp_path, config)

        context = pipeline.run()
        ese_mesh = context.get_store_notnull(ESEMesh)

        assert ese_mesh.vertices.shape == ese_mesh.scalp_vertices.shape
        assert ese_mesh.vertices.shape[0] > 0

    def test_run_exports_artifacts(self, tmp_path) -> None:
        config = Stage2Config(k_neighbors=30)
        pipeline = build_pipeline(tmp_path, config)

        context = pipeline.run()
        ese_mesh = context.get_store_notnull(ESEMesh)

        stage2_dir = tmp_path / "stage2"
        assert (stage2_dir / "ese_mesh.ply").exists()
        assert np.array_equal(np.load(stage2_dir / "ese_vertices.npy"), ese_mesh.vertices)
        assert np.array_equal(np.load(stage2_dir / "ese_faces.npy"), ese_mesh.faces)
        assert np.array_equal(np.load(stage2_dir / "normals.npy"), ese_mesh.normals)
        assert np.array_equal(np.load(stage2_dir / "quality.npy"), ese_mesh.quality)

        pairs = json.loads((stage2_dir / "point_pairs.json").read_text())
        assert pairs["n_points"] == ese_mesh.vertices.shape[0]
        assert len(pairs["scalp_vertices"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["ese_vertices"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["normals"]) == ese_mesh.vertices.shape[0]
        assert len(pairs["quality"]) == ese_mesh.vertices.shape[0]

        written_config = json.loads((stage2_dir / "stage2_config.json").read_text())
        assert written_config["stage2"]["k_neighbors"] == 30
