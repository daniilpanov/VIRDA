#!/usr/bin/env python
"""Run VIRDA Stage 2: PCA normals and ESE construction."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.surface_extractor import MeshData
from core.ese_config import ESEConfig
from core.pca_normal_estimator import NormalResult
from api.stage2 import run_stage2
from api import exporter
import numpy as np
import csv


def load_stage1(project_dir: Path) -> tuple[MeshData, ESEConfig]:
    """Load Stage 1 outputs from project directory."""
    import trimesh
    mesh_path = project_dir / "mesh" / "scalp.ply"
    tm = trimesh.load(str(mesh_path), process=False)
    mesh = MeshData(
        vertices=np.array(tm.vertices, dtype=np.float64),
        faces=np.array(tm.faces, dtype=np.int64),
    )
    config_path = project_dir / "config" / "parameters.json"
    config = ESEConfig.load(config_path)
    return mesh, config


def main():
    parser = argparse.ArgumentParser(description="VIRDA Stage 2: ESE Construction")
    parser.add_argument("--project-dir", default="patient_project", help="Project directory")
    parser.add_argument("--radius", type=float, default=10.0, help="Neighborhood radius in mm")
    parser.add_argument("--k-neighbors", type=int, default=None, help="Use k nearest neighbors")
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--weighted", action="store_true", help="Use distance-weighted PCA")
    parser.add_argument("--weight-sigma", type=float, default=5.0)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    mesh, ese_config = load_stage1(project_dir)

    ese, normal_result, qc = run_stage2(
        mesh,
        radius_mm=args.radius,
        k_neighbors=args.k_neighbors,
        min_neighbors=args.min_neighbors,
        weighted=args.weighted,
        weight_sigma=args.weight_sigma,
        ese_config=ese_config,
    )

    exporter.export_ese_pairs(ese, project_dir / "mesh" / "ese_pairs.csv")
    exporter.export_normals_csv(normal_result, mesh, project_dir / "mesh" / "normals.csv")
    exporter.export_vertices_csv(
        MeshData(vertices=ese.scalp_vertices, faces=np.empty((0, 3), dtype=np.int64)),
        project_dir / "mesh" / "scalp_vertices.csv",
    )
    exporter.export_vertices_csv(
        MeshData(vertices=ese.ese_vertices, faces=np.empty((0, 3), dtype=np.int64)),
        project_dir / "mesh" / "ese_vertices.csv",
    )

    print("\n" + qc.summary())
    return 0 if qc.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
