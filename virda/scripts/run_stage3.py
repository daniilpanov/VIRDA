#!/usr/bin/env python
"""Run VIRDA Stage 3: Electrode localization from measurements."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ese_generator import ESEResult
from core.fiducial_manager import FiducialManager
from api.stage3 import run_stage3, load_measurements
from api import exporter
import csv
import numpy as np


def load_stage2(project_dir: Path) -> ESEResult:
    """Load Stage 2 outputs from project directory."""
    pairs_path = project_dir / "mesh" / "ese_pairs.csv"
    scalp_verts = []
    ese_verts = []
    normals = []
    quality = []
    with open(pairs_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scalp_verts.append([float(row["scalp_x"]), float(row["scalp_y"]), float(row["scalp_z"])])
            ese_verts.append([float(row["ese_x"]), float(row["ese_y"]), float(row["ese_z"])])
            normals.append([float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"])])
            quality.append(float(row["pca_quality"]))

    return ESEResult(
        ese_points=[],
        scalp_vertices=np.array(scalp_verts),
        ese_vertices=np.array(ese_verts),
        normals=np.array(normals),
        quality=np.array(quality),
        head_centroid=np.array(scalp_verts).mean(axis=0),
    )


def main():
    parser = argparse.ArgumentParser(description="VIRDA Stage 3: Electrode Localization")
    parser.add_argument("--project-dir", default="patient_project", help="Project directory")
    parser.add_argument("--measurements", required=True, help="Path to measurements CSV or JSON")
    parser.add_argument("--fiducials", default=None, help="Path to fiducials JSON (if not in project)")
    parser.add_argument("--fiducial-ids", nargs="+", default=None, help="Subset of fiducials")
    parser.add_argument("--max-residual", type=float, default=10.0, help="Flag threshold in mm")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    ese = load_stage2(project_dir)

    fid_path = args.fiducials or str(project_dir / "fiducials" / "fiducials.json")
    fiducial_mgr = FiducialManager.load(fid_path)
    fiducial_ids = args.fiducial_ids or list(fiducial_mgr.get_all_fiducials().keys())

    measurements = load_measurements(args.measurements, fiducial_ids)

    result, qc = run_stage3(
        ese=ese,
        fiducial_mgr=fiducial_mgr,
        measurements=measurements,
        max_residual_threshold=args.max_residual,
        fiducial_ids=args.fiducial_ids,
    )

    exporter.export_localization_csv(result, project_dir / "electrodes.csv")
    exporter.export_localization_json(result, project_dir / "electrodes.json")

    print("\n" + qc.summary())
    return 0 if qc.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
