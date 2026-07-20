#!/usr/bin/env python
"""Run full VIRDA pipeline: Stage 1 → Stage 2 → Stage 3."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.stage1 import run_stage1
from api.stage2 import run_stage2
from api.stage3 import run_stage3, load_measurements
from api import exporter


def main():
    parser = argparse.ArgumentParser(description="VIRDA Full Pipeline")
    parser.add_argument("mri_path", help="Path to MRI file or DICOM directory")
    parser.add_argument("--project-dir", default="patient_project", help="Output project directory")
    parser.add_argument("--measurements", required=True, help="Path to measurements CSV or JSON")
    parser.add_argument("--fiducials", default=None, help="Path to fiducials JSON")
    parser.add_argument("--fiducial-ids", nargs="+", default=None)
    parser.add_argument("--segmentation-method", default="threshold")
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--close-radius", type=int, default=3)
    parser.add_argument("--min-component", type=int, default=10000)
    parser.add_argument("--mesh-smooth-iterations", type=int, default=0)
    parser.add_argument("--ese-offset", type=float, default=5.0)
    parser.add_argument("--radius", type=float, default=10.0, help="PCA neighborhood radius mm")
    parser.add_argument("--max-residual", type=float, default=10.0)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)

    print("=== Stage 1 ===")
    mesh, fiducial_mgr, ese_config, segmentation, mri, qc1 = run_stage1(
        mri_path=args.mri_path,
        segmentation_method=args.segmentation_method,
        smooth_sigma=args.smooth_sigma,
        close_radius=args.close_radius,
        min_component_size=args.min_component,
        mesh_smooth_iterations=args.mesh_smooth_iterations,
        ese_offset_mm=args.ese_offset,
    )

    exporter.export_mesh_ply(mesh, project_dir / "mesh" / "scalp.ply")
    exporter.export_vertices_csv(mesh, project_dir / "mesh" / "vertices.csv")
    exporter.export_faces_csv(mesh, project_dir / "mesh" / "faces.csv")
    fiducial_mgr.save(project_dir / "fiducials" / "fiducials.json")
    ese_config.save(project_dir / "config" / "parameters.json")
    exporter.export_segmentation_nifti(
        segmentation.mask, mri.affine, project_dir / "segmentation" / "head_segmentation.nii.gz"
    )

    print("\n=== Stage 2 ===")
    ese, normal_result, qc2 = run_stage2(
        mesh,
        radius_mm=args.radius,
        ese_config=ese_config,
    )

    exporter.export_ese_pairs(ese, project_dir / "mesh" / "ese_pairs.csv")
    exporter.export_normals_csv(normal_result, mesh, project_dir / "mesh" / "normals.csv")

    print("\n=== Stage 3 ===")
    fiducial_ids = args.fiducial_ids or list(fiducial_mgr.get_all_fiducials().keys())
    measurements = load_measurements(args.measurements, fiducial_ids)

    result, qc3 = run_stage3(
        ese=ese,
        fiducial_mgr=fiducial_mgr,
        measurements=measurements,
        max_residual_threshold=args.max_residual,
        fiducial_ids=args.fiducial_ids,
    )

    exporter.export_localization_csv(result, project_dir / "electrodes.csv")
    exporter.export_localization_json(result, project_dir / "electrodes.json")

    all_passed = qc1.all_passed and qc2.all_passed and qc3.all_passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
