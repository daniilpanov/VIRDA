#!/usr/bin/env python
"""Run VIRDA Stage 1: MRI loading, segmentation, mesh generation."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.stage1 import run_stage1
from api import exporter


def main():
    parser = argparse.ArgumentParser(description="VIRDA Stage 1: Head Surface Mesh Generation")
    parser.add_argument("mri_path", help="Path to MRI file (NIfTI) or DICOM directory")
    parser.add_argument("--project-dir", default="patient_project", help="Output project directory")
    parser.add_argument("--segmentation-method", default="threshold", choices=["threshold", "region_grow"])
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--close-radius", type=int, default=3)
    parser.add_argument("--min-component", type=int, default=10000)
    parser.add_argument("--mesh-smooth-iterations", type=int, default=0)
    parser.add_argument("--mesh-smooth-lambda", type=float, default=0.5)
    parser.add_argument("--ese-offset", type=float, default=5.0, help="ESE offset in mm")
    args = parser.parse_args()

    mesh, fiducial_mgr, ese_config, segmentation, mri, qc = run_stage1(
        mri_path=args.mri_path,
        segmentation_method=args.segmentation_method,
        smooth_sigma=args.smooth_sigma,
        close_radius=args.close_radius,
        min_component_size=args.min_component,
        mesh_smooth_iterations=args.mesh_smooth_iterations,
        mesh_smooth_lambda=args.mesh_smooth_lambda,
        ese_offset_mm=args.ese_offset,
    )

    project_dir = Path(args.project_dir)
    exporter.export_mesh_ply(mesh, project_dir / "mesh" / "scalp.ply")
    exporter.export_vertices_csv(mesh, project_dir / "mesh" / "vertices.csv")
    exporter.export_faces_csv(mesh, project_dir / "mesh" / "faces.csv")
    exporter.export_normals_csv(
        None, mesh, project_dir / "mesh" / "normals.csv"
    ) if hasattr(mesh, 'vertex_normals') else None
    fiducial_mgr.save(project_dir / "fiducials" / "fiducials.json")
    ese_config.save(project_dir / "config" / "parameters.json")
    exporter.export_segmentation_nifti(
        segmentation.mask, mri.affine, project_dir / "segmentation" / "head_segmentation.nii.gz"
    )

    print("\n" + qc.summary())
    return 0 if qc.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
