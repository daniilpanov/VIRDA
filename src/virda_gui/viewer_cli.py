"""Standalone CLI for the interactive 3D viewer.

Usage examples::

    virda-gui-viewer --nifti <scan.nii.gz>
    virda-gui-viewer --mesh <final_mesh.ply>
    virda-gui-viewer --nifti <scan.nii.gz> --mesh <final_mesh.ply> \\
        --fiducials <input/fiducials.json>
    virda-gui-viewer --nifti <scan.nii.gz> --mesh <final_mesh.ply> \\
        --electrodes <electrodes.tsv>
    virda-gui-viewer --nifti <scan.nii.gz> --mesh <final_mesh.ply> \\
        --electrodes <electrodes_scalp.json> --electrodes <electrodes.tsv>

At least one of ``--nifti`` or ``--mesh`` is required.  ``--electrodes``
accepts Stage 3 JSON or a tabular file with ``name``, ``x``, ``y``, ``z``
columns, optionally followed by a color for the whole group.  May be repeated
to overlay multiple electrode groups.  Tabular files are treated as
FreeSurfer cRAS; the frame is detected automatically or forced with
``--electrodes-cras`` (requires ``--nifti``).
"""

import argparse
from collections.abc import Callable

from PySide6.QtWidgets import QApplication, QMainWindow

from virda_gui.viewer import ViewerWidget
from virda_gui.viewer_loaders import parse_electrode_specs


def show_viewer(
    nifti_path: str | None = None,
    mesh_path: str | None = None,
    fiducials_path: str | None = None,
    normals_path: str | None = None,
    downsample_stride: int = 1,
    mesh_opacity: float = 0.6,
    normals_scale: float = 3.0,
    normals_density: int = 500,
    electrode_specs: list[tuple[str, str | None]] | None = None,
    electrodes_cras: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Launch the interactive 3D viewer window.

    Parameters map directly to the CLI flags of ``virda-gui-viewer``.  At
    least one of *nifti_path* or *mesh_path* must be provided.
    *electrode_specs* is a list of ``(path, color)`` pairs as produced by
    :func:`virda_gui.viewer_loaders.parse_electrode_specs`; pass
    ``electrodes_cras=True`` to force the FreeSurfer cRAS -> scanner RAS
    conversion of tabular electrode files.  *log* receives progress and QC
    messages.  The call blocks until the viewer window is closed.
    """
    if not nifti_path and not mesh_path:
        raise ValueError("at least one of nifti_path or mesh_path is required")
    if electrodes_cras and not nifti_path:
        raise ValueError("electrodes_cras requires nifti_path")

    app = QApplication.instance() or QApplication([])

    window = QMainWindow()
    window.setWindowTitle("VIRDA — scalp mesh and/or MRI volume")
    widget = ViewerWidget(log=log)
    widget.sceneFailed.connect(lambda message: log(f"ERROR: 3D viewer failed: {message}"))
    window.setCentralWidget(widget)
    window.resize(960, 720)
    window.show()

    widget.load(
        nifti_path=nifti_path,
        mesh_path=mesh_path,
        fiducials_path=fiducials_path,
        normals_path=normals_path,
        downsample_stride=downsample_stride,
        mesh_opacity=mesh_opacity,
        normals_scale=normals_scale,
        normals_density=normals_density,
        electrode_specs=electrode_specs,
        electrodes_cras=electrodes_cras,
    )

    if QApplication.instance() is app:
        app.exec()


def main() -> None:
    """CLI entry point for ``virda-gui-viewer``."""
    parser = argparse.ArgumentParser(
        prog="virda-gui-viewer",
        description="Interactive 3D viewer: scalp mesh and/or MRI volume.",
    )
    parser.add_argument("--nifti", help="Path to the T1-weighted NIfTI scan.")
    parser.add_argument("--mesh", help="Path to the scalp mesh (PLY).")
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Voxel stride for volume downsampling (1 = full resolution).",
    )
    parser.add_argument(
        "--mesh-opacity", type=float, default=0.6, help="Scalp mesh opacity (0..1)."
    )
    parser.add_argument(
        "--fiducials",
        help="Path to fiducials JSON (input/fiducials.json).",
    )
    parser.add_argument(
        "--normals",
        help="Path to normals file (normals.npy) for visualisation.",
    )
    parser.add_argument(
        "--normals-scale",
        type=float,
        default=3.0,
        help="Visual length of normal arrows in scene units (default: 3.0).",
    )
    parser.add_argument(
        "--normals-density",
        type=int,
        default=500,
        help="Show one normal per N vertices (default: 500).",
    )
    parser.add_argument(
        "--electrodes",
        action="append",
        nargs="*",
        default=[],
        metavar="FILE [COLOR]",
        help=(
            "Electrodes file: Stage 3 JSON (.json) "
            "or tabular with name/x/y/z columns (.tsv/.csv), "
            "optionally followed by a color for the whole group, e.g. "
            "--electrodes electrodes.tsv yellow. May be repeated to overlay "
            "multiple groups (without a color each group gets the next "
            "palette entry)."
        ),
    )
    parser.add_argument(
        "--electrodes-cras",
        action="store_true",
        help=(
            "Force cRAS -> scanner RAS conversion of tabular electrode files "
            "(.tsv/.csv) using the --nifti affine. Without the flag the frame "
            "is detected automatically when a --mesh is supplied (the frame "
            "whose points sit closer to the mesh wins). Stage 3 JSON output "
            "is always already in scanner RAS."
        ),
    )
    args = parser.parse_args()

    if not args.nifti and not args.mesh:
        parser.error("at least one of --nifti or --mesh is required")
    if args.electrodes_cras and not args.nifti:
        parser.error("--electrodes-cras requires --nifti")

    try:
        specs = parse_electrode_specs(args.electrodes)
    except ValueError as exc:
        parser.error(str(exc))

    show_viewer(
        nifti_path=args.nifti,
        mesh_path=args.mesh,
        fiducials_path=args.fiducials,
        normals_path=args.normals,
        downsample_stride=args.downsample,
        mesh_opacity=args.mesh_opacity,
        normals_scale=args.normals_scale,
        normals_density=args.normals_density,
        electrode_specs=specs,
        electrodes_cras=args.electrodes_cras,
    )


if __name__ == "__main__":
    main()
