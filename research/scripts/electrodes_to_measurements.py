#!/usr/bin/env python3
"""Convert electrodes.tsv + coordsystem.json to a Stage 3 measurements JSON.

Reads electrode positions and fiducial coordinates and computes distances
from each electrode to each fiducial.  The output JSON can be passed to
``virda --measurements-path``.

Usage:
    python electrodes_to_measurements.py <project_dir> [--output measurements.json]

The project_dir must contain:
    - electrodes.tsv   (columns: name, x, y, z in FreeSurfer cRAS mm)
    - coordsystem.json (MNE-style with FiducialsCoordinates in MRI-RAS)
    - head.nii.gz      (used for the cRAS -> scanner RAS conversion;
                       override with --nifti)

Coordinate frames: electrodes.tsv stores FreeSurfer cRAS (volume-centered
RAS) while the coordsystem fiducials ("MRI") are in scanner RAS.  The
electrode positions are converted to scanner RAS via the NIfTI affine
before computing distances.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_KEY_MAP = {"NASION": "NAS", "LPA": "LPA", "RPA": "RPA"}


def _load_electrode_positions(electrodes_tsv: Path) -> dict[str, np.ndarray]:
    """Read electrodes.tsv and return {name: cRAS [x, y, z]}."""
    positions: dict[str, np.ndarray] = {}
    with electrodes_tsv.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("name"):
                continue
            parts = line.split("\t")
            name = parts[0]
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            positions[name] = coords
    return positions


def _cras_to_scanner_ras_offset(nifti_path: Path) -> np.ndarray:
    """Compute the cRAS-to-scanner-RAS offset from the NIfTI affine.

    FreeSurfer cRAS is centered at the volume midpoint, so the offset is
    the affine applied to the center voxel coordinate.
    """
    import nibabel as nib

    img = nib.load(str(nifti_path))
    affine = np.array(img.affine, dtype=np.float64)
    shape = np.array(img.shape)
    center_voxel = (shape[:3] - 1) / 2.0
    return np.array((affine @ np.append(center_voxel, 1.0))[:3], dtype=np.float64)


def _load_fiducial_coords(coordsystem_json: Path) -> dict[str, np.ndarray]:
    """Read coordsystem.json and return {fiducial_id: MRI-RAS coords}.

    The file stores fiducials as ``{"NASION": {"Head": [...], "MRI": [...]}}``.
    MRI coordinates are already in scanner RAS (the world space of the scalp
    mesh), so no cRAS offset is needed for them.

    Keys are normalised to the short IDs used by the pipeline
    (``NAS``, ``LPA``, ``RPA``).
    """
    data = json.loads(coordsystem_json.read_text(encoding="utf-8"))
    coordsystem = data.get("coordsystem", data)
    fiducials_raw = coordsystem.get("FiducialsCoordinates", {})
    result: dict[str, np.ndarray] = {}
    for fid, coord in fiducials_raw.items():
        mri = coord.get("MRI", coord.get("Head")) if isinstance(coord, dict) else coord
        if mri is None:
            continue
        key = _KEY_MAP.get(fid, fid)
        result[key] = np.array(mri, dtype=np.float64)
    return result


def _build_measurements(
    electrode_positions: dict[str, np.ndarray],
    fiducial_coords: dict[str, np.ndarray],
) -> dict:
    """Build the measurements JSON structure.

    Electrode positions and fiducial coordinates are expected in the same
    coordinate frame (scanner RAS).
    """
    electrodes = []
    for name, pos in electrode_positions.items():
        measured = {
            fid: float(np.linalg.norm(pos - fcoords)) for fid, fcoords in fiducial_coords.items()
        }
        electrodes.append({"electrode_id": name, "measured_distances": measured})
    return {"electrodes": electrodes}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert electrodes.tsv to a Stage 3 measurements JSON.",
    )
    parser.add_argument("project_dir", help="Path to the project directory.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: <project_dir>/measurements.json).",
    )
    parser.add_argument(
        "--nifti",
        default=None,
        help="NIfTI used for the cRAS -> scanner RAS conversion "
        "(default: <project_dir>/head.nii.gz).",
    )
    args = parser.parse_args(argv)

    project = Path(args.project_dir)
    electrodes_tsv = project / "electrodes.tsv"
    coordsystem_json = project / "coordsystem.json"

    for path in (electrodes_tsv, coordsystem_json):
        if not path.is_file():
            print(f"Error: missing {path}", file=sys.stderr)
            sys.exit(1)

    nifti_path = Path(args.nifti) if args.nifti else project / "head.nii.gz"
    if not nifti_path.is_file():
        print(
            f"Error: missing {nifti_path} (required for the cRAS -> scanner RAS conversion)",
            file=sys.stderr,
        )
        sys.exit(1)

    offset = _cras_to_scanner_ras_offset(nifti_path)
    electrode_positions = _load_electrode_positions(electrodes_tsv)
    electrode_positions = {name: pos + offset for name, pos in electrode_positions.items()}
    fiducial_coords = _load_fiducial_coords(coordsystem_json)

    measurements = _build_measurements(electrode_positions, fiducial_coords)

    output_path = Path(args.output) if args.output else project / "measurements.json"
    output_path.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    print(f"Wrote {len(measurements['electrodes'])} electrodes -> {output_path}")


if __name__ == "__main__":
    main()
