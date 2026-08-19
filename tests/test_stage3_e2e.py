"""End-to-end integration test using the real mne-sample dataset.

Runs the full VIRDA pipeline (Stage 1 -> 2 -> 3) on the mne-sample NIfTI
with known electrode positions, verifies localization accuracy.

Measurement semantics: per the Stage 3 spec
(docs/task/VIRDA_Stage3_Real_Electrode_Locations.md) the measured distances
are taken from each electrode *center*, which lies on the ESE surface
(scalp offset outward by ``ese_offset_mm`` along the local normal), not
from the scalp contact point itself.  The test therefore synthesizes
measurements from the paired ESE vertices of the true scalp positions
(with small seeded noise) and checks that Stage 3 recovers the original
scalp positions via the ESE->scalp mapping.

Note on coordinate systems: electrodes.tsv stores positions in FreeSurfer
cRAS (volume-centered RAS) while the NIfTI mesh and coordsystem.json
fiducials are in scanner RAS. The test transforms electrode positions to
scanner RAS via the NIfTI affine before computing distances.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from virda.models.ese_mesh import ESEMesh

MNE_DIR = Path(__file__).resolve().parent.parent / "test-data" / "mne-sample"

MEASUREMENT_NOISE_SIGMA_MM = 1.0


def _compute_scanner_ras_offset() -> np.ndarray:
    """Compute the cRAS-to-scanner-RAS offset from the NIfTI affine."""
    import nibabel as nib

    nifti_image = nib.load(str(MNE_DIR / "head.nii.gz"))
    assert isinstance(nifti_image, nib.Nifti1Image)
    affine = np.array(nifti_image.affine)
    shape = np.array(nifti_image.shape)
    center_voxel = (shape[:3] - 1) / 2.0
    return np.array((affine @ np.append(center_voxel, 1.0))[:3], dtype=np.float64)


_SCANNER_RAS_OFFSET: np.ndarray = _compute_scanner_ras_offset()


def _cras_to_scanner_ras(positions: np.ndarray) -> np.ndarray:
    """Transform positions from FreeSurfer cRAS to NIfTI scanner RAS.

    The NIfTI affine maps voxel indices to scanner RAS. FreeSurfer cRAS
    is centered at the volume midpoint, so the offset is the affine
    applied to the center voxel coordinate.
    """
    return positions + _SCANNER_RAS_OFFSET  # type: ignore[no-any-return]


def _load_electrode_positions() -> dict[str, np.ndarray]:
    """Read electrodes.tsv and return {name: cRAS [x, y, z]}."""
    positions: dict[str, np.ndarray] = {}
    with (MNE_DIR / "electrodes.tsv").open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("name"):
                continue
            parts = line.split("\t")
            name = parts[0]
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            positions[name] = coords
    return positions


def _generate_measurements_json(
    tmp_path: Path,
    electrode_positions: dict[str, np.ndarray],
    fiducial_coords: dict[str, np.ndarray],
    ese_mesh: ESEMesh,
) -> Path:
    """Write a measurements JSON with distances from electrode body centers.

    For every true scalp position the paired ESE vertex of the nearest
    scalp vertex is used as the electrode center; Gaussian noise is added
    to the distances to simulate real measurement error.
    """
    import json

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(42)
    scalp_vertices = np.asarray(ese_mesh.scalp_vertices)
    ese_vertices = np.asarray(ese_mesh.vertices)
    true_positions = np.array([_cras_to_scanner_ras(p) for p in electrode_positions.values()])
    _, nearest = cKDTree(scalp_vertices).query(true_positions)
    body_centers = ese_vertices[nearest]

    electrodes = []
    for (name, _), center in zip(electrode_positions.items(), body_centers, strict=True):
        measured = {
            fid: float(
                np.linalg.norm(center - fcoords) + rng.normal(0.0, MEASUREMENT_NOISE_SIGMA_MM)
            )
            for fid, fcoords in fiducial_coords.items()
        }
        electrodes.append({"electrode_id": name, "measured_distances": measured})
    path = tmp_path / "electrodes.json"
    path.write_text(json.dumps({"electrodes": electrodes}, indent=2))
    return path


@pytest.mark.integration
def test_stage3_full_pipeline_with_mne_sample(tmp_path: Path) -> None:
    from virda.config import VirdaSettings, build_config, load_config_file
    from virda.main import run, run_stage3
    from virda.models.coordsystem import Coordsystem
    from virda.models.ese_mesh import ESEMesh

    coordsystem_data = load_config_file(MNE_DIR / "coordsystem.json")
    coordsystem: Coordsystem = coordsystem_data["coordsystem"]
    fiducials = coordsystem.to_fiducials()
    fiducial_coords = {f.fiducial_id: f.coordinates for f in fiducials.items}

    electrode_positions = _load_electrode_positions()
    assert len(electrode_positions) == 60

    config = build_config(
        settings=VirdaSettings(),
        config_files=[MNE_DIR / "coordsystem.json"],
        overrides={
            "nifti_path": str(MNE_DIR / "head.nii.gz"),
            "project_dir": str(tmp_path),
        },
    )

    stage1_result, ese_mesh, _ = run(config)
    assert isinstance(ese_mesh, ESEMesh)

    measurements_path = _generate_measurements_json(
        tmp_path, electrode_positions, fiducial_coords, ese_mesh
    )

    electrodes = run_stage3(config, stage1_result, ese_mesh, measurements_path)

    assert electrodes is not None
    localized = [e for e in electrodes.items if e.is_localized]
    assert len(localized) == 60

    errors_mm: list[float] = []
    for electrode in localized:
        eid = electrode.electrode_id
        assert eid is not None
        true_pos = _cras_to_scanner_ras(electrode_positions[eid])
        error = float(np.linalg.norm(electrode.scalp_coords - true_pos))
        errors_mm.append(error)

    errors = np.array(errors_mm)
    mean_err = float(errors.mean())
    median_err = float(np.median(errors))
    max_err = float(errors.max())
    pct_5mm = float((errors <= 5.0).sum() / len(errors) * 100)
    pct_10mm = float((errors <= 10.0).sum() / len(errors) * 100)
    pct_20mm = float((errors <= 20.0).sum() / len(errors) * 100)

    print(
        f"\n  Localized {len(localized)}/{len(electrodes.items)} electrodes\n"
        f"  Mean error   : {mean_err:.2f} mm\n"
        f"  Median error : {median_err:.2f} mm\n"
        f"  Max error    : {max_err:.2f} mm\n"
        f"  Within 5 mm  : {pct_5mm:.1f}%\n"
        f"  Within 10 mm : {pct_10mm:.1f}%\n"
        f"  Within 20 mm : {pct_20mm:.1f}%"
    )

    stage3_dir = tmp_path / "localization"
    assert (stage3_dir / "electrodes.json").exists()
    assert (stage3_dir / "electrode_coords.csv").exists()
    assert (stage3_dir / "localization_summary.json").exists()

    assert median_err < 4.0
    assert mean_err < 6.0
    assert pct_5mm >= 50.0
    assert pct_10mm >= 90.0
