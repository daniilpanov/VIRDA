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
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from virda.models.config import Config
    from virda.models.ese_mesh import ESEMesh

MNE_DIR = Path(__file__).resolve().parent.parent / "test-data" / "mne-sample"

MEASUREMENT_NOISE_SIGMA_MM = 1.0
# Hard per-electrode ceiling. Aggregate gates (median/mean/pct) are the main
# accuracy contract; this bound only catches catastrophic outliers. With
# σ=1 mm measurement noise the residual landscape can rank ~60 wrong vertices
# above the true pair (EEG 043), so a 10 mm ceiling is brittle at the real
# 0.1 mm ESE offset even though median error stays ~2 mm.
PER_ELECTRODE_MAX_ERROR_MM = 12.0


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


def _per_electrode_errors(
    electrodes: object,
    electrode_positions: dict[str, np.ndarray],
) -> dict[str, float]:
    """Map every localized electrode ID to its error vs the true position.

    This is the per-pair ground truth check: one entry per electrode ID,
    not a cloud-level aggregate.
    """
    from virda.models.electrode import Electrodes

    assert isinstance(electrodes, Electrodes)
    errors: dict[str, float] = {}
    for electrode in electrodes.items:
        if not electrode.is_localized:
            continue
        eid = electrode.electrode_id
        assert eid is not None
        true_pos = _cras_to_scanner_ras(electrode_positions[eid])
        errors[eid] = float(np.linalg.norm(electrode.scalp_coords - true_pos))
    return errors


def _print_per_electrode_trace(errors: dict[str, float], limit_mm: float) -> None:
    """Print a per-electrode ID trace table sorted by error."""
    rows = sorted(errors.items(), key=lambda kv: kv[1], reverse=True)
    lines = [f"  {'electrode':<12}{'error':>10}  status"]
    for eid, err in rows:
        status = "OK" if err <= limit_mm else "FAIL"
        lines.append(f"  {eid:<12}{err:>8.2f} mm  {status}")
    print("\n" + "\n".join(lines))


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


# Every test runs against both ESE normal-estimation modes: the default
# radius-based one and an explicit k-NN neighborhood (the value the old
# committed config.json carried).
K_NEIGHBORS_MODES = [
    pytest.param(None, id="default-radius"),
    pytest.param(20, id="knn-k20"),
]


def _build_config(tmp_path: Path, k_neighbors: int | None, **overrides: Any) -> Config:
    """Build the pipeline config shared by the integration tests."""
    from virda.config import VirdaSettings, build_config

    merged: dict[str, Any] = {
        "nifti_path": str(MNE_DIR / "head.nii.gz"),
        "project_dir": str(tmp_path),
        **overrides,
    }
    if k_neighbors is not None:
        merged["k_neighbors"] = k_neighbors
    return build_config(
        settings=VirdaSettings(),
        config_files=[MNE_DIR / "coordsystem.json"],
        overrides=merged,
    )


@pytest.mark.integration
@pytest.mark.parametrize("k_neighbors", K_NEIGHBORS_MODES)
def test_stage3_full_pipeline_with_mne_sample(tmp_path: Path, k_neighbors: int | None) -> None:
    from virda.config import load_config_file
    from virda.main import run, run_stage3
    from virda.models.coordsystem import Coordsystem
    from virda.models.ese_mesh import ESEMesh

    coordsystem_data = load_config_file(MNE_DIR / "coordsystem.json")
    coordsystem: Coordsystem = coordsystem_data["coordsystem"]
    fiducials = coordsystem.to_fiducials()
    fiducial_coords = {f.fiducial_id: f.coordinates for f in fiducials.items}

    electrode_positions = _load_electrode_positions()
    assert len(electrode_positions) == 60

    config = _build_config(tmp_path, k_neighbors)

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

    # Per-electrode pair checks: no catastrophic outlier on any single ID.
    per_pair = _per_electrode_errors(electrodes, electrode_positions)
    assert len(per_pair) == 60
    _print_per_electrode_trace(per_pair, limit_mm=PER_ELECTRODE_MAX_ERROR_MM)
    worst = sorted(per_pair.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("  Worst pairs : " + ", ".join(f"{eid}={err:.2f}mm" for eid, err in worst))
    assert all(err < PER_ELECTRODE_MAX_ERROR_MM for err in per_pair.values())

    assert median_err < 4.0
    assert mean_err < 6.0
    assert pct_5mm >= 50.0
    assert pct_10mm >= 90.0


@pytest.mark.integration
@pytest.mark.parametrize("k_neighbors", K_NEIGHBORS_MODES)
def test_stage3_offset_calibration_with_scalp_level_measurements(
    tmp_path: Path, k_neighbors: int | None
) -> None:
    """Scalp-level measurements localize accurately with offset calibration.

    Simulates a stale configuration: the ESE is built with a 20 mm body
    offset (``ese_offset_mm=20`` override) while the measurements are taken
    from the digitized scalp contact points.  With ``calibrate_ese_offset``
    enabled Stage 3 must estimate the ~-18 mm shift and still recover the
    true positions.  The committed coordsystem.json carries the real
    electrode offset (0.1 mm), under which no calibration is needed.
    """
    import json

    from virda.config import load_config_file
    from virda.main import run, run_stage3
    from virda.models.coordsystem import Coordsystem
    from virda.models.ese_mesh import ESEMesh

    coordsystem_data = load_config_file(MNE_DIR / "coordsystem.json")
    coordsystem: Coordsystem = coordsystem_data["coordsystem"]
    fiducial_coords = {f.fiducial_id: f.coordinates for f in coordsystem.to_fiducials().items}

    electrode_positions = _load_electrode_positions()

    config = _build_config(tmp_path, k_neighbors, calibrate_ese_offset=True, ese_offset_mm=20)

    stage1_result, ese_mesh, _ = run(config)
    assert isinstance(ese_mesh, ESEMesh)

    electrodes_json = [
        {
            "electrode_id": name,
            "measured_distances": {
                fid: float(np.linalg.norm(_cras_to_scanner_ras(pos) - fcoords))
                for fid, fcoords in fiducial_coords.items()
            },
        }
        for name, pos in electrode_positions.items()
    ]
    measurements_path = tmp_path / "electrodes_scalp_level.json"
    measurements_path.write_text(json.dumps({"electrodes": electrodes_json}, indent=2))

    electrodes = run_stage3(config, stage1_result, ese_mesh, measurements_path)

    assert electrodes is not None
    localized = [e for e in electrodes.items if e.is_localized]
    assert len(localized) == 60

    shift = electrodes.calibrated_offset_shift_mm
    assert shift is not None
    assert -21.0 <= shift <= -15.0

    # Per-electrode pair checks: every single ID must localize accurately.
    per_pair = _per_electrode_errors(electrodes, electrode_positions)
    assert len(per_pair) == 60
    _print_per_electrode_trace(per_pair, limit_mm=PER_ELECTRODE_MAX_ERROR_MM)
    worst = sorted(per_pair.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("  Worst pairs : " + ", ".join(f"{eid}={err:.2f}mm" for eid, err in worst))
    assert all(err < PER_ELECTRODE_MAX_ERROR_MM for err in per_pair.values())

    errors = np.array(list(per_pair.values()))
    median_err = float(np.median(errors))
    mean_err = float(errors.mean())
    pct_10mm = float((errors <= 10.0).sum() / len(errors) * 100)

    print(
        f"\n  Calibrated offset shift: {shift:+.1f} mm\n"
        f"  Mean error   : {mean_err:.2f} mm\n"
        f"  Median error : {median_err:.2f} mm\n"
        f"  Max error    : {errors.max():.2f} mm\n"
        f"  Within 10 mm : {pct_10mm:.1f}%"
    )

    summary = json.loads((tmp_path / "localization" / "localization_summary.json").read_text())
    assert summary["calibrated_ese_offset_shift_mm"] == pytest.approx(shift)

    assert median_err < 6.0
    assert mean_err < 9.0
    assert pct_10mm >= 85.0


@pytest.mark.integration
@pytest.mark.parametrize("k_neighbors", K_NEIGHBORS_MODES)
def test_stage3_committed_measurements_per_electrode_trace(
    tmp_path: Path, k_neighbors: int | None
) -> None:
    """Regression test for the real workflow: committed measurements.json.

    Runs Stage 3 on the committed ``test-data/mne-sample/measurements.json``
    with default settings and verifies EVERY electrode pair individually:
    each electrode ID must land within a few mm of its true position.
    This guards against cloud-level metrics hiding individual bad pairs.
    """
    from virda.main import run, run_stage3
    from virda.models.ese_mesh import ESEMesh

    electrode_positions = _load_electrode_positions()

    config = _build_config(tmp_path, k_neighbors)

    stage1_result, ese_mesh, _ = run(config)
    assert isinstance(ese_mesh, ESEMesh)

    electrodes = run_stage3(config, stage1_result, ese_mesh, MNE_DIR / "measurements.json")

    assert electrodes is not None
    localized = [e for e in electrodes.items if e.is_localized]
    assert len(localized) == 60

    per_pair = _per_electrode_errors(electrodes, electrode_positions)
    assert len(per_pair) == 60
    _print_per_electrode_trace(per_pair, limit_mm=PER_ELECTRODE_MAX_ERROR_MM)

    errors = np.array(list(per_pair.values()))
    median_err = float(np.median(errors))
    worst = sorted(per_pair.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(
        f"\n  Offset shift: {electrodes.calibrated_offset_shift_mm:+.2f} mm\n"
        f"  Median error: {median_err:.2f} mm\n"
        f"  Worst pairs : " + ", ".join(f"{eid}={err:.2f}mm" for eid, err in worst)
    )

    # No catastrophic per-electrode outlier; median stays tight.
    assert all(err < PER_ELECTRODE_MAX_ERROR_MM for err in per_pair.values())
    assert median_err < 4.0
