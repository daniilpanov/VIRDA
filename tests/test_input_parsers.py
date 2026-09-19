"""Tests for the :mod:`virda.input_parsers` file-path parsers and shims."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from tests.helpers.measurements import make_measurements_file
from tests.helpers.pipelines import save_test_fiducials
from virda.input_parsers import (
    parse_config,
    parse_coordsystem,
    parse_electrodes_table,
    parse_fiducials,
    parse_measurements,
    parse_mri_volume,
)
from virda.models.config import Config
from virda.models.coordsystem import Coordsystem
from virda.models.electrode import Electrodes
from virda.models.fiducial import Fiducials
from virda.models.mri_volume import MRIVolume


def test_parse_fiducials(tmp_path: Path) -> None:
    path = save_test_fiducials(tmp_path / "fiducials.json")

    result = parse_fiducials(path)

    assert isinstance(result, Fiducials)
    assert result.ids == ["NAS", "LPA"]


def test_parse_coordsystem(tmp_path: Path) -> None:
    path = tmp_path / "coordsystem.json"
    path.write_text(
        json.dumps(
            {
                "CoordinateSystem": "RAS",
                "CoordinateUnits": "mm",
                "FiducialsCoordinates": {
                    "NASION": {"Head": [0.0, 10.0, 0.0], "MRI": [0.0, 90.0, 0.0]}
                },
                "ElectrodeOffset": 2.5,
            }
        )
    )

    result = parse_coordsystem(path)

    assert isinstance(result, Coordsystem)
    assert result.coordinate_system == "RAS"
    assert result.electrode_offset_mm == 2.5


def test_parse_measurements(tmp_path: Path) -> None:
    path = make_measurements_file(tmp_path / "measurements.json", points=np.zeros((2, 3)))

    result = parse_measurements(path)

    assert isinstance(result, Electrodes)
    assert [electrode.electrode_id for electrode in result.items] == ["E0", "E1"]


def test_parse_mri_volume(tmp_path: Path) -> None:
    path = tmp_path / "tiny_volume.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), path)

    result = parse_mri_volume(path)

    assert isinstance(result, MRIVolume)
    assert result.data.shape == (4, 4, 4)
    assert result.spacing == (1.0, 1.0, 1.0)


def test_parse_electrodes_table(tmp_path: Path) -> None:
    path = tmp_path / "electrodes.tsv"
    path.write_text(
        "name\tx\ty\tz\n"
        "Fz\t1.0\t2.0\t3.0\n"
        "Cz\t4.0\t5.0\t6.0\n"
    )

    positions, residuals, flags, measured, names = parse_electrodes_table(path)

    np.testing.assert_allclose(positions, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert residuals.shape == (2,)
    assert flags.shape == (2,)
    assert measured == [{}, {}]
    assert names == ["Fz", "Cz"]


def test_parse_config(tmp_path: Path) -> None:
    path = tmp_path / "pipeline_config.json"
    path.write_text(json.dumps({"ese_offset_mm": 2.5, "mesh_density_percent": 40}))

    result = parse_config(path)

    assert isinstance(result, Config)
    assert result.ese_offset_mm == 2.5
    assert result.mesh_density_percent == 40.0


def test_refactor_shims_keep_import_paths() -> None:
    from virda.config import VirdaSettings
    from virda.main import (
        _ESE_PARAMS,
        _validate_required_group,
        _warn_partial_neighborhood,
        main,
        run,
        run_stage3,
    )
    from virda_cli.main import main as cli_main

    assert callable(run)
    assert callable(run_stage3)
    assert callable(main)
    assert callable(cli_main)
    assert callable(_validate_required_group)
    assert callable(_warn_partial_neighborhood)
    assert _ESE_PARAMS == ("ese_offset_mm",)
    assert VirdaSettings is not None