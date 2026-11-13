"""Unit tests for :mod:`virda_gui_new.services.artifact_saving`.

Each saver writes tiny real files into ``tmp_path`` (NIfTI via nibabel,
PLY via trimesh) and the written files are read back or checked on disk.
The localization savers write the virda exporter formats, so the CSV
point files are read back through the virda point importer.
"""

import json
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import trimesh

from virda.io.importers.electrode_points import import_electrode_points
from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask
from virda_gui_new.api.ops import ScalpSurface
from virda_gui_new.services.artifact_loading import load_fiducial_points
from virda_gui_new.services.artifact_saving import (
    save_ese_mesh,
    save_final_mesh,
    save_localization,
    save_scalp_surface,
    save_volume_copy,
)

_VOLUME = MRIVolume(
    np.zeros((4, 4, 4), dtype="float32"), np.eye(4), (1.0, 1.0, 1.0), ("R", "A", "S")
)

_FIDUCIAL_ROWS = [
    ("NAS", cast("Any", np.array([0.0, 0.0, 1.0]))),
    ("LPA", cast("Any", np.array([1.0, 0.0, 0.0]))),
    ("RPA", cast("Any", np.array([-1.0, 0.0, 0.0]))),
]
_MEASUREMENTS = {"Fz": {"NAS": 120.5, "LPA": 131.2}}


def test_save_scalp_surface_writes_mask_and_base_mesh(tmp_path: Path) -> None:
    surface = ScalpSurface(
        mask=SegmentationMask(np.ones((3, 3, 3), dtype="uint8")),
        mesh=Mesh(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0, 1, 2]]),
        ),
    )

    save_scalp_surface(tmp_path, _VOLUME, surface)

    mask_path = tmp_path / "segmentation" / "head_mask.nii.gz"
    assert mask_path.is_file()
    image = nib.load(mask_path)
    assert cast("Any", image).get_fdata().sum() == 27
    assert np.asarray(cast("Any", image).dataobj).dtype == np.uint8
    assert np.allclose(cast("Any", image).affine, np.eye(4))

    mesh_path = tmp_path / "scalp" / "base_mesh.ply"
    assert mesh_path.is_file()
    assert len(cast("Any", trimesh.load(mesh_path)).vertices) == 3


def test_save_final_mesh_writes_project_ply(tmp_path: Path) -> None:
    save_final_mesh(tmp_path, Mesh(np.eye(4, 3), np.array([[0, 1, 2]])))

    assert (tmp_path / "scalp" / "final_mesh.ply").is_file()


def test_save_volume_copy_keeps_the_source_in_the_project(tmp_path: Path) -> None:
    """The copy makes the project self-contained: the scanner finds it."""
    source = tmp_path / "elsewhere" / "head.nii.gz"
    source.parent.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), "float32"), np.eye(4)), source)

    written = save_volume_copy(tmp_path, source)

    assert written == tmp_path / "input" / "head.nii.gz"
    assert written is not None and written.is_file()


def test_save_volume_copy_of_a_project_file_keeps_it(tmp_path: Path) -> None:
    """Copying a file onto itself leaves the original in place."""
    source = tmp_path / "input" / "head.nii.gz"
    source.parent.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), "float32"), np.eye(4)), source)

    written = save_volume_copy(tmp_path, source)

    assert written == source
    assert source.is_file()


def test_save_ese_mesh_writes_project_ply(tmp_path: Path) -> None:
    save_ese_mesh(tmp_path, Mesh(np.eye(4, 3), np.array([[0, 1, 2]])))

    assert (tmp_path / "ese" / "mesh.ply").is_file()


def test_save_localization_writes_all_files(tmp_path: Path) -> None:
    electrodes = Electrodes(
        [
            Electrode(
                "A",
                {"NAS": 1.0},
                ese_coords=cast("Any", np.array([1.0, 2.0, 3.0])),
                scalp_coords=cast("Any", np.array([4.0, 5.0, 6.0])),
            ),
            Electrode(
                "B",
                {"LPA": 2.0},
                ese_coords=cast("Any", np.array([7.0, 8.0, 9.0])),
                scalp_coords=None,
            ),
        ]
    )

    save_localization(tmp_path, _FIDUCIAL_ROWS, _MEASUREMENTS, electrodes)

    fiducials = load_fiducial_points(tmp_path / "localization" / "fiducials.json")
    assert fiducials.labels == ("NAS", "LPA", "RPA")

    document = json.loads(
        (tmp_path / "localization" / "measurements.json").read_text(encoding="utf-8")
    )
    assert document == {
        "electrodes": [{"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}}]
    }

    scalp_path = tmp_path / "localization" / "electrodes_scalp.csv"
    with scalp_path.open(newline="", encoding="utf-8") as fh:
        header = fh.readline().strip()
    # The exact batch pipeline header of export_electrodes_scalp.
    assert header == "electrode_id,x,y,z,residual_mm,flagged"
    scalp = import_electrode_points(scalp_path)
    assert [point.electrode_id for point in scalp] == ["A"]
    assert scalp[0].coordinates.tolist() == [4.0, 5.0, 6.0]

    ese = import_electrode_points(tmp_path / "localization" / "electrodes_ese.csv")
    assert [point.electrode_id for point in ese] == ["A", "B"]
    assert [point.coordinates.tolist() for point in ese] == [
        [1.0, 2.0, 3.0],
        [7.0, 8.0, 9.0],
    ]


def test_save_localization_skips_files_without_coordinates(tmp_path: Path) -> None:
    electrodes = Electrodes([Electrode("A", {"NAS": 1.0})])

    save_localization(tmp_path, _FIDUCIAL_ROWS, _MEASUREMENTS, electrodes)

    assert (tmp_path / "localization" / "fiducials.json").is_file()
    assert (tmp_path / "localization" / "measurements.json").is_file()
    assert not (tmp_path / "localization" / "electrodes_scalp.csv").exists()
    assert not (tmp_path / "localization" / "electrodes_ese.csv").exists()


def test_save_localization_rejects_invalid_fiducial_rows(tmp_path: Path) -> None:
    electrodes = Electrodes([])

    try:
        save_localization(tmp_path, [], _MEASUREMENTS, electrodes)
    except Exception as error:  # noqa: BLE001 - the popup surfaces any error
        assert "fiducial" in str(error).lower()
    else:
        raise AssertionError("expected empty fiducial rows to fail")
