"""Unit tests for :mod:`virda_gui_new.services.project_import`."""

import json
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import trimesh

from virda.io.fiducial_helpers import save_fiducials
from virda.models.electrode import Electrodes
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.project_import import prepare_project


def _write_nifti(path: Path) -> None:
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), "float32"), np.eye(4)), path)


def _write_ply(path: Path) -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 1, 3], [1, 2, 3]]).export(path)


def _write_fiducials(path: Path) -> None:
    fiducials = Fiducials(items=[Fiducial("c1", "nas", np.array([0.0, 1.0, 2.0]), "world")])
    save_fiducials(path, fiducials)


def _write_coordsystem(path: Path) -> None:
    """MNE-style coordsystem.json: millimetre MRI fiducials."""
    path.write_text(
        json.dumps(
            {
                "CoordinateSystem": "MRI",
                "CoordinateUnits": "mm",
                "FiducialsCoordinates": {"NAS": {"Head": [1.0, 2.0, 3.0], "MRI": [1.0, 2.0, 3.0]}},
            }
        ),
        encoding="utf-8",
    )


def _write_measurements(path: Path) -> None:
    path.write_text(
        json.dumps({"electrodes": [{"electrode_id": "Fz", "measured_distances": {"NAS": 120.5}}]}),
        encoding="utf-8",
    )


def _write_outdated_point_rows(path: Path) -> None:
    """Write the outdated electrode format (electrode_id / coords keys)."""
    path.write_text(
        json.dumps(
            [
                {
                    "electrode_id": "EEG 002",
                    "coords": [-20.5, 76.6, 62.7],
                    "flagged": True,
                }
            ]
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "input").mkdir()
    (tmp_path / "mesh").mkdir()
    (tmp_path / "ese").mkdir()
    (tmp_path / "localization").mkdir()
    return tmp_path


def test_prepare_project_loads_valid_artifacts(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_ply(project / "ese" / "ese_mesh.ply")
    _write_fiducials(project / "input" / "fiducials.json")

    prepared = prepare_project(project)

    assert prepared.failures == ()
    assert isinstance(prepared.loaded.nifti, MRIVolume)
    assert isinstance(prepared.loaded.scalp_mesh, Mesh)
    assert isinstance(prepared.loaded.ese_mesh, Mesh)
    assert isinstance(prepared.loaded.fiducials, LabeledPoints)
    assert prepared.loaded.fiducials.labels == ("nas",)
    assert prepared.recoverable_step == 4


def test_prepare_project_rejects_outdated_point_format(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_ply(project / "ese" / "ese_mesh.ply")
    _write_outdated_point_rows(project / "localization" / "electrodes_scalp.json")

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert "electrodes_scalp.json" in prepared.failures[0]
    assert "electrode points" in prepared.failures[0]
    assert prepared.loaded.scalp_points is None
    # Everything else still seeds; recovery follows the loaded artifacts.
    assert prepared.loaded.nifti is not None
    assert prepared.loaded.scalp_mesh is not None
    assert prepared.recoverable_step == 3


def test_prepare_project_rejects_empty_point_file(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    (project / "localization" / "electrodes_scalp.json").write_text("[]", encoding="utf-8")

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert prepared.loaded.scalp_points is None
    assert prepared.recoverable_step == 0


def test_prepare_project_rejects_non_array_point_file(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    (project / "localization" / "electrodes_scalp.json").write_text("{}", encoding="utf-8")

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert prepared.loaded.scalp_points is None


def test_prepare_project_empty_directory(project: Path) -> None:
    prepared = prepare_project(project)

    assert prepared.failures == ()
    assert prepared.loaded == type(prepared.loaded)()
    assert prepared.recoverable_step == -1


def test_prepare_project_loads_measurements_and_base_mesh(project: Path) -> None:
    base = project / "scalp"
    base.mkdir(exist_ok=True)
    _write_ply(base / "base_mesh.ply")
    _write_measurements(project / "input" / "measurements.json")
    (project / "input" / "pipeline_config.json").write_text("{}", encoding="utf-8")

    prepared = prepare_project(project)

    assert prepared.failures == ()
    assert isinstance(prepared.loaded.base_mesh, Mesh)
    assert isinstance(prepared.loaded.measurements, Electrodes)
    assert prepared.loaded.measurements.items[0].electrode_id == "Fz"
    # pipeline_config stays a scanned path: nothing loads it yet.
    assert prepared.loaded.pipeline_config == project / "input" / "pipeline_config.json"
    assert prepared.recoverable_step == 1


def test_prepare_project_converts_coordsystem_to_fiducials(project: Path) -> None:
    _write_coordsystem(project / "input" / "coordsystem.json")

    prepared = prepare_project(project)

    assert prepared.failures == ()
    points = prepared.loaded.fiducials
    assert isinstance(points, LabeledPoints)
    assert points.labels == ("NAS",)
    assert points.points.tolist() == [[1.0, 2.0, 3.0]]
    assert prepared.recoverable_step == 4


def test_prepare_project_fiducials_win_over_coordsystem(project: Path) -> None:
    _write_coordsystem(project / "input" / "coordsystem.json")
    _write_fiducials(project / "somewhere" / "fiducials.json")

    prepared = prepare_project(project)

    assert prepared.failures == ()
    assert cast("LabeledPoints", prepared.loaded.fiducials).labels == ("nas",)


def test_prepare_project_final_mesh_wins_over_base_mesh(project: Path) -> None:
    base = project / "scalp"
    base.mkdir(exist_ok=True)
    _write_ply(base / "base_mesh.ply")
    _write_ply(project / "mesh" / "final_mesh.ply")

    prepared = prepare_project(project)

    assert prepared.failures == ()
    assert isinstance(prepared.loaded.scalp_mesh, Mesh)
    assert isinstance(prepared.loaded.base_mesh, Mesh)
    assert prepared.recoverable_step == 2


def test_prepare_project_corrupt_measurements_is_reported(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    (project / "input" / "measurements.json").write_text("not json", encoding="utf-8")

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert "measurements.json" in prepared.failures[0]
    assert prepared.loaded.measurements is None
    assert prepared.recoverable_step == 0


def test_prepare_project_corrupt_mesh_is_reported(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    (project / "mesh" / "final_mesh.ply").write_text("not a mesh", encoding="utf-8")

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert "final_mesh.ply" in prepared.failures[0]
    assert prepared.loaded.scalp_mesh is None
    assert prepared.recoverable_step == 0
