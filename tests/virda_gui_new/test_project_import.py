"""Unit tests for :mod:`virda_gui_new.services.project_import`."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

from virda.io.fiducial_helpers import save_fiducials
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


def _write_point_csv(path: Path, rows: list[str]) -> None:
    """Write the pipeline coords CSV header plus the given raw rows."""
    lines = ["electrode_id,x,y,z,residual_mm,flagged", *rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def test_prepare_project_rejects_invalid_point_rows(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    _write_ply(project / "mesh" / "final_mesh.ply")
    _write_ply(project / "ese" / "ese_mesh.ply")
    _write_point_csv(project / "localization" / "electrodes_scalp.csv", ["A1,oops,2.0,3.0,,no"])

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert "electrodes_scalp.csv" in prepared.failures[0]
    assert "coordinate" in prepared.failures[0]
    assert prepared.loaded.scalp_points is None
    # Everything else still seeds; recovery follows the loaded artifacts.
    assert prepared.loaded.nifti is not None
    assert prepared.loaded.scalp_mesh is not None
    assert prepared.recoverable_step == 3


def test_prepare_project_rejects_empty_point_file(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    _write_point_csv(project / "localization" / "electrodes_scalp.csv", [])

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert "no electrode points found" in prepared.failures[0]
    assert prepared.loaded.scalp_points is None
    assert prepared.recoverable_step == 0


def test_prepare_project_rejects_point_file_without_coordinate_columns(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    (project / "localization" / "electrodes_scalp.csv").write_text(
        "not a table\n", encoding="utf-8"
    )

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert prepared.loaded.scalp_points is None


def test_prepare_project_loads_electrode_point_csvs(project: Path) -> None:
    _write_point_csv(project / "localization" / "electrodes_scalp.csv", ["A1,1.0,2.0,3.0,,no"])
    _write_point_csv(project / "localization" / "electrodes_ese.csv", ["E1,9.0,8.0,7.0,,no"])

    prepared = prepare_project(project)

    assert prepared.failures == ()
    scalp = prepared.loaded.scalp_points
    ese = prepared.loaded.ese_points
    assert scalp is not None
    assert ese is not None
    assert scalp.labels == ("A1",)
    assert scalp.points.tolist() == [[1.0, 2.0, 3.0]]
    assert ese.points.tolist() == [[9.0, 8.0, 7.0]]
    assert prepared.recoverable_step == 4


def test_prepare_project_empty_directory(project: Path) -> None:
    prepared = prepare_project(project)

    assert prepared.failures == ()
    assert prepared.loaded == type(prepared.loaded)()
    assert prepared.recoverable_step == -1


def test_prepare_project_corrupt_mesh_is_reported(project: Path) -> None:
    _write_nifti(project / "input" / "a.nii")
    (project / "mesh" / "final_mesh.ply").write_text("not a mesh", encoding="utf-8")

    prepared = prepare_project(project)

    assert len(prepared.failures) == 1
    assert "final_mesh.ply" in prepared.failures[0]
    assert prepared.loaded.scalp_mesh is None
    assert prepared.recoverable_step == 0
