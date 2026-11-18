"""Unit tests for the manager's geometry service."""

import numpy as np
import pytest

from tests.virda_gui_new.fixtures.steps import advance_to
from virda.mesh.center_of_mass import center_of_mass
from virda.models.coordinates import CoordinateSystem
from virda.models.fiducial import FiducialType
from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.steps import NIFTI, STEPS
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_loading
from virda_gui_new.services.steps_manager import StepsManager

# NAS on the +y axis, LPA/RPA symmetric on the x axis: the fiducials
# frame axes come out axis-aligned.
_NAS = np.array([0.0, 100.0, 0.0])
_LPA = np.array([-70.0, 0.0, 0.0])
_RPA = np.array([70.0, 0.0, 0.0])
_AFFINE = np.array(
    [
        [0.5, 0.0, 0.0, 10.0],
        [0.0, 0.5, 0.0, -5.0],
        [0.0, 0.0, 0.5, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _make_manager(mesh: Mesh | None = None, fiducials: LabeledPoints | None = None) -> StepsManager:
    """Manager advanced to step 4 with optional mesh and fiducial inputs."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    advance_to(steps, 4)
    if mesh is not None:
        steps.set_step_context_item(1, "scalp_surface", mesh)
        steps.set_step_context_item(2, "mesh", mesh)
    if fiducials is not None:
        steps.set_global_item("fiducials", fiducials)
    return steps


def test_head_origin_is_the_mesh_center_of_mass() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    mesh = Mesh(vertices, np.zeros((1, 3), dtype=int))
    geometry = _make_manager(mesh=mesh).geometry

    origin = geometry.context.head_origin
    assert origin is not None
    assert np.allclose(origin.as_array(), center_of_mass(mesh))
    assert geometry.is_available(CoordinateSystem.HEAD)


def test_head_origin_follows_the_mesh() -> None:
    first = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
    second = Mesh(np.ones((4, 3)), np.zeros((1, 3), dtype=int))
    manager = _make_manager(mesh=first)

    geometry = manager.geometry
    head_origin = geometry.context.head_origin
    assert head_origin is not None
    assert np.allclose(head_origin.as_array(), np.zeros(3))

    manager.set_step_context_item(2, "mesh", second)

    head_origin = manager.geometry.context.head_origin
    assert head_origin is not None
    assert np.allclose(head_origin.as_array(), np.ones(3))


def test_step_2_mesh_wins_over_the_step_1_scalp_surface() -> None:
    base = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
    final = Mesh(np.ones((4, 3)), np.zeros((1, 3), dtype=int))
    manager = _make_manager()
    manager.set_step_context_item(1, "scalp_surface", base)
    manager.set_step_context_item(2, "mesh", final)

    head_origin = manager.geometry.context.head_origin
    assert head_origin is not None
    assert np.allclose(head_origin.as_array(), np.ones(3))


def test_a_broken_step_2_mesh_is_not_taken_for_the_base_surface() -> None:
    """A step-2 value that is not a mesh leaves the state without a mesh."""
    manager = _make_manager()
    manager.set_step_context_item(
        1, "scalp_surface", Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
    )
    manager.set_step_context_item(2, "mesh", object())

    assert manager.active_scalp_mesh() is None


def test_head_needs_a_mesh() -> None:
    geometry = _make_manager().geometry

    assert geometry.context.head_origin is None
    assert not geometry.is_available(CoordinateSystem.HEAD)
    with pytest.raises(ValueError, match="scalp mesh centroid"):
        geometry.from_world(np.zeros((1, 3)), CoordinateSystem.HEAD)


def test_fiducials_match_the_store_labels() -> None:
    points = LabeledPoints(
        np.array([_NAS, _LPA, _RPA, [9.0, 9.0, 9.0]]),
        ("NASION", "LPA", "rpa", "Extra"),
    )
    geometry = _make_manager(fiducials=points).geometry

    assert set(geometry.context.fiducials) == {"NAS", "LPA", "RPA"}
    nas = geometry.context.fiducials[FiducialType.NAS]
    assert np.allclose(nas.as_array(), _NAS)
    assert geometry.is_available(CoordinateSystem.FIDUCIALS)
    assert np.allclose(geometry.from_world(_NAS, CoordinateSystem.FIDUCIALS), [0.0, 0.0, 0.0])


def test_an_empty_store_has_no_fiducials() -> None:
    geometry = _make_manager().geometry

    assert geometry.context.fiducials == {}
    assert not geometry.is_available(CoordinateSystem.FIDUCIALS)


def test_world_is_always_available() -> None:
    geometry = _make_manager().geometry

    assert geometry.is_available(CoordinateSystem.WORLD)
    converted = geometry.from_world(np.array([[1.0, 2.0, 3.0]]), CoordinateSystem.WORLD)
    assert np.allclose(converted, [[1.0, 2.0, 3.0]])


def test_voxel_needs_the_step_0_volume() -> None:
    manager = _make_manager()

    assert not manager.geometry.is_available(CoordinateSystem.VOXEL)

    manager.set_step_context_item(
        NIFTI.index,
        "volume",
        MRIVolume(np.zeros((2, 2, 2)), _AFFINE, (0.5, 0.5, 0.5), ("R", "A", "S")),
    )

    geometry = manager.geometry
    assert geometry.is_available(CoordinateSystem.VOXEL)
    assert np.allclose(geometry.from_world(_NAS, CoordinateSystem.VOXEL), [-20.0, 210.0, -6.0])


def test_a_path_volume_is_loaded_once(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "scan.nii.gz"
    calls: list[object] = []

    def fake_load(source) -> MRIVolume:
        calls.append(source)
        return MRIVolume(np.zeros((2, 2, 2)), _AFFINE, (0.5, 0.5, 0.5), ("R", "A", "S"))

    monkeypatch.setattr(artifact_loading, "load_volume", fake_load)
    manager = _make_manager()
    manager.set_step_context_item(NIFTI.index, "volume", path)

    assert manager.geometry.is_available(CoordinateSystem.VOXEL)
    manager.set_global_item("fiducials", LabeledPoints(np.array([_NAS]), ("NAS",)))
    assert manager.geometry.is_available(CoordinateSystem.VOXEL)
    assert calls == [path]


def test_the_geometry_instance_is_stable_across_reads() -> None:
    manager = _make_manager()

    geometry = manager.geometry

    assert manager.geometry is geometry
