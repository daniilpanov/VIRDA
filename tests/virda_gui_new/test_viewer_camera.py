"""Unit tests for the viewer camera helpers."""

from virda_gui_new.widgets.viewer3d import _apply_camera_state, _camera_state


class _FakeCamera:
    """The camera attributes the helpers read and write."""

    def __init__(self) -> None:
        self.position = (1.0, 2.0, 3.0)
        self.focal_point = (0.0, 0.0, 0.0)
        self.up = (0.0, 0.0, 1.0)
        self.view_angle = 30.0
        self.parallel_projection = False


def test_the_camera_state_round_trips() -> None:
    camera = _FakeCamera()

    state = _camera_state(camera)
    camera.position = (9.0, 9.0, 9.0)
    camera.focal_point = (1.0, 1.0, 1.0)
    camera.up = (0.0, 1.0, 0.0)
    camera.view_angle = 45.0
    camera.parallel_projection = True
    _apply_camera_state(camera, state)

    assert camera.position == (1.0, 2.0, 3.0)
    assert camera.focal_point == (0.0, 0.0, 0.0)
    assert camera.up == (0.0, 0.0, 1.0)
    assert camera.view_angle == 30.0
    assert camera.parallel_projection is False


def test_the_camera_state_notices_a_moved_camera() -> None:
    camera = _FakeCamera()
    before = _camera_state(camera)

    camera.position = (5.0, 5.0, 5.0)

    assert _camera_state(camera) != before


def test_the_reset_view_frames_the_meshes(viewer) -> None:
    """The camera frames the meshes of the scene, not everything drawn."""
    viewer._reset_camera()

    focal_point = viewer._plotter.camera.focal_point

    assert tuple(round(float(value), 6) for value in focal_point) == (0.0, 0.0, 0.0)


def test_the_reset_button_lives_outside_the_layers_panel(viewer) -> None:
    """The panel is rebuilt with every scene; the button is not."""
    from PySide6.QtWidgets import QGroupBox, QPushButton

    panels = viewer.findChildren(QGroupBox)
    assert panels
    for panel in panels:
        assert panel.findChildren(QPushButton) == []
    assert any(button.text() == "Reset view" for button in viewer.findChildren(QPushButton))
