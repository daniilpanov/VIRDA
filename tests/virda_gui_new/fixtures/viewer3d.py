import numpy as np
import pytest
import pyvista as pv

from virda_gui_new.dto.viewer3d import LabeledPoints


def points(n: int) -> LabeledPoints:
    rng = np.random.default_rng(n)
    return LabeledPoints(rng.random((n, 3)), tuple(f"p{i}" for i in range(n)))


@pytest.fixture
def viewer(qapp):
    """A scalp + ESE scene with points on both and two fiducials."""
    from virda_gui_new.widgets.viewer3d import SceneViewer3D

    widget = SceneViewer3D(
        scalp=pv.Sphere(),
        scalp_points=points(3),
        ese=pv.Sphere(radius=0.4),
        ese_points=points(2),
        fiducials=points(2),
    )
    yield widget
    widget.deleteLater()
