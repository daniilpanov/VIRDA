"""Unit tests for :class:`virda_gui_new.services.localization.runner`.

Runs under the offscreen Qt platform; ``localize_electrodes`` is mocked, so no
heavy mesh work runs; only payload delivery is exercised.
"""

import time
from typing import cast

import numpy as np
import pytest

from virda.models.coordinates import CoordinateSystem, Point
from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.mesh import Mesh
from virda_gui_new.services.localization import LiveLocalizationRunner

_ESE = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
_SCALP = Mesh(np.zeros((3, 3)), np.zeros((1, 3), dtype=int))
_NORMALS = np.ones((4, 3))
_FIDUCIALS = Fiducials(
    [Fiducial("a", "a", Point(x=0.0, y=0.0, z=0.0, system=CoordinateSystem.WORLD))]
)
_ELECTRODES = Electrodes([Electrode(electrode_id="A", measured_distances={"a": 1.0})])


@pytest.fixture
def runners(qapp):
    """Collect runners and stop their worker threads at teardown."""
    created: list[LiveLocalizationRunner] = []
    yield created
    for runner in created:
        runner.stop()


def _make_runner(runners) -> LiveLocalizationRunner:
    runner = LiveLocalizationRunner(_ESE, _NORMALS, _SCALP)
    runners.append(runner)
    return runner


def _pump(qapp, deadline_s: float) -> None:
    """Process events until the debounce timer and worker thread land."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_missing_ese_mesh_raises(qapp) -> None:
    with pytest.raises(ValueError, match="run the ESE mesh step first"):
        LiveLocalizationRunner(cast(Mesh, None), _NORMALS, _SCALP)


def test_submit_delivers_latest_result(qapp, runners, monkeypatch) -> None:
    results: list[object] = []
    failures: list[str] = []
    calls: list[tuple[object, ...]] = []

    def fake_localize(
        ese, normals, scalp, fiducials, electrodes, *, config, fixed_offset_shift_mm=None
    ):
        calls.append((ese, normals, scalp, fiducials, electrodes, config))
        return {"seq": len(calls), "eid": electrodes.items[0].electrode_id}

    monkeypatch.setattr(
        "virda_gui_new.services.localization.runner.localize_electrodes", fake_localize
    )
    runner = _make_runner(runners)
    runner.localized.connect(results.append)
    runner.failed.connect(failures.append)

    runner.submit(_FIDUCIALS, _ELECTRODES)
    second = Electrodes([Electrode(electrode_id="B", measured_distances={"a": 2.0})])
    runner.submit(_FIDUCIALS, second)  # supersedes the first snapshot
    _pump(qapp, 2.0)
    qapp.processEvents()

    assert failures == []
    assert results == [{"seq": 1, "eid": "B"}]
    assert len(calls) == 1  # second submit superseded the first debounce
    assert calls[0][0] is _ESE and calls[0][1] is _NORMALS and calls[0][2] is _SCALP


def test_failed_path(qapp, runners, monkeypatch) -> None:
    results: list[object] = []
    failures: list[str] = []

    def boom(*_args, **_kwargs):
        raise RuntimeError("no fiducials on surface")

    monkeypatch.setattr("virda_gui_new.services.localization.runner.localize_electrodes", boom)
    runner = _make_runner(runners)
    runner.localized.connect(results.append)
    runner.failed.connect(failures.append)

    runner.submit(_FIDUCIALS, _ELECTRODES)
    _pump(qapp, 2.0)
    qapp.processEvents()

    assert results == []
    assert failures == ["no fiducials on surface"]


def test_stop_silences_pending_submit(qapp, runners, monkeypatch) -> None:
    results: list[object] = []
    monkeypatch.setattr(
        "virda_gui_new.services.localization.runner.localize_electrodes", lambda *_args, **_kw: None
    )
    runner = _make_runner(runners)
    runner.localized.connect(results.append)

    runner.submit(_FIDUCIALS, _ELECTRODES)
    runner.stop()
    _pump(qapp, 2.0)
    qapp.processEvents()

    assert results == []


def test_busy_tracks_a_successful_run(qapp, runners, monkeypatch) -> None:
    busy: list[bool] = []
    monkeypatch.setattr(
        "virda_gui_new.services.localization.runner.localize_electrodes",
        lambda *_args, **_kw: Electrodes([]),
    )
    runner = _make_runner(runners)
    runner.busyChanged.connect(busy.append)

    runner.submit(_FIDUCIALS, _ELECTRODES)
    _pump(qapp, 2.0)

    # One submit schedules one debounced run: busy turns on once, off once.
    assert busy == [True, False]


def test_busy_tracks_a_failed_run(qapp, runners, monkeypatch) -> None:
    busy: list[bool] = []
    failures: list[str] = []

    def boom(*_args, **_kwargs):
        raise RuntimeError("no fiducials on surface")

    monkeypatch.setattr("virda_gui_new.services.localization.runner.localize_electrodes", boom)
    runner = _make_runner(runners)
    runner.busyChanged.connect(busy.append)
    runner.failed.connect(failures.append)

    runner.submit(_FIDUCIALS, _ELECTRODES)
    _pump(qapp, 2.0)

    assert failures == ["no fiducials on surface"]
    assert busy == [True, False]


def test_busy_stays_on_when_a_result_submits_the_next_run(qapp, runners, monkeypatch) -> None:
    """A listener that submits from ``localized`` keeps the busy state.

    The debounce of the new run is already armed when the previous
    result lands, so no ``False`` slips between the two runs.
    """
    busy: list[bool] = []
    monkeypatch.setattr(
        "virda_gui_new.services.localization.runner.localize_electrodes",
        lambda *_args, **_kw: Electrodes([]),
    )
    runner = _make_runner(runners)
    runner.busyChanged.connect(busy.append)
    runner.localized.connect(lambda _result: runner.submit(_FIDUCIALS, _ELECTRODES))

    runner.submit(_FIDUCIALS, _ELECTRODES)
    _pump(qapp, 2.0)

    assert busy == [True]


def test_stop_releases_the_busy_state(qapp, runners, monkeypatch) -> None:
    busy: list[bool] = []
    monkeypatch.setattr(
        "virda_gui_new.services.localization.runner.localize_electrodes",
        lambda *_args, **_kw: Electrodes([]),
    )
    runner = _make_runner(runners)
    runner.busyChanged.connect(busy.append)

    runner.submit(_FIDUCIALS, _ELECTRODES)
    runner.stop()

    assert busy == [True, False]
