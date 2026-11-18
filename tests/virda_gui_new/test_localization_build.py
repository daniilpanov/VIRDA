"""Unit tests for :mod:`virda_gui_new.services.localization.build`."""

import numpy as np
import pytest

from virda_gui_new.services.localization.build import (
    blocked_reason,
    build_electrodes,
    build_fiducials,
    measurement_counts,
)


def test_build_fiducials_happy() -> None:
    rows = [("Nasion", np.array([0.0, 1.0, 2.0])), ("  LPA ", np.array([3.0, 4.0, 5.0]))]

    fiducials = build_fiducials(rows)

    assert [(f.fiducial_id, f.name) for f in fiducials.items] == [
        ("Nasion", "Nasion"),
        ("LPA", "LPA"),
    ]
    first = fiducials.items[0]
    assert first.coordinates.shape == (3,)
    assert first.coordinate_system == "world"
    assert first.definition_method == "manual"
    assert first.weight == 1.0


def test_build_fiducials_rejects_empty_name() -> None:
    rows = [("   ", np.zeros(3))]

    with pytest.raises(ValueError, match="non-empty"):
        build_fiducials(rows)


def test_build_fiducials_rejects_duplicate_names() -> None:
    rows = [("Nasion", np.zeros(3)), ("  Nasion ", np.ones(3))]

    with pytest.raises(ValueError, match="unique"):
        build_fiducials(rows)


def test_build_fiducials_rejects_bad_shape() -> None:
    rows = [("Nasion", np.zeros(4))]

    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        build_fiducials(rows)


def test_build_electrodes_maps_rows() -> None:
    table = {"A": {"Alpha": 10.0}, "B": {"Alpha": 12.0, "Beta": 5.0}}

    electrodes = build_electrodes(table)

    assert [e.electrode_id for e in electrodes.items] == ["A", "B"]
    assert electrodes.items[0].measured_distances == {"Alpha": 10.0}
    assert electrodes.items[1].measured_distances == {"Alpha": 12.0, "Beta": 5.0}
    assert electrodes.items[0].ese_coords is None
    assert not electrodes.items[0].flagged


def test_build_electrodes_blocks_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_electrodes({"A": {"Alpha": 1.0}, "A ": {"Beta": 2.0}})


@pytest.mark.parametrize(
    ("n_fiducials", "n_complete", "n_incomplete", "expected"),
    [
        (0, 0, 0, "not enough fiducials (need 3)"),
        (2, 1, 0, "not enough fiducials (need 3)"),
        (3, 0, 0, "no measurements"),
        (3, 1, 0, None),
        (3, 0, 1, "1 electrode(s) need all three measured distances"),
        (3, 1, 2, "2 electrode(s) need all three measured distances"),
    ],
)
def test_blocked_reason_matrix(
    n_fiducials: int, n_complete: int, n_incomplete: int, expected: str | None
) -> None:
    assert blocked_reason(n_fiducials, n_complete, n_incomplete) == expected


def test_measurement_counts_separates_the_complete_rows() -> None:
    table = {
        "A": {"NAS": 1.0, "LPA": 2.0, "RPA": 3.0},
        "B": {"NAS": 1.0, "LPA": 2.0},
    }

    assert measurement_counts(table) == (1, 1)


def test_measurement_counts_knows_the_aliases_of_the_nasion_point() -> None:
    table = {"A": {"NASION": 1.0, "LPA": 2.0, "RPA": 3.0}}

    assert measurement_counts(table) == (1, 0)


def test_measurement_counts_ignores_an_empty_row() -> None:
    assert measurement_counts({"A": {}}) == (0, 0)
