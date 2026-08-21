import json
from pathlib import Path

import numpy as np
import pytest

from virda_gui.viewer import (
    _cras_to_scanner_ras_offset,
    _load_electrodes,
)


def _write_tsv(path: Path, rows: list[tuple[str, float, float, float]]) -> None:
    lines = ["name\tx\ty\tz"]
    lines += [f"{name}\t{x}\t{y}\t{z}" for name, x, y, z in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCrasToScannerRasOffset:
    def test_identity_affine_centers_at_midpoint(self) -> None:
        affine = np.eye(4)
        offset = _cras_to_scanner_ras_offset(affine, (64, 64, 64))
        assert offset == pytest.approx([31.5, 31.5, 31.5])

    def test_scaling_and_translation_affine(self) -> None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [-10.0, 5.0, 3.0]
        offset = _cras_to_scanner_ras_offset(affine, (33, 33, 33))
        # center voxel (16, 16, 16) scaled by 2 plus translation
        assert offset == pytest.approx([22.0, 37.0, 35.0])


class TestTabularCrasConversion:
    def test_offset_applied_to_points(self, tmp_path: Path) -> None:
        path = tmp_path / "electrodes.tsv"
        _write_tsv(path, [("Fz", 1.0, 2.0, 3.0), ("Cz", 4.0, 5.0, 6.0)])
        offset = np.array([10.0, -1.0, 0.5])

        points, residuals, flags, measured, names = _load_electrodes(str(path), cras_offset=offset)

        assert np.asarray(points) == pytest.approx(np.array([[11.0, 1.0, 3.5], [14.0, 4.0, 6.5]]))
        assert len(residuals) == 2
        assert not flags.any()
        assert measured == [{}, {}]
        assert names == ["Fz", "Cz"]

    def test_without_offset_points_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "electrodes.tsv"
        _write_tsv(path, [("Fz", 1.0, 2.0, 3.0)])

        points, _, _, _, _ = _load_electrodes(str(path))

        assert np.asarray(points) == pytest.approx(np.array([[1.0, 2.0, 3.0]]))

    def test_missing_name_column_generates_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "electrodes.csv"
        path.write_text("x,y,z\n1,2,3\n4,5,6\n", encoding="utf-8")

        _, _, _, _, names = _load_electrodes(str(path))

        assert names == ["E001", "E002"]

    def test_blank_name_cell_generates_id(self, tmp_path: Path) -> None:
        path = tmp_path / "electrodes.tsv"
        lines = ["name\tx\ty\tz", "\t1\t2\t3"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        _, _, _, _, names = _load_electrodes(str(path))

        assert names == ["E001"]

    def test_json_group_ignores_cras_offset(self, tmp_path: Path) -> None:
        path = tmp_path / "electrodes.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "electrode_id": "E1",
                        "coords": [1.0, 2.0, 3.0],
                        "residual_error": 0.5,
                        "flagged": False,
                    }
                ]
            ),
            encoding="utf-8",
        )
        offset = np.array([100.0, 100.0, 100.0])

        points, residuals, flags, _, names = _load_electrodes(str(path), cras_offset=offset)

        assert np.asarray(points) == pytest.approx(np.array([[1.0, 2.0, 3.0]]))
        assert residuals == pytest.approx([0.5])
        assert not flags.any()
        assert names == ["E1"]

    def test_json_without_ids_generates_names(self, tmp_path: Path) -> None:
        path = tmp_path / "electrodes.json"
        path.write_text(
            json.dumps(
                [
                    {"coords": [1.0, 2.0, 3.0]},
                    {"coords": [4.0, 5.0, 6.0], "electrode_id": "Cz"},
                ]
            ),
            encoding="utf-8",
        )

        _, _, _, _, names = _load_electrodes(str(path))

        assert names == ["E001", "Cz"]
