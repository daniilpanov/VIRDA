import csv
from pathlib import Path

import numpy as np
import pytest

from virda.io.importers.electrodes_table import import_electrodes_table


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestImportElectrodesTable:
    def test_tsv_with_names(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "electrodes.tsv",
            "name\tx\ty\tz\nFz\t0.0\t0.0\t10.0\nCz\t1.0\t2.0\t3.0\n",
        )

        positions, residuals, flags, measured, names = import_electrodes_table(path)

        np.testing.assert_array_almost_equal(positions, [[0.0, 0.0, 10.0], [1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(residuals, [0.0, 0.0])
        np.testing.assert_array_equal(flags, [False, False])
        assert measured == [{}, {}]
        assert names == ["Fz", "Cz"]

    def test_csv_autodetects_delimiter(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "electrodes.csv", "name,x,y,z\nPz,5.0,6.0,7.0\n")

        positions, _residuals, _flags, _measured, names = import_electrodes_table(path)

        np.testing.assert_array_almost_equal(positions, [[5.0, 6.0, 7.0]])
        assert names == ["Pz"]

    def test_columns_matched_case_insensitively(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "electrodes.tsv", "Name\tX\tY\tZ\nOz\t1.0\t1.0\t1.0\n")

        positions, _residuals, _flags, _measured, names = import_electrodes_table(path)

        np.testing.assert_array_almost_equal(positions, [[1.0, 1.0, 1.0]])
        assert names == ["Oz"]

    def test_missing_names_get_sequential_ids(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "electrodes.tsv", "x\ty\tz\n1.0\t0.0\t0.0\n2.0\t0.0\t0.0\n")

        _positions, _residuals, _flags, _measured, names = import_electrodes_table(path)

        assert names == ["E001", "E002"]


class TestImportElectrodesTableErrors:
    def test_missing_coordinate_column(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "electrodes.tsv", "name\tx\ty\nFz\t0.0\t0.0\n")

        with pytest.raises(ValueError, match="missing required column 'z'"):
            import_electrodes_table(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "electrodes.tsv", "")

        with pytest.raises(csv.Error, match="delimiter"):
            import_electrodes_table(path)
