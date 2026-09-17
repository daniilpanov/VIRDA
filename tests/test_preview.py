"""Unit tests for :mod:`virda_gui.preview` pure helpers (no QApplication)."""

from pathlib import Path

import numpy as np
import pytest

from virda_gui.preview import (
    load_npy_table,
    npy_columns_for,
    npy_table_rows,
    parse_csv_tsv,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestParseCsvTsv:
    def test_parse_comma_separated(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "electrode_coords.csv", "id,x,y,z\nE1,1,2,3\nE2,4,5,6\n")
        headers, rows = parse_csv_tsv(path)
        assert headers == ["id", "x", "y", "z"]
        assert rows == [["E1", "1", "2", "3"], ["E2", "4", "5", "6"]]

    def test_parse_tab_separated(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "electrodes.tsv", "name\tx\ty\tz\nA\t0.1\t0.2\t0.3\n")
        headers, rows = parse_csv_tsv(path)
        assert headers == ["name", "x", "y", "z"]
        assert rows == [["A", "0.1", "0.2", "0.3"]]

    def test_header_only_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "empty.csv", "a,b,c\n")
        headers, rows = parse_csv_tsv(path)
        assert headers == ["a", "b", "c"]
        assert rows == []

    def test_ragged_rows_are_padded(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "ragged.csv", "a,b,c\n1,2\n1,2,3,4\n")
        headers, rows = parse_csv_tsv(path)
        assert headers == ["a", "b", "c"]
        assert rows == [["1", "2", ""], ["1", "2", "3"]]

    def test_utf8_bom_is_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.csv"
        path.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n")
        headers, rows = parse_csv_tsv(path)
        assert headers == ["a", "b"]
        assert rows == [["1", "2"]]

    def test_quoted_fields(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "quoted.csv", 'name,note\n"E,1","has, comma"\n')
        headers, rows = parse_csv_tsv(path)
        assert headers == ["name", "note"]
        assert rows == [["E,1", "has, comma"]]


class TestNpyColumnMap:
    def test_vertices_mask(self) -> None:
        assert npy_columns_for("scalp_vertices.npy") == ["X", "Y", "Z"]

    def test_faces_mask(self) -> None:
        assert npy_columns_for("ese_faces.npy") == ["A", "B", "C"]

    def test_adjacency_mask(self) -> None:
        assert npy_columns_for("scalp_face_adjacency.npy") == ["Face A", "Face B"]

    def test_normals_mask(self) -> None:
        assert npy_columns_for("normals.npy") == ["NX", "NY", "NZ"]

    def test_quality_mask(self) -> None:
        assert npy_columns_for("quality.npy") == ["Quality"]

    def test_mask_is_case_insensitive(self) -> None:
        assert npy_columns_for("Scalp_Vertices.npy") == ["X", "Y", "Z"]

    def test_unknown_name_returns_none(self) -> None:
        assert npy_columns_for("custom_data.npy") is None


class TestLoadNpyTable:
    def test_2d_array_with_masked_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "scalp_vertices.npy"
        np.save(path, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        headers, array = load_npy_table(path)
        assert headers == ["X", "Y", "Z"]
        assert array.shape == (2, 3)

    def test_1d_array_gets_value_column(self, tmp_path: Path) -> None:
        path = tmp_path / "quality.npy"
        np.save(path, np.array([0.1, 0.2]))
        headers, array = load_npy_table(path)
        assert headers == ["Value"]
        assert array.shape == (2,)

    def test_column_count_mismatch_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "scalp_faces.npy"
        np.save(path, np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))
        headers, _ = load_npy_table(path)
        assert headers == ["Col 0", "Col 1", "Col 2", "Col 3"]

    def test_3d_array_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "volume.npy"
        np.save(path, np.zeros((2, 2, 2)))
        with pytest.raises(ValueError, match="3D"):
            load_npy_table(path)


class TestNpyTableRows:
    def test_2d_rows(self) -> None:
        rows = npy_table_rows(np.array([[1, 2], [3, 4]]))
        assert rows == [["1", "2"], ["3", "4"]]

    def test_1d_columns_as_rows(self) -> None:
        rows = npy_table_rows(np.array([0.1, 0.2]))
        assert rows == [["0.1"], ["0.2"]]

    def test_scalar_single_cell(self) -> None:
        rows = npy_table_rows(np.array(5.0))
        assert rows == [["5.0"]]
