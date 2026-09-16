"""Unit tests for :mod:`virda_gui.preview` pure helpers (no QApplication)."""

from pathlib import Path

from virda_gui.preview import parse_csv_tsv


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
