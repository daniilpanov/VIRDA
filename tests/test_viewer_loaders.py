import json
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from virda_gui.viewer import (
    _cras_decision_message,
    _cras_to_scanner_ras_offset,
    _detect_cras_conversion,
    _load_electrodes,
    _parse_electrode_specs,
    _resolve_group_color,
)


def _write_tsv(path: Path, rows: list[tuple[str, float, float, float]]) -> None:
    lines = ["name\tx\ty\tz"]
    lines += [f"{name}\t{x}\t{y}\t{z}" for name, x, y, z in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sphere_points(n: int = 400, radius: float = 80.0) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    y = 1.0 - 2.0 * idx / (n - 1)
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = idx * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), y]) * radius


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


class TestCrasAutoDetection:
    OFFSET = np.array([4.77, -8.54, -26.79])

    def test_detects_cras_when_shifted_points_fit_mesh(self) -> None:
        mesh = _sphere_points()
        true_pts = mesh[::40].copy()
        raw = true_pts - self.OFFSET

        converted, d_raw, d_shift = _detect_cras_conversion(raw, mesh, None, True, self.OFFSET)

        assert converted is True
        assert d_raw > 10.0
        assert d_shift == pytest.approx(0.0, abs=1e-6)

    def test_keeps_scanner_ras_when_already_fitting(self) -> None:
        mesh = _sphere_points()
        true_pts = mesh[::40].copy()

        converted, d_raw, d_shift = _detect_cras_conversion(true_pts, mesh, None, True, self.OFFSET)

        assert converted is False
        assert d_raw == pytest.approx(0.0, abs=1e-6)
        assert d_shift > 10.0

    def test_ambiguous_half_shift_prefers_no_conversion(self) -> None:
        mesh = _sphere_points()
        true_pts = mesh[::40].copy()
        raw = true_pts - 0.5 * self.OFFSET

        converted, d_raw, d_shift = _detect_cras_conversion(raw, mesh, None, True, self.OFFSET)

        assert converted is False
        assert d_shift == pytest.approx(d_raw, rel=0.2)

    def test_voxel_scene_with_rotated_affine(self) -> None:
        angle = 0.3
        affine = np.eye(4)
        affine[:3, :3] = 2.0 * np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        affine[:3, 3] = [10.0, 20.0, 30.0]
        true_pts = _sphere_points()[::40].copy()
        mesh_voxel = true_pts @ np.linalg.inv(affine[:3, :3]).T + np.linalg.inv(affine)[:3, 3]
        raw = true_pts - self.OFFSET

        converted, _, d_shift = _detect_cras_conversion(
            raw, mesh_voxel, affine, mm_scene=False, cras_offset=self.OFFSET
        )

        assert converted is True
        assert d_shift == pytest.approx(0.0, abs=1e-6)

    def test_decision_message_formatting(self) -> None:
        message = _cras_decision_message("electrodes.tsv", True, 27.3, 2.94)

        assert message == "electrodes.tsv: cRAS detected (median dist 27.3 -> 2.9 mm), converted"

        message = _cras_decision_message("electrodes.tsv", False, 2.1, 28.6)

        assert (
            message == "electrodes.tsv: scanner RAS assumed (median dist 2.1 -> 28.6 mm), unchanged"
        )


class TestElectrodeSpecs:
    def test_path_only(self) -> None:
        assert _parse_electrode_specs([["a.tsv"]]) == [("a.tsv", None)]

    def test_path_with_color(self) -> None:
        assert _parse_electrode_specs([["a.tsv", "yellow"], ["b.json", "green"]]) == [
            ("a.tsv", "yellow"),
            ("b.json", "green"),
        ]

    def test_rejects_extra_values(self) -> None:
        with pytest.raises(ValueError, match="expects FILE .COLOR., got 3"):
            _parse_electrode_specs([["a.tsv", "yellow", "extra"]])

    def test_explicit_color_wins(self) -> None:
        assert _resolve_group_color("green", 0) == "green"

    def test_palette_fallback_rotates(self) -> None:
        assert _resolve_group_color(None, 0) == "yellow"
        assert _resolve_group_color(None, 7) == "lime"

    def test_rejects_unknown_color(self) -> None:
        with pytest.raises(ValueError, match="invalid electrode color"):
            _resolve_group_color("not-a-color", 0)


class TestIntensifyColor:
    def test_returns_valid_distinct_color(self) -> None:
        from virda_gui.viewer import _intensify_color

        for base in ("yellow", "lime", "magenta", "cyan", "orange", "white", "#34eb89"):
            boosted = _intensify_color(base)
            assert pv.Color(boosted) is not None
            assert pv.Color(boosted).float_rgb != pv.Color(base).float_rgb

    def test_preserves_hue(self) -> None:
        import colorsys

        from virda_gui.viewer import _intensify_color

        base = pv.Color("orange").float_rgb
        boosted = pv.Color(_intensify_color("orange")).float_rgb
        h0 = colorsys.rgb_to_hls(*base)[0]
        h1 = colorsys.rgb_to_hls(*boosted)[0]
        assert abs(h0 - h1) < 1e-3
