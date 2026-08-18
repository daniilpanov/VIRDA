import argparse

import pytest

from virda.main import (
    _ESE_PARAMS,
    _validate_required_group,
    _warn_partial_neighborhood,
)
from virda.models.config import Config


def _ns(**kwargs: object) -> argparse.Namespace:
    """Build an argparse.Namespace with the given attributes."""
    return argparse.Namespace(**kwargs)


class TestValidateRequiredGroup:
    def test_all_provided(self) -> None:
        args = _ns(n_electrodes=32, ese_offset_mm=2.5, ese_reference="electrode_body_center")
        _validate_required_group(args, "ESE", _ESE_PARAMS)

    def test_none_provided(self) -> None:
        args = _ns(n_electrodes=None, ese_offset_mm=None, ese_reference=None)
        _validate_required_group(args, "ESE", _ESE_PARAMS)

    def test_partial_raises(self) -> None:
        args = _ns(n_electrodes=None, ese_offset_mm=2.5, ese_reference="electrode_body_center")
        with pytest.raises(SystemExit, match="--ese-offset-mm, --ese-reference"):
            _validate_required_group(args, "ESE", _ESE_PARAMS)

    def test_single_param_raises(self) -> None:
        args = _ns(n_electrodes=32, ese_offset_mm=None, ese_reference=None)
        with pytest.raises(SystemExit, match="--n-electrodes"):
            _validate_required_group(args, "ESE", _ESE_PARAMS)

    def test_two_of_three_raises(self) -> None:
        args = _ns(n_electrodes=32, ese_offset_mm=2.5, ese_reference=None)
        with pytest.raises(SystemExit, match="--ese-reference"):
            _validate_required_group(args, "ESE", _ESE_PARAMS)

    def test_config_all_provided(self) -> None:
        config = Config(
            n_electrodes=32,
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
        )
        _validate_required_group(config, "ESE", _ESE_PARAMS)

    def test_config_coordsystem_n_electrodes(self) -> None:
        config = Config(
            n_electrodes=60,
            ese_offset_mm=5.0,
            ese_reference="electrode_body_center",
        )
        _validate_required_group(config, "ESE", _ESE_PARAMS)

    def test_config_partial_raises(self) -> None:
        config = Config(ese_offset_mm=2.5, ese_reference="electrode_body_center")
        with pytest.raises(SystemExit, match="--ese-offset-mm, --ese-reference"):
            _validate_required_group(config, "ESE", _ESE_PARAMS)


class TestWarnPartialNeighborhood:
    def test_warns_when_ese_not_configured(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = _ns(
            k_neighbors=30,
            neighborhood_radius_mm=None,
            pca_sigma_mm=None,
            min_neighbors=None,
            use_weighted_pca=None,
            n_electrodes=None,
            ese_offset_mm=None,
            ese_reference=None,
        )
        config = Config()
        _warn_partial_neighborhood(args, config)
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "--k-neighbors" in captured.err
        assert "Stage 2 will be skipped" in captured.err

    def test_no_warn_when_ese_configured(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = _ns(
            k_neighbors=30,
            neighborhood_radius_mm=None,
            pca_sigma_mm=None,
            min_neighbors=None,
            use_weighted_pca=None,
            n_electrodes=32,
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
        )
        config = Config(
            n_electrodes=32,
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
        )
        _warn_partial_neighborhood(args, config)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_no_warn_when_no_neighborhood_params(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = _ns(
            k_neighbors=None,
            neighborhood_radius_mm=None,
            pca_sigma_mm=None,
            min_neighbors=None,
            use_weighted_pca=None,
            n_electrodes=None,
            ese_offset_mm=None,
            ese_reference=None,
        )
        config = Config()
        _warn_partial_neighborhood(args, config)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_lists_only_missing_ese_params(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = _ns(
            k_neighbors=30,
            neighborhood_radius_mm=None,
            pca_sigma_mm=None,
            min_neighbors=None,
            use_weighted_pca=None,
            n_electrodes=32,
            ese_offset_mm=None,
            ese_reference=None,
        )
        config = Config(n_electrodes=32)
        _warn_partial_neighborhood(args, config)
        captured = capsys.readouterr()
        assert "--ese-offset-mm" in captured.err
        assert "--ese-reference" in captured.err
        assert "--n-electrodes" not in captured.err
