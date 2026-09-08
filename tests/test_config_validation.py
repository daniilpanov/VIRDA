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
        args = _ns(ese_offset_mm=2.5, ese_reference="electrode_body_center")
        _validate_required_group(args, "ESE", _ESE_PARAMS)

    def test_none_provided(self) -> None:
        args = _ns(ese_offset_mm=None, ese_reference=None)
        _validate_required_group(args, "ESE", _ESE_PARAMS)

    def test_config_all_provided(self) -> None:
        config = Config(
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
        )
        _validate_required_group(config, "ESE", _ESE_PARAMS)


class TestWarnPartialNeighborhood:
    def test_warns_when_ese_not_configured(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = _ns(
            k_neighbors=30,
            neighborhood_radius_mm=None,
            pca_sigma_mm=None,
            min_neighbors=None,
            use_weighted_pca=None,
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
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
        )
        config = Config(
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
            ese_offset_mm=None,
            ese_reference=None,
        )
        config = Config()
        _warn_partial_neighborhood(args, config)
        captured = capsys.readouterr()
        assert captured.err == ""
