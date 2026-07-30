"""Tests for core.ese_config module."""

import json
import tempfile
from pathlib import Path

import pytest

from virda.core.ese_config import ESEConfig


class TestESEConfig:
    def test_default_config(self):
        cfg = ESEConfig()
        assert cfg.offset_mm == 5.0
        assert cfg.reference_point == "center_of_external_surface"

    def test_invalid_offset(self):
        with pytest.raises(ValueError, match="positive"):
            ESEConfig(offset_mm=-1.0)
        with pytest.raises(ValueError, match="positive"):
            ESEConfig(offset_mm=0.0)

    def test_invalid_reference_point(self):
        with pytest.raises(ValueError, match="Unknown"):
            ESEConfig(reference_point="invalid_point")

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            cfg = ESEConfig(offset_mm=7.5, description="test")
            cfg.save(path)

            loaded = ESEConfig.load(path)
            assert loaded.offset_mm == 7.5
            assert loaded.description == "test"

    def test_repr(self):
        cfg = ESEConfig(offset_mm=3.0)
        assert "3.0" in repr(cfg)
