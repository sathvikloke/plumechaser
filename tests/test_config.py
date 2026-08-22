"""Config loading/validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plumechaser.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = REPO_ROOT / "config" / "default.yaml"


def _write_variant(tmp_path: Path, mutate) -> Path:
    base = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    mutate(base)
    p = tmp_path / "variant.yaml"
    p.write_text(yaml.safe_dump(base))
    return p


def test_default_config_loads(tmp_path):
    cfg = load_config(DEFAULT)
    assert set(cfg.basins) == {"korpezhe", "permian", "region_c"}
    assert cfg.basins["korpezhe"].role == "champion"
    assert cfg.basins["region_c"].role == "coverage"
    assert cfg.tropomi.persistence_passes == 2
    assert cfg.ime.ueff_slope == 0.33


def test_missing_champion_rejected(tmp_path):
    def strip_champion(base):
        for b in base["basins"].values():
            b["role"] = "coverage"

    with pytest.raises(ConfigError, match="champion"):
        load_config(_write_variant(tmp_path, strip_champion))


def test_inverted_bbox_rejected(tmp_path):
    def invert(base):
        base["basins"]["korpezhe"]["bbox"] = [60.0, 38.5, 58.0, 40.0]

    with pytest.raises(ConfigError, match="bbox"):
        load_config(_write_variant(tmp_path, invert))


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/config.yaml")
