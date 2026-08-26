"""Honesty gates are load-bearing, so they get real tests."""

from __future__ import annotations

import numpy as np
import pytest

from plumechaser.config import load_config
from plumechaser.retrieve.gates import evaluate_gates


def _field(rng, sigma_ppb, shape=(120, 120)):
    return rng.normal(0.0, sigma_ppb, size=shape)


def test_clean_retrieval_passes_both_gates():
    rng = np.random.default_rng(20270307)
    field = _field(rng, 12.0)
    mask = np.zeros(field.shape, bool)
    mask[50:58, 50:58] = True  # 64 px of 14400 -> 0.4%

    v = evaluate_gates(field, mask)

    assert not v.artifact_dominated
    assert v.reasons == ()
    assert v.sigma_col_ppb == pytest.approx(12.0, rel=0.15)
    assert v.mask_fraction == pytest.approx(64 / 14400)
    assert "passed" in v.verdict


def test_noisy_background_trips_sigma_gate():
    rng = np.random.default_rng(1)
    field = _field(rng, 210.0)  # the Korpezhe dune-margin regime
    mask = np.zeros(field.shape, bool)
    mask[0:5, 0:5] = True

    v = evaluate_gates(field, mask)

    assert v.artifact_dominated
    assert any("sigma_col" in r for r in v.reasons)
    assert "ARTIFACT-DOMINATED" in v.verdict


def test_oversized_mask_trips_mask_gate_even_with_quiet_background():
    rng = np.random.default_rng(2)
    field = _field(rng, 8.0)
    mask = np.zeros(field.shape, bool)
    mask[:40, :] = True  # 33% of the window

    v = evaluate_gates(field, mask)

    assert v.artifact_dominated
    assert any("plume mask" in r for r in v.reasons)
    # the quiet background must not rescue it
    assert not any("sigma_col" in r for r in v.reasons)


def test_mask_swallowing_window_is_artifact_dominated():
    field = np.ones((30, 30))
    v = evaluate_gates(field, np.ones((30, 30), bool))
    assert v.artifact_dominated
    assert v.mask_fraction == 1.0


def test_limits_are_honoured_and_come_from_config():
    rng = np.random.default_rng(3)
    field = _field(rng, 100.0)
    mask = np.zeros(field.shape, bool)

    assert evaluate_gates(field, mask, sigma_col_ppb_limit=80.0).artifact_dominated
    assert not evaluate_gates(field, mask, sigma_col_ppb_limit=500.0).artifact_dominated

    cfg = load_config("config/default.yaml")
    assert cfg.gates.sigma_col_ppb_limit == 80.0
    assert cfg.gates.mask_fraction_limit == 0.15


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shapes differ"):
        evaluate_gates(np.zeros((4, 4)), np.zeros((5, 5), bool))
    with pytest.raises(ValueError, match="valid shapes differ"):
        evaluate_gates(
            np.zeros((4, 4)), np.zeros((4, 4), bool), valid=np.ones((5, 5), bool)
        )


def test_constant_filled_nodata_cannot_deflate_the_sigma_gate():
    """Half the window filled with a constant must not rescue a noisy scene.

    This is the Korpezhe tile-edge case: ~43% of the retrieval window is
    nodata and gets filled, which a naive MAD would read as very low noise.
    """
    rng = np.random.default_rng(20270307)
    field = rng.normal(0.0, 300.0, size=(100, 100))
    valid = np.ones(field.shape, bool)
    valid[:43, :] = False
    field[~valid] = 0.0  # the constant fill
    mask = np.zeros(field.shape, bool)

    naive = evaluate_gates(field, mask)
    honest = evaluate_gates(field, mask, valid=valid)

    assert honest.sigma_col_ppb > naive.sigma_col_ppb
    assert honest.sigma_col_ppb == pytest.approx(300.0, rel=0.15)
    assert honest.artifact_dominated
    assert honest.n_window_px == int(valid.sum())


def test_mask_fraction_is_relative_to_valid_pixels():
    field = np.zeros((100, 100))
    valid = np.zeros(field.shape, bool)
    valid[:50, :] = True          # 5000 valid px
    mask = np.zeros(field.shape, bool)
    mask[:10, :] = True           # 1000 masked px, all inside the valid half

    v = evaluate_gates(field, mask, valid=valid)

    assert v.n_window_px == 5000
    assert v.mask_fraction == pytest.approx(0.20)
    assert v.artifact_dominated  # 20% > 15% limit


def test_verdict_serialises_for_provenance():
    rng = np.random.default_rng(4)
    v = evaluate_gates(_field(rng, 300.0), np.zeros((120, 120), bool))
    d = v.as_dict()
    assert d["artifact_dominated"] is True
    assert isinstance(d["gate_reasons"], list) and d["gate_reasons"]
    assert set(d) >= {"sigma_col_ppb", "mask_fraction", "n_mask_px", "verdict"}
