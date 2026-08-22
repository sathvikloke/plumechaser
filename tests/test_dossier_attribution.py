"""Dossier rendering + attribution context tests."""

from __future__ import annotations

import pandas as pd

from plumechaser.attribute.context import (
    MULTI_SOURCE_SENTINCEL,
    infrastructure_context,
)
from plumechaser.report.dossier import DossierInput, render_dossier
from plumechaser.retrieve.ime import ImeResult


def _quant():
    return ImeResult(
        q_kg_h=4321.0, ci_low=2100.0, ci_high=8800.0,
        ime_kg=1500.0, ueff_ms=1.77, length_m=640.0, n_pixels=1024,
    )


def test_dossier_renders_quantified(tmp_path):
    d = DossierInput(
        event_id="EVT-2025-001", basin="korpezhe", det_date="2025-06-01",
        lon=59.1, lat=39.2, event_class="persistent",
        z_peak=6.7, persistence_passes=5,
        persistence_dates=["2025-05-28", "2025-06-01"],
        cue_action="cue_sentinel2", cue_reason="persistent source (5 passes)",
        quant=_quant(), u10_ms=4.0, wind_source="ERA5",
        context_verdict="single_candidate",
        context_candidates=[{"name": "Well Pad 7", "type": "oil_gas", "dist_km": 1.2}],
        provenance="config_sha256: abc\ncommit: deadbeef",
    )
    out = render_dossier(d, tmp_path / "d.html")
    html = out.read_text()
    assert "4321.0" in html and "EVT-2025-001" in html
    assert "not attribution" in html and "deadbeef" in html


def test_dossier_detection_only_path(tmp_path):
    d = DossierInput(
        event_id="EVT-2025-002", basin="permian", det_date="2025-08-11",
        lon=-102.3, lat=31.8, event_class="transient", z_peak=9.9,
        persistence_passes=1, persistence_dates=["2025-08-11"],
        cue_action="cue_sentinel2", cue_reason="transient ultra-emission (z=9.9)",
    )
    out = render_dossier(d, tmp_path / "d2.html")
    assert "Detection-only event" in out.read_text()


def test_density_rule_triggers_multi_source(tmp_path):
    rows = [{"name": f"f{i}", "type": "well", "lat": 39.0 + i * 0.001,
             "lon": 59.0 + i * 0.001} for i in range(7)]
    csv = tmp_path / "flight.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    res = infrastructure_context(
        59.0, 39.0, flight_csv=str(csv),
        search_radius_km=5.0, density_rule_per_radius=5,
    )
    assert res.verdict == MULTI_SOURCE_SENTINCEL
    assert len(res.candidates) == 7


def test_single_candidate_and_empty_layer(tmp_path):
    csv = tmp_path / "one.csv"
    pd.DataFrame([{"name": "Pad A", "type": "oil_gas", "lat": 39.001, "lon": 59.001}]).to_csv(
        csv, index=False
    )
    res = infrastructure_context(59.0, 39.0, flight_csv=str(csv))
    assert res.verdict == "single_candidate"
    assert res.candidates.iloc[0]["name"] == "Pad A"

    empty = infrastructure_context(0.0, 0.0)  # no layers supplied
    assert empty.verdict == "no_infrastructure" and empty.candidates.empty
