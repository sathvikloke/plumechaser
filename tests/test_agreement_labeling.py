"""End-to-end agreement study, clustering, labeling protocol, atlas limits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plumechaser.atlas.limits import limit_surface, min_detectable_rate
from plumechaser.evaluate.agreement import assign_branch, cluster_ids, run_agreement
from plumechaser.labeling.queue import (
    build_render_pack,
    cohen_kappa,
    collect_labels,
    control_hit_rate,
)


def _events(rows):
    df = pd.DataFrame(rows, columns=["id", "date", "lon", "lat"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


class TestClustering:
    def test_spatial_temporal_grouping(self):
        events = _events(
            [
                ("a", "2025-01-01", 59.0, 39.0),
                ("b", "2025-01-02", 59.1, 39.05),   # same cluster as a
                ("c", "2025-06-01", 59.0, 39.0),     # too late -> new cluster
                ("d", "2025-01-02", 20.0, 10.0),     # far away -> new cluster
            ]
        )
        ids = cluster_ids(events, distance_km=25.0, window_days=14)
        assert ids[events.index[events["id"] == "a"][0]] == ids[
            events.index[events["id"] == "b"][0]
        ]
        assert len(set(ids)) == 3


class TestAgreementStudy:
    def _run(self):
        dets = _events(
            [
                ("d1", "2025-03-02", 59.00, 39.00),
                ("d2", "2025-04-11", 59.20, 39.30),
                ("d3", "2025-07-08", 4.60, 31.40),   # false positive
            ]
        )
        strict = _events(
            [
                ("r1", "2025-03-03", 59.05, 39.05),
                ("r2", "2025-07-09", 55.00, 25.00),  # no nearby detection
            ]
        )
        return run_agreement(
            dets, strict, None,
            radius_km=25.0, window_days=3,
            cluster_distance_km=25.0, cluster_window_days=14,
            bootstrap_draws=200, seed=7,
        )

    def test_strict_mode_metrics(self):
        reports = self._run()
        rep = reports["strict"]
        m = rep.metrics
        assert m["n_detected"] == 3 and m["n_reference"] == 2
        assert m["n_matched"] == 1          # d1<->r1; r2 unmatched; d3 FP
        assert m["precision"] == pytest.approx(1 / 3)
        assert m["recall"] == pytest.approx(1 / 2)
        assert rep.precision_ci is not None and rep.recall_ci is not None

    def test_sensitivity_grid_and_branches(self):
        rep = self._run()["strict"]
        assert len(rep.sensitivity) == 9
        assert set(rep.sensitivity.columns) >= {"cluster_dist_factor", "precision"}
        assert rep.interpretation.startswith("Strict mode benchmarks")

    def test_branch_assignment_rule(self):
        assert assign_branch(45) == "full"
        assert assign_branch(20) == "intermediate"
        assert assign_branch(5) == "descriptive"


class TestLabelingProtocol:
    def _pack(self, tmp_path):
        scenes = {f"cand{i}": np.random.default_rng(i).normal(0, 10, (8, 8)) for i in range(5)}
        controls = {"ctrl_known_plume": np.full((8, 8), 50.0)}
        return build_render_pack(scenes, tmp_path / "pack", controls=controls)

    def test_pack_manifest_controls_flagged_and_hashed(self, tmp_path):
        pack = self._pack(tmp_path)
        import json

        manifest = json.loads((pack / "manifest.json").read_text())
        assert manifest["pack_sha256"]
        flags = [e["is_control"] for e in manifest["entries"]]
        assert sum(flags) == 1 and len(flags) == 6   # 5 candidates + 1 control

    def test_kappa_and_control_hit_rate(self, tmp_path):
        pack = self._pack(tmp_path)
        labels = pd.DataFrame(
            {
                "id": ["cand0", "cand1", "ctrl_known_plume"],
                "pass1_morphology": ["plume_like", "not_plume", "plume_like"],
                "pass2_final": ["real_plume", "artifact", "real_plume"],
            }
        )
        saved = tmp_path / "labels.csv"
        labels.to_csv(saved, index=False)
        loaded = collect_labels(saved)
        assert cohen_kappa(["p", "p"], ["p", "p"]) == 1.0
        assert cohen_kappa(["p", "a"], ["a", "p"]) < 0
        assert control_hit_rate(loaded, pack / "manifest.json") == 1.0


class TestAtlasLimits:
    def test_min_rate_monotonicity(self):
        base = dict(
            sigma_col_ppb=12.0, k_sigma=3.0, min_pixels=3, pixel_area_m2=1113.0**2,
            u10_ms=3.0, typical_plume_length_m=800.0,
        )
        q0 = min_detectable_rate(**base)
        assert q0 > 0
        assert min_detectable_rate(**{**base, "sigma_col_ppb": 24.0}) > q0
        assert min_detectable_rate(**{**base, "min_pixels": 6}) > q0
        assert min_detectable_rate(**{**base, "u10_ms": 6.0}) > q0

    def test_limit_surface_structure(self):
        surface = limit_surface(
            {"korpezhe": {"surface_class": "homogeneous_arid"},
             "permian": {"surface_class": "heterogeneous"}},
            seasons=["JJA", "DJF"],
            u10_by_basin_season={("korpezhe", "JJA"): 4.0},
            sigma_by_class_season={
                ("homogeneous_arid", "JJA"): 12.0,
                ("heterogeneous", "JJA"): 25.0,
            },
            k_sigma=3.0, min_pixels=3, pixel_size_m=1113,
            lengths_by_class={"homogeneous_arid": 800.0, "heterogeneous": 1200.0},
        )
        assert set(surface) == {"korpezhe", "permian"}
        assert set(surface["korpezhe"]) == {"JJA"}       # missing season skipped
