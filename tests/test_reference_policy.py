"""Reference-pass selector + cue policy + persistence verification tests."""

from __future__ import annotations

from datetime import date

from plumechaser.cue.policy import CueDecision, decide_cue, write_cue_manifest
from plumechaser.cue.reference import CandidatePass, select_reference_pass
from plumechaser.detect.verify import Candidate, persist_candidates


def _pass(pid="p1", dt=5, cloud=0.0, corr=0.99, mbsp=0.5):
    return CandidatePass(pass_id=pid, dt_days=dt, cloud_fraction=cloud,
                         surface_corr=corr, mbsp_abs_sigma=mbsp)


class TestReferenceSelector:
    def test_picks_best_score(self):
        good = _pass("clean", dt=6, cloud=0.01)
        meh = _pass("meh", dt=11, cloud=0.04, corr=0.96)
        v = select_reference_pass([good, meh])
        assert v.chosen == "clean"

    def test_rejection_reasons(self):
        cands = [
            _pass("inside", dt=1),
            _pass("far", dt=20),
            _pass("cloudy", dt=6, cloud=0.4),
            _pass("unstable", dt=6, corr=0.80),
            _pass("plumed", dt=6, mbsp=9.0),
            _pass("ok", dt=7),
        ]
        v = select_reference_pass(cands)
        assert v.chosen == "ok"
        assert set(v.rejected) == {"inside", "far", "cloudy", "unstable", "plumed"}

    def test_all_rejected_returns_none(self):
        v = select_reference_pass([_pass("bad", dt=1)])
        assert v.chosen is None and v.score is None


class TestCuePolicy:
    def test_persistent_cued(self):
        action, reason = decide_cue(persistence_passes=5, z_peak=3.2)
        assert action == "cue_sentinel2" and "persistent" in reason

    def test_strong_transient_cued(self):
        action, _ = decide_cue(persistence_passes=1, z_peak=8.0)
        assert action == "cue_sentinel2"

    def test_weak_transient_watched(self):
        action, reason = decide_cue(persistence_passes=1, z_peak=3.5)
        assert action == "watch" and "weak" in reason


class TestPersistence:
    def test_single_pass_rejected_multi_confirmed(self):
        dets = [
            Candidate("a", date(2025, 1, 1), 59.0, 39.0, 4.0, source_pass_ids=["pa"]),
            Candidate("b", date(2025, 1, 3), 59.05, 39.02, 5.0, source_pass_ids=["pb"]),
            Candidate("c", date(2025, 2, 1), 60.0, 40.0, 9.0, source_pass_ids=["pc"]),
        ]
        confirmed, rejected = persist_candidates(dets, min_passes=2, gap_days=3)
        assert [c.candidate_id for c in confirmed] == ["a"]
        assert confirmed[0].score == 5.0
        assert set(confirmed[0].source_pass_ids) == {"pa", "pb"}
        assert [c.candidate_id for c in rejected] == ["c"]

    def test_gap_enforced(self):
        dets = [
            Candidate("a", date(2025, 1, 1), 59.0, 39.0, 4.0),
            Candidate("b", date(2025, 1, 10), 59.0, 39.0, 4.0),
        ]
        confirmed, rejected = persist_candidates(dets, min_passes=2, gap_days=3)
        assert not confirmed and len(rejected) == 2


def test_manifest_roundtrip(tmp_path):
    d = CueDecision(
        decision_id="e1", det_date=date(2025, 6, 1), lon=59.0, lat=39.0,
        persistence_passes=5, class_="persistent", action="cue_sentinel2",
        reason="test", counterfactuals={"x": "y"},
    )
    p = write_cue_manifest([d], tmp_path / "cue.jsonl")
    text = p.read_text()
    assert '"action": "cue_sentinel2"' in text
