"""Bundle builder tests (hermetic)."""

from __future__ import annotations

import json

import pytest

from plumechaser.report.bundle import bundle_integrity, write_bundle
from plumechaser.report.dossier import DossierInput


@pytest.fixture
def dossier():
    return DossierInput(
        event_id="EVT-TEST-1", basin="korpezhe", det_date="2025-06-01",
        lon=59.0, lat=39.0, event_class="persistent",
        z_peak=7.4, persistence_passes=4,
        persistence_dates=["2025-05-28", "2025-06-01"],
        cue_action="cue_sentinel2", cue_reason="persistent source (4 passes)",
    )


def test_write_bundle_layout(tmp_path, dossier):
    bdir = write_bundle(
        dossier, tmp_path,
        tropomi_png=b"\x89PNG fake",
        extra={"power_branch": "full"},
    )
    assert (bdir / "provenance.json").exists()
    assert (bdir / "dossier.html").exists()
    assert (bdir / "tropomi_png").read_bytes() == b"\x89PNG fake"

    prov = json.loads((bdir / "provenance.json").read_text())
    assert prov["event_id"] == "EVT-TEST-1"
    assert prov["code_commit"] == "unknown" or len(prov["code_commit"]) >= 7
    assert prov["config_sha256"] and len(prov["config_sha256"]) == 64
    assert prov.get("power_branch") == "full"
    assert "EVT-TEST-1" in (bdir / "dossier.html").read_text()


def test_bundle_integrity_changes_with_content(tmp_path, dossier):
    b1 = write_bundle(dossier, tmp_path)
    d1 = bundle_integrity(b1)

    dossier2 = DossierInput(**{**vars(dossier), "z_peak": 9.9})
    b2 = write_bundle(dossier2, tmp_path / "other")
    d2 = bundle_integrity(b2)
    assert d1 != d2
