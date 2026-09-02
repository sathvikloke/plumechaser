"""The fair-floor demo must never open on a retracted result.

Found during the 2026-08-25 audit: the replay dashboard listed every bundle
and defaulted to index 0 of a sorted list, which put a withdrawn MARS-S2L
run first. These tests pin the classification that prevents that.
"""

from __future__ import annotations

from plumechaser.report.dashboard import (
    DIAGNOSTIC_EVENT_IDS,
    STATUS_DIAGNOSTIC,
    STATUS_QUOTABLE,
    STATUS_UNREADABLE,
    STATUS_WITHDRAWN,
    WITHDRAWN_EVENT_IDS,
    bundle_status,
)


def test_withdrawn_runs_are_classified_by_committed_id():
    for eid in WITHDRAWN_EVENT_IDS:
        assert bundle_status({"event_id": eid}) == STATUS_WITHDRAWN


def test_diagnostic_runs_are_classified_by_committed_id():
    for eid in DIAGNOSTIC_EVENT_IDS:
        assert bundle_status({"event_id": eid}) == STATUS_DIAGNOSTIC


def test_the_specific_bundle_that_sorted_first_is_withdrawn():
    """`EVT-20260427-P82-MARSS2L` sorts to index 0 and was the default view."""
    assert bundle_status({"event_id": "EVT-20260427-P82-MARSS2L"}) == STATUS_WITHDRAWN


def test_corrected_runs_remain_quotable():
    for eid in ("EVT-20260805-K26-MARSS2L-v6", "EVT-20260427-P82-MARSS2L-v4"):
        assert bundle_status({"event_id": eid}) == STATUS_QUOTABLE


def test_denylist_does_not_depend_on_the_on_disk_flag():
    """bundles/ is gitignored and write_bundle rewrites provenance wholesale.

    A re-run of a superseded event id would strip an on-disk flag, so the
    committed id list has to stand on its own.
    """
    stripped = {"event_id": "EVT-20260427-P82-MARSS2L"}  # no `superseded` key
    assert bundle_status(stripped) == STATUS_WITHDRAWN


def test_on_disk_flag_is_still_honoured_for_future_bundles():
    assert bundle_status({"event_id": "EVT-FUTURE", "superseded": True}) == STATUS_WITHDRAWN
    assert bundle_status(
        {"event_id": "EVT-FUTURE", "result_status": "diagnostic"}
    ) == STATUS_DIAGNOSTIC
    assert bundle_status({"event_id": "EVT-FUTURE", "do_not_quote": True}) == STATUS_WITHDRAWN


def test_diagnostic_beats_a_generic_superseded_flag():
    """Bundle markings set `superseded` on diagnostics too.

    Checking the generic flag first mislabelled every intermediate audit step
    as a retracted result, which overstates how much was withdrawn.
    """
    for eid in DIAGNOSTIC_EVENT_IDS:
        assert bundle_status({"event_id": eid, "superseded": True}) == STATUS_DIAGNOSTIC
    assert bundle_status(
        {"event_id": "EVT-FUTURE", "superseded": True, "result_status": "diagnostic"}
    ) == STATUS_DIAGNOSTIC


def test_classification_fails_closed():
    assert bundle_status(None) == STATUS_UNREADABLE
    assert bundle_status("not a dict") == STATUS_UNREADABLE


def test_bundle_name_is_used_when_event_id_is_missing():
    assert bundle_status({}, "EVT-20260427-P82-MARSS2L") == STATUS_WITHDRAWN
