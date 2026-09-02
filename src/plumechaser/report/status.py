"""Which cached results may be quoted, and which are audit trail only.

Single source of truth for every consumer of ``bundles/`` -- the replay
dashboard, the figure builder, the flux auditor, and the bundle writer. It
lives here rather than in any one of them because a denylist that exists in
three copies is a denylist that will disagree with itself.

The 2026-08-25 flux audit withdrew the MARS-S2L detections produced under the
input-scale bug (see docs/SUPERSEDED_RESULTS.md). Those bundles stay on disk
on purpose: they are the evidence that the audit happened. What must not
happen is one of them being read back as a current result.

Why the ids are committed here rather than read from the bundles
---------------------------------------------------------------
``bundles/`` is gitignored, and ``report.bundle.write_bundle`` rewrites
``provenance.json`` wholesale, so any flag written into a bundle can be lost
by a re-clone or destroyed by a re-run. The committed list survives both.
On-disk flags are still honoured, as an additional signal, never as the only
one. Classification fails closed: unreadable or unrecognised is not quotable.
"""

from __future__ import annotations

__all__ = [
    "DIAGNOSTIC_EVENT_IDS",
    "STATUS_DIAGNOSTIC",
    "STATUS_QUOTABLE",
    "STATUS_UNREADABLE",
    "STATUS_WITHDRAWN",
    "WITHDRAWN_EVENT_IDS",
    "bundle_status",
    "is_quotable",
]

# Retracted: produced from 0-1 reflectance where MARS-S2L requires DN
# (reflectance x 10000), which invalidated both the scene scores and the
# fluxes. Superseded by -v6 (Korpezhe) and -v4 (Permian).
WITHDRAWN_EVENT_IDS = frozenset({
    "EVT-20260805-K26-MARSS2L",
    "EVT-20260805-K26-MARSS2L-v2",
    "EVT-20260427-P82-MARSS2L",
})

# Intermediate audit steps, each isolating a single variable. Individually
# they look like results -- v5 records scene_score 0.9985 -- and none is one.
DIAGNOSTIC_EVENT_IDS = frozenset({
    "EVT-20260805-K26-MARSS2L-v3",
    "EVT-20260805-K26-MARSS2L-v4",
    "EVT-20260805-K26-MARSS2L-v5",
    "EVT-20260427-P82-MARSS2L-v2",
    "EVT-20260427-P82-MARSS2L-v3",
})

STATUS_QUOTABLE = "quotable"
STATUS_WITHDRAWN = "withdrawn"
STATUS_DIAGNOSTIC = "diagnostic"
STATUS_UNREADABLE = "unreadable"


def bundle_status(meta: dict | None, bundle_name: str = "") -> str:
    """Classify one bundle's provenance.

    Args:
        meta: parsed provenance.json, or None if it could not be read.
        bundle_name: directory name, used when provenance lacks an event_id.

    Returns:
        One of the ``STATUS_*`` constants.
    """
    if not isinstance(meta, dict):
        return STATUS_UNREADABLE
    event_id = str(meta.get("event_id") or bundle_name)

    # Committed ids win over on-disk flags. Order matters: bundle markings set
    # `superseded` on diagnostic runs too, so the specific classification has
    # to be consulted before the generic "not current" flag, or every
    # diagnostic gets mislabelled as a retracted result.
    if event_id in WITHDRAWN_EVENT_IDS:
        return STATUS_WITHDRAWN
    if event_id in DIAGNOSTIC_EVENT_IDS:
        return STATUS_DIAGNOSTIC
    if meta.get("result_status") == "diagnostic":
        return STATUS_DIAGNOSTIC
    if meta.get("superseded") or meta.get("do_not_quote"):
        return STATUS_WITHDRAWN
    return STATUS_QUOTABLE


def is_quotable(meta: dict | None, bundle_name: str = "") -> bool:
    """True only for results that may appear in a figure, table or headline."""
    return bundle_status(meta, bundle_name) == STATUS_QUOTABLE
