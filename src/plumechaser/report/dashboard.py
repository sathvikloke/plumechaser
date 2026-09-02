"""Replay dashboard (Streamlit) over cached event bundles.

Zero live-API dependency by design: the fair laptop reads pre-rendered
bundles (see docs/RUNBOOK.md bundle layout), each carrying provenance.json.
Public-tier coarsening and the 30-day delay rule are enforced here too --
the demo shows resolved cases that are already public via MARS post-delay,
with the courtesy caption baked into the header.

Withdrawn results
-----------------
This is the screen a judge looks at, so it must never present a retracted
number as current. The 2026-08-25 audit withdrew the MARS-S2L detections
produced under the input-scale bug (docs/SUPERSEDED_RESULTS.md), and those
bundles are still on disk deliberately, as the audit trail. Classification
comes from ``report.status``, the single source of truth shared with the
figure builder and the flux auditor.
"""

from __future__ import annotations

import json
from pathlib import Path

from plumechaser.report.status import (
    DIAGNOSTIC_EVENT_IDS,
    STATUS_DIAGNOSTIC,
    STATUS_QUOTABLE,
    STATUS_UNREADABLE,
    STATUS_WITHDRAWN,
    WITHDRAWN_EVENT_IDS,
    bundle_status,
)

__all__ = [
    "DIAGNOSTIC_EVENT_IDS",
    "STATUS_DIAGNOSTIC",
    "STATUS_QUOTABLE",
    "STATUS_UNREADABLE",
    "STATUS_WITHDRAWN",
    "WITHDRAWN_EVENT_IDS",
    "bundle_status",
    "run_dashboard",
]


def _read_meta(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_dashboard(bundles_dir: str | Path = "bundles") -> None:  # pragma: no cover
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="PlumeChaser replay", page_icon="🛰", layout="wide")
    st.title("PlumeChaser — methane super-emitter replay")
    st.caption(
        "Replays cached analysis bundles. Public tier applies MARS-style "
        "courtesy: coordinates coarsened to 0.01°, events ≥30 days old."
    )

    root = Path(bundles_dir)
    if not root.exists():
        st.warning(f"No bundles directory found at {root.resolve()}")
        return
    manifests = sorted(root.glob("*/provenance.json"))
    if not manifests:
        st.info("Bundles appear here once a hindcast run has produced them.")
        return

    rows = []
    for m in manifests:
        meta = _read_meta(m)
        status = bundle_status(meta, m.parent.name)
        rows.append({
            "bundle": m.parent.name,
            "status": status,
            **{k: (meta or {}).get(k) for k in ("basin", "det_date", "event_class")},
        })
    table = pd.DataFrame(rows)

    quotable = table[table["status"] == STATUS_QUOTABLE]["bundle"].tolist()
    others = table[table["status"] != STATUS_QUOTABLE]["bundle"].tolist()

    show_all = st.sidebar.checkbox(
        "Show withdrawn / diagnostic bundles", value=False,
        help="Retracted and intermediate runs are kept as the audit trail. "
             "They are hidden by default so they cannot be shown by accident.",
    )
    # Quotable bundles first, so index 0 is never a retracted result.
    options = quotable + (others if show_all else [])
    if not options:
        st.error(
            "No quotable bundles. Every bundle on disk is withdrawn or "
            "diagnostic — see docs/SUPERSEDED_RESULTS.md."
        )
        return

    choice = st.sidebar.selectbox("Event bundle", options)
    if others:
        st.sidebar.caption(f"{len(others)} bundle(s) hidden as withdrawn/diagnostic.")

    meta = _read_meta(root / choice / "provenance.json") or {}
    status = bundle_status(meta, choice)

    if status == STATUS_WITHDRAWN:
        st.error(
            "**WITHDRAWN RESULT — DO NOT QUOTE.** Produced before the "
            "2026-08-25 input-scale fix. Retained only as the audit trail; "
            "see docs/SUPERSEDED_RESULTS.md for what replaced it."
        )
    elif status == STATUS_DIAGNOSTIC:
        st.warning(
            "**DIAGNOSTIC RUN — NOT A RESULT.** One step of the flux audit, "
            "isolating a single variable. Not a finding on its own."
        )
    elif status == STATUS_UNREADABLE:
        st.error("Unreadable provenance.json — treated as not quotable.")

    st.subheader(meta.get("event_id", choice))
    cols = st.columns(4)
    cols[0].metric("Basin", str(meta.get("basin")))
    cols[1].metric("Detected", str(meta.get("det_date")))
    cols[2].metric("Class", str(meta.get("event_class")))

    # Rule 4: absolute fluxes stay unquotable while the audit is open, and a
    # withdrawn bundle never shows one regardless.
    q = meta.get("quant") or {}
    if q and status == STATUS_QUOTABLE:
        cols[3].metric("Rate Q", "withheld", "under audit")
        st.info(
            "Absolute emission rates are withheld pending the flux audit "
            "(docs/S2_REAL_DATA_FINDINGS.md). Detection and evidence stand."
        )
    elif q:
        cols[3].metric("Rate Q", "withheld", "withdrawn run")

    gates = meta.get("gates") or {}
    if gates:
        verdict = gates.get("verdict", "")
        (st.warning if gates.get("artifact_dominated") else st.success)(
            f"Honesty gates: {verdict}"
        )
        for reason in gates.get("gate_reasons", []):
            st.caption(f"• {reason}")

    panels = (("tropomi_png", "TROPOMI screening"), ("mbmp_png", "S2 MBMP dXCH4"))
    for img_key, title in panels:
        # Bundles write "<key>.png"; matching the bare key found nothing.
        candidates = [root / choice / f"{img_key}.png", root / choice / img_key]
        shown = next((p for p in candidates if p.exists()), None)
        if shown is not None:
            st.image(str(shown), caption=title)
        else:
            st.markdown(f"_{title}: not cached in this bundle_")

    dossier = root / choice / "dossier.html"
    if dossier.exists() and status == STATUS_QUOTABLE:
        st.download_button(
            "Download evidence dossier (HTML)", dossier.read_bytes(), dossier.name
        )
    elif dossier.exists():
        st.caption("Dossier download disabled for withdrawn/diagnostic bundles.")

    if status == STATUS_QUOTABLE:
        st.json(meta, expanded=False)
    else:
        with st.expander("Raw provenance (contains withdrawn values)"):
            st.json(meta, expanded=False)
