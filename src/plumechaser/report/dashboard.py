"""Replay dashboard (Streamlit) over cached event bundles.

Zero live-API dependency by design: the fair laptop reads pre-rendered
bundles (see docs/RUNBOOK.md bundle layout), each carrying provenance.json.
Public-tier coarsening and the 30-day delay rule are enforced here too --
the demo shows resolved cases that are already public via MARS post-delay,
with the courtesy caption baked into the header.
"""

from __future__ import annotations

import json
from pathlib import Path


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
        meta = json.loads(m.read_text())
        rows.append(
            {"bundle": m.parent.name,
             **{k: meta.get(k) for k in ("basin", "det_date", "event_class")}}
        )
    table = pd.DataFrame(rows)
    choice = st.sidebar.selectbox("Event bundle", table["bundle"])
    meta = json.loads((root / choice / "provenance.json").read_text())

    st.subheader(meta.get("event_id", choice))
    cols = st.columns(4)
    cols[0].metric("Basin", str(meta.get("basin")))
    cols[1].metric("Detected", str(meta.get("det_date")))
    cols[2].metric("Class", str(meta.get("event_class")))
    q = meta.get("quant") or {}
    if q:
        cols[3].metric(
            "Rate Q",
            f"{q.get('q_kg_h', 0):,.0f} kg/h",
            f"{q.get('ci_low', 0):,.0f}–{q.get('ci_high', 0):,.0f} CI",
        )

    panels = (("tropomi_png", "TROPOMI screening"), ("mbmp_png", "S2 MBMP dXCH4"))
    for img_key, title in panels:
        p = root / choice / f"{img_key}"
        if p.exists():
            st.image(str(p), caption=title)
        else:
            st.markdown(f"_{title}: not cached in this bundle_")

    dossier = root / choice / "dossier.html"
    if dossier.exists():
        st.download_button("Download evidence dossier (HTML)", dossier.read_bytes(), dossier.name)

    st.json(meta, expanded=False)
