"""Evidence-dossier generation: one self-contained HTML file per event.

Fields follow the frozen dossier spec: event header, TROPOMI panel, cue-log
excerpt, MBMP map + mask, Q +/- CI with wind provenance, infrastructure
context WITH the density-rule verdict, and a full provenance block
(data versions, code commit,
config hash, render-pack hash).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from jinja2 import Template

TEMPLATE = Template(
    """\
<!doctype html>
<html><head><meta charset="utf-8">
<title>PlumeChaser dossier {{ e.event_id }}</title>
<style>
 body{font-family:-apple-system,Segoe UI,Arial,sans-serif;
      max-width:900px;margin:2rem auto;color:#111}
 h1{font-size:1.4rem} table{border-collapse:collapse;width:100%;margin:0.6rem 0}
 td,th{border:1px solid #ccc;padding:4px 8px;font-size:0.9rem;text-align:left}
 .verdict{background:#fff3cd;padding:8px;border-left:4px solid #b8860b;margin:0.5rem 0}
 .meta{color:#555;font-size:0.78rem;white-space:pre-wrap}
</style></head><body>
<h1>PlumeChaser evidence dossier — {{ e.event_id }}</h1>
<p><b>{{ e.basin }}</b> · detection {{ e.det_date }} ·
lon {{ "%.3f"|format(e.lon) }}, lat {{ "%.3f"|format(e.lat) }} ·
class: {{ e.event_class }}</p>

<h3>TROPOMI screening</h3>
<table>
 <tr><th>peak z-score</th><td>{{ "%.2f"|format(e.z_peak) }}</td></tr>
 <tr><th>persistence</th><td>{{ e.persistence_passes }} passes ({{ e.persistence_dates }})</td></tr>
 <tr><th>cue decision</th><td>{{ e.cue_action }} — {{ e.cue_reason }}</td></tr>
</table>

<h3>Quantification</h3>
{% if e.quant %}
<table>
 <tr><th>rate Q</th><td><b>{{ "%.1f"|format(e.quant.q_kg_h) }} kg/h</b>
     ({{ "%0.0f"|format(e.quant.ci_low) }}–{{ "%0.0f"|format(e.quant.ci_high) }}
     95% MC CI)</td></tr>
 <tr><th>IME / U_eff / L</th><td>{{ "%.0f"|format(e.quant.ime_kg) }} kg ·
     {{ "%.2f"|format(e.quant.ueff_ms) }} m/s · {{ "%.0f"|format(e.quant.length_m) }} m</td></tr>
 <tr><th>wind source</th><td>{{ e.wind_source }} (U10={{ "%.1f"|format(e.u10_ms) }} m/s)</td></tr>
 <tr><th>plume pixels</th><td>{{ e.quant.n_pixels }}</td></tr>
</table>
<p class="meta">Calibration caveat: band-integrated absorption coefficients are
literature-seeded simplifications pending RTM-LUT replacement; rates carry the
IME method's documented 30–90% envelope.</p>
{% else %}
<p>Detection-only event: no clean reference pass within window.</p>
{% endif %}

<h3>Infrastructure context (not attribution)</h3>
<div class="verdict">{{ e.context_verdict }}</div>
{% if e.context_candidates %}
<table>
 <tr><th>name</th><th>type</th><th>dist km</th></tr>
 {% for c in e.context_candidates %}
 <tr><td>{{ c.name }}</td><td>{{ c.type }}</td><td>{{ "%.1f"|format(c.dist_km) }}</td></tr>
 {% endfor %}
</table>
{% endif %}

<h3>Provenance</h3>
<pre class="meta">{{ e.provenance }}</pre>
</body></html>
"""
)


@dataclass
class DossierInput:
    event_id: str
    basin: str
    det_date: str
    lon: float
    lat: float
    event_class: str
    z_peak: float
    persistence_passes: int
    persistence_dates: list[str]
    cue_action: str
    cue_reason: str
    quant: object | None = None          # ImeResult or None (detection-only)
    u10_ms: float | None = None
    wind_source: str = "ERA5"
    context_verdict: str = "no_infrastructure"
    context_candidates: list[dict] = field(default_factory=list)
    provenance: str = ""


def render_dossier(dossier: DossierInput, out_path: str | Path) -> Path:
    """Render the dossier to a standalone HTML file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    e = dict(vars(dossier))
    e["quant"] = asdict(dossier.quant) if dossier.quant is not None else None
    out.write_text(TEMPLATE.render(e=e), encoding="utf-8")
    return out
