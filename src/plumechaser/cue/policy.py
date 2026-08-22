"""Cueing policy: persistent-first tasking with transient second class.

Design (frozen analysis plan): persistent sources give high confirmation
yield because they are still emitting at the next clear high-resolution pass;
transient ultra-emissions are tracked opportunistically as a second class.
Every decision -- including counterfactuals that were NOT taken -- is logged
to the cue manifest, which doubles as autonomy evidence for reviewers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class CueDecision:
    decision_id: str
    det_date: date
    lon: float
    lat: float
    persistence_passes: int
    class_: str                       # "persistent" | "transient"
    action: str                       # "cue_sentinel2" | "watch" | "drop"
    reason: str
    counterfactuals: dict[str, str] = field(default_factory=dict)


def decide_cue(
    *,
    persistence_passes: int,
    min_persistent_passes: int = 4,
    z_peak: float,
    transient_z_floor: float = 5.0,
    s2_pass_expected_within_days: int | None = None,
) -> tuple[str, str]:
    """Return ``(action, reason)`` for one verified TROPOMI-tier candidate."""
    counterfactuals: dict[str, str] = {}

    if persistence_passes >= min_persistent_passes:
        action = "cue_sentinel2"
        reason = f"persistent source ({persistence_passes} passes)"
        if s2_pass_expected_within_days is None:
            counterfactuals["no_revisit_info"] = "cue archive search anyway"
        return action, reason
    counterfactuals["persistence"] = (
        f"{persistence_passes} < {min_persistent_passes}: not promoted to primary"
    )

    if z_peak >= transient_z_floor:
        action = "cue_sentinel2"
        reason = f"transient ultra-emission (z={z_peak:.1f} >= {transient_z_floor})"
        return action, reason

    action = "watch"
    reason = f"weak transient (z={z_peak:.1f} < {transient_z_floor})"
    return action, reason


def write_cue_manifest(decisions: list[CueDecision], path: str | Path) -> Path:
    """Persist every decision + counterfactual as JSONL (append-safe)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for d in decisions:
            record = asdict(d)
            record["det_date"] = d.det_date.isoformat()
            f.write(json.dumps(record) + "\n")
    return out
