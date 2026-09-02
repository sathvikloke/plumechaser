"""RTM-derived calibration for the MBMP retrieval.

Replaces the single effective absorption coefficients in
``config/default.yaml`` (``mbmp.alpha_*_per_ppb``) with a geometry-dependent
curve measured against UNEP's production radiative-transfer LUT.

Background
----------
The 2026-08-25 flux audit measured our simplified coefficients against
marss2l's RTM LUT and found they understate methane columns by **2.5x to
6.3x** depending on solar geometry (median 4.4x). Two consequences made a
fix mandatory before the analysis-plan freeze:

1. Every ppb our own chain reports is too small by a geometry-dependent
   factor, which propagates into sigma_col, the observability atlas, and any
   IME flux.
2. The honesty gate is expressed in ppb, so the *same* physical scene noise
   produced two different gate verdicts depending on which chain measured
   it. A threshold is not meaningful until the two scales agree.

The spread is the point: a single scalar correction would be wrong by up to
2.5x at the extremes, which is why this is a curve and not a new constant.

Parameterisation
----------------
Our retrieval works in log space,
``x = ln(B11/B12)_target - ln(B11/B12)_reference``, and the stored fit is

    dXCH4_ppb = c1*x + c2*x^2 + c3*x^3

forced through the origin, per (satellite, SZA, VZA). ``c1`` is the
linear-regime sensitivity; the simplified chain's equivalent is
``1 / (alpha_b12 - alpha_b11)``. The quadratic and cubic terms matter: the
RTM is markedly non-linear, and a purely linear coefficient underestimates
strong plumes by tens of percent.

Provenance and citation
-----------------------
``config/rtm_calibration.json`` holds OUR fits to a curve we sampled; it is
not a copy of anyone's lookup table. Regenerate with
``scripts/calibrate_alpha.py`` (needs ``.venv-mars``). The physics is Varon
et al. (2021, AMT 14, 2771) and Gorrono et al. (2023, AMT 16, 89); the LUT
sampled ships with ``marss2l`` (LGPL-3.0, UNEP/IMEO) and must be cited
wherever these numbers appear.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "CalibrationError",
    "RtmCalibration",
    "load_calibration",
    "simplified_c1",
]

DEFAULT_PATH = Path("config/rtm_calibration.json")


class CalibrationError(RuntimeError):
    """Raised when the RTM calibration file is missing or malformed."""


def simplified_c1(alpha_b12_per_ppb: float, alpha_b11_per_ppb: float) -> float:
    """The simplified chain's ppb-per-unit-log-ratio, for comparison."""
    d_alpha = alpha_b12_per_ppb - alpha_b11_per_ppb
    if d_alpha <= 0:
        raise ValueError("alpha_b12 must exceed alpha_b11")
    return 1.0 / d_alpha


@dataclass(frozen=True)
class RtmCalibration:
    """Geometry-dependent ppb-from-log-ratio curve, per satellite."""

    satellites: tuple[str, ...]
    sza_grid: np.ndarray
    vza_grid: np.ndarray
    # coeffs[sat][i_sza, j_vza, k] for k in (c1, c2, c3)
    coeffs: dict[str, np.ndarray]
    max_relative_residual: float
    marss2l_version: str

    def _interp_coeffs(self, satellite: str, sza: float, vza: float) -> np.ndarray:
        sat = satellite.upper()
        if sat not in self.coeffs:
            # S2C shares the MSI design; fall back rather than fail a campaign.
            sat = "S2B" if "S2B" in self.coeffs else self.satellites[0]
        table = self.coeffs[sat]

        # Clamp to the measured envelope; extrapolating an RTM fit is worse
        # than admitting the edge value.
        s = float(np.clip(sza, self.sza_grid[0], self.sza_grid[-1]))
        v = float(np.clip(vza, self.vza_grid[0], self.vza_grid[-1]))

        i = int(np.clip(np.searchsorted(self.sza_grid, s) - 1, 0, len(self.sza_grid) - 2))
        j = int(np.clip(np.searchsorted(self.vza_grid, v) - 1, 0, len(self.vza_grid) - 2))
        s0, s1 = self.sza_grid[i], self.sza_grid[i + 1]
        v0, v1 = self.vza_grid[j], self.vza_grid[j + 1]
        ts = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
        tv = 0.0 if v1 == v0 else (v - v0) / (v1 - v0)

        c00, c01 = table[i, j], table[i, j + 1]
        c10, c11 = table[i + 1, j], table[i + 1, j + 1]
        return (c00 * (1 - ts) * (1 - tv) + c01 * (1 - ts) * tv
                + c10 * ts * (1 - tv) + c11 * ts * tv)

    def ppb_from_log_ratio(
        self,
        log_ratio: np.ndarray | float,
        satellite: str,
        sza: float,
        vza: float,
    ) -> np.ndarray | float:
        """Column enhancement (ppb) from ``ln(B11/B12)_t - ln(B11/B12)_ref``.

        Negative x (a ratio moving the other way) is mapped through the same
        curve with sign preserved, so noise stays symmetric about zero rather
        than being rectified into a positive bias.
        """
        c1, c2, c3 = self._interp_coeffs(satellite, sza, vza)
        x = np.asarray(log_ratio, dtype=np.float64)
        mag = np.abs(x)
        out = np.sign(x) * (c1 * mag + c2 * mag**2 + c3 * mag**3)
        return float(out) if np.isscalar(log_ratio) or out.ndim == 0 else out

    def c1(self, satellite: str, sza: float, vza: float) -> float:
        """Linear-regime ppb per unit log-ratio at this geometry."""
        return float(self._interp_coeffs(satellite, sza, vza)[0])

    def effective_d_alpha(self, satellite: str, sza: float, vza: float) -> float:
        """``alpha_b12 - alpha_b11`` equivalent implied by the RTM at this geometry."""
        return 1.0 / self.c1(satellite, sza, vza)


@lru_cache(maxsize=4)
def load_calibration(path: str | Path = DEFAULT_PATH) -> RtmCalibration:
    """Load and index the measured calibration. Cached per path."""
    p = Path(path)
    if not p.exists():
        raise CalibrationError(
            f"{p} not found — regenerate with scripts/calibrate_alpha.py "
            f"(requires .venv-mars)"
        )
    try:
        raw = json.loads(p.read_text())
        entries = raw["entries"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise CalibrationError(f"{p} is malformed: {exc}") from exc
    if not entries:
        raise CalibrationError(f"{p} contains no entries")

    sats = tuple(sorted({e["satellite"] for e in entries}))
    szas = np.array(sorted({e["sza"] for e in entries}), dtype=float)
    vzas = np.array(sorted({e["vza"] for e in entries}), dtype=float)

    coeffs: dict[str, np.ndarray] = {
        s: np.full((len(szas), len(vzas), 3), np.nan) for s in sats
    }
    for e in entries:
        i = int(np.searchsorted(szas, e["sza"]))
        j = int(np.searchsorted(vzas, e["vza"]))
        coeffs[e["satellite"]][i, j] = (e["c1"], e["c2"], e["c3"])

    for s, table in coeffs.items():
        if np.isnan(table).any():
            raise CalibrationError(
                f"{p}: incomplete (satellite, SZA, VZA) grid for {s}"
            )

    return RtmCalibration(
        satellites=sats,
        sza_grid=szas,
        vza_grid=vzas,
        coeffs=coeffs,
        max_relative_residual=float(raw.get("max_relative_residual", float("nan"))),
        marss2l_version=str(raw.get("marss2l_version", "unknown")),
    )
