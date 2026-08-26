"""Configuration loading and validation.

The YAML file under config/ is the single source of truth. Every module
receives typed dataclasses instead of raw dicts so that a malformed config
fails fast at startup rather than deep inside a hindcast run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing required keys or has bad values."""


@dataclass(frozen=True)
class Basin:
    name: str
    role: str  # "champion" | "coverage"
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    surface_class: str
    elevation_hpa: float


@dataclass(frozen=True)
class TropomiCfg:
    collection: str
    band: str
    qa_band: str
    screening_pixel_size_m: int
    min_qa: float
    climatology_window_days: int
    z_threshold: float
    min_blob_pixels: int
    persistence_passes: int
    persistence_gap_days: int


@dataclass(frozen=True)
class ReferenceCfg:
    window_before_days: int
    window_after_days: int
    margin_days: int
    min_surface_corr: float
    max_reference_mbsp_sigma: float
    w_cloud: float
    w_corr: float
    w_proximity: float


@dataclass(frozen=True)
class Sentinel2Cfg:
    collection: str
    pixel_size_m: int
    max_cloud_fraction: float
    reference: ReferenceCfg


@dataclass(frozen=True)
class MbmpCfg:
    alpha_b12_per_ppb: float
    alpha_b11_per_ppb: float
    plume_threshold_sigma: float
    morphological_median_size: int


@dataclass(frozen=True)
class ImeCfg:
    ueff_slope: float
    ueff_intercept: float
    mc_samples: int
    wind_noise_frac: float
    mask_inclusion_prob: float
    retrieval_noise_ppb: float
    ci_percentiles: tuple[float, float]


@dataclass(frozen=True)
class GatesCfg:
    """Honesty-gate limits, shared by every retrieval path.

    Defaults reproduce the pre-registered values so that a config written
    before the ``gates`` section existed still loads and still gates.
    """

    sigma_col_ppb_limit: float = 80.0
    mask_fraction_limit: float = 0.15


@dataclass(frozen=True)
class EvaluationCfg:
    match_radius_km: float
    match_window_days: int
    cluster_distance_km: float
    cluster_window_days: int
    bootstrap_draws: int
    random_seed: int


@dataclass(frozen=True)
class Paths:
    mirrors: Path
    manifests: Path
    outputs: Path
    bundles: Path


@dataclass(frozen=True)
class Config:
    basins: dict[str, Basin]
    tropomi: TropomiCfg
    sentinel2: Sentinel2Cfg
    mbmp: MbmpCfg
    ime: ImeCfg
    evaluation: EvaluationCfg
    paths: Paths
    gates: GatesCfg = field(default_factory=GatesCfg)
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _require(d: dict[str, Any], *keys: str) -> Any:
    node: Any = d
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise ConfigError(f"missing config key: {'.'.join(keys)}")
        node = node[k]
    return node


def load_config(path: str | Path = "config/default.yaml") -> Config:
    """Load and validate the pipeline configuration."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    basins: dict[str, Basin] = {}
    for name, spec in _require(raw, "basins").items():
        bbox = tuple(float(x) for x in spec["bbox"])
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ConfigError(f"basin '{name}': invalid bbox {bbox}")
        basins[name] = Basin(
            name=name,
            role=spec["role"],
            bbox=bbox,  # type: ignore[arg-type]
            surface_class=spec["surface_class"],
            elevation_hpa=float(spec["elevation_hpa"]),
        )
    if not any(b.role == "champion" for b in basins.values()):
        raise ConfigError("at least one basin must have role='champion'")

    t = raw["tropomi"]
    tropomi = TropomiCfg(
        collection=t["collection"],
        band=t["band"],
        qa_band=t["qa_band"],
        screening_pixel_size_m=int(t["screening_pixel_size_m"]),
        min_qa=float(t["min_qa"]),
        climatology_window_days=int(t["climatology_window_days"]),
        z_threshold=float(t["z_threshold"]),
        min_blob_pixels=int(t["min_blob_pixels"]),
        persistence_passes=int(t["persistence"]["min_passes"]),
        persistence_gap_days=int(t["persistence"]["max_gap_days"]),
    )

    r = raw["sentinel2"]["reference"]
    s2 = Sentinel2Cfg(
        collection=raw["sentinel2"]["collection"],
        pixel_size_m=int(raw["sentinel2"]["pixel_size_m"]),
        max_cloud_fraction=float(raw["sentinel2"]["max_cloud_fraction"]),
        reference=ReferenceCfg(
            window_before_days=int(r["window_before_days"]),
            window_after_days=int(r["window_after_days"]),
            margin_days=int(r["margin_days"]),
            min_surface_corr=float(r["min_surface_corr"]),
            max_reference_mbsp_sigma=float(r["max_reference_mbsp_sigma"]),
            w_cloud=float(r["weights"]["cloud"]),
            w_corr=float(r["weights"]["corr"]),
            w_proximity=float(r["weights"]["proximity"]),
        ),
    )

    m = raw["mbmp"]
    mbmp = MbmpCfg(
        alpha_b12_per_ppb=float(m["alpha_b12_per_ppb"]),
        alpha_b11_per_ppb=float(m["alpha_b11_per_ppb"]),
        plume_threshold_sigma=float(m["plume_threshold_sigma"]),
        morphological_median_size=int(m["morphological_median_size"]),
    )

    i = raw["ime"]
    ime = ImeCfg(
        ueff_slope=float(i["ueff_slope"]),
        ueff_intercept=float(i["ueff_intercept"]),
        mc_samples=int(i["mc_samples"]),
        wind_noise_frac=float(i["wind_noise_frac"]),
        mask_inclusion_prob=float(i["mask_inclusion_prob"]),
        retrieval_noise_ppb=float(i["retrieval_noise_ppb"]),
        ci_percentiles=(float(i["ci_percentiles"][0]), float(i["ci_percentiles"][1])),
    )

    e = raw["evaluation"]
    evaluation = EvaluationCfg(
        match_radius_km=float(e["match_radius_km"]),
        match_window_days=int(e["match_window_days"]),
        cluster_distance_km=float(e["cluster_distance_km"]),
        cluster_window_days=int(e["cluster_window_days"]),
        bootstrap_draws=int(e["bootstrap_draws"]),
        random_seed=int(e["random_seed"]),
    )

    g = raw.get("gates") or {}
    gates = GatesCfg(
        sigma_col_ppb_limit=float(g.get("sigma_col_ppb_limit", 80.0)),
        mask_fraction_limit=float(g.get("mask_fraction_limit", 0.15)),
    )
    if gates.sigma_col_ppb_limit <= 0 or not 0 < gates.mask_fraction_limit <= 1:
        raise ConfigError(f"gates: implausible limits {gates}")

    paths = Paths(
        mirrors=Path(_require(raw, "paths", "mirrors")),
        manifests=Path(_require(raw, "paths", "manifests")),
        outputs=Path(_require(raw, "paths", "outputs")),
        bundles=Path(_require(raw, "paths", "bundles")),
    )

    return Config(
        basins=basins,
        tropomi=tropomi,
        sentinel2=s2,
        mbmp=mbmp,
        ime=ime,
        evaluation=evaluation,
        paths=paths,
        gates=gates,
        raw=raw,
    )


def config_sha256(cfg_path: str | Path = "config/default.yaml") -> str:
    """Stable hash of the config file, recorded in every run manifest."""
    return hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()
