"""
分布相似度 (DISTRIBUTION similarity) — per scenario CATEGORY, does the PET distribution
of the default-value-rendered parameterized scenarios match the real-world one?

PIPELINE
  for every labeled scenario in the category:
    real  PET ← the GT *_pet_optimized.parquet (real_source="parquet", default) or
                recomputed on the scenario window with the same estimator
                (real_source="recompute")
    param PET ← render the scenario with `sample_cfg` (default 5pt-default =
                default-value render), time-parameterize by the real speed profile,
                run the SAME PET estimator (interaction_sim.pet) vs the real ego
  then compare the two samples per category (|PET|, finite values only).

DIVERGENCES (real sample R, param sample P)
  ks_stat, ks_p   two-sample Kolmogorov–Smirnov: sup_x |F_R(x) − F_P(x)|  (scipy)
  wasserstein     1-D earth-mover distance W₁(R, P) in seconds            (scipy)
  jsd             Jensen–Shannon divergence (base 2, ∈[0,1]) between histograms on
                  shared bins over the pooled finite range
  frac_conflict_* share of scenarios with a finite PET (conflict happened) on each
                  side — an inf-PET (no conflict) never enters the histogram, so this
                  reports how often parameterization *destroys/creates* the conflict.

INPUT   dataset (+ group_by ∈ {superclass, scenario_type, agent_class}, sample_cfg,
        estimator, per_class/labels/limit subsetting)
OUTPUT  pet_samples()  → long per-scenario DataFrame (ego, actor, category,
                         real_pet, param_pet, param_pet_type)
        compare()      → per-category DataFrame of divergences + pooled 'ALL' row
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from .. import config as C, criticality as CRIT
from . import core, interaction_sim as IS


def divergences(real_vals, param_vals, bins: int = 20) -> dict:
    """Distribution distances between two 1-D samples (finite values only)."""
    r = np.asarray(real_vals, float)
    p = np.asarray(param_vals, float)
    rf, pf = r[np.isfinite(r)], p[np.isfinite(p)]
    out = {"n_real": int(len(rf)), "n_param": int(len(pf)),
           "frac_conflict_real": float(len(rf) / len(r)) if len(r) else float("nan"),
           "frac_conflict_param": float(len(pf) / len(p)) if len(p) else float("nan"),
           "median_real": float(np.median(rf)) if len(rf) else float("nan"),
           "median_param": float(np.median(pf)) if len(pf) else float("nan"),
           "ks_stat": float("nan"), "ks_p": float("nan"),
           "wasserstein": float("nan"), "jsd": float("nan")}
    if len(rf) < 2 or len(pf) < 2:
        return out
    ks = ks_2samp(rf, pf)
    out["ks_stat"], out["ks_p"] = float(ks.statistic), float(ks.pvalue)
    out["wasserstein"] = float(wasserstein_distance(rf, pf))
    lo, hi = min(rf.min(), pf.min()), max(rf.max(), pf.max())
    if hi - lo < 1e-9:
        out["jsd"] = 0.0
    else:
        edges = np.linspace(lo, hi, bins + 1)
        hr, _ = np.histogram(rf, bins=edges)
        hp, _ = np.histogram(pf, bins=edges)
        out["jsd"] = float(jensenshannon(hr + 1e-12, hp + 1e-12, base=2) ** 2)
    return out


def _real_pet(dataset: str, s: dict, real_source: str, estimator: str) -> float:
    if real_source == "parquet":
        return abs(CRIT.gt_criticality(dataset, s["ego"], s["actor"])["gt_pet"])
    a = core.real_traj(dataset, s["actor"], s["min_frame"], s["max_frame"])
    e = core.real_traj(dataset, s["ego"], s["min_frame"], s["max_frame"])
    if a is None or e is None:
        return float("nan")
    return IS.pet(a, e, estimator)["pet_abs"]


def pet_samples(dataset: str, group_by: str = "superclass",
                sample_cfg: C.SampleConfig | None = None,
                per_class: int | None = None, labels=None, limit: int | None = None,
                estimator: str = "bbox", real_source: str = "parquet",
                verbose: bool = True) -> pd.DataFrame:
    """|PET| per labeled scenario, real vs default-render param (long format).
    real_source: 'parquet' = GT table (whole-track window, the real-world reference);
    'recompute' = same estimator on the scenario window (estimator-consistent)."""
    scenarios = core.list_scenarios(dataset, labels=labels, per_class=per_class,
                                    group_key=group_by, limit=limit)
    rows = []
    for k, s in enumerate(scenarios):
        real_pet = _real_pet(dataset, s, real_source, estimator)
        param_pet, param_type = float("nan"), "error"
        try:
            p = core.param_traj(dataset, s["ego"], s["actor"],
                                s["min_frame"], s["max_frame"], sample_cfg)
            e = core.real_traj(dataset, s["ego"], s["min_frame"], s["max_frame"])
            if p is not None and e is not None:
                pr = IS.pet(p, e, estimator)
                param_pet, param_type = pr["pet_abs"], pr["pet_type"]
        except Exception as exc:                       # keep sweeping on bad scenarios
            param_type = f"error:{type(exc).__name__}"
        rows.append({**{k2: s[k2] for k2 in
                        ("dataset", "ego", "actor", "min_frame", "max_frame", "label")},
                     "category": s[group_by], "real_pet": real_pet,
                     "param_pet": param_pet, "param_pet_type": param_type})
        if verbose and (k + 1) % 25 == 0:
            print(f"  [{k + 1}/{len(scenarios)}] pet_samples...", flush=True)
    return pd.DataFrame(rows)


def compare(dataset: str, group_by: str = "superclass",
            sample_cfg: C.SampleConfig | None = None,
            per_class: int | None = None, labels=None, limit: int | None = None,
            estimator: str = "bbox", real_source: str = "parquet",
            bins: int = 20, samples: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-category PET-distribution divergences (+ pooled 'ALL' row). Pass a
    precomputed `samples` frame (from pet_samples) to skip the sweep."""
    if samples is None:
        samples = pet_samples(dataset, group_by, sample_cfg, per_class, labels,
                              limit, estimator, real_source)
    out = []
    for cat, g in samples.groupby("category"):
        out.append({"dataset": dataset, "category": cat, "n_scenarios": len(g),
                    **divergences(g["real_pet"], g["param_pet"], bins)})
    out.append({"dataset": dataset, "category": "ALL", "n_scenarios": len(samples),
                **divergences(samples["real_pet"], samples["param_pet"], bins)})
    return pd.DataFrame(out)
