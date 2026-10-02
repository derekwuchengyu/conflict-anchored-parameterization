"""Shared config/loaders for exp_logical_bg — the N-vs-N per-logical-scenario
background-collision experiment (one representative concrete scenario per
logical class, 50 ours-LHS renders vs 50 fresh per-centre KDE samples,
common-horizon exposure equalization, full Task-C hit decomposition).

Read-only reuse:
  sr-tlkeep-experiment/scripts/41_validity_layers.py   (vl41: Tracks, grid_traj,
      esmini_agent, score_ego, physics helpers, RED/ISIM/SIMC/HSTATS)
  sr-tlkeep-experiment/scripts/47_bg_diagnosis.py      (bg47: hit_detail,
      _sat_depth, truncate — the Task-C decomposition machinery)
  sr-tlkeep-experiment/core/{svd_param,kde_sampling}.py (subset SVD+KDE fits)
  sr-tlkeep-experiment/data/<subset>/                   (real_meta, vectors)
  exp_cp3d10/results/esmini_runs/<cls>/<sid>/cp3d10_base.xosc (2-param base)
  exp_cp3d10/map_background.py                          (tyms.xodr basemap)
All NEW artifacts stay inside exp_logical_bg/.

Class mapping (logical class -> concrete machinery):
  keeptl : validity keeptl (n=50),  render cls keeptl,  KDE fit subset keeptl
  cutin  : validity cutin88 (n=88), render cls cutin88, KDE fit subset cutin_dir
           (r2l westbound n=34, the largest direction subset per Task B);
           candidates restricted to the 34 westbound scenarios
  tlkeep : validity tlkeep (n=298), render cls tlkeep,  KDE fit subset tlkeep
           (NOTE: the class-level comparisons used KDE samples from the
           all-label-2 906-vector fit; here the tlkeep-298 subset fit is used
           for parallelism with the other two classes — documented deviation)

Criticality band (ego layer): keeptl/tlkeep = minTTC<1.5 s OR |PET|<1 s;
cutin classes = minTTC<1.5 s ONLY (convention from the class-level runs).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
RESULTS = EXP / "results"
FIGS = EXP / "figs"
RUNS = RESULTS / "esmini_runs"

CY = Path("/home/hcis-s19/Documents/ChengYu")
SR = CY / "sr-tlkeep-experiment"
SR_SCRIPTS = SR / "scripts"
SR_RES = SR / "results"
CP3RUNS = CY / "exp_cp3d10" / "results" / "esmini_runs"

FPS = 30.0
SEED = 20260718                 # stats/sampling convention
N_SAMPLES = 50
BOX_MAJOR = 2.0                 # continuous LHS box: Crit_Off_Major in +-2 m
BOX_MINOR = 1.0                 # Crit_Off_Minor in +-1 m
D_KDE = 3                       # reduced dimension of the subset SVD fits
NT, NY, NTH = 50, 2, 1          # scenario-vector layout (04_generate_svd)

# Task-C decomposition bins (47_bg_diagnosis conventions)
GRAZE_F, SHORT_F = 2, 9
TANGENT_M = 0.1
NEAR_S = 2.0

# logical class -> concrete machinery
CLASSES = ["keeptl", "cutin", "tlkeep"]
CFG = {
    "keeptl": dict(vl_cls="keeptl", render_cls="keeptl", fit_subset="keeptl",
                   pet_band=True,
                   desc="left turn (label-1 n2e, n=50)"),
    "cutin":  dict(vl_cls="cutin88", render_cls="cutin88",
                   fit_subset="cutin_dir", pet_band=False,
                   desc="cut-in, r2l westbound direction subset (n=34 of 88)"),
    "tlkeep": dict(vl_cls="tlkeep", render_cls="tlkeep", fit_subset="tlkeep",
                   pet_band=True,
                   desc="keep/straight (label-2 westbound, n=298)"),
}

# figure palette — dataviz reference instance, light mode
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
C_CLEAN = "#2a78d6"            # series blue: bg-clean sample
C_GRAZE = "#ec835a"            # status serious: grazing / near-tangent-only hit
C_SOLID = "#d03b3b"            # status critical: solid hit (>=3 f and >=0.1 m)
C_BG = "#c3c2b7"               # background tracks
GRID = "#e8e7e2"


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SR_SCRIPTS / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cache = {}


def vl41():
    """41_validity_layers as a module (Tracks, grid_traj, esmini_agent, ...)."""
    if "vl41" not in _cache:
        sys.path.insert(0, str(SR_SCRIPTS))
        _cache["vl41"] = _load("vl41", "41_validity_layers.py")
    return _cache["vl41"]


def bg47():
    """47_bg_diagnosis as a module (hit_detail, _sat_depth, truncate)."""
    if "bg47" not in _cache:
        vl41()                                    # 47 re-loads it; keep order
        _cache["bg47"] = _load("bg47", "47_bg_diagnosis.py")
    return _cache["bg47"]


def subset_meta(subset: str):
    import pandas as pd
    return pd.read_csv(SR / "data" / subset / "real_meta.csv")


def subset_vectors(subset: str):
    d = np.load(SR / "data" / subset / "real_vectors.npz", allow_pickle=True)
    return d["X_raw"], np.asarray(d["keys"]).astype(str)


def stored_fit(subset: str) -> dict:
    import json
    p = SR_RES / ("svd_fit.json" if subset == "tlkeep_906"
                  else f"{subset}_svd_fit.json")
    return json.load(open(p))


def base_xosc(render_cls: str, sid: str) -> Path:
    return CP3RUNS / render_cls / sid / "cp3d10_base.xosc"


def lhs_box(n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube over the continuous box [-BOX_MAJOR,+BOX_MAJOR] x
    [-BOX_MINOR,+BOX_MINOR] -> (n, 2) [major, minor]."""
    out = np.empty((n, 2))
    for j, half in enumerate((BOX_MAJOR, BOX_MINOR)):
        strata = (rng.permutation(n) + rng.random(n)) / n     # (0,1)
        out[:, j] = (strata * 2.0 - 1.0) * half
    return out


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))
