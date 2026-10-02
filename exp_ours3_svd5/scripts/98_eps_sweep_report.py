#!/usr/bin/env python3
"""Question (e), part 2: the epsilon-ball sweep FIGURES, the fixed-eps
RECOMMENDATION and the REPORT.

This script is a pure OFFLINE reader.  It recomputes nothing about coverage:
every coverage number it prints is read out of the frozen sweep tables written
by scripts/97_eps_sweep.py

    results/eps_sweep/eps_sweep_coverage.csv     (per-arm curve, 3 spaces)
    results/eps_sweep/eps_sweep_corner.csv       (same, rare 20% reals only)
    results/eps_sweep/eps_sweep_joint_grid.csv   (full 2-D eps_int x eps_path)
    results/eps_sweep/eps_real_nn_values.csv     (per-scenario real-to-real NN)
    results/eps_sweep/eps_real_nn_distributions.csv

plus results/e6_real_reference.csv (per-class descriptor SDs, used only to
translate z units into seconds / metres / m/s) and
results/final/tables3/gen_metrics.csv (the thesis-table cells, used as a gate).

NOTHING EXISTING IS MODIFIED.  New outputs only:
    figures/eps_sweep/fig1..fig8_*.png
    results/eps_sweep/tables3_coverage_at_fixed_eps.{csv,md}
    results/eps_sweep/eps_rank_flips.csv
    results/eps_sweep/eps_regimes.csv
    results/eps_sweep/eps_sweep_report_numbers.json
    results/eps_sweep/EPS_SWEEP_REPORT.md
    results/eps_sweep/EPS_SWEEP_REPORT_zh.md
    results/eps_sweep/98_stdout.txt (when run with the tee wrapper)

TWO DATA HAZARDS handled explicitly (both found by the verifier of 97):
  (H1) In eps_sweep_coverage.csv / eps_sweep_corner.csv the JOINT rows carry
       eps_mult = NaN at ALL 46 eps_int values of each class grid, not only at
       the 31 shared global ones.  Filtering "eps_mult is null" and then
       aggregating across classes silently mixes single-class values.  Every
       cross-class aggregation here filters on eps_int.isin(GRID_INT) instead.
  (H2) The CURRENT per-class rule is the eps_mult == 1.0 row, NOT the joint
       diagonal row that happens to share the same eps_int (the diagonal pairs
       that eps_int with the pooled-quantile partner, a different eps_path).

Run:
  MPLCONFIGDIR=/tmp/ours3_svd5_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hcis-s19/micromamba/envs/nps/bin/python -B scripts/98_eps_sweep_report.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ours3_svd5_mpl")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
OUT = RES / "eps_sweep"
FIG = PROJECT / "figures" / "eps_sweep"
FIG.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# constants
# ─────────────────────────────────────────────────────────────────────────────
# the three arms that HAVE a coverage@1000 cell in the thesis tables.
# svd_d5_kde_executed_E3 is a 4th KDE arm but its pool is 43-100 draws, so it
# has no budget-1000 row at all; it appears only on the budget figure (fig8).
ARMS = ["ours3_disk_kde", "svd_d5_kde_matched_analytic", "sakura_route_kde"]
ARM_B100 = ARMS + ["svd_d5_kde_executed_E3"]
SHORT = {
    "ours3_disk_kde": "Ours+KDE",
    "svd_d5_kde_matched_analytic": "SVD_d5+KDE",
    "sakura_route_kde": "SAKURA-route+KDE",
    "svd_d5_kde_executed_E3": "SVD_d5+KDE (exec E3)",
}
TINY = {"ours3_disk_kde": "Ours", "svd_d5_kde_matched_analytic": "SVD",
        "sakura_route_kde": "SAKURA", "svd_d5_kde_executed_E3": "SVDexec"}
# Okabe-Ito, colour-blind safe; held constant across every figure in this file
COLOR = {
    "ours3_disk_kde": "#0072B2",              # blue
    "svd_d5_kde_matched_analytic": "#D55E00",  # vermillion
    "sakura_route_kde": "#009E73",             # bluish green
    "svd_d5_kde_executed_E3": "#CC79A7",       # reddish purple
}
CLASSES = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr"]
CLASS_LABEL = {
    "tlkeep": "Agent-straight (ego LT)\ntlkeep",
    "keeptl": "Agent-LT N→E\nkeeptl",
    "keeptl_sw": "Agent-LT S→W\nkeeptl_sw",
    "cutinl": "Cut-in (left)\ncutinl",
    "cutinr": "Cut-in (right)\ncutinr",
}
CLASS_ONELINE = {
    "tlkeep": "Agent-straight (ego LT) / tlkeep",
    "keeptl": "Agent-LT N→E / keeptl",
    "keeptl_sw": "Agent-LT S→W / keeptl_sw",
    "cutinl": "Cut-in (left) / cutinl",
    "cutinr": "Cut-in (right) / cutinr",
}
CLASS_ZH = {
    "tlkeep": "直行代理（ego 左轉）tlkeep",
    "keeptl": "左轉代理 N→E / keeptl",
    "keeptl_sw": "左轉代理 S→W / keeptl_sw",
    "cutinl": "左側切入 / cutinl",
    "cutinr": "右側切入 / cutinr",
}
DESC = ["pet", "d_min", "alpha", "conflict_x", "conflict_y", "u_c"]
DESC_UNIT = {"pet": "s", "d_min": "m", "alpha": "deg",
             "conflict_x": "m", "conflict_y": "m", "u_c": "m/s"}
SPACES = ["interaction", "path", "joint"]
SPACE_LABEL = {"interaction": "interaction (6-D z)", "path": "path (DTW, m)",
               "joint": "joint (interaction ∧ path)"}

# the current per-class rule, read live from results/table34_manifest.json (v3 rerun 2026-09-14: the hard-coded copy
# still held the pre-rerun cutinl pair (0.9312, 0.8743); the v3 manifest gives (1.1086, 0.8819))
CUR = {cls: (float(e["interaction"]), float(e["path"]))
       for cls, e in json.loads((RES / "table34_manifest.json").read_text())["eps"].items()
       if cls in ("cutinl", "cutinr", "keeptl", "keeptl_sw", "tlkeep")}
assert len(CUR) == 5, sorted(CUR)

# regime thresholds (stated up front so the bands are reproducible)
SAT = 0.95     # saturation  : best arm coverage@1000 > SAT
STARV = 0.05   # starvation  : best arm coverage@1000 < STARV

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


TIE = 1e-12   # scripts/90_tables3_assemble.py line 67: bold ties are |diff| <= 1e-12


def bold_set(series):
    """Best arm(s) of a group, with scripts/90's exact-tie tolerance."""
    return set(series[series >= series.max() - TIE].index) if len(series) else set()


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# load
# ─────────────────────────────────────────────────────────────────────────────
say("=" * 78)
say("98_eps_sweep_report.py  --  figures + fixed-eps recommendation + report")
say("=" * 78)

COV = pd.read_csv(OUT / "eps_sweep_coverage.csv")
CRN = pd.read_csv(OUT / "eps_sweep_corner.csv")
JG = pd.read_csv(OUT / "eps_sweep_joint_grid.csv")
NNV = pd.read_csv(OUT / "eps_real_nn_values.csv")
NND = pd.read_csv(OUT / "eps_real_nn_distributions.csv")
GEN = pd.read_csv(RES / "final" / "tables3" / "gen_metrics.csv")
REAL = pd.read_csv(RES / "e6_real_reference.csv")
say(f"loaded  coverage {COV.shape}  corner {CRN.shape}  joint_grid {JG.shape} "
    f" nn_values {NNV.shape}  gen_metrics {GEN.shape}  real {REAL.shape}")

# the 31 shared GLOBAL grid values (hazard H1: derive them from the spaces that
# are clean, i.e. interaction / path, never from the joint rows)
GRID_INT = np.array(sorted(COV.loc[(COV.space == "interaction") & COV.eps_mult.isna(), "eps_int"].unique()))
GRID_PATH = np.array(sorted(COV.loc[(COV.space == "path") & COV.eps_mult.isna(), "eps_path"].unique()))
assert len(GRID_INT) == 31 and len(GRID_PATH) == 31, (len(GRID_INT), len(GRID_PATH))
say(f"global grids: eps_int {len(GRID_INT)} values [{GRID_INT.min()}, {GRID_INT.max()}]; "
    f"eps_path {len(GRID_PATH)} values [{GRID_PATH.min()}, {GRID_PATH.max()}]")

OPBASE = dict(pool="all", real_set="matched489", budget=1000)


def op(df, space, arms=ARMS, budget=1000):
    """Thesis operating point: pool=all, matched489, budget=b, the given space."""
    s = df[(df.pool == "all") & (df.real_set == "matched489") & (df.budget == budget)
           & (df.arm.isin(arms)) & (df.space == space)]
    return s


def sweep_tab(df, space, arms=ARMS, budget=1000):
    """Seed-mean curve on the 31 SHARED global grid values only (hazard H1).

    Returns a long frame: subset, arm, eps (the swept axis), cov, lo, hi.
    """
    s = op(df, space, arms, budget)
    if space == "path":
        s = s[s.eps_path.isin(GRID_PATH)]
        key = "eps_path"
    else:
        s = s[s.eps_int.isin(GRID_INT)]
        key = "eps_int"
    g = (s.groupby(["subset", "arm", key])
           .agg(cvg=("coverage_mean", "mean"), lo=("coverage_min", "min"),
                hi=("coverage_max", "max"), nat=("coverage_natural_order", "mean"),
                n_real=("n_real", "max"), n_seeds=("seed", "nunique"),
                eps_path_partner=("eps_path", "mean"))
           .reset_index().rename(columns={key: "eps"}))
    return g


def current_cells(df, space, arms=ARMS, budget=1000):
    """The CURRENT per-class rule cells: the eps_mult == 1.0 rows (hazard H2)."""
    s = op(df, space, arms, budget)
    s = s[s.eps_mult == 1.0]
    return (s.groupby(["subset", "arm"])
              .agg(cvg=("coverage_mean", "mean"), lo=("coverage_min", "min"),
                   hi=("coverage_max", "max"), n_real=("n_real", "max"),
                   eps_int=("eps_int", "mean"), eps_path=("eps_path", "mean"))
              .reset_index())


TAB = {sp: sweep_tab(COV, sp) for sp in SPACES}
TABC = {sp: sweep_tab(CRN, sp) for sp in SPACES}
CURTAB = {sp: current_cells(COV, sp) for sp in SPACES}
CURTABC = {sp: current_cells(CRN, sp) for sp in SPACES}

# ─────────────────────────────────────────────────────────────────────────────
# GATE: my seed-mean of the frozen sweep must equal the thesis-table cells
# ─────────────────────────────────────────────────────────────────────────────
say("\n-- GATE: current-rule cells vs results/final/tables3/gen_metrics.csv --")
gate_rows, gmax = [], 0.0
for metric, src in (("coverage", CURTAB), ("corner_coverage", CURTABC)):
    g = GEN[(GEN.metric == metric) & (GEN.pool == "all") & (GEN.real_set == "matched489")
            & (GEN.budget == 1000) & (GEN.arm.isin(ARMS))]
    for sp in SPACES:
        gg = g[g.space == sp]
        t = src[sp].set_index(["subset", "arm"])["cvg"]
        for r in gg.itertuples():
            mine = t.get((r._2, r.arm))
            if mine is None:
                gate_rows.append((metric, sp, r._2, r.arm, r.value, None, None))
                continue
            d = abs(float(mine) - float(r.value))
            gmax = max(gmax, d)
            gate_rows.append((metric, sp, r._2, r.arm, r.value, mine, d))
GATE = pd.DataFrame(gate_rows, columns=["metric", "space", "subset", "arm",
                                        "gen_metrics", "mine", "absdiff"])
n_gate = int(GATE.absdiff.notna().sum())
n_miss = int(GATE.absdiff.isna().sum())
assert n_miss == 0, GATE[GATE.absdiff.isna()]
assert gmax < 1e-12, gmax
say(f"GATE PASS: {n_gate} cells, 0 missing, max |diff| = {gmax:.3e}  (tolerance 1e-12)")

# hazard H1 demonstration, for the report
leak = (COV[(COV.space == "joint") & COV.eps_mult.isna() & (COV.pool == "all")
            & (COV.real_set == "matched489") & (COV.arm == "ours3_disk_kde")]
        .groupby("eps_int")["subset"].nunique())
N_LEAK = int((leak < 5).sum())
LEAK_VALS = sorted(np.round(leak[leak < 5].index.values, 6).tolist())
say(f"hazard H1: joint rows carry {len(leak)} distinct eps_int, {N_LEAK} of them single-class "
    f"(filtered out everywhere below)")

# ─────────────────────────────────────────────────────────────────────────────
# A. REGIMES
# ─────────────────────────────────────────────────────────────────────────────
say("\n" + "=" * 78)
say("A. REGIMES  (starvation: best arm < %.2f ; saturation: best arm > %.2f)" % (STARV, SAT))
say("=" * 78)


def regimes(tab, space):
    rows = []
    for cls in CLASSES:
        t = tab[tab['subset'] == cls].pivot_table(index="eps", columns="arm", values="cvg")
        if t.empty:
            continue
        best = t.max(axis=1)
        e = best.index.values
        st = e[best.values < STARV]
        sa = e[best.values > SAT]
        # discriminative band = first grid eps above the last starved one,
        # last grid eps below the first saturated one
        lo = e[np.searchsorted(e, st.max(), "right")] if len(st) and st.max() < e[-1] else e[0]
        if len(sa):
            i = int(np.searchsorted(e, sa.min(), "left")) - 1
            hi = e[max(i, 0)]
        else:
            hi = e[-1]
        rows.append(dict(space=space, subset=cls, n_real=int(tab[tab['subset'] == cls].n_real.max()),
                         starv_last=float(st.max()) if len(st) else np.nan,
                         disc_lo=float(lo), disc_hi=float(hi),
                         sat_first=float(sa.min()) if len(sa) else np.nan,
                         best_max=float(best.max()),
                         best_at_disc_lo=float(best.loc[lo]), best_at_disc_hi=float(best.loc[hi]),
                         cur_eps=float(CUR[cls][1] if space == "path" else CUR[cls][0])))
    return pd.DataFrame(rows)


REG = pd.concat([regimes(TAB[sp], sp) for sp in SPACES], ignore_index=True)
REG["cur_in_band"] = (REG.cur_eps >= REG.disc_lo) & (REG.cur_eps <= REG.disc_hi)
BAND = {}
for sp in SPACES:
    r = REG[REG.space == sp]
    BAND[sp] = (float(r.disc_lo.max()), float(r.disc_hi.min()))
    say(f"\n[{sp}]  intersection of the 5 class bands = [{BAND[sp][0]:.4f}, {BAND[sp][1]:.4f}]")
    for r2 in r.itertuples():
        say(f"   {r2.subset:10s} n={r2.n_real:3d}  starved<= {f4(r2.starv_last):>8s}  "
            f"discriminative [{r2.disc_lo:.4f}, {r2.disc_hi:.4f}]  saturated>= {f4(r2.sat_first):>8s}  "
            f"best_max={r2.best_max:.4f}  current={r2.cur_eps:.4f} {'IN' if r2.cur_in_band else 'OUT of'} band")
REG.to_csv(OUT / "eps_regimes.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# B. RANK STABILITY
# ─────────────────────────────────────────────────────────────────────────────
say("\n" + "=" * 78)
say("B. RANK STABILITY  (winner = highest coverage@1000, seed-mean)")
say("=" * 78)

flip_rows = []
for sp in SPACES:
    for cls in CLASSES:
        t = TAB[sp][TAB[sp]['subset'] == cls].pivot_table(index="eps", columns="arm", values="cvg")
        t = t.dropna(axis=1, how="all")
        if t.empty:
            continue
        e = t.index.values
        w = t.idxmax(axis=1).values
        cur = CUR[cls][1] if sp == "path" else CUR[cls][0]
        band = REG[(REG.space == sp) & (REG['subset'] == cls)].iloc[0]
        for i in range(1, len(e)):
            if w[i] != w[i - 1]:
                a, b = t.iloc[i - 1], t.iloc[i]
                flip_rows.append(dict(
                    space=sp, subset=cls, eps_before=float(e[i - 1]), eps_after=float(e[i]),
                    winner_before=w[i - 1], winner_after=w[i],
                    cov_before_winner=float(a[w[i - 1]]), cov_before_challenger=float(a[w[i]]),
                    cov_after_winner=float(b[w[i]]), cov_after_challenger=float(b[w[i - 1]]),
                    margin_before=float(a[w[i - 1]] - a[w[i]]), margin_after=float(b[w[i]] - b[w[i - 1]]),
                    brackets_current_eps=bool(e[i - 1] <= cur <= e[i]),
                    inside_discriminative_band=bool(e[i - 1] >= band.disc_lo and e[i] <= band.disc_hi),
                ))
FLIPS = pd.DataFrame(flip_rows)
FLIPS.to_csv(OUT / "eps_rank_flips.csv", index=False)
say(f"{len(FLIPS)} rank flips on the 31-point global grid "
    f"({int(FLIPS.inside_discriminative_band.sum())} of them inside the class's discriminative band)")


def winner_at(sp, cls, eps, tab=None):
    tab = tab or TAB
    t = tab[sp][tab[sp]['subset'] == cls].pivot_table(index="eps", columns="arm", values="cvg").dropna(axis=1, how="all")
    i = int(np.argmin(np.abs(t.index.values - eps)))
    r = t.iloc[i]
    return r.idxmax(), float(r.max()), float(t.index.values[i]), r.to_dict()


def stability(sp, cls, eps, tab=None):
    """Maximal winner-constant grid interval containing eps; multiplicative radius."""
    tab = tab or TAB
    t = tab[sp][tab[sp]['subset'] == cls].pivot_table(index="eps", columns="arm", values="cvg").dropna(axis=1, how="all")
    e, w = t.index.values, t.idxmax(axis=1).values
    i = int(np.argmin(np.abs(e - eps)))
    lo = hi = i
    while lo > 0 and w[lo - 1] == w[i]:
        lo -= 1
    while hi < len(e) - 1 and w[hi + 1] == w[i]:
        hi += 1
    rl = np.inf if lo == 0 else e[i] / e[lo - 1]
    rh = np.inf if hi == len(e) - 1 else e[hi + 1] / e[i]
    return w[i], float(e[lo]), float(e[hi]), float(min(rl, rh))


ROB = []
for sp in SPACES:
    say(f"\n[{sp}] winner at the CURRENT class eps, and how far that survives")
    for cls in CLASSES:
        cur = CUR[cls][1] if sp == "path" else CUR[cls][0]
        cc = CURTAB[sp][CURTAB[sp]['subset'] == cls].set_index("arm")["cvg"]
        if cc.empty:
            continue
        wcur, vcur = cc.idxmax(), cc.max()
        run = cc.drop(wcur).max() if len(cc) > 1 else np.nan
        w, lo, hi, rad = stability(sp, cls, cur)
        band = REG[(REG.space == sp) & (REG['subset'] == cls)].iloc[0]
        const_in_band = (FLIPS[(FLIPS.space == sp) & (FLIPS['subset'] == cls)
                               & FLIPS.inside_discriminative_band].empty)
        t2 = TAB[sp][TAB[sp]['subset'] == cls].pivot_table(index="eps", columns="arm", values="cvg").dropna(axis=1, how="all")
        mcom = (t2.index.values >= BAND[sp][0]) & (t2.index.values <= BAND[sp][1])
        wins_common = sorted(set(t2[mcom].idxmax(axis=1))) if mcom.any() else []
        ROB.append(dict(space=sp, subset=cls, cur_eps=cur, winner=wcur, cov=vcur,
                        winner_constant_over_common_band=(len(wins_common) == 1),
                        winners_in_common_band=";".join(TINY[x] for x in wins_common),
                        runner_up=float(run), margin=float(vcur - run) if np.isfinite(run) else np.nan,
                        stable_lo=lo, stable_hi=hi, stability_radius=rad,
                        winner_constant_over_band=const_in_band,
                        band_lo=float(band.disc_lo), band_hi=float(band.disc_hi)))
        say(f"   {cls:10s} eps={cur:.4f} winner={TINY[wcur]:7s} cov={vcur:.4f} "
            f"(runner-up {f4(run)}, margin {f4(vcur - run)})  winner constant on "
            f"[{lo:.4f},{hi:.4f}] (x{rad:.2f})  const over own band: {const_in_band}  "
            f"const over all-class band: {len(wins_common) == 1} ({';'.join(TINY[x] for x in wins_common)})")
ROB = pd.DataFrame(ROB)

# ─────────────────────────────────────────────────────────────────────────────
# C. FIXED-EPS RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────
say("\n" + "=" * 78)
say("C. FIXED-EPS RECOMMENDATION")
say("=" * 78)

# pooled real-to-real NN distributions (matched489)
nn = NNV[NNV.real_set == "matched489"]
NN_INT = np.sort(nn[nn.space == "interaction"].nn_dist.values)
NN_PATH = np.sort(nn[nn.space == "path"].nn_dist.values)
POOL_MED_INT = float(np.median(NN_INT))
POOL_MED_PATH = float(np.median(NN_PATH))
say(f"pooled real-to-real NN (matched489): interaction median {POOL_MED_INT:.4f} z over n={len(NN_INT)}; "
    f"path median {POOL_MED_PATH:.4f} m over n={len(NN_PATH)}")
say("per-class spread: interaction %.4f..%.4f (x%.2f) ; path %.4f..%.4f (x%.2f)"
    % (min(v[0] for v in CUR.values()), max(v[0] for v in CUR.values()),
       max(v[0] for v in CUR.values()) / min(v[0] for v in CUR.values()),
       min(v[1] for v in CUR.values()), max(v[1] for v in CUR.values()),
       max(v[1] for v in CUR.values()) / min(v[1] for v in CUR.values())))


def ecdf_q(sorted_vals, x):
    return float(np.searchsorted(sorted_vals, x, "right") / len(sorted_vals))


# the joint diagonal: eps_int -> the pooled-quantile-matched eps_path partner
DIAG = (COV[(COV.space == "joint") & COV.eps_mult.isna() & (COV.pool == "all")
            & (COV.real_set == "matched489") & (COV.arm == "ours3_disk_kde")
            & (COV.budget == 1000) & (COV['subset'] == "tlkeep")]
        [["eps_int", "eps_path", "eps_quantile"]].drop_duplicates().sort_values("eps_int"))
DIAGMAP = dict(zip(np.round(DIAG.eps_int.values, 6), DIAG.eps_path.values))
DIAGQ = dict(zip(np.round(DIAG.eps_int.values, 6), DIAG.eps_quantile.values))

# candidate screen over every global eps_int that is a diagonal anchor
cand_rows = []
for ei in GRID_INT:
    ep = DIAGMAP[round(float(ei), 6)]
    ok_band = {sp: (BAND[sp][0] <= (ep if sp == "path" else ei) <= BAND[sp][1]) for sp in SPACES}
    # bold changes vs the current rule, joint space, coverage and corner
    ch_cov, ch_cor = [], []
    for cls in CLASSES:
        cc = CURTAB["joint"][CURTAB["joint"]['subset'] == cls].set_index("arm")["cvg"]
        nw = TAB["joint"][(TAB["joint"]['subset'] == cls) & np.isclose(TAB["joint"].eps, ei)].set_index("arm")["cvg"]
        if not cc.empty and not nw.empty and bold_set(cc) != bold_set(nw):
            ch_cov.append(cls)
        cc2 = CURTABC["joint"][CURTABC["joint"]['subset'] == cls].set_index("arm")["cvg"]
        nw2 = TABC["joint"][(TABC["joint"]['subset'] == cls) & np.isclose(TABC["joint"].eps, ei)].set_index("arm")["cvg"]
        if not cc2.empty and not nw2.empty and bold_set(cc2) != bold_set(nw2):
            ch_cor.append(cls)
    rads = [stability("joint", c, ei)[3] for c in CLASSES]
    cand_rows.append(dict(eps_int=float(ei), eps_path=float(ep), q=float(DIAGQ[round(float(ei), 6)]),
                          in_band_int=ok_band["interaction"], in_band_path=ok_band["path"],
                          in_band_joint=ok_band["joint"],
                          in_all_bands=all(ok_band.values()),
                          n_bold_change_cov=len(ch_cov), bold_change_cov=";".join(ch_cov),
                          n_bold_change_corner=len(ch_cor), bold_change_corner=";".join(ch_cor),
                          min_stability_radius=float(np.min(rads)),
                          n_classes_stable_pm50=int(sum(r >= 1.5 for r in rads))))
CAND = pd.DataFrame(cand_rows)
say("\ncandidate screen (diagonal anchors on the 31-point global eps_int grid):")
say(CAND[(CAND.eps_int >= 0.24) & (CAND.eps_int <= 2.05)].to_string(index=False,
    columns=["eps_int", "eps_path", "q", "in_all_bands", "n_bold_change_cov",
             "n_bold_change_corner", "min_stability_radius", "n_classes_stable_pm50"]))

# ---- the choice -----------------------------------------------------------
REC_INT = 0.75
REC_PATH = float(DIAGMAP[round(REC_INT, 6)])          # 0.354791... m
REC_PATH_ROUND = 0.35
ALT_INT = 0.884054            # the ONLY anchor that changes no bold cell at all
ALT_PATH = float(DIAGMAP[round(ALT_INT, 6)])          # 0.451288... m
ALT2_INT = 1.0                # the loose variant: geometric centre of the joint band
ALT2_PATH = float(DIAGMAP[round(ALT2_INT, 6)])        # 0.544154... m
Q_REC = ecdf_q(NN_INT, REC_INT), ecdf_q(NN_PATH, REC_PATH)
Q_ALT = ecdf_q(NN_INT, ALT_INT), ecdf_q(NN_PATH, ALT_PATH)
Q_ALT2 = ecdf_q(NN_INT, ALT2_INT), ecdf_q(NN_PATH, ALT2_PATH)
BAND_GEOM_MID = float(np.sqrt(BAND["joint"][0] * BAND["joint"][1]))
say(f"\nFIRST  choice: eps_int = {REC_INT} z, eps_path = {REC_PATH:.6f} m (print as {REC_PATH_ROUND} m); "
    f"pooled NN percentile {100*Q_REC[0]:.1f}% / {100*Q_REC[1]:.1f}%")
say(f"SECOND choice: eps_int = {ALT_INT} z, eps_path = {ALT_PATH:.6f} m; "
    f"pooled NN percentile {100*Q_ALT[0]:.1f}% / {100*Q_ALT[1]:.1f}%  (0 bold changes)")
say(f"THIRD (loose) : eps_int = {ALT2_INT} z, eps_path = {ALT2_PATH:.6f} m; "
    f"pooled NN percentile {100*Q_ALT2[0]:.1f}% / {100*Q_ALT2[1]:.1f}%")
say(f"geometric centre of the joint discriminative band = {BAND_GEOM_MID:.4f} z")
Q50_INT_GRID = float(GRID_INT[np.argmin(np.abs(GRID_INT - POOL_MED_INT))])
say(f"pooled-median (q=0.50) variant: eps_int {POOL_MED_INT:.4f} -> nearest grid {Q50_INT_GRID}, "
    f"eps_path {POOL_MED_PATH:.4f} m")

# physical translation of 1 z and of eps_int
rm = REAL[(REAL.in_matched489.astype(bool)) & (REAL.int_finite.astype(bool))]
SD = rm.groupby("subset")[DESC].std(ddof=1)
say("\nwhat eps_int means physically (per-class real SD, matched489, int_finite):")
say(SD.round(3).to_string())
PHYS = {}
for cls in CLASSES:
    PHYS[cls] = {k: dict(sd=float(SD.loc[cls, k]),
                         one_axis=float(REC_INT * SD.loc[cls, k]),
                         isotropic=float(REC_INT / np.sqrt(6) * SD.loc[cls, k])) for k in DESC}
say(f"\neps_int = {REC_INT} z, if the whole gap sits in ONE descriptor (0.75 x SD):")
for cls in CLASSES:
    say("   %-10s " % cls + "  ".join(f"{k}={PHYS[cls][k]['one_axis']:.2f}{DESC_UNIT[k]}" for k in DESC))
say(f"eps_int = {REC_INT} z, if the gap is spread evenly over all six (0.75/sqrt(6) x SD):")
for cls in CLASSES:
    say("   %-10s " % cls + "  ".join(f"{k}={PHYS[cls][k]['isotropic']:.2f}{DESC_UNIT[k]}" for k in DESC))

# (v) unreachable / trivially reachable
say("\n(v) reachability at the recommended eps")
REACH = []
for cls in CLASSES:
    nnc = nn[nn['subset'] == cls]
    ni = nnc[nnc.space == "interaction"].nn_dist.values
    npth = nnc[nnc.space == "path"].nn_dist.values
    t = TAB["joint"][(TAB["joint"]['subset'] == cls) & np.isclose(TAB["joint"].eps, REC_INT)].set_index("arm")["cvg"]
    tc = TAB["joint"][(TAB["joint"]['subset'] == cls) & np.isclose(TAB["joint"].eps, CUR[cls][0])]
    n_real = int(TAB["joint"][TAB["joint"]['subset'] == cls].n_real.max())
    REACH.append(dict(subset=cls, n_real_int=len(ni), n_real_path=len(npth), n_real_joint=n_real,
                      iso_int=int((ni > REC_INT).sum()), iso_path=int((npth > REC_PATH).sum()),
                      dense_int=int((ni <= REC_INT / 2).sum()), dense_path=int((npth <= REC_PATH / 2).sum()),
                      best_cov=float(t.max()), n_missed_by_best=int(round(n_real * (1 - t.max()))),
                      all_cov=float(t.min()), n_hit_by_all=int(round(n_real * t.min()))))
REACH = pd.DataFrame(REACH)
say(REACH.to_string(index=False))

# ---- rank robustness of the recommendation to +-50% -----------------------
say("\n(iii) rank robustness of the FIRST choice to +-50% in eps_int")
ROB50 = []
for sp in SPACES:
    for cls in CLASSES:
        anchor = REC_PATH if sp == "path" else REC_INT
        t = TAB[sp][TAB[sp]['subset'] == cls].pivot_table(index="eps", columns="arm", values="cvg").dropna(axis=1, how="all")
        m = (t.index.values >= 0.5 * anchor) & (t.index.values <= 1.5 * anchor)
        ws = set(t[m].idxmax(axis=1)) if m.any() else set()
        w0, v0, e0, _ = winner_at(sp, cls, anchor)
        ROB50.append(dict(space=sp, subset=cls, anchor=anchor, winner=w0, cov=v0,
                          n_winners_pm50=len(ws), winners_pm50=";".join(sorted(TINY[x] for x in ws)),
                          stable=len(ws) <= 1))
ROB50 = pd.DataFrame(ROB50)
for sp in SPACES:
    r = ROB50[ROB50.space == sp]
    say(f"   [{sp}] {int(r.stable.sum())}/{len(r)} classes keep one winner over +-50%: "
        + ", ".join(f"{x.subset}:{x.winners_pm50}" for x in r.itertuples()))

# ─────────────────────────────────────────────────────────────────────────────
# D. THESIS TABLES RECOMPUTED AT THE FIXED EPS
# ─────────────────────────────────────────────────────────────────────────────
say("\n" + "=" * 78)
say("D. THESIS TABLES AT THE FIXED EPS")
say("=" * 78)

TBL_OF = {"keeptl": ("Table 左轉 (left turn)", "Agent-LT N→E"),
          "keeptl_sw": ("Table 左轉 (left turn)", "Agent-LT S→W"),
          "cutinl": ("Table Cut-in", "Cut-in (left)"),
          "cutinr": ("Table Cut-in", "Cut-in (right)"),
          "tlkeep": ("Table Appendix — tlkeep", "Agent-straight (ego LT) tlkeep")}
TBL_N = {"keeptl": 50, "keeptl_sw": 82, "cutinl": 61, "cutinr": 20, "tlkeep": 298}
METHOD_LABEL = {"sakura_route_kde": "SAKURA-route+KDE",
                "svd_d5_kde_matched_analytic": "SVD_d5+KDE",
                "ours3_disk_kde": "Ours+KDE"}
MORDER = ["sakura_route_kde", "svd_d5_kde_matched_analytic", "ours3_disk_kde"]


drows = []
for cls in CLASSES:
    cc = CURTAB["joint"][CURTAB["joint"]['subset'] == cls].set_index("arm")["cvg"]
    cf = TAB["joint"][(TAB["joint"]['subset'] == cls) & np.isclose(TAB["joint"].eps, REC_INT)].set_index("arm")["cvg"]
    kc = CURTABC["joint"][CURTABC["joint"]['subset'] == cls].set_index("arm")["cvg"]
    kf = TABC["joint"][(TABC["joint"]['subset'] == cls) & np.isclose(TABC["joint"].eps, REC_INT)].set_index("arm")["cvg"]
    bc, bf, bkc, bkf = bold_set(cc), bold_set(cf), bold_set(kc), bold_set(kf)
    for arm in MORDER:
        if arm not in cc.index:
            continue
        drows.append(dict(
            table=TBL_OF[cls][0], group=TBL_OF[cls][1], subset=cls, n_group=TBL_N[cls],
            n_real_coverage=int(CURTAB["joint"][CURTAB["joint"]['subset'] == cls].n_real.max()),
            n_rare=int(CURTABC["joint"][CURTABC["joint"]['subset'] == cls].n_real.max()),
            method=METHOD_LABEL[arm], arm=arm,
            eps_int_current=CUR[cls][0], eps_path_current=CUR[cls][1],
            eps_int_fixed=REC_INT, eps_path_fixed=REC_PATH,
            coverage_current=float(cc[arm]), coverage_fixed=float(cf[arm]),
            coverage_delta=float(cf[arm] - cc[arm]),
            corner_current=float(kc[arm]), corner_fixed=float(kf[arm]),
            corner_delta=float(kf[arm] - kc[arm]),
            bold_coverage_current=arm in bc, bold_coverage_fixed=arm in bf,
            bold_coverage_moves=(arm in bc) != (arm in bf),
            bold_corner_current=arm in bkc, bold_corner_fixed=arm in bkf,
            bold_corner_moves=(arm in bkc) != (arm in bkf),
        ))
D = pd.DataFrame(drows)
D.to_csv(OUT / "tables3_coverage_at_fixed_eps.csv", index=False)
n_mv_cov = int(D.bold_coverage_moves.sum())
n_mv_cor = int(D.bold_corner_moves.sum())
GRP_MV_COV = sorted(D.loc[D.bold_coverage_moves, "subset"].unique().tolist())
GRP_MV_COR = sorted(D.loc[D.bold_corner_moves, "subset"].unique().tolist())
say(f"{n_mv_cov} of {len(D)} coverage cells change bold status ({len(GRP_MV_COV)} groups: {GRP_MV_COV}); "
    f"{n_mv_cor} of {len(D)} corner cells ({len(GRP_MV_COR)} groups: {GRP_MV_COR})")


def md_table_D():
    L = []
    L.append(f"Joint space, pool = all, real_set = matched489, budget = 1000, mean over 3 seeds x 20 orderings.  "
             f"Current = the per-class median-NN rule; Fixed = eps_int {REC_INT} z / eps_path {REC_PATH_ROUND} m "
             f"(exact {REC_PATH:.6f} m).  **bold** = best arm of the group under that rule; "
             f"a cell flagged MOVE changes bold status.\n")
    L.append("| Table | Scenario group | n | n_real (cov / rare) | Method | Cov@1000 current | Cov@1000 fixed | Δ | bold moves | Corner current | Corner fixed | Δ | bold moves |")
    L.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | :-: | ---: | ---: | ---: | :-: |")
    last = None
    for cls in CLASSES:
        d = D[D['subset'] == cls]
        for i, r in enumerate(d.itertuples()):
            g = f"{r.table} | {r.group} | {r.n_group} | {r.n_real_coverage} / {r.n_rare}" if i == 0 else " |  |  | "
            cc = f"**{r.coverage_current:.3f}**" if r.bold_coverage_current else f"{r.coverage_current:.3f}"
            cf = f"**{r.coverage_fixed:.3f}**" if r.bold_coverage_fixed else f"{r.coverage_fixed:.3f}"
            kc = f"**{r.corner_current:.3f}**" if r.bold_corner_current else f"{r.corner_current:.3f}"
            kf = f"**{r.corner_fixed:.3f}**" if r.bold_corner_fixed else f"{r.corner_fixed:.3f}"
            L.append(f"| {g} | {r.method} | {cc} | {cf} | {r.coverage_delta:+.3f} | "
                     f"{'**MOVE**' if r.bold_coverage_moves else ''} | {kc} | {kf} | {r.corner_delta:+.3f} | "
                     f"{'**MOVE**' if r.bold_corner_moves else ''} |")
        last = cls
    return "\n".join(L)


(OUT / "tables3_coverage_at_fixed_eps.md").write_text(
    "## Thesis coverage / corner-coverage cells at the recommended FIXED epsilon\n\n" + md_table_D() +
    "\n\n- Source rows: results/eps_sweep/eps_sweep_coverage.csv and eps_sweep_corner.csv, "
    "space=joint, pool=all, real_set=matched489, budget=1000; `Current` = the rows with eps_mult == 1.0, "
    "`Fixed` = the joint-diagonal rows at eps_int == 0.75.\n"
    "- The current cells reproduce results/final/tables3/gen_metrics.csv exactly "
    f"({n_gate} cells checked, max |diff| {gmax:.1e}).\n"
    "- The RareCase table (Special 39_180) has no coverage column at all (one recorded scenario; it reports "
    "Recovered fraction instead), so it does not appear here.\n"
    "- SAKURA has no tlkeep arm, so the appendix group carries two methods.\n", encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# E. SHOULD EPS BE SPLIT?
# ─────────────────────────────────────────────────────────────────────────────
say("\n" + "=" * 78)
say("E. SPLIT EPS?  interaction vs path")
say("=" * 78)

# (ii) correlation of a scenario's interaction NN with its path NN
from scipy.stats import spearmanr, pearsonr  # noqa: E402
W = nn.pivot_table(index=["subset", "scenario_uid"], columns="space", values="nn_dist").dropna()
CORR = [dict(scope="pooled", n=len(W),
             spearman=float(spearmanr(W.interaction, W.path).statistic),
             pearson=float(pearsonr(W.interaction, W.path)[0]))]
for cls, g in W.groupby(level=0):
    CORR.append(dict(scope=cls, n=len(g),
                     spearman=float(spearmanr(g.interaction, g.path).statistic),
                     pearson=float(pearsonr(g.interaction, g.path)[0])))
CORR = pd.DataFrame(CORR)
say(CORR.to_string(index=False))

# (iii) elasticity at the operating point, from the 2-D joint grid
JGO = JG[(JG.pool == "all") & (JG.real_set == "matched489") & (JG.budget == 1000) & (JG.arm.isin(ARMS))]
JGM = JGO.groupby(["subset", "arm", "eps_int", "eps_path"])["coverage_mean"].mean().reset_index()
ELA = []
for cls in CLASSES:
    for arm in ARMS:
        s = JGM[(JGM['subset'] == cls) & (JGM.arm == arm)]
        if s.empty:
            continue
        piv = s.pivot_table(index="eps_int", columns="eps_path", values="coverage_mean")
        EI, EP = piv.index.values, piv.columns.values
        for tag, (ei, ep) in (("current", CUR[cls]), ("fixed", (REC_INT, REC_PATH))):
            i = int(np.argmin(np.abs(EI - ei)))
            j = int(np.argmin(np.abs(EP - ep)))
            i = min(max(i, 1), len(EI) - 2)
            j = min(max(j, 1), len(EP) - 2)
            di = (piv.iloc[i + 1, j] - piv.iloc[i - 1, j]) / (np.log(EI[i + 1]) - np.log(EI[i - 1]))
            dj = (piv.iloc[i, j + 1] - piv.iloc[i, j - 1]) / (np.log(EP[j + 1]) - np.log(EP[j - 1]))
            ELA.append(dict(operating_point=tag, subset=cls, arm=arm,
                            eps_int_cell=float(EI[i]), eps_path_cell=float(EP[j]),
                            coverage=float(piv.iloc[i, j]),
                            d_cov_dlog_eps_int=float(di), d_cov_dlog_eps_path=float(dj),
                            binding=("tie" if abs(di - dj) <= 0.05 * max(abs(di), abs(dj), 1e-12)
                                     else ("interaction" if di > dj else "path")),
                            ratio_int_over_path=float(di / dj) if dj != 0 else np.inf))
ELA = pd.DataFrame(ELA)
say("\nelasticity dC/dlog(eps) at the CURRENT operating point (joint grid, central difference):")
say(ELA[ELA.operating_point == "current"].to_string(index=False,
    columns=["subset", "arm", "coverage", "d_cov_dlog_eps_int", "d_cov_dlog_eps_path", "binding"]))
say("\nelasticity at the FIXED operating point:")
say(ELA[ELA.operating_point == "fixed"].to_string(index=False,
    columns=["subset", "arm", "eps_int_cell", "eps_path_cell", "coverage",
             "d_cov_dlog_eps_int", "d_cov_dlog_eps_path", "binding"]))

# (iv) one combined normalised distance: L-inf is exactly today's rule at t=1;
#      L2 is bracketed between L-inf(t/sqrt2) and L-inf(t) because the balls nest.
say("\n(iv) a SINGLE combined normalised distance")
say("    L-inf with per-class median-NN normalisation at t=1 IS the current rule "
    "(eps_mult == 1.0 rows), so 'one combined number' changes nothing unless the "
    "normaliser or the norm changes.")
LINF = []
for m in (1.0, 1.5, 2.0):
    s = COV[(COV.space == "joint") & (COV.eps_mult == m) & (COV.pool == "all")
            & (COV.real_set == "matched489") & (COV.budget == 1000) & (COV.arm.isin(ARMS))]
    g = s.groupby(["subset", "arm"]).coverage_mean.mean().reset_index()
    g["t"] = m
    LINF.append(g)
LINF = pd.concat(LINF, ignore_index=True)
say("    L-inf (per-class normalisation) coverage at t = 1, 1.5, 2:")
say(LINF.pivot_table(index=["subset", "arm"], columns="t", values="coverage_mean").round(4).to_string())

# L2 bracket around the FIXED (pooled) normalisation, t = 1
# Rigorous bracket.  Any grid rectangle [0,a] x [0,b] with (a/e_i)^2+(b/e_p)^2 <= 1
# is INSCRIBED in the L2 ball, so its coverage is a valid lower bound; the
# smallest rectangle that CONTAINS the L2 ball is the L-inf cell itself, so that
# is a valid upper bound.  We take the best inscribed rectangle per arm.
L2, L2CELL = [], {}
for cls in CLASSES:
    sub = JGM[JGM['subset'] == cls]
    ins = sub[(sub.eps_int / REC_INT) ** 2 + (sub.eps_path / REC_PATH) ** 2 <= 1.0]
    up = TAB["joint"][(TAB["joint"]['subset'] == cls) & np.isclose(TAB["joint"].eps, REC_INT)].set_index("arm")["cvg"]
    for arm in ARMS:
        if arm not in up.index:
            continue
        ia = ins[ins.arm == arm]
        if ia.empty:
            lo, cell = np.nan, (np.nan, np.nan)
        else:
            k = ia.coverage_mean.idxmax()
            lo, cell = float(ia.loc[k, "coverage_mean"]), (float(ia.loc[k, "eps_int"]), float(ia.loc[k, "eps_path"]))
        L2CELL[(cls, arm)] = cell
        L2.append(dict(subset=cls, arm=arm, best_inscribed_eps_int=cell[0], best_inscribed_eps_path=cell[1],
                       l2_lower=lo, l2_upper=float(up[arm])))
L2 = pd.DataFrame(L2)
L2SEP = []
for cls in CLASSES:
    d = L2[L2['subset'] == cls]
    if d.empty:
        continue
    w = d.loc[d.l2_upper.idxmax(), "arm"]
    lowin = float(d.loc[d.arm == w, "l2_lower"].iloc[0])
    hiothers = float(d.loc[d.arm != w, "l2_upper"].max()) if len(d) > 1 else -np.inf
    L2SEP.append(dict(subset=cls, l_inf_winner=w, winner_l2_lower=lowin,
                      best_other_l2_upper=hiothers, winner_provably_unchanged=bool(lowin > hiothers)))
L2SEP = pd.DataFrame(L2SEP)
say(f"    L2 bracket at t=1 with the POOLED normaliser ({REC_INT} z, {REC_PATH:.4f} m): "
    "lower bound = the best grid rectangle inscribed in the L2 ball (per arm), "
    "upper bound = the L-inf cell itself.")
say(L2.round(4).to_string(index=False))
say(L2SEP.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# structural path ceilings (for F)
# ─────────────────────────────────────────────────────────────────────────────
say("\n-- structural path-coverage ceilings (kernel-centre availability) --")
_cc = ["arm", "subset", "seed", "kernel_center_uid", "centre_in_matched489"]
d1 = pd.read_parquet(RES / "e6_descriptors.parquet", columns=_cc)
d2 = pd.read_parquet(RES / "e6_descriptors_sakura.parquet", columns=_cc)
dd = pd.concat([d1, d2], ignore_index=True)
dd = dd[dd.centre_in_matched489.astype(bool)]
nreal = REAL[REAL.in_matched489.astype(bool)].groupby("subset").size()
CEIL = []
for arm in ARM_B100:
    sa = dd[dd.arm == arm]
    per, per_union, tot_c = {}, {}, []
    tot_n = 0
    for cls in CLASSES:
        n = int(nreal.get(cls, 0))
        sc = sa[sa['subset'] == cls]
        if n == 0 or sc.empty:
            per[cls] = per_union[cls] = None
            continue
        # per-seed distinct centres (coverage@1000 is computed inside ONE seed's pool,
        # then averaged over seeds), and the union over seeds for reference
        ks = sc.groupby("seed").kernel_center_uid.nunique()
        per[cls] = float(ks.mean()) / n
        per_union[cls] = int(sc.kernel_center_uid.nunique()) / n
        tot_c.append(float(ks.mean()))
        tot_n += n
    CEIL.append(dict(arm=arm, pooled_ceiling_per_seed=(sum(tot_c) / tot_n) if tot_n else np.nan,
                     n_real=tot_n,
                     **{f"ceil_{c}": per.get(c) for c in CLASSES},
                     **{f"union_{c}": per_union.get(c) for c in CLASSES}))
CEIL = pd.DataFrame(CEIL)
say(CEIL.round(4).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
say("\n" + "=" * 78)
say("FIGURES")
say("=" * 78)
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})
CURC = "#444444"
RECC = "#000000"
FIGS = []


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    FIGS.append(str(p))
    say(f"  wrote {p}")


def curve_panel(ax, tab, cls, space, band=True):
    t = tab[tab['subset'] == cls]
    for arm in ARMS:
        s = t[t.arm == arm].sort_values("eps")
        if s.empty:
            continue
        ax.plot(s["eps"], s["cvg"], color=COLOR[arm], lw=1.6, label=SHORT[arm], zorder=3)
        if band:
            ax.fill_between(s["eps"], s["lo"], s["hi"], color=COLOR[arm], alpha=0.16, lw=0, zorder=2)
    cur = CUR[cls][1] if space == "path" else CUR[cls][0]
    ax.axvline(cur, color=CURC, ls="--", lw=1.0, zorder=4)
    rec = REC_PATH if space == "path" else REC_INT
    ax.axvline(rec, color=RECC, ls=":", lw=1.3, zorder=4)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    n = int(t.n_real.max()) if len(t) else 0
    ax.set_title(f"{CLASS_LABEL[cls]}  (n={n})")


def fig_curves(tab, space, name, title, corner=False):
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.3), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        curve_panel(ax, tab, cls, space)
        ax.set_xlabel("$\\varepsilon_{path}$ [m, DTW]" if space == "path" else "$\\varepsilon_{int}$ [z units]")
    axes[0].set_ylabel(("corner " if corner else "") + "coverage@1000")
    h = [Line2D([], [], color=COLOR[a], lw=2, label=SHORT[a]) for a in ARMS]
    h += [Line2D([], [], color=CURC, ls="--", lw=1.2, label="current per-class $\\varepsilon$"),
          Line2D([], [], color=RECC, ls=":", lw=1.4,
                 label=f"recommended fixed $\\varepsilon$ ({REC_INT} z / {REC_PATH_ROUND} m)"),
          Patch(facecolor="0.6", alpha=0.3, label="min-max over 20 orderings x 3 seeds")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(title, y=1.04, fontsize=10.5)
    save(fig, name)


fig_curves(TAB["interaction"], "interaction", "fig1_coverage_vs_eps_interaction.png",
           "Fig 1  Interaction-space coverage@1000 vs $\\varepsilon_{int}$ "
           "(pool=all, real_set=matched489, joint→interaction rule)")
fig_curves(TAB["path"], "path", "fig2_coverage_vs_eps_path.png",
           "Fig 2  Path-space coverage@1000 vs $\\varepsilon_{path}$ (DTW to the own kernel centre)")
fig_curves(TAB["joint"], "joint", "fig3_coverage_vs_eps_joint_diagonal.png",
           "Fig 3  Joint coverage@1000 along the quantile-matched diagonal "
           "($\\varepsilon_{path}$ = pooled-NN partner of $\\varepsilon_{int}$)")

# fig4 joint grid heatmaps
fig, axes = plt.subplots(3, 5, figsize=(17.5, 10.6), gridspec_kw=dict(hspace=0.52, wspace=0.28))
for r, arm in enumerate(ARMS):
    for c, cls in enumerate(CLASSES):
        ax = axes[r, c]
        s = JGM[(JGM['subset'] == cls) & (JGM.arm == arm)]
        if s.empty:
            ax.text(0.5, 0.5, f"{SHORT[arm]}\nhas no {cls} arm", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.4")
            ax.set_axis_off()
            continue
        piv = s.pivot_table(index="eps_path", columns="eps_int", values="coverage_mean")
        X, Y = np.meshgrid(piv.columns.values, piv.index.values)
        m = ax.pcolormesh(X, Y, piv.values, cmap="viridis", vmin=0, vmax=1, shading="nearest")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot(CUR[cls][0], CUR[cls][1], marker="x", ms=8, mew=2, color="w", zorder=5)
        ax.plot(REC_INT, REC_PATH, marker="+", ms=10, mew=2, color="r", zorder=5)
        vc = CURTAB["joint"][(CURTAB["joint"]['subset'] == cls) & (CURTAB["joint"].arm == arm)]["cvg"]
        vf = TAB["joint"][(TAB["joint"]['subset'] == cls) & (TAB["joint"].arm == arm)
                          & np.isclose(TAB["joint"].eps, REC_INT)]["cvg"]
        ax.set_title(f"{SHORT[arm]} — {CLASS_ONELINE[cls]}\n"
                     f"x current {float(vc.iloc[0]):.3f}   + fixed {float(vf.iloc[0]):.3f}", fontsize=7.5)
        if r == 2:
            ax.set_xlabel("$\\varepsilon_{int}$ [z]")
        if c == 0:
            ax.set_ylabel("$\\varepsilon_{path}$ [m]")
cb = fig.colorbar(m, ax=axes, fraction=0.015, pad=0.012)
cb.set_label("joint coverage@1000")
fig.suptitle("Fig 4  Joint coverage@1000 over the full ($\\varepsilon_{int}$, $\\varepsilon_{path}$) grid  "
             "(white x = current per-class rule, red + = recommended fixed $\\varepsilon$)", y=0.955, fontsize=11)
save(fig, "fig4_joint_grid_heatmaps.png")

# fig5 rank map
fig, axes = plt.subplots(5, 3, figsize=(13.8, 9.4), sharex="col", gridspec_kw=dict(hspace=0.30))
for r, cls in enumerate(CLASSES):
    for c, sp in enumerate(SPACES):
        ax = axes[r, c]
        t = TAB[sp][TAB[sp]['subset'] == cls].pivot_table(index="eps", columns="arm", values="cvg").dropna(axis=1, how="all")
        e = t.index.values
        w = t.idxmax(axis=1).values
        best = t.max(axis=1).values
        second = np.sort(t.values, axis=1)[:, -2] if t.shape[1] > 1 else np.zeros(len(e))
        edges = np.sqrt(np.r_[e[0] ** 2 / e[1], e[:-1] * e[1:], e[-1] ** 2 / e[-2]])
        for i in range(len(e)):
            ax.axvspan(edges[i], edges[i + 1], color=COLOR[w[i]],
                       alpha=0.25 + 0.65 * min((best[i] - second[i]) / 0.3, 1.0), lw=0)
        ax.plot(e, best, color="0.1", lw=1.1, zorder=4)
        reg = REG[(REG.space == sp) & (REG['subset'] == cls)].iloc[0]
        ax.axvspan(edges[0], reg.disc_lo, color="w", alpha=0.55, lw=0, zorder=3)
        if np.isfinite(reg.sat_first):
            ax.axvspan(reg.disc_hi, edges[-1], color="w", alpha=0.55, lw=0, zorder=3)
        cur = CUR[cls][1] if sp == "path" else CUR[cls][0]
        rec = REC_PATH if sp == "path" else REC_INT
        ax.axvline(cur, color=CURC, ls="--", lw=1.1, zorder=6)
        ax.axvline(rec, color=RECC, ls=":", lw=1.4, zorder=6)
        ax.set_xscale("log"); ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, 1.02)
        if r == 0:
            ax.set_title(SPACE_LABEL[sp], fontsize=9.5)
        ax.plot([reg.disc_lo, reg.disc_hi], [0.035, 0.035], color="0.1", lw=3.2,
                solid_capstyle="butt", zorder=7)
        if c == 0:
            ax.set_ylabel(CLASS_LABEL[cls], fontsize=7, labelpad=2)
        if r == 4:
            ax.set_xlabel("$\\varepsilon_{path}$ [m]" if sp == "path" else "$\\varepsilon_{int}$ [z]")
h = [Patch(facecolor=COLOR[a], alpha=0.75, label=f"{SHORT[a]} is best") for a in ARMS]
h += [Patch(facecolor="w", edgecolor="0.6", label="starved / saturated (washed out)"),
      Line2D([], [], color="0.1", lw=3.2, label="discriminative band"),
      Line2D([], [], color="0.1", lw=1.2, label="best-arm coverage@1000"),
      Line2D([], [], color=CURC, ls="--", lw=1.2, label="current per-class $\\varepsilon$"),
      Line2D([], [], color=RECC, ls=":", lw=1.4, label="recommended fixed $\\varepsilon$")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.075))
fig.suptitle("Fig 5  Who wins as a function of $\\varepsilon$  (band colour = winning arm, "
             "saturation = margin over the runner-up)", y=0.965, fontsize=11)
save(fig, "fig5_rank_map.png")

# fig6 real-to-real NN distributions
CLSCOL = {"tlkeep": "#332288", "keeptl": "#88CCEE", "keeptl_sw": "#44AA99",
          "cutinl": "#DDCC77", "cutinr": "#AA4499"}
fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
for c, sp in enumerate(["interaction", "path"]):
    ax = axes[0, c]
    for cls in CLASSES:
        v = np.sort(nn[(nn['subset'] == cls) & (nn.space == sp)].nn_dist.values)
        if not len(v):
            continue
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=CLSCOL[cls], lw=1.4,
                label=f"{cls} (n={len(v)})")
        ax.axvline(CUR[cls][1] if sp == "path" else CUR[cls][0], color=CLSCOL[cls], ls="--", lw=0.9, alpha=0.8)
    pool = NN_PATH if sp == "path" else NN_INT
    ax.step(pool, np.arange(1, len(pool) + 1) / len(pool), where="post", color="k", lw=2.0,
            label=f"pooled (n={len(pool)})")
    rec = REC_PATH if sp == "path" else REC_INT
    ax.axvline(rec, color="r", ls=":", lw=2.0)
    q = ecdf_q(pool, rec)
    ax.annotate(f"fixed $\\varepsilon$ = {rec:.3f}\npooled q = {q:.3f}", xy=(rec, q),
                xytext=(0.06, 0.82), textcoords="axes fraction", color="r", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="r", lw=1.0))
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("real-to-real NN distance  " + ("[m, DTW]" if sp == "path" else "[z units]"))
    ax.set_ylabel("ECDF")
    ax.set_title(f"{SPACE_LABEL[sp]}  —  dashed = that class's current $\\varepsilon$")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax2 = axes[1, c]
    data = [nn[(nn['subset'] == cls) & (nn.space == sp)].nn_dist.values for cls in CLASSES]
    bp = ax2.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False)
    ax2.set_yticks(range(1, len(CLASSES) + 1))
    ax2.set_yticklabels([CLASS_ONELINE[c2].split(" / ")[1] for c2 in CLASSES], fontsize=7.5)
    for patch, cls in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(CLSCOL[cls]); patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("k")
    for i, cls in enumerate(CLASSES):
        ax2.plot(CUR[cls][1] if sp == "path" else CUR[cls][0], i + 1, marker="D", ms=5, color=CURC, zorder=5)
    ax2.axvline(rec, color="r", ls=":", lw=2.0)
    ax2.set_xscale("log")
    ax2.set_xlabel("real-to-real NN distance  " + ("[m, DTW]" if sp == "path" else "[z units]"))
    ax2.set_title("per class (box = IQR, whiskers 1.5 IQR, no fliers); ◆ = current $\\varepsilon$ (= the median)")
fig.suptitle("Fig 6  Real-to-real nearest-neighbour distances — the scale the $\\varepsilon$-ball has to match",
             y=0.985, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "fig6_real_nn_distributions.png")

# fig7 corner coverage
fig, axes = plt.subplots(3, 5, figsize=(16.5, 8.4), sharey=True)
for r, sp in enumerate(SPACES):
    for c, cls in enumerate(CLASSES):
        ax = axes[r, c]
        curve_panel(ax, TABC[sp], cls, sp)
        nr = int(TABC[sp][TABC[sp]['subset'] == cls].n_real.max())
        ax.set_title(f"{CLASS_ONELINE[cls]}\n{SPACE_LABEL[sp]}  (k_rare={nr})", fontsize=7.5)
        ax.set_xlabel("$\\varepsilon_{path}$ [m]" if sp == "path" else "$\\varepsilon_{int}$ [z]", fontsize=8)
    axes[r, 0].set_ylabel("corner coverage@1000")
h = [Line2D([], [], color=COLOR[a], lw=2, label=SHORT[a]) for a in ARMS]
h += [Line2D([], [], color=CURC, ls="--", lw=1.2, label="current per-class $\\varepsilon$"),
      Line2D([], [], color=RECC, ls=":", lw=1.4, label="recommended fixed $\\varepsilon$")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.035))
fig.suptitle("Fig 7  Corner coverage@1000 (top-20% most NN-isolated real scenarios) vs $\\varepsilon$",
             y=0.975, fontsize=11)
fig.tight_layout(rect=(0, 0.01, 1, 0.94))
save(fig, "fig7_corner_coverage_vs_eps.png")

# fig8 budget x eps
REP_EPS = [0.5, REC_INT, 1.25]
fig, axes = plt.subplots(len(REP_EPS), 5, figsize=(16.5, 8.4), sharey=True)
for r, ei in enumerate(REP_EPS):
    ep = DIAGMAP[round(float(ei), 6)]
    for c, cls in enumerate(CLASSES):
        ax = axes[r, c]
        for arm in ARM_B100:
            xs, ys, lo, hi = [], [], [], []
            for b in (10, 100, 1000):
                s = COV[(COV.space == "joint") & (COV.pool == "all") & (COV.real_set == "matched489")
                        & (COV.budget == b) & (COV.arm == arm) & (COV['subset'] == cls)
                        & np.isclose(COV.eps_int, ei)]
                s = s[s.eps_mult.isna()]
                if s.empty:
                    continue
                xs.append(b); ys.append(s.coverage_mean.mean())
                lo.append(s.coverage_min.min()); hi.append(s.coverage_max.max())
            if not xs:
                continue
            ls = "-" if arm in ARMS else "--"
            ax.plot(xs, ys, ls, marker="o", ms=3.5, color=COLOR[arm], lw=1.5, label=SHORT[arm])
            ax.fill_between(xs, lo, hi, color=COLOR[arm], alpha=0.14, lw=0)
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{CLASS_ONELINE[cls]}\n$\\varepsilon$ = ({ei:g} z, {ep:.3f} m)"
                     + ("  ← recommended" if ei == REC_INT else ""), fontsize=7.5)
        if r == len(REP_EPS) - 1:
            ax.set_xlabel("budget b (draws)")
    axes[r, 0].set_ylabel("joint coverage@b")
h = [Line2D([], [], color=COLOR[a], lw=2, ls="-" if a in ARMS else "--", label=SHORT[a]) for a in ARM_B100]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.035))
fig.suptitle("Fig 8  Budget x $\\varepsilon$: joint coverage@b for three $\\varepsilon$ on the quantile-matched "
             "diagonal (dashed = the executed-E3 SVD KDE arm, pool 43-100 draws, so it has no b=1000)",
             y=0.975, fontsize=11)
fig.tight_layout(rect=(0, 0.01, 1, 0.94))
save(fig, "fig8_budget_x_eps.png")

# ─────────────────────────────────────────────────────────────────────────────
# numbers for the report
# ─────────────────────────────────────────────────────────────────────────────
def curve_at(sp, cls, eps, arm, tab=None):
    tab = tab or TAB
    t = tab[sp][(tab[sp]['subset'] == cls) & (tab[sp].arm == arm)]
    if t.empty:
        return None
    i = int(np.argmin(np.abs(t.eps.values - eps)))
    return float(t["cvg"].values[i])


NUM = dict(
    gate=dict(n_cells=n_gate, max_absdiff=gmax),
    grids=dict(n_int=len(GRID_INT), n_path=len(GRID_PATH),
               int_min=float(GRID_INT.min()), int_max=float(GRID_INT.max()),
               path_min=float(GRID_PATH.min()), path_max=float(GRID_PATH.max())),
    hazard_h1=dict(n_single_class_joint_eps=N_LEAK, values=LEAK_VALS),
    bands={sp: BAND[sp] for sp in SPACES},
    recommendation=dict(eps_int=REC_INT, eps_path=REC_PATH, eps_path_print=REC_PATH_ROUND,
                        q_int=Q_REC[0], q_path=Q_REC[1],
                        alt_eps_int=ALT_INT, alt_eps_path=ALT_PATH, alt_q=Q_ALT,
                        pooled_median_int=POOL_MED_INT, pooled_median_path=POOL_MED_PATH),
    bold_moves=dict(coverage=n_mv_cov, corner=n_mv_cor,
                    coverage_groups=GRP_MV_COV, corner_groups=GRP_MV_COR),
    n_flips=len(FLIPS), n_flips_in_band=int(FLIPS.inside_discriminative_band.sum()),
)
write = lambda p, o: Path(p).write_text(json.dumps(o, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
write(OUT / "eps_sweep_report_numbers.json", dict(
    numbers=NUM, regimes=REG.to_dict("records"), robustness=ROB.to_dict("records"),
    robust_pm50=ROB50.to_dict("records"), reachability=REACH.to_dict("records"),
    correlations=CORR.to_dict("records"), elasticity=ELA.to_dict("records"),
    l2_bracket=L2.to_dict("records"), l2_separation=L2SEP.to_dict("records"),
    ceilings=CEIL.to_dict("records"), candidates=CAND.to_dict("records"),
    physical_units=PHYS, figures=FIGS))
say("\nwrote " + str(OUT / "eps_sweep_report_numbers.json"))

# ─────────────────────────────────────────────────────────────────────────────
# POOLED cross-class curves, with EQUAL class denominators per comparison
# (the verifier of 97 showed that mixing a 5-class arm with a 4-class arm
#  inflates the 4-class arm; every pooled number below names its class set)
# ─────────────────────────────────────────────────────────────────────────────
say("\n" + "=" * 78)
say("POOLED cross-class curves (real-weighted, EQUAL denominators)")
say("=" * 78)
CL4 = ["keeptl", "keeptl_sw", "cutinl", "cutinr"]          # the classes SAKURA has
CL5 = CLASSES
OURS, SVD, SAK = ARMS


def pooled_curve(sp, arms, classes):
    t = TAB[sp][TAB[sp]['subset'].isin(classes) & TAB[sp].arm.isin(arms)].copy()
    t["w"] = t["cvg"] * t["n_real"]
    g = t.groupby(["arm", "eps"]).agg(num=("w", "sum"), den=("n_real", "sum")).reset_index()
    g["cov"] = g.num / g.den
    piv = g.pivot(index="eps", columns="arm", values="cov")
    den = g.groupby("arm").den.max().to_dict()
    return piv, den


def first_cross(piv, a, b):
    """smallest grid eps where arm b >= arm a (a leads below it)."""
    d = piv[a] - piv[b]
    m = d.values < 0
    if not m.any():
        return None, None
    i = int(np.argmax(m))
    return float(piv.index.values[i]), (float(piv[a].values[i]), float(piv[b].values[i]))


POOLED = {}
for tag, arms, classes in (("ours_vs_svd_5class", [OURS, SVD], CL5),
                           ("three_arms_4class", ARMS, CL4)):
    for sp in SPACES:
        piv, den = pooled_curve(sp, arms, classes)
        spread = (piv.max(axis=1) - piv.min(axis=1))
        pk = float(spread.idxmax())
        cr, cv = first_cross(piv, OURS, SVD)
        POOLED[f"{tag}|{sp}"] = dict(
            classes=classes, n_real={k: int(v) for k, v in den.items()},
            spread_peak_eps=pk, spread_peak=float(spread.max()),
            ours_leads_up_to=cr, values_at_cross=cv,
            curve={f"{e:g}": {a: float(piv.loc[e, a]) for a in piv.columns} for e in piv.index})
        say(f"\n[{tag} | {sp}]  n_real {den}")
        say(f"   max-min arm spread peaks at eps={pk:.4f} (gap {spread.max():.4f})")
        if cr is not None:
            say(f"   Ours leads up to eps={cr:.4f} exclusive; at that eps Ours {cv[0]:.4f} vs SVD {cv[1]:.4f}")
        else:
            say("   Ours leads over SVD at every grid eps")
        show = [0.25, 0.5, 0.75, 1.0, 2.0] if sp != "path" else [0.1, 0.25, 0.5, 1.0, 3.0]
        for e in show:
            i = int(np.argmin(np.abs(piv.index.values - e)))
            say("      eps=%-8.4f " % piv.index.values[i]
                + "  ".join(f"{TINY[a]}={piv.iloc[i][a]:.4f}" for a in piv.columns))

# per-anchor comparison of the three candidate fixed eps
say("\n-- the three candidate fixed eps, side by side (joint, coverage@1000) --")
ANCH = []
for tag, ei in (("FIRST 0.75z/0.35m", REC_INT), ("SECOND 0.88z/0.45m", ALT_INT),
                ("THIRD 1.0z/0.54m", ALT2_INT)):
    for cls in CLASSES:
        t = TAB["joint"][(TAB["joint"]['subset'] == cls) & np.isclose(TAB["joint"].eps, ei)].set_index("arm")["cvg"]
        tc = TABC["joint"][(TABC["joint"]['subset'] == cls) & np.isclose(TABC["joint"].eps, ei)].set_index("arm")["cvg"]
        cc = CURTAB["joint"][CURTAB["joint"]['subset'] == cls].set_index("arm")["cvg"]
        kc = CURTABC["joint"][CURTABC["joint"]['subset'] == cls].set_index("arm")["cvg"]
        ANCH.append(dict(anchor=tag, eps_int=ei, eps_path=DIAGMAP[round(float(ei), 6)], subset=cls,
                         winner_current=";".join(sorted(bold_set(cc))), winner_fixed=";".join(sorted(bold_set(t))),
                         cov_winner_fixed=float(t.max()), cov_runnerup_fixed=float(t.drop(t.idxmax()).max()) if len(t) > 1 else np.nan,
                         corner_winner_current=";".join(sorted(bold_set(kc))),
                         corner_winner_fixed=";".join(sorted(bold_set(tc))),
                         bold_moves=bold_set(cc) != bold_set(t),
                         corner_bold_moves=bold_set(kc) != bold_set(tc)))
ANCH = pd.DataFrame(ANCH)
say(ANCH.to_string(index=False, columns=["anchor", "subset", "winner_current", "winner_fixed",
                                         "bold_moves", "corner_winner_current", "corner_winner_fixed",
                                         "corner_bold_moves"]))
for tag in ANCH.anchor.unique():
    d = ANCH[ANCH.anchor == tag]
    say(f"   {tag}: {int(d.bold_moves.sum())} coverage-bold group changes, "
        f"{int(d.corner_bold_moves.sum())} corner-bold group changes")

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
say("\n" + "=" * 78)
say("REPORT")
say("=" * 78)


def mdtab(df, cols, heads, fmts, align=None):
    align = align or ["---:" if isinstance(f, str) and f != "s" else "---" for f in fmts]
    L = ["| " + " | ".join(heads) + " |", "| " + " | ".join(align) + " |"]
    for r in df.itertuples():
        cells = []
        for c, f in zip(cols, fmts):
            v = getattr(r, c)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("–")
            elif f == "s":
                cells.append(str(v))
            else:
                cells.append(format(v, f))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


AN = lambda arm: SHORT[arm]
REGP = REG.copy()
REGP["cls"] = REGP['subset'].map(CLASS_ONELINE)
REGP["sp"] = REGP.space.map(SPACE_LABEL)
REGP["band"] = REGP.apply(lambda r: f"[{r.disc_lo:.4f}, {r.disc_hi:.4f}]", axis=1)
REGP["curin"] = np.where(REGP.cur_in_band, "yes", "**NO**")

FB = FLIPS[FLIPS.inside_discriminative_band].copy()
FB["cls"] = FB['subset'].map(CLASS_ONELINE)
FB["sp"] = FB.space
FB["wb"] = FB.winner_before.map(TINY)
FB["wa"] = FB.winner_after.map(TINY)
FB["cross"] = np.where(FB.brackets_current_eps, "**yes**", "")

ROB["band_common_lo"] = ROB.space.map(lambda x: BAND[x][0])
ROB["band_common_hi"] = ROB.space.map(lambda x: BAND[x][1])
SAFE = ROB[ROB.winner_constant_over_common_band & (ROB.winners_in_common_band == ROB.winner.map(TINY))]
DIFFW = ROB[ROB.winner_constant_over_common_band & (ROB.winners_in_common_band != ROB.winner.map(TINY))]

ROBP = ROB.copy()
ROBP["cls"] = ROBP['subset'].map(CLASS_ONELINE)
ROBP["w"] = ROBP.winner.map(TINY)
ROBP["stable"] = ROBP.apply(lambda r: f"[{r.stable_lo:.4f}, {r.stable_hi:.4f}] (x{r.stability_radius:.2f})", axis=1)
ROBP["verdict"] = np.where(ROBP.winner_constant_over_band, "**eps-ROBUST**", "eps-fragile")
ROBP["common"] = np.where(ROBP.winner_constant_over_common_band,
                          "**stable** (" + ROBP.winners_in_common_band + ")",
                          "flips (" + ROBP.winners_in_common_band + ")")

ELAF = ELA[ELA.operating_point == "fixed"].copy()
ELAF["cls"] = ELAF['subset'].map(CLASS_ONELINE)
ELAF["a"] = ELAF.arm.map(TINY)
ELAC = ELA[ELA.operating_point == "current"].copy()
ELAC["cls"] = ELAC['subset'].map(CLASS_ONELINE)
ELAC["a"] = ELAC.arm.map(TINY)

CORRP = CORR.copy()
CORRP["lab"] = CORRP.scope.map(lambda x: CLASS_ONELINE.get(x, "**pooled (all 5 classes)**"))

L2P = L2.copy()
L2P["cls"] = L2P['subset'].map(CLASS_ONELINE)
L2P["a"] = L2P.arm.map(TINY)
L2SP = L2SEP.copy()
L2SP["cls"] = L2SP['subset'].map(CLASS_ONELINE)
L2SP["w"] = L2SP.l_inf_winner.map(TINY)
L2SP["v"] = np.where(L2SP.winner_provably_unchanged, "**yes**", "undecided")

REACHP = REACH.copy()
REACHP["cls"] = REACHP['subset'].map(CLASS_ONELINE)

CEILP = CEIL.copy()
CEILP["a"] = CEILP.arm.map(SHORT)

PHYSD = pd.DataFrame([dict(cls=CLASS_ONELINE[c], **{k: PHYS[c][k]["one_axis"] for k in DESC},
                           iso=PHYS[c]["pet"]["isotropic"]) for c in CLASSES])

ANCHS = (ANCH.groupby("anchor")
         .agg(eps_int=("eps_int", "first"), eps_path=("eps_path", "first"),
              n_cov=("bold_moves", "sum"), n_cor=("corner_bold_moves", "sum"))
         .reset_index())
ANCHS["q"] = ANCHS.eps_int.map(lambda e: ecdf_q(NN_INT, e))

P5 = POOLED["ours_vs_svd_5class|joint"]["curve"]
P5I = POOLED["ours_vs_svd_5class|interaction"]["curve"]
P5P = POOLED["ours_vs_svd_5class|path"]["curve"]
P4J = POOLED["three_arms_4class|joint"]["curve"]


def pline(curve, e, arms):
    k = min(curve.keys(), key=lambda x: abs(float(x) - e))
    return k, {a: curve[k][a] for a in arms if a in curve[k]}


FIGREL = [f"figures/eps_sweep/{Path(f).name}" for f in FIGS]

_b = ELA[(ELA.operating_point == "fixed") & (ELA.arm == OURS)].set_index("subset").binding.to_dict()
_gp = {k: [CLASS_ONELINE[c].split(" / ")[0] for c in CLASSES if _b.get(c) == k] for k in ("path", "interaction", "tie")}
_gz = {k: [CLASS_ZH[c] for c in CLASSES if _b.get(c) == k] for k in ("path", "interaction", "tie")}
BINDSENT = "; ".join(f"`eps_{'path' if k == 'path' else 'int'}` binds in {', '.join(v)}" if k != "tie"
                     else f"the two are within 5 % of each other in {', '.join(v)}"
                     for k, v in _gp.items() if v)
BINDSENT_ZH = "；".join((("軌跡 ε 主導" if k == "path" else "互動 ε 主導") + "：" + "、".join(v)) if k != "tie"
                       else ("兩者相差 5 % 以內：" + "、".join(v)) for k, v in _gz.items() if v)

_pl = []
for _sp, _cu, _es in (("interaction", P5I, [0.25, 0.5, 0.75, 1.0, 2.0]),
                      ("path", P5P, [0.1, 0.25, 0.5, 1.0, 3.0]),
                      ("joint", P5, [0.5, 0.75, 1.0, 1.5, 2.0])):
    _lead = POOLED["ours_vs_svd_5class|" + _sp]["ours_leads_up_to"]
    _lead = f"{_lead:.4f}" if _lead else "always"
    for _e in _es:
        _k, _v = pline(_cu, _e, [OURS, SVD])
        _pl.append(f"| {_sp} | {_k} | {_v[OURS]:.4f} | {_v[SVD]:.4f} | {_lead} |")
POOLED_MD = "\n".join(_pl)

EXEC = f"""## Executive summary — the three questions, answered

1. **How does epsilon change coverage?** Every arm's curve has the same three regimes — starvation, a
   discriminative band, saturation. All five classes are discriminative together only inside
   `eps_int` ∈ [{BAND['interaction'][0]:.4f}, {BAND['interaction'][1]:.4f}] z, `eps_path` ∈ [{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m and, jointly,
   `eps_int` ∈ [{BAND['joint'][0]:.4f}, {BAND['joint'][1]:.4f}] z; outside that window the epsilon choice stops carrying information
   (Fig 1–3, 5).
2. The **path axis is the tight one**: its all-class discriminative window is only
   [{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m wide, against [{BAND['interaction'][0]:.2f}, {BAND['interaction'][1]:.2f}] z for interaction.
   Two of the five current per-class `eps_path` values ({', '.join(REG[(REG.space=='path')&(~REG.cur_in_band)]['subset'])}) already sit outside it.
3. **Recommended fixed pair: `eps_int` = {REC_INT} z and `eps_path` = {REC_PATH_ROUND} m** (exactly {REC_PATH:.4f} m,
   the pooled-quantile partner of {REC_INT} z at q = {Q_REC[0]:.3f}). It is the only round grid point inside every
   class's discriminative band in all three spaces.
4. Where it sits: the {100*Q_REC[0]:.0f}-th percentile of the pooled real-to-real nearest-neighbour distribution
   (interaction, n = {len(NN_INT)}) and the {100*Q_REC[1]:.0f}-th of the path one (n = {len(NN_PATH)}) — i.e. still "about as close as two
   recorded scenarios of the same class typically are", the same idea as today, only pooled instead of per class.
5. Cost of adopting it: **{n_mv_cov} of {len(D)} coverage cells and {n_mv_cor} of {len(D)} corner cells change bold status**
   ({', '.join(GRP_MV_COV)} for coverage, {', '.join(GRP_MV_COR)} for the corner tie). Everything else in the three
   thesis tables is unchanged (Section D and results/eps_sweep/tables3_coverage_at_fixed_eps.md).
6. Both moves favour ours, so **say so in the text**. The minimum-disruption alternative is
   `eps_int` = {ALT_INT:.3f} z / `eps_path` = {ALT_PATH:.3f} m ({int(ANCHS[ANCHS.anchor.str.startswith('SECOND')].n_cov.iloc[0])} coverage move,
   {int(ANCHS[ANCHS.anchor.str.startswith('SECOND')].n_cor.iloc[0])} corner move), at the price of leaving the path-axis discriminative band.
7. **Which conclusions survive any epsilon?** Inside the all-class joint window
   [{BAND['joint'][0]:.4f}, {BAND['joint'][1]:.4f}] z, only {int(ROB[ROB.space=='joint'].winner_constant_over_common_band.sum())} of the 5 class winners never change:
   {', '.join(CLASS_ONELINE[r['subset']] for _, r in ROB[(ROB.space=='joint')&ROB.winner_constant_over_common_band].iterrows())} — both Ours+KDE.
   Agent-LT N→E, Agent-LT S→W and Cut-in (right) each flip inside that window, so those three coverage
   sentences must carry the epsilon they were read at.
8. **Should epsilon be split into path and interaction? YES — keep two numbers.** They are in different units
   (dimensionless z vs metres), a scenario isolated in one space is largely not isolated in the other
   (pooled Spearman {CORR[CORR.scope=='pooled'].spearman.iloc[0]:.3f}, per class {CORR[CORR.scope!='pooled'].spearman.min():+.3f}…{CORR[CORR.scope!='pooled'].spearman.max():+.3f}), and the two thresholds
   bind in different classes (Section E-iii): at the recommended point {BINDSENT}.
9. What you may fix is **one quantile q**, not one scalar: read `eps_int` and `eps_path` off each space's own pooled
   real-NN ECDF at the same q. q = 0.50 is today's rule; q = {Q_REC[0]:.2f} is the recommendation above.
10. A single scalar is only possible with an explicit weighting, and the current per-class rule already *is* one
    (an L-infinity ball in units of each class's median NN, at radius 1). Replacing L-inf by L2 is undecidable at
    this grid resolution in 3 of 5 classes (Section E-iv), so there is nothing to gain from collapsing the two.
"""

METHOD = f"""## What was swept, and how to read the numbers

- Source of every coverage number: `results/eps_sweep/eps_sweep_coverage.csv` (full real set),
  `eps_sweep_corner.csv` (the top-20 % most NN-isolated reals) and `eps_sweep_joint_grid.csv`
  (the 2-D surface), all written by `scripts/97_eps_sweep.py`. This script recomputes no coverage.
- Reporting configuration, identical to the thesis tables: space = joint, pool = all, real_set = matched489,
  budget = 1000, mean over 3 seeds × 20 orderings. Bands in the figures are min–max over those 60 curves.
- Arms: the three that actually have a coverage@1000 cell — {', '.join(SHORT[a] for a in ARMS)}.
  `svd_d5_kde_executed_E3` is a fourth KDE arm but its pool is 43–100 draws, so it has no budget-1000 value at
  all; it appears only on Fig 8.
- Grids: {len(GRID_INT)} shared global `eps_int` values from {GRID_INT.min()} to {GRID_INT.max()} z and {len(GRID_PATH)} shared global
  `eps_path` values from {GRID_PATH.min()} to {GRID_PATH.max()} m. Every cross-class aggregation below is restricted to those
  shared values.
- **Gate.** Re-reading the current-rule rows (`eps_mult == 1.0`) reproduces every coverage /
  corner-coverage cell of `results/final/tables3/gen_metrics.csv`: {n_gate} cells, 0 missing,
  max |diff| = {gmax:.2e} (tolerance 1e-12).
- **Two traps in the sweep CSVs, avoided here.** (H1) The joint rows carry `eps_mult = NaN` at all
  {len(leak)} `eps_int` values of each class grid, of which {N_LEAK} exist for one class only; filtering on
  `eps_mult` and aggregating across classes silently mixes single-class numbers, so this script filters on
  the {len(GRID_INT)} shared values explicitly. (H2) The *current* rule is the `eps_mult == 1.0` row, not the
  joint-diagonal row with the same `eps_int` — the diagonal pairs that `eps_int` with a different `eps_path`.
- Bold ties use scripts/90's tolerance (|Δ| ≤ 1e-12), so the keeptl corner tie is preserved.
"""

SEC_A = f"""## A. The effect of epsilon — three regimes

Definitions used throughout (fixed before looking at the data):
**starvation** = the best arm's coverage@1000 < {STARV:.2f}; **saturation** = the best arm's coverage@1000 > {SAT:.2f};
**discriminative band** = the grid values strictly between the two.

{mdtab(REGP, ["sp", "cls", "n_real", "starv_last", "band", "sat_first", "best_max", "cur_eps", "curin"],
       ["space", "class", "n_real", "starved up to", "discriminative band", "saturated from", "best arm max",
        "current eps", "current eps in band?"],
       ["s", "s", "d", ".4f", "s", ".4f", ".4f", ".4f", "s"])}

Intersection of the five class bands:
**interaction [{BAND['interaction'][0]:.4f}, {BAND['interaction'][1]:.4f}] z**,
**path [{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m**,
**joint [{BAND['joint'][0]:.4f}, {BAND['joint'][1]:.4f}] z** (on the quantile-matched diagonal).

Three things follow.

1. *The path axis is the binding constraint on any fixed choice.* Its all-class window is a factor
   {BAND['path'][1]/BAND['path'][0]:.2f} wide; the interaction one is a factor {BAND['interaction'][1]/BAND['interaction'][0]:.2f} wide. The two cut-in classes saturate the path
   space at 0.5 m, which is why the current cutinl (`eps_path` {CUR['cutinl'][1]:.4f} m) and cutinr ({CUR['cutinr'][1]:.4f} m)
   thresholds sit **outside** the common band — their path column is already almost non-discriminative.
2. *The interaction space never truly starves at realistic epsilon but saturates early.* At `eps_int` = 1.1631 z
   four of five classes are above {SAT:.2f}; the interaction ranking read at large epsilon says nothing.
3. *The joint space is the slowest to saturate* (tlkeep's best arm only reaches {REG[(REG.space=='joint')&(REG['subset']=='tlkeep')].best_max.iloc[0]:.4f} at `eps_int` 6 z),
   which is why the thesis reports it.

Pooled cross-class curves, real-weighted, **with equal class denominators** (mixing a 5-class arm with the
4-class SAKURA arm inflates SAKURA — every row below names its class set):

Ours+KDE vs SVD_d5+KDE, all five classes (interaction / joint n = {POOLED['ours_vs_svd_5class|joint']['n_real'][OURS]}, path n = {POOLED['ours_vs_svd_5class|path']['n_real'][OURS]}):

| space | eps | Ours+KDE | SVD_d5+KDE | Ours leads up to |
| --- | ---: | ---: | ---: | ---: |
""" + POOLED_MD + f"""

The arm separation is largest exactly where the recommendation lands: the max–min spread over the three arms
peaks at `eps_int` = {POOLED['three_arms_4class|joint']['spread_peak_eps']:.4f} on the joint diagonal (gap {POOLED['three_arms_4class|joint']['spread_peak']:.4f}, four common classes) and at
`eps_int` = {POOLED['ours_vs_svd_5class|joint']['spread_peak_eps']:.4f} for Ours-vs-SVD over all five (gap {POOLED['ours_vs_svd_5class|joint']['spread_peak']:.4f}).

Figures: `{FIGREL[0]}`, `{FIGREL[1]}`, `{FIGREL[2]}`, `{FIGREL[3]}`.
"""

SEC_B = f"""## B. Rank stability — which coverage conclusions are epsilon-robust

{len(FLIPS)} winner changes occur on the 31-point global grid; **{len(FB)} of them fall inside the class's own
discriminative band**, i.e. they are not artefacts of a starved or saturated regime.

{mdtab(FB, ["sp", "cls", "eps_before", "wb", "cov_before_winner", "eps_after", "wa", "cov_after_winner", "cross"],
       ["space", "class", "eps before", "winner", "cov", "eps after", "winner", "cov", "brackets the current eps?"],
       ["s", "s", ".4f", "s", ".4f", ".4f", "s", ".4f", "s"])}

Read at each class's **current** epsilon, and how far that verdict survives:

{mdtab(ROBP, ["space", "cls", "cur_eps", "w", "cov", "runner_up", "margin", "stable", "verdict", "common"],
       ["space", "class", "current eps", "winner", "winner cov", "runner-up", "margin", "winner constant on",
        "constant over its OWN band?", "over the ALL-CLASS band?"],
       ["s", "s", ".4f", "s", ".4f", ".4f", ".4f", "s", "s", "s"])}

The last column is the one that matters for a fixed epsilon: it asks whether the winner is the same at *every*
epsilon a fixed choice could sensibly take, i.e. anywhere in the all-class discriminative band
(joint [{BAND['joint'][0]:.4f}, {BAND['joint'][1]:.4f}] z, interaction [{BAND['interaction'][0]:.4f}, {BAND['interaction'][1]:.4f}] z, path [{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m).

**What this means for the thesis text.**

- *Safe to state with no epsilon attached* — the winner is the same at every epsilon in the all-class band
  **and** it is the winner the thesis already prints:
  {'; '.join(f"{CLASS_ONELINE[r['subset']]} / {r.space} → {r.winners_in_common_band}" for _, r in SAFE.iterrows())}.
  In the **joint** space (the one the tables report) that is {int(len(SAFE[SAFE.space=='joint']))} of 5 classes:
  {', '.join(CLASS_ONELINE[r['subset']] for _, r in SAFE[SAFE.space=='joint'].iterrows())} — both **Ours+KDE**.
  tlkeep's joint winner does flip, but only at {float(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='tlkeep')].eps_after.iloc[0]):.4f} z, far above the top of the all-class band.
- *Epsilon-fragile — must carry the epsilon.* In the joint space: Agent-LT N→E changes hands
  {int(len(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='keeptl')]))} times between {FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='keeptl')].eps_before.min():.4f} and {FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='keeptl')].eps_after.max():.4f} z before settling on SVD;
  Agent-LT S→W goes Ours→SVD at {float(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='keeptl_sw')].eps_after.iloc[0]):.4f} z; Cut-in (right) goes Ours→SVD at
  {float(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='cutinr')].eps_after.iloc[0]):.4f} z. The current per-class thresholds happen to sit on opposite sides of those three
  flips, which is precisely why the published table shows a mixed picture.
- *A third category: the current epsilon is outside the all-class band, so the "stable" winner is not the one
  printed.* {'; '.join(f"{CLASS_ONELINE[r['subset']]} / {r.space}: printed {TINY[r.winner]} at eps {r.cur_eps:.4f}, but {r.winners_in_common_band} everywhere in [{r.band_common_lo:.4f}, {r.band_common_hi:.4f}]" for _, r in DIFFW.iterrows()) if len(DIFFW) else 'none'}.
  These are classes whose current threshold is already past its saturation onset (Section A), so the printed
  winner is read in a regime where the metric no longer separates the arms.
- The stability radius column says how much room there is: the current keeptl joint verdict survives only a
  ×{float(ROB[(ROB.space=='joint')&(ROB['subset']=='keeptl')].stability_radius.iloc[0]):.2f} change in epsilon, and the cutinl one a ×{float(ROB[(ROB.space=='joint')&(ROB['subset']=='cutinl')].stability_radius.iloc[0]):.2f} change.
  **A reader who re-derives your epsilon with a slightly different rule can overturn three of the five class
  verdicts but not the other two.**

Figure: `{FIGREL[4]}` (coloured band = winning arm, colour intensity = margin over the runner-up, the dark bar
at the bottom of each panel = that class's discriminative band).
"""

SEC_C = f"""## C. The fixed-epsilon recommendation

### FIRST choice — `eps_int` = {REC_INT} z, `eps_path` = {REC_PATH_ROUND} m

(exact `eps_path` = {REC_PATH:.6f} m, the pooled-ECDF quantile partner of {REC_INT} z; both are measured grid
values, so every number below is read, not interpolated.)

**(i) Inside every class's discriminative band.** It is the *only* round grid point that is:
{REC_INT} z ∈ interaction [{BAND['interaction'][0]:.4f}, {BAND['interaction'][1]:.4f}] ✔, joint [{BAND['joint'][0]:.4f}, {BAND['joint'][1]:.4f}] ✔ and
{REC_PATH:.4f} m ∈ path [{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] ✔. The full candidate screen (`in_all_bands` column of the
`candidates` block in `eps_sweep_report_numbers.json`) leaves only three anchors passing all three bands —
{', '.join(f'{v:.4f}' for v in CAND[CAND.in_all_bands].eps_int)} z — and {REC_INT} has the largest rank-stability radius of the three
({float(CAND[np.isclose(CAND.eps_int, REC_INT)].min_stability_radius.iloc[0]):.3f} vs {float(CAND[np.isclose(CAND.eps_int, 0.642490)].min_stability_radius.iloc[0]):.3f} and {float(CAND[np.isclose(CAND.eps_int, 0.676433)].min_stability_radius.iloc[0]):.3f}).

**(ii) Physically interpretable.** `eps_path` = {REC_PATH_ROUND} m is a DTW distance in metres: two target paths are
"the same" if they stay about a third of a metre apart on average — roughly a tenth of a 3.5 m lane,
and far below the ≈3 m lateral offset a lane change produces. `eps_int` = {REC_INT} z is an L2 distance in the 6-D
z-scored interaction space. If the whole gap sits in a single descriptor it is 0.75 × that class's real SD:

{mdtab(PHYSD, ["cls"] + DESC, ["class", "PET [s]", "d_min [m]", "α [deg]", "c_x [m]", "c_y [m]", "u_c [m/s]"],
       ["s"] + [".2f"] * 6)}

If instead the gap is spread evenly over all six descriptors each one moves by 0.75/√6 = {REC_INT/np.sqrt(6):.3f} SD,
e.g. {PHYS['tlkeep']['pet']['isotropic']:.2f} s of PET and {PHYS['tlkeep']['d_min']['isotropic']:.2f} m of minimum distance for tlkeep, or
{PHYS['cutinr']['d_min']['isotropic']:.2f} m and {PHYS['cutinr']['pet']['isotropic']:.2f} s for Cut-in (right). **This is a loose ball in the single-descriptor
direction and a tight one in the isotropic direction** — worth one sentence in the thesis, because a reader
who only sees "0.75 z" will not guess that it permits a 3.7 s PET difference on tlkeep.

**(iii) Rank robustness to ±50 %.** Over `eps_int` ∈ [{0.5*REC_INT:.4f}, {1.5*REC_INT:.4f}] z the class winner is unchanged in
{int(ROB50[(ROB50.space=='joint')].stable.sum())}/5 classes in the joint space, {int(ROB50[(ROB50.space=='interaction')].stable.sum())}/5 in interaction and {int(ROB50[(ROB50.space=='path')].stable.sum())}/5 in path
(`robust_pm50` in the JSON). No anchor anywhere on the grid does better than 3/5 in the joint space — the
±50 % test is simply harder than the data can support, which is itself the honest finding of Section B.

**(iv) Position on the pooled real-to-real NN distribution.** `eps_int` {REC_INT} z is the
**{100*Q_REC[0]:.2f}-th percentile** of the pooled interaction NN distribution (n = {len(NN_INT)}, median {POOL_MED_INT:.4f} z);
`eps_path` {REC_PATH:.4f} m is the **{100*Q_REC[1]:.2f}-th** of the pooled path one (n = {len(NN_PATH)}, median {POOL_MED_PATH:.4f} m).
The strict pooled-median (q = 0.50) variant would be {POOL_MED_INT:.4f} z / {POOL_MED_PATH:.4f} m; it gives the same bold
pattern as {REC_INT} z (nearest grid anchor {Q50_INT_GRID:.4f}), so the choice between q = 0.50 and q = {Q_REC[0]:.2f} is not
load-bearing. Note the per-class spread the fixed value replaces: interaction {min(v[0] for v in CUR.values()):.4f}…{max(v[0] for v in CUR.values()):.4f} z
(×{max(v[0] for v in CUR.values())/min(v[0] for v in CUR.values()):.2f}) but path {min(v[1] for v in CUR.values()):.4f}…{max(v[1] for v in CUR.values()):.4f} m (×{max(v[1] for v in CUR.values())/min(v[1] for v in CUR.values()):.2f}) — **the interaction axis barely
changes, the path axis changes a lot**, and tlkeep is the class that moves most
(its `eps_path` goes {CUR['tlkeep'][1]:.4f} → {REC_PATH:.4f} m, +{100*(REC_PATH/CUR['tlkeep'][1]-1):.0f} %).

**(v) How many real scenarios become unreachable or trivially reachable.**

{mdtab(REACHP, ["cls", "n_real_joint", "iso_int", "iso_path", "dense_int", "dense_path", "best_cov",
                "n_missed_by_best", "all_cov", "n_hit_by_all"],
       ["class", "n_real", "isolated in interaction (NN > eps_int)", "isolated in path (NN > eps_path)",
        "dense in interaction (NN ≤ eps/2)", "dense in path", "best-arm cov@1000",
        "missed by the best arm", "cov reached by ALL arms", "hit by every arm"],
       ["s", "d", "d", "d", "d", "d", ".4f", "d", ".4f", "d"])}

"Isolated" means no *other recorded* scenario of the class lies within the fixed epsilon, so the generator
cannot cover it by landing near a neighbour — it has to reproduce that scenario itself. At {REC_INT} z /
{REC_PATH:.3f} m that is {int(REACH.iso_int.sum())}/{int(REACH.n_real_int.sum())} reals in interaction and {int(REACH.iso_path.sum())}/{int(REACH.n_real_path.sum())} in path. "Trivially reachable" in the
sense of *covered by every one of the three arms* is only {int(REACH.n_hit_by_all.sum())} reals in total, so the fixed epsilon does not
hand out free coverage.

### SECOND choice — `eps_int` = {ALT_INT:.4f} z, `eps_path` = {ALT_PATH:.4f} m (minimum disruption)

{mdtab(ANCHS, ["anchor", "eps_int", "eps_path", "q", "n_cov", "n_cor"],
       ["candidate", "eps_int [z]", "eps_path [m]", "pooled q (interaction)", "coverage groups whose bold moves",
        "corner groups whose bold moves"], ["s", ".4f", ".4f", ".4f", "d", "d"])}

This is the anchor that leaves the published coverage column bold-identical
({int(ANCHS[ANCHS.anchor.str.startswith('SECOND')].n_cov.iloc[0])} coverage move). Its trade-off: `eps_path` {ALT_PATH:.4f} m is outside the all-class path band
[{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m — it sits between the top of the two cut-in classes' discriminative band
({BAND['path'][1]:.4f} m) and their saturation onset (0.5 m), so the path column barely separates the arms there;
and it sits at the {100*Q_ALT[0]:.0f}-th pooled NN
percentile, which is harder to describe as "as close as two real scenarios typically are".
A third, looser variant `eps_int` = {ALT2_INT} z / `eps_path` = {ALT2_PATH:.4f} m sits at the geometric centre of the joint
band ({BAND_GEOM_MID:.4f} z) but moves the Agent-LT S→W coverage bold to SVD.

### Which method does the choice favour?

Honestly: **the FIRST choice favours ours in both cells that move** — Cut-in (right) coverage goes
SVD {float(D[(D['subset']=='cutinr')&(D.arm==SVD)].coverage_current.iloc[0]):.3f} → {float(D[(D['subset']=='cutinr')&(D.arm==SVD)].coverage_fixed.iloc[0]):.3f} while ours goes {float(D[(D['subset']=='cutinr')&(D.arm==OURS)].coverage_current.iloc[0]):.3f} → {float(D[(D['subset']=='cutinr')&(D.arm==OURS)].coverage_fixed.iloc[0]):.3f}, and the Agent-LT N→E
corner tie breaks in ours' favour ({float(D[(D['subset']=='keeptl')&(D.arm==SVD)].corner_fixed.iloc[0]):.3f} vs {float(D[(D['subset']=='keeptl')&(D.arm==OURS)].corner_fixed.iloc[0]):.3f}). The mechanism is not arbitrary — a
smaller epsilon rewards landing *on* the recorded scenario, which is what a conflict-anchored sampler does,
while a larger epsilon rewards spreading mass, which is what the SVD+KDE fit does — but it is a real
directional effect and the thesis should print the sentence "the fixed epsilon was chosen from the
discriminative-band criterion, not from the ranking; it moves two cells, both towards ours" rather than let a
reviewer find it. If you would rather not defend that, take the SECOND choice and accept a weaker path axis.

Figure: `{FIGREL[5]}` (where the fixed epsilon sits on each class's NN distribution).
"""

SEC_D = f"""## D. The thesis tables recomputed at the fixed epsilon

Full table: **`results/eps_sweep/tables3_coverage_at_fixed_eps.md`** and `.csv`
(the `.csv` carries `bold_coverage_moves` / `bold_corner_moves` flags per cell).

{md_table_D()}

- {n_mv_cov} of {len(D)} coverage cells and {n_mv_cor} of {len(D)} corner cells change bold status.
- The RareCase table (Special 39_180) has no coverage column — it reports a recovered fraction against a
  single recorded scenario — so it is unaffected by the epsilon choice in its coverage sense, though its
  `Recovered frac.` cell does use the cutinl class epsilon and would move with it.
- The absolute level shifts a lot even where the ranking does not: tlkeep Ours+KDE goes
  {float(D[(D['subset']=='tlkeep')&(D.arm==OURS)].coverage_current.iloc[0]):.3f} → {float(D[(D['subset']=='tlkeep')&(D.arm==OURS)].coverage_fixed.iloc[0]):.3f} because its current `eps_path` ({CUR['tlkeep'][1]:.4f} m) is the tightest of the
  five. If you adopt the fixed epsilon, every coverage number in the thesis changes and the *captions*
  must change with them.

Figures: `{FIGREL[3]}` (the operating point on the 2-D surface), `{FIGREL[6]}` (corner), `{FIGREL[7]}` (budget × epsilon).
"""

SEC_E = f"""## E. Should epsilon be split into path and interaction?

**Recommendation: yes — keep two numbers, and tie them together through one quantile, not one scalar.**
The evidence, in the order the question asks for it.

**(i) They are incommensurable.** `eps_int` is a dimensionless L2 distance in a 6-D z-scored space;
`eps_path` is a DTW distance in metres. A single scalar can only exist through an explicit weighting
w such that `d = max(d_int/w_int, d_path/w_path)` or an L2 analogue — i.e. you do not remove the second
number, you hide it inside w. This is not a philosophical point: with `eps_int` = {REC_INT} z ≈ {PHYS['tlkeep']['pet']['one_axis']:.1f} s of tlkeep
PET and `eps_path` = {REC_PATH_ROUND} m, there is no exchange rate between "seconds of PET" and "metres of DTW"
that the data supplies.

**(ii) Isolation in one space does not predict isolation in the other.**
Correlation between a real scenario's interaction NN distance and its path NN distance
(`results/eps_sweep/eps_real_nn_values.csv`, matched489):

{mdtab(CORRP, ["lab", "n", "spearman", "pearson"], ["scope", "n", "Spearman ρ", "Pearson r"],
       ["s", "d", "+.4f", "+.4f"])}

Pooled Spearman {CORR[CORR.scope=='pooled'].spearman.iloc[0]:.3f} — and **within** class it ranges from
{CORR[CORR.scope!='pooled'].spearman.min():+.3f} ({CORR.loc[CORR[CORR.scope!='pooled'].spearman.idxmin(),'scope']}) to {CORR[CORR.scope!='pooled'].spearman.max():+.3f} ({CORR.loc[CORR[CORR.scope!='pooled'].spearman.idxmax(),'scope']}).
The high pooled Pearson ({CORR[CORR.scope=='pooled'].pearson.iloc[0]:.3f}) is mostly a between-class effect (classes that are sparse are sparse in
both spaces); inside a class, a scenario with an unusual interaction signature is usually *not* the one with
an unusual path. A single combined radius would therefore be too tight for one space and too loose for the
other, scenario by scenario.

**(iii) Elasticity: which threshold actually controls the joint number.**
d(coverage)/d(log eps) at the recommended operating point, central difference on the 2-D grid
(the `eps_path` grid cell used is {float(ELAF.eps_path_cell.iloc[0]):.6f} m, the nearest grid value to {REC_PATH:.4f} m):

{mdtab(ELAF, ["cls", "a", "coverage", "d_cov_dlog_eps_int", "d_cov_dlog_eps_path", "binding"],
       ["class", "arm", "coverage", "dC/dlog eps_int", "dC/dlog eps_path", "binding threshold"],
       ["s", "s", ".4f", "+.4f", "+.4f", "s"])}

At the **current** operating point the split is even sharper: for tlkeep/Ours the path elasticity
({float(ELAC[(ELAC['subset']=='tlkeep')&(ELAC.arm==OURS)].d_cov_dlog_eps_path.iloc[0]):.4f}) is {float(ELAC[(ELAC['subset']=='tlkeep')&(ELAC.arm==OURS)].d_cov_dlog_eps_path.iloc[0])/float(ELAC[(ELAC['subset']=='tlkeep')&(ELAC.arm==OURS)].d_cov_dlog_eps_int.iloc[0]):.0f}× the interaction one ({float(ELAC[(ELAC['subset']=='tlkeep')&(ELAC.arm==OURS)].d_cov_dlog_eps_int.iloc[0]):.4f}), while for
keeptl/Ours it is the other way round ({float(ELAC[(ELAC['subset']=='keeptl')&(ELAC.arm==OURS)].d_cov_dlog_eps_int.iloc[0]):.4f} vs {float(ELAC[(ELAC['subset']=='keeptl')&(ELAC.arm==OURS)].d_cov_dlog_eps_path.iloc[0]):.4f}) and for cutinl/Ours the path
derivative is exactly {float(ELAC[(ELAC['subset']=='cutinl')&(ELAC.arm==OURS)].d_cov_dlog_eps_path.iloc[0]):.4f} (already at its ceiling). **Different classes are governed by different
thresholds**, so one shared knob would be moving the wrong one in at least one class. This is the same fact the
2-D heatmaps show as an L-shape (Fig 4): below the knee `eps_int` binds, left of it `eps_path` binds.

**(iv) What a single combined normalised distance would do.**
First, note the current rule *already is* one: covering real *i* means
max(d_int/m_int(class), d_path/m_path(class)) ≤ 1, an **L-infinity ball of radius 1** in units of each class's
own median NN. The `eps_mult` ∈ {{1, 1.5, 2}} rows are that same combined distance at radius t:

{mdtab(LINF.pivot_table(index=["subset", "arm"], columns="t", values="coverage_mean").reset_index()
       .assign(cls=lambda d: d['subset'].map(CLASS_ONELINE), a=lambda d: d.arm.map(TINY)),
       ["cls", "a", 1.0, 1.5, 2.0], ["class", "arm", "t = 1", "t = 1.5", "t = 2"],
       ["s", "s", ".4f", ".4f", ".4f"]) if False else
 LINF.pivot_table(index=["subset", "arm"], columns="t", values="coverage_mean").round(4).to_markdown()}

So "adopt one combined number" is not a change at all unless you change the *normaliser* (per class → pooled,
which is exactly the fixed-epsilon recommendation) or the *norm* (L∞ → L2). Replacing L∞ by L2 at radius 1
with the pooled normaliser ({REC_INT} z, {REC_PATH:.4f} m) gives, rigorously, coverage between the best grid rectangle
inscribed in the L2 ball and the L∞ cell itself:

{mdtab(L2P, ["cls", "a", "l2_lower", "l2_upper"], ["class", "arm", "L2 coverage ≥", "L2 coverage ≤"],
       ["s", "s", ".4f", ".4f"])}

and the winner is provably unchanged in only {int(L2SEP.winner_provably_unchanged.sum())} of 5 classes:

{mdtab(L2SP, ["cls", "w", "winner_l2_lower", "best_other_l2_upper", "v"],
       ["class", "L∞ winner", "its guaranteed L2 floor", "best rival's L2 ceiling", "ranking provably unchanged?"],
       ["s", "s", ".4f", ".4f", "s"])}

In other words the L2 variant cannot be shown to preserve the ranking at this grid resolution, and it buys
nothing interpretable. **Do not adopt it.**

**(v) Recommendation.** Keep `eps_int` and `eps_path` as two separate numbers, fix them **both** by reading one
quantile q off each space's own **pooled** real-to-real NN distribution
(`results/eps_sweep/eps_real_nn_distributions.csv`, rows `scope = pooled`). Today's rule is q = 0.50 *per class*;
the recommendation is q ≈ {Q_REC[0]:.2f} *pooled*, which is `eps_int` = {REC_INT} z and `eps_path` = {REC_PATH_ROUND} m. Report both,
state the quantile, and state the joint rule (a draw must satisfy both, on the same draw).
"""

SEC_F = f"""## F. Limitations — why the epsilon choice is less decisive than it looks


**F1. The KDEs are in-sample.** Every KDE arm is fitted on the same recorded scenarios the coverage is then
measured against; each draw already knows which real it is centred on. Coverage@1000 is therefore a
*reachability* statement about the sampler's spread around known points, not a generalisation statement,
and epsilon is the knob that decides how generous that statement is. No epsilon repairs this.


**F2. Path coverage allows exactly one centre per real scenario.** A draw covers real *i* in path only if
`kernel_center_uid == i`; a draw that happens to fly right along *i*'s path but is centred on *j* never
counts. That makes path coverage hard-capped by kernel-centre availability, independently of epsilon:

{mdtab(CEILP, ["a", "pooled_ceiling_per_seed"] + [f"ceil_{c}" for c in CLASSES],
       ["arm", "pooled ceiling (per seed)"] + [CLASS_ONELINE[c].split(" / ")[1] for c in CLASSES],
       ["s", ".4f"] + [".4f"] * 5)}

{SHORT[OURS]} can never exceed {float(CEIL[CEIL.arm==OURS].pooled_ceiling_per_seed.iloc[0]):.4f} pooled ({float(CEIL[CEIL.arm==OURS].ceil_cutinr.iloc[0]):.2f} on Cut-in (right), where only
{int(round(float(CEIL[CEIL.arm==OURS].ceil_cutinr.iloc[0]) * 20))} of 20 reals have a kernel centre at all). The flat right-hand tails in Fig 2 are this ceiling,
**not** saturation — the path curves never reach 1.0 for any epsilon.

**F3. The valid-pool prefix rule is not epsilon-neutral.** The `pool = valid` variant keeps the contiguous
Table-5-flagged prefix of a pool, so its 19 random orderings permute a shorter list than the `all` pool's;
only `coverage_natural_order` is strictly comparable between the two pools. The thesis uses `pool = all`,
and so does every number in this report.

**F4. Different draw counts per arm.** {SHORT[OURS]} and {SHORT[SVD]} and {SHORT[SAK]} each supply 1000 draws per seed, but
`svd_d5_kde_executed_E3` supplies only 43–100, so it has no coverage@1000 at all and is excluded from Fig 1–7.
Comparing arms at a common budget is a choice; at budget 10 or 100 the ranking is different (Fig 8).

**F5. Class denominators differ by space.** Interaction and joint are scored over the reals with all six
descriptors finite (n = {POOLED['ours_vs_svd_5class|joint']['n_real'][OURS]}), path over all reals (n = {POOLED['ours_vs_svd_5class|path']['n_real'][OURS]}). Joint coverage can therefore exceed path
coverage in a class where they differ (cutinr, 15 vs 20); that is a denominator artefact, not a bug.

**F6. Three of the five classes are small.** Cut-in (right) has {int(REACH[REACH['subset']=='cutinr'].n_real_joint.iloc[0])} rankable reals and {int(REACH[REACH['subset']=='cutinr'].n_real_joint.iloc[0])//5}
corner reals, so one scenario is worth {100/int(REACH[REACH['subset']=='cutinr'].n_real_joint.iloc[0]):.1f} coverage points and {100/3:.0f} corner points. The cutinr rank flip
at {float(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='cutinr')].eps_after.iloc[0]):.4f} z rests on a handful of scenarios.

**F7. The joint diagonal uses a pooled quantile map.** `eps_path` on the diagonal is the pooled-ECDF partner of
`eps_int`, the same map in every class. That is deliberate (it is the fixed-epsilon question) but it is not a
per-class quantile match; a per-class map would shift each class's diagonal slightly.

**F8. The grid truncates.** Path and joint coverage are read with `searchsorted` indices, so a draw whose DTW
exceeds 10 m, or whose self-centre interaction distance exceeds 6 z, never covers. That is exact at every
grid point but no coverage can be read beyond `eps_int` 6 z / `eps_path` 10 m, and the largest observed real
NN distances ({NN_INT.max():.2f} z, {NN_PATH.max():.2f} m) are close to those ends.

**F9. Epsilon is not validity.** Nothing in this sweep asks whether a covering draw is a *drivable* scenario.
A larger epsilon buys coverage from draws that may be off-road or in background collision; the validity
columns of the thesis tables are the counterweight and they are epsilon-independent.
"""

SRC = f"""## Sources and reproduction

Inputs (all read-only):

| file | what was read |
| --- | --- |
| `results/eps_sweep/eps_sweep_coverage.csv` | every coverage@b curve, 3 spaces × 10 arms × 2 pools × 2 real sets |
| `results/eps_sweep/eps_sweep_corner.csv` | the same restricted to the top-20 % NN-isolated reals |
| `results/eps_sweep/eps_sweep_joint_grid.csv` | the 2-D (eps_int × eps_path) surface, matched489 / all / budget 1000 |
| `results/eps_sweep/eps_real_nn_values.csv` | per-scenario real-to-real NN distances (both spaces) |
| `results/eps_sweep/eps_real_nn_distributions.csv` | their per-class and pooled summaries |
| `results/e6_real_reference.csv` | per-class descriptor SDs (z → seconds / metres / m/s only) |
| `results/e6_descriptors.parquet`, `results/e6_descriptors_sakura.parquet` | kernel-centre availability (Limitation 2) |
| `results/final/tables3/gen_metrics.csv` | the gate: the published coverage / corner cells |
| `results/table34_manifest.json` | the current per-class eps (hard-coded in the script's `CUR`) |

Outputs of this script (all new):

| file | what |
| --- | --- |
| `results/eps_sweep/EPS_SWEEP_REPORT.md` / `_zh.md` | this report |
| `results/eps_sweep/tables3_coverage_at_fixed_eps.csv` / `.md` | Section D |
| `results/eps_sweep/eps_rank_flips.csv` | every winner change on the grid (Section B) |
| `results/eps_sweep/eps_regimes.csv` | the starvation / discriminative / saturation boundaries (Section A) |
| `results/eps_sweep/eps_sweep_report_numbers.json` | every number quoted above, machine-readable |
| `figures/eps_sweep/fig1…fig8_*.png` | the eight figures, 200 dpi |
| `results/eps_sweep/98_stdout.txt` | the full console log of this run |

Reproduce:

```bash
cd /home/hcis-s19/Documents/ChengYu/exp_ours3_svd5
MPLCONFIGDIR=/tmp/ours3_svd5_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
  /home/hcis-s19/micromamba/envs/nps/bin/python -B scripts/98_eps_sweep_report.py
```

(the script writes its own console log to `results/eps_sweep/98_stdout.txt`; do not redirect stdout there,
the script would overwrite the redirect at the end)

The script asserts its own gate against `gen_metrics.csv` ({n_gate} cells, max |diff| {gmax:.1e}) and fails loudly
if the frozen sweep tables ever stop reproducing the published cells.

The eight figures named in this report (`fig1…fig8`) are produced by this script. The eight earlier PNGs in the
same directory (`eps_sweep_interaction.png` etc.) were produced by an unversioned scratch script during the
sweep run and are **not** reproducible; they are left untouched but should not be cited.
"""

REPORT = ("# Epsilon-ball sweep — how much does ε decide, and can one fixed ε replace the per-class rule?\n\n"
          f"*Generated by `scripts/98_eps_sweep_report.py` from the frozen sweep tables of "
          f"`scripts/97_eps_sweep.py`. Every number is read from a named file and column; nothing is estimated.*\n\n"
          + EXEC + "\n" + METHOD + "\n" + SEC_A + "\n" + SEC_B + "\n" + SEC_C + "\n" + SEC_D + "\n"
          + SEC_E + "\n" + SEC_F + "\n" + SRC)
(OUT / "EPS_SWEEP_REPORT.md").write_text(REPORT, encoding="utf-8")
say("wrote " + str(OUT / "EPS_SWEEP_REPORT.md") + f"  ({len(REPORT.splitlines())} lines)")

# ─────────────────────────────────────────────────────────────────────────────
# Chinese report — identical numbers, all read from the same variables
# ─────────────────────────────────────────────────────────────────────────────
def zhcls(c):
    return CLASS_ZH[c]


REGZ = REGP.copy()
REGZ["cls"] = REGZ['subset'].map(CLASS_ZH)
REGZ["sp"] = REGZ.space.map({"interaction": "互動 (6 維 z)", "path": "軌跡 (DTW, m)", "joint": "聯合 (互動 ∧ 軌跡)"})
REGZ["curin"] = np.where(REGZ.cur_in_band, "是", "**否**")
FBZ = FB.copy(); FBZ["cls"] = FBZ['subset'].map(CLASS_ZH)
ROBZ = ROBP.copy(); ROBZ["cls"] = ROBZ['subset'].map(CLASS_ZH)
ROBZ["verdict"] = np.where(ROBZ.winner_constant_over_band, "**對 ε 穩健**", "對 ε 敏感")
ROBZ["common"] = np.where(ROBZ.winner_constant_over_common_band,
                          "**穩定** (" + ROBZ.winners_in_common_band + ")",
                          "翻轉 (" + ROBZ.winners_in_common_band + ")")
CORRZ = CORR.copy(); CORRZ["lab"] = CORRZ.scope.map(lambda x: CLASS_ZH.get(x, "**合併（五類）**"))
ELAFZ = ELAF.copy(); ELAFZ["cls"] = ELAFZ['subset'].map(CLASS_ZH)
ELAFZ["bind"] = ELAFZ.binding.map({"interaction": "互動 ε", "path": "軌跡 ε", "tie": "相同"})
REACHZ = REACHP.copy(); REACHZ["cls"] = REACHZ['subset'].map(CLASS_ZH)
L2PZ = L2P.copy(); L2PZ["cls"] = L2PZ['subset'].map(CLASS_ZH)
L2SZ = L2SP.copy(); L2SZ["cls"] = L2SZ['subset'].map(CLASS_ZH)
L2SZ["v"] = np.where(L2SZ.winner_provably_unchanged, "**是**", "無法判定")
CEILZ = CEILP.copy()
PHYSZ = PHYSD.copy(); PHYSZ["cls"] = [CLASS_ZH[c] for c in CLASSES]
ANCHZ = ANCHS.copy()
DZ = D.copy(); DZ["cls"] = DZ['subset'].map(CLASS_ZH)
DZ["m1"] = np.where(DZ.bold_coverage_moves, "**變**", "")
DZ["m2"] = np.where(DZ.bold_corner_moves, "**變**", "")

ZH = f"""# ε 球掃描 —— ε 到底決定了多少，一個固定 ε 能不能取代現在的逐類規則？

*本檔由 `scripts/98_eps_sweep_report.py` 從 `scripts/97_eps_sweep.py` 凍結的掃描表產生。每個數字都可追到檔名與欄位，沒有任何估計值。內容與 `EPS_SWEEP_REPORT.md` 完全對應，數字一致。*

## 摘要 —— 直接回答三個問題

1. **ε 怎麼影響 coverage？** 每一條曲線都有同樣三個區間：飢餓 → 可分辨 → 飽和。五個類別「同時」可分辨的範圍只有
   互動 `eps_int` ∈ [{BAND['interaction'][0]:.4f}, {BAND['interaction'][1]:.4f}] z、軌跡 `eps_path` ∈ [{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m、
   聯合 `eps_int` ∈ [{BAND['joint'][0]:.4f}, {BAND['joint'][1]:.4f}] z；超出這個窗口，ε 的選擇就不再帶資訊（圖 1–3、5）。
2. **軌跡軸才是卡住的那一個**：它的全類可分辨窗口只有 [{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m（倍率 {BAND['path'][1]/BAND['path'][0]:.2f}），
   互動軸則有 [{BAND['interaction'][0]:.4f}, {BAND['interaction'][1]:.4f}] z（倍率 {BAND['interaction'][1]/BAND['interaction'][0]:.2f}）。目前五個逐類 `eps_path` 有兩個
   （{'、'.join(CLASS_ZH[c] for c in REG[(REG.space=='path')&(~REG.cur_in_band)]['subset'])}）已經落在窗口之外。
3. **建議的固定值：`eps_int` = {REC_INT} z、`eps_path` = {REC_PATH_ROUND} m**（精確值 {REC_PATH:.6f} m，是 {REC_INT} z 在合併分位
   q = {Q_REC[0]:.4f} 上的對應值）。它是唯一一個在三個空間、五個類別的可分辨帶裡都成立的「整數」格點。
4. 它落在哪裡：合併後真實場景兩兩最近鄰距離分布的第 **{100*Q_REC[0]:.2f}** 百分位（互動，n = {len(NN_INT)}，中位數 {POOL_MED_INT:.4f} z）
   與第 **{100*Q_REC[1]:.2f}** 百分位（軌跡，n = {len(NN_PATH)}，中位數 {POOL_MED_PATH:.4f} m）。意義跟現在一樣是「兩筆真實場景通常有多近」，
   只是把「逐類」換成「合併」。
5. 採用的代價：**{n_mv_cov} / {len(D)} 個 coverage 儲存格與 {n_mv_cor} / {len(D)} 個 corner 儲存格的粗體會換人**
   （coverage：{'、'.join(CLASS_ZH[c] for c in GRP_MV_COV)}；corner：{'、'.join(CLASS_ZH[c] for c in GRP_MV_COR)}）。三張論文表其餘部分不變（第 D 節）。
6. 這兩個變動都對 ours 有利，**所以要在論文裡明講**。若要把變動降到最低，另一個選項是
   `eps_int` = {ALT_INT:.4f} z / `eps_path` = {ALT_PATH:.4f} m（coverage 0 個群組變動、corner {int(ANCHS[ANCHS.anchor.str.startswith('SECOND')].n_cor.iloc[0])} 個），
   代價是它已經離開軌跡軸的可分辨帶。
7. **哪些結論不受 ε 影響？** 在全類聯合窗口 [{BAND['joint'][0]:.4f}, {BAND['joint'][1]:.4f}] z 內，五個類別只有
   {int(ROB[ROB.space=='joint'].winner_constant_over_common_band.sum())} 個的勝方從頭到尾不變：{'、'.join(CLASS_ZH[r['subset']] for _, r in ROB[(ROB.space=='joint')&ROB.winner_constant_over_common_band].iterrows())}，兩者都是 Ours+KDE。
   {CLASS_ZH['keeptl']}、{CLASS_ZH['keeptl_sw']}、{CLASS_ZH['cutinr']} 在窗口內都會翻轉，這三句話一定要把 ε 一起寫出來。
8. **ε 要不要分成軌跡與互動？要，維持兩個數字。** 兩者單位不可通約（無因次 z vs 公尺），
   一個在互動空間孤立的場景通常不是在軌跡空間孤立的那一個（合併 Spearman {CORR[CORR.scope=='pooled'].spearman.iloc[0]:.4f}，
   逐類 {CORR[CORR.scope!='pooled'].spearman.min():+.4f}…{CORR[CORR.scope!='pooled'].spearman.max():+.4f}），而且兩個門檻在不同類別各自主導（第 E-iii 節）：
   在建議操作點上，{BINDSENT_ZH}。
9. 可以固定的是**一個分位數 q**，不是一個純量：在各自空間的合併最近鄰 ECDF 上讀同一個 q。現在的規則是逐類 q = 0.50，
   建議改成合併 q ≈ {Q_REC[0]:.2f}。
10. 要壓成單一純量，一定要有明確權重；而現在的逐類規則**本來就是**一個（以各類中位數最近鄰為單位的 L∞ 球，半徑 1）。
    把 L∞ 換成 L2 在目前的格點解析度下，五類中有 {5-int(L2SEP.winner_provably_unchanged.sum())} 類無法證明排名不變（第 E-iv 節），沒有任何好處。

## 讀法與來源

- 所有 coverage 數字來自 `results/eps_sweep/eps_sweep_coverage.csv`、`eps_sweep_corner.csv`、`eps_sweep_joint_grid.csv`
  （皆由 `scripts/97_eps_sweep.py` 產生）。本腳本不重算任何 coverage。
- 報告設定與論文表完全一致：space = joint、pool = all、real_set = matched489、budget = 1000、3 seeds × 20 orderings 平均。
- 網格：{len(GRID_INT)} 個共用 `eps_int` 格點，從 {GRID_INT.min()} 到 {GRID_INT.max()} z；{len(GRID_PATH)} 個共用 `eps_path` 格點，
  從 {GRID_PATH.min()} 到 {GRID_PATH.max()} m。以下所有跨類別彙總都只用這些共用格點。
- 方法（arm）只有三個真的有 coverage@1000：{'、'.join(SHORT[a] for a in ARMS)}。
  `svd_d5_kde_executed_E3` 的樣本池只有 43–100 筆，沒有 budget 1000，只出現在圖 8。
- **驗證閘**：以 `eps_mult == 1.0` 的列重讀，可完全重現 `results/final/tables3/gen_metrics.csv` 的每一個
  coverage / corner 儲存格：{n_gate} 格、0 缺漏、最大誤差 {gmax:.2e}（容差 1e-12）。
- **CSV 的兩個陷阱（本腳本已避開）**：(H1) joint 列在每個類別網格的全部 {len(leak)} 個 `eps_int` 上 `eps_mult` 都是 NaN，
  其中 {N_LEAK} 個只有單一類別；若用 `eps_mult` 過濾再跨類別彙總會悄悄變成單類數字，因此本腳本一律用
  {len(GRID_INT)} 個共用格點過濾。(H2)「現在的規則」是 `eps_mult == 1.0` 的列，**不是**同一個 `eps_int` 的對角線列
  （對角線配的是另一個 `eps_path`）。
- 粗體相同值的判定沿用 scripts/90 的容差（|Δ| ≤ 1e-12），所以 keeptl 的 corner 平手被保留。

## A. ε 的效果 —— 三個區間

定義（先訂再看資料）：**飢餓** = 最佳方法 coverage@1000 < {STARV:.2f}；**飽和** = 最佳方法 coverage@1000 > {SAT:.2f}；
**可分辨帶** = 兩者之間的格點。

{mdtab(REGZ, ["sp", "cls", "n_real", "starv_last", "band", "sat_first", "best_max", "cur_eps", "curin"],
       ["空間", "類別", "n_real", "飢餓至", "可分辨帶", "飽和起點", "最佳方法上限", "現行 ε", "現行 ε 在帶內？"],
       ["s", "s", "d", ".4f", "s", ".4f", ".4f", ".4f", "s"])}

五類交集：**互動 [{BAND['interaction'][0]:.4f}, {BAND['interaction'][1]:.4f}] z、軌跡 [{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m、
聯合 [{BAND['joint'][0]:.4f}, {BAND['joint'][1]:.4f}] z**。

三個推論：(1) 固定 ε 真正被卡住的是軌跡軸，兩個 cut-in 類別在 0.5 m 就飽和，所以現行的
cutinl（{CUR['cutinl'][1]:.4f} m）與 cutinr（{CUR['cutinr'][1]:.4f} m）已經在交集之外；
(2) 互動空間很早就飽和（`eps_int` = 1.1631 z 時五類有四類 > {SAT:.2f}），大 ε 下的互動排名沒有意義；
(3) 聯合空間最慢飽和（tlkeep 的最佳方法在 6 z 也只有 {REG[(REG.space=='joint')&(REG['subset']=='tlkeep')].best_max.iloc[0]:.4f}），所以論文報聯合空間是對的。

合併跨類曲線（以真實數加權，**分母對齊**；把 5 類的方法跟只有 4 類的 SAKURA 混在一起會灌水）：
Ours+KDE vs SVD_d5+KDE，五類（互動/聯合 n = {POOLED['ours_vs_svd_5class|joint']['n_real'][OURS]}，軌跡 n = {POOLED['ours_vs_svd_5class|path']['n_real'][OURS]}）：

| 空間 | ε | Ours+KDE | SVD_d5+KDE | Ours 領先到 |
| --- | ---: | ---: | ---: | ---: |
{POOLED_MD}

三方法在四個共同類別上的最大－最小差距，在聯合對角線的 `eps_int` = {POOLED['three_arms_4class|joint']['spread_peak_eps']:.4f} 達到最大
（差距 {POOLED['three_arms_4class|joint']['spread_peak']:.4f}）；Ours-vs-SVD 五類則在 `eps_int` = {POOLED['ours_vs_svd_5class|joint']['spread_peak_eps']:.4f} 最大（差距 {POOLED['ours_vs_svd_5class|joint']['spread_peak']:.4f}）。
**建議值正好落在方法最容易被分開的位置。**

圖：`{FIGREL[0]}`、`{FIGREL[1]}`、`{FIGREL[2]}`、`{FIGREL[3]}`。

## B. 排名穩定性 —— 哪些 coverage 結論禁得起換 ε

全域 31 點網格上共有 {len(FLIPS)} 次勝方變動，其中 **{len(FB)} 次落在該類別自己的可分辨帶內**（不是飢餓或飽和造成的假象）。

{mdtab(FBZ, ["sp", "cls", "eps_before", "wb", "cov_before_winner", "eps_after", "wa", "cov_after_winner"],
       ["空間", "類別", "翻轉前 ε", "勝方", "coverage", "翻轉後 ε", "勝方", "coverage"],
       ["s", "s", ".4f", "s", ".4f", ".4f", "s", ".4f"])}

在各類**現行** ε 上讀到的勝方，以及這個判斷能撐多遠：

{mdtab(ROBZ, ["space", "cls", "cur_eps", "w", "cov", "runner_up", "margin", "stable", "verdict", "common"],
       ["空間", "類別", "現行 ε", "勝方", "勝方 coverage", "第二名", "差距", "勝方不變區間", "在自己帶內不變？", "在全類帶內？"],
       ["s", "s", ".4f", "s", ".4f", ".4f", ".4f", "s", "s", "s"])}

最後一欄才是固定 ε 真正要看的：它問「在任何一個合理的固定 ε 上，勝方會不會一樣」。

**對論文文字的意義**

- *可以不附 ε 直接寫的*（全類帶內勝方不變，且與論文現在印的一致）：
  {'；'.join(f"{CLASS_ZH[r['subset']]} / {r.space} → {r.winners_in_common_band}" for _, r in SAFE.iterrows())}。
  在論文報的**聯合**空間裡只有 {int(len(SAFE[SAFE.space=='joint']))} 類：{'、'.join(CLASS_ZH[r['subset']] for _, r in SAFE[SAFE.space=='joint'].iterrows())}，兩者都是 **Ours+KDE**。
- *對 ε 敏感、必須附上 ε 的*（聯合空間）：{CLASS_ZH['keeptl']} 在 {FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='keeptl')].eps_before.min():.4f}–{FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='keeptl')].eps_after.max():.4f} z 之間換手
  {int(len(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='keeptl')]))} 次後定於 SVD；{CLASS_ZH['keeptl_sw']} 在 {float(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='keeptl_sw')].eps_after.iloc[0]):.4f} z 由 Ours 轉 SVD；
  {CLASS_ZH['cutinr']} 在 {float(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='cutinr')].eps_after.iloc[0]):.4f} z 由 Ours 轉 SVD。現行逐類門檻剛好分別落在這三個翻轉點的兩側，
  這正是論文表看起來勝負交錯的原因。
- *第三種情況：現行 ε 落在全類帶之外，所以「穩定的勝方」跟論文印的不是同一個*：
  {'；'.join(f"{CLASS_ZH[r['subset']]} / {r.space}：ε = {r.cur_eps:.4f} 印的是 {TINY[r.winner]}，但在 [{r.band_common_lo:.4f}, {r.band_common_hi:.4f}] 全區都是 {r.winners_in_common_band}" for _, r in DIFFW.iterrows()) if len(DIFFW) else '無'}。

- 「勝方不變區間」欄說明還有多少空間：現行 {CLASS_ZH['keeptl']} 的聯合判斷只撐得住 ×{float(ROB[(ROB.space=='joint')&(ROB['subset']=='keeptl')].stability_radius.iloc[0]):.2f} 的 ε 變動，
  {CLASS_ZH['cutinl']} 則有 ×{float(ROB[(ROB.space=='joint')&(ROB['subset']=='cutinl')].stability_radius.iloc[0]):.2f}。**一個用稍微不同規則重算 ε 的讀者，可以推翻五個類別判斷中的三個，但推不翻另外兩個。**

圖：`{FIGREL[4]}`。

## C. 固定 ε 的建議

### 第一選擇 —— `eps_int` = {REC_INT} z、`eps_path` = {REC_PATH_ROUND} m（精確 {REC_PATH:.6f} m）

**(i) 在每一類的可分辨帶內。** 通過三個空間全部條件的錨點只有
{'、'.join(f'{v:.4f}' for v in CAND[CAND.in_all_bands].eps_int)} z 三個，其中只有 {REC_INT} 是整數，而且排名穩定半徑最大
（{float(CAND[np.isclose(CAND.eps_int, REC_INT)].min_stability_radius.iloc[0]):.3f}，另外兩個是 {float(CAND[np.isclose(CAND.eps_int, 0.642490)].min_stability_radius.iloc[0]):.3f} 與 {float(CAND[np.isclose(CAND.eps_int, 0.676433)].min_stability_radius.iloc[0]):.3f}）。

**(ii) 物理意義。** `eps_path` = {REC_PATH_ROUND} m 就是 DTW 的公尺：兩條目標軌跡平均差約三分之一公尺才算「同一條」，
大約 3.5 m 車道寬的十分之一，遠小於一次變換車道造成的 ≈3 m 側向位移。`eps_int` = {REC_INT} z 是 6 維 z 空間的 L2 距離；
若差距全部集中在單一描述子，等於 0.75 × 該類真實標準差：

{mdtab(PHYSZ, ["cls"] + DESC, ["類別", "PET [s]", "d_min [m]", "α [度]", "c_x [m]", "c_y [m]", "u_c [m/s]"],
       ["s"] + [".2f"] * 6)}

若差距平均分到六個描述子，每個只動 0.75/√6 = {REC_INT/np.sqrt(6):.3f} 個標準差
（例如 tlkeep 的 PET {PHYS['tlkeep']['pet']['isotropic']:.2f} s、d_min {PHYS['tlkeep']['d_min']['isotropic']:.2f} m；{CLASS_ZH['cutinr']} 的 d_min {PHYS['cutinr']['d_min']['isotropic']:.2f} m、PET {PHYS['cutinr']['pet']['isotropic']:.2f} s）。**單方向很鬆、等向很緊**，論文要補一句，
否則讀者看到「0.75 z」不會知道它在 tlkeep 上允許 {PHYS['tlkeep']['pet']['one_axis']:.1f} s 的 PET 差距。

**(iii) ±50 % 的排名穩健度。** `eps_int` ∈ [{0.5*REC_INT:.4f}, {1.5*REC_INT:.4f}] z 內勝方不變的類別數：
聯合 {int(ROB50[(ROB50.space=='joint')].stable.sum())}/5、互動 {int(ROB50[(ROB50.space=='interaction')].stable.sum())}/5、軌跡 {int(ROB50[(ROB50.space=='path')].stable.sum())}/5。
網格上沒有任何錨點在聯合空間能超過 3/5——±50 % 這個測試本來就比資料能支撐的還嚴，這也正是 B 節的誠實結論。

**(iv) 在合併最近鄰分布上的位置。** {REC_INT} z 是互動合併分布的第 {100*Q_REC[0]:.2f} 百分位（n = {len(NN_INT)}，中位數 {POOL_MED_INT:.4f} z）；
{REC_PATH:.4f} m 是軌跡合併分布的第 {100*Q_REC[1]:.2f} 百分位（n = {len(NN_PATH)}，中位數 {POOL_MED_PATH:.4f} m）。
嚴格取合併中位數（q = 0.50）會得到 {POOL_MED_INT:.4f} z / {POOL_MED_PATH:.4f} m，粗體結果與 {REC_INT} z 相同，所以 q 取 0.50 或 {Q_REC[0]:.2f} 不是關鍵。
被取代掉的逐類離散程度：互動 {min(v[0] for v in CUR.values()):.4f}…{max(v[0] for v in CUR.values()):.4f} z（×{max(v[0] for v in CUR.values())/min(v[0] for v in CUR.values()):.2f}）、
軌跡 {min(v[1] for v in CUR.values()):.4f}…{max(v[1] for v in CUR.values()):.4f} m（×{max(v[1] for v in CUR.values())/min(v[1] for v in CUR.values()):.2f}）——**互動軸幾乎不變，軌跡軸變很多**，其中 tlkeep 動最大
（`eps_path` 從 {CUR['tlkeep'][1]:.4f} 變成 {REC_PATH:.4f} m，+{100*(REC_PATH/CUR['tlkeep'][1]-1):.0f} %）。

**(v) 有多少真實場景變成「構不到」或「隨便就中」。**

{mdtab(REACHZ, ["cls", "n_real_joint", "iso_int", "iso_path", "dense_int", "dense_path", "best_cov",
                "n_missed_by_best", "all_cov", "n_hit_by_all"],
       ["類別", "n_real", "互動孤立 (NN > ε)", "軌跡孤立 (NN > ε)", "互動稠密 (NN ≤ ε/2)", "軌跡稠密",
        "最佳方法 cov@1000", "最佳方法漏掉", "三方法都覆蓋的比例", "三方法都覆蓋"],
       ["s", "d", "d", "d", "d", "d", ".4f", "d", ".4f", "d"])}

「孤立」= 同類別中沒有**其他真實場景**落在固定 ε 內，生成器不能靠「落在鄰居附近」覆蓋它，必須真的重現它。
在 {REC_INT} z / {REC_PATH:.3f} m 下，互動是 {int(REACH.iso_int.sum())}/{int(REACH.n_real_int.sum())}、軌跡是 {int(REACH.iso_path.sum())}/{int(REACH.n_real_path.sum())}。
「三個方法全都覆蓋」的真實場景總共只有 {int(REACH.n_hit_by_all.sum())} 筆，所以固定 ε 並沒有白送 coverage。

### 第二選擇 —— `eps_int` = {ALT_INT:.4f} z、`eps_path` = {ALT_PATH:.4f} m（變動最小）

{mdtab(ANCHZ, ["anchor", "eps_int", "eps_path", "q", "n_cov", "n_cor"],
       ["候選", "eps_int [z]", "eps_path [m]", "合併 q（互動）", "coverage 粗體變動群組數", "corner 粗體變動群組數"],
       ["s", ".4f", ".4f", ".4f", "d", "d"])}

它讓已發表的 coverage 欄粗體完全不動。代價：`eps_path` {ALT_PATH:.4f} m 已在全類軌跡帶
[{BAND['path'][0]:.4f}, {BAND['path'][1]:.4f}] m 之外——它落在兩個 cut-in 類別可分辨帶上緣（{BAND['path'][1]:.4f} m）與飽和起點（0.5 m）之間，
軌跡欄在那裡幾乎不再區分方法；
而且它在合併最近鄰分布的第 {100*Q_ALT[0]:.0f} 百分位，比較難用「兩筆真實場景通常有多近」來解釋。
第三個更寬的選項 `eps_int` = {ALT2_INT} z / `eps_path` = {ALT2_PATH:.4f} m 落在聯合帶的幾何中心（{BAND_GEOM_MID:.4f} z），
但會把 {CLASS_ZH['keeptl_sw']} 的 coverage 粗體移給 SVD。

### 這個選擇偏袒誰？

誠實地說：**第一選擇在兩個變動的儲存格上都偏向 ours**。{CLASS_ZH['cutinr']} 的 coverage 由
SVD {float(D[(D['subset']=='cutinr')&(D.arm==SVD)].coverage_current.iloc[0]):.3f} → {float(D[(D['subset']=='cutinr')&(D.arm==SVD)].coverage_fixed.iloc[0]):.3f}、ours {float(D[(D['subset']=='cutinr')&(D.arm==OURS)].coverage_current.iloc[0]):.3f} → {float(D[(D['subset']=='cutinr')&(D.arm==OURS)].coverage_fixed.iloc[0]):.3f}；
{CLASS_ZH['keeptl']} 的 corner 平手被打破，倒向 ours（{float(D[(D['subset']=='keeptl')&(D.arm==SVD)].corner_fixed.iloc[0]):.3f} vs {float(D[(D['subset']=='keeptl')&(D.arm==OURS)].corner_fixed.iloc[0]):.3f}）。
機制不是偶然——ε 越小越獎勵「直接落在那筆真實場景上」，這正是 conflict-anchored 取樣在做的事；ε 越大越獎勵把機率質量攤開，
這是 SVD+KDE 在做的事——但方向性是真的。論文該主動寫一句：「固定 ε 是依可分辨帶準則選的，不是依排名選的；
它動了兩格，兩格都往 ours 走。」不想辯護這一點，就改用第二選擇，代價是軌跡軸變弱。

圖：`{FIGREL[5]}`。

## D. 用固定 ε 重算論文表

完整表：**`results/eps_sweep/tables3_coverage_at_fixed_eps.md`** 與 `.csv`。

{mdtab(DZ, ["cls", "method", "coverage_current", "coverage_fixed", "coverage_delta", "m1",
            "corner_current", "corner_fixed", "corner_delta", "m2"],
       ["場景群組", "方法", "現行 Cov@1000", "固定 ε Cov@1000", "Δ", "粗體變動",
        "現行 Corner", "固定 ε Corner", "Δ", "粗體變動"],
       ["s", "s", ".3f", ".3f", "+.3f", "s", ".3f", ".3f", "+.3f", "s"])}

- {n_mv_cov} / {len(D)} 個 coverage 儲存格、{n_mv_cor} / {len(D)} 個 corner 儲存格的粗體狀態改變。
- RareCase 表（Special 39_180）沒有 coverage 欄（只有一筆真實場景，改報還原比例），所以不在此表內。
- 就算排名不變，**絕對數值變化很大**：tlkeep 的 Ours+KDE 由 {float(D[(D['subset']=='tlkeep')&(D.arm==OURS)].coverage_current.iloc[0]):.3f} 變成
  {float(D[(D['subset']=='tlkeep')&(D.arm==OURS)].coverage_fixed.iloc[0]):.3f}，因為它現行的 `eps_path`（{CUR['tlkeep'][1]:.4f} m）是五類中最緊的。採用固定 ε 就等於論文裡每一個
  coverage 數字都要換，圖說也要跟著改。

圖：`{FIGREL[3]}`、`{FIGREL[6]}`、`{FIGREL[7]}`。

## E. ε 需要分成軌跡與互動嗎？

**建議：需要 —— 維持兩個數字，用一個分位數把它們綁在一起，而不是用一個純量。**

**(i) 單位不可通約。** `eps_int` 是 6 維 z 空間的無因次 L2 距離，`eps_path` 是公尺的 DTW。要壓成一個純量，
就必須寫出權重 w，讓 `d = max(d_int/w_int, d_path/w_path)`——第二個數字沒有消失，只是藏進 w 裡。
資料本身並沒有提供「PET 的秒」與「DTW 的公尺」之間的匯率。

**(ii) 在一個空間孤立，不代表在另一個空間也孤立。**

{mdtab(CORRZ, ["lab", "n", "spearman", "pearson"], ["範圍", "n", "Spearman ρ", "Pearson r"],
       ["s", "d", "+.4f", "+.4f"])}

合併 Spearman 只有 {CORR[CORR.scope=='pooled'].spearman.iloc[0]:.4f}；**類別內**更是從 {CORR[CORR.scope!='pooled'].spearman.min():+.4f}（{CLASS_ZH[CORR.loc[CORR[CORR.scope!='pooled'].spearman.idxmin(),'scope']]}）
到 {CORR[CORR.scope!='pooled'].spearman.max():+.4f}（{CLASS_ZH[CORR.loc[CORR[CORR.scope!='pooled'].spearman.idxmax(),'scope']]}）。合併的 Pearson 偏高（{CORR[CORR.scope=='pooled'].pearson.iloc[0]:.4f}）主要是類別之間的效果（稀疏的類別兩邊都稀疏）；
在類別內，互動特徵奇特的場景通常不是軌跡奇特的那一筆。

**(iii) 彈性分析：到底是哪個門檻在控制聯合數字。**
在建議操作點上的 d(coverage)/d(log ε)（2 維網格中央差分；實際用到的 `eps_path` 格點是 {float(ELAF.eps_path_cell.iloc[0]):.6f} m）：

{mdtab(ELAFZ, ["cls", "a", "coverage", "d_cov_dlog_eps_int", "d_cov_dlog_eps_path", "bind"],
       ["類別", "方法", "coverage", "dC/dlog eps_int", "dC/dlog eps_path", "主導門檻"],
       ["s", "s", ".4f", "+.4f", "+.4f", "s"])}

在**現行**操作點上差異更誇張：tlkeep/Ours 的軌跡彈性 {float(ELAC[(ELAC['subset']=='tlkeep')&(ELAC.arm==OURS)].d_cov_dlog_eps_path.iloc[0]):.4f} 是互動彈性
{float(ELAC[(ELAC['subset']=='tlkeep')&(ELAC.arm==OURS)].d_cov_dlog_eps_int.iloc[0]):.4f} 的 {float(ELAC[(ELAC['subset']=='tlkeep')&(ELAC.arm==OURS)].d_cov_dlog_eps_path.iloc[0])/float(ELAC[(ELAC['subset']=='tlkeep')&(ELAC.arm==OURS)].d_cov_dlog_eps_int.iloc[0]):.0f} 倍；keeptl/Ours 則相反（{float(ELAC[(ELAC['subset']=='keeptl')&(ELAC.arm==OURS)].d_cov_dlog_eps_int.iloc[0]):.4f} vs {float(ELAC[(ELAC['subset']=='keeptl')&(ELAC.arm==OURS)].d_cov_dlog_eps_path.iloc[0]):.4f}）；
cutinl/Ours 的軌跡導數正好是 {float(ELAC[(ELAC['subset']=='cutinl')&(ELAC.arm==OURS)].d_cov_dlog_eps_path.iloc[0]):.4f}（已到天花板）。**不同類別由不同門檻主導**，所以共用一個旋鈕至少在一個類別上會轉錯。
這跟 2 維熱圖的 L 形（圖 4）是同一件事：拐點以下 `eps_int` 卡住，拐點左邊 `eps_path` 卡住。

**(iv) 單一合併正規化距離會怎樣。**
先注意：現行規則**本來就是**一個——覆蓋真實場景 i 等於 max(d_int/m_int(類), d_path/m_path(類)) ≤ 1，
也就是以各類中位數最近鄰為單位、半徑 1 的 **L∞ 球**。`eps_mult` ∈ {{1, 1.5, 2}} 就是同一個距離在半徑 t 的值：

{LINF.pivot_table(index=["subset", "arm"], columns="t", values="coverage_mean").round(4).to_markdown()}

所以「改用一個合併數字」本身不是改變，除非換**正規化基準**（逐類 → 合併，也就是本報告的建議）或換**範數**（L∞ → L2）。
以合併基準（{REC_INT} z、{REC_PATH:.4f} m）把 L∞ 換成半徑 1 的 L2，嚴格可得的上下界（下界 = 內接於 L2 球的最佳網格矩形，
上界 = L∞ 格點本身）：

{mdtab(L2PZ, ["cls", "a", "l2_lower", "l2_upper"], ["類別", "方法", "L2 coverage ≥", "L2 coverage ≤"],
       ["s", "s", ".4f", ".4f"])}

五類中只有 {int(L2SEP.winner_provably_unchanged.sum())} 類可以「證明」排名不變：

{mdtab(L2SZ, ["cls", "w", "winner_l2_lower", "best_other_l2_upper", "v"],
       ["類別", "L∞ 勝方", "它保證的 L2 下界", "最強對手的 L2 上界", "排名可證明不變？"],
       ["s", "s", ".4f", ".4f", "s"])}

也就是說 L2 版本在這個網格解析度下無法證明保留排名，而且也沒有更好解釋。**不建議採用。**

**(v) 建議。** `eps_int` 與 `eps_path` 維持兩個數字，兩個都用**同一個分位數 q** 從各自空間的**合併**真實最近鄰分布讀出
（`results/eps_sweep/eps_real_nn_distributions.csv` 的 `scope = pooled` 列）。現行是逐類 q = 0.50；建議改成合併
q ≈ {Q_REC[0]:.2f}，即 `eps_int` = {REC_INT} z、`eps_path` = {REC_PATH_ROUND} m。論文要同時報這兩個數字、報 q，並說明聯合規則
（同一筆 draw 必須同時滿足兩者）。

## F. 誠實的限制 —— 為什麼 ε 的選擇沒有看起來那麼決定性


**F1. 所有 KDE 都是 in-sample。** 每個 KDE 方法都是用同一批真實場景擬合的，每一筆 draw 都知道自己以哪一筆真實場景為中心。
coverage@1000 因此是「取樣器在已知點附近散得多開」的**可達性**陳述，不是泛化陳述；ε 只是決定這個陳述有多寬鬆。換任何 ε 都補不了這一點。

**F2. 軌跡 coverage 每筆真實場景只允許一個中心。** draw 只有在 `kernel_center_uid == i` 時才算覆蓋 i；就算它剛好沿著 i 的路徑走，
只要中心是 j 就不算。因此軌跡 coverage 有一個與 ε 無關的硬天花板：

{mdtab(CEILZ, ["a", "pooled_ceiling_per_seed"] + [f"ceil_{c}" for c in CLASSES],
       ["方法", "合併天花板（單 seed）"] + [CLASS_ZH[c] for c in CLASSES],
       ["s", ".4f"] + [".4f"] * 5)}

{SHORT[OURS]} 合併永遠不會超過 {float(CEIL[CEIL.arm==OURS].pooled_ceiling_per_seed.iloc[0]):.4f}（{CLASS_ZH['cutinr']} 只有 {float(CEIL[CEIL.arm==OURS].ceil_cutinr.iloc[0]):.2f}，
20 筆真實場景中只有 {int(round(float(CEIL[CEIL.arm==OURS].ceil_cutinr.iloc[0]) * 20))} 筆有中心）。圖 2 右邊的平台是這個天花板，**不是飽和**。

**F3. valid 池的前綴規則對 ε 不中立。** `pool = valid` 只保留 Table-5 通過旗標的連續前綴，它的 19 個隨機順序打亂的是較短的清單，
因此只有 `coverage_natural_order` 在兩個池之間嚴格可比。論文與本報告全部用 `pool = all`。

**F4. 各方法的 draw 數不同。** 三個主要方法各有每 seed 1000 筆，但 `svd_d5_kde_executed_E3` 只有 43–100 筆，
完全沒有 coverage@1000，因此不出現在圖 1–7。在同一預算下比較是一個選擇；budget 10 或 100 時排名不同（圖 8）。

**F5. 分母隨空間不同。** 互動與聯合只算六個描述子都有限的真實場景（n = {POOLED['ours_vs_svd_5class|joint']['n_real'][OURS]}），軌跡算全部（n = {POOLED['ours_vs_svd_5class|path']['n_real'][OURS]}）。
所以在 cutinr（15 vs 20）聯合 coverage 有可能大於軌跡 coverage，那是分母造成的，不是錯誤。

**F6. 五類中有三類很小。** {CLASS_ZH['cutinr']} 只有 {int(REACH[REACH['subset']=='cutinr'].n_real_joint.iloc[0])} 筆可排名的真實場景、{int(REACH[REACH['subset']=='cutinr'].n_real_joint.iloc[0])//5} 筆 corner，
一筆就值 {100/int(REACH[REACH['subset']=='cutinr'].n_real_joint.iloc[0]):.1f} 個 coverage 百分點、{100/3:.0f} 個 corner 百分點。cutinr 在 {float(FLIPS[(FLIPS.space=='joint')&(FLIPS['subset']=='cutinr')].eps_after.iloc[0]):.4f} z 的翻轉只靠幾筆場景。

**F7. 聯合對角線用的是合併分位映射**，每個類別用同一條映射。這是刻意的（因為問題就是固定 ε），但它不是逐類分位匹配。

**F8. 網格有截斷。** 軌跡與聯合用 `searchsorted` 索引，DTW 超過 10 m 或自身中心互動距離超過 6 z 的 draw 一律不算覆蓋。
在每個格點上都精確，但 ε 不能讀到 `eps_int` 6 z / `eps_path` 10 m 以外，而最大的真實最近鄰距離
（{NN_INT.max():.2f} z、{NN_PATH.max():.2f} m）已經接近這兩端。

**F9. ε 不等於有效性。** 本掃描完全沒有問「覆蓋到的那筆 draw 是不是可行駛的場景」。ε 放大會從可能出地圖或撞背景車的 draw 買到 coverage；
論文表的有效性欄才是制衡，而那些欄與 ε 無關。

## 來源與重現

輸入（全部唯讀）：`results/eps_sweep/eps_sweep_coverage.csv`、`eps_sweep_corner.csv`、`eps_sweep_joint_grid.csv`、
`eps_real_nn_values.csv`、`eps_real_nn_distributions.csv`、`results/e6_real_reference.csv`、
`results/e6_descriptors.parquet`、`results/e6_descriptors_sakura.parquet`、`results/final/tables3/gen_metrics.csv`、
`results/table34_manifest.json`（現行逐類 ε）。

輸出（全部新增）：`results/eps_sweep/EPS_SWEEP_REPORT.md` / `_zh.md`、`tables3_coverage_at_fixed_eps.csv` / `.md`、
`eps_rank_flips.csv`、`eps_regimes.csv`、`eps_sweep_report_numbers.json`、`98_stdout.txt`、`figures/eps_sweep/fig1…fig8_*.png`。

重現：

```bash
cd /home/hcis-s19/Documents/ChengYu/exp_ours3_svd5
MPLCONFIGDIR=/tmp/ours3_svd5_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
  /home/hcis-s19/micromamba/envs/nps/bin/python -B scripts/98_eps_sweep_report.py
```

(the script writes its own console log to `results/eps_sweep/98_stdout.txt`; do not redirect stdout there,
the script would overwrite the redirect at the end)

腳本本身帶驗證閘（對 `gen_metrics.csv` {n_gate} 格、最大誤差 {gmax:.1e}），凍結表一旦不再重現已發表數字就會直接失敗。

本報告引用的八張圖（`fig1…fig8`）由本腳本產生。同一目錄下另外八張較早的 PNG（`eps_sweep_interaction.png` 等）
是掃描當時由未版本化的暫存腳本產生、**無法重現**，本次未更動，但不應引用。
"""
(OUT / "EPS_SWEEP_REPORT_zh.md").write_text(ZH, encoding="utf-8")
say("wrote " + str(OUT / "EPS_SWEEP_REPORT_zh.md") + f"  ({len(ZH.splitlines())} lines)")

# stdout log
(OUT / "98_stdout.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("wrote " + str(OUT / "98_stdout.txt"))
print("\nDONE")
