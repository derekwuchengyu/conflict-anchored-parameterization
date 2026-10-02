#!/usr/bin/env python3
"""89_gen_metrics_tables3.py — stage A of results/final/tables3 (generative-arm metrics).

Computes, from the frozen descriptor pools, the numbers the paper-style tables need for the
GENERATIVE arms (ours3_disk_kde, svd_d5_kde_matched_analytic, sakura_route_kde; the executed
svd_d5_kde_executed_E3 as a detail arm) and for the 39_180 RareCase arms:

  1. reproduction gate  — coverage@{10,100,1000} (interaction / path / joint, pool all, real_set
                          matched489, 3 seeds, 20 orderings) recomputed with the UNCHANGED
                          scripts/46_coverage_similarity.py routines (hit_matrices / orderings /
                          budget_curve, imported as a module; NOT re-derived) and compared with
                          results/table3_coverage.csv, results/final/T3.csv and
                          results/table3_coverage_sakura.csv. The script aborts if any value
                          differs by more than 1e-9.
  2. corner coverage    — the same coverage rule restricted to the class's RARE real scenarios:
                          the top-20 % most isolated reals by nearest-neighbour distance to the
                          other reals of the same class in the same standardized 6-D interaction
                          space that defines eps (k = ceil(0.2 * n_rankable)). Listed in
                          rare_scenes.csv.
  3. variance ratio     — per class / arm / seed / pool: var(z_gen) / var(z_real_class) per
                          descriptor, trace ratio (mean of six), generalized-variance ratio
                          (det Sigma_gen / det Sigma_real)^(1/6), and the per-centre spread.
  4. RareCase 39_180    — recovered fraction (joint / interaction / path, cutinl eps), trace
                          variance ratio (cutinl class std), n_valid / n_executed, background
                          solid rate and off-road rate, each with the exact source column.

z-space: the 46 convention — each descriptor standardized with the mean / std (ddof = 1) of the
matched489 REAL scenarios of the class that have all six descriptors finite (train-frozen,
`real_scaling`). eps_interaction = median real-to-real NN distance in that space (cvlib._eps);
eps_path = median real-to-real NN DTW. Both are asserted equal to results/table34_manifest.json.

Outputs (results/final/tables3/): gen_metrics.csv (long), rare_scenes.csv, gen_metrics_manifest.json.
No existing file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
OUT = RES / "final" / "tables3"
E9 = RES / "e9_39_180"

# ── scripts/46 imported as a module (its coverage routine is reused verbatim, not re-derived) ──
_spec = importlib.util.spec_from_file_location("cov46", PROJECT / "scripts/46_coverage_similarity.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["cov46"] = M
_spec.loader.exec_module(M)
DESC, SPACES, BUDGETS, NORD = M.DESC, M.SPACES, M.BUDGETS, M.NORD
ZC = [f"z_{k}" for k in DESC]
CLASSES = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr"]
REAL_SETS = ["matched489", "disk469"]
SEEDS = [20260910, 20260911, 20260912]
GEN_ARMS = {  # arm -> (source parquet, label, primary?)
    "ours3_disk_kde": ("e6_descriptors.parquet", "Ours+KDE (ours3 disk + KDE, executed)", True),
    "svd_d5_kde_matched_analytic": ("e6_descriptors.parquet", "SVD_d5+KDE (matched KDE, analytic decode)", True),
    "sakura_route_kde": ("e6_descriptors_sakura.parquet", "SAKURA-route+KDE (offset, v_avg; executed)", True),
    "svd_d5_kde_executed_E3": ("e6_descriptors.parquet", "SVD executed (timed Polyline, E3) KDE, first 100/class/seed [detail]", False),
}
RARE_FRAC = 0.2
SPECIAL_UID = "HetroD/00/39_180/2424-3002"
TOL = 1e-9

# 39_180 arms: (source, sample-level table, label, class data used)
RARECASE_ARMS = {
    "ours3_condkde_a2641": ("T8", "Ours+KDE (conditional KDE, class h borrowed from cutinl)", "bandwidth"),
    "ours3_sobol_a2641": ("T8", "Ours Sobol (expert ranges, no class data) [detail]", "none"),
    "svd_extbasis_gauss_h1": ("T8", "SVD_d5+KDE nearest definable: external cutinl LOGO basis + Gaussian h_loo [detail; needs external basis+bandwidth]", "basis+bandwidth"),
    "svd_rawjitter_s0.5": ("T8", "SVD raw-vector jitter sigma 0.5 m (null) [detail]", "none"),
    "sakura_route_kde_cond_39_180": ("sakura_cond", "SAKURA-route+KDE conditional (cutinl bandwidth)", "bandwidth (cutinl fit)"),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nanmax_abs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    if (np.isfinite(a) != np.isfinite(b)).any():
        return np.inf
    return float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0


class Rows:
    """Collector for the long-format gen_metrics.csv."""

    def __init__(self):
        self.rows = []

    def add(self, table_group, cls, real_set, arm, pool, metric, space, budget, values, n_real=np.nan, n_rare=np.nan,
            n_samples=np.nan, source="", note="", seeds=None):
        v = np.asarray([x for x in values if x is not None], float)
        fin = v[np.isfinite(v)]
        self.rows.append(dict(table_group=table_group, **{"class": cls}, real_set=real_set, arm=arm, pool=pool, metric=metric,
                              space=space, budget=budget,
                              value=float(fin.mean()) if len(fin) else np.nan,
                              value_seed_min=float(fin.min()) if len(fin) else np.nan,
                              value_seed_max=float(fin.max()) if len(fin) else np.nan,
                              n_seeds=int(len(fin)), n_real=n_real, n_rare=n_rare, n_samples=n_samples, source=source, note=note,
                              seeds=";".join(str(s) for s in seeds) if seeds else ""))


# ═══════════════════════════════════════════════════════════════════════════
# real reference, scaling, eps (all asserted against the frozen 46 outputs)
# ═══════════════════════════════════════════════════════════════════════════
def load_real_and_eps(checks):
    real = pd.read_csv(RES / "e6_real_reference.csv", low_memory=False)
    assert real.scenario_uid.is_unique      # 489 before the 2026-09-14 cutinl exclusion, 486 after
    real["int_finite"] = real.int_finite.astype(bool)
    real["in_disk469"] = real.in_disk469.astype(bool)
    real["in_matched489"] = real.in_matched489.astype(bool)
    scaling = M.real_scaling(real)                      # 46 line ~ 'def real_scaling'
    man = json.loads((RES / "table34_manifest.json").read_text())
    d_scale = 0.0
    for cls in CLASSES:
        for k in DESC:
            d_scale = max(d_scale, abs(scaling[cls]["mean"][k] - man["real_scaling"][cls]["mean"][k]),
                          abs(scaling[cls]["sd"][k] - man["real_scaling"][cls]["sd"][k]))
        assert scaling[cls]["n"] == man["real_scaling"][cls]["n"], cls
    Zr = M.zscore(real, scaling)
    d_z = nanmax_abs(Zr, real[ZC].to_numpy(float))
    eps = {}
    for cls, g in real.groupby("subset"):
        gi = g[g.int_finite]
        eps[cls] = dict(interaction=M.cv_eps(gi[ZC].to_numpy(float)), path=float(np.nanmedian(g.nn_dtw_m)),
                        n_real_int=int(len(gi)), n_real=int(len(g)))
    d_eps = max(max(abs(eps[c]["interaction"] - man["eps"][c]["interaction"]), abs(eps[c]["path"] - man["eps"][c]["path"])) for c in CLASSES)
    checks["real_scaling_vs_table34_manifest_max_abs_diff"] = d_scale
    checks["real_z_recomputed_vs_e6_real_reference_max_abs_diff"] = d_z
    checks["eps_vs_table34_manifest_max_abs_diff"] = d_eps
    assert d_scale < TOL and d_z < 1e-9 and d_eps < TOL, (d_scale, d_z, d_eps)
    # in the matched489 z-space the real class variance is 1 by construction (ddof=1)
    for cls in CLASSES:
        v = real[real.subset.eq(cls) & real.int_finite][ZC].var(ddof=1).to_numpy()
        assert np.allclose(v, 1.0, atol=1e-9), (cls, v)
    return real, scaling, eps


def load_pools(scaling, real, checks):
    d6 = pd.read_parquet(RES / "e6_descriptors.parquet")
    ds = pd.read_parquet(RES / "e6_descriptors_sakura.parquet")
    pools = {}
    for arm, (src, _label, _p) in GEN_ARMS.items():
        df = (d6 if src == "e6_descriptors.parquet" else ds)
        A = df[df.arm.eq(arm)].copy().reset_index(drop=True)
        assert len(A), arm
        # z columns of the pool are exactly the 46 z-scaling (recomputed here from pet..u_c)
        Z = M.zscore(A, scaling)
        d = nanmax_abs(Z, A[ZC].to_numpy(float))
        checks[f"pool_z_recomputed_max_abs_diff[{arm}]"] = d
        assert d < 1e-9, (arm, d)
        assert (A.int_finite.to_numpy() == np.isfinite(Z).all(axis=1)).all(), arm
        # real_* of the kernel centre = the real reference row of that scenario
        rr = real.set_index("scenario_uid")
        for k in DESC:
            ref = rr[k].reindex(A.kernel_center_uid).to_numpy(float)
            dd = nanmax_abs(ref, A[f"real_{k}"].to_numpy(float))
            assert dd < 1e-9, (arm, k, dd)
        assert A.groupby(["subset", "seed"]).draw.apply(lambda v: v.is_unique).all(), arm
        pools[arm] = A
    return pools


# ═══════════════════════════════════════════════════════════════════════════
# coverage (46 machinery) — returns per (space) the ordering curves for the full and rare real sets
# ═══════════════════════════════════════════════════════════════════════════
def pool_slice(A, seed, pool):
    """46 lines 'for pool in ("all", "valid")': the valid pool is the contiguous Table-5-flagged prefix."""
    Gfull = A[A.seed.eq(seed)].sort_values("draw").reset_index(drop=True)
    if pool == "all":
        G = Gfull
    else:
        flagged = Gfull.table5_available.to_numpy()
        n_prefix = int(np.argmin(flagged)) if not flagged.all() else len(flagged)
        if n_prefix == 0:
            return None, None
        G = Gfull.iloc[:n_prefix].reset_index(drop=True)
    valid_mask = np.array([bool(v) if pd.notna(v) else False for v in G.valid_gate.to_numpy()])
    return G, (None if pool == "all" else valid_mask)


def coverage_curves(G, Rall, Rsets, eps_cls, vm, n_ord, row_masks):
    """H = 46.hit_matrices; for each space and each named real subset (row mask on Rall) the
    46.budget_curve over 46.orderings. Returns {space: {mask_name: (curve (n_ord x M), nR)}}."""
    H_all, _D = M.hit_matrices(G, Rall, eps_cls["interaction"], eps_cls["path"], 1.0, vm)
    Mn = len(G)
    ords = M.orderings(Mn, n_ord)
    out = {}
    for space in SPACES:
        sel_space = Rall.scenario_uid.isin(Rsets[space].scenario_uid).to_numpy()
        out[space] = {}
        for name, mask in row_masks.items():
            sel = sel_space & mask
            nR = int(sel.sum())
            if nR == 0:
                out[space][name] = (None, 0)
                continue
            out[space][name] = (M.budget_curve(H_all[space][sel], ords), nR)
    return out


def curve_stats(curve, b):
    c = curve[:, b - 1]
    return float(c.mean()), float(c.min()), float(c.max()), float(c[0])


# ═══════════════════════════════════════════════════════════════════════════
# rare scenes
# ═══════════════════════════════════════════════════════════════════════════
def rare_tables(real, eps):
    rows = []
    rare_uids = {}
    # matched489 ranks first so the disk469 rows can carry the matched489 rank as well
    rank_m489 = {}
    for real_set in REAL_SETS:
        for cls in CLASSES:
            inset = real.in_disk469 if real_set == "disk469" else real.in_matched489
            Rall = real[real.subset.eq(cls) & inset]
            R = Rall[Rall.int_finite].reset_index(drop=True)
            Z = R[ZC].to_numpy(float)
            D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
            np.fill_diagonal(D, np.inf)
            nn = D.min(axis=1)
            nn_uid = R.scenario_uid.to_numpy()[D.argmin(axis=1)]
            if real_set == "matched489":
                # the class eps is the median of exactly these NN distances (cvlib._eps on the same Z)
                assert abs(float(np.median(nn)) - eps[cls]["interaction"]) < TOL, cls
            order = np.lexsort((R.scenario_uid.to_numpy(), -nn))      # most isolated first; ties by uid
            rank = np.empty(len(R), int)
            rank[order] = np.arange(1, len(R) + 1)
            n_rank, n_tot = len(R), len(Rall)
            k = int(math.ceil(RARE_FRAC * n_rank))
            k_tot = int(math.ceil(RARE_FRAC * n_tot))
            is_rare = rank <= k
            if real_set == "matched489":
                rank_m489.update({u: int(r) for u, r in zip(R.scenario_uid, rank)})
            rare_uids[(cls, real_set)] = dict(uids=list(R.scenario_uid[is_rare]), k=k, n_rankable=n_rank, n_total=n_tot,
                                              k_of_total=k_tot, uids_k_of_total=list(R.scenario_uid[rank <= k_tot]))
            for i, r in R.iterrows():
                rows.append({"class": cls, "real_set": real_set, "scenario_uid": r.scenario_uid, "scenario_id": r.scenario_id,
                             "nn_dist": float(nn[i]), "nn_uid": nn_uid[i], "rank": int(rank[i]), "is_rare": bool(is_rare[i]),
                             "k_rare": k, "n_real_rankable": n_rank, "n_real_total": n_tot,
                             "is_rare_k_of_total": bool(rank[i] <= k_tot), "k_of_total": k_tot,
                             "rank_in_matched489": rank_m489.get(r.scenario_uid, np.nan),
                             "nn_dist_over_eps": float(nn[i] / eps[cls]["interaction"]), "eps_interaction": eps[cls]["interaction"],
                             "pet": r.pet, "d_min": r.d_min, "alpha": r.alpha, "conflict_x": r.conflict_x, "conflict_y": r.conflict_y, "u_c": r.u_c,
                             "z_pet": r.z_pet, "z_d_min": r.z_d_min, "z_alpha": r.z_alpha, "z_conflict_x": r.z_conflict_x,
                             "z_conflict_y": r.z_conflict_y, "z_u_c": r.z_u_c,
                             "in_disk469": bool(r.in_disk469), "group_id": r.group_id})
    return pd.DataFrame(rows), rare_uids


# ═══════════════════════════════════════════════════════════════════════════
# variance ratio
# ═══════════════════════════════════════════════════════════════════════════
def gv_ratio(Zg, Zr):
    """(det Sigma_gen / det Sigma_real)^(1/6); nan when either covariance is singular or too few rows."""
    if len(Zg) < len(DESC) + 1 or len(Zr) < len(DESC) + 1:
        return np.nan
    sg, ldg = np.linalg.slogdet(np.cov(Zg, rowvar=False, ddof=1))
    sr, ldr = np.linalg.slogdet(np.cov(Zr, rowvar=False, ddof=1))
    if sg <= 0 or sr <= 0:
        return np.nan
    return float(np.exp((ldg - ldr) / len(DESC)))


def variance_block(G, Rz, centre_col="kernel_center_uid"):
    Zg = G[ZC].to_numpy(float)
    var_g = Zg.var(axis=0, ddof=1) if len(Zg) >= 2 else np.full(len(DESC), np.nan)
    var_r = Rz.var(axis=0, ddof=1)
    ratio = var_g / var_r
    out = dict(trace_ratio=float(np.mean(ratio)), gv_ratio=gv_ratio(Zg, Rz), n=int(len(Zg)),
               trace_gen=float(var_g.sum()), trace_real=float(var_r.sum()))
    for j, k in enumerate(DESC):
        out[f"ratio_{k}"] = float(ratio[j])
    # per-centre spread: trace of var(z) of each centre's own finite samples (centres with >= 2)
    tr = []
    counts = G.groupby(centre_col).size()
    for _c, gg in G.groupby(centre_col):
        if len(gg) >= 2:
            tr.append(float(gg[ZC].to_numpy(float).var(axis=0, ddof=1).sum()))
    out.update(per_centre_trace_median=float(np.median(tr)) if tr else np.nan,
               per_centre_trace_ratio=float(np.median(tr) / var_r.sum()) if tr else np.nan,
               n_centres=int(counts.size), n_centres_ge2=int(len(tr)),
               samples_per_centre_median=float(counts.median()) if counts.size else np.nan)
    return out


# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-orderings", type=int, default=NORD)
    args = ap.parse_args()
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    checks, repro = {}, []
    R = Rows()

    real, scaling, eps = load_real_and_eps(checks)
    pools = load_pools(scaling, real, checks)
    print("[real/eps] scaling, z and eps identical to the frozen 46 outputs", flush=True)

    # frozen references for the reproduction gate
    t3_seed = pd.read_csv(RES / "table3_coverage.csv", low_memory=False)
    t3_final = pd.read_csv(RES / "final" / "T3.csv", low_memory=False)
    t3_sak = pd.read_csv(RES / "table3_coverage_sakura.csv", low_memory=False)

    def ref_seed(arm, cls, real_set, seed, pool, space, b):
        src = t3_sak if arm == "sakura_route_kde" else t3_seed
        r = src[src.arm.eq(arm) & src.subset.eq(cls) & src.real_set.eq(real_set) & src.seed.eq(seed) & src.pool.eq(pool)
                & src.space.eq(space) & src.budget.eq(b) & src.eps_mult.eq(1.0)]
        if len(r) != 1:
            return None
        r = r.iloc[0]
        return float(r.coverage_mean), float(r.coverage_min), float(r.coverage_max), float(r.coverage_natural_order), int(r.n_real)

    def ref_final(arm, cls, real_set, pool, space, b):
        r = t3_final[t3_final.block.eq("coverage") & t3_final.arm.eq(arm) & t3_final.subset.eq(cls) & t3_final.real_set.eq(real_set)
                     & t3_final.pool.eq(pool) & t3_final.space.eq(space) & t3_final.budget.eq(b)]
        if len(r) != 1:
            return None
        r = r.iloc[0]
        return float(r.value), float(r.value_seed_min), float(r.value_seed_max), int(r.n_seeds)

    rare_df, rare_uids = rare_tables(real, eps)
    rare_df.to_csv(OUT / "rare_scenes.csv", index=False)
    print("[rare] k per class (matched489): " + ", ".join(f"{c} {rare_uids[(c, 'matched489')]['k']}/{rare_uids[(c, 'matched489')]['n_rankable']}" for c in CLASSES), flush=True)

    # ── coverage (full real set = reproduction gate) + corner coverage ──
    cov_store = {}   # (arm, cls, real_set, pool, seed) -> {space: {name: (mean,min,max,nat,b,nR)}}
    for arm, A in pools.items():
        for cls in CLASSES:
            Acls = A[A.subset.eq(cls)]
            if not len(Acls):
                continue
            for real_set in REAL_SETS:
                inset = real.in_disk469 if real_set == "disk469" else real.in_matched489
                Rall = real[real.subset.eq(cls) & inset].reset_index(drop=True)
                Rsets = dict(interaction=Rall[Rall.int_finite], path=Rall, joint=Rall[Rall.int_finite])
                ru = rare_uids[(cls, real_set)]
                masks = {"full": np.ones(len(Rall), bool),
                         "rare": Rall.scenario_uid.isin(ru["uids"]).to_numpy(),
                         "rare_k_of_total": Rall.scenario_uid.isin(ru["uids_k_of_total"]).to_numpy()}
                for pool in ("all", "valid"):
                    for seed in sorted(Acls.seed.unique()):
                        G, vm = pool_slice(Acls, seed, pool)
                        if G is None:
                            continue
                        Mn = len(G)
                        curves = coverage_curves(G, Rall, Rsets, eps[cls], vm, args.n_orderings, masks)
                        n_valid = int(np.sum(vm)) if vm is not None else Mn
                        rec = {}
                        for space in SPACES:
                            rec[space] = {}
                            for name in masks:
                                curve, nR = curves[space][name]
                                if curve is None:
                                    continue
                                for b in sorted(set(BUDGETS + [Mn])):
                                    if b <= Mn:
                                        rec[space][(name, b)] = (*curve_stats(curve, b), nR, Mn, n_valid)
                        cov_store[(arm, cls, real_set, pool, seed)] = rec
                        # reproduction gate against the per-seed frozen tables (full real set)
                        for space in SPACES:
                            for (name, b), (mean_, min_, max_, nat_, nR, Mn_, _nv) in rec[space].items():
                                if name != "full":
                                    continue
                                ref = ref_seed(arm, cls, real_set, seed, pool, space, b)
                                if ref is None:
                                    repro.append(dict(arm=arm, subset=cls, real_set=real_set, seed=seed, pool=pool, space=space, budget=b,
                                                      mine=mean_, ref=np.nan, ref_table="(no frozen row)", abs_diff=np.nan, n_real=nR))
                                    continue
                                d = max(abs(mean_ - ref[0]), abs(min_ - ref[1]), abs(max_ - ref[2]), abs(nat_ - ref[3]))
                                repro.append(dict(arm=arm, subset=cls, real_set=real_set, seed=seed, pool=pool, space=space, budget=b,
                                                  mine=mean_, ref=ref[0], ref_table="table3_coverage_sakura.csv" if arm == "sakura_route_kde" else "table3_coverage.csv",
                                                  abs_diff=d, n_real=nR, n_real_ref=ref[4]))
    repro_df = pd.DataFrame(repro)
    repro_df.to_csv(OUT / "reproduction_check_per_seed.csv", index=False)

    # seed-aggregated coverage rows + T3.csv comparison
    repro_final = []
    for arm, A in pools.items():
        for cls in CLASSES:
            for real_set in REAL_SETS:
                for pool in ("all", "valid"):
                    keys = [k for k in cov_store if k[:4] == (arm, cls, real_set, pool)]
                    if not keys:
                        continue
                    seeds = sorted(k[4] for k in keys)
                    for space in SPACES:
                        for name in ("full", "rare", "rare_k_of_total"):
                            budgets = sorted({b for k in keys for (n, b) in cov_store[k][space] if n == name})
                            for b in budgets:
                                vals = [cov_store[k][space].get((name, b)) for k in keys]
                                vals = [v for v in vals if v is not None]
                                if not vals:
                                    continue
                                means = [v[0] for v in vals]
                                nR, Mn, nv = vals[0][4], vals[0][5], int(np.mean([v[6] for v in vals]))
                                is_pool = b == Mn
                                ru = rare_uids[(cls, real_set)]
                                if name == "full":
                                    src = f"recomputed with scripts/46 hit_matrices/orderings/budget_curve; == results/final/T3.csv ({'seed-mean' if arm != 'sakura_route_kde' else 'table3_coverage_sakura.csv seed-mean'})"
                                    R.add("coverage", cls, real_set, arm, pool, "coverage", space, b, means, n_real=nR, n_rare=np.nan, n_samples=Mn,
                                          source=src, seeds=seeds,
                                          note=f"mean over {len(seeds)} seed(s) of the 20-ordering mean; budget==pool size -> ordering-invariant" + ("" if pool == "all" else f"; valid pool = Table-5 flagged prefix, n_valid mean {nv}"))
                                    ref = ref_final(arm, cls, real_set, pool, space, b) if arm != "sakura_route_kde" else None
                                    if ref is not None:
                                        d = max(abs(np.mean(means) - ref[0]), abs(min(means) - ref[1]), abs(max(means) - ref[2]))
                                        repro_final.append(dict(arm=arm, subset=cls, real_set=real_set, pool=pool, space=space, budget=b, mine=float(np.mean(means)),
                                                                ref=ref[0], n_seeds_mine=len(means), n_seeds_ref=ref[3], abs_diff=d, ref_table="results/final/T3.csv"))
                                    elif arm == "sakura_route_kde":
                                        s = t3_sak[t3_sak.arm.eq(arm) & t3_sak.subset.eq(cls) & t3_sak.real_set.eq(real_set) & t3_sak.pool.eq(pool)
                                                   & t3_sak.space.eq(space) & t3_sak.budget.eq(b) & t3_sak.eps_mult.eq(1.0)]
                                        if len(s):
                                            d = max(abs(np.mean(means) - s.coverage_mean.mean()), abs(min(means) - s.coverage_mean.min()), abs(max(means) - s.coverage_mean.max()))
                                            repro_final.append(dict(arm=arm, subset=cls, real_set=real_set, pool=pool, space=space, budget=b, mine=float(np.mean(means)),
                                                                    ref=float(s.coverage_mean.mean()), n_seeds_mine=len(means), n_seeds_ref=int(s.seed.nunique()), abs_diff=d,
                                                                    ref_table="results/table3_coverage_sakura.csv (seed-mean)"))
                                else:
                                    metric = "corner_coverage" if name == "rare" else "corner_coverage_k_of_total"
                                    kk = ru["k"] if name == "rare" else ru["k_of_total"]
                                    R.add("corner_coverage", cls, real_set, arm, pool, metric, space, b, means, n_real=ru["n_rankable"], n_rare=kk, n_samples=Mn,
                                          source="same coverage rule as T3 restricted to the rare real subset (rare_scenes.csv is_rare" + ("" if name == "rare" else "_k_of_total") + ")",
                                          seeds=seeds,
                                          note=(f"rare = top-{int(RARE_FRAC*100)}% most isolated reals by NN distance in the eps z-space, k=ceil(0.2*{'n_rankable' if name == 'rare' else 'n_total'})={kk}; "
                                                f"NN computed among the {real_set} reals of the class with all six descriptors finite" + ("" if pool == "all" else f"; valid pool = Table-5 flagged prefix, n_valid mean {nv}")))
    repro_final_df = pd.DataFrame(repro_final)
    repro_final_df.to_csv(OUT / "reproduction_check_seed_mean.csv", index=False)

    # reproduction gate verdict (pool all, matched489, budgets 10/100/1000, three primary arms)
    gate = repro_df[repro_df.pool.eq("all") & repro_df.real_set.eq("matched489") & repro_df.arm.isin(["ours3_disk_kde", "svd_d5_kde_matched_analytic", "sakura_route_kde"])
                    & repro_df.budget.isin([10, 100, 1000])]
    missing = gate[gate.ref.isna()]
    worst = float(np.nanmax(gate.abs_diff)) if len(gate) else np.nan
    gate_final = repro_final_df[repro_final_df.pool.eq("all") & repro_final_df.real_set.eq("matched489") & repro_final_df.budget.isin([10, 100, 1000])]
    worst_final = float(np.nanmax(gate_final.abs_diff)) if len(gate_final) else np.nan
    all_worst = float(np.nanmax(repro_df.abs_diff)) if repro_df.abs_diff.notna().any() else np.nan
    checks.update(reproduction_gate_n_cells=int(len(gate)), reproduction_gate_missing_reference_rows=int(len(missing)),
                  reproduction_gate_max_abs_diff_per_seed=worst, reproduction_gate_max_abs_diff_seed_mean_vs_T3=worst_final,
                  reproduction_all_cells_max_abs_diff=all_worst, reproduction_all_cells_n=int(repro_df.abs_diff.notna().sum()))
    print(f"[repro] gate cells {len(gate)} (missing ref {len(missing)}), max |diff| per-seed {worst:.3e}, seed-mean vs T3 {worst_final:.3e}; "
          f"all {int(repro_df.abs_diff.notna().sum())} cells (incl. valid pool, disk469): max |diff| {all_worst:.3e}", flush=True)
    if len(missing) or not (worst <= TOL and worst_final <= TOL):
        print(gate[gate.abs_diff.isna() | (gate.abs_diff > TOL)].to_string(), flush=True)
        raise SystemExit("REPRODUCTION GATE FAILED — stopping (no variant invented)")

    # ── variance ratio ──
    for arm, A in pools.items():
        for cls in CLASSES:
            Acls = A[A.subset.eq(cls)]
            if not len(Acls):
                continue
            for real_set in REAL_SETS:
                inset = real.in_disk469 if real_set == "disk469" else real.in_matched489
                Rz = real[real.subset.eq(cls) & inset & real.int_finite][ZC].to_numpy(float)
                for pool in ("all", "valid"):
                    per_seed = []
                    seeds = []
                    for seed in sorted(Acls.seed.unique()):
                        G, vm = pool_slice(Acls, seed, pool)
                        if G is None:
                            continue
                        if vm is not None:
                            G = G[vm]
                        G = G[G.int_finite]
                        if len(G) < 2:
                            continue
                        per_seed.append(variance_block(G, Rz))
                        seeds.append(seed)
                    if not per_seed:
                        continue
                    n_s = int(np.mean([p["n"] for p in per_seed]))
                    common = f"z standardized with the matched489 class real mean/std (46 real_scaling); real variance from the {real_set} class reals with all six descriptors finite (one row per scenario, n={len(Rz)}); gen samples int_finite" + ("" if pool == "all" else " AND valid_gate (Table-5 flagged prefix)")
                    R.add("variance", cls, real_set, arm, pool, "variance_ratio_trace", "interaction", np.nan, [p["trace_ratio"] for p in per_seed], n_real=len(Rz), n_samples=n_s,
                          source="mean over the six per-descriptor var(z_gen)/var(z_real) (ddof=1)", note=common + "; 1.0 = same spread as the recorded class", seeds=seeds)
                    R.add("variance", cls, real_set, arm, pool, "variance_ratio_gv", "interaction", np.nan, [p["gv_ratio"] for p in per_seed], n_real=len(Rz), n_samples=n_s,
                          source="(det Sigma_gen / det Sigma_real)^(1/6), slogdet, nan if singular", note=common, seeds=seeds)
                    for k in DESC:
                        R.add("variance", cls, real_set, arm, pool, f"variance_ratio_{k}", "interaction", np.nan, [p[f"ratio_{k}"] for p in per_seed], n_real=len(Rz), n_samples=n_s,
                              source="var(z_gen)/var(z_real) (ddof=1)", note=common, seeds=seeds)
                    R.add("variance", cls, real_set, arm, pool, "per_centre_trace_median", "interaction", np.nan, [p["per_centre_trace_median"] for p in per_seed], n_real=len(Rz), n_samples=n_s,
                          source="median over kernel centres (>=2 finite samples) of trace var(z) of the centre's own samples",
                          note=common + f"; n_centres mean {np.mean([p['n_centres'] for p in per_seed]):.1f}, with >=2 {np.mean([p['n_centres_ge2'] for p in per_seed]):.1f}, samples/centre median {np.mean([p['samples_per_centre_median'] for p in per_seed]):.1f}", seeds=seeds)
                    R.add("variance", cls, real_set, arm, pool, "per_centre_trace_ratio", "interaction", np.nan, [p["per_centre_trace_ratio"] for p in per_seed], n_real=len(Rz), n_samples=n_s,
                          source="per_centre_trace_median / trace(Sigma_real) (= /6 for matched489)", note=common, seeds=seeds)
                    R.add("variance", cls, real_set, arm, pool, "trace_var_gen", "interaction", np.nan, [p["trace_gen"] for p in per_seed], n_real=len(Rz), n_samples=n_s,
                          source="sum of the six var(z_gen); trace(Sigma_real)=" + f"{per_seed[0]['trace_real']:.6f}", note=common, seeds=seeds)

    # ── RareCase 39_180 ──
    rr = real.set_index("scenario_uid")
    if SPECIAL_UID in rr.index:
        r0 = rr.loc[SPECIAL_UID]
        sc = scaling["cutinl"]
        eps_i, eps_p = eps["cutinl"]["interaction"], eps["cutinl"]["path"]
        checks["rarecase_39_180_reference_source"] = "current cutinl real reference / scaling / eps"
    else:
        # v3 rerun 2026-09-14: 39_180 left the cutinl cohort (relabelled). The rarecase / Table III pipeline is frozen by
        # user decision, so this block uses the pre-rerun cutinl scaling / eps and the 39_180 real row extracted from
        # _snapshot_20260914.tar (verified to reproduce the stored recovered_* flags of the frozen 46b cond parquet).
        frozen = json.loads((RES / "special_39_180_frozen_reference.json").read_text())
        r0 = pd.Series(frozen["real_row"])
        sc = frozen["cutinl_real_scaling"]
        eps_i, eps_p = frozen["cutinl_eps"]["interaction"], frozen["cutinl_eps"]["path"]
        checks["rarecase_39_180_reference_source"] = "frozen pre-rerun (results/special_39_180_frozen_reference.json)"
    z0 = np.array([(float(r0[k]) - sc["mean"][k]) / sc["sd"][k] for k in DESC])
    t8 = pd.read_csv(RES / "final" / "T8.csv", low_memory=False).set_index("arm")
    t8s = pd.read_csv(E9 / "T8_samples.csv", low_memory=False)
    pet9 = pd.read_csv(E9 / "e9_bbox_pet_paired.csv", low_memory=False).drop_duplicates("sample_id").set_index("sample_id")
    rep = t8s[t8s.arm.eq("replay")].iloc[0]
    for k, src in M.DESC_SRC.items():
        assert abs(float(rep[src]) - float(r0[k])) < 1e-6, (k, rep[src], r0[k])
    cond = pd.read_parquet(RES / "e6_descriptors_sakura_39_180_cond.parquet")
    t5sak_sum = pd.read_csv(RES / "table5_validity_summary_sakura.csv", low_memory=False)
    t5sak_smp = pd.read_csv(RES / "table5_validity_samples_sakura.csv", low_memory=False)
    bg_esm = pd.read_csv(E9 / "e9_esmini_bg_collisions.csv", low_memory=False)
    checks["e9_esmini_bg_collisions_runs"] = {a: dict(n=int(len(g)), agent_bg_leq_gt_end_gt0=int((g.agent_bg_frames_leq_gt_end > 0).sum()),
                                                      analytic_all_solid_gt0=int((g.analytic_bg_all_solid_hits > 0).sum()))
                                              for a, g in bg_esm.groupby("arm")}
    rc_rows = []
    for arm, (src, label, cdu) in RARECASE_ARMS.items():
        if src == "T8":
            S = t8s[t8s.arm.eq(arm)].copy().reset_index(drop=True)
            # PET in T8_samples == e9_bbox_pet_paired generated_pet (verified), non-PET from the 31 paired tables
            p = pet9.generated_pet.reindex(S.sample_id).to_numpy(float)
            assert nanmax_abs(p, S.pet.to_numpy(float)) < 1e-9, arm
            X = np.column_stack([S[M.DESC_SRC[k]].to_numpy(float) for k in DESC])
            dtw = S.err_dtw.to_numpy(float)
            valid_primary = S.valid.fillna(False).astype(bool).to_numpy()
            valid_def = "T8_samples.csv `valid` (no teleport / wrong-way / a_lat>5 / v>25 / >5% outside driving+shoulder union; scripts/52)"
            seeds_arr = S.seed.astype(int).to_numpy()
            n_exec, n_valid_t8 = int(t8.loc[arm, "n_executed"]), int(t8.loc[arm, "n_valid"])
            assert n_exec == len(S) and n_valid_t8 == int(valid_primary.sum()), arm
            bg = float(t8.loc[arm, "bg_solid_rate_all_gtsupport"]); bg_src = "results/final/T8.csv:bg_solid_rate_all_gtsupport"
            bg_valid = float(t8.loc[arm, "bg_solid_rate_all_valid_gtsupport"]); bg_valid_src = "results/final/T8.csv:bg_solid_rate_all_valid_gtsupport"
            bg_l0 = float(t8.loc[arm, "bg_solid_rate_label0_gtsupport"]); bg_l0_src = "results/final/T8.csv:bg_solid_rate_label0_gtsupport"
            offroad_n = int(t8.loc[arm, "offroad_n"]); off_src = "results/final/T8.csv:offroad_n / n_executed"
            teleport_n = int(t8.loc[arm, "teleport_n"])
            sample_note = "descriptors: results/e9_39_180/T8_samples.csv pet,min_dist,conflict_angle,conflict_x,conflict_y,agent_arr_speed; path: err_dtw (traj_dtw vs GT target path)"
        else:
            S = cond.copy().reset_index(drop=True)
            X = np.column_stack([S[k].to_numpy(float) for k in DESC])
            dtw = S.dtw.to_numpy(float)
            seeds_arr = S.seed.astype(int).to_numpy()
            # Table-5 flags of the cond arm (horizon-invariant flags; taken at 'full')
            f5 = t5sak_smp[t5sak_smp.arm.eq(arm) & t5sak_smp.horizon.eq("full")].drop_duplicates("sample_id").set_index("sample_id").reindex(S.sample_id)
            assert f5.teleport.notna().all(), arm
            t8like = ~(f5.teleport.astype(bool) | f5.wrongway.astype(bool) | f5.phys_gate1.astype(bool) | f5.offroad_vl.astype(bool)).to_numpy()
            valid_primary = t8like
            valid_def = ("T8-like approximation from Table-5 flags (no teleport / wrong-way / physics gate1 (v>25 or a_lat>5) / off-road VL); "
                         "the off-road rule (VL windowing) differs from T8's 33_e7 driving+shoulder-union rule and there is no bg-hit term")
            gate_full = S.valid_gate.map(lambda v: bool(v) if pd.notna(v) else False).to_numpy()
            n_exec = len(S); n_valid_t8 = int(t8like.sum())
            row5 = t5sak_sum[t5sak_sum.arm.eq(arm) & t5sak_sum.scope.eq("special_39_180")].set_index("horizon")
            bg = float(row5.loc["chmed", "any_solid_rate"]); bg_src = "results/table5_validity_summary_sakura.csv:any_solid_rate (special_39_180, horizon chmed) — Table-5 OBB at the chmed horizon, NOT T8's GT-support horizon"
            bg_valid = np.nan; bg_valid_src = "n/a (Table 5 has no valid-only bg rate)"
            bg_l0 = np.nan; bg_l0_src = "n/a"
            offroad_n = int(f5.offroad_vl.astype(bool).sum()); off_src = "results/table5_validity_samples_sakura.csv:offroad_vl (full) count / n — VL rule, not T8's union rule"
            assert abs(offroad_n / n_exec - float(row5.loc["full", "offroad_vl_rate"])) < 1e-9
            teleport_n = int(f5.teleport.astype(bool).sum())
            sample_note = "descriptors/dtw: results/e6_descriptors_sakura_39_180_cond.parquet (46b); recovered_* recomputed here and asserted equal to the stored columns"
        Z = (X - np.array([sc["mean"][k] for k in DESC])) / np.array([sc["sd"][k] for k in DESC])
        fin = np.isfinite(Z).all(axis=1)
        dist = np.full(len(Z), np.inf)
        dist[fin] = np.linalg.norm(Z[fin] - z0, axis=1)
        rec_i = dist <= eps_i
        rec_p = np.isfinite(dtw) & (dtw <= eps_p)
        rec_j = rec_i & rec_p
        if src != "T8":
            assert (rec_i == S.recovered_interaction.to_numpy(bool)).all() and (rec_p == S.recovered_path.to_numpy(bool)).all() and (rec_j == S.recovered_joint.to_numpy(bool)).all(), arm
        pools_rc = {"all": np.ones(len(S), bool), "valid": valid_primary}
        if src != "T8":
            pools_rc["valid_table5_gate"] = gate_full
            # cross-check against the frozen 46b table (all-draws and valid-only recovered rates, per seed and pooled)
            cond_ref = pd.read_csv(RES / "table3_sakura_39_180_cond.csv").set_index("scope")
            dmax = 0.0
            for scope, m in [("all_seeds", np.ones(len(S), bool))] + [(f"seed_{s}", seeds_arr == s) for s in sorted(set(seeds_arr))]:
                for space, rec in (("interaction", rec_i), ("path", rec_p), ("joint", rec_j)):
                    dmax = max(dmax, abs(float(rec[m].mean()) - float(cond_ref.loc[scope, f"recovered_{space}_rate"])),
                               abs(float((rec & gate_full)[m].mean()) - float(cond_ref.loc[scope, f"recovered_{space}_rate_valid_only"])))
            checks["sakura_cond_39_180_recovered_rates_vs_table3_sakura_39_180_cond_max_abs_diff"] = dmax
            assert dmax < TOL, dmax
        for pool, pm in pools_rc.items():
            seeds_here = sorted(set(seeds_arr))
            pool_def = "" if pool == "all" else (valid_def if pool == "valid" else "the full Table-5 valid_gate (teleport/off-road VL/phys gate1/bg solid chmed/sampler-invalid)")
            note_pool = "" if pool == "all" else f"; pool '{pool}' = draws passing {pool_def} (n={int(pm.sum())})"

            def frac(rec, s, among_valid=False):
                """46b convention for the valid pools: (recovered AND valid) / all draws of the seed — invalid draws cost
                budget and cannot recover; among_valid=True gives the conditional rate recovered / valid draws instead."""
                ms = seeds_arr == s
                if among_valid:
                    return float(rec[pm & ms].mean()) if (pm & ms).any() else np.nan
                return float((rec & pm)[ms].mean()) if ms.any() else np.nan

            for space, rec in (("interaction", rec_i), ("path", rec_p), ("joint", rec_j)):
                R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, pool, "recovered_fraction", space, len(S), [frac(rec, s) for s in seeds_here], n_real=1, n_rare=1, n_samples=len(S),
                      source=sample_note + f"; eps = cutinl class (interaction {eps_i:.6f} z, path {eps_p:.6f} m); z with the cutinl class real mean/std",
                      note=f"'coverage' for n=1 = fraction of draws within eps of the single recorded 39_180 descriptor vector ({space}); value = mean over the 3 seeds x 100 draws" + note_pool
                           + ("; denominator = all draws of the seed (recovered AND valid; 46b recovered_*_rate_valid_only convention)" if pool != "all" else ""), seeds=seeds_here)
                if pool != "all":
                    R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, pool, "recovered_fraction_among_valid", space, len(S), [frac(rec, s, True) for s in seeds_here], n_real=1, n_rare=1,
                          n_samples=int(pm.sum()), source="as recovered_fraction", note=f"conditional rate: recovered / valid draws of the seed ({space})" + note_pool, seeds=seeds_here)
            # overall (pooled over seeds) too
            for space, rec in (("interaction", rec_i), ("joint", rec_j)):
                R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, pool, "recovered_fraction_pooled_seeds", space, len(S), [float((rec & pm).mean())],
                      n_real=1, n_rare=1, n_samples=len(S), source="as above, all 300 draws pooled", note="single value (no seed split); denominator = all draws" + note_pool)
            # variance (trace ratio vs the cutinl class real std => real var = 1 per descriptor)
            tr, gv, per_desc = [], [], {k: [] for k in DESC}
            for s in seeds_here:
                m = pm & (seeds_arr == s) & fin
                if m.sum() >= 2:
                    v = Z[m].var(axis=0, ddof=1)
                    tr.append(float(v.mean())); gv.append(gv_ratio(Z[m], real[real.subset.eq("cutinl") & real.in_matched489 & real.int_finite][ZC].to_numpy(float)))
                    for j, k in enumerate(DESC):
                        per_desc[k].append(float(v[j]))
                else:
                    tr.append(np.nan); gv.append(np.nan)
                    for k in DESC:
                        per_desc[k].append(np.nan)
            n_fin = int((pm & fin).sum())
            R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, pool, "variance_ratio_trace", "interaction", np.nan, tr, n_real=46, n_samples=n_fin,
                  source="mean over six of var(z_draws) (ddof=1) with z standardized by the cutinl class real mean/std (real var = 1); draws with all six descriptors finite",
                  note="six descriptors available per draw, so the trace ratio is used (no IQR fallback)" + note_pool, seeds=seeds_here)
            R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, pool, "variance_ratio_gv", "interaction", np.nan, gv, n_real=46, n_samples=n_fin,
                  source="(det Sigma_draws / det Sigma_cutinl_real)^(1/6)", note=note_pool, seeds=seeds_here)
            for k in DESC:
                R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, pool, f"variance_ratio_{k}", "interaction", np.nan, per_desc[k], n_real=46, n_samples=n_fin,
                      source="var(z_draws)/1", note=note_pool, seeds=seeds_here)
        # validity / background / off-road (single values from the frozen tables)
        R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "valid_rate", "", n_exec, [n_valid_t8 / n_exec], n_real=1, n_samples=n_exec,
              source=("results/final/T8.csv:n_valid / n_executed" if src == "T8" else "count of T8-like valid from results/table5_validity_samples_sakura.csv flags / n"),
              note=f"n_valid={n_valid_t8}, n_executed={n_exec}; valid = {valid_def}; teleport_n={teleport_n}")
        if src != "T8":
            R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "valid_rate_table5_gate", "", n_exec, [float(gate_full.mean())], n_real=1, n_samples=n_exec,
                  source="results/e6_descriptors_sakura_39_180_cond.parquet:valid_gate (== table3_sakura_39_180_cond.csv n_valid_gate)",
                  note="full Table-5 gate incl. bg solid chmed and sampler-invalid (offset/speed clip); much stricter than T8's valid")
        R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "bg_solid_rate", "", n_exec, [bg], n_real=1, n_samples=n_exec, source=bg_src,
              note="T8: analytic OBB solid (>2 frames, depth>=0.1 m) vs all rec00 vehicle tracks within GT support 2424-2923" if src == "T8" else "Table-5 substrate; horizon differs from T8 (chmed vs GT support)")
        if src == "T8":
            R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "valid", "bg_solid_rate", "", n_exec, [bg_valid], n_real=1, n_samples=n_valid_t8, source=bg_valid_src, note="among valid draws")
            R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "bg_solid_rate_label0", "", n_exec, [bg_l0], n_real=1, n_samples=n_exec, source=bg_l0_src, note="vs the 14 label-0 partners")
        R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "offroad_rate", "", n_exec, [offroad_n / n_exec], n_real=1, n_samples=n_exec, source=off_src,
              note=f"offroad_n={offroad_n}" + ("; >5% of points outside the tyms.xodr driving+shoulder union (33_e7 recipe)" if src == "T8" else "; VL-windowed off-road rule"))
        R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "D_m_T8", "", n_exec, [float(t8.loc[arm, "D_m"]) if arm in t8.index else np.nan], n_real=1, n_samples=n_exec,
              source="results/final/T8.csv:D_m (medians over valid scored draws; b_k over the T8 arm set)" if arm in t8.index else "n/a (arm not in T8)",
              note="similarity reference only; not the table2_sakura_arms 5-arm D_m")
        rc_rows.append(dict(arm=arm, label=label, class_data_used=cdu, n_executed=n_exec, n_valid=n_valid_t8, teleport_n=teleport_n, offroad_n=offroad_n, bg_solid=bg,
                            rec_joint_all=float(rec_j.mean()), rec_int_all=float(rec_i.mean()), rec_path_all=float(rec_p.mean()),
                            rec_joint_valid=float(rec_j[valid_primary].mean()) if valid_primary.any() else np.nan, n_int_finite=int(fin.sum())))
    # reconstruction arms of the RareCase table (validity / background only; similarity comes from table2_sakura_arms.csv in stage B)
    for arm in ("svd_matched_fullfit", "svd_matched_logo"):
        r = t8.loc[arm]
        R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "valid_rate", "", 1, [float(r.n_valid) / float(r.n_executed)], n_real=1, n_samples=1,
              source="results/final/T8.csv:n_valid / n_executed", note=f"analytic decode; off-road (offroad_n={int(r.offroad_n)}) => invalid; used_class_data={r.used_class_data}")
        R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "bg_solid_rate", "", 1, [float(r.bg_solid_rate_all_gtsupport)], n_real=1, n_samples=1,
              source="results/final/T8.csv:bg_solid_rate_all_gtsupport", note="single decode")
        R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "offroad_rate", "", 1, [float(r.offroad_n) / float(r.n_executed)], n_real=1, n_samples=1,
              source="results/final/T8.csv:offroad_n / n_executed", note="single decode")
    R.add("rarecase_39_180", "special_39_180", "single_39_180", "svd_d5_singleton_N1", "all", "undefined", "", 0, [np.nan], n_real=1, n_samples=0,
          source="results/final/T8.csv row svd_d5_singleton_N1", note=str(t8.loc["svd_d5_singleton_N1", "definable"]))
    # ours3_disk default + SAKURA plain/route for 39_180 from the Table-5 sample files where present
    t5_main = pd.read_csv(RES / "table5_validity_samples.csv", usecols=["arm", "scenario_uid", "sample_id", "horizon", "scored", "any_solid", "offroad_vl", "teleport", "phys_gate1", "valid_all"], low_memory=False)
    for arm, src_df, src_name in (("ours3_disk", t5_main, "results/table5_validity_samples.csv"), ("sakura_plain", t5sak_smp, "results/table5_validity_samples_sakura.csv"),
                                  ("sakura_route", t5sak_smp, "results/table5_validity_samples_sakura.csv"), ("sakura_plain_extra", t5sak_smp, "results/table5_validity_samples_sakura.csv")):
        s = src_df[src_df.arm.eq(arm) & src_df.scenario_uid.eq(SPECIAL_UID)]
        if not len(s):
            continue
        ch = s[s.horizon.eq("chmed")]; fu = s[s.horizon.eq("full")]
        if len(ch):
            R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "bg_solid_rate", "", len(ch), [float(ch.any_solid.astype(float).mean())], n_real=1, n_samples=len(ch),
                  source=f"{src_name}:any_solid (scenario 39_180, horizon chmed)", note=f"Table-5 substrate, {len(ch)} sample(s) of this scenario; NOT T8's GT-support horizon")
        if len(fu):
            R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "offroad_rate", "", len(fu), [float(fu.offroad_vl.astype(float).mean())], n_real=1, n_samples=len(fu),
                  source=f"{src_name}:offroad_vl (scenario 39_180, horizon full)", note="VL rule")
            R.add("rarecase_39_180", "special_39_180", "single_39_180", arm, "all", "valid_rate_table5_gate", "", len(fu), [float(fu.valid_all.astype(float).mean())], n_real=1, n_samples=len(fu),
                  source=f"{src_name}:valid_all (full)", note="Table-5 valid_all")

    # ── write ──
    gm = pd.DataFrame(R.rows)
    cols = ["table_group", "class", "real_set", "arm", "pool", "metric", "space", "budget", "value", "value_seed_min", "value_seed_max",
            "n_seeds", "n_real", "n_rare", "n_samples", "source", "note", "seeds"]
    gm = gm[cols]
    gm.to_csv(OUT / "gen_metrics.csv", index=False)

    definitions = {
        "z_space": "6-D (signed PET s, d_min m, alpha deg, conflict_x m, conflict_y m, u_c m/s) standardized per class with the mean/std (ddof=1) of the matched489 REAL scenarios of the class having all six descriptors finite (scripts/46 real_scaling; train-frozen). The same z is used for eps, coverage, corner coverage, variance and the 39_180 recovery.",
        "eps": "eps_interaction = median real-to-real nearest-neighbour distance in z (cvlib._eps); eps_path = median real-to-real NN DTW of arc-resampled GT target paths; asserted equal to results/table34_manifest.json",
        "coverage": "scripts/46 hit_matrices/orderings/budget_curve reused as-is: a real scenario is interaction-covered by any finite sample within eps_interaction, path-covered only by samples rendered for it (kernel centre == scenario) with DTW <= eps_path, joint = both on the same sample; coverage@b = fraction of reals covered among the first b draws, mean over 20 orderings (natural + default_rng(60000+o) permutations, o=1..19), per seed, then mean/min/max over seeds. budget == pool size is ordering-invariant. Valid pool = contiguous Table-5-flagged prefix; invalid draws cost budget and cannot cover.",
        "corner_coverage": f"same rule restricted to the rare real subset: rare = top-{int(RARE_FRAC*100)}% most isolated reals by NN distance to the other reals of the same class in the same z-space (k = ceil(0.2 * n_rankable), n_rankable = reals of the real_set with all six descriptors finite; the alternative cut k = ceil(0.2 * n_total) is also reported as corner_coverage_k_of_total). NN for disk469 is computed among the disk469 reals (rank_in_matched489 also listed).",
        "variance_ratio": "per class/arm/seed/pool on int_finite samples: var(z_gen)/var(z_real) per descriptor (ddof=1; real = deduplicated class reals of the real_set; == 1 per descriptor for matched489 by construction), trace ratio = mean of the six, generalized-variance ratio = (det Sigma_gen/det Sigma_real)^(1/6) (nan if singular), per-centre spread = median over kernel centres (>=2 finite samples) of trace var(z) of the centre's own samples (and its ratio to trace Sigma_real). 1.0 = same spread as the recorded class.",
        "rarecase_39_180": "recovered fraction = fraction of draws within eps (cutinl class eps and z-scaling) of the single recorded 39_180 descriptor vector (interaction / path / joint); variance = trace ratio of the draws' z standardized by the cutinl class real std (six descriptors available; no IQR fallback needed); valid_rate = n_valid/n_executed with the exact source column; bg_solid_rate and offroad_rate from T8 (GT-support horizon, 33_e7 union) for the e9 arms and from Table-5 files (chmed horizon, VL rule) for the SAKURA arm — the definitions differ and are recorded per row.",
        "bold_rules_for_stage_B": "similarity/collision/off-road lower is best; coverage/corner higher; variance closest to 1.0; chosen among the generated sub-rows of the same group; real reference never bolded.",
    }
    inputs = [RES / "e6_descriptors.parquet", RES / "e6_descriptors_sakura.parquet", RES / "e6_descriptors_sakura_39_180_cond.parquet", RES / "e6_real_reference.csv",
              RES / "table34_manifest.json", RES / "table3_coverage.csv", RES / "table3_coverage_sakura.csv", RES / "table3_sakura_39_180_cond.csv", RES / "final" / "T3.csv",
              RES / "final" / "T8.csv", E9 / "T8_samples.csv", E9 / "e9_bbox_pet_paired.csv", E9 / "e9_esmini_bg_collisions.csv", RES / "table5_validity_summary_sakura.csv",
              RES / "table5_validity_samples_sakura.csv", RES / "table5_validity_samples.csv", PROJECT / "scripts/46_coverage_similarity.py", M.CV_SOURCE, M.DTW_SOURCE]
    man = dict(script=str(Path(__file__).resolve()), script_sha256=sha256(Path(__file__).resolve()), generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               elapsed_s=round(time.monotonic() - t0, 1), n_orderings=args.n_orderings, ordering_rng="default_rng(60000+o), o=1..19; order 0 = natural draw order (46.orderings)",
               seeds=SEEDS, classes=CLASSES, real_sets=REAL_SETS, arms={a: dict(source=s, label=l, primary=p) for a, (s, l, p) in GEN_ARMS.items()},
               rarecase_arms={a: dict(source=s, label=l, class_data_used=c) for a, (s, l, c) in RARECASE_ARMS.items()},
               eps=eps, real_scaling=scaling, rare_k={f"{c}|{rs}": dict(k=v["k"], n_rankable=v["n_rankable"], n_total=v["n_total"], k_of_total=v["k_of_total"], uids=v["uids"])
                                                        for (c, rs), v in rare_uids.items()},
               reuses="scripts/46_coverage_similarity.py imported via importlib: real_scaling, zscore, cv_eps (cvlib._eps), hit_matrices, orderings, budget_curve, DESC/SPACES/BUDGETS/NORD, DESC_SRC",
               checks=checks, definitions=definitions, rarecase_summary=rc_rows,
               inputs=[dict(path=str(p), sha256=sha256(p)) for p in inputs],
               outputs=[str(OUT / f) for f in ("gen_metrics.csv", "rare_scenes.csv", "reproduction_check_per_seed.csv", "reproduction_check_seed_mean.csv", "gen_metrics_manifest.json")])
    (OUT / "gen_metrics_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)) + "\n")

    # ── compact summary ──
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.max_rows", 400, "display.float_format", lambda v: f"{v:.3f}"):
        c = gm[gm.table_group.eq("coverage") & gm.real_set.eq("matched489") & gm.pool.eq("all") & gm.budget.eq(1000) & gm.space.eq("joint")]
        print("\n== coverage@1000 joint (matched489, pool all; mean [seed min, max]) ==")
        print(c.assign(v=c.apply(lambda r: f"{r.value:.3f} [{r.value_seed_min:.3f}, {r.value_seed_max:.3f}]", axis=1)).pivot(index="class", columns="arm", values="v").to_string())
        cc = gm[gm.table_group.eq("corner_coverage") & gm.real_set.eq("matched489") & gm.pool.eq("all") & gm.metric.eq("corner_coverage") & gm.budget.eq(1000)]
        for space in SPACES:
            s = cc[cc.space.eq(space)]
            print(f"\n== corner coverage@1000 {space} (matched489, pool all, rare k=ceil(0.2 n_rankable)) ==")
            print(s.assign(v=s.apply(lambda r: f"{r.value:.3f} [{r.value_seed_min:.3f}, {r.value_seed_max:.3f}] k={int(r.n_rare)}", axis=1)).pivot(index="class", columns="arm", values="v").to_string())
        v = gm[gm.table_group.eq("variance") & gm.real_set.eq("matched489") & gm.metric.isin(["variance_ratio_trace", "variance_ratio_gv", "per_centre_trace_ratio"])]
        for pool in ("all", "valid"):
            s = v[v.pool.eq(pool)]
            print(f"\n== variance (matched489, pool {pool}; mean [seed min, max]) ==")
            print(s.assign(v=s.apply(lambda r: f"{r.value:.3f} [{r.value_seed_min:.3f}, {r.value_seed_max:.3f}] n={int(r.n_samples)}", axis=1)).pivot_table(index=["class", "metric"], columns="arm", values="v", aggfunc="first").to_string())
        rc = gm[gm.table_group.eq("rarecase_39_180") & gm.metric.isin(["recovered_fraction", "variance_ratio_trace", "valid_rate", "bg_solid_rate", "offroad_rate"])]
        print("\n== RareCase 39_180 ==")
        print(rc[["arm", "pool", "metric", "space", "value", "value_seed_min", "value_seed_max", "n_samples", "source"]].to_string(index=False))
    print(f"\n[done] {len(gm)} metric rows -> {OUT / 'gen_metrics.csv'}; rare scenes {len(rare_df)} rows; {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
