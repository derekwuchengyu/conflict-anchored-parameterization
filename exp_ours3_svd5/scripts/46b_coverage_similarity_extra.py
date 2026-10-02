#!/usr/bin/env python3
"""Tables 3 / 4 for the SAKURA arms (scripts/46_coverage_similarity.py is NOT edited; imported as a module).

Arms added (same real reference, eps, budgets 10/100/1000, 20 orderings, eps x {1,1.5,2}, 3 seeds as 46):
  sakura_route_kde        dependent sampler, executed (runs/sakura_route_kde_s<seed>, 1000 per class per seed)
  sakura_route_defaults   nominal, 1 per scene (routed where a route exists, plain chord otherwise)
  sakura_plain            nominal, 1 per scene = the 192 sakura_bc + sakura_plain_extra (cutinr 20)
  sakura_route_kde_cond_39_180   singleton conditional arm: recovery of the single real scene 39_180
                          (cutinl scaling / eps), reported separately (table3_sakura_39_180_cond.csv)
Real reference = 46.load_real() (matched489 / disk469), z-scaling = 46.real_scaling, eps = 46's rule
(asserted equal to results/table34_manifest.json), coverage machinery = 46.hit_matrices / budget_curve /
orderings, DTW = 46._dtw_task (verbatim traj_dtw) with a separate cache, validity gate = 46.join_table5 on
results/table5_validity_samples_sakura.csv (45b), similarity = the 46 Table 4 loop (W1, POT joint W1,
variance / IQR ratio, out-of-support, count-matched bracket, per-centre span).
Outputs: results/table3_coverage_sakura.csv, table3_first_hit_sakura.csv, table3_variation_sakura.csv,
table3_budget_curve_points_sakura.csv, table4_similarity_sakura.csv, table3_sakura_39_180_cond.csv,
e6_descriptors_sakura.parquet, table34_sakura_manifest.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
_spec = importlib.util.spec_from_file_location("cov46", PROJECT / "scripts/46_coverage_similarity.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["cov46"] = M          # registered so the ProcessPool can pickle M._dtw_task
_spec.loader.exec_module(M)
DESC, SPACES, BUDGETS, EMULT, MATCH_N, REAL_SETS, UNITS = M.DESC, M.SPACES, M.BUDGETS, M.EMULT, M.MATCH_N, M.REAL_SETS, M.UNITS
CLASSES = ["keeptl", "keeptl_sw", "cutinl", "cutinr"]
SEEDS = [20260910, 20260911, 20260912]
SPECIAL_UID, SPECIAL_CLS = "HetroD/00/39_180/2424-3002", "special_39_180"
NEW_ARMS = {
    "sakura_route_kde": dict(label="SAKURA-route + KDE (offset, v_avg) (executed)", kind="dependent", seeded=True),
    "sakura_route_defaults": dict(label="SAKURA-route defaults nominal (1 per scene)", kind="nominal", seeded=False),
    "sakura_plain": dict(label="SAKURA plain chord nominal (192 sakura_bc + extra cutinr)", kind="nominal", seeded=False),
    "sakura_route_kde_cond_39_180": dict(label="39_180 class-borrowed conditional SAKURA-route KDE (cutinl bandwidth)", kind="dependent", seeded=True),
}
M.ARMS.update(NEW_ARMS)
M.TABLE5_ARM.update({"sakura_route_kde": "sakura_route_kde", "sakura_route_defaults": "sakura_route", "sakura_plain": "sakura_plain",
                     "sakura_route_kde_cond_39_180": "sakura_route_kde_cond_39_180"})
M.TABLE5 = RES / "table5_validity_samples_sakura.csv"


def kde_meta(paired):
    meta = []
    for sj in paired.sample_json:
        s = json.loads(Path(sj).read_text())
        sc = s["context"]["source_context"]
        req, app = sc["sampler_parameters_requested"], sc["sampler_parameters_applied"]
        meta.append(dict(seed=int(sc["seed"]), draw=int(sc["draw"]), center=sc["kernel_center_scenario_uid"], h=float(sc["h"]),
                         offset_clipped=bool(sc["offset_clipped"]), speed_clipped=bool(sc["speed_clipped"]),
                         req_offset=req["offset_m"], req_v=req["v_avg_kmh"], app_offset=app["offset_m"], app_v=app["v_avg_kmh"],
                         route_found=bool(sc.get("route_found", False)), upstream_status=s.get("status"), run_id=s.get("run_id")))
    return pd.DataFrame(meta)


def load_kde_arm(arm, tag_of_seed, seeds):
    frames = []
    for seed in seeds:
        paired_path = RES / f"sak_{tag_of_seed(seed)}_nonpet_paired.csv"
        pet_path = RES / f"bbox_pet_sak_{tag_of_seed(seed)}_paired.csv"
        paired = pd.read_csv(paired_path, low_memory=False)
        pet = pd.read_csv(pet_path, low_memory=False)
        m = kde_meta(paired)
        assert (m.center.to_numpy() == paired.scenario_uid.to_numpy()).all(), f"{arm}: centre differs from the rendered scenario"
        inv = (m.offset_clipped | m.speed_clipped).to_numpy()
        reason = np.where(m.offset_clipped & m.speed_clipped, "offset_clipped+speed_clipped", np.where(m.offset_clipped, "offset_clipped", np.where(m.speed_clipped, "speed_clipped", "")))
        frames.append(M.base_rows(arm, paired, pet, m.seed, m.draw, m.center,
                                  dict(sampler_invalid=inv, sampler_invalid_reason=reason, kde_h=m.h.to_numpy(),
                                       requested_offset_m=m.req_offset.to_numpy(), requested_v_avg_kmh=m.req_v.to_numpy(),
                                       applied_offset_m=m.app_offset.to_numpy(), applied_v_avg_kmh=m.app_v.to_numpy(),
                                       offset_clipped=m.offset_clipped.to_numpy(), speed_clipped=m.speed_clipped.to_numpy(),
                                       route_found=m.route_found.to_numpy(), upstream_status=m.upstream_status.to_numpy(),
                                       run_id=m.run_id.to_numpy(), input_paired_path=str(paired_path))))
    out = pd.concat(frames, ignore_index=True)
    assert out.sample_id.is_unique, arm
    return out


def compute_dtw_cached(df, workers, cache_path):
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["arm", "sample_id", "dtw", "dtw_coverage", "n_gen", "trajectory_path"])
    key = set(zip(cache.arm, cache.sample_id))
    todo = df[df.trajectory_path.astype(str).ne("") & df.dtw.isna() & ~pd.Series(list(zip(df.arm, df.sample_id)), index=df.index).isin(key)]
    tasks = [((r.arm, r.sample_id), r.trajectory_path, r.metadata_min_frame, r.metadata_max_frame, r.source_tracks, r.scenario_id) for r in todo.itertuples()]
    print(f"[dtw] {len(tasks)} samples to compute ({len(cache)} cached)", flush=True)
    if tasks:
        new = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, d, cov, ng in ex.map(M._dtw_task, tasks, chunksize=64):
                new.append(dict(arm=k[0], sample_id=k[1], dtw=d, dtw_coverage=cov, n_gen=ng))
        newdf = pd.DataFrame(new)
        newdf["trajectory_path"] = [t[1] for t in tasks]
        cache = pd.concat([cache, newdf], ignore_index=True).drop_duplicates(["arm", "sample_id"], keep="last")
        cache.to_csv(cache_path, index=False)
    c = cache.set_index(["arm", "sample_id"])
    idx = pd.MultiIndex.from_arrays([df.arm, df.sample_id])
    hit = idx.isin(c.index)
    df.loc[hit, "dtw"] = c.dtw.reindex(idx[hit]).to_numpy()
    df.loc[hit, "dtw_coverage"] = c.dtw_coverage.reindex(idx[hit]).to_numpy()
    df.loc[hit, "dtw_source"] = "traj_dtw(180_trajdtw_aggregate.py) on the saved trace"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-orderings", type=int, default=M.NORD)
    args = ap.parse_args()
    t_start = time.monotonic()
    gt = M._GT
    real = M.load_real()
    scaling = M.real_scaling(real)
    Zr = M.zscore(real, scaling)
    for j, k in enumerate(DESC):
        real[f"z_{k}"] = Zr[:, j]
    real_nn = M.real_path_eps(real, gt, args.workers)
    real = real.merge(real_nn[["scenario_uid", "nn_dtw_m"]], on="scenario_uid", how="left")
    eps = {}
    for cls, g in real.groupby("subset"):
        gi = g[g.int_finite]
        eps[cls] = dict(interaction=M.cv_eps(gi[[f"z_{k}" for k in DESC]].to_numpy(float)), path=float(np.nanmedian(g.nn_dtw_m)),
                        n_real_int=int(len(gi)), n_real=int(len(g)), n_real_pet_nonfinite=int((~g.pet_finite).sum()))
    ref_eps = json.loads((RES / "table34_manifest.json").read_text())["eps"]
    for cls in eps:
        assert abs(eps[cls]["interaction"] - ref_eps[cls]["interaction"]) < 1e-9 and abs(eps[cls]["path"] - ref_eps[cls]["path"]) < 1e-9, cls
    print("[eps] identical to table34_manifest.json", flush=True)

    arms = {}
    arms["sakura_route_kde"] = load_kde_arm("sakura_route_kde", lambda s: f"sakura_route_kde_s{s}", SEEDS)
    pet_route = pd.read_csv(RES / "bbox_pet_sak_sakura_route_paired.csv", low_memory=False)
    arms["sakura_route_defaults"] = M.load_nominal("sakura_route_defaults", RES / "sak_sakura_route_nonpet_paired.csv", pet_route)
    pet_all = pd.read_csv(RES / "bbox_pet_all_methods_paired.csv", low_memory=False)
    bc = M.load_nominal("sakura_plain", RES / "sakura_population_nonpet_paired.csv", pet_all[pet_all.method.eq("sakura_bc")])
    pet_extra = pd.read_csv(RES / "bbox_pet_sak_sakura_plain_extra_paired.csv", low_memory=False)
    extra = M.load_nominal("sakura_plain", RES / "sak_sakura_plain_extra_nonpet_paired.csv", pet_extra)
    bc["source_rows"], extra["source_rows"] = "sakura_bc (existing 192)", "sakura_plain_extra (85)"
    dup = extra.set_index(["subset", "scenario_uid"]).index.isin(bc.set_index(["subset", "scenario_uid"]).index)
    print(f"[sakura_plain] extra rows already among the 192 sakura_bc scenes (dropped): {extra[dup].sample_id.tolist()}", flush=True)
    arms["sakura_plain"] = pd.concat([bc, extra[~dup]], ignore_index=True)
    cond = load_kde_arm("sakura_route_kde_cond_39_180", lambda s: "sakura_route_kde_cond_39_180", [None])
    df = pd.concat(list(arms.values()) + [cond], ignore_index=True)
    # the special scene is not a class of the real set: keep only class rows for Tables 3/4; cond + special copies handled below
    special_rows = df[df.subset.eq(SPECIAL_CLS)].copy()
    df = df[~df.subset.eq(SPECIAL_CLS)].reset_index(drop=True)
    for c in ("dtw", "dtw_coverage"):
        df[c] = np.nan
        special_rows[c] = np.nan
    df["dtw_source"], special_rows["dtw_source"] = "", ""
    df = compute_dtw_cached(df, args.workers, RES / "e6_dtw_cache_sakura.csv")
    special_rows = compute_dtw_cached(special_rows.reset_index(drop=True), args.workers, RES / "e6_dtw_cache_sakura.csv")
    df = M.join_table5(df)
    special_rows = M.join_table5(special_rows)
    rr = real.set_index("scenario_uid")
    for k in DESC:
        df[f"real_{k}"] = rr[k].reindex(df.kernel_center_uid).to_numpy()
    df["centre_in_matched489"] = df.kernel_center_uid.isin(rr.index)
    df["centre_in_disk469"] = df.kernel_center_uid.map(rr.in_disk469).fillna(False).astype(bool)
    Z = M.zscore(df, scaling)
    for j, k in enumerate(DESC):
        df[f"z_{k}"] = Z[:, j]
    df["int_finite"] = np.isfinite(Z).all(axis=1)
    df["eps_interaction"] = df.subset.map(lambda c: eps[c]["interaction"])
    df["eps_path"] = df.subset.map(lambda c: eps[c]["path"])
    df["dist_to_centre_interaction"] = np.sqrt(sum((df[f"z_{k}"] - (df[f"real_{k}"] - df.subset.map(lambda c, k=k: scaling[c]["mean"][k])) / df.subset.map(lambda c, k=k: scaling[c]["sd"][k])) ** 2 for k in DESC))
    df["recovered_interaction"] = df.dist_to_centre_interaction <= df.eps_interaction
    df["recovered_path"] = df.dtw <= df.eps_path
    df["recovered_joint"] = df.recovered_interaction & df.recovered_path
    df.to_parquet(RES / "e6_descriptors_sakura.parquet", index=False)
    print(f"[descriptors] {len(df)} rows; per arm: {df.arm.value_counts().to_dict()}", flush=True)

    ARMS = {a: NEW_ARMS[a] for a in ("sakura_route_kde", "sakura_route_defaults", "sakura_plain")}
    cov_rows, curve_rows, fh_rows = [], [], []
    for cls in CLASSES:
        for real_set in REAL_SETS:
            Rall = real[real.subset.eq(cls) & (real.in_disk469 if real_set == "disk469" else real.in_matched489)]
            Rsets = dict(interaction=Rall[Rall.int_finite], path=Rall, joint=Rall[Rall.int_finite])
            for arm, spec in ARMS.items():
                A = df[df.arm.eq(arm) & df.subset.eq(cls)]
                if not len(A):
                    continue
                seeds = sorted(A.seed.unique()) if spec["seeded"] else [0]
                for seed in seeds:
                    Gfull = A[A.seed.eq(seed)].sort_values("draw").reset_index(drop=True)
                    for pool in ("all", "valid"):
                        G = Gfull
                        if pool == "valid":
                            flagged = G.table5_available.to_numpy()
                            n_prefix = int(np.argmin(flagged)) if not flagged.all() else len(flagged)
                            if n_prefix == 0:
                                continue
                            G = Gfull.iloc[:n_prefix].reset_index(drop=True)
                        Mn = len(G)
                        valid_mask = np.array([bool(v) if pd.notna(v) else False for v in G.valid_gate.to_numpy()])
                        vm = None if pool == "all" else valid_mask
                        for m in EMULT:
                            H_all, _D = M.hit_matrices(G, Rall, eps[cls]["interaction"], eps[cls]["path"], m, vm)
                            for space in SPACES:
                                Rs = Rsets[space]
                                sel = Rall.scenario_uid.isin(Rs.scenario_uid).to_numpy()
                                Hm = H_all[space][sel]
                                nR = int(sel.sum())
                                if nR == 0:
                                    continue
                                if spec["kind"] == "dependent":
                                    ords = M.orderings(Mn, args.n_orderings)
                                    curve = M.budget_curve(Hm, ords)
                                    for b in sorted(set(BUDGETS + [Mn])):
                                        if b <= Mn:
                                            c = curve[:, b - 1]
                                            cov_rows.append(dict(subset=cls, real_set=real_set, arm=arm, arm_label=spec["label"], seed=seed, pool=pool, eps_mult=m,
                                                                 space=space, budget=b, budget_is_pool_size=b == Mn, n_real=nR, n_samples=Mn,
                                                                 n_samples_valid=int(valid_mask.sum()), n_samples_flagged=int(G.table5_available.sum()),
                                                                 coverage_mean=float(c.mean()), coverage_min=float(c.min()), coverage_max=float(c.max()),
                                                                 coverage_natural_order=float(c[0]), n_orderings=len(ords)))
                                    if m == 1.0 and real_set == "matched489":
                                        for b in sorted(set([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, Mn])):
                                            if b <= Mn:
                                                curve_rows.append(dict(subset=cls, arm=arm, seed=seed, pool=pool, space=space, budget=b,
                                                                       coverage_mean=float(curve[:, b - 1].mean()), coverage_min=float(curve[:, b - 1].min()),
                                                                       coverage_max=float(curve[:, b - 1].max())))
                                    if m == 1.0:
                                        fh = np.full(Hm.shape[0], np.inf)
                                        for i in range(Hm.shape[0]):
                                            j = np.flatnonzero(Hm[i])
                                            if len(j):
                                                fh[i] = j[0] + 1
                                        fin = np.isfinite(fh)
                                        fh_rows.append(dict(subset=cls, real_set=real_set, arm=arm, arm_label=spec["label"], seed=seed, pool=pool, space=space,
                                                            n_real=nR, pool_size=Mn, n_hit=int(fin.sum()), censored_fraction=float(1 - fin.mean()) if nR else np.nan,
                                                            first_hit_median_all_censored=float(np.median(fh)) if nR else np.nan,
                                                            first_hit_median_hits_only=float(np.median(fh[fin])) if fin.any() else np.inf,
                                                            first_hit_p90_hits_only=float(np.percentile(fh[fin], 90)) if fin.any() else np.inf,
                                                            hit_at_1_fraction=float((fh == 1).mean()) if nR else np.nan))
                                else:
                                    c = float(Hm.any(axis=1).mean()) if nR else np.nan
                                    cov_rows.append(dict(subset=cls, real_set=real_set, arm=arm, arm_label=spec["label"], seed=seed, pool=pool, eps_mult=m,
                                                         space=space, budget=Mn, budget_is_pool_size=True, n_real=nR, n_samples=Mn,
                                                         n_samples_valid=int(valid_mask.sum()), n_samples_flagged=int(G.table5_available.sum()),
                                                         coverage_mean=c, coverage_min=c, coverage_max=c, coverage_natural_order=c, n_orderings=0))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(RES / "table3_coverage_sakura.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(RES / "table3_budget_curve_points_sakura.csv", index=False)
    fh_df = pd.DataFrame(fh_rows)
    fh_df.to_csv(RES / "table3_first_hit_sakura.csv", index=False)

    var_rows = []
    for cls in CLASSES:
        for arm, spec in ARMS.items():
            A = df[df.arm.eq(arm) & df.subset.eq(cls)]
            if not len(A) or spec["kind"] != "dependent":
                continue
            for scope, sub in [("all_seeds", A)] + [(f"seed_{s}", A[A.seed.eq(s)]) for s in sorted(A.seed.unique())]:
                g = sub.groupby("kernel_center_uid")
                counts = g.size()
                multi = counts[counts >= 2].index
                row = dict(subset=cls, arm=arm, arm_label=spec["label"], scope=scope, n_samples=len(sub), n_centres=int(counts.size),
                           n_centres_ge2=int(len(multi)), samples_per_centre_median=float(counts.median()),
                           samples_per_centre_min=int(counts.min()), samples_per_centre_max=int(counts.max()),
                           sampler_invalid_share=float(sub.sampler_invalid.mean()))
                for k in DESC + ["dtw"]:
                    iq = g[k].agg(lambda v: (lambda f: np.subtract(*np.percentile(f, [75, 25])) if len(f) >= 2 else np.nan)(v[np.isfinite(v.astype(float))].astype(float)))
                    row[f"iqr_median_{k}"] = float(np.nanmedian(iq.loc[multi])) if len(multi) else np.nan
                    sp = g[k].agg(lambda v: (lambda f: f.max() - f.min() if len(f) >= 2 else np.nan)(v[np.isfinite(v.astype(float))].astype(float)))
                    row[f"span_median_{k}"] = float(np.nanmedian(sp.loc[multi])) if len(multi) else np.nan
                for space in SPACES:
                    rec = g[f"recovered_{space}"].mean()
                    row[f"recovery_{space}_median_over_centres"] = float(rec.median())
                    row[f"recovery_{space}_pooled"] = float(sub[f"recovered_{space}"].mean())
                    row[f"centres_with_any_recovery_{space}"] = float(g[f"recovered_{space}"].any().mean())
                var_rows.append(row)
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(RES / "table3_variation_sakura.csv", index=False)

    import ot
    sim_rows = []
    rng = np.random.default_rng(20260910)
    for cls in CLASSES:
        R = real[real.subset.eq(cls)]
        for arm, spec in ARMS.items():
            A = df[df.arm.eq(arm) & df.subset.eq(cls)]
            if not len(A):
                continue
            scopes = [("all_seeds", A)] + ([(f"seed_{s}", A[A.seed.eq(s)]) for s in sorted(A.seed.unique())] if spec["seeded"] and A.seed.nunique() > 1 else [])
            if spec["kind"] == "dependent" and A.sampler_invalid.any():
                scopes.append(("all_seeds_sampler_valid", A[~A.sampler_invalid.astype(bool)]))
            for scope, sub in scopes:
                for k in DESC + ["dtw"]:
                    rv = R[k].astype(float).to_numpy() if k != "dtw" else np.zeros(len(R))
                    rv = rv[np.isfinite(rv)]
                    gv = sub[k].astype(float).to_numpy()
                    gv = gv[np.isfinite(gv)]
                    d = dict(subset=cls, arm=arm, arm_label=spec["label"], scope=scope, descriptor=k, unit=UNITS[k],
                             n_real=len(rv), n_arm_finite=len(gv), n_arm=len(sub), n_arm_nonfinite=int(len(sub) - len(gv)),
                             sampler_invalid_share=float(sub.sampler_invalid.mean()), real_median=float(np.median(rv)) if len(rv) else np.nan,
                             arm_median=float(np.median(gv)) if len(gv) else np.nan)
                    if k == "dtw":
                        d.update(w1=float(np.mean(gv)) if len(gv) else np.nan, variance_ratio=np.nan, iqr_ratio=np.nan,
                                 out_of_real_support=np.nan, real_range_covered=np.nan, arm_p90=float(np.percentile(gv, 90)) if len(gv) else np.nan)
                    elif len(rv) >= 2 and len(gv) >= 2:
                        lo, hi = rv.min(), rv.max()
                        d.update(w1=float(wasserstein_distance(rv, gv)), variance_ratio=float(np.var(gv, ddof=1) / max(np.var(rv, ddof=1), 1e-12)),
                                 iqr_ratio=float((np.percentile(gv, 75) - np.percentile(gv, 25)) / max(np.percentile(rv, 75) - np.percentile(rv, 25), 1e-9)),
                                 out_of_real_support=float(np.mean((gv < lo) | (gv > hi))),
                                 real_range_covered=float(np.mean((rv >= gv.min()) & (rv <= gv.max()))))
                    else:
                        d.update(w1=np.nan, variance_ratio=np.nan, iqr_ratio=np.nan, out_of_real_support=np.nan, real_range_covered=np.nan)
                    if spec["kind"] == "dependent" and k != "dtw":
                        rr_ = R.set_index("scenario_uid")[k].astype(float)
                        for mn in [None] + MATCH_N:
                            hits = tot = 0
                            for ck, gg in sub.groupby("kernel_center_uid"):
                                if ck not in rr_.index or not np.isfinite(rr_[ck]):
                                    continue
                                v = gg[k].astype(float).to_numpy()
                                v = v[np.isfinite(v)]
                                if mn is not None and len(v) > mn:
                                    v = rng.choice(v, mn, replace=False)
                                if len(v) < 2:
                                    continue
                                tot += 1
                                hits += int(v.min() <= rr_[ck] <= v.max())
                            tag = "all" if mn is None else f"match{mn}"
                            d[f"bracket_rate_{tag}"] = hits / tot if tot else np.nan
                            d[f"bracket_n_centres_{tag}"] = tot
                        sp = sub.groupby("kernel_center_uid")[k].agg(lambda v: (lambda f: f.max() - f.min() if len(f) >= 2 else np.nan)(v[np.isfinite(v.astype(float))].astype(float)))
                        d["per_centre_span_median"] = float(np.nanmedian(sp)) if sp.notna().any() else np.nan
                    sim_rows.append(d)
                ZR = R[R.int_finite][[f"z_{k}" for k in DESC]].to_numpy(float)
                ZG = sub[sub.int_finite][[f"z_{k}" for k in DESC]].to_numpy(float)
                if len(ZR) >= 2 and len(ZG) >= 2:
                    Mx = ot.dist(ZR, ZG, metric="euclidean")
                    w1j = float(ot.emd2(np.full(len(ZR), 1 / len(ZR)), np.full(len(ZG), 1 / len(ZG)), Mx, numItermax=1_000_000))
                else:
                    w1j = np.nan
                sim_rows.append(dict(subset=cls, arm=arm, arm_label=spec["label"], scope=scope, descriptor="joint6_zscored", unit="z",
                                     n_real=len(ZR), n_arm_finite=len(ZG), n_arm=len(sub), n_arm_nonfinite=int(len(sub) - len(ZG)),
                                     sampler_invalid_share=float(sub.sampler_invalid.mean()), w1=w1j))
    sim = pd.DataFrame(sim_rows)
    seeded = sim[sim.scope.str.startswith("seed_")]
    if len(seeded):
        sp = seeded.groupby(["subset", "arm", "descriptor"]).w1.agg(["min", "max", "std"]).rename(columns=dict(min="w1_seed_min", max="w1_seed_max", std="w1_seed_std"))
        sim = sim.merge(sp.reset_index(), on=["subset", "arm", "descriptor"], how="left")
    sim.to_csv(RES / "table4_similarity_sakura.csv", index=False)

    # singleton 39_180: recovery of its own real descriptors by the class-borrowed conditional arm (cutinl scaling / eps)
    if SPECIAL_UID in rr.index:
        cond_status = "recomputed"
        cond_rows = []
        sp = special_rows[special_rows.arm.eq("sakura_route_kde_cond_39_180")].copy()
        r0 = rr.loc[SPECIAL_UID]
        sc_ = scaling["cutinl"]
        zc = np.column_stack([(sp[k].astype(float).to_numpy() - sc_["mean"][k]) / sc_["sd"][k] for k in DESC])
        z0 = np.array([(float(r0[k]) - sc_["mean"][k]) / sc_["sd"][k] for k in DESC])
        sp["dist_to_centre_interaction"] = np.sqrt(((zc - z0) ** 2).sum(axis=1))
        sp["recovered_interaction"] = sp.dist_to_centre_interaction <= eps["cutinl"]["interaction"]
        sp["recovered_path"] = sp.dtw <= eps["cutinl"]["path"]
        sp["recovered_joint"] = sp.recovered_interaction & sp.recovered_path
        for scope, sub in [("all_seeds", sp)] + [(f"seed_{s}", sp[sp.seed.eq(s)]) for s in sorted(sp.seed.unique())]:
            vg = sub.valid_gate.map(lambda v: bool(v) if pd.notna(v) else False)
            row = dict(arm="sakura_route_kde_cond_39_180", class_data_used="bandwidth (cutinl fit)", scope=scope, n=len(sub),
                       n_valid_gate=int(vg.sum()), sampler_invalid_share=float(sub.sampler_invalid.mean()),
                       n_pet_nonfinite=int((~np.isfinite(sub.pet.astype(float))).sum()),
                       eps_interaction=eps["cutinl"]["interaction"], eps_path=eps["cutinl"]["path"])
            for space in SPACES:
                row[f"recovered_{space}_rate"] = float(sub[f"recovered_{space}"].mean())
                row[f"recovered_{space}_rate_valid_only"] = float((sub[f"recovered_{space}"] & vg).mean())
                row[f"any_recovered_{space}"] = bool(sub[f"recovered_{space}"].any())
            for k in DESC + ["dtw"]:
                v = sub[k].astype(float)
                v = v[np.isfinite(v)]
                row[f"{k}_real"] = float(r0[k]) if k != "dtw" else 0.0
                row[f"{k}_median"] = float(v.median()) if len(v) else np.nan
                row[f"{k}_iqr"] = float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) >= 2 else np.nan
                row[f"{k}_bracket"] = bool(len(v) >= 2 and v.min() <= row[f"{k}_real"] <= v.max()) if k != "dtw" else None
            cond_rows.append(row)
        cond_df = pd.DataFrame(cond_rows)
        cond_df.to_csv(RES / "table3_sakura_39_180_cond.csv", index=False)
        sp.to_parquet(RES / "e6_descriptors_sakura_39_180_cond.parquet", index=False)
    else:
        # v3 rerun 2026-09-14: 39_180 left the cutinl real reference (relabelled). The rarecase / Table III pipeline is
        # frozen by user decision, so table3_sakura_39_180_cond.csv and e6_descriptors_sakura_39_180_cond.parquet keep
        # their pre-rerun values (computed with the pre-rerun cutinl scaling / eps) and are not rewritten here.
        cond_status = "frozen (pre-rerun values kept; 39_180 not in the v3 cutinl real reference)"
        cond_df = pd.read_csv(RES / "table3_sakura_39_180_cond.csv")
        print(f"[39_180 cond] {cond_status}", flush=True)

    man = dict(script=str(Path(__file__)), generated=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=time.monotonic() - t_start,
               reuses="scripts/46_coverage_similarity.py (module import): load_real, real_scaling, zscore, real_path_eps, base_rows, load_nominal, hit_matrices, budget_curve, orderings, join_table5, _dtw_task",
               eps=eps, eps_identical_to_table34_manifest=True, budgets=BUDGETS, eps_multipliers=EMULT, n_orderings=args.n_orderings,
               seeds=SEEDS, real_sets=REAL_SETS, table5_source=str(M.TABLE5),
               counts=df.groupby(["arm", "subset"]).size().unstack(fill_value=0).to_dict(),
               sources=[dict(path=str(p), sha256=M.sha256(p)) for p in
                        [RES / "table34_manifest.json", RES / "table5_validity_samples_sakura.csv", RES / "sakura_population_nonpet_paired.csv",
                         RES / "bbox_pet_all_methods_paired.csv", RES / "sak_sakura_route_nonpet_paired.csv", RES / "bbox_pet_sak_sakura_route_paired.csv",
                         RES / "sak_sakura_plain_extra_nonpet_paired.csv", RES / "bbox_pet_sak_sakura_plain_extra_paired.csv"]
                        + [RES / f"sak_sakura_route_kde_s{s}_nonpet_paired.csv" for s in SEEDS] + [RES / f"bbox_pet_sak_sakura_route_kde_s{s}_paired.csv" for s in SEEDS]
                        if Path(p).exists()],
               outputs=[str(RES / o) for o in ["table3_coverage_sakura.csv", "table3_first_hit_sakura.csv", "table3_variation_sakura.csv",
                                               "table3_budget_curve_points_sakura.csv", "table4_similarity_sakura.csv", "table3_sakura_39_180_cond.csv",
                                               "e6_descriptors_sakura.parquet", "e6_dtw_cache_sakura.csv"]])
    M.write_json(RES / "table34_sakura_manifest.json", man)
    c = cov[cov.real_set.eq("matched489") & cov.eps_mult.eq(1.0) & cov.pool.eq("all") & cov.budget.isin([100, 1000])]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.3f}"):
        print(c.groupby(["subset", "arm", "space", "budget"]).coverage_mean.mean().unstack("budget").to_string())
        print(cond_df[["scope", "n", "n_valid_gate", "recovered_interaction_rate", "recovered_path_rate", "recovered_joint_rate"]].to_string(index=False))
    print(f"[done] {time.monotonic() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
