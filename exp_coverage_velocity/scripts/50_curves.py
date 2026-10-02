"""Coverage velocity tables from the three sample sets (v2, post-review).

Metrics (frozen ruler, cvlib):
  path_w100   alpha-weighted 100-D positions (upstream frac_covered), eps = median real NN
  path_w101   + duration (Eq. 21)
  arc         GEOMETRY ONLY: arc-length-resampled 50-pt path, rms metres per point,
              eps_arc = median real NN  (separates shape from timing/overrun)
  interaction z-scored {pet, min_dist, conflict_angle, agent_arr_speed}, eps = median real NN
  path_both   path_w100 AND arc on the SAME sample (timing AND geometry) -- the
              strict path coverage; w100 alone is blind to geometry when the
              agent dwells, arc alone is blind to timing
  joint       path_w100 AND interaction on the SAME sample
Pools: 'all' = every ok sample; 'valid' = additionally not teleported, |a_lat|<=5,
       v<=25 m/s, duration not clamped.  'valid' is the primary pool.
Tolerance sweep: eps_mult in {1, 1.5, 2} (eps is data-derived and knife-edge).
Arms: ref (zero-jitter own params), cond (E-A: centre forced, 1000 draws),
      uncond (E-B: dependent sampling, 10000 draws; 20 random orderings for IQR).
Unconditional samples whose centre was excluded from the KDE fit (kde_fit.json)
are dropped (sakura keeptl: the 2157_2038 spawn-artifact offset).
SVD ceilings: in-sample rank-3 reachable fraction AND leave-one-out (basis refit
without the scenario) reachable fraction.

Output: results/real_interaction.csv, eps.json, first_hit_per_scenario.csv,
        table_first_hit.csv, table_budget_curve.csv, table_final.csv,
        validity.csv, corner_running_min.csv
Usage: micromamba run -n nps python 50_curves.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import cvrender as R  # noqa: E402
from hetero_param.sweep import core as SWC  # noqa: E402

T = L.load()
IKEYS = ["pet", "min_dist", "conflict_angle", "agent_arr_speed"]
METRICS = ["path_w100", "path_w101", "arc", "path_both", "interaction", "joint"]
POOLS = ("all", "valid")
EMULT = (1.0, 1.5, 2.0)
NORD = 20
kde_fit = json.load(open(L.RESULTS / "kde_fit.json"))

# ── real references ──────────────────────────────────────────────────────────
rt = pd.read_parquet(T.ddir / "real_tracks.parquet")
rows, arcs = [], []
for sid in T.keys:
    a = rt[(rt.scenario_id == sid) & (rt.role == "actor")].sort_values("frame")
    e = rt[(rt.scenario_id == sid) & (rt.role == "ego")].sort_values("frame")
    agent = SWC._derived_kinematics_copy(R._traj(a.frame.values, a.x.values, a.y.values, a.heading_deg.values,
                                                 a.speed.values, a.length.iloc[0], a.width.iloc[0]))
    ego = R._traj(e.frame.values, e.x.values, e.y.values, e.heading_deg.values, e.speed.values,
                  e.length.iloc[0], e.width.iloc[0])
    rows.append(dict(scenario_id=sid, **R.describe(agent, ego)))
    arcs.append(L.arc_resample(a.x.values, a.y.values, L.NT).reshape(-1))
real_i = pd.DataFrame(rows).set_index("scenario_id").loc[T.keys]
real_i.to_csv(L.RESULTS / "real_interaction.csv")
ZI = real_i[IKEYS].values.astype(float)
RF = np.isfinite(ZI).all(1)
IM, ISD = ZI[RF].mean(0), ZI[RF].std(0, ddof=1)
Zr = (ZI - IM) / ISD
Zr[~RF] = np.nan
ARC_R = np.asarray(arcs) / np.sqrt(L.NT)            # rms-per-point metres
eps_int, eps_arc = L._eps(Zr[RF]), L._eps(ARC_R)
print(f"[real] eps interaction {eps_int:.4f} ({RF.sum()}/{T.N} finite; non-finite {[T.keys[j] for j in np.flatnonzero(~RF)]}), "
      f"eps arc {eps_arc:.3f} m rms/pt, eps w100 {T.eps['w100']:.4f}, w101 {T.eps['w101']:.4f}")
REAL = {"path_w100": T.w(T.X, "w100"), "path_w101": T.w(T.X, "w101"), "arc": ARC_R, "interaction": Zr}
EPS = {"path_w100": T.eps["w100"], "path_w101": T.eps["w101"], "arc": eps_arc, "interaction": eps_int}
VALID_REAL = {k: (RF if k in ("interaction", "joint") else np.ones(T.N, bool)) for k in METRICS}
json.dump({**EPS, "interaction_mean": IM.tolist(), "interaction_sd": ISD.tolist(), "keys": IKEYS,
           "n_real_interaction": int(RF.sum())}, open(L.RESULTS / "eps.json", "w"), indent=2)


def feats(df):
    X = np.stack(df.vector.values).astype(float)
    Z = (df[IKEYS].values.astype(float) - IM) / ISD
    Z[~np.isfinite(Z).all(1)] = np.nan
    return {"path_w100": T.w(X, "w100"), "path_w101": T.w(X, "w101"),
            "arc": np.stack(df.arc.values).astype(float) / np.sqrt(L.NT), "interaction": Z}


def dists(F, i):
    """(M,) distance of every sample to real i per metric, in eps units."""
    out = {}
    for k in ("path_w100", "path_w101", "arc", "interaction"):
        out[k] = np.linalg.norm(F[k] - REAL[k][i], axis=1) / EPS[k]
    out["path_both"] = np.maximum(out["path_w100"], out["arc"])        # timing AND geometry
    out["joint"] = np.maximum(out["path_w100"], out["interaction"])
    return {k: np.nan_to_num(v, nan=np.inf) for k, v in out.items()}


def sample_valid(df):
    X = np.stack(df.vector.values).astype(float)
    k = L.kinematics(T, X)
    v = (k.v_max.values <= 25.0) & (k.alat_max.values <= 5.0)
    if "dur_raw" in df:
        v &= df.dur_raw.values > 0.5
    if "teleport" in df:
        v &= ~df.teleport.fillna(False).astype(bool).values
    return v, k


# ── load sets ────────────────────────────────────────────────────────────────
sets, val = {}, []
for m in ("svd_kde", "ours", "sakura"):
    f = L.RESULTS / f"renders_{m}.parquet"
    if not f.exists():
        print(f"[skip] {f.name} missing")
        continue
    df = pd.read_parquet(f)
    df = df[df.ok.astype(bool)].copy()
    excl = set(kde_fit.get(m, {}).get("excluded", []))
    df = df.reset_index(drop=True)
    v, kin = sample_valid(df)
    if excl:                                   # artifact-centre draws cost budget but cannot cover
        bad = ((df.arm == "uncond") & df.centre.isin(excl)).values
        v = v & ~bad
        print(f"[{m}] {int(bad.sum())} uncond samples from KDE-excluded centres {sorted(excl)} marked invalid")
    df["valid"] = v
    sets[m] = df
    for arm, g in df.groupby("arm"):
        kk = kin.loc[g.index]
        val.append(dict(method=m, arm=arm, n=len(g), valid_frac=float(g.valid.mean()),
                        pet_finite=float(np.isfinite(g.pet.astype(float)).mean()),
                        teleport=float(g.teleport.mean()) if "teleport" in g else np.nan,
                        v_max_p99=float(np.percentile(kk.v_max, 99)), alat_p99=float(np.percentile(kk.alat_max, 99)),
                        frac_v_gt25=float((kk.v_max > 25).mean()), frac_alat_gt5=float((kk.alat_max > 5).mean())))
    print(f"[{m}] rows={len(df)} arms={df.arm.value_counts().to_dict()} valid={df.valid.mean():.3f}")
pd.DataFrame(val).to_csv(L.RESULTS / "validity.csv", index=False)

# ── E-A: first hit (conditional) + ref ───────────────────────────────────────
fh_rows, corner_rm = [], []
for m, df in sets.items():
    ref = df[df.arm == "ref"].set_index("centre")
    cond = df[df.arm == "cond"]
    for pool in POOLS:
        for i, sid in enumerate(T.keys):
            g = cond[cond.centre == sid].sort_values("draw")
            row = dict(method=m, pool=pool, scenario_id=sid, n_draws=len(g))
            if sid in ref.index and (pool == "all" or bool(ref.loc[sid, "valid"])):
                Dr = dists(feats(ref.loc[[sid]]), i)
                for k in METRICS:
                    row[f"ref_{k}"] = float(Dr[k][0]); row[f"ref_hit_{k}"] = bool(Dr[k][0] <= 1.0)
            else:
                for k in METRICS:
                    row[f"ref_{k}"] = np.inf; row[f"ref_hit_{k}"] = False
            if len(g) == 0:
                for k in METRICS:
                    row[f"first_{k}"] = np.inf; row[f"min_{k}"] = np.inf; row[f"n_hit_{k}"] = 0
                fh_rows.append(row)
                continue
            D = dists(feats(g), i)
            if pool == "valid":                  # invalid draws still cost a render but cannot cover
                bad = ~g.valid.values
                D = {k: np.where(bad, np.inf, v) for k, v in D.items()}
            for k in METRICS:
                j = np.flatnonzero(D[k] <= 1.0)
                row[f"first_{k}"] = float(g.draw.values[j[0]]) if len(j) else np.inf
                row[f"min_{k}"] = float(D[k].min())
                row[f"n_hit_{k}"] = int(len(j))
            fh_rows.append(row)
            if sid == L.CORNER:
                corner_rm.append(pd.DataFrame(dict(method=m, pool=pool, draw=g.draw.values,
                                                   **{f"run_min_{k}": np.minimum.accumulate(D[k]) * EPS.get(k, 1.0)
                                                      for k in ("path_w100", "arc", "interaction")})))
fh = pd.DataFrame(fh_rows)
fh.to_csv(L.RESULTS / "first_hit_per_scenario.csv", index=False)
pd.concat(corner_rm).to_csv(L.RESULTS / "corner_running_min.csv", index=False)

summ = []
for (m, pool), g0 in fh.groupby(["method", "pool"]):
    for k in METRICS:
        g = g0[g0.scenario_id.map(lambda s_: VALID_REAL[k][T.idx[s_]])]
        f = g[f"first_{k}"].values
        fin = np.isfinite(f)
        c = g0[g0.scenario_id == L.CORNER].iloc[0]
        summ.append(dict(method=m, pool=pool, metric=k, n_real=int(len(g)),
                         ref_hit_frac=float(g[f"ref_hit_{k}"].mean()),
                         hit_at_1=float((f == 1).mean()), ever_hit_frac=float(fin.mean()),
                         first_med_all=float(np.median(f)),
                         first_med_hit=float(np.median(f[fin])) if fin.any() else np.inf,
                         first_p90_hit=float(np.percentile(f[fin], 90)) if fin.any() else np.inf,
                         corner_first=float(c[f"first_{k}"]), corner_min=float(c[f"min_{k}"]) * EPS.get(k, 1.0)))
summ = pd.DataFrame(summ)
summ.to_csv(L.RESULTS / "table_first_hit.csv", index=False)
print(summ[summ.pool == "valid"].round(3).to_string(index=False))

# ── E-B: coverage vs budget (unconditional) ──────────────────────────────────
ceil = pd.read_csv(L.RESULTS / "svdkde_ceiling.csv")
loo = {sp: np.array([T.holdout_floor([j], space=sp)[j] for j in range(T.N)]) for sp in ("w100", "w101")}
curve, final = [], []
for m, df in sets.items():
    unc = df[df.arm == "uncond"].sort_values("draw")
    F = feats(unc)
    M = len(unc)
    Dm0 = {k: np.stack([dists(F, i)[k] for i in range(T.N)]) for k in METRICS}        # (N, M) eps units
    for pool in POOLS:
        Dm = Dm0 if pool == "all" else {k: np.where(~unc.valid.values[None, :], np.inf, v) for k, v in Dm0.items()}
        for em in EMULT:
            H = {k: (Dm[k] <= em)[VALID_REAL[k]] for k in METRICS}
            for o in range(NORD):
                perm = np.arange(M) if o == 0 else L.rng(60000 + o).permutation(M)
                for k in METRICS:
                    cov = np.maximum.accumulate(H[k][:, perm], axis=1).mean(0)
                    for b in L.BUDGETS:
                        if b <= M:
                            curve.append(dict(method=m, pool=pool, eps_mult=em, metric=k, order=o, budget=b,
                                              coverage=float(cov[b - 1])))
                    if o == 0:
                        full = np.flatnonzero(cov >= 1.0)
                        keys_v = np.array(T.keys)[VALID_REAL[k]]
                        cum_last = np.maximum.accumulate(H[k], axis=1)[:, -1]
                        d = dict(method=m, pool=pool, eps_mult=em, metric=k, coverage_at_max=float(cov[-1]),
                                 n_samples=M, n_real=int(VALID_REAL[k].sum()),
                                 budget_to_100=int(full[0] + 1) if len(full) else np.inf,
                                 n_uncovered=int((~cum_last).sum()), uncovered=";".join(keys_v[~cum_last]))
                        if m == "svd_kde" and k in ("path_w100", "path_w101"):
                            sp = k[-4:]
                            d["ceiling_rank3"] = float((ceil[ceil.space == sp].floor.values <= em * T.eps[sp]).mean())
                            d["ceiling_rank3_loo"] = float((loo[sp] <= em * T.eps[sp]).mean())
                        final.append(d)
curve = pd.DataFrame(curve)
curve.to_csv(L.RESULTS / "table_budget_curve.csv", index=False)
fin = pd.DataFrame(final)
fin.to_csv(L.RESULTS / "table_final.csv", index=False)
piv = curve[(curve.order == 0) & (curve.pool == "valid") & (curve.eps_mult == 1.0)].pivot_table(
    index=["metric", "method"], columns="budget", values="coverage")
print(piv.round(2).to_string())
print(fin[(fin.pool == "valid")].drop(columns="uncovered").round(3).to_string(index=False))
