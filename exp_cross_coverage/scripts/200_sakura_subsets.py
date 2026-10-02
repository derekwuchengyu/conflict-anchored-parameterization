"""SAKURA (UN R157 / start-end-only NURBS, constant speed) baseline for the
subsets sr-tlkeep never rendered (cutinl, special_39_180, ...).

Mirrors 80_cp3_method.py exactly — same base-xosc generation, same esmini
replay surgery, same SWC.descriptors(estimator="bbox") conventions, same
paired |Δ| aggregation — but with the SAKURA sample config:
    SampleConfig("none", placement="none", n_points=0), speed_model="const"
so the resulting numbers are directly comparable with cp3_interaction*.csv.

Usage: CC_SUBSET=cutinl micromamba run -n nps python scripts/200_sakura_subsets.py
Outputs: results/sakura_trajectories{SUF}.parquet,
         results/sakura_descriptors{SUF}.parquet,
         results/sakura_interaction{SUF}.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib as L  # noqa: E402
from lib import PL  # noqa: E402
from run_label_lib import trim_lead_still, circ_mean_deg, wrap180  # noqa: E402
from hetero_param.sweep import core as SWC  # noqa: E402
from hetero_param.similarity.core import Traj  # noqa: E402
from hetero_param import esmini_exec as EX  # noqa: E402
from hetero_param import generate as GEN  # noqa: E402
from hetero_param.config import SampleConfig  # noqa: E402

FPS = L.FPS
METHOD = "sakura"
CFG = SampleConfig("none", placement="none", n_points=0)
SPEED_MODEL = "const"
XBASE = L.RUNS / "_xosc_base_sakura"
IKEYS = ["pet", "pet_abs", "min_ttc", "min_dist", "conflict_angle",
         "closing_speed", "drac", "agent_arr_speed", "agent_arr_accel",
         "agent_arr_heading"]


def gen_base(ego, actor, mf, xf, label=None):
    XBASE.mkdir(parents=True, exist_ok=True)
    hits = list(XBASE.glob(f"*_{ego}_{actor}_f{mf + 1}.xosc"))
    if hits:
        return hits[0]
    return GEN.generate(PL.DATASET, ego, [actor], [], mf, xf,
                        label if label is not None else L.LABEL, XBASE,
                        sample_cfg=CFG, speed_model=SPEED_MODEL)


def _gen_one(args):
    ego, actor, mf, xf, label = args
    try:
        base = gen_base(ego, actor, mf, xf, label)
        return (ego, actor, str(base) if base else None, None)
    except Exception as e:  # noqa: BLE001
        return (ego, actor, None, f"{type(e).__name__}: {e}")


def _render_one(job):
    sid, ego, actor, mf, xf, base = job
    try:
        run_dir = L.RUNS / Path(base).stem
        run_dir.mkdir(parents=True, exist_ok=True)
        xosc = run_dir / "sakbase.xosc"
        csv = run_dir / "sakbase.csv"
        if not csv.exists() or not xosc.exists():
            EX.to_esmini_replay(Path(base), PL.DATASET, ego, actor, mf, xf,
                                xosc, None, agent_replay=False)
        res = EX.run_and_extract(xosc, PL.DATASET, ego, actor, mf, xf, csv)
        if res is None or res.get("agent") is None:
            return (sid, None, "no agent traj")
        return (sid, res, None)
    except Exception as e:  # noqa: BLE001
        return (sid, None, f"{type(e).__name__}: {e}")


def _traj_df(g, Lw, W):
    t = g.frame.values.astype(float) / FPS
    sp = g.speed.values.astype(float)
    acc = np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp)
    return Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
                y=g.y.values.astype(float),
                heading=g.heading_deg.values.astype(float), speed=sp, accel=acc,
                fps=FPS, length=float(Lw), width=float(W), meta={})


G_JOB = {}


def _desc_one(sid):
    try:
        g, eg = G_JOB[sid]
        agent = _traj_df(g, g.length.iloc[0], g.width.iloc[0])
        ego = _traj_df(eg, eg.length.iloc[0], eg.width.iloc[0])
        d = SWC.descriptors(agent, ego, estimator="bbox")
        d.update({"scenario_id": sid})
        return d
    except Exception as e:  # noqa: BLE001
        return {"scenario_id": sid, "error": f"{type(e).__name__}: {e}"}


def main(smoke):
    meta = pd.read_csv(L.DDIR / "real_meta.csv")
    if smoke:
        meta = meta.head(smoke)
    t0 = time.time()

    gen_args = [(int(r.ego), int(r.actor), int(r.min_frame), int(r.max_frame),
                 int(r.label) if "label" in meta.columns and r.label
                 and int(r.label) not in (77, 88) else None)
                for r in meta.itertuples()]
    with Pool(6) as pool:
        gen_out = pool.map(_gen_one, gen_args, chunksize=4)
    bases = {f"{e}_{a}": b for (e, a, b, err) in gen_out if b}
    for (e, a, b, err) in gen_out:
        if err:
            print(f"  gen fail {e}_{a}: {err}", flush=True)
    print(f"gen ok={len(bases)} fail={len(gen_args) - len(bases)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    jobs = [(f"{int(r.ego)}_{int(r.actor)}", int(r.ego), int(r.actor),
             int(r.min_frame), int(r.max_frame),
             bases[f"{int(r.ego)}_{int(r.actor)}"])
            for r in meta.itertuples()
            if f"{int(r.ego)}_{int(r.actor)}" in bases]
    rows, n_err = [], 0
    with ThreadPoolExecutor(12) as tp:
        for (sid, res, err) in tp.map(_render_one, jobs):
            if err:
                n_err += 1
                print(f"  render fail {sid}: {err}", flush=True)
                continue
            rows.extend(PL.trajs_to_rows(res, sid, METHOD, "base"))
    tr = pd.DataFrame(rows, columns=PL.ROW_COLUMNS)
    tr.to_parquet(L.RESULTS / f"sakura_trajectories{L.SUF}.parquet")
    print(f"rendered {len(jobs) - n_err}/{len(jobs)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    n_tp = 0
    for (sid, role), g in tr.groupby(["scenario_id", "role"]):
        if role != "agent":
            continue
        g = g.sort_values("frame")
        eg = tr[(tr.scenario_id == sid) & (tr.role == "ego")].sort_values("frame")
        if len(eg) == 0:
            continue
        if L.is_teleport(g):
            n_tp += 1
            continue
        G_JOB[sid] = (trim_lead_still(g), eg)
    print(f"descriptors: {len(G_JOB)} jobs ({n_tp} teleport-excluded)",
          flush=True)
    with Pool(8) as pool:
        out = [r for r in pool.imap_unordered(_desc_one, list(G_JOB),
                                              chunksize=16) if r]
    dd = pd.DataFrame(out)
    if "error" in dd.columns:
        dd = dd[dd.error.isna()]
    dd.to_parquet(L.RESULTS / f"sakura_descriptors{L.SUF}.parquet")

    # ---- paired interaction similarity vs real (identical to 80_cp3) -------
    sr = pd.read_parquet(L.RDESC)
    if "error" in sr.columns:
        sr = sr[sr.error.isna()]
    rd = sr[sr.set == "real"].set_index("scenario_id")
    mu_head = circ_mean_deg(rd["agent_arr_heading"].values)
    dd = dd.set_index("scenario_id")
    irows = []
    from scipy import stats as _st
    for key in IKEYS:
        diffs = []
        for sid in dd.index:
            if sid not in rd.index:
                continue
            gv, rv = float(dd.loc[sid, key]), float(rd.loc[sid, key])
            if key == "agent_arr_heading":
                gv = float(wrap180(gv - mu_head))
                rv = float(wrap180(rv - mu_head))
            if np.isfinite(gv) and np.isfinite(rv):
                d_ = gv - rv
                if key == "agent_arr_heading":
                    d_ = float(wrap180(d_))
                diffs.append(abs(d_))
        g_all = dd[key].astype(float).values
        r_all = rd[key].astype(float).values
        gf = g_all[np.isfinite(g_all)]
        rf = r_all[np.isfinite(r_all)]
        irows.append({
            "method": METHOD, "key": key,
            "frac_finite": float(np.isfinite(g_all).mean()),
            "real_frac_finite": float(np.isfinite(r_all).mean()),
            "wasserstein_1d": float(_st.wasserstein_distance(rf, gf))
            if len(gf) > 2 else np.nan,
            "paired_mae_median": float(np.median(diffs)) if diffs else np.nan,
            "paired_mae_p90": float(np.percentile(diffs, 90)) if diffs else np.nan,
            "n_paired": len(diffs)})
    cds, ats = [], []
    for sid in dd.index:
        if sid not in rd.index:
            continue
        rg, rr = dd.loc[sid], rd.loc[sid]
        if np.isfinite(rg.conflict_x) and np.isfinite(rr.conflict_x):
            cds.append(float(np.hypot(rg.conflict_x - rr.conflict_x,
                                      rg.conflict_y - rr.conflict_y)))
        tg = (rg.agent_arr_frame - rg.ego_arr_frame) / FPS
        tr_ = (rr.agent_arr_frame - rr.ego_arr_frame) / FPS
        if np.isfinite(tg) and np.isfinite(tr_):
            ats.append(abs(float(tg - tr_)))
    irows.append({"method": METHOD, "key": "conflict_point_dist_m",
                  "paired_mae_median": float(np.median(cds)) if cds else np.nan,
                  "paired_mae_p90": float(np.percentile(cds, 90)) if cds else np.nan,
                  "n_paired": len(cds)})
    irows.append({"method": METHOD, "key": "arr_time_gap_s",
                  "paired_mae_median": float(np.median(ats)) if ats else np.nan,
                  "paired_mae_p90": float(np.percentile(ats, 90)) if ats else np.nan,
                  "n_paired": len(ats)})
    sak = pd.DataFrame(irows)
    sak.to_csv(L.RESULTS / f"sakura_interaction{L.SUF}.csv", index=False)
    print(sak[["key", "paired_mae_median", "paired_mae_p90", "n_paired"]]
          .round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    main(args.smoke)
