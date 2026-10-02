"""New method variant "cp3d10": NURBS interior points = critical position
(minPET anchor) ± 10 m arc-length (3 points), plus the usual start/end (and
the two heading-fix points) — SampleConfig(anchor="pet", placement="critdist",
n_points=3, step_m=10). Speed handling identical to petq3 (legacy real-speed
event anchored at the minPET frame).

Pipeline: gen base xosc (project-local) → esmini base render → interaction
descriptors → PAIRED interaction similarity |Δ(gen, real)| per scenario,
merged with sr-tlkeep's tlkeep_fidelity_interaction.csv (sakura / svd_d3 /
svd_kde / ours-petq3) for the comparison table + fig7.

Usage: micromamba run -n nps python 80_cp3_method.py [--smoke N]
Outputs: results/cp3_descriptors.parquet, results/cp3_interaction.csv,
         results/interaction_comparison.csv, figs/fig7_interaction_sim.png
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
CFG = SampleConfig("cp3d10", anchor="pet", placement="critdist",
                   n_points=3, step_m=10.0)
XBASE = L.RUNS / "_xosc_base_cp3"
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
                        sample_cfg=CFG, speed_model=None)


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
        xosc = run_dir / "cp3base.xosc"
        csv = run_dir / "cp3base.csv"
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
                continue
            rows.extend(PL.trajs_to_rows(res, sid, "cp3d10", "base"))
    tr = pd.DataFrame(rows, columns=PL.ROW_COLUMNS)
    tr.to_parquet(L.RESULTS / f"cp3_trajectories{L.SUF}.parquet")
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
    dd.to_parquet(L.RESULTS / f"cp3_descriptors{L.SUF}.parquet")

    # ---- paired interaction similarity vs real (sr-tlkeep conventions) -----
    sr = pd.read_parquet(L.RDESC)
    if "error" in sr.columns:
        sr = sr[sr.error.isna()]
    rd = sr[sr.set == "real"].set_index("scenario_id")
    mu_head = circ_mean_deg(rd["agent_arr_heading"].values)
    dd = dd.set_index("scenario_id")
    irows = []
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
        from scipy import stats as _st
        g_all = dd[key].astype(float).values
        r_all = rd[key].astype(float).values
        gf = g_all[np.isfinite(g_all)]
        rf = r_all[np.isfinite(r_all)]
        irows.append({
            "method": "cp3d10", "key": key,
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
    irows.append({"method": "cp3d10", "key": "conflict_point_dist_m",
                  "paired_mae_median": float(np.median(cds)) if cds else np.nan,
                  "paired_mae_p90": float(np.percentile(cds, 90)) if cds else np.nan,
                  "n_paired": len(cds)})
    irows.append({"method": "cp3d10", "key": "arr_time_gap_s",
                  "paired_mae_median": float(np.median(ats)) if ats else np.nan,
                  "paired_mae_p90": float(np.percentile(ats, 90)) if ats else np.nan,
                  "n_paired": len(ats)})
    cp3 = pd.DataFrame(irows)
    cp3.to_csv(L.RESULTS / f"cp3_interaction{L.SUF}.csv", index=False)

    other_f = L.SR / "results" / f"{L.NAME}_fidelity_interaction.csv"
    if other_f.exists():
        comp = pd.concat([pd.read_csv(other_f), cp3], ignore_index=True)
    else:
        print(f"(no sr comparison file for {L.NAME} — cp3d10 only)")
        comp = cp3
    comp.to_csv(L.RESULTS / f"interaction_comparison{L.SUF}.csv", index=False)
    show = comp[comp.key.isin(["pet", "min_dist", "agent_arr_speed",
                               "conflict_angle", "closing_speed",
                               "conflict_point_dist_m", "arr_time_gap_s"])]
    print(show.pivot_table(index="key", columns="method",
                           values="paired_mae_median").round(3).to_string(),
          flush=True)
    print("CP3 DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None)
    main(ap.parse_args().smoke)
