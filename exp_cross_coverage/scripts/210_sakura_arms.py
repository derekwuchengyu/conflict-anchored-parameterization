"""SAKURA arms: `sakura` (start/end NURBS chord) and `sakura_route`
(planned road route when one exists, NURBS chord otherwise).

Identical generation / esmini-replay / descriptor path as 80_cp3_method.py and
200_sakura_subsets.py, so the numbers drop straight into
results/interaction_comparison*.csv alongside cp3d10.

  --fan   also render the 27-point {min,mid,max}^3 parameter cross
          (Offset x EndSpeed x Duration, lib.cross_plan) for the coverage figure.

Usage:
  CC_SUBSET=keeptl micromamba run -n nps python scripts/210_sakura_arms.py --arm sakura_route [--fan]
Outputs (per subset SUF, per arm):
  results/<arm>_trajectories{SUF}.parquet, <arm>_descriptors{SUF}.parquet,
  results/<arm>_interaction{SUF}.csv, results/route_plan{SUF}.csv
  --fan: results/<arm>_fan_descriptors{SUF}.parquet
"""
from __future__ import annotations

import argparse
import json
import os
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
import sakura_route as SR  # noqa: E402
from run_label_lib import trim_lead_still, circ_mean_deg, wrap180  # noqa: E402
from hetero_param.sweep import core as SWC  # noqa: E402
from hetero_param.similarity.core import Traj, real_traj  # noqa: E402
from hetero_param import esmini_exec as EX  # noqa: E402
from hetero_param import generate as GEN  # noqa: E402
from hetero_param.config import SampleConfig  # noqa: E402

FPS = L.FPS
CFG = SampleConfig("none", placement="none", n_points=0)   # SAKURA: start/end only
SPEED_MODEL = "const"
XBASE = L.RUNS / "_xosc_base_sakura"
TAGPFX = {"sakura": "sak", "sakura_route": "sakrt"}
IKEYS = ["pet", "pet_abs", "min_ttc", "min_dist", "conflict_angle",
         "closing_speed", "drac", "agent_arr_speed", "agent_arr_accel",
         "agent_arr_heading"]


# ── base xosc (shared by both arms) ──────────────────────────────────────────
def _gen_one(args):
    ego, actor, mf, xf, label = args
    try:
        XBASE.mkdir(parents=True, exist_ok=True)
        hits = list(XBASE.glob(f"*_{ego}_{actor}_f{mf + 1}.xosc"))
        base = hits[0] if hits else GEN.generate(
            PL.DATASET, ego, [actor], [], mf, xf,
            label if label is not None else L.LABEL, XBASE,
            sample_cfg=CFG, speed_model=SPEED_MODEL)
        return (ego, actor, str(base) if base else None, None)
    except Exception as e:  # noqa: BLE001
        return (ego, actor, None, f"{type(e).__name__}: {e}")


# ── route planning (main process; cached to disk) ────────────────────────────
def plan_all(meta) -> dict:
    cache_f = L.RESULTS / f"route_plan{L.SUF}.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    cmap = None
    for r in meta.itertuples():
        sid = f"{int(r.ego)}_{int(r.actor)}"
        if sid in cache:
            continue
        t = real_traj(PL.DATASET, int(r.actor), int(r.min_frame), int(r.max_frame))
        if t is None or len(t.x) < 2:
            cache[sid] = None
            continue
        if cmap is None:
            cmap = GEN._get_map(PL.DATASET)
        try:
            wp = SR.plan_route(cmap, t.x[0], t.y[0], t.x[-1], t.y[-1])
        except Exception:  # noqa: BLE001
            wp = None
        cache[sid] = wp
    cache_f.write_text(json.dumps(cache))
    return cache


# ── render ───────────────────────────────────────────────────────────────────
ROUTES: dict = {}
ARM = "sakura"
AVG_SPEED = True
SCREEN_TELEPORT = os.environ.get("SAK_SCREEN_TELEPORT", "1") != "0"


def _speed_override(sid, ego, actor, mf, xf, overrides):
    """Route arm: constant speed = planned-route length / recorded travel time,
    so the agent traverses the planned route over the same window the recording
    used (SAKURA is defined by a constant AVERAGE speed; the pipeline's "const"
    hook uses the instantaneous START speed, which stalls stop-and-go agents and
    de-synchronises a speed-driven route)."""
    wp = ROUTES.get(sid)
    if not wp:
        return overrides
    t = real_traj(PL.DATASET, actor, mf, xf)
    if t is None or len(t.frame) < 2:
        return overrides
    dur = (float(t.frame[-1]) - float(t.frame[0])) / FPS
    if dur <= 0:
        return overrides
    v_kmh = 3.6 * SR.route_length(wp) / dur
    ov = dict(overrides or {})
    base_end = ov.get("Agent1_1_SA_EndSpeed")
    ov["Agent1_Speed"] = v_kmh
    ov["Agent1_1_SA_EndSpeed"] = v_kmh if base_end is None else base_end
    return ov


def _render_one(job):
    sid, ego, actor, mf, xf, base, tag, overrides = job
    try:
        run_dir = L.RUNS / Path(base).stem
        run_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{TAGPFX[ARM]}{'' if AVG_SPEED or ARM != 'sakura_route' else 'v0'}{tag}"
        xosc, csv = run_dir / f"{stem}.xosc", run_dir / f"{stem}.csv"
        if not csv.exists() or not xosc.exists():
            ov = overrides
            if ARM == "sakura_route" and AVG_SPEED:
                ov = _speed_override(sid, ego, actor, mf, xf, overrides)
            EX.to_esmini_replay(Path(base), PL.DATASET, ego, actor, mf, xf,
                                xosc, ov, agent_replay=False)
            if ARM == "sakura_route" and ROUTES.get(sid):
                SR.apply_route(xosc, [tuple(w)[:3] for w in ROUTES[sid]])
        res = EX.run_and_extract(xosc, PL.DATASET, ego, actor, mf, xf, csv)
        if res is None or res.get("agent") is None:
            return (sid, tag, None, "no agent traj")
        return (sid, tag, res, None)
    except Exception as e:  # noqa: BLE001
        return (sid, tag, None, f"{type(e).__name__}: {e}")


def _traj_df(g):
    t = g.frame.values.astype(float) / FPS
    sp = g.speed.values.astype(float)
    return Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
                y=g.y.values.astype(float),
                heading=g.heading_deg.values.astype(float), speed=sp,
                accel=np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp),
                fps=FPS, length=float(g.length.iloc[0]),
                width=float(g.width.iloc[0]), meta={})


G_JOB: dict = {}


def _desc_one(key):
    try:
        g, eg = G_JOB[key]
        d = SWC.descriptors(_traj_df(g), _traj_df(eg), estimator="bbox")
        d.update({"scenario_id": key[0], "tag": key[1]})
        return d
    except Exception as e:  # noqa: BLE001
        return {"scenario_id": key[0], "tag": key[1],
                "error": f"{type(e).__name__}: {e}"}


def descriptors_from(tr):
    G_JOB.clear()
    n_tp = 0
    # one pass: (scenario, tag, role) -> frame-sorted slice.  Filtering `tr` per
    # group inside the loop is O(n^2) and dominates the fan stage.
    groups = {k: g.sort_values("frame")
              for k, g in tr.groupby(["scenario_id", "tag", "role"], sort=False)}
    for (sid, tag, role), g in groups.items():
        if role != "agent":
            continue
        eg = groups.get((sid, tag, "ego"))
        if eg is None or len(eg) == 0:
            continue
        if SCREEN_TELEPORT and L.is_teleport(g):
            n_tp += 1
            continue
        G_JOB[(sid, tag)] = (trim_lead_still(g), eg)
    with Pool(8) as pool:
        out = [r for r in pool.imap_unordered(_desc_one, list(G_JOB), chunksize=16) if r]
    dd = pd.DataFrame(out)
    if "error" in dd.columns:
        dd = dd[dd.error.isna()]
    return dd, n_tp


def paired_table(dd, rd, method):
    mu_head = circ_mean_deg(rd["agent_arr_heading"].values)
    dd = dd.set_index("scenario_id")
    rows = []
    from scipy import stats as _st
    for key in IKEYS:
        diffs = []
        for sid in dd.index:
            if sid not in rd.index:
                continue
            gv, rv = float(dd.loc[sid, key]), float(rd.loc[sid, key])
            if key == "agent_arr_heading":
                gv, rv = float(wrap180(gv - mu_head)), float(wrap180(rv - mu_head))
            if np.isfinite(gv) and np.isfinite(rv):
                d_ = gv - rv
                if key == "agent_arr_heading":
                    d_ = float(wrap180(d_))
                diffs.append(abs(d_))
        g_all = dd[key].astype(float).values
        r_all = rd[key].astype(float).values
        gf, rf = g_all[np.isfinite(g_all)], r_all[np.isfinite(r_all)]
        rows.append({"method": method, "key": key,
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
    for k, v in (("conflict_point_dist_m", cds), ("arr_time_gap_s", ats)):
        rows.append({"method": method, "key": k,
                     "paired_mae_median": float(np.median(v)) if v else np.nan,
                     "paired_mae_p90": float(np.percentile(v, 90)) if v else np.nan,
                     "n_paired": len(v)})
    return pd.DataFrame(rows)


def main(arm, fan, smoke, avg_speed=True):
    global ARM, AVG_SPEED
    ARM, AVG_SPEED = arm, avg_speed
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
    print(f"[{arm}] base xosc ok={len(bases)}/{len(gen_args)} ({time.time()-t0:.0f}s)",
          flush=True)

    if arm == "sakura_route":
        ROUTES.update({k: v for k, v in plan_all(meta).items() if v})
        planned = sum(1 for r in meta.itertuples()
                      if f"{int(r.ego)}_{int(r.actor)}" in ROUTES)
        print(f"[{arm}] route planned for {planned}/{len(meta)} scenarios "
              f"({100*planned/len(meta):.0f}%) ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame([{"scenario_id": f"{int(r.ego)}_{int(r.actor)}",
                       "route_found": f"{int(r.ego)}_{int(r.actor)}" in ROUTES,
                       "n_waypoints": len(ROUTES.get(f"{int(r.ego)}_{int(r.actor)}", []))}
                      for r in meta.itertuples()]).to_csv(
            L.RESULTS / f"route_plan{L.SUF}.csv", index=False)

    jobs = []
    for r in meta.itertuples():
        sid = f"{int(r.ego)}_{int(r.actor)}"
        if sid not in bases:
            continue
        args = (sid, int(r.ego), int(r.actor), int(r.min_frame), int(r.max_frame),
                bases[sid])
        jobs.append(args + ("base", None))
        if fan:
            _, plan = L.cross_plan(Path(bases[sid]))
            for tag, ov in plan.items():
                jobs.append(args + (tag, ov))

    rows, n_err = [], 0
    with ThreadPoolExecutor(12) as tp:
        for (sid, tag, res, err) in tp.map(_render_one, jobs):
            if err:
                n_err += 1
                continue
            rows.extend(PL.trajs_to_rows(res, sid, arm, tag))
    tr = pd.DataFrame(rows, columns=PL.ROW_COLUMNS)
    print(f"[{arm}] rendered {len(jobs)-n_err}/{len(jobs)} ({time.time()-t0:.0f}s)",
          flush=True)

    base_tr = tr[tr.tag == "base"]
    base_tr.to_parquet(L.RESULTS / f"{arm}_trajectories{L.SUF}.parquet")
    dd, n_tp = descriptors_from(base_tr)
    dd.to_parquet(L.RESULTS / f"{arm}_descriptors{L.SUF}.parquet")
    print(f"[{arm}] base descriptors {len(dd)} ({n_tp} teleport-excluded)", flush=True)

    sr = pd.read_parquet(L.RDESC)
    if "error" in sr.columns:
        sr = sr[sr.error.isna()]
    rd = sr[sr.set == "real"].set_index("scenario_id")
    if "tag" in dd.columns and len(dd):
        tab = paired_table(dd.drop(columns=["tag"]), rd, arm)
        tab.to_csv(L.RESULTS / f"{arm}_interaction{L.SUF}.csv", index=False)
        print(tab[["key", "paired_mae_median", "n_paired"]].round(3).to_string(index=False),
              flush=True)
    else:   # every base render screened out (single-scenario specials) — the
            # interaction csv for those is produced separately; keep going so
            # the fan still gets written
        print(f"[{arm}] no scored base render — skipping the paired table", flush=True)

    if fan:
        fd, n_tp_f = descriptors_from(tr[tr.tag != "base"])
        fd.to_parquet(L.RESULTS / f"{arm}_fan_descriptors{L.SUF}.parquet")
        print(f"[{arm}] fan descriptors {len(fd)} ({n_tp_f} teleport-excluded) "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sakura", "sakura_route"], default="sakura_route")
    ap.add_argument("--fan", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--start-speed", action="store_true",
                    help="route arm: keep the pipeline's start-speed constant "
                         "instead of the route-average speed")
    a = ap.parse_args()
    main(a.arm, a.fan, a.smoke, avg_speed=not a.start_speed)
