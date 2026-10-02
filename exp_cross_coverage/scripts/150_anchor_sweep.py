"""Anchor sweep — decoupled critical POSITION (CP placement) × critical FRAME
(speed event) anchors, each ∈ {pet, min_dist, traj_cross} (3×3 = 9 combos),
rendered with the subset's best d from the 2026-08-14 d-sweep.

Every combo uses force_anchor=True + speed_model="real" + speed_anchor=<frame
anchor>: the 3 interior CPs sit at (pos-anchor position ± d) arc-length, the
speed event ramps to the agent's REAL speed at the frame-anchor frame, timed
to reach it. The legacy cp3 baseline (pipeline pet frame, window-max EndSpeed)
is loaded from the d-sweep artifacts for comparison (combo tag "legacy").

Usage: CC_SUBSET=keeptl [CC_CONFLICT_W=8] micromamba run -n nps \
           python scripts/150_anchor_sweep.py
Outputs: results/anchor_sweep_{subset}[_w8].csv,
         results/anch{P}{F}_{descriptors,trajectories}_{subset}[_w8].parquet
"""
from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib as L  # noqa: E402
from lib import PL  # noqa: E402
from run_label_lib import trim_lead_still, wrap180  # noqa: E402
from hetero_param.sweep import core as SWC  # noqa: E402
from hetero_param.similarity.core import Traj  # noqa: E402
from hetero_param import esmini_exec as EX  # noqa: E402
from hetero_param import generate as GEN  # noqa: E402
from hetero_param import interaction as IX  # noqa: E402
from hetero_param import parampath as PP  # noqa: E402
from hetero_param.config import SampleConfig  # noqa: E402

FPS = L.FPS
CW = float(os.environ.get("CC_CONFLICT_W", 0)) or None
WSUF = f"_w{CW:g}" if CW else ""
SUBSET_STEP = {"cutinl": 10, "keeptl": 10, "keeptl_sw": 10,
               "special_39_180": 10, "special_1786_1797": 5,  # v3 keys: ±5
               "uturn_859_881": 5}
STEP = SUBSET_STEP.get(L.NAME, 10)
ANCH = {"p": "pet", "m": "min_dist", "x": "traj_cross"}
COMBOS = [(P, F) for P in "pmx" for F in "pmx"]
KEYS = ["pet", "min_dist", "agent_arr_speed", "conflict_angle",
        "agent_arr_heading"]


def _patch_conflict_weight(xosc: Path, w: float):
    tree = ET.parse(xosc)
    cps = [cp for cp in tree.getroot().iter("ControlPoint")
           if cp.get("weight") is not None]
    if len(cps) == 3:
        cps[1].set("weight", f"{w:g}")
        tree.write(xosc)
    else:
        print(f"[conflict-w] SKIP {xosc.name}: {len(cps)} weighted CPs",
              flush=True)


def _traj_df(g, Lw, W):
    t = g.frame.values.astype(float) / FPS
    sp = g.speed.values.astype(float)
    acc = np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp)
    return Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
                y=g.y.values.astype(float),
                heading=g.heading_deg.values.astype(float), speed=sp, accel=acc,
                fps=FPS, length=float(Lw), width=float(W), meta={})


def _pfv(P: str, F: str) -> str:
    """Cache version for p-anchored arms: the 2026-08-15 pet_frame orientation
    fix (agent's own boundary frame) invalidates every gen that used anchor=
    'pet' for pos or frame — bump their cache namespace, keep m/x untouched."""
    return "_pf2" if "p" in (P, F) else ""


def _xbase_for(P: str, F: str) -> Path:
    return L.RUNS / f"_xosc_base_anch_{P}{F}_s{STEP}{_pfv(P, F)}"


def _gen_one(args):
    """PROCESS-isolated gen (global sampling state — see the 120 race lesson)."""
    P, F, ego, actor, mf, xf, label = args
    cfg = SampleConfig(f"anch{P}{F}s{STEP}", anchor=ANCH[P],
                       placement="critdist", n_points=3, step_m=float(STEP))
    xbase = _xbase_for(P, F)
    try:
        hits = list(xbase.glob(f"*_{ego}_{actor}_f{mf + 1}.xosc"))
        if hits:
            return (ego, actor, str(hits[0]), None)
        b = GEN.generate(PL.DATASET, ego, [actor], [], mf, xf,
                         label if label else L.LABEL, xbase,
                         sample_cfg=cfg, force_anchor=True,
                         speed_model="real", speed_anchor=ANCH[F])
        return (ego, actor, str(b) if b else None, None)
    except Exception as e:  # noqa: BLE001
        return (ego, actor, None, f"{type(e).__name__}: {e}")


def gen_render_combo(P: str, F: str, meta: pd.DataFrame):
    xbase = _xbase_for(P, F)
    xbase.mkdir(parents=True, exist_ok=True)
    gen_args = [(P, F, int(r.ego), int(r.actor), int(r.min_frame),
                 int(r.max_frame),
                 int(r.label) if "label" in meta.columns and r.label
                 and int(r.label) not in (77, 88) else None)
                for r in meta.itertuples()]
    with Pool(6) as pool:
        gen_out = pool.map(_gen_one, gen_args, chunksize=2)
    bases = {f"{e}_{a}": b for (e, a, b, err) in gen_out if b}
    for (e, a, b, err) in gen_out:
        if err:
            print(f"[{P}{F}] gen fail {e}_{a}: {err[:120]}", flush=True)

    tag = f"anch{P}{F}s{STEP}{'w%g' % CW if CW else ''}{_pfv(P, F)}base"

    def render_one(job):
        sid, ego, actor, mf, xf, base = job
        try:
            run_dir = L.RUNS / Path(base).stem
            run_dir.mkdir(parents=True, exist_ok=True)
            xosc = run_dir / f"{tag}.xosc"
            csv = run_dir / f"{tag}.csv"
            if not csv.exists() or not xosc.exists():
                EX.to_esmini_replay(Path(base), PL.DATASET, ego, actor, mf,
                                    xf, xosc, None, agent_replay=False)
                if CW:
                    _patch_conflict_weight(xosc, CW)
            res = EX.run_and_extract(xosc, PL.DATASET, ego, actor, mf, xf, csv)
            if res is None or res.get("agent") is None:
                return (sid, None)
            return (sid, res)
        except Exception:  # noqa: BLE001
            return (sid, None)

    jobs = [(f"{int(r.ego)}_{int(r.actor)}", int(r.ego), int(r.actor),
             int(r.min_frame), int(r.max_frame),
             bases[f"{int(r.ego)}_{int(r.actor)}"])
            for r in meta.itertuples()
            if f"{int(r.ego)}_{int(r.actor)}" in bases]
    rows = []
    with ThreadPoolExecutor(12) as tp:
        for (sid, res) in tp.map(render_one, jobs):
            if res:
                rows.extend(PL.trajs_to_rows(res, sid, f"anch{P}{F}", "base"))
    print(f"[{P}{F}] gen ok={len(bases)}/{len(gen_args)} rendered "
          f"{len(set(r[0] for r in rows))}", flush=True)
    return pd.DataFrame(rows, columns=PL.ROW_COLUMNS)


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


def descriptors_for(tr: pd.DataFrame):
    global G_JOB
    G_JOB = {}
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
    with Pool(8) as pool:
        out = [r for r in pool.imap_unordered(_desc_one, list(G_JOB),
                                              chunksize=8) if r]
    dd = pd.DataFrame(out)
    if "error" in dd.columns:
        dd = dd[dd.error.isna()]
    return dd, n_tp


def legacy_desc_traj():
    """The d-sweep cp3 artifacts at this subset's STEP (weight-aware)."""
    if STEP == 10 and not CW:
        d = L.RESULTS / f"cp3_descriptors{L.SUF}.parquet"
        t = L.RESULTS / f"cp3_trajectories{L.SUF}.parquet"
    else:
        d = L.RESULTS / f"cp3s{STEP}_descriptors{L.SUF}{WSUF}.parquet"
        t = L.RESULTS / f"cp3s{STEP}_trajectories{L.SUF}{WSUF}.parquet"
    return (d, t) if d.exists() and t.exists() else (None, None)


def main():
    meta = pd.read_csv(L.DDIR / "real_meta.csv")
    sr = pd.read_parquet(L.RDESC)
    if "error" in sr.columns:
        sr = sr[sr.error.isna()]
    rd = sr[sr.set == "real"].set_index("scenario_id") if "set" in sr.columns \
        else sr.set_index("scenario_id")

    # anchor-frame availability audit (fallbacks collapse combos!)
    na_pet = na_md = na_tc = 0
    for r in meta.itertuples():
        petf, mdf, tcf = PP.anchor_frames(PL.DATASET, int(r.ego), int(r.actor),
                                          int(r.min_frame), int(r.max_frame))
        na_pet += petf is None
        na_md += mdf is None
        na_tc += tcf is None
    print(f"[{L.NAME}] anchor availability: pet missing {na_pet}/{len(meta)}, "
          f"min_dist missing {na_md}, traj_cross missing {na_tc}", flush=True)

    combo_desc, combo_tr = {}, {}
    for (P, F) in COMBOS:
        t0 = time.time()
        d_file = L.RESULTS / f"anch{P}{F}_descriptors{L.SUF}{WSUF}.parquet"
        t_file = L.RESULTS / f"anch{P}{F}_trajectories{L.SUF}{WSUF}.parquet"
        if d_file.exists() and t_file.exists():
            dd = pd.read_parquet(d_file)
            if "error" in dd.columns:
                dd = dd[dd.error.isna()]
            tr = pd.read_parquet(t_file)
            n_tp = -1
        else:
            tr = gen_render_combo(P, F, meta)
            tr.to_parquet(t_file)
            dd, n_tp = descriptors_for(tr)
            dd.to_parquet(d_file)
        combo_desc[(P, F)] = (dd.set_index("scenario_id") if len(dd) else
                              pd.DataFrame(
                                  index=pd.Index([], name="scenario_id")))
        combo_tr[(P, F)] = tr
        print(f"[{P}{F}] {len(dd)} descriptor rows (tp={n_tp}, "
              f"{time.time() - t0:.0f}s)", flush=True)

    # legacy baseline from the d-sweep
    ld, lt = legacy_desc_traj()
    if ld is not None:
        dd = pd.read_parquet(ld)
        if "error" in dd.columns:
            dd = dd[dd.error.isna()]
        combo_desc[("legacy", "legacy")] = (
            dd.set_index("scenario_id") if len(dd) else
            pd.DataFrame(index=pd.Index([], name="scenario_id")))
        combo_tr[("legacy", "legacy")] = pd.read_parquet(lt)

    # real interaction windows
    real_tracks = pd.read_parquet(L.DDIR / "real_tracks.parquet")
    R = {(sid, role): g.sort_values("frame")
         for (sid, role), g in real_tracks.groupby(["scenario_id", "role"])}
    half = int(round(2.0 * FPS))
    RW = {}
    for sid in meta.scenario_id:
        if (sid, "actor") in R and (sid, "ego") in R:
            ra, re_ = R[(sid, "actor")], R[(sid, "ego")]
            RW[sid] = IX._pair_interaction(
                ra.frame.values, ra.x.values, ra.y.values,
                re_.frame.values, re_.x.values, re_.y.values, FPS)

    sum_rows = []
    for (P, F), dd in combo_desc.items():
        for key in KEYS + ["conflict_point"]:
            diffs = []
            for sid in dd.index:
                if sid not in rd.index:
                    continue
                if key == "conflict_point":
                    if np.isfinite(dd.loc[sid, "conflict_x"]) and \
                            np.isfinite(rd.loc[sid, "conflict_x"]):
                        diffs.append(float(np.hypot(
                            dd.loc[sid, "conflict_x"] - rd.loc[sid, "conflict_x"],
                            dd.loc[sid, "conflict_y"] - rd.loc[sid, "conflict_y"])))
                    continue
                gv, rv = float(dd.loc[sid, key]), float(rd.loc[sid, key])
                if not (np.isfinite(gv) and np.isfinite(rv)):
                    continue
                d_ = (abs(float(wrap180(gv % 360 - rv % 360)))
                      if key == "agent_arr_heading" else abs(gv - rv))
                diffs.append(d_)
            if diffs:
                sum_rows.append({"pos": P, "frm": F, "key": key,
                                 "n": len(diffs),
                                 "mae_median": float(np.median(diffs)),
                                 "mae_p90": float(np.percentile(diffs, 90))})
        tr = combo_tr[(P, F)]
        ag = {sid: g.sort_values("frame") for (sid, role), g in
              tr.groupby(["scenario_id", "role"]) if role == "agent"}
        eg = {sid: g.sort_values("frame") for (sid, role), g in
              tr.groupby(["scenario_id", "role"]) if role == "ego"}
        pd_, gd_ = [], []
        for sid in dd.index:
            o_r = RW.get(sid)
            if not o_r or sid not in ag or sid not in eg or sid not in rd.index:
                continue
            a, e = ag[sid], eg[sid]
            o_p = IX._pair_interaction(a.frame.values, a.x.values, a.y.values,
                                       e.frame.values, e.x.values, e.y.values,
                                       FPS)
            if not o_p:
                continue
            ps = IX._profile_similarity(o_r, o_p, half)
            gw = IX._dtw1d(IX._gap_window(o_r, half), IX._gap_window(o_p, half))
            if np.isfinite(ps):
                pd_.append(1.0 - float(ps))
            if np.isfinite(gw):
                gd_.append(float(gw))
        for key, vals in (("profile_dissim", pd_), ("gap_dtw", gd_)):
            if vals:
                sum_rows.append({"pos": P, "frm": F, "key": key,
                                 "n": len(vals),
                                 "mae_median": float(np.median(vals)),
                                 "mae_p90": float(np.percentile(vals, 90))})
    comp = pd.DataFrame(sum_rows)
    comp.to_csv(L.RESULTS / f"anchor_sweep{L.SUF}{WSUF}.csv", index=False)
    piv = comp.pivot_table(index="key", columns=["pos", "frm"],
                           values="mae_median")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
