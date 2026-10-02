"""step_m sweep — how does the ±step arc-length spacing of the 3 interior
NURBS points (cp3) change GT fidelity on a class? E.g. the hook-turn's cusp is
~5 m across: ±10 m points straddle it entirely, ±2 m should hug it.

For each step in --steps: gen base xosc (SampleConfig critdist step_m=step,
cached per-step in esmini_runs/_xosc_base_cp3s{step}/), esmini render (tag
cp3s{step}base), descriptors (teleport-screened), paired |Δ| vs the subset's
real GT descriptors. step 10 reuses the existing cp3d10 artifacts when found.

Outputs: results/stepm_sweep_{subset}.csv,
         results/cp3s{step}_descriptors_{subset}.parquet,
         figs/fig12_stepm_{subset}.png
           (a) median |Δ| per key × step  (b) per-scenario slopegraph for the
           geometry keys  (c) map overlays: worst scenarios, GT vs all steps
Usage: CC_SUBSET=hookturn micromamba run -n nps python 120_stepm_sweep.py \
           --steps 10,3,2

CC_CONFLICT_W=8 additionally overrides the NURBS weight of the CONFLICT
(middle) interior control point only — the ±d flank points keep the pipeline
default 5.0. Renders get w{W} tags + all outputs a _w{W} suffix, and the
step-10 reuse of the weight-5 cp3d10 artifacts is disabled.
"""
from __future__ import annotations

import argparse
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
from hetero_param.config import SampleConfig  # noqa: E402

sys.path.insert(0, "/home/hcis-s19/Documents/ChengYu/trajectory_plots")
import xodr_lanes  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FPS = L.FPS
CW = float(os.environ.get("CC_CONFLICT_W", 0)) or None   # conflict-CP weight
WSUF = f"_w{CW:g}" if CW else ""
INK, GRAY, LANE = "#33322e", "#c9c7c0", "#dedcd4"
STEP_COL = {10: "#d99000", 5: "#0f8a60", 3: "#2a78d6", 2: "#c22e2e"}
XODR = "/home/hcis-s19/Documents/ChengYu/retrieval-scenarios/data/map/tyms.xodr"
KEYS = ["pet", "min_dist", "agent_arr_speed", "conflict_angle",
        "agent_arr_heading"]
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6f6e68", "ytick.color": "#6f6e68", "axes.grid": True,
    "grid.color": "#eceae5", "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 11})


def _traj_df(g, Lw, W):
    t = g.frame.values.astype(float) / FPS
    sp = g.speed.values.astype(float)
    acc = np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp)
    return Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
                y=g.y.values.astype(float),
                heading=g.heading_deg.values.astype(float), speed=sp, accel=acc,
                fps=FPS, length=float(Lw), width=float(W), meta={})


def _xbase_for(step: int) -> Path:
    return L.RUNS / ("_xosc_base_cp3" if step == 10
                     else f"_xosc_base_cp3s{step}")


def _gen_step_one(args):
    """Module-level worker: MUST run in a separate PROCESS (Pool) — generation
    goes through global sampling state (set_active_sample_config) and shared
    Stage A debug dirs; threads race and corrupt each other's configs."""
    step, ego, actor, mf, xf, label = args
    cfg = SampleConfig(f"cp3s{step}", anchor="pet", placement="critdist",
                       n_points=3, step_m=float(step))
    xbase = _xbase_for(step)
    try:
        hits = list(xbase.glob(f"*_{ego}_{actor}_f{mf + 1}.xosc"))
        if hits:
            return (ego, actor, str(hits[0]), None)
        b = GEN.generate(PL.DATASET, ego, [actor], [], mf, xf,
                         label if label else L.LABEL, xbase,
                         sample_cfg=cfg, speed_model=None)
        return (ego, actor, str(b) if b else None, None)
    except Exception as e:  # noqa: BLE001
        return (ego, actor, None, f"{type(e).__name__}: {e}")


def _patch_conflict_weight(xosc: Path, w: float):
    """Set the NURBS weight of the CONFLICT interior control point only.
    The pipeline stamps weight="5.0" on the interior shape points (cp3:
    crit-d, crit, crit+d in path order); flanks keep 5.0. CAVEAT (review
    2026-08-14): sampling.py's legacy spacing guards can DROP interior
    points (incl. crit) on short/slow trajectories — only a complete 3-CP
    set guarantees middle == crit, so anything else is left unpatched with
    a loud warning rather than silently weighting a flank."""
    tree = ET.parse(xosc)
    cps = [cp for cp in tree.getroot().iter("ControlPoint")
           if cp.get("weight") is not None]
    if len(cps) == 3:
        cps[1].set("weight", f"{w:g}")
        tree.write(xosc)
    else:
        print(f"[conflict-w] SKIP {xosc.name}: {len(cps)} weighted CPs "
              f"(need 3 to identify crit) — renders at default weight",
              flush=True)


def interior_cps(step: int, sid: str, mf: int):
    """Weighted (interior/shape) NURBS control points of the agent's base
    xosc for one step — the actual sampled points crit±d (may be < 3 when
    sampling's spacing guards dropped some)."""
    hits = list(_xbase_for(step).glob(f"*_{sid}_f{mf + 1}.xosc"))
    if not hits:
        return None
    best = []
    for nb in ET.parse(hits[0]).getroot().iter("Nurbs"):
        cps = []
        for cp in nb.iter("ControlPoint"):
            if cp.get("weight") is None:
                continue
            wp = cp.find(".//WorldPosition")
            if wp is not None:
                cps.append((float(wp.get("x")), float(wp.get("y"))))
        if len(cps) > len(best):
            best = cps
    return np.array(best) if best else None


def gen_render_step(step: int, meta: pd.DataFrame):
    """Gen + render one step; returns long trajectories DataFrame."""
    xbase = _xbase_for(step)
    xbase.mkdir(parents=True, exist_ok=True)
    gen_args = [(step, int(r.ego), int(r.actor), int(r.min_frame),
                 int(r.max_frame),
                 int(r.label) if "label" in meta.columns and r.label
                 and int(r.label) not in (77, 88) else None)
                for r in meta.itertuples()]
    with Pool(6) as pool:
        gen_out = pool.map(_gen_step_one, gen_args, chunksize=2)
    bases = {f"{e}_{a}": b for (e, a, b, err) in gen_out if b}
    for (e, a, b, err) in gen_out:
        if err:
            print(f"[s{step}] gen fail {e}_{a}: {err[:120]}", flush=True)

    if CW:
        tag = f"cp3s{step}w{CW:g}base"
    else:
        tag = f"cp3s{step}base" if step != 10 else "cp3base"

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
                rows.extend(PL.trajs_to_rows(res, sid, f"cp3s{step}", "base"))
    print(f"[s{step}] gen ok={len(bases)}/{len(gen_args)} rendered "
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


def main(steps):
    meta = pd.read_csv(L.DDIR / "real_meta.csv")
    sr = pd.read_parquet(L.RDESC)
    if "error" in sr.columns:
        sr = sr[sr.error.isna()]
    rd = sr[sr.set == "real"].set_index("scenario_id")

    per_step_desc, per_step_tr, sum_rows = {}, {}, []
    for step in steps:
        t0 = time.time()
        d_file = L.RESULTS / f"cp3s{step}_descriptors{L.SUF}{WSUF}.parquet"
        t_file = (L.RESULTS / f"cp3_trajectories{L.SUF}.parquet"
                  if step == 10 and not CW
                  else L.RESULTS / f"cp3s{step}_trajectories{L.SUF}{WSUF}.parquet")
        if step == 10 and not CW and \
                (L.RESULTS / f"cp3_descriptors{L.SUF}.parquet").exists():
            dd = pd.read_parquet(L.RESULTS / f"cp3_descriptors{L.SUF}.parquet")
            if "error" in dd.columns:
                dd = dd[dd.error.isna()]
            tr = pd.read_parquet(t_file)
            n_tp = -1
        elif d_file.exists() and t_file.exists():
            dd = pd.read_parquet(d_file)
            if "error" in dd.columns:
                dd = dd[dd.error.isna()]
            tr = pd.read_parquet(t_file)
            n_tp = -1
        else:
            tr = gen_render_step(step, meta)
            tr.to_parquet(t_file)
            dd, n_tp = descriptors_for(tr)
            dd.to_parquet(d_file)
        # a step can end up with ZERO valid rows (e.g. single-scenario subset
        # whose render teleports) — keep an empty frame so the step reports
        # "invalid" instead of crashing
        per_step_desc[step] = (dd.set_index("scenario_id") if len(dd) else
                               pd.DataFrame(
                                   index=pd.Index([], name="scenario_id")))
        per_step_tr[step] = tr
        print(f"[s{step}] {len(dd)} descriptor rows (tp={n_tp}, "
              f"{time.time() - t0:.0f}s)", flush=True)

    # real interaction windows (for profile/gap shape metrics)
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

    # paired |Δ| per step
    for step in steps:
        dd = per_step_desc[step]
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
                sum_rows.append({"step": step, "key": key, "n": len(diffs),
                                 "mae_median": float(np.median(diffs)),
                                 "mae_p90": float(np.percentile(diffs, 90))})
        # interaction-window SHAPE fidelity (profile_dissim = 1 - profile_sim,
        # gap_dtw) — same conventions as 130_single_report
        tr = per_step_tr[step]
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
                sum_rows.append({"step": step, "key": key, "n": len(vals),
                                 "mae_median": float(np.median(vals)),
                                 "mae_p90": float(np.percentile(vals, 90))})
    comp = pd.DataFrame(sum_rows)
    comp.to_csv(L.RESULTS / f"stepm_sweep{L.SUF}{WSUF}.csv", index=False)
    print(comp.pivot_table(index="key", columns="step",
                           values="mae_median").round(3).to_string())

    # ---- fig12 -------------------------------------------------------------
    roads, _j = xodr_lanes.parse(XODR)
    # worst scenarios under step=10 min_dist delta for the map panels
    dd10 = per_step_desc[steps[0]]
    dsel = []
    for sid in dd10.index:
        if sid in rd.index and np.isfinite(dd10.loc[sid, "min_dist"]):
            dsel.append((abs(dd10.loc[sid, "min_dist"]
                             - rd.loc[sid, "min_dist"]), sid))
    show = [sid for _, sid in sorted(dsel, reverse=True)[:4]]

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15])
    # (a) median |Δ| vs step
    axa = fig.add_subplot(gs[0, 0:2])
    kx = np.arange(len(KEYS) + 1)
    klabs = [k.replace("agent_arr_", "arr_").replace("_", "\n")
             for k in KEYS] + ["conflict\npoint"]
    for j, step in enumerate(steps):
        d = comp[comp.step == step].set_index("key")
        norm = comp.groupby("key").mae_median.max()
        xs, ys = [], []
        for xi, k in zip(kx, KEYS + ["conflict_point"]):
            if k not in d.index:        # step invalid (teleport) for this key
                continue
            xs.append(xi)
            ys.append(d.loc[k, "mae_median"] / max(norm[k], 1e-9))
        axa.plot(np.array(xs) + (j - 1) * 0.12, ys, "o", ms=8,
                 color=STEP_COL[step], label=f"±{step} m")
        for xi, y, k in zip(xs, ys, [k for k in KEYS + ["conflict_point"]
                                     if k in d.index]):
            axa.text(xi + (j - 1) * 0.12, y + 0.03,
                     f"{d.loc[k, 'mae_median']:.2f}", fontsize=6.5,
                     ha="center", color=STEP_COL[step])
    axa.set_xticks(kx, klabs, fontsize=8)
    axa.set_ylabel("median |Δ| (per-key normalized to worst step)")
    axa.set_title("(a) GT-fidelity vs point spacing — labels = raw median |Δ|",
                  fontsize=9.5, loc="left")
    axa.legend(frameon=False, fontsize=8.5)
    axa.grid(axis="x", visible=False)
    # (b) per-scenario slopegraph (min_dist)
    axb = fig.add_subplot(gs[0, 2:4])
    for sid in dd10.index:
        ys = []
        for step in steps:
            dd = per_step_desc[step]
            v = (abs(float(dd.loc[sid, "min_dist"]) -
                     float(rd.loc[sid, "min_dist"]))
                 if sid in dd.index and sid in rd.index else np.nan)
            ys.append(v)
        axb.plot(range(len(steps)), ys, "-o", ms=4, lw=1.0, color="#9a9a94",
                 alpha=0.6)
    med = [(comp[(comp.step == s) & (comp.key == "min_dist")].mae_median.iloc[0]
            if len(comp[(comp.step == s) & (comp.key == "min_dist")])
            else np.nan) for s in steps]
    axb.plot(range(len(steps)), med, "-o", ms=9, lw=2.5, color=INK,
             label="median")
    axb.set_xticks(range(len(steps)), [f"±{s} m" for s in steps])
    axb.set_ylabel("per-scenario |Δ min_dist| [m]")
    axb.set_title("(b) min_dist error per scenario across spacings",
                  fontsize=9.5, loc="left")
    axb.legend(frameon=False, fontsize=8.5)
    axb.grid(axis="x", visible=False)
    # (c) map overlays
    tr_by_step = {s: {sid: g.sort_values("frame") for (sid, role), g in
                      per_step_tr[s].groupby(["scenario_id", "role"])
                      if role == "agent"} for s in steps}
    for i, sid in enumerate(show):
        ax = fig.add_subplot(gs[1, i])
        for road in roads.values():
            try:
                for _lid, (inner, outer) in road.lane_edges(ds=0.5).items():
                    ax.plot(inner[:, 0], inner[:, 1], color=LANE, lw=0.6,
                            zorder=0)
                    ax.plot(outer[:, 0], outer[:, 1], color=LANE, lw=0.6,
                            zorder=0)
            except Exception:  # noqa: BLE001
                continue
        a = R[(sid, "actor")]
        ax.plot(a.x, a.y, color=INK, lw=2.4, zorder=3, label="GT agent")
        ax.plot(a.x.iloc[0], a.y.iloc[0], "o", ms=6, color=INK, zorder=4)
        mf_sid = int(meta.set_index("scenario_id").loc[sid, "min_frame"])
        for step in steps:
            g = tr_by_step[step].get(sid)
            if g is not None:
                ax.plot(g.x, g.y, color=STEP_COL[step], lw=1.6, ls="--",
                        zorder=4, label=f"±{step} m")
            cps = interior_cps(step, sid, mf_sid)
            if cps is not None:
                ax.plot(cps[:, 0], cps[:, 1], "s", ms=5.5,
                        mfc=STEP_COL[step], mec="white", mew=0.7, ls="none",
                        zorder=6)
        if sid in rd.index and np.isfinite(rd.loc[sid, "conflict_x"]):
            ax.plot(rd.loc[sid, "conflict_x"], rd.loc[sid, "conflict_y"],
                    "x", ms=10, color="#c22e2e", mew=2.2, zorder=6)
        pad = 12
        ax.set_xlim(a.x.min() - pad, a.x.max() + pad)
        ax.set_ylim(a.y.min() - pad, a.y.max() + pad)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(sid, fontsize=9, loc="left")
        if i == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{L.NAME}{WSUF and f' (conflict-CP weight {CW:g})'} — cp3 "
                 f"point-spacing sweep "
                 f"({' / '.join(f'±{s} m' for s in steps)}): fidelity of the "
                 f"default render vs GT", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(L.FIGS / f"fig12_stepm{L.SUF}{WSUF}.png", dpi=150)
    print(f"saved fig12_stepm{L.SUF}{WSUF}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="10,3,2")
    main([int(s) for s in ap.parse_args().steps.split(",")])
