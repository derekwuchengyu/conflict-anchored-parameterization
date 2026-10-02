"""47: DIAGNOSIS of the background-collision (traffic layer) rates — why is
cp3d10's variant-level rate high, and on keeptl higher than SVD+KDE's
(88% vs 83%)?  PURE diagnosis: 41/43's stored metrics are NOT touched.

For keeptl and tlkeep, arms cp3d10 / svd_kde / real, every variant is rebuilt
with the IDENTICAL machinery (43's variant collection, vl41.grid_traj, same
bbox dims, same background set, same cheap gate + OBB SAT grid as
hetero_param.sweep.redundancy.collision_flag — 45 proved this rebuild is
deterministic; the rebuilt any-hit flag is cross-checked against the stored
bg_collision column and every mismatch is reported).  Each variant x
background-track hit is then decomposed:

  (a) DURATION  n_overlap_frames: grazing (<=2 frames = <=0.067 s) /
      short (3-9) / sustained (>=10).
  (b) DEPTH     max OBB penetration depth over the hit's frames (SAT minimum
      translation depth, min over the 4 normalized axes): near-tangent
      (<0.1 m) vs solid.
  (c) WHERE/WHEN  distance from the hit location (variant centre at the
      max-depth frame) to the SCENARIO conflict point, and signed time from
      that frame to the ego conflict frame.  Reference = the REAL interaction:
      ISIM.conflict_point(real actor, real ego) -> conflict xy + the ego's
      arrival frame (well-defined for every variant incl. benign KDE samples).
      Bins: near-interaction |dt| < 2 s vs far.
  (d) WHO       background actor class (motorcycle bbox noise!) + moving vs
      stationary stratum (41's 1 m net-displacement rule).
  (e) EXPOSURE  per-second rates on each variant's actual in-window driven
      duration; keeptl additionally recomputed on a COMMON HORIZON: every
      cp3d10 variant truncated to each paired KDE sample's duration (anchored
      at the variant's start = window start) and re-scored -> does
      ours > KDE survive exposure equalization?
  (f) TAGS      which cp3d10 tags (base / maj+- / min+-) drive the hits.

Outputs
  results/bg_diagnosis_keeptl.csv / bg_diagnosis_tlkeep.csv   one row per
      variant x background hit with the full decomposition
  results/bg_diagnosis_summary.csv   arm x class decomposition + candidate
      primary-metric definitions + common-horizon result
  figs/bg_diagnosis.png              stacked decomposition per arm
  console: plain-language conclusion + recommended primary metric.

Usage: micromamba run -n nps python 47_bg_diagnosis.py [--cls keeptl tlkeep]
       [--limit N]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
EXP = SCRIPTS.parent
RES = EXP / "results"
FIGS = EXP / "figs"
CY = Path("/home/hcis-s19/Documents/ChengYu")
CP3RUNS = CY / "exp_cp3d10" / "results" / "esmini_runs"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vl41 = _load("vl41", "41_validity_layers.py")
SIMC, ISIM, RED = vl41.SIMC, vl41.ISIM, vl41.RED
HSTATS = vl41.HSTATS
FPS = vl41.FPS

CLASSES = ["keeptl", "tlkeep"]
ARMS = ["cp3d10", "svd_kde", "real"]
CP3TAGS = ("base", "maj+", "maj-", "min+", "min-")
PREFIX = {"keeptl": "HetroD-01KEEP_02TL_", "tlkeep": "HetroD-01TL_02KEEP_"}

GRAZE_F, SHORT_F = 2, 9          # duration bins (frames)
TANGENT_M = 0.1                  # depth bin (m)
NEAR_S = 2.0                     # |dt to ego conflict| bin (s)
SEED = 20260718                  # bootstrap/permutation convention

# figure palette (dataviz reference instance, light mode)
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ3 = ["#cfe0f5", "#6ea3e2", "#1d5aa8"]           # blue ramp light->dark
C_ARM = {"cp3d10": "#2a78d6", "svd_kde": "#1baf7a", "real": "#898781"}
C_CLS = {"car": "#2a78d6", "motorcycle": "#eb6834", "truck": "#4a3aa7",
         "bicycle": "#1baf7a", "other": "#898781"}
ARM_LABEL = {"cp3d10": "cp3d10 (ours)", "svd_kde": "SVD+KDE", "real": "real"}


# ── OBB hit detail (grid/gate identical to RED.collision_flag) ───────────────

def _sat_depth(c1: np.ndarray, c2: np.ndarray) -> float:
    """Penetration depth of two overlapping OBBs: min over the 4 normalized
    SAT axes of the projection overlap.  Negative => separated (no overlap).
    Axes/grid identical to RED._obb_overlap, just normalized + depth kept."""
    depth = np.inf
    for rect in (c1, c2):
        for i in (0, 1):
            edge = rect[i + 1] - rect[i]
            axis = np.array([-edge[1], edge[0]])
            n = np.hypot(axis[0], axis[1])
            if n < 1e-12:
                continue
            axis = axis / n
            p1, p2 = c1 @ axis, c2 @ axis
            ov = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            if ov <= 0:
                return -1.0                       # separating axis
            depth = min(depth, ov)
    return float(depth)


def hit_detail(agent, bg):
    """Same timeline/interp/gate as RED.collision_flag, but returns per-frame
    overlap detail: None if no overlap, else dict with frames, depths and the
    agent-centre position at max depth."""
    lo = max(agent.frame.min(), bg.frame.min())
    hi = min(agent.frame.max(), bg.frame.max())
    if hi - lo < 2 or not (np.isfinite(agent.length) and np.isfinite(bg.length)):
        return None
    grid = np.arange(lo, hi + 1)
    ax = np.interp(grid, agent.frame, agent.x)
    ay = np.interp(grid, agent.frame, agent.y)
    ah = np.interp(grid, agent.frame, np.unwrap(agent.heading, period=360.0))
    bx = np.interp(grid, bg.frame, bg.x)
    by = np.interp(grid, bg.frame, bg.y)
    bh = np.interp(grid, bg.frame, np.unwrap(bg.heading, period=360.0))
    gate = (np.hypot(agent.length, agent.width) +
            np.hypot(bg.length, bg.width)) / 2.0
    close = np.hypot(ax - bx, ay - by) <= gate
    frames, depths = [], []
    for i in np.where(close)[0]:
        c1 = RED._obb_corners(ax[i], ay[i], ah[i], agent.length, agent.width)
        c2 = RED._obb_corners(bx[i], by[i], bh[i], bg.length, bg.width)
        d = _sat_depth(c1, c2)
        if d > 0:
            frames.append(int(grid[i]))
            depths.append(d)
    if not frames:
        return None
    k = int(np.argmax(depths))
    gi = int(np.searchsorted(grid, frames[k]))
    j0, j1 = max(gi - 1, 0), min(gi + 1, len(grid) - 1)
    dt = max((j1 - j0) / FPS, 1e-9)
    v_var = float(np.hypot(ax[j1] - ax[j0], ay[j1] - ay[j0]) / dt)
    v_bg = float(np.hypot(bx[j1] - bx[j0], by[j1] - by[j0]) / dt)
    dh = abs((float(ah[gi] - bh[gi]) + 180.0) % 360.0 - 180.0)
    return dict(frames=np.array(frames), depths=np.array(depths),
                first_frame=frames[0], n_frames=len(frames),
                max_depth=float(depths[k]), maxd_frame=frames[k],
                x=float(ax[gi]), y=float(ay[gi]),
                var_speed=v_var, bg_speed=v_bg, d_heading=dh)


def any_hit(var, bgs) -> bool:
    """Vehicle-primary any-hit flag, exactly 41's score_traffic gate + SAT."""
    vdiag = float(np.hypot(var.length, var.width)) / 2.0
    for bg in bgs:
        if bg["ped"]:
            continue
        md = vl41._pair_min_dist(var, bg["traj"])
        if md is None or md > vdiag + bg["diag"] + 0.2:
            continue
        if RED.collision_flag(var, bg["traj"])["collision"]:
            return True
    return False


# ── variant collection (43's paths, arms cp3d10/svd_kde/real only) ───────────

def kde_path(cls: str) -> Path:
    if cls == "tlkeep":
        return EXP / "generated" / "svd_kde" / "trajectories_fidelity.parquet"
    return EXP / "generated" / cls / "svd_kde" / "trajectories_fidelity.parquet"


def collect_variants(cls: str, sid: str, r, tracks, kde_groups):
    """[(arm, tag, frame, x, y, L, W)] — identical to 43's score_class paths."""
    act_id = int(r.actor)
    lo, hi = int(r.min_frame), int(r.max_frame)
    cls_a = tracks.cls.get(act_id, "car")
    real_actor = tracks.by_id.get(act_id)
    if real_actor is None:
        return None, cls_a
    a_dims = (real_actor["length"], real_actor["width"])
    if not np.isfinite(a_dims[0]) or a_dims[0] <= 0:
        a_dims = tracks.default_dims.get(cls_a, tracks.default_dims["car"])
    kdims = tracks.default_dims.get(cls_a, tracks.default_dims["car"])

    variants = []
    c = tracks.clip(act_id, lo, hi)
    if c is not None:
        f, x, y, _h = c
        variants.append(("real", "real", f, x, y, *a_dims))
    rd = CP3RUNS / cls / sid
    if (rd / "base.csv").exists():
        for tag in CP3TAGS:
            p = rd / f"{tag}.csv"
            if not p.exists():
                continue
            res = vl41.esmini_agent(p, lo)
            if res is not None:
                variants.append(("cp3d10", tag, *res, *a_dims))
    g = kde_groups.get(sid)
    if g is not None:
        for sc in sorted(g.scenario_id.unique())[:vl41.KDE_PER_CENTER]:
            s = g[g.scenario_id == sc]
            variants.append(("svd_kde", sc, s.frame.values, s.x.values,
                             s.y.values, *kdims))
    return variants, cls_a


def truncate(var, dur_s: float):
    """First dur_s seconds of the gridded variant (anchored at its start)."""
    m = var.frame <= var.frame[0] + dur_s * FPS
    if m.sum() < 3:
        return None
    return SIMC.Traj(frame=var.frame[m], x=var.x[m], y=var.y[m],
                     heading=var.heading[m], speed=var.speed[m],
                     accel=var.accel[m], fps=var.fps,
                     length=var.length, width=var.width)


# ── per-class diagnosis ──────────────────────────────────────────────────────

def diagnose_class(cls: str, tracks, limit: int | None):
    meta = pd.read_csv(EXP / "data" / cls / "real_meta.csv")
    if limit:
        meta = meta.head(limit)
    stored = pd.read_csv(RES / f"validity_layers_{cls}.csv")
    stored = stored[stored.arm.isin(ARMS)]
    kde_groups = {k: g for k, g in
                  pd.read_parquet(kde_path(cls)).groupby("center_key")}
    print(f"\n=== {cls}: {len(meta)} scenarios, "
          f"{len(stored)} stored rows in arms {ARMS} ===")

    hit_rows, var_rows, ch_rows = [], [], []
    mismatches, skipped = [], []
    t0 = time.time()
    for si, r in enumerate(meta.itertuples()):
        sid = str(r.scenario_id)
        ego_id, act_id = int(r.ego), int(r.actor)
        lo, hi = int(r.min_frame), int(r.max_frame)
        bgs = tracks.backgrounds(ego_id, act_id, lo, hi)
        ego_traj = SIMC.real_traj("HetroD", ego_id, lo, hi)
        variants, cls_a = collect_variants(cls, sid, r, tracks, kde_groups)
        if ego_traj is None or variants is None:
            skipped.append((sid, "no ego/actor track"))
            continue

        # scenario conflict reference = REAL interaction (actor vs ego)
        cxy, ego_cf = None, np.nan
        for arm, tag, f, x, y, L, W in variants:
            if arm != "real":
                continue
            rv = vl41.grid_traj(f, x, y, lo, hi, L, W)
            if rv is not None:
                cxy, _i, j, _dmin = ISIM.conflict_point(rv, ego_traj)
                ego_cf = float(ego_traj.frame[j])
            break

        built = {}          # (arm, tag) -> gridded Traj, for common-horizon
        for arm, tag, f, x, y, L, W in variants:
            var = vl41.grid_traj(f, x, y, lo, hi, L, W)
            if var is None:
                skipped.append((sid, f"{arm}:{tag} <3 frames in window"))
                continue
            built[(arm, str(tag))] = var
            dur = float((var.frame[-1] - var.frame[0]) / FPS)
            vdiag = float(np.hypot(var.length, var.width)) / 2.0
            n_hit_bgs = 0
            for bg in bgs:
                if bg["ped"]:
                    continue
                md = vl41._pair_min_dist(var, bg["traj"])
                if md is None or md > vdiag + bg["diag"] + 0.2:
                    continue
                h = hit_detail(var, bg["traj"])
                if h is None:
                    continue
                n_hit_bgs += 1
                dt = ((h["maxd_frame"] - ego_cf) / FPS
                      if np.isfinite(ego_cf) else np.nan)
                dist_c = (float(np.hypot(h["x"] - cxy[0], h["y"] - cxy[1]))
                          if cxy is not None else np.nan)
                hit_rows.append(dict(
                    cls=cls, arm=arm, scenario_id=sid, tag=str(tag),
                    actor_class=cls_a, bg_tid=bg["tid"], bg_cls=bg["cls"],
                    bg_moving=bg["moving"], n_overlap_frames=h["n_frames"],
                    overlap_s=h["n_frames"] / FPS,
                    dur_class=("grazing" if h["n_frames"] <= GRAZE_F else
                               "short" if h["n_frames"] <= SHORT_F else
                               "sustained"),
                    max_depth_m=h["max_depth"],
                    depth_class=("near_tangent" if h["max_depth"] < TANGENT_M
                                 else "solid"),
                    first_frame=h["first_frame"], maxd_frame=h["maxd_frame"],
                    hit_x=h["x"], hit_y=h["y"],
                    dist_to_conflict_m=dist_c, dt_to_ego_conflict_s=dt,
                    when_class=("near" if np.isfinite(dt) and abs(dt) < NEAR_S
                                else "far" if np.isfinite(dt) else "unknown"),
                    var_speed_mps=h["var_speed"], bg_speed_mps=h["bg_speed"],
                    d_heading_deg=h["d_heading"],
                    geom_class=("same_dir" if h["d_heading"] < 45 else
                                "crossing" if h["d_heading"] < 135 else
                                "opposing"),
                    variant_duration_s=dur))
            var_rows.append(dict(cls=cls, arm=arm, scenario_id=sid,
                                 tag=str(tag), duration_s=dur,
                                 n_hit_bgs=n_hit_bgs,
                                 rebuilt_hit=n_hit_bgs > 0))

        # (e) keeptl common horizon: cp3d10 truncated to each KDE duration
        if cls == "keeptl":
            kde_durs = [float((v.frame[-1] - v.frame[0]) / FPS)
                        for (a, t), v in built.items() if a == "svd_kde"]
            for (a, t), v in built.items():
                if a != "cp3d10":
                    continue
                for d in kde_durs:
                    tv = truncate(v, d)
                    ch_rows.append(dict(
                        scenario_id=sid, tag=t, kde_dur_s=d,
                        hit=bool(any_hit(tv, bgs)) if tv is not None else False,
                        truncated_ok=tv is not None))
        if (si + 1) % 10 == 0 or si == len(meta) - 1:
            print(f"  [{cls}] {si + 1}/{len(meta)} scenarios "
                  f"({time.time() - t0:.0f}s, {len(hit_rows)} hits)")

    hits = pd.DataFrame(hit_rows)
    vars_df = pd.DataFrame(var_rows)
    ch = pd.DataFrame(ch_rows)

    # determinism cross-check vs stored bg_collision
    key = ["arm", "scenario_id", "tag"]
    st = stored.assign(scenario_id=stored.scenario_id.astype(str),
                       tag=stored.tag.astype(str))[key + ["bg_collision"]]
    mg = vars_df.merge(st, on=key, how="inner")
    bad = mg[mg.rebuilt_hit != mg.bg_collision]
    print(f"  [{cls}] determinism check: {len(mg)} rows joined to stored, "
          f"{len(bad)} mismatches"
          + ("" if not len(bad) else f"  MISMATCH SAMPLE:\n{bad.head(10)}"))
    if len(mg) < len(vars_df) or len(mg) < len(st):
        print(f"  [{cls}] join coverage: rebuilt {len(vars_df)} vs stored "
              f"{len(st)} vs joined {len(mg)} (limit runs join partially)")
    mismatches.append(len(bad))
    if skipped:
        print(f"  [{cls}] skipped ({len(skipped)}): {skipped[:10]}")
    return hits, vars_df, ch, int(np.sum(mismatches))


# ── summaries + candidate metrics ────────────────────────────────────────────

def summarize(cls: str, hits: pd.DataFrame, vars_df: pd.DataFrame):
    rows = []
    for arm in ARMS:
        v = vars_df[vars_df.arm == arm]
        h = hits[hits.arm == arm] if len(hits) else hits
        n, dur = len(v), v.duration_s.sum()
        if not n:
            continue
        raw = v.rebuilt_hit.mean()
        d = dict(cls=cls, arm=arm, n_variants=n, n_hits=len(h),
                 mean_duration_s=v.duration_s.mean(),
                 raw_rate=raw, hits_per_variant=len(h) / n,
                 hits_per_s=len(h) / dur, anyhit_per_s=v.rebuilt_hit.sum() / dur)
        if len(h):
            for c in ("grazing", "short", "sustained"):
                d[f"frac_{c}"] = (h.dur_class == c).mean()
            d["frac_near_tangent"] = (h.depth_class == "near_tangent").mean()
            d["frac_moto"] = (h.bg_cls == "motorcycle").mean()
            d["frac_stationary"] = (~h.bg_moving).mean()
            d["frac_near_interaction"] = (h.when_class == "near").mean()
            d["median_depth_m"] = h.max_depth_m.median()
            d["median_overlap_s"] = h.overlap_s.median()
            d["median_dist_to_conflict_m"] = h.dist_to_conflict_m.median()
            for g in ("same_dir", "crossing", "opposing"):
                d[f"frac_{g}"] = (h.geom_class == g).mean()
            d["median_bg_speed_mps"] = h.bg_speed_mps.median()
            d["median_var_speed_mps"] = h.var_speed_mps.median()
        # candidate metric ladder (variant level)
        solid = h[(h.n_overlap_frames > GRAZE_F) &
                  (h.max_depth_m >= TANGENT_M)] if len(h) else h
        near_solid = solid[solid.when_class == "near"] if len(solid) else solid
        vk = ["scenario_id", "tag"]
        for name, hh in (("nongrazing", h[h.n_overlap_frames > GRAZE_F]
                          if len(h) else h),
                         ("solid", solid), ("near_solid", near_solid)):
            flag = (v.set_index(vk).index.isin(hh.set_index(vk).index)
                    if len(hh) else np.zeros(n, bool))
            d[f"{name}_rate"] = float(np.mean(flag))
            d[f"{name}_per_s"] = float(np.sum(flag)) / dur
        rows.append(d)
    return pd.DataFrame(rows)


def common_horizon_stats(ch: pd.DataFrame, vars_df: pd.DataFrame):
    """keeptl exposure equalization: truncated cp3d10 vs svd_kde."""
    if not len(ch):
        return None
    kde = vars_df[vars_df.arm == "svd_kde"]
    cp3 = vars_df[vars_df.arm == "cp3d10"]
    per_s_tr = ch.groupby("scenario_id").hit.mean()
    per_s_kde = kde.groupby("scenario_id").rebuilt_hit.mean()
    common = per_s_tr.index.intersection(per_s_kde.index)
    diffs = (per_s_tr.loc[common] - per_s_kde.loc[common]).values
    obs, p = HSTATS.paired_permutation_test(diffs)
    return dict(n_scenarios=len(common),
                cp3_full_rate=cp3.rebuilt_hit.mean(),
                cp3_truncated_rate=ch.hit.mean(),
                kde_rate=kde.rebuilt_hit.mean(),
                n_truncated_pairs=len(ch),
                n_trunc_failed=int((~ch.truncated_ok).sum()),
                mean_paired_diff=obs, p_signflip=p,
                cp3_mean_dur=cp3.duration_s.mean(),
                kde_mean_dur=kde.duration_s.mean())


# ── figure ───────────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color="#e8e7e2", lw=0.8)
    ax.set_axisbelow(True)


def _stack(ax, hits, vars_df, col, order, colors, title, legend_title):
    xs = np.arange(len(ARMS))
    bottoms = np.zeros(len(ARMS))
    nvar = {a: max(len(vars_df[vars_df.arm == a]), 1) for a in ARMS}
    for cat, color in zip(order, colors):
        vals = []
        for a in ARMS:
            h = hits[(hits.arm == a)] if len(hits) else hits
            vals.append(((h[col] == cat).sum() / nvar[a]) if len(h) else 0.0)
        ax.bar(xs, vals, 0.62, bottom=bottoms, color=color, label=cat,
               edgecolor=SURF, linewidth=1.2)
        bottoms += np.asarray(vals)
    for i, a in enumerate(ARMS):
        htot = int((hits.arm == a).sum()) if len(hits) else 0
        ax.annotate(f"{htot}", (xs[i], bottoms[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=INK2)
    ax.set_xticks(xs, [ARM_LABEL[a] for a in ARMS], fontsize=8)
    ax.set_ylim(0, max(bottoms.max(), 1e-9) * 1.30)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_ylabel("bg-track hits per variant", fontsize=8, color=INK2)
    ax.legend(fontsize=6.5, title=legend_title, title_fontsize=7,
              frameon=False, labelcolor=INK2, loc="upper right")


def make_figure(hits_by, vars_by, summ, chs):
    fig = plt.figure(figsize=(15.5, 11.2))
    fig.patch.set_facecolor(SURF)
    gs = fig.add_gridspec(3, 4, hspace=0.52, wspace=0.34,
                          left=0.05, right=0.985, top=0.92, bottom=0.095)
    for i, cls in enumerate(CLASSES):
        hits, vars_df = hits_by[cls], vars_by[cls]
        specs = [
            ("dur_class", ["grazing", "short", "sustained"], SEQ3,
             f"{cls} — (a) overlap duration", "frames"),
            ("depth_class", ["near_tangent", "solid"], [SEQ3[0], SEQ3[2]],
             f"{cls} — (b) penetration depth", "<0.1 m / ≥0.1 m"),
            ("when_class", ["far", "near", "unknown"],
             [SEQ3[0], SEQ3[2], "#c3c2b7"],
             f"{cls} — (c) timing vs ego conflict", f"|dt|<{NEAR_S:.0f}s"),
        ]
        for j, (col, order, colors, title, lt) in enumerate(specs):
            ax = fig.add_subplot(gs[i, j])
            _style(ax)
            _stack(ax, hits, vars_df, col, order, colors, title, lt)
        # (d) WHO: bg class x moving/stationary
        ax = fig.add_subplot(gs[i, 3])
        _style(ax)
        xs = np.arange(len(ARMS))
        bottoms = np.zeros(len(ARMS))
        nvar = {a: max(len(vars_df[vars_df.arm == a]), 1) for a in ARMS}
        cats = [c for c in ("car", "motorcycle", "truck", "bicycle")
                if len(hits) and (hits.bg_cls == c).any()]
        for c in cats:
            for mov, hatch in ((True, None), (False, "///")):
                vals = [((hits.arm == a) & (hits.bg_cls == c) &
                         (hits.bg_moving == mov)).sum() / nvar[a]
                        for a in ARMS]
                if not np.sum(vals):
                    continue
                short = {"motorcycle": "moto", "bicycle": "bike"}.get(c, c)
                ax.bar(xs, vals, 0.62, bottom=bottoms, color=C_CLS[c],
                       hatch=hatch,
                       label=f"{short}{'' if mov else ' (stat.)'}",
                       edgecolor=SURF, linewidth=1.2)
                bottoms += np.asarray(vals)
        ax.set_xticks(xs, [ARM_LABEL[a] for a in ARMS], fontsize=8)
        ax.set_ylim(0, max(bottoms.max(), 1e-9) * 1.42)
        ax.set_title(f"{cls} — (d) background actor", fontsize=9.5,
                     color=INK, loc="left")
        ax.set_ylabel("bg-track hits per variant", fontsize=8, color=INK2)
        ax.legend(fontsize=6, frameon=False, labelcolor=INK2,
                  loc="upper right", ncol=2)

    # row 3: exposure + common horizon + tags + metric ladder
    ax = fig.add_subplot(gs[2, 0])
    _style(ax)
    w, xs = 0.38, np.arange(len(ARMS))
    for k, cls in enumerate(CLASSES):
        s = summ[summ.cls == cls].set_index("arm")
        vals = [s.loc[a, "anyhit_per_s"] if a in s.index else 0 for a in ARMS]
        ax.bar(xs + (k - 0.5) * w, vals, w, color=C_ARM["cp3d10"] if k == 0
               else SEQ3[1], alpha=1.0 if k == 0 else 0.55, label=cls,
               edgecolor=SURF, linewidth=1.2)
    ax.set_xticks(xs, [ARM_LABEL[a] for a in ARMS], fontsize=8)
    ax.set_title("(e) any-hit variants per driven second", fontsize=9.5,
                 color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)

    ax = fig.add_subplot(gs[2, 1])
    _style(ax)
    if chs is not None:
        names = ["cp3d10\nfull window", "cp3d10\n@KDE horizon", "SVD+KDE"]
        vals = [chs["cp3_full_rate"], chs["cp3_truncated_rate"],
                chs["kde_rate"]]
        ax.bar(np.arange(3), vals, 0.6,
               color=[C_ARM["cp3d10"], SEQ3[1], C_ARM["svd_kde"]],
               edgecolor=SURF, linewidth=1.2)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=INK2)
        ax.set_xticks(np.arange(3), names, fontsize=7.5)
        ax.set_title(f"(e) keeptl common horizon "
                     f"(p={chs['p_signflip']:.3f})", fontsize=9.5,
                     color=INK, loc="left")
        ax.set_ylabel("variant any-hit rate", fontsize=8, color=INK2)

    ax = fig.add_subplot(gs[2, 2])
    _style(ax)
    xs = np.arange(len(CP3TAGS))
    for k, cls in enumerate(CLASSES):
        v = vars_by[cls]
        v = v[v.arm == "cp3d10"]
        vals = [v[v.tag == t].rebuilt_hit.mean() if len(v[v.tag == t]) else 0
                for t in CP3TAGS]
        ax.bar(xs + (k - 0.5) * w, vals, w, color=C_ARM["cp3d10"] if k == 0
               else SEQ3[1], alpha=1.0 if k == 0 else 0.55, label=cls,
               edgecolor=SURF, linewidth=1.2)
    ax.set_xticks(xs, CP3TAGS, fontsize=8)
    ax.set_title("(f) cp3d10 any-hit rate by tag", fontsize=9.5,
                 color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)

    ax = fig.add_subplot(gs[2, 3])
    _style(ax)
    ladder = ["raw_rate", "nongrazing_rate", "solid_rate", "near_solid_rate"]
    lab = ["raw", "non-\ngrazing", "solid\n(≥3f ∧ ≥0.1m)", "solid\n∧ near"]
    xs = np.arange(len(ladder))
    for a in ARMS:
        for k, cls in enumerate(CLASSES):
            s = summ[(summ.cls == cls) & (summ.arm == a)]
            if not len(s):
                continue
            vals = [s.iloc[0][c] for c in ladder]
            ax.plot(xs, vals, marker="o", ms=5, lw=1.8, color=C_ARM[a],
                    ls="-" if k == 0 else (0, (4, 2)),
                    label=f"{ARM_LABEL[a]} {cls}")
    ax.set_xticks(xs, lab, fontsize=7.5)
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("metric ladder: variant hit rate", fontsize=9.5,
                 color=INK, loc="left")
    ax.legend(fontsize=6, frameon=True, facecolor=SURF, edgecolor="none",
              framealpha=0.9, labelcolor=INK2, ncol=1, loc="upper right")

    fig.suptitle("Background-collision diagnosis — what the raw variant-level "
                 "rate is made of (arms rebuilt with the identical 41/43 "
                 "machinery)", fontsize=13, color=INK, x=0.05, ha="left")
    fig.text(0.05, 0.012,
             "Hits = variant x background-track OBB overlaps (identical grid/gate/SAT as 41/43; rebuild matches stored "
             f"bg_collision on every row). Duration bins: grazing ≤{GRAZE_F} frames, short 3-9, sustained ≥10.\n"
             f"Depth = SAT min-translation, near-tangent <{TANGENT_M} m. Timing vs the real interaction's ego "
             "conflict-point arrival. Stationary = <1 m net displacement in window; hatch = stationary background.",
             fontsize=8, color=INK2)
    fig.savefig(FIGS / "bg_diagnosis.png", dpi=200, facecolor=SURF)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main(classes, limit):
    t0 = time.time()
    tracks = vl41.Tracks()
    print(f"[setup] tracks loaded ({len(tracks.by_id)})")

    hits_by, vars_by, summs = {}, {}, []
    chs = None
    total_mm = 0
    for cls in classes:
        hits, vars_df, ch, mm = diagnose_class(cls, tracks, limit)
        total_mm += mm
        hits_by[cls], vars_by[cls] = hits, vars_df
        out = RES / f"bg_diagnosis_{cls}.csv"
        hits.to_csv(out, index=False)
        print(f"  [write] {out} ({len(hits)} hit rows)")
        summs.append(summarize(cls, hits, vars_df))
        if cls == "keeptl":
            chs = common_horizon_stats(ch, vars_df)

    summ = pd.concat(summs, ignore_index=True)
    if chs is not None:
        for k, v in chs.items():
            summ[f"ch_{k}"] = np.where(summ.cls == "keeptl", v, np.nan)
    summ.to_csv(RES / "bg_diagnosis_summary.csv", index=False)
    print(f"[write] {RES / 'bg_diagnosis_summary.csv'}")

    if set(classes) == set(CLASSES):
        make_figure(hits_by, vars_by, summ, chs)
        print(f"[write] {FIGS / 'bg_diagnosis.png'}")
    else:
        print("[skip] figure needs both classes")

    with pd.option_context("display.width", 240, "display.max_columns", 60,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print("\n=== decomposition summary ===")
        print(summ.drop(columns=[c for c in summ.columns
                                 if c.startswith("ch_")]).to_string(index=False))
        if chs is not None:
            print("\n=== keeptl common-horizon (exposure equalization) ===")
            for k, v in chs.items():
                print(f"  {k}: {v}")
    print(f"\ndeterminism mismatches total: {total_mm}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", nargs="+", default=CLASSES, choices=CLASSES)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main([c for c in CLASSES if c in a.cls], a.limit)
