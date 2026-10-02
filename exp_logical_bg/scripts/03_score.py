"""03: score the 50-vs-50 arms with the full Task-C background-collision
decomposition + ego criticality, on three horizons:

  full   each sample's full in-window trajectory (the raw class-level metric);
  ch     COMMON HORIZON (primary, per the experiment design): every trajectory
         truncated to the SAME time span = min driven duration over ALL 100
         samples of the scenario, anchored at the labeled WINDOW START
         (absolute frame <= min_frame + dur*FPS) — exposure exactly equal;
  chmed  sensitivity horizon: min of the two methods' MEDIAN durations
         (robust to single short-KDE outliers; samples shorter than it keep
         their full length).

Machinery is byte-identical to the class-level comparisons: vl41.grid_traj /
Tracks / score_ego, bg47.hit_detail (SAT depth = exact OBB minimum translation)
on the 1/30 s interpolation grid.  Conflict reference = the SCENARIO-level real
interaction (ISIM.conflict_point of real actor vs real ego + ego arrival
frame), as recommended by the Task-C diagnosis.  The real actor is pushed
through the identical loop as the zero reference.

Per-sample flags (per horizon)
  any_hit      >=1 OBB overlap with any background vehicle track
  any_solid    PRIMARY hit metric: >=1 hit with >2 overlap frames AND
               SAT depth >= 0.1 m (the diagnosis "solid" ladder rung —
               excludes grazing/near-tangent bbox noise)
  any_near_solid  solid hit within |dt| < 2 s of the real ego-conflict arrival
  ego_critical band per class: keeptl/tlkeep minTTC<1.5 s OR |PET|<1 s;
               cutin minTTC<1.5 s ONLY
  bg_clean_critical = (not any_solid) AND ego_critical

Outputs: results/samples.csv, results/hits.csv, results/summary.csv
Usage: micromamba run -n nps python 03_score.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lbg_lib as L  # noqa: E402

vl = L.vl41()
bg47 = L.bg47()
SIMC, ISIM = vl.SIMC, vl.ISIM
FPS = L.FPS


def trunc_abs(var, lo: int, dur_s: float):
    """Truncate to absolute frame <= lo + dur_s*FPS (window-start anchor)."""
    m = var.frame <= lo + dur_s * FPS
    if m.sum() < 3:
        return None
    return SIMC.Traj(frame=var.frame[m], x=var.x[m], y=var.y[m],
                     heading=var.heading[m], speed=var.speed[m],
                     accel=var.accel[m], fps=var.fps,
                     length=var.length, width=var.width)


def score_variant(var, bgs, cxy, ego_cf, cls_a):
    """Task-C decomposition of one gridded variant vs all background vehicles."""
    vdiag = float(np.hypot(var.length, var.width)) / 2.0
    hits = []
    for bg in bgs:
        if bg["ped"]:
            continue
        md = vl._pair_min_dist(var, bg["traj"])
        if md is None or md > vdiag + bg["diag"] + 0.2:
            continue
        h = bg47.hit_detail(var, bg["traj"])
        if h is None:
            continue
        dt = ((h["maxd_frame"] - ego_cf) / FPS if np.isfinite(ego_cf)
              else np.nan)
        dist_c = (float(np.hypot(h["x"] - cxy[0], h["y"] - cxy[1]))
                  if cxy is not None else np.nan)
        hits.append(dict(
            actor_class=cls_a, bg_tid=bg["tid"], bg_cls=bg["cls"],
            bg_moving=bg["moving"], n_overlap_frames=h["n_frames"],
            overlap_s=h["n_frames"] / FPS,
            dur_class=("grazing" if h["n_frames"] <= L.GRAZE_F else
                       "short" if h["n_frames"] <= L.SHORT_F else "sustained"),
            max_depth_m=h["max_depth"],
            depth_class=("near_tangent" if h["max_depth"] < L.TANGENT_M
                         else "solid"),
            first_frame=h["first_frame"], maxd_frame=h["maxd_frame"],
            hit_x=h["x"], hit_y=h["y"],
            dist_to_conflict_m=dist_c, dt_to_ego_conflict_s=dt,
            when_class=("near" if np.isfinite(dt) and abs(dt) < L.NEAR_S
                        else "far" if np.isfinite(dt) else "unknown"),
            var_speed_mps=h["var_speed"], bg_speed_mps=h["bg_speed"],
            d_heading_deg=h["d_heading"],
            geom_class=("same_dir" if h["d_heading"] < 45 else
                        "crossing" if h["d_heading"] < 135 else "opposing")))
    solid = [h for h in hits if h["n_overlap_frames"] > L.GRAZE_F
             and h["max_depth_m"] >= L.TANGENT_M]
    near_solid = [h for h in solid if h["when_class"] == "near"]
    flags = dict(
        n_hit_bgs=len(hits), any_hit=len(hits) > 0,
        n_solid=len(solid), any_solid=len(solid) > 0,
        any_near_solid=len(near_solid) > 0,
        any_sustained_deep=any(h["dur_class"] == "sustained"
                               and h["depth_class"] == "solid" for h in hits),
        max_depth_m=max((h["max_depth_m"] for h in hits), default=np.nan))
    return hits, flags


def main():
    t0 = time.time()
    tracks = vl.Tracks()
    print(f"[setup] tracks loaded ({len(tracks.by_id)})")
    sel = pd.read_csv(L.RESULTS / "selection.csv")
    chosen = sel[sel.chosen].set_index("cls").scenario_id.astype(str)
    manifest = pd.read_csv(L.RESULTS / "ours_manifest.csv")
    kde = pd.read_parquet(L.RESULTS / "kde_samples.parquet")

    sample_rows, hit_rows = [], []
    for cls in L.CLASSES:
        cfg = L.CFG[cls]
        sid = chosen[cls]
        meta = L.subset_meta(cfg["fit_subset"])
        meta["scenario_id"] = meta.scenario_id.astype(str)
        r = meta[meta.scenario_id == sid].iloc[0]
        ego_id, act_id = int(r.ego), int(r.actor)
        lo, hi = int(r.min_frame), int(r.max_frame)
        cls_a = tracks.cls.get(act_id, "car")
        bgs = tracks.backgrounds(ego_id, act_id, lo, hi)
        n_bg_veh = sum(1 for b in bgs if not b["ped"])
        ego_traj = SIMC.real_traj("HetroD", ego_id, lo, hi)
        if ego_traj is None:
            raise RuntimeError(f"{cls}/{sid}: no ego track")
        a_dims = (tracks.by_id[act_id]["length"], tracks.by_id[act_id]["width"])
        if not np.isfinite(a_dims[0]) or a_dims[0] <= 0:
            a_dims = tracks.default_dims.get(cls_a, tracks.default_dims["car"])
        kdims = tracks.default_dims.get(cls_a, tracks.default_dims["car"])

        # collect variants (arm, sample, gridded Traj, extra)
        variants = []
        c = tracks.clip(act_id, lo, hi)
        if c is None:
            raise RuntimeError(f"{cls}/{sid}: no real actor in window")
        f, x, y, _h = c
        real_var = vl.grid_traj(f, x, y, lo, hi, *a_dims)
        variants.append(("real", "real", real_var, {}))

        om = manifest[(manifest.cls == cls)
                      & (manifest.scenario_id.astype(str) == sid)]
        n_fail = 0
        for m in om.itertuples():
            res = vl.esmini_agent(Path(m.csv), lo)
            var = None if res is None else vl.grid_traj(*res, lo, hi, *a_dims)
            if var is None:
                n_fail += 1
                continue
            variants.append(("ours", m.sample, var,
                             dict(off_major=m.off_major,
                                  off_minor=m.off_minor)))
        kg = kde[(kde.cls == cls) & (kde.scenario_id.astype(str) == sid)]
        for sk, g in kg.groupby("sample"):
            var = vl.grid_traj(g.frame.values, g.x.values, g.y.values,
                               lo, hi, *kdims)
            if var is None:
                n_fail += 1
                continue
            variants.append(("svd_kde", sk, var,
                             dict(kde_duration_s=float(g.duration_s.iloc[0]))))

        # conflict reference = real interaction
        cxy, _i, j, _dmin = ISIM.conflict_point(real_var, ego_traj)
        ego_cf = float(ego_traj.frame[j])

        # horizons
        durs = {(a, s): float((v.frame[-1] - v.frame[0]) / FPS)
                for a, s, v, _e in variants if a != "real"}
        d_ours = [d for (a, _s), d in durs.items() if a == "ours"]
        d_kde = [d for (a, _s), d in durs.items() if a == "svd_kde"]
        ch_dur = min(min(d_ours), min(d_kde))
        chmed_dur = min(float(np.median(d_ours)), float(np.median(d_kde)))
        win_s = (hi - lo) / FPS
        print(f"\n=== {cls}/{sid}: {len(variants) - 1} samples "
              f"(+real), {n_bg_veh} bg vehicles, window {win_s:.1f}s ===")
        print(f"  spans: ours [{min(d_ours):.1f},{max(d_ours):.1f}] "
              f"med {np.median(d_ours):.1f}s | kde "
              f"[{min(d_kde):.1f},{max(d_kde):.1f}] med "
              f"{np.median(d_kde):.1f}s -> ch={ch_dur:.1f}s "
              f"chmed={chmed_dur:.1f}s; render/grid failures: {n_fail}")

        for arm, sample, var, extra in variants:
            for hz, dur in (("full", None), ("ch", ch_dur),
                            ("chmed", chmed_dur)):
                vh = var if dur is None else trunc_abs(var, lo, dur)
                if vh is None:
                    sample_rows.append(dict(cls=cls, scenario_id=sid, arm=arm,
                                            sample=sample, horizon=hz,
                                            scored=False, **extra))
                    continue
                hits, flags = score_variant(vh, bgs, cxy, ego_cf, cls_a)
                eg = vl.score_ego(vh, ego_traj, use_pet=cfg["pet_band"])
                driven = float((vh.frame[-1] - vh.frame[0]) / FPS)
                row = dict(cls=cls, scenario_id=sid, arm=arm, sample=sample,
                           horizon=hz, scored=True, driven_s=driven,
                           **flags, **eg,
                           bg_clean_critical=(not flags["any_solid"])
                           and eg["ego_critical"],
                           bg_clean_raw_critical=(not flags["any_hit"])
                           and eg["ego_critical"], **extra)
                sample_rows.append(row)
                for h in hits:
                    hit_rows.append(dict(cls=cls, scenario_id=sid, arm=arm,
                                         sample=sample, horizon=hz, **h))
        print(f"  scored ({time.time() - t0:.0f}s total)")

    samples = pd.DataFrame(sample_rows)
    hits = pd.DataFrame(hit_rows)

    # excess-over-real: ignore hits on bg tracks that the REAL actor replay
    # itself overlaps in the full window (recording/bbox-noise neighbors —
    # the real-actor zero-reference correction at bg-track level)
    real_tids = {cls: set(hits[(hits.cls == cls) & (hits.arm == "real")
                              & (hits.horizon == "full")].bg_tid)
                 for cls in L.CLASSES}
    print("\nreal-actor-overlapped bg tracks (excluded in *_excl_real):",
          {k: sorted(v) for k, v in real_tids.items()})
    key = ["cls", "arm", "horizon", "sample"]
    if len(hits):
        solid_excl = hits[(hits.n_overlap_frames > L.GRAZE_F)
                          & (hits.max_depth_m >= L.TANGENT_M)
                          & ~hits.apply(lambda r: r.bg_tid
                                        in real_tids[r.cls], axis=1)]
        flag = set(map(tuple, solid_excl[key].values))
    else:
        flag = set()
    samples["any_solid_excl_real"] = [
        tuple(r) in flag for r in samples[key].values]
    samples["bg_clean_excl_critical"] = (~samples.any_solid_excl_real
                                         & samples.ego_critical.fillna(False)
                                         .astype(bool))
    samples.to_csv(L.RESULTS / "samples.csv", index=False)
    hits.to_csv(L.RESULTS / "hits.csv", index=False)

    # ── summary: cls x arm x horizon ─────────────────────────────────────────
    summ = []
    for (cls, arm, hz), g in samples[samples.scored].groupby(
            ["cls", "arm", "horizon"]):
        n = len(g)
        hh = hits[(hits.cls == cls) & (hits.arm == arm)
                  & (hits.horizon == hz)]
        d = dict(cls=cls, arm=arm, horizon=hz, n=n,
                 mean_driven_s=g.driven_s.mean(),
                 n_hits=len(hh))
        d["n_bg_tracks_hit"] = hh.bg_tid.nunique() if len(hh) else 0
        for col in ("any_hit", "any_solid", "any_near_solid",
                    "any_sustained_deep", "any_solid_excl_real",
                    "ego_critical", "ego_collision", "bg_clean_critical",
                    "bg_clean_raw_critical", "bg_clean_excl_critical"):
            k = int(g[col].sum())
            lo_ci, hi_ci = L.wilson_ci(k, n)
            d[f"{col}_k"] = k
            d[f"{col}_rate"] = k / n
            d[f"{col}_ci_lo"], d[f"{col}_ci_hi"] = lo_ci, hi_ci
        dur = g.driven_s.sum()
        d["anyhit_per_s"] = g.any_hit.sum() / dur
        d["solid_per_s"] = g.any_solid.sum() / dur
        d["hits_per_s"] = len(hh) / dur
        if len(hh):
            for c in ("grazing", "short", "sustained"):
                d[f"frac_{c}"] = (hh.dur_class == c).mean()
            d["frac_near_tangent"] = (hh.depth_class == "near_tangent").mean()
            d["frac_near_interaction"] = (hh.when_class == "near").mean()
            d["frac_moto"] = (hh.bg_cls == "motorcycle").mean()
            d["frac_stationary"] = (~hh.bg_moving).mean()
            d["median_depth_m"] = hh.max_depth_m.median()
            d["median_overlap_s"] = hh.overlap_s.median()
            d["median_dist_to_conflict_m"] = hh.dist_to_conflict_m.median()
            for gc in ("same_dir", "crossing", "opposing"):
                d[f"frac_{gc}"] = (hh.geom_class == gc).mean()
        summ.append(d)
    sm = pd.DataFrame(summ)
    sm.to_csv(L.RESULTS / "summary.csv", index=False)
    print(f"\n[write] samples.csv ({len(samples)}), hits.csv ({len(hits)}), "
          f"summary.csv ({len(sm)})")

    show = ["cls", "arm", "horizon", "n", "mean_driven_s", "any_hit_rate",
            "any_solid_rate", "any_solid_ci_lo", "any_solid_ci_hi",
            "any_solid_excl_real_rate", "n_bg_tracks_hit",
            "any_near_solid_rate", "solid_per_s", "ego_critical_rate",
            "bg_clean_critical_rate", "bg_clean_excl_critical_rate",
            "frac_moto", "frac_near_interaction"]
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:.3f}"):
        for hz in ("full", "ch", "chmed"):
            print(f"\n=== horizon {hz} ===")
            print(sm[sm.horizon == hz][[c for c in show if c in sm.columns]]
                  .to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
