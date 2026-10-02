#!/usr/bin/env python3
"""Table 5 validity / background scoring of the SAKURA arms on the SAME substrate as scripts/45_validity_bg.py.

scripts/45_validity_bg.py is NOT edited: this script imports it as a module and reuses its substrate
(vl41.Tracks / grid_traj / score_ego, bg47.hit_detail SAT depth, e7 drivable union + trim_lead_still,
lib.teleport_info/is_teleport, exit_angle, 07.latacc_vmax, trunc_abs, wilson_ci) through the functions
45 defines: load_sample_geometry, compute_heavy, horizon_flags, rate_block, paired_stats.  The per-scene
loop of 45.score_scenario is replicated here with ONE change: the ch / chmed horizon cutoffs are the
per-scene values already published in results/table5_validity_samples.csv (so the new arms are scored
on exactly Table 5's horizons instead of horizons re-derived from the new arms); the horizon that 45's
rule would give from the new arms alone is kept as a diagnostic column (chmed_dur_own_s).  The real
replay reference is re-scored through the same loop and checked against Table 5's real rows.

Arms scored (all executed, esmini):
  sakura_plain_extra           runs/sakura_plain_defaults_extra   (cutinr, special_39_180, cutinl copy of 39_180)
  sakura_route                 runs/sakura_route_defaults
  sakura_route_kde             runs/sakura_route_kde_s<seed>       (3 seeds x 1000 per class)
  sakura_route_kde_noclip      runs/sakura_route_kde_noclip_pilot  (unclipped offset pilot, 100 per class)
  sakura_route_kde_cond_39_180 runs/sakura_route_kde_39_180_cond   (class-borrowed conditional, 100 x 3 seeds)
plus, for the summary / paired tests, the Table 5 rows of real, ours3_disk and sakura_bc copied read-only
(sakura_plain = sakura_bc (192) + sakura_plain_extra).
sampler-invalid for the KDE arms = offset_clipped OR speed_clipped (recorded per draw by 85).
PET (ego-critical band) from the bbox-PET files of the new arms (results/bbox_pet_sak_*_paired.csv).

Outputs: results/table5_validity_summary_sakura.csv, table5_validity_samples_sakura.csv,
table5_validity_stats_sakura.csv, table5_validity_hits_sakura.csv, table5_sakura_manifest.json
(same columns as the Table 5 files).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
XC = PROJECT.parent / "exp_cross_coverage"
SPECIAL_UID, SPECIAL_CLS = "HetroD/00/39_180/2424-3002", "special_39_180"
SPECIAL_TRACKS = XC / "data/special_39_180/real_tracks.parquet"
# 2026-09-14 v3 rerun: cutinl scenes relabelled away from cut-in-left (same list as scripts/10 EXCLUDED_SCENARIOS).
EXCLUDED_CUTINL = {"39_180", "39_189", "1669_1657"}
CLASSES = ["keeptl", "keeptl_sw", "cutinl", "cutinr", SPECIAL_CLS]
NEW_ARMS = {
    "sakura_plain_extra": dict(batch="sakura_plain_defaults_extra", kde=False, label="SAKURA plain chord, extra scenes (cutinr / special_39_180 / cutinl 39_180) (executed)"),
    "sakura_route": dict(batch="sakura_route_defaults", kde=False, label="SAKURA-route defaults (210 recipe; plain chord where no route) (executed)"),
    "sakura_route_kde": dict(batch="sakura_route_kde_s*", kde=True, label="SAKURA-route + KDE (offset, v_avg), executed, offset clipped to base +-0.5 m"),
    "sakura_route_kde_noclip": dict(batch="sakura_route_kde_noclip_pilot", kde=True, label="SAKURA-route + KDE UNCLIPPED offset pilot (first 100 draws, seed 20260910)"),
    "sakura_route_kde_cond_39_180": dict(batch="sakura_route_kde_39_180_cond", kde=True, label="39_180 class-borrowed conditional SAKURA-route KDE (cutinl bandwidth)"),
}
PET_FILES = {
    "sakura_plain_extra": ["bbox_pet_sak_sakura_plain_extra_paired.csv"],
    "sakura_route": ["bbox_pet_sak_sakura_route_paired.csv"],
    "sakura_route_kde": [f"bbox_pet_sak_sakura_route_kde_s{s}_paired.csv" for s in (20260910, 20260911, 20260912)],
    "sakura_route_kde_noclip": ["bbox_pet_sak_sakura_route_kde_noclip_paired.csv"],
    "sakura_route_kde_cond_39_180": ["bbox_pet_sak_sakura_route_kde_cond_39_180_paired.csv"],
}
COPY_ARMS = ["real", "ours3_disk", "sakura_bc"]

_spec = importlib.util.spec_from_file_location("validity45", PROJECT / "scripts/45_validity_bg.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["validity45"] = M
_spec.loader.exec_module(M)                    # substrate loaded; main() not run
M.LOG = RES / "45b_progress.log"
M.CACHE = RES / "table5_cache_sakura"          # separate cache: Table 5's cache dir is not touched
vl, bg47, ISIM, SIMC = M.vl, M.bg47, M.ISIM, M.SIMC
FPS = M.FPS


def log(msg):
    M.log(msg)


def latest_batches(pattern):
    out = []
    for d in sorted((PROJECT / "runs").glob(pattern)):
        lb = d / "latest_batch.json"
        if lb.exists():
            out.append(Path(json.loads(lb.read_text())["batch_dir"]))
    return out


def collect(arm, spec):
    roots = latest_batches(spec["batch"])
    out, n_status = [], {}
    for root in roots:
        for sj in sorted(root.glob("*/sample.json")):
            s = json.loads(sj.read_text())
            st = s.get("status")
            n_status[st] = n_status.get(st, 0) + 1
            if st != "completed":
                continue
            job, ctx = s["job"], s["context"]
            sc = ctx.get("source_context", {})
            d = dict(arm=arm, arm_label=spec["label"], kind="executed", scenario_uid=ctx["scenario_uid"], cls=ctx["subset"],
                     sample_id=s["sample_id"], traj_path=str(sj.parent / "trajectory.parquet"), run_id=s.get("run_id"),
                     sample_json=str(sj), sampler_invalid=False, sampler_invalid_reason="", mode=job["arm"],
                     kde_seed=sc.get("seed", ""), kde_draw=sc.get("draw", ""), route_found=bool(ctx.get("apply_route", False)),
                     offset_clipped=bool(sc.get("offset_clipped", False)), speed_clipped=bool(sc.get("speed_clipped", False)))
            if spec["kde"]:
                reasons = [r for r, f in (("offset_clipped", d["offset_clipped"]), ("speed_clipped", d["speed_clipped"])) if f]
                d["sampler_invalid"] = bool(reasons)
                d["sampler_invalid_reason"] = "+".join(reasons)
            out.append(d)
    return out, dict(roots=[str(r) for r in roots], status_counts=n_status)


def score_scenario_fixed(task):
    """45.score_scenario with Table 5's per-scene horizon cutoffs (own-rule chmed kept as diagnostic)."""
    cls, scen, samples = task["cls"], task["scen"], task["samples"]
    tracks, area, offroad_vl = M._G["tracks"], M._G["area"], M._G["offroad_vl"]
    uid, lo, hi = scen["uid"], scen["lo"], scen["hi"]
    srows, hrows, notes = [], [], []
    try:
        bgs = tracks.backgrounds(scen["ego"], scen["actor"], lo, hi)
        ego_traj = SIMC.real_traj("HetroD", scen["ego"], lo, hi)
        if ego_traj is None:
            return dict(uid=uid, srows=[], hrows=[], notes=[f"{uid}: no ego track"])
        n_bg_veh = sum(1 for b in bgs if not b["ped"])
        n_bg_ped = sum(1 for b in bgs if b["ped"])
        real = next(s for s in samples if s["arm"] == "real")
        real_geo = M.load_sample_geometry(real, scen)
        real_var = vl.grid_traj(real_geo["frame"], real_geo["x"], real_geo["y"], lo, hi, scen["L"], scen["W"]) if real_geo else None
        if real_var is None:
            return dict(uid=uid, srows=[], hrows=[], notes=[f"{uid}: real actor <3 frames in window"])
        cxy, _i, j, _dmin = ISIM.conflict_point(real_var, ego_traj)
        ego_cf = float(ego_traj.frame[j])
        real_md = ISIM.min_distance(real_var, ego_traj)
        real_xy = np.column_stack([real_var.x, real_var.y])
        cdir = M.CACHE / cls
        cdir.mkdir(parents=True, exist_ok=True)
        built, n_cache_hit = [], 0
        for s in samples:
            geo = M.load_sample_geometry(s, scen)
            if geo is None:
                built.append((s, None, None))
                continue
            cp = cdir / f"{geo['sha']}_{M.SCORER_VERSION}.json"
            heavy = None
            if cp.exists():
                try:
                    heavy = json.loads(cp.read_text())
                    if heavy.get("scenario_uid") != uid:
                        heavy = None
                    else:
                        n_cache_hit += 1
                except Exception:  # noqa: BLE001
                    heavy = None
            if heavy is None:
                heavy = M.compute_heavy(s, geo, scen, bgs, real_xy, offroad_vl, area)
                heavy["scenario_uid"] = uid
                tmp = cp.with_suffix(".tmp")
                tmp.write_text(json.dumps(heavy, default=lambda o: None if o is None else
                                          (float(o) if isinstance(o, (np.floating,)) else bool(o) if isinstance(o, np.bool_) else str(o))))
                os.replace(tmp, cp)
            built.append((s, geo, heavy))
        real_hits = next(h for s, g, h in built if s["arm"] == "real")
        real_tids = {h["tid"] for h in (real_hits.get("hits") or []) if not h["ped"]}
        durs = {}
        for s, g, h in built:
            if s["arm"] != "real" and h is not None and h.get("gridded"):
                durs.setdefault(s["arm"], []).append(h["driven_s"])
        all_d = [d for v in durs.values() for d in v]
        own_ch = min(all_d) if all_d else None
        own_chmed = min(float(np.median(v)) for v in durs.values()) if durs else None
        # Table 5 cutoffs (fixed); fall back to the own rule only when Table 5 has no row for the scene
        ch_dur = task["ch_dur"] if task["ch_dur"] is not None else own_ch
        chmed_dur = task["chmed_dur"] if task["chmed_dur"] is not None else own_chmed
        horizon_source = "table5_validity_samples.csv" if task["chmed_dur"] is not None else "own_rule_45 (no Table 5 row)"
        horizons = [("full", None), ("ch", ch_dur), ("chmed", chmed_dur)]
        for s, geo, heavy in built:
            base = dict(cls=cls, arm=s["arm"], arm_label=s["arm_label"], kind=s["kind"], scenario_uid=uid, scenario_id=scen["scenario_id"],
                        group_id=scen["group_id"], sample_id=s["sample_id"], mode=s.get("mode", ""), kde_seed=s.get("kde_seed", ""),
                        kde_draw=s.get("kde_draw", ""), window_s=(hi - lo) / FPS, n_bg_vehicles=n_bg_veh, n_bg_peds=n_bg_ped,
                        real_conflict_ego_frame=ego_cf, real_min_dist_frame=real_md["min_dist_frame"], ch_dur_s=ch_dur, chmed_dur_s=chmed_dur,
                        sampler_invalid=bool(s.get("sampler_invalid", False)), sampler_invalid_reason=s.get("sampler_invalid_reason", ""),
                        traj_sha256=geo["sha"] if geo else "", horizon_source=horizon_source, ch_dur_own_s=own_ch, chmed_dur_own_s=own_chmed,
                        route_found=s.get("route_found", ""), offset_clipped=s.get("offset_clipped", ""), speed_clipped=s.get("speed_clipped", ""))
            if geo is None or heavy is None or not heavy.get("gridded"):
                reason = "decode_failed_or_unreadable" if geo is None else "<3 frames in window"
                for hz, _d in horizons:
                    srows.append(dict(base, horizon=hz, scored=False, unscored_reason=reason))
                continue
            var = vl.grid_traj(geo["frame"], geo["x"], geo["y"], lo, hi, scen["L"], scen["W"])
            heavy_cols = {("full_" + k if k in ("driven_s", "grid_f0", "grid_f1") else k): heavy[k]
                          for k in heavy if k not in ("hits", "version", "scenario_uid", "gridded")}
            for hz, dur in horizons:
                vh = var if dur is None else M.trunc_abs(var, lo, dur)
                if vh is None:
                    srows.append(dict(base, horizon=hz, scored=False, unscored_reason="truncated <3 frames", **heavy_cols))
                    continue
                cutoff = None if dur is None else lo + dur * FPS
                flags, hits = M.horizon_flags(heavy["hits"], cutoff, ego_cf, real_tids)
                eg = vl.score_ego(vh, ego_traj, use_pet=False)
                driven = float((vh.frame[-1] - vh.frame[0]) / FPS)
                row = dict(base, horizon=hz, scored=True, unscored_reason="", driven_s=driven, exposure_s=driven,
                           hits_per_s=flags["n_hit_bgs"] / driven if driven > 0 else np.nan,
                           solid_per_s=flags["n_solid"] / driven if driven > 0 else np.nan, **flags,
                           ego_collision=bool(eg["ego_collision"]), ego_min_ttc=float(eg["ego_min_ttc"]),
                           ego_crit_ttc=bool(np.isfinite(eg["ego_min_ttc"]) and eg["ego_min_ttc"] < M.TTC_K), **heavy_cols)
                srows.append(row)
                for h in hits:
                    hrows.append(dict(cls=cls, arm=s["arm"], scenario_uid=uid, sample_id=s["sample_id"], horizon=hz, **h))
        notes.append(f"{uid}/{cls}: {len(built)} samples, {n_cache_hit} cache hits, ch={ch_dur}, chmed={chmed_dur} ({horizon_source})")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"{uid}: ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    return dict(uid=uid, srows=srows, hrows=hrows, notes=notes)


def run_chunk(chunk):
    return [score_scenario_fixed(t) for t in chunk]


def pet_lookup_new():
    gen, real = {}, {}
    for arm, files in PET_FILES.items():
        for f in files:
            p = RES / f
            if not p.exists():
                log(f"WARN missing PET file {p}")
                continue
            d = pd.read_csv(p, low_memory=False)
            for r in d.itertuples():
                gen[(arm, r.sample_id)] = (float(r.generated_pet) if pd.notna(r.generated_pet) else np.nan, str(r.generated_pet_type))
                if pd.notna(r.real_pet):
                    real.setdefault(r.scenario_uid, (float(r.real_pet), str(r.real_pet_type)))
    return gen, real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    log(f"=== 45b_validity_extra_arms start (workers={args.workers}) ===")
    reg = M.scenario_registry()
    reg = reg[reg.subset.isin(CLASSES)]
    special_rows = reg[reg.scenario_uid.eq(SPECIAL_UID)]
    if len(special_rows):
        special = special_rows.iloc[0].to_dict()
    else:
        # v3 rerun: 39_180 is no longer a cutinl cohort member, but the special_39_180 key keeps its pre-rerun
        # registry row (frozen from _snapshot_20260914.tar results/svd_d5_cases.csv, fullfit mode).
        special = pd.read_csv(RES / "special_39_180_registry_row.csv", dtype={"scenario_id": str}).iloc[0].to_dict()
    special.update(subset=SPECIAL_CLS, source_tracks=str(SPECIAL_TRACKS))
    reg = pd.concat([reg, pd.DataFrame([special])], ignore_index=True)
    reg_by_key = {(r.subset, r.scenario_uid): r._asdict() for r in reg.itertuples(index=False)}
    log(f"registry: {len(reg)} (class, scene) keys")

    t5 = pd.read_csv(RES / "table5_validity_samples.csv", low_memory=False)
    t5_h = t5[t5.horizon.eq("chmed")].drop_duplicates("scenario_uid").set_index("scenario_uid")[["ch_dur_s", "chmed_dur_s"]]

    samples, arm_meta = [], {}
    for arm, spec in NEW_ARMS.items():
        ss, info = collect(arm, spec)
        kept = []
        for s in ss:
            if (s["cls"], s["scenario_uid"]) not in reg_by_key:
                if s["cls"] == "cutinl" and s["scenario_uid"].split("/")[2] in EXCLUDED_CUTINL:
                    continue      # v3 rerun: removed from the cutinl cohort at the root, so dropped here too
                raise RuntimeError(f"{s['sample_json']}: ({s['cls']}, {s['scenario_uid']}) not in registry")
            kept.append(s)
        n_dropped = len(ss) - len(kept)
        ss = kept
        if n_dropped:
            log(f"arm {arm}: dropped {n_dropped} samples of excluded cutinl scenes {sorted(EXCLUDED_CUTINL)}")
        samples += ss
        arm_meta[arm] = dict(info, n_completed=len(ss), n_dropped_excluded_cutinl=n_dropped, label=spec["label"])
        log(f"arm {arm}: {len(ss)} completed samples {info['status_counts']} from {info['roots']}")
    # real references (one per (class, scene) key)
    src_cache = {}
    for r in reg.itertuples(index=False):
        g = M.load_real_actor(r.source_tracks, r.scenario_id, src_cache)
        assert g is not None and g.track_id.eq(int(r.actor)).all(), r.scenario_uid
        g = g[(g.frame >= r.metadata_min_frame) & (g.frame <= r.metadata_max_frame)]
        samples.append(dict(arm="real", arm_label=M.ARM_LABEL["real"], kind="real", scenario_uid=r.scenario_uid, cls=r.subset,
                            sample_id=f"real__{r.scenario_id}", actor_df=g[["frame", "x", "y", "speed", "length", "width"]].copy(),
                            sampler_invalid=False, sampler_invalid_reason=""))
    del src_cache
    log("loading HetroD tracks (vl41.Tracks) ...")
    tracks = vl.Tracks()
    SIMC._raw_tracks("HetroD")
    log("building drivable union (E7 + VL windowing) ...")
    area = M.e7.drivable_union()
    offroad_vl = vl.load_offroad()
    M._G.update(tracks=tracks, area=area, offroad_vl=offroad_vl)
    scen_by_key = {}
    for r in reg.itertuples(index=False):
        t = tracks.by_id.get(int(r.actor))
        cls_a = tracks.cls.get(int(r.actor), "car")
        L_, W_ = (t["length"], t["width"]) if t is not None else (np.nan, np.nan)
        if not np.isfinite(L_) or L_ <= 0:
            L_, W_ = tracks.default_dims.get(cls_a, tracks.default_dims["car"])
        scen_by_key[(r.subset, r.scenario_uid)] = dict(uid=r.scenario_uid, scenario_id=r.scenario_id, cls=r.subset, ego=int(r.ego), actor=int(r.actor),
                                                       lo=int(r.metadata_min_frame), hi=int(r.metadata_max_frame), group_id=r.group_id,
                                                       L=float(L_), W=float(W_), actor_class=cls_a)
    by_key = {}
    for s in samples:
        by_key.setdefault((s["cls"], s["scenario_uid"]), []).append(s)
    tasks = []
    for cls in CLASSES:
        keys = [k for k in by_key if k[0] == cls and len(by_key[k]) > 1]
        if args.limit:
            keys = keys[:args.limit]
        for i in range(0, len(keys), args.chunk):
            tasks.append([dict(cls=cls, scen=scen_by_key[k], samples=by_key[k],
                               ch_dur=float(t5_h.ch_dur_s[k[1]]) if k[1] in t5_h.index and pd.notna(t5_h.ch_dur_s[k[1]]) else None,
                               chmed_dur=float(t5_h.chmed_dur_s[k[1]]) if k[1] in t5_h.index and pd.notna(t5_h.chmed_dur_s[k[1]]) else None)
                          for k in keys[i:i + args.chunk]])
    n_rows = sum(len(x["samples"]) for t in tasks for x in t)
    log(f"{len(tasks)} tasks, {n_rows} sample rows to score")
    srows, hrows, notes = [], [], []
    ctx = mp.get_context("fork")
    done = 0
    with ctx.Pool(args.workers) as pool:
        for res_list in pool.imap_unordered(run_chunk, tasks):
            for res in res_list:
                srows += res["srows"]
                hrows += res["hrows"]
                notes += res["notes"]
            done += 1
            if done % 10 == 0 or done == len(tasks):
                log(f"task {done}/{len(tasks)} ({len(srows)} rows, {time.time() - t0:.0f}s)")
    errs = [n for n in notes if "ERROR" in n]
    for n in errs:
        log(n)
    new = pd.DataFrame(srows)
    hits_df = pd.DataFrame(hrows)

    # PET from the new arms' bbox-PET files (never recomputed); real PET from Table 5's real rows / the new files
    gen_pet, real_pet = pet_lookup_new()
    t5_real_pet = t5[t5.arm.eq("real") & t5.horizon.eq("full")].drop_duplicates("scenario_uid").set_index("scenario_uid")

    def pet_of(r):
        if r.arm == "real":
            v = real_pet.get(r.scenario_uid)
            if v is None and r.scenario_uid in t5_real_pet.index and t5_real_pet.pet_available[r.scenario_uid]:
                v = (float(t5_real_pet.pet_s[r.scenario_uid]), str(t5_real_pet.pet_type[r.scenario_uid]))
        else:
            v = gen_pet.get((r.arm, r.sample_id))
        return v if v is not None else (np.nan, "n/a")

    pets = [pet_of(r) for r in new.itertuples()]
    new["pet_s"] = [p[0] for p in pets]
    new["pet_type"] = [p[1] for p in pets]
    new["pet_available"] = new.pet_type != "n/a"
    new["pet_zero"] = np.where(new.pet_available, (new.pet_type == "time_overlap") | (new.pet_s == 0), np.nan)
    new["pet_lt1"] = np.where(new.pet_available, new.pet_s.abs() < M.PET_G, np.nan)
    sc = new.scored.fillna(False).astype(bool)
    new["ego_critical"] = np.where(sc, new.ego_crit_ttc.fillna(False).astype(bool) | (new.pet_available & (new.pet_s.abs() < M.PET_G)), np.nan)
    for c in ("teleport", "shift_m25_any_solid", "shift_p25_any_solid"):
        new[c] = new[c].map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v))
    new["valid_all"] = np.where(sc, ~(new.any_solid.fillna(False).astype(bool) | new.ped_any_solid.fillna(False).astype(bool)
                                      | new.offroad_vl.fillna(False).astype(bool) | new.teleport.fillna(False).astype(bool)
                                      | new.wrongway.fillna(False).astype(bool) | new.phys_gate1.fillna(False).astype(bool)
                                      | new.sampler_invalid.fillna(False).astype(bool)), np.nan)
    new["valid_and_critical"] = np.where(sc, new.valid_all.fillna(False).astype(bool) & new.ego_critical.fillna(False).astype(bool), np.nan)

    # consistency of the re-scored real reference with Table 5's real rows (same substrate, same horizons)
    chk_cols = ["any_solid", "any_hit", "offroad_vl", "offroad_e7", "wrongway", "phys_gate1", "driven_s"]
    a = new[new.arm.eq("real") & ~new.cls.eq(SPECIAL_CLS)].set_index(["scenario_uid", "horizon"])
    b = t5[t5.arm.eq("real")].set_index(["scenario_uid", "horizon"])
    common = a.index.intersection(b.index)
    mism = {c: int((~np.isclose(a.loc[common, c].astype(float).fillna(-1).to_numpy(), b.loc[common, c].astype(float).fillna(-1).to_numpy(), atol=1e-6)).sum()) for c in chk_cols}
    log(f"real-reference consistency vs Table 5 on {len(common)} (scene, horizon) rows: mismatches {mism}")

    # copied Table 5 rows (read-only) for the reference arms; sakura_plain = sakura_bc + sakura_plain_extra
    copied = t5[t5.arm.isin(COPY_ARMS) & t5.cls.isin(CLASSES) & ~t5.arm.eq("real")].copy()
    copied["horizon_source"] = "table5_validity_samples.csv (copied row)"
    for c in ("ch_dur_own_s", "chmed_dur_own_s", "route_found", "offset_clipped", "speed_clipped"):
        copied[c] = np.nan
    bc_rows = copied[copied.arm.eq("sakura_bc")]
    extra_rows = new[new.arm.eq("sakura_plain_extra")]
    dup = extra_rows.set_index(["cls", "scenario_uid"]).index.isin(bc_rows.set_index(["cls", "scenario_uid"]).index)
    log(f"sakura_plain: extra rows already among the 192 sakura_bc scenes dropped from the combined arm: {sorted(extra_rows[dup].sample_id.unique())}")
    plain = pd.concat([bc_rows, extra_rows[~dup]], ignore_index=True)
    plain["arm"] = "sakura_plain"
    plain["arm_label"] = "SAKURA plain chord = sakura_bc (192, copied from Table 5) + sakura_plain_extra (executed)"
    samples_df = pd.concat([new, copied, plain], ignore_index=True)
    # The v3 route render excludes the user-frozen 39_180 singleton. Preserve its
    # pre-v3 route validity rows from the verified snapshot for Table III/D9.
    frozen_prefix = RES / "special_39_180_table5_sakura_frozen_"
    frozen_samples = pd.read_csv(str(frozen_prefix) + "samples.csv", low_memory=False)
    frozen_hits = pd.read_csv(str(frozen_prefix) + "hits.csv", low_memory=False)
    assert len(frozen_samples) == 3 and set(frozen_samples.horizon) == {"full", "ch", "chmed"}
    assert len(frozen_hits) == 10
    assert not ((samples_df.cls == "special_39_180") & (samples_df.arm == "sakura_route")).any()
    samples_df = pd.concat([samples_df, frozen_samples], ignore_index=True)
    hits_df = pd.concat([hits_df, frozen_hits], ignore_index=True)
    samples_df.to_csv(RES / "table5_validity_samples_sakura.csv", index=False)
    hits_df.to_csv(RES / "table5_validity_hits_sakura.csv", index=False)
    log(f"[write] table5_validity_samples_sakura.csv ({len(samples_df)}), hits ({len(hits_df)})")

    scored = samples_df[samples_df.scored.fillna(False).astype(bool)].copy()
    arms = [a for a in ["sakura_bc", "sakura_plain_extra", "sakura_plain", "sakura_route", "sakura_route_kde",
                        "sakura_route_kde_noclip", "sakura_route_kde_cond_39_180", "ours3_disk"] if (scored.arm == a).any()]
    for a in arms:
        M.ARM_KIND[a] = "executed"
    rng = np.random.default_rng(M.SEED)
    summ = []
    for hz in ("full", "ch", "chmed"):
        S = scored[scored.horizon == hz]
        real_by_uid = S[S.arm == "real"].drop_duplicates("scenario_uid").set_index("scenario_uid")
        for scope in ["pooled"] + CLASSES:
            SS = S if scope == "pooled" else S[S.cls == scope]
            for a in ["real"] + arms:
                g = SS[SS.arm == a]
                if not len(g):
                    continue
                d = dict(scope=scope, cls=scope, arm=a, arm_label=g.arm_label.iloc[0], kind="real" if a == "real" else "executed", horizon=hz,
                         n_attempted=int(((samples_df.horizon == hz) & (samples_df.arm == a) & ((samples_df.cls == scope) if scope != "pooled" else True)).sum()))
                d.update(M.rate_block(g, real_by_uid, rng))
                if a == "real":
                    for c in M.DELTA_COLS:
                        d[f"{c}_delta_real"] = 0.0
                summ.append(d)
    summary = pd.DataFrame(summ)
    frozen_summary = pd.read_csv(str(frozen_prefix) + "summary.csv", low_memory=False)
    assert len(frozen_summary) == 3 and set(frozen_summary.horizon) == {"full", "ch", "chmed"}
    frozen_key = (summary.scope == "special_39_180") & (summary.arm == "sakura_route")
    summary = pd.concat([summary[~frozen_key], frozen_summary], ignore_index=True)
    summary.to_csv(RES / "table5_validity_summary_sakura.csv", index=False)
    M.CLASSES = CLASSES
    stats = M.paired_stats(scored, [a for a in arms if a != "ours3_disk"])
    stats.to_csv(RES / "table5_validity_stats_sakura.csv", index=False)
    log(f"[write] summary ({len(summary)}), stats ({len(stats)})")
    manifest = dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), script="scripts/45b_validity_extra_arms.py",
                    reuses="scripts/45_validity_bg.py (imported module; substrate, load_sample_geometry, compute_heavy, horizon_flags, rate_block, paired_stats, thresholds, SEED)",
                    scorer_version=M.SCORER_VERSION, seed=M.SEED, workers=args.workers, classes=CLASSES,
                    horizons="ch / chmed cutoffs fixed to Table 5's per-scene values (results/table5_validity_samples.csv); own-rule values kept as ch_dur_own_s / chmed_dur_own_s",
                    cache_dir=str(M.CACHE), arms=arm_meta, copied_reference_arms=COPY_ARMS,
                    sampler_invalid_rule="KDE arms: offset_clipped OR speed_clipped (recorded per draw by scripts/85)",
                    frozen_special_route="3 validity sample/summary rows and 10 hit rows for special_39_180/sakura_route, copied from verified _snapshot_20260914.tar; no v3 rerender",
                    pet_files={a: [str(RES / f) for f in fs] for a, fs in PET_FILES.items()},
                    real_reference_consistency=dict(n_rows=int(len(common)), mismatches=mism),
                    n_scored_per_arm={a: int(((samples_df.arm == a) & (samples_df.horizon == "full") & samples_df.scored.fillna(False).astype(bool)).sum()) for a in ["real"] + arms},
                    n_attempted_per_arm={a: int(((samples_df.arm == a) & (samples_df.horizon == "full")).sum()) for a in ["real"] + arms},
                    inputs_sha256={p: M.sha256_file(RES / p) for p in ["svd_d5_cases.csv", "table5_validity_samples.csv",
                        "special_39_180_table5_sakura_frozen_samples.csv", "special_39_180_table5_sakura_frozen_summary.csv",
                        "special_39_180_table5_sakura_frozen_hits.csv", "special_39_180_table5_sakura_frozen_manifest.json"]},
                    errors=errs, elapsed_s=time.time() - t0,
                    outputs=["results/table5_validity_samples_sakura.csv", "results/table5_validity_hits_sakura.csv",
                             "results/table5_validity_summary_sakura.csv", "results/table5_validity_stats_sakura.csv", "results/table5_sakura_manifest.json"])
    (RES / "table5_sakura_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    show = ["scope", "arm", "n", "mean_driven_s", "any_solid_rate", "any_solid_delta_real", "any_near_solid_rate", "offroad_vl_rate",
            "teleport_rate", "wrongway_rate", "phys_gate1_rate", "sampler_invalid_rate", "valid_all_rate", "valid_and_critical_rate"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.3f}"):
        print("\n=== horizon chmed ===")
        print(summary[summary.horizon == "chmed"][show].to_string(index=False))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
