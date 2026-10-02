#!/usr/bin/env python3
"""SAKURA baseline arms: build + render (plain chord, road-following route, route + KDE).

Arms (label of every output = the arm name):
  sakura_plain_defaults_extra   plain SAKURA (start->end NURBS chord, constant speed) on the
                                exp_cross_coverage _xosc_base_sakura bases for the subsets the
                                existing 192 sakura_bc samples do not cover (cutinr, special_39_180,
                                plus the cutinl population copy of 39_180); same rule as
                                scripts/25_sakura_generate.py: ONLY Agent1_1_SA_EndSpeed is overridden
                                to the recorded target endpoint speed (hypot(vx,vy) at the last
                                recorded target frame inside the metadata window).  The 192 existing
                                sakura_bc samples (runs/sakura_defaults/d833696f8ea2e990) are reused
                                as-is for keeptl / keeptl_sw / cutinl and are NOT re-rendered.
  sakura_route_defaults         SAKURA-route, exp_cross_coverage/scripts/210_sakura_arms.py recipe:
                                where a planned route exists (results/route_plan_<cls>.json) the NURBS
                                FollowTrajectory is replaced by an AssignRouteAction over the planned
                                waypoints + a stop-at-goal event (sakura_route.apply_route) and the
                                constant speed = planned route length / recorded travel time is written
                                to BOTH Agent1_Speed and Agent1_1_SA_EndSpeed (210._speed_override);
                                where no route exists the base is rendered exactly as plain SAKURA
                                (210 fallback: overrides untouched, base constant start speed).
  sakura_route_kde_s<seed>      generative SAKURA-route: per scene parameter vector
                                (Agent1_Offset of the base = recorded lane offset [m],
                                 v_avg [km/h] = route length / recorded duration, or GT path length /
                                 duration where no route); one de Gelder KDE per class fitted exactly as
                                scripts/23b_ours3_disk_kde.py fits ours (AST-loaded cvlib.ParamKDE:
                                per-column standardisation, LOO scalar bandwidth, dependent sampling);
                                1000 draws per class per seed (fixed pools under plans/), kernel centre =
                                the drawn scene (its base, route, ego).  Executed offset = clip(requested,
                                centre base offset +- 0.5 m) (pipeline DistributionRange; larger offsets
                                make esmini snap to a neighbouring lane), speed = max(0, requested);
                                both clips are RECORDED per draw (requested vs applied + flags).
  sakura_route_kde_noclip_pilot the first 100 draws of seed 20260910 per class executed with the
                                UNCLIPPED offset (speed still max(0)) to report the teleport/snap rate.
  sakura_route_kde_39_180_cond  class-borrowed conditional arm for the singleton 39_180: cutinl fit,
                                conditional draws centred on 39_180's own (offset, v) with the cutinl h,
                                100 x 3 seeds; label 'bandwidth' (class data used), as in
                                results/e9_39_180/T8_REPORT.md.

Rendering = the same AST-loaded adapter functions of hetero_param/esmini_exec.py as
scripts/20_ours3_generate.py (to_esmini_replay, agent_replay=False), then sakura_route.apply_route
where routed, esmini headless with the identical flags of 20, parse_tidy of 20 (AST/module load).
Sample layout = the 20 layout (sample.json + trajectory.parquet + every xosc/csv retained) so that
scripts/31_ours3_nonpet.py resolve_context() / 32_bbox_pet.py / 45b / 46b accept the samples.

Usage:
  python -B scripts/85_sakura_arms.py --stage plan
  python -B scripts/85_sakura_arms.py --stage render --arms plain route kde noclip cond --workers 8
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
XC = ROOT / "exp_cross_coverage"
XBASE = XC / "esmini_runs/_xosc_base_sakura"
RUNNER_20 = PROJECT / "scripts/20_ours3_generate.py"
SAKURA_25 = PROJECT / "scripts/25_sakura_generate.py"
SAKURA_ROUTE_SRC = XC / "scripts/sakura_route.py"
SAKURA_210 = XC / "scripts/210_sakura_arms.py"
KDE_SOURCE = ROOT / "sr-tlkeep-experiment/core/kde_sampling.py"
CV_SOURCE = ROOT / "exp_coverage_velocity/scripts/cvlib.py"
CASES = PROJECT / "results/svd_d5_cases.csv"
SAKURA_BC_RUN = PROJECT / "runs/sakura_defaults/d833696f8ea2e990"
CLASSES = ["keeptl", "keeptl_sw", "cutinl", "cutinr"]
ESMINI_SEED = 20260914      # fixed esmini --seed (v3 rerun): makes SAKURA road routing reproducible
SEEDS = [20260910, 20260911, 20260912]
POOL_SIZE, PILOT_N, COND_N = 1000, 100, 100
OFFSET_CLIP_M = 0.5
FPS = 30.0
SPEED_END, SPEED_START, OFFSET = "Agent1_1_SA_EndSpeed", "Agent1_Speed", "Agent1_Offset"
KDE_COLS = ["offset_m", "v_avg_kmh"]
SPECIALS = {
    "39_180": dict(scenario_id="39_180", ego=39, actor=180, min_frame=2424, max_frame=3002,
                   subset="special_39_180", scenario_uid="HetroD/00/39_180/2424-3002",
                   source_tracks=str(XC / "data/special_39_180/real_tracks.parquet"),
                   route_json=str(XC / "results/route_plan_special_39_180.json"),
                   group_id="g_a892cf18c80191f4", donor_class="cutinl", label=8),
    # 859_881 belongs to no fitted class, so its borrowed bandwidth comes from the same
    # left-turn cohort that donates the SVD basis, and it has no planned route (0 % routed).
    "859_881": dict(scenario_id="859_881", ego=859, actor=881, min_frame=14435, max_frame=15711,
                    subset="uturn_859_881", scenario_uid="HetroD/00/859_881/14435-15711",
                    source_tracks=str(XC / "data/uturn_859_881/real_tracks.parquet"),
                    route_json=str(PROJECT / "results/route_plan_uturn_859_881.json"),
                    group_id="g_uturn_859_881", donor_class="keeptl", label=77),
}
SPECIAL = SPECIALS["39_180"]
SUF = ""
ARM_METHOD = {"plain": "sakura_plain", "route": "sakura_route", "kde": "sakura_route_kde",
              "noclip": "sakura_route_kde_noclip", "cond": "sakura_route_kde_cond_{sid}"}
ARM_BATCH = {"plain": "sakura_plain_defaults_extra", "route": "sakura_route_defaults",
             "kde": "sakura_route_kde_s{seed}", "noclip": "sakura_route_kde_noclip_pilot",
             "cond": "sakura_route_kde_{sid}_cond"}

_spec = importlib.util.spec_from_file_location("ours3_runner_20", RUNNER_20)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)                       # parse_tidy, sha256, atomic_json, ast_functions, literal_constant, params, signature, trajectory
_spec25 = importlib.util.spec_from_file_location("sakura_25", SAKURA_25)
S25 = importlib.util.module_from_spec(_spec25)
_spec25.loader.exec_module(S25)                   # verify (plain SAKURA checks), adapter_functions
_specsr = importlib.util.spec_from_file_location("sakura_route_xc", SAKURA_ROUTE_SRC)
SR = importlib.util.module_from_spec(_specsr)
_specsr.loader.exec_module(SR)                    # route_length, apply_route (pure; plan_route unused: plans are cached on disk)
MIN_FREE = H.MIN_FREE


# ── helpers ──────────────────────────────────────────────────────────────────
def json_safe(v):
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_kde():
    """AST-load the unchanged upstream ParamKDE + LOO bandwidth (exactly as scripts/23b)."""
    namespace = {"np": np}
    for path, names in ((KDE_SOURCE, {"loo_log_likelihood", "loo_bandwidth"}), (CV_SOURCE, {"ParamKDE"})):
        parsed = ast.parse(path.read_text())
        nodes = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes} == names
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace["ParamKDE"]


def ast_source_hash(path, name):
    src = Path(path).read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest()
    raise KeyError(name)


# ── scene registry ───────────────────────────────────────────────────────────
def load_raw_all():
    return pd.read_parquet(H.RAW, columns=["trackId", "frame", "xCenter", "yCenter", "heading",
                                           "xVelocity", "yVelocity", "length", "width"])


def target_window(raw_all, ego, target, lo, hi):
    raw = raw_all[raw_all.trackId.isin([ego, target]) & raw_all.frame.between(lo, hi)].sort_values(["trackId", "frame"])
    t = raw[raw.trackId.eq(target)].drop_duplicates("frame")
    return raw, t


def scene_table(raw_all):
    """All fullfit scenes of the four SAKURA classes + the special 39_180; base / route / kinematic facts."""
    cases = pd.read_csv(CASES, dtype={"recording": str, "scenario_id": str})
    cases = cases[cases["mode"].eq("fullfit") & cases.subset.isin(CLASSES)].drop_duplicates("scenario_uid")
    rows = []
    routes = {}
    for cls in CLASSES:
        p = XC / f"results/route_plan_{cls}.json"
        routes[cls] = json.loads(p.read_text())
    rj = Path(SPECIAL["route_json"])
    routes[SPECIAL["subset"]] = json.loads(rj.read_text()) if rj.exists() else {}
    specs = [dict(subset=r.subset, scenario_id=str(r.scenario_id), scenario_uid=r.scenario_uid, ego=int(r.ego),
                  actor=int(r.actor), min_frame=int(r.metadata_min_frame), max_frame=int(r.metadata_max_frame),
                  source_tracks=str(r.source_tracks), group_id=r.group_id, label=int(r.label))
             for r in cases.itertuples()]
    specs.append(dict(subset=SPECIAL["subset"], scenario_id=SPECIAL["scenario_id"], scenario_uid=SPECIAL["scenario_uid"],
                      ego=SPECIAL["ego"], actor=SPECIAL["actor"], min_frame=SPECIAL["min_frame"], max_frame=SPECIAL["max_frame"],
                      source_tracks=SPECIAL["source_tracks"], group_id=SPECIAL["group_id"],
                      label=SPECIAL.get("label", 8)))
    for s in specs:
        hits = sorted(XBASE.glob(f"*_{s['ego']}_{s['actor']}_f{s['min_frame'] + 1}.xosc"))
        row = dict(s, base=str(hits[0]) if hits else None, base_found=bool(hits), n_base_candidates=len(hits))
        wp = routes[s["subset"]].get(s["scenario_id"])
        row.update(route_found=bool(wp), n_waypoints=len(wp) if wp else 0,
                   route_length_m=float(SR.route_length(wp)) if wp else np.nan)
        _raw, t = target_window(raw_all, s["ego"], s["actor"], s["min_frame"], s["max_frame"])
        assert len(t) >= 2 and t.frame.is_unique
        dur = (float(t.frame.iloc[-1]) - float(t.frame.iloc[0])) / FPS
        gt_len = float(np.hypot(np.diff(t.xCenter.to_numpy(float)), np.diff(t.yCenter.to_numpy(float))).sum())
        end = t.iloc[-1]
        row.update(target_frames=int(len(t)), recorded_duration_s=dur, gt_path_length_m=gt_len,
                   target_gt_min_frame=int(t.frame.min()), target_gt_max_frame=int(t.frame.max()),
                   endpoint_speed_kmh=float(np.hypot(end.xVelocity, end.yVelocity) * 3.6), endpoint_speed_frame=int(end.frame),
                   v_avg_route_kmh=float(3.6 * row["route_length_m"] / dur) if wp and dur > 0 else np.nan,
                   v_avg_gt_kmh=float(3.6 * gt_len / dur) if dur > 0 else np.nan)
        row["v_avg_kmh"] = row["v_avg_route_kmh"] if wp else row["v_avg_gt_kmh"]
        row["v_avg_source"] = "route_length/recorded_duration" if wp else "gt_path_length/recorded_duration (no route)"
        if row["base"]:
            p = H.params(row["base"])
            row.update(base_sha256=H.sha256(row["base"]), base_offset_m=float(p[OFFSET]),
                       base_start_speed_kmh=float(p[SPEED_START]), base_end_speed_kmh=float(p[SPEED_END]),
                       base_sa_duration_s=float(p["Agent1_1_SA_DynamicDuration"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, routes


# ── job construction ─────────────────────────────────────────────────────────
def base_job(arm, s, job_id, overrides, apply_route, waypoints, geometry_variant_id, is_nominal, source_context, speed_rule):
    return dict(job_id=job_id, arm=arm, method=ARM_METHOD[arm], scenario_id=s["scenario_id"], scenario_uid=s["scenario_uid"],
                **{"class": s["subset"]}, dataset="HetroD", recording="00", ego=int(s["ego"]), target=int(s["actor"]),
                min_frame=int(s["min_frame"]), max_frame=int(s["max_frame"]), fps=FPS,
                source_xosc=s["base"], source_sha256=s["base_sha256"], source_tracks=s["source_tracks"],
                raw_tracks_path=str(H.RAW), map_path=str(H.MAP), geometry_variant_id=geometry_variant_id,
                overrides={k: float(v) for k, v in overrides.items()}, apply_route=bool(apply_route),
                waypoints=[list(w) for w in waypoints] if waypoints else None, route_found=bool(s["route_found"]),
                route_length_m=None if not s["route_found"] else float(s["route_length_m"]),
                recorded_duration_s=float(s["recorded_duration_s"]), gt_path_length_m=float(s["gt_path_length_m"]),
                is_nominal_default=bool(is_nominal), speed_rule=speed_rule, source_context=source_context,
                group_id=s["group_id"])


def jobs_plain(scenes):
    """Arm A: cutinr + special_39_180 (+ the cutinl population copy of 39_180)."""
    sel = scenes[scenes.base_found & (scenes.subset.eq("cutinr") | scenes.subset.eq(SPECIAL["subset"])
                                      | (scenes.subset.eq("cutinl") & scenes.scenario_id.eq("39_180")))]
    jobs = []
    for s in sel.to_dict("records"):
        sc = dict(subset=s["subset"], scenario_uid=s["scenario_uid"], source_tracks=s["source_tracks"], method="sakura_plain",
                  baseline_family="sakura_plain_xosc_base_sakura", route_found=False,
                  not_in_existing_192=True,
                  note=("cutinl population-membership copy of the special scene (not among the 192 sakura_bc samples)"
                        if s["subset"] == "cutinl" else "class not covered by the 192 existing sakura_bc samples"))
        jobs.append(base_job("plain", s, f"sakura_plain_{s['subset']}_{s['scenario_id']}",
                             {SPEED_END: s["endpoint_speed_kmh"]}, False, None,
                             "sakura_plain_nurbs_chord_const_speed_endspeed_override", True, sc,
                             "25 rule: only Agent1_1_SA_EndSpeed overridden to the recorded target endpoint speed"))
    return jobs


def route_overrides(s):
    """210._speed_override: constant speed = planned route length / recorded travel time on both speed params."""
    if not s["route_found"]:
        return {}, "210 fallback (no route): overrides untouched -> base constant start speed, plain NURBS chord"
    v = float(s["v_avg_route_kmh"])
    return {SPEED_START: v, SPEED_END: v}, "210._speed_override: Agent1_Speed = Agent1_1_SA_EndSpeed = 3.6 * route_length / recorded_duration"


def jobs_route(scenes, routes):
    jobs = []
    for s in scenes[scenes.base_found].to_dict("records"):
        wp = routes[s["subset"]].get(s["scenario_id"]) if s["route_found"] else None
        ov, rule = route_overrides(s)
        sc = dict(subset=s["subset"], scenario_uid=s["scenario_uid"], source_tracks=s["source_tracks"], method="sakura_route",
                  baseline_family="sakura_route_210", route_found=bool(wp), n_waypoints=len(wp) if wp else 0,
                  route_plan=str(XC / f"results/route_plan_{s['subset']}.json"),
                  fallback_label=None if wp else "no planned route -> plain SAKURA chord (label: sakura_route = plain chord)")
        jobs.append(base_job("route", s, f"sakura_route_{s['subset']}_{s['scenario_id']}", ov, bool(wp), wp,
                             "sakura_route_assignroute_planned_waypoints" if wp else "sakura_plain_nurbs_chord_no_route",
                             True, sc, rule))
    return jobs


def kde_jobs_from_pool(arm, table, scenes_by_uid, routes, seed, plan_paths, fit, clip):
    jobs = []
    for r in table.to_dict("records"):
        s = scenes_by_uid[r["center_scenario_uid"]]
        wp = routes[s["subset"]].get(s["scenario_id"]) if s["route_found"] else None
        v = float(r["applied_v_avg_kmh"])
        off = float(r["applied_offset_m"])
        ov = {OFFSET: off, SPEED_START: v, SPEED_END: v}
        sc = dict(subset=s["subset"], scenario_uid=s["scenario_uid"], source_tracks=s["source_tracks"], method=ARM_METHOD[arm],
                  sampling_method=r["sampling_method"], seed=int(seed), draw=int(r["draw"]), planned_pool_size=int(r["planned_pool_size"]),
                  kernel_center_index=int(r["center_index"]), kernel_center_scenario_id=s["scenario_id"],
                  kernel_center_scenario_uid=s["scenario_uid"], h=float(fit["h"]), kde_fit_class=fit["class"],
                  class_data_used=fit["class_data_used"],
                  sampler_parameters_requested=dict(offset_m=float(r["requested_offset_m"]), v_avg_kmh=float(r["requested_v_avg_kmh"])),
                  sampler_parameters_applied=dict(offset_m=off, v_avg_kmh=v),
                  offset_clipped=bool(r["offset_clipped"]), offset_clip_applied=bool(clip),
                  offset_clip_rule=(f"clip(requested, centre base offset -+ {OFFSET_CLIP_M} m)" if clip else "NONE (unclipped pilot)"),
                  offset_clip_bounds_m=[float(r["offset_clip_lo"]), float(r["offset_clip_hi"])] if clip else None,
                  speed_clipped=bool(r["speed_clipped"]), speed_clip_rule="max(0.0, requested_v_avg_kmh)",
                  sampler_invalid=bool(r["offset_clipped"]) or bool(r["speed_clipped"]),
                  rejected_or_resampled=False, sampling_plan=str(plan_paths["pool"]), fit_path=str(plan_paths["fit"]),
                  center_parameters=dict(offset_m=float(s["base_offset_m"]), v_avg_kmh=float(s["v_avg_kmh"]), v_avg_source=s["v_avg_source"]),
                  route_found=bool(wp), n_waypoints=len(wp) if wp else 0,
                  fit_scope="all scenes of the class with a plain SAKURA base (in-sample exploratory fit); one fit per class")
        jobs.append(base_job(arm, s, r["job_id"], ov, bool(wp), wp,
                             "sakura_route_assignroute_planned_waypoints" if wp else "sakura_plain_nurbs_chord_no_route",
                             False, sc, ("KDE draw: Agent1_Offset = applied offset; Agent1_Speed = Agent1_1_SA_EndSpeed = applied v_avg"
                                         + ("; route waypoints" if wp else "; plain chord (no route)"))))
    return jobs


# ── KDE planning (23b mirror) ────────────────────────────────────────────────
def plan_kde(scenes, routes, args):
    ParamKDE = load_kde()
    source_paths = [Path(__file__).resolve(), CASES, KDE_SOURCE, CV_SOURCE, RUNNER_20, SAKURA_25, SAKURA_ROUTE_SRC, SAKURA_210] + \
        [XC / f"results/route_plan_{c}.json" for c in CLASSES] + [Path(SPECIAL["route_json"])]
    sources = [{"path": str(p), "sha256": H.sha256(p)} for p in source_paths]
    fit_input = scenes[scenes.base_found & scenes.subset.isin(CLASSES)][["subset", "scenario_uid", "base_sha256", "base_offset_m", "v_avg_kmh"]]
    fit_digest = hashlib.sha256(fit_input.to_csv(index=False).encode()).hexdigest()
    source_key = hashlib.sha256(json.dumps(sources + [{"fit_input_sha256": fit_digest}], sort_keys=True).encode()).hexdigest()[:16]
    plan_root = PROJECT / "plans/sakura_route_kde" / source_key
    plan_root.mkdir(parents=True, exist_ok=True)
    if not (plan_root / "planner_snapshot.py").exists():
        (plan_root / "planner_snapshot.py").write_text(Path(__file__).read_text())
    by_class = {c: scenes[scenes.base_found & scenes.subset.eq(c)].reset_index(drop=True) for c in CLASSES}
    fit_path = plan_root / "fit.json"
    saved = json.loads(fit_path.read_text()) if fit_path.exists() else None
    if saved is not None:
        assert saved["sources"] == sources and saved["fit_input_sha256"] == fit_digest
    models, fits = {}, {}
    for cls in CLASSES:
        g = by_class[cls]
        raw = g[["base_offset_m", "v_avg_kmh"]].to_numpy(float)
        assert raw.shape == (len(g), 2) and np.isfinite(raw).all() and len(g) >= 2
        npz = plan_root / f"fit_{cls}.npz"
        if saved is not None:
            arrays = np.load(npz)
            model = ParamKDE.__new__(ParamKDE)
            model.raw, model.cols = arrays["raw"], KDE_COLS
            model.m, model.sd, model.Z = arrays["mean"], arrays["sd"], arrays["standardized"]
            model.h = float(arrays["h"])
            model.N, model.p = model.Z.shape
            assert np.array_equal(model.raw, raw)
            elapsed = saved["fits"][cls]["fit_s"]
        else:
            t0 = time.perf_counter()
            model = ParamKDE(raw, KDE_COLS)
            elapsed = time.perf_counter() - t0
            np.savez_compressed(npz, raw=model.raw, mean=model.m, sd=model.sd, standardized=model.Z, h=model.h)
        models[cls] = model
        fits[cls] = {"class": cls, "n_fit": int(len(g)), "dimensions": 2, "columns": KDE_COLS, "mean": model.m.tolist(),
                     "sd": model.sd.tolist(), "h": float(model.h), "fit_s": elapsed,
                     "center_scenario_ids": g.scenario_id.tolist(), "center_scenario_uids": g.scenario_uid.tolist(),
                     "n_centres_with_route": int(g.route_found.sum()), "class_data_used": "own class (dependent KDE)",
                     "standardization": "per-column mean and population std; near-zero std replaced with 1 (cvlib.ParamKDE)",
                     "bandwidth": "unchanged upstream LOO log-likelihood grid + golden-section refinement (kde_sampling.loo_bandwidth)",
                     "parameter_definition": dict(offset_m="Agent1_Offset of the plain SAKURA base (recorded lane offset)",
                                                  v_avg_kmh="3.6 * route_length / recorded_duration where a route exists, else 3.6 * GT path length / recorded_duration")}
        print(f"[sakura KDE fit] {cls}: N={len(g)}, p=2, h={model.h:.9g}, mean={model.m.round(4).tolist()}, sd={model.sd.round(4).tolist()}", flush=True)
    if saved is None:
        write_json(fit_path, dict(fits=fits, sources=sources, fit_input_sha256=fit_digest, variable_columns=KDE_COLS,
                                  fixed_context="kernel centre's base xosc (geometry, SA duration, delay), planned route, replay ego",
                                  fit_scope="in-sample exploratory fit; no train/test split"))
    # keyed by uid within the four classes only: the special_39_180 row shares its uid with the cutinl member 39_180
    scenes_by_uid = {s["scenario_uid"]: s for s in scenes.to_dict("records") if s["base_found"] and s["subset"] in CLASSES}
    pools = {}
    for seed in SEEDS:
        for cls in CLASSES:
            model, g = models[cls], by_class[cls]
            pool_path = plan_root / f"pool_{cls}_seed{seed}_n{POOL_SIZE}.parquet"
            timing_path = pool_path.with_suffix(".json")
            if pool_path.exists() and timing_path.exists():
                table = pd.read_parquet(pool_path)
                assert len(table) == POOL_SIZE
            else:
                rng = np.random.default_rng(seed)
                t0 = time.perf_counter()
                requested, centers = model.dependent(POOL_SIZE, rng)
                sampling_s = time.perf_counter() - t0
                rows = []
                for k, (p, idx) in enumerate(zip(requested, centers), 1):
                    c = g.iloc[int(idx)]
                    lo, hi = c.base_offset_m - OFFSET_CLIP_M, c.base_offset_m + OFFSET_CLIP_M
                    off_app = float(np.clip(p[0], lo, hi))
                    v_app = float(max(0.0, p[1]))
                    rows.append(dict(job_id=f"{cls}__seed{seed}__draw{k:06d}", subset=cls, seed=seed, draw=k, planned_pool_size=POOL_SIZE,
                                     sampling_method="sakura_route_kde_dependent", center_index=int(idx),
                                     center_scenario_id=c.scenario_id, center_scenario_uid=c.scenario_uid, center_route_found=bool(c.route_found),
                                     requested_offset_m=float(p[0]), requested_v_avg_kmh=float(p[1]),
                                     applied_offset_m=off_app, applied_v_avg_kmh=v_app,
                                     offset_clip_lo=float(lo), offset_clip_hi=float(hi), center_offset_m=float(c.base_offset_m), center_v_avg_kmh=float(c.v_avg_kmh),
                                     offset_clipped=bool(p[0] < lo or p[0] > hi), speed_clipped=bool(p[1] < 0),
                                     unclipped_offset_m=float(p[0]), h=float(model.h), source_xosc=c.base, source_xosc_sha256=c.base_sha256))
                table = pd.DataFrame(rows)
                table.to_parquet(pool_path, index=False)
                table.to_csv(pool_path.with_suffix(".csv"), index=False)
                write_json(timing_path, dict(subset=cls, seed=seed, n_generated_parameter_vectors=POOL_SIZE, n_rejected=0, n_resampled=0,
                                             n_offset_clipped=int(table.offset_clipped.sum()), n_speed_clipped=int(table.speed_clipped.sum()),
                                             sampling_s=sampling_s, pool_sha256=H.sha256(pool_path),
                                             rng="numpy.random.default_rng(seed), restarted per class",
                                             vectorized_draw_order="upstream ParamKDE.dependent(M): all uniform centers then all Gaussian noises",
                                             fit_path=str(fit_path), fit_h=float(model.h), offset_clip_m=OFFSET_CLIP_M))
            pools[(cls, seed)] = dict(path=pool_path, table=table, timing=json.loads(timing_path.read_text()))
    # conditional arm for 39_180 (cutinl fit, external centre = 39_180's own parameters)
    sp = scenes[scenes.subset.eq(SPECIAL["subset"])].iloc[0]
    model = models["cutinl"]
    x0 = np.array([sp.base_offset_m, sp.v_avg_kmh], float)
    z0 = (x0 - model.m) / model.sd
    cond = {}
    for seed in ([] if args.classes_only else SEEDS):
        pool_path = plan_root / f"pool_cond{SPECIAL['scenario_id']}_seed{seed}_n{COND_N}.parquet"
        timing_path = pool_path.with_suffix(".json")
        if pool_path.exists() and timing_path.exists():
            table = pd.read_parquet(pool_path)
        else:
            rng = np.random.default_rng(seed)
            t0 = time.perf_counter()
            Z = z0 + model.h * rng.standard_normal((COND_N, 2))     # ParamKDE.conditional with an external centre
            requested = model.to_raw(Z)
            sampling_s = time.perf_counter() - t0
            lo, hi = sp.base_offset_m - OFFSET_CLIP_M, sp.base_offset_m + OFFSET_CLIP_M
            rows = []
            for k, p in enumerate(requested, 1):
                rows.append(dict(job_id=f"{SPECIAL['subset']}__cond__seed{seed}__draw{k:06d}", subset=SPECIAL["subset"], seed=seed, draw=k,
                                 planned_pool_size=COND_N,
                                 sampling_method=f"sakura_route_kde_conditional_external_centre_{SPECIAL['donor_class']}_bandwidth",
                                 center_index=-1, center_scenario_id=sp.scenario_id, center_scenario_uid=sp.scenario_uid, center_route_found=False,
                                 requested_offset_m=float(p[0]), requested_v_avg_kmh=float(p[1]),
                                 applied_offset_m=float(np.clip(p[0], lo, hi)), applied_v_avg_kmh=float(max(0.0, p[1])),
                                 offset_clip_lo=float(lo), offset_clip_hi=float(hi), center_offset_m=float(x0[0]), center_v_avg_kmh=float(x0[1]),
                                 offset_clipped=bool(p[0] < lo or p[0] > hi), speed_clipped=bool(p[1] < 0), unclipped_offset_m=float(p[0]),
                                 h=float(model.h), source_xosc=sp.base, source_xosc_sha256=sp.base_sha256))
            table = pd.DataFrame(rows)
            table.to_parquet(pool_path, index=False)
            table.to_csv(pool_path.with_suffix(".csv"), index=False)
            write_json(timing_path, dict(subset=SPECIAL["subset"], seed=seed, n=COND_N,
                                         class_data_used=f"bandwidth ({SPECIAL['donor_class']} class fit: h, mean, sd)",
                                         centre=dict(offset_m=float(x0[0]), v_avg_kmh=float(x0[1]), v_avg_source=sp.v_avg_source, z=z0.tolist()),
                                         n_offset_clipped=int(table.offset_clipped.sum()), n_speed_clipped=int(table.speed_clipped.sum()),
                                         sampling_s=sampling_s, pool_sha256=H.sha256(pool_path), fit_path=str(fit_path), fit_h=float(model.h),
                                         rule=f"Z = z0 + h * N(0, I) with z0 = (x0 - mean) / sd of the {SPECIAL['donor_class']} fit; to_raw; same clip rules as the class arm"))
        cond[seed] = dict(path=pool_path, table=table, timing=json.loads(timing_path.read_text()))
    return dict(plan_root=plan_root, source_key=source_key, fit_path=fit_path, fits=fits, pools=pools, cond=cond,
                sources=sources, scenes_by_uid=scenes_by_uid)


# ── xosc verification ────────────────────────────────────────────────────────
def verify_route_xosc(original, executable, overrides, routed, n_wp):
    before, after = H.params(original), H.params(executable)
    assert set(before) == set(after), "ParameterDeclaration set changed"
    for k, v in before.items():
        if k in overrides:
            assert abs(float(after[k]) - float(overrides[k])) < 1e-12, (k, after[k], overrides[k])
        else:
            assert v == after[k], f"Fixed declaration changed: {k}"
    src, dst = ET.parse(original).getroot(), ET.parse(executable).getroot()
    assert H.signature(src.find("Entities")) == H.signature(dst.find("Entities")), "CatalogReference changed"
    src_nurbs = src.findall(".//Nurbs")
    assert len(src_nurbs) == 1 and len(src_nurbs[0].findall("ControlPoint")) == 4
    assert not [c for c in src_nurbs[0].findall("ControlPoint") if c.get("weight") is not None]
    dst_nurbs = dst.findall(".//Nurbs")
    routes = dst.findall(".//AssignRouteAction")
    stops = [e for e in dst.iter("Event") if "StopAtGoal" in e.get("name", "")]
    events = {e.get("name") for e in dst.iter("Event")}
    assert {"Agent1_StartEvent", "Agent1_SpeedEvent"} <= events, "original speed events missing"
    if routed:
        assert len(dst_nurbs) == 0, "Nurbs remains after apply_route"
        assert len(routes) == 1 and len(routes[0].findall(".//Waypoint")) == n_wp
        assert len(stops) == 1, "stop-at-goal event count"
        offsets = [w.find(".//LanePosition").get("offset") for w in routes[0].findall(".//Waypoint")]
        assert offsets[0] == "0" and all(o == f"${OFFSET}" for o in offsets[1:]), offsets
    else:
        assert len(dst_nurbs) == 1 and H.signature(src_nurbs[0]) == H.signature(dst_nurbs[0]), "chord geometry changed"
        assert not routes and not stops
    return dict(overridden_declarations=sorted(overrides), fixed_declarations_unchanged=True, catalog_reference_entities_unchanged=True,
                routed=bool(routed), nurbs_removed=bool(routed), n_route_waypoints=n_wp if routed else 0,
                route_waypoint_offset="first 0, then $Agent1_Offset" if routed else None, stop_at_goal_added=bool(routed),
                chord_geometry_unchanged=not routed, original_speed_events_present=True)


# ── worker ───────────────────────────────────────────────────────────────────
_W = {}   # per-process state: raw_all (fork), adapter namespace


def _adapter(job):
    """AST-load the four esmini_exec adapter functions once per process (exactly as 20/25/26)."""
    fps = job["fps"]
    if "ns" not in _W:
        ns = {"np": np, "pd": pd, "ET": ET, "Path": Path,
              "paths": SimpleNamespace(dataset_of=lambda _d: {"fps": fps, "xodr": job["map_path"]}),
              "SIMC": SimpleNamespace(real_traj=lambda _d, actor, lo, hi: H.trajectory(_W["raw"], actor, lo, hi, fps)),
              "CATALOG_ENTITIES": True, "_CATALOG_DIRS": H.literal_constant(H.EX_SOURCE, "_CATALOG_DIRS")}
        H.ast_functions(H.EX_SOURCE, ["_standard_catalogs", "_ego_replay_story", "_sim_time_stop", "to_esmini_replay"], ns)
        _W["ns"] = ns
    return _W["ns"]


def run_job(task):
    job, batch, run_id = task["job"], Path(task["batch"]), task["run_id"]
    wd = batch / job["job_id"]
    wd.mkdir(exist_ok=True)
    sample_path = wd / "sample.json"
    if sample_path.exists():
        try:
            old = json.loads(sample_path.read_text())
            if old.get("status") == "completed" and all(Path(a["path"]).exists() for a in old["artifacts"]):
                return dict(sample_id=job["job_id"], status="completed", reused=True, sample_dir=str(wd), error=None)
        except Exception:  # noqa: BLE001
            pass
    t0 = time.perf_counter()
    source_copy, executable, csv_path = wd / "source.xosc", wd / "run.xosc", wd / "run.csv"
    job_public = {k: v for k, v in job.items() if k != "waypoints"}
    report = dict(sample_id=job["job_id"], sample_dir=str(wd), job=job_public, run_id=run_id, method=job["method"], arm=job["arm"],
                  is_nominal_default=job["is_nominal_default"], status="preparing", all_xosc_retained=True, scoring_performed=False,
                  parameters_requested={}, parameters_applied={})
    sim_start = time.perf_counter()
    try:
        raw_all = _W["raw_all"]
        raw = raw_all[raw_all.trackId.isin([job["ego"], job["target"]]) & raw_all.frame.between(job["min_frame"], job["max_frame"])]
        raw = raw.sort_values(["trackId", "frame"])
        _W["raw"] = raw
        target = raw[raw.trackId.eq(job["target"])].drop_duplicates("frame")
        assert len(target) > 1
        original = H.params(job["source_xosc"])
        sc = job["source_context"]
        context = dict(method=job["method"], arm=job["arm"], baseline_family=sc.get("baseline_family", job["method"]),
                       scenario_id=job["scenario_id"], scenario_uid=job["scenario_uid"], dataset=job["dataset"], recording=job["recording"],
                       subset=job["class"], ego=job["ego"], target=job["target"], fps=job["fps"], source_tracks=job["source_tracks"],
                       raw_tracks_path=job["raw_tracks_path"], metadata_window_frames=[job["min_frame"], job["max_frame"]],
                       target_gt_support_frames=[int(target.frame.min()), int(target.frame.max())],
                       geometry_variant_id=job["geometry_variant_id"], source_xosc=job["source_xosc"], source_sha256=job["source_sha256"],
                       route_found=job["route_found"], apply_route=job["apply_route"], n_waypoints=len(job["waypoints"] or []),
                       route_length_m=job["route_length_m"], recorded_duration_s=job["recorded_duration_s"], gt_path_length_m=job["gt_path_length_m"],
                       speed_rule=job["speed_rule"], overrides=job["overrides"],
                       fixed_parameter_declarations={k: v for k, v in original.items() if k not in job["overrides"]},
                       source_end_speed_raw=original[SPEED_END], source_start_speed_raw=original[SPEED_START], source_offset_raw=original[OFFSET],
                       fixed_duration_raw=original["Agent1_1_SA_DynamicDuration"], is_nominal_default=job["is_nominal_default"],
                       declared_end_speed_is_measured_terminal_speed=False, catalogue_entities=True, original_adapter_stop_margin_s=1.0,
                       no_stop_at_goal_added=not job["apply_route"], no_endpoint_constraints=not job["apply_route"],
                       source_context=sc, group_id=job["group_id"])
        if job["arm"] == "plain":
            requested = dict(endpoint_end_speed_kmh=float(job["overrides"][SPEED_END]))
            context.update(default_end_speed_kmh=requested["endpoint_end_speed_kmh"], default_endpoint_end_speed_kmh=requested["endpoint_end_speed_kmh"],
                           speed_source="hypot(raw.xVelocity, raw.yVelocity) at last recorded target frame within metadata window")
        elif job["arm"] == "route":
            requested = {k: float(v) for k, v in job["overrides"].items()}
            requested["route_avg_speed_kmh"] = float(job["overrides"].get(SPEED_START, np.nan)) if job["overrides"] else None
        else:
            requested = dict(sc["sampler_parameters_requested"])
            requested.update({k: float(v) for k, v in job["overrides"].items()})
        applied = {k: float(v) for k, v in job["overrides"].items()}
        if job["arm"] in ("kde", "noclip", "cond"):
            applied.update(sc["sampler_parameters_applied"])
        report.update(context=context, parameters_requested=requested, parameters_applied=applied, load_context_s=time.perf_counter() - t0)
        H.atomic_json(wd / "context.json", json_safe(context))
        H.atomic_json(wd / "parameters.json", json_safe(dict(method=job["method"], requested=requested, applied=applied, overrides=job["overrides"],
                                                             fixed_declarations=context["fixed_parameter_declarations"])))
        shutil.copyfile(job["source_xosc"], source_copy)
        assert H.sha256(source_copy) == job["source_sha256"]
        t_prep = time.perf_counter()
        ns = _adapter(job)
        ns["to_esmini_replay"](source_copy, job["dataset"], job["ego"], job["target"], job["min_frame"], job["max_frame"], executable,
                               job["overrides"] or None, agent_replay=False)
        if job["apply_route"]:
            ok = SR.apply_route(executable, [tuple(w)[:3] for w in job["waypoints"]])
            assert ok, "apply_route matched nothing"
        if job["arm"] == "plain":
            report["checks"] = S25.verify(source_copy, executable, float(job["overrides"][SPEED_END]))
        else:
            report["checks"] = verify_route_xosc(source_copy, executable, job["overrides"], job["apply_route"], len(job["waypoints"] or []))
        report["prepare_s"] = time.perf_counter() - t_prep
        cmd = ["nice", "-n", "10", str(task["binary"]), "--osc", str(executable), "--headless", "--csv_logger", str(csv_path),
               "--fixed_timestep", str(1.0 / job["fps"]), "--logfile_path", str(wd / "esmini.log"), "--disable_stdout",
               # v3 2026-09-14: without --seed esmini draws a new random seed per run and its road routing picks
               # junction branches at random, so identical SAKURA run.xosc files gave different trajectories
               "--seed", str(ESMINI_SEED)]
        report.update(command=cmd, subprocess_cwd=str(wd), status="executing")
        H.atomic_json(sample_path, json_safe(report))
        sim_start = time.perf_counter()
        with (wd / "stdout.txt").open("wb") as so, (wd / "stderr.txt").open("wb") as se:
            res = subprocess.run(cmd, cwd=wd, stdout=so, stderr=se, timeout=60)
        report.update(returncode=res.returncode, simulate_s=time.perf_counter() - sim_start)
        if res.returncode != 0:
            raise RuntimeError(f"esmini returned {res.returncode}")
        t_parse = time.perf_counter()
        tidy = H.parse_tidy(csv_path, raw, job["job_id"], context)
        tidy.to_parquet(wd / "trajectory.parquet", index=False)
        tidy.to_csv(wd / "trajectory.csv", index=False, float_format="%.12g")
        tg = tidy[tidy.role.eq("target") & tidy.within_metadata_window]
        report.update(status="completed", parse_export_s=time.perf_counter() - t_parse, trajectory_rows=len(tidy),
                      trajectory_path=str(wd / "trajectory.parquet"),
                      observed_time_range_s=[float(tidy.time_s.min()), float(tidy.time_s.max())],
                      target_rows_in_window=int(len(tg)),
                      target_driven_s_in_window=float((tg.frame.max() - tg.frame.min()) / job["fps"]) if len(tg) > 1 else 0.0)
    except subprocess.TimeoutExpired:
        report.update(status="timeout", timeout_s=60, simulate_s=time.perf_counter() - sim_start)
    except Exception as exc:  # noqa: BLE001
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["total_s"] = time.perf_counter() - t0
    report["artifacts"] = [dict(path=str(p), sha256=H.sha256(p), size_bytes=p.stat().st_size)
                           for p in sorted(wd.iterdir()) if p.is_file() and p.name != "sample.json"]
    H.atomic_json(sample_path, json_safe(report))
    return dict(sample_id=job["job_id"], status=report["status"], reused=False, sample_dir=str(wd), error=report.get("error"),
                total_s=report["total_s"], simulate_s=report.get("simulate_s"))


def _init_worker(raw_all):
    _W["raw_all"] = raw_all


# ── batch driver ─────────────────────────────────────────────────────────────
def render_batch(batch_name, jobs, extra_protocol, raw_all, workers, limit=None):
    if limit:
        jobs = jobs[:limit]
    out = PROJECT / "runs" / batch_name
    out.mkdir(parents=True, exist_ok=True)
    src_paths = {Path(__file__).resolve(), RUNNER_20, SAKURA_25, SAKURA_ROUTE_SRC, SAKURA_210, H.EX_SOURCE, H.RAW, H.MAP, H.BINARY, CASES}
    src_paths.update(Path(j["source_xosc"]) for j in jobs)
    src_paths.update(Path(j["source_tracks"]) for j in jobs)
    src_paths.update(Path(j["source_context"]["route_plan"]) for j in jobs if j["source_context"].get("route_plan"))
    for cat in ("Vehicles/VehicleCatalog.xosc", "Pedestrians/PedestrianCatalog.xosc", "Controllers/ControllerCatalog.xosc", "Environments/EnvironmentCatalog.xosc"):
        src_paths.add(Path("/opt/Catalogs") / cat)
    sources = [dict(path=str(p), sha256=H.sha256(p), size_bytes=p.stat().st_size) for p in sorted(src_paths) if p.exists()]
    protocol = dict(batch_name=batch_name, method=jobs[0]["method"] if jobs else None, arm=jobs[0]["arm"] if jobs else None,
                    jobs=[{k: v for k, v in j.items() if k != "waypoints"} for j in jobs], n_jobs=len(jobs), sources=sources,
                    esmini_flags=["--headless", "--csv_logger", "--fixed_timestep 1/fps", "--logfile_path", "--disable_stdout",
                                  "--seed", str(ESMINI_SEED)],
                    esmini_seed=ESMINI_SEED,
                    subprocess_timeout_s=60, nice=10, worker_count=workers, min_free_disk_bytes=MIN_FREE,
                    adapter="hetero_param/esmini_exec.py to_esmini_replay (AST-loaded as 20/25/26), agent_replay=False; sakura_route.apply_route where routed",
                    verbatim_function_hashes=dict(apply_route=ast_source_hash(SAKURA_ROUTE_SRC, "apply_route"),
                                                  route_length=ast_source_hash(SAKURA_ROUTE_SRC, "route_length"),
                                                  speed_override_210=ast_source_hash(SAKURA_210, "_speed_override"),
                                                  verify_25=ast_source_hash(SAKURA_25, "verify"), parse_tidy_20=ast_source_hash(RUNNER_20, "parse_tidy")),
                    all_xosc_retained=True, scoring_performed=False, **extra_protocol)
    run_id = hashlib.sha256(json.dumps(json_safe(protocol), sort_keys=True).encode()).hexdigest()[:16]
    batch = out / run_id
    batch.mkdir(exist_ok=True)
    local_bin = PROJECT / "bin/esmini"
    assert local_bin.resolve() == H.BINARY.resolve()
    assert not (local_bin.parent / "config.yml").exists(), "Unexpected local esmini config"
    if not (batch / "runner_snapshot.py").exists():
        shutil.copyfile(__file__, batch / "runner_snapshot.py")
    H.atomic_json(batch / "protocol.json", json_safe(protocol))
    write_json(batch / "jobs.json", dict(batch_name=batch_name, run_id=run_id, jobs=jobs))
    tasks = [dict(job=j, batch=str(batch), run_id=run_id, binary=str(local_bin)) for j in jobs]
    t0 = time.perf_counter()
    results = []
    status = dict(run_id=run_id, batch_dir=str(batch), batch_name=batch_name, status="running", n_jobs=len(jobs))
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(raw_all,)) as pool:
        for n, r in enumerate(pool.imap_unordered(run_job, tasks, chunksize=1), 1):
            results.append(r)
            if n % 100 == 0 or n == len(tasks):
                counts = pd.Series([x["status"] for x in results]).value_counts().to_dict()
                print(f"[{batch_name}] {n}/{len(tasks)} {counts} ({time.perf_counter() - t0:.0f}s)", flush=True)
                H.atomic_json(batch / "batch_status.json", json_safe(dict(status, completed_so_far=n, status_counts=counts)))
            if shutil.disk_usage(batch).free < MIN_FREE:
                status.update(status="paused_low_disk")
                H.atomic_json(batch / "batch_status.json", json_safe(status))
                pool.terminate()
                raise RuntimeError("PAUSED: less than 10 GiB free; no files deleted")
    by_id = {r["sample_id"]: r for r in results}
    rows = []
    for j in jobs:
        r = by_id[j["job_id"]]
        sc = j["source_context"]
        rows.append(dict(sample_id=j["job_id"], status=r["status"], sample_dir=r["sample_dir"], error=r.get("error"),
                         scenario_id=j["scenario_id"], scenario_uid=j["scenario_uid"], subset=j["class"], method=j["method"], arm=j["arm"],
                         route_found=j["route_found"], apply_route=j["apply_route"], n_waypoints=len(j["waypoints"] or []),
                         is_nominal_default=j["is_nominal_default"], seed=sc.get("seed"), draw=sc.get("draw"),
                         kernel_center_scenario_uid=sc.get("kernel_center_scenario_uid"),
                         offset_clipped=sc.get("offset_clipped"), speed_clipped=sc.get("speed_clipped"),
                         **{f"override_{k}": v for k, v in j["overrides"].items()}, total_s=r.get("total_s"), simulate_s=r.get("simulate_s")))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(batch / "manifest.csv", index=False)
    counts = manifest.status.value_counts().to_dict()
    status.update(status="completed" if counts.get("completed", 0) == len(jobs) else "completed_with_failures",
                  status_counts=counts, elapsed_s=time.perf_counter() - t0, manifest=str(batch / "manifest.csv"))
    H.atomic_json(batch / "batch_status.json", json_safe(status))
    H.atomic_json(out / "latest_batch.json", json_safe(dict(batch_dir=str(batch), run_id=run_id, status=status["status"], manifest=str(batch / "manifest.csv"))))
    for s in sources:
        assert H.sha256(s["path"]) == s["sha256"], f"Input changed during the run: {s['path']}"
    print(json.dumps(json_safe(status)), flush=True)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("plan", "render"), default="plan")
    ap.add_argument("--arms", nargs="*", default=["plain", "route", "kde", "noclip", "cond"], choices=list(ARM_METHOD))
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N jobs of every batch")
    ap.add_argument("--special", choices=list(SPECIALS), default="39_180",
                    help="which singleton rare scenario the special/cond arms are built for")
    ap.add_argument("--only-special", action="store_true",
                    help="restrict the plain/route arms to the special subset (leaves the class renders alone)")
    ap.add_argument("--classes-only", action="store_true",
                    help="exclude frozen special scenes and skip conditional rarecase pools")
    ap.add_argument("--label-v2", action="store_true", help="updated 859_881 label, separate singleton batches")
    args = ap.parse_args()
    if args.label_v2 and args.special != "859_881":
        ap.error("--label-v2 requires --special 859_881")
    if args.classes_only and (args.only_special or "cond" in args.arms):
        ap.error("--classes-only cannot be combined with --only-special or the cond arm")
    global SPECIAL, SUF
    SPECIAL = SPECIALS[args.special]
    if args.label_v2:
        entry = json.loads((ROOT / "HetroD-labeler/data/00_labeled_scenarios.json").read_text())["859_881"]
        lo, hi = int(entry["min_frame"]), int(entry["max_frame"])
        SPECIAL = dict(SPECIAL, min_frame=lo, max_frame=hi,
                       scenario_uid=f"HetroD/00/859_881/{lo}-{hi}")
    SUF = ("" if args.special == "39_180" else f"_{SPECIAL['subset']}") + ("_labelv2" if args.label_v2 else "")
    t0 = time.perf_counter()
    raw_all = load_raw_all()
    scenes, routes = scene_table(raw_all)
    scenes.to_csv(PROJECT / f"results/sakura_arms_scenes{SUF}.csv", index=False)
    summ = scenes.groupby("subset").agg(n_scenes=("scenario_uid", "size"), n_base=("base_found", "sum"), n_route=("route_found", "sum")).reset_index()
    summ["routed_fraction_of_bases"] = [float(scenes[scenes.subset.eq(c) & scenes.base_found].route_found.mean()) for c in summ.subset]
    print(summ.to_string(index=False), flush=True)
    plan = plan_kde(scenes, routes, args)
    write_json(PROJECT / f"results/sakura_arms_plan{SUF}.json",
               dict(plan_root=str(plan["plan_root"]), source_key=plan["source_key"], fits=plan["fits"],
                    pools={f"{c}_seed{s}": dict(path=str(v["path"]), **v["timing"]) for (c, s), v in plan["pools"].items()},
                    cond={f"seed{s}": dict(path=str(v["path"]), **v["timing"]) for s, v in plan["cond"].items()},
                    scene_summary=summ.to_dict("records"), sources=plan["sources"]))
    if args.stage == "plan":
        print(f"[plan] done in {time.perf_counter() - t0:.0f}s", flush=True)
        return 0
    scenes_by_uid = plan["scenes_by_uid"]
    sp = scenes[scenes.subset.eq(SPECIAL["subset"])].iloc[0].to_dict()
    arm_scenes = scenes[scenes.subset.eq(SPECIAL["subset"])] if args.only_special else scenes
    if args.classes_only:
        arm_scenes = arm_scenes[arm_scenes.subset.isin(CLASSES)]
    statuses = {}
    if "plain" in args.arms:
        statuses["plain"] = render_batch(ARM_BATCH["plain"] + SUF, jobs_plain(arm_scenes),
                                         dict(rule="25_sakura_generate.py: only Agent1_1_SA_EndSpeed overridden to recorded endpoint speed; bases = exp_cross_coverage/_xosc_base_sakura",
                                              existing_192_reused_from=str(SAKURA_BC_RUN)), raw_all, args.workers, args.limit)
    if "route" in args.arms:
        statuses["route"] = render_batch(ARM_BATCH["route"] + SUF, jobs_route(arm_scenes, routes),
                                         dict(rule="210_sakura_arms.py sakura_route recipe (avg speed on both speed params, apply_route where route_found, plain chord otherwise)"),
                                         raw_all, args.workers, args.limit)
    if "kde" in args.arms:
        for seed in args.seeds:
            jobs = []
            for cls in CLASSES:
                pool = plan["pools"][(cls, seed)]
                jobs += kde_jobs_from_pool("kde", pool["table"], scenes_by_uid, routes, seed,
                                           dict(pool=pool["path"], fit=plan["fit_path"]), plan["fits"][cls], clip=True)
            statuses[f"kde_s{seed}"] = render_batch(ARM_BATCH["kde"].format(seed=seed), jobs,
                                                    dict(seed=seed, pool_size=POOL_SIZE, offset_clip_m=OFFSET_CLIP_M, fit=plan["fits"],
                                                         plan_root=str(plan["plan_root"])), raw_all, args.workers, args.limit)
    if "noclip" in args.arms:
        jobs = []
        for cls in CLASSES:
            pool = plan["pools"][(cls, SEEDS[0])]
            t = pool["table"].head(PILOT_N).copy()
            t["applied_offset_m"] = t["unclipped_offset_m"]
            t["job_id"] = t.job_id.str.replace("__draw", "__noclip__draw", regex=False)
            jobs += kde_jobs_from_pool("noclip", t, scenes_by_uid, routes, SEEDS[0], dict(pool=pool["path"], fit=plan["fit_path"]), plan["fits"][cls], clip=False)
        statuses["noclip"] = render_batch(ARM_BATCH["noclip"], jobs,
                                          dict(seed=SEEDS[0], pilot_n_per_class=PILOT_N, offset_clip="NONE (unclipped pilot; same first 100 draws as sakura_route_kde_s20260910)"),
                                          raw_all, args.workers, args.limit)
    if "cond" in args.arms:
        jobs = []
        sby = dict(scenes_by_uid)
        sby[SPECIAL["scenario_uid"]] = sp
        for seed in args.seeds:
            c = plan["cond"][seed]
            donor = SPECIAL["donor_class"]
            fit = dict(plan["fits"][donor], class_data_used=f"bandwidth ({donor} class fit borrowed: h, mean, sd; "
                                                            f"NOT learned from {SPECIAL['scenario_id']} alone)")
            jobs += kde_jobs_from_pool("cond", c["table"], sby, routes, seed, dict(pool=c["path"], fit=plan["fit_path"]), fit, clip=True)
        statuses["cond"] = render_batch(ARM_BATCH["cond"].format(sid=SPECIAL["scenario_id"]) + ("_labelv2" if args.label_v2 else ""), jobs,
                                        dict(seeds=args.seeds, n_per_seed=COND_N, class_data_used="bandwidth",
                                             kde_fit_class=SPECIAL["donor_class"],
                                             label="class-borrowed conditional arm; answers 'what an outside bandwidth builds', not the singleton question (DECISIONS.md #5/#6)"),
                                        raw_all, args.workers, args.limit)
    write_json(PROJECT / f"results/sakura_arms_render_status{SUF}.json", dict(statuses=statuses, elapsed_s=time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
