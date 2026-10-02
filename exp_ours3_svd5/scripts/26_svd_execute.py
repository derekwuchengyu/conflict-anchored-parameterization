#!/usr/bin/env python3
"""E3 arm: execute SVD d5 decoded polylines in esmini.

Label of every output of this script: "SVD executed (timed Polyline, E3)".

The algebraic SVD baseline (results/matched_cohort/svd_d5_nonpet_paired.csv,
results/bbox_pet_matched_paired.csv) decodes each 101-vector into 50 xy samples
plus one duration and scores them with derived kinematics; it never runs a
simulator. This script keeps that decode convention exactly
(frame = metadata_min_frame + linspace(0, duration, 50) * fps, see
scripts/30_interaction_metrics.py) and writes the decoded samples as a timed
OpenSCENARIO Polyline (Vertex time + WorldPosition x, y, h) for the challenge
agent, then executes it headless with the SAME replay ego, the SAME adapter
functions (hetero_param/esmini_exec.py: _standard_catalogs, _ego_replay_story,
_sim_time_stop, to_esmini_replay, AST-loaded as scripts/20_ours3_generate.py
does) and the SAME tidy trajectory schema (parse_tidy of 20).

Base xosc surgery, beyond what to_esmini_replay already does for ours3:
  1. Agent1 FollowTrajectoryAction: the Nurbs shape (7 ControlPoints incl. the
     three weighted method control points) is REPLACED by a Polyline of the 50
     decoded vertices; TimeReference None -> Timing absolute (scale 1, offset 0);
     followingMode stays "position".
  2. Every SpeedAction of the challenge agent is REMOVED: the Action
     "Agent1_StartSpeedAction" (step to $Agent1_Speed) inside "Agent1_StartEvent"
     and the whole Event "Agent1_SpeedEvent" (linear ramp to
     $Agent1_1_SA_EndSpeed).  With a timed polyline the vertex times define the
     speed; SpeedActions would contest the longitudinal domain.  Referenced
     events (StoryboardElementStateCondition chain StartEvent -> DummyEvent ->
     TrajectoryEvent) survive; only the SpeedAction actions are dropped.
  3. Nothing else changes: ParameterDeclarations (values kept, EndSpeed not
     overridden), CatalogReference entities, Init teleports, the spawn teleport,
     the ego replay story and the SimulationTime StopTrigger (window + 1 s
     margin) are identical to the ours3 executable.

Stages: smoke (5 fullfit reconstructions per class, ADE gate), recon (all
cohort-eligible reconstructions x {fullfit, logo}), kde (first 100 draws per
class per seed; numeric_ok=False rows are executed with applied duration and
keep their flags).  Layout: runs/svd_d5_executed_<stage>/<run_id>/<sample_id>/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RAW = ROOT / "HetroD-labeler/data/00_tracks.parquet"
EX_SOURCE = ROOT / "hetero-param/hetero_param/esmini_exec.py"
MAP = ROOT / "retrieval-scenarios/data/map/tyms.xodr"
BINARY = ROOT / "esmini/bin/esmini"
RUNNER_20 = PROJECT / "scripts/20_ours3_generate.py"
POPULATION = PROJECT / "results/ours3_population.csv"
ANALYTIC_PAIRED = PROJECT / "results/matched_cohort/svd_d5_nonpet_paired.csv"
MIN_FREE = 10 * 1024 ** 3
CLASSES = ["cutinl", "cutinr", "keeptl", "keeptl_sw", "tlkeep"]
FPS = 30.0
NT = 50
EXECUTION_LABEL = "SVD executed (timed Polyline, E3)"
METHOD = "svd_d5_executed"
STARTUP_SKIP_STEPS = 5          # same startup-lag skip as exp_kmp_axes/03_build_template.py
SMOKE_ADE_GATE_M, SMOKE_MAX_GATE_M = 0.05, 0.20

_spec = importlib.util.spec_from_file_location("ours3_runner_20", RUNNER_20)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)          # parse_tidy, sha256, atomic_json, ast_functions, literal_constant, params, trajectory


def json_safe(value):
    """Replace non-finite floats by None so 20's allow_nan=False writer accepts the report."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_ex_functions(raw_window, fps):
    """AST-load the four adapter functions exactly as 20_ours3_generate.load_pure_functions does."""
    namespace = {"np": np, "pd": pd, "ET": ET, "Path": Path}
    namespace["paths"] = SimpleNamespace(dataset_of=lambda _dataset: {"fps": fps, "xodr": str(MAP)})
    namespace["SIMC"] = SimpleNamespace(real_traj=lambda _dataset, actor, lo, hi: G.trajectory(raw_window, actor, lo, hi, fps))
    namespace["CATALOG_ENTITIES"] = True
    namespace["_CATALOG_DIRS"] = G.literal_constant(EX_SOURCE, "_CATALOG_DIRS")
    G.ast_functions(EX_SOURCE, ["_standard_catalogs", "_ego_replay_story", "_sim_time_stop", "to_esmini_replay"], namespace)
    return namespace


def decode(vector, duration, min_frame, fps):
    """Legacy decode convention of 30_interaction_metrics.py; heading by finite differences."""
    vector = np.asarray(vector, float)
    assert vector.shape == (101,) and np.isfinite(vector[:100]).all()
    xy = vector[:100].reshape(NT, 2)
    assert np.isfinite(duration) and duration > 0
    t = np.linspace(0.0, float(duration), NT)
    frame = float(min_frame) + t * fps
    d = np.diff(xy, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    h = np.zeros(NT)
    last = None
    for i in range(NT - 1):
        if seg[i] > 1e-9:
            last = float(np.arctan2(d[i, 1], d[i, 0]))
        elif last is None:
            j = next((k for k in range(i + 1, NT - 1) if seg[k] > 1e-9), None)
            last = float(np.arctan2(d[j, 1], d[j, 0])) if j is not None else 0.0
        h[i] = last
    h[NT - 1] = h[NT - 2]
    return dict(x=xy[:, 0], y=xy[:, 1], t=t, frame=frame, h=h, duration_s=float(duration),
                zero_length_segments=int((seg <= 1e-9).sum()), path_length_m=float(seg.sum()))


def polyline_surgery(xosc_path, poly):
    """Replace the Agent1 Nurbs by the timed decoded Polyline; drop Agent1 SpeedActions."""
    tree = ET.parse(xosc_path)
    root = tree.getroot()
    agent_fta = None
    for fta in root.iter("FollowTrajectoryAction"):
        traj = fta.find(".//Trajectory")
        if traj is not None and traj.get("name") != "EgoReplayTrajectory":
            assert agent_fta is None, "more than one non-ego FollowTrajectoryAction"
            agent_fta = fta
    assert agent_fta is not None, "no agent FollowTrajectoryAction"
    traj = agent_fta.find(".//Trajectory")
    shape = traj.find("Shape")
    nurbs = shape.find("Nurbs")
    assert nurbs is not None and shape.find("Polyline") is None
    removed = dict(nurbs_control_points=len(nurbs.findall("ControlPoint")),
                   nurbs_weighted_control_points=len([c for c in nurbs.findall("ControlPoint") if c.get("weight") is not None]),
                   nurbs_knots=len(nurbs.findall("Knot")), nurbs_order=nurbs.get("order"),
                   trajectory_name=traj.get("name"))
    shape.remove(nurbs)
    polyline = ET.SubElement(shape, "Polyline")
    for x, y, t, h in zip(poly["x"], poly["y"], poly["t"], poly["h"]):
        v = ET.SubElement(polyline, "Vertex", time=f"{t:.6f}")
        pos = ET.SubElement(v, "Position")
        ET.SubElement(pos, "WorldPosition", x=f"{x:.4f}", y=f"{y:.4f}", h=f"{h:.5f}")
    tref = agent_fta.find("TimeReference")
    removed["time_reference_before"] = [c.tag for c in tref] if tref is not None else None
    if tref is None:
        tref = ET.SubElement(agent_fta, "TimeReference")
    for child in list(tref):
        tref.remove(child)
    ET.SubElement(tref, "Timing", domainAbsoluteRelative="absolute", scale="1", offset="0")
    fm = agent_fta.find("TrajectoryFollowingMode")
    removed["following_mode_before"] = fm.get("followingMode") if fm is not None else None
    if fm is None:
        fm = ET.SubElement(agent_fta, "TrajectoryFollowingMode")
    fm.set("followingMode", "position")
    referenced = {sb.get("storyboardElementRef") for sb in root.iter("StoryboardElementStateCondition")}
    removed_actions, removed_events = [], []
    for man in root.iter("Maneuver"):
        for ev in list(man.findall("Event")):
            sa_acts = [a for a in ev.findall("Action") if a.find(".//SpeedAction") is not None
                       and a.find(".//FollowTrajectoryAction") is None]
            if not sa_acts:
                continue
            others = [a for a in ev.findall("Action") if a not in sa_acts]
            if not others and ev.get("name") not in referenced:
                man.remove(ev)
                removed_events.append(ev.get("name"))
                removed_actions.extend(f"{ev.get('name')}/{a.get('name')}" for a in sa_acts)
            else:
                assert others, f"referenced pure-SpeedAction event {ev.get('name')}"
                for a in sa_acts:
                    ev.remove(a)
                    removed_actions.append(f"{ev.get('name')}/{a.get('name')}")
    removed.update(removed_speed_action_actions=removed_actions, removed_events=removed_events,
                   time_reference_after="Timing absolute scale=1 offset=0", following_mode_after="position",
                   polyline_vertices=int(NT))
    ET.indent(tree, space="    ")
    tree.write(xosc_path, encoding="utf-8", xml_declaration=True)
    return removed


def verify_executable(base_xosc, adapter_xosc, executable, poly):
    base = ET.parse(base_xosc).getroot()
    mid = ET.parse(adapter_xosc).getroot()
    dst = ET.parse(executable).getroot()
    assert G.params(base_xosc) == G.params(executable), "ParameterDeclarations changed"
    assert G.signature(base.find("Entities")) == G.signature(dst.find("Entities")), "CatalogReference changed"
    assert dst.find(".//Nurbs") is None, "Nurbs remains"
    ego_story = [s for s in dst.find("Storyboard").findall("Story") if s.get("name") == "story_EgoReplay"]
    mid_story = [s for s in mid.find("Storyboard").findall("Story") if s.get("name") == "story_EgoReplay"]
    assert len(ego_story) == 1 and G.signature(ego_story[0]) == G.signature(mid_story[0]), "ego replay story changed"
    assert G.signature(dst.find("Storyboard/StopTrigger")) == G.signature(mid.find("Storyboard/StopTrigger")), "stop trigger changed"
    assert G.signature(dst.find("Storyboard/Init")) == G.signature(mid.find("Storyboard/Init")), "Init changed"
    agent_speed_actions = [sa for sa in dst.iter("SpeedAction")]
    story_speed = [sa for story in dst.find("Storyboard").findall("Story") for sa in story.iter("SpeedAction")]
    assert not story_speed, "story SpeedAction remains"
    poly_verts = [v for tr in dst.iter("Trajectory") if tr.get("name") != "EgoReplayTrajectory" for v in tr.iter("Vertex")]
    assert len(poly_verts) == NT
    wx = np.array([float(v.find("Position/WorldPosition").get("x")) for v in poly_verts])
    wy = np.array([float(v.find("Position/WorldPosition").get("y")) for v in poly_verts])
    wt = np.array([float(v.get("time")) for v in poly_verts])
    assert np.max(np.abs(wx - poly["x"])) <= 5.00001e-5 and np.max(np.abs(wy - poly["y"])) <= 5.00001e-5
    assert np.max(np.abs(wt - poly["t"])) <= 5.00001e-7 and np.all(np.diff(wt) > 0)
    stop_value = float(dst.find("Storyboard/StopTrigger//SimulationTimeCondition").get("value"))
    return dict(parameter_declarations_unchanged=True, catalog_reference_entities_unchanged=True,
                init_unchanged=True, ego_replay_story_unchanged=True, stop_trigger_unchanged=True,
                nurbs_removed=True, story_speed_actions_remaining=0, init_speed_actions_remaining=len(agent_speed_actions),
                polyline_vertices=len(poly_verts), vertex_rounding_max_m=float(max(np.max(np.abs(wx - poly["x"])), np.max(np.abs(wy - poly["y"])))),
                sim_stop_s=stop_value)


def fidelity(tidy, poly, fps, agent_delay_s):
    """Executed Agent1 trace sampled at the vertex times versus the analytic polyline.

    The base xosc spawns Agent1 after $Agent1_Delay (= the target's GT start
    minus the metadata window start; 0.033 s when clock aligned).  Its event
    chain then needs ~3-4 steps before the timed polyline takes over, so the
    agent stands at its spawn teleport until then.  "post_startup" vertices are
    those at t >= Agent1_Delay + 5/fps (the exp_kmp_axes 5-step skip).
    """
    a = tidy[tidy.role.eq("target")].sort_values("time_s")
    ts, xs, ys = a.time_s.to_numpy(float), a.x.to_numpy(float), a.y.to_numpy(float)
    t_end = float(ts[-1])
    inside = poly["t"] <= t_end + 1e-9
    ex = np.interp(poly["t"][inside], ts, xs)
    ey = np.interp(poly["t"][inside], ts, ys)
    dev = np.hypot(ex - poly["x"][inside], ey - poly["y"][inside])
    post = poly["t"][inside] >= agent_delay_s + STARTUP_SKIP_STEPS / fps - 1e-9
    # first logged step at which the agent is within 1 cm of the polyline (startup lag)
    within = ts <= poly["t"][-1] + 1e-9
    px = np.interp(ts[within], poly["t"], poly["x"])
    py = np.interp(ts[within], poly["t"], poly["y"])
    step_dev = np.hypot(xs[within] - px, ys[within] - py)
    hit = np.nonzero(step_dev < 0.01)[0]
    after = ts > poly["t"][-1] + 1e-9
    post_steps = np.hypot(np.diff(xs), np.diff(ys))[after[1:]] if after.sum() >= 1 else np.array([])
    return dict(polyline_ends_before_agent_start=bool(poly["t"][-1] < agent_delay_s),
                post_polyline_rows=int(after.sum()),
                post_polyline_max_step_m=float(post_steps.max()) if len(post_steps) else 0.0,
                post_polyline_path_m=float(post_steps.sum()) if len(post_steps) else 0.0,ade_all_vertices_m=float(dev.mean()), max_all_vertices_m=float(dev.max()),
                ade_post_startup_m=float(dev[post].mean()) if post.any() else float("nan"),
                max_post_startup_m=float(dev[post].max()) if post.any() else float("nan"),
                n_vertices_evaluated=int(inside.sum()), n_vertices_post_startup=int(post.sum()),
                startup_skip_s=agent_delay_s + STARTUP_SKIP_STEPS / fps, agent_delay_s=agent_delay_s,
                n_vertices_before_agent_delay=int((poly["t"] < agent_delay_s - 1e-9).sum()),
                first_step_within_1cm_s=float(ts[within][hit[0]]) if len(hit) else float("nan"),
                first_vertex_deviation_m=float(dev[0]),
                polyline_end_s=float(poly["t"][-1]), logged_end_s=t_end,
                polyline_truncated_by_stop=bool(poly["t"][-1] > t_end + 1e-9),
                vertices_beyond_log=int((~inside).sum()))


def population_rows():
    pop = pd.read_csv(POPULATION)
    pop = pop[pop.status.eq("eligible")].copy()
    assert pop.scenario_uid.is_unique      # 489 before the 2026-09-14 cutinl exclusion, 486 after
    return pop.set_index("scenario_uid")


def jobs_for(stage, seeds):
    pop = population_rows()
    analytic = pd.read_csv(ANALYTIC_PAIRED)
    jobs, sources = [], set()
    if stage in ("smoke", "recon"):
        for cls in CLASSES:
            npz = PROJECT / f"models/svd_d5_matched/{cls}/reconstructions.npz"
            sources.add(npz)
            with np.load(npz, allow_pickle=False) as z:
                data = {k: z[k] for k in ("X_rec_fullfit", "X_rec_logo", "scenario_uid", "cohort_eligible", "scenario_id")}
            digest = G.sha256(npz)
            idx = np.nonzero(data["cohort_eligible"])[0]
            modes = ("fullfit",) if stage == "smoke" else ("fullfit", "logo")
            if stage == "smoke":
                idx = idx[:5]
            for i in idx:
                uid = str(data["scenario_uid"][i])
                p = pop.loc[uid]
                assert p.subset == cls
                for mode in modes:
                    vector = data[f"X_rec_{mode}"][i]
                    ref = analytic[analytic.scenario_uid.eq(uid) & analytic["mode"].eq(mode)]
                    assert len(ref) == 1 and int(ref.iloc[0].case_index) == int(i), (uid, mode)
                    assert abs(float(ref.iloc[0].generated_duration_s) - float(vector[100])) < 1e-9
                    assert int(ref.iloc[0].metadata_min_frame) == int(p.min_frame)
                    jobs.append(dict(job_id=f"{cls}__{p.scenario_id}__{mode}", scenario_id=str(p.scenario_id), **{"class": cls},
                        dataset="HetroD", recording="00", ego=int(p.ego), target=int(p.actor),
                        min_frame=int(p.min_frame), max_frame=int(p.max_frame), source_xosc=str(p.base), fps=FPS,
                        geometry_variant_id=f"svd_d5_{mode}_executed_timed_polyline",
                        raw_tracks_path=str(RAW), map_path=str(MAP),
                        vector=[float(v) for v in vector], duration_s=float(vector[100]),
                        source_context=dict(subset=cls, scenario_uid=uid, source_tracks=str(p.source_tracks),
                            mode=mode, case_index=int(i), reconstruction_npz=str(npz.relative_to(PROJECT)),
                            reconstruction_npz_sha256=digest, reconstruction_array_key=f"X_rec_{mode}",
                            raw_duration_s=float(vector[100]), applied_duration_s=float(vector[100]),
                            duration_clipped=False, numeric_ok=True, center_scenario_uid=uid,
                            analytic_paired_csv=str(ANALYTIC_PAIRED), analytic_case_index=int(ref.index[0]),
                            decode_convention="metadata_min_frame + linspace(0, duration, 50) * fps (30_interaction_metrics.py)",
                            execution_label=EXECUTION_LABEL, method=METHOD)))
    elif stage == "kde":
        for seed in seeds:
            for cls in CLASSES:
                npz = PROJECT / f"generated/svd_d5_kde_matched/{cls}/seed_{seed}.npz"
                sources.add(npz)
                with np.load(npz, allow_pickle=False) as z:
                    data = {k: z[k] for k in ("X", "center_case_index", "center_scenario_uid", "raw_duration_s",
                                              "applied_duration_s", "duration_clipped", "numeric_ok", "numeric_status", "attempt_id")}
                    h_loo, design = float(z["h_loo"]), str(z["sampling_design"])
                digest = G.sha256(npz)
                for j in range(100):
                    uid = str(data["center_scenario_uid"][j])
                    p = pop.loc[uid]
                    assert p.subset == cls
                    vector = data["X"][j]
                    applied = float(data["applied_duration_s"][j])
                    assert abs(applied - max(float(data["raw_duration_s"][j]), 0.5)) < 1e-9
                    assert abs(float(vector[100]) - float(data["raw_duration_s"][j])) < 1e-9
                    jobs.append(dict(job_id=str(data["attempt_id"][j]), scenario_id=str(p.scenario_id), **{"class": cls},
                        dataset="HetroD", recording="00", ego=int(p.ego), target=int(p.actor),
                        min_frame=int(p.min_frame), max_frame=int(p.max_frame), source_xosc=str(p.base), fps=FPS,
                        geometry_variant_id="svd_d5_kde_executed_timed_polyline",
                        raw_tracks_path=str(RAW), map_path=str(MAP),
                        vector=[float(v) for v in vector], duration_s=applied,
                        source_context=dict(subset=cls, scenario_uid=uid, source_tracks=str(p.source_tracks),
                            mode="kde", seed=int(seed), draw_index=int(j), attempt_id=str(data["attempt_id"][j]),
                            center_case_index=int(data["center_case_index"][j]), center_scenario_uid=uid,
                            kde_npz=str(npz.relative_to(PROJECT)), kde_npz_sha256=digest, kde_array_key="X",
                            raw_duration_s=float(data["raw_duration_s"][j]), applied_duration_s=applied,
                            duration_clipped=bool(data["duration_clipped"][j]), numeric_ok=bool(data["numeric_ok"][j]),
                            numeric_status=str(data["numeric_status"][j]), h_loo=h_loo, sampling_design=design,
                            decode_convention="center metadata_min_frame + linspace(0, applied_duration, 50) * fps",
                            execution_label=EXECUTION_LABEL, method=METHOD)))
    else:
        raise ValueError(stage)
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    return jobs, sources


def run_job(job, batch, local_bin, run_id, raw_all, source_digest):
    wd = batch / job["job_id"]
    wd.mkdir(exist_ok=True)
    report_path = wd / "sample.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old.get("status") == "completed" and all((wd / n).exists() for n in
                ("patched_source.xosc", "run.xosc", "run.csv", "trajectory.parquet")):
            print(f"[svd-exec] reuse completed {job['job_id']}", flush=True)
            return old
    start = time.perf_counter()
    job_public = {k: v for k, v in job.items() if k != "vector"}
    report = dict(sample_id=job["job_id"], sample_dir=str(wd), job=job_public, run_id=run_id, method=METHOD,
                  execution_label=EXECUTION_LABEL, status="preparing", all_xosc_retained=True, scoring_performed=False,
                  parameters_requested={}, parameters_applied={}, is_nominal_default=False)
    adapter_xosc, executable, csv_path = wd / "patched_source.xosc", wd / "run.xosc", wd / "run.csv"
    sim_start = time.perf_counter()
    try:
        raw = raw_all[raw_all.trackId.isin([job["ego"], job["target"]]) & raw_all.frame.between(job["min_frame"], job["max_frame"])]
        raw = raw.sort_values(["trackId", "frame"])
        target = raw[raw.trackId == job["target"]]
        assert len(target) >= 2 and len(raw[raw.trackId == job["ego"]]) >= 2
        funcs = load_ex_functions(raw, job["fps"])
        poly = decode(job["vector"], job["duration_s"], job["min_frame"], job["fps"])
        p = G.params(job["source_xosc"])
        context = dict(scenario_id=job["scenario_id"], dataset=job["dataset"], recording=job["recording"],
            subset=job["class"], ego=job["ego"], target=job["target"],
            metadata_window_frames=[job["min_frame"], job["max_frame"]],
            target_gt_support_frames=[int(target.frame.min()), int(target.frame.max())], fps=job["fps"],
            geometry_variant_id=job["geometry_variant_id"], source_xosc=job["source_xosc"],
            source_context=job["source_context"], method=METHOD, execution_label=EXECUTION_LABEL,
            agent_shape="timed Polyline of the 50 decoded SVD d5 samples (absolute Timing, position mode)",
            agent_heading_source="finite differences of decoded xy; first vertex = direction to the second",
            agent_speed_source="implied by vertex timing; no SpeedAction",
            fixed_parameter_declarations=p, no_stop_at_goal_added=True, no_endpoint_constraints=True,
            catalogue_entities=True, original_adapter_stop_margin_s=1.0,
            agent_delay_s=float(p["Agent1_Delay"]),
            clock_offset_frames=int(job["min_frame"]) - int(target.frame.min()),
            esmini_traj_filter_note="esmini default --traj_filter 0.1 drops polyline vertices closer than 0.1 m (same flags as 20_ours3_generate.py)",
            decoded_duration_s=poly["duration_s"], decoded_path_length_m=poly["path_length_m"],
            decoded_zero_length_segments=poly["zero_length_segments"])
        report.update(context=context, load_context_s=time.perf_counter() - start)
        report["decoded_polyline"] = dict(x=poly["x"].tolist(), y=poly["y"].tolist(), t=poly["t"].tolist(),
                                          frame=poly["frame"].tolist(), h_rad=poly["h"].tolist())
        prep = time.perf_counter()
        funcs["to_esmini_replay"](Path(job["source_xosc"]), job["dataset"], job["ego"], job["target"],
            job["min_frame"], job["max_frame"], adapter_xosc, None, agent_replay=False)
        shutil.copyfile(adapter_xosc, executable)
        report["xosc_surgery"] = polyline_surgery(executable, poly)
        report["checks"] = verify_executable(Path(job["source_xosc"]), adapter_xosc, executable, poly)
        report["prepare_s"] = time.perf_counter() - prep
        cmd = ["nice", "-n", "10", str(local_bin), "--osc", str(executable), "--headless",
               "--csv_logger", str(csv_path), "--fixed_timestep", str(1.0 / job["fps"]),
               "--logfile_path", str(wd / "esmini.log"), "--disable_stdout"]
        report.update(command=cmd, subprocess_cwd=str(wd), status="executing")
        G.atomic_json(report_path, json_safe(report))
        sim_start = time.perf_counter()
        with (wd / "stdout.txt").open("wb") as stdout, (wd / "stderr.txt").open("wb") as stderr:
            result = subprocess.run(cmd, cwd=wd, stdout=stdout, stderr=stderr, timeout=60)
        report.update(simulate_s=time.perf_counter() - sim_start, returncode=result.returncode)
        if result.returncode != 0:
            raise RuntimeError(f"esmini returned {result.returncode}")
        parse_start = time.perf_counter()
        tidy = G.parse_tidy(csv_path, raw, job["job_id"], context)
        # additive flag only (scorers ignore it): rows at or before the last decoded vertex time
        tidy["within_decoded_polyline_support"] = tidy.time_s <= poly["t"][-1] + 0.5 / job["fps"]
        tidy.to_parquet(wd / "trajectory.parquet", index=False)
        tidy.to_csv(wd / "trajectory.csv", index=False, float_format="%.12g")
        report["fidelity"] = fidelity(tidy, poly, job["fps"], float(p["Agent1_Delay"]))
        report.update(status="completed", parse_export_s=time.perf_counter() - parse_start,
            trajectory_rows=len(tidy), observed_time_range_s=[float(tidy.time_s.min()), float(tidy.time_s.max())],
            trajectory_path=str(wd / "trajectory.parquet"))
    except subprocess.TimeoutExpired:
        report.update(status="timeout", timeout_s=60, simulate_s=time.perf_counter() - sim_start)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["total_s"] = time.perf_counter() - start
    report["artifacts"] = [{"path": str(q), "sha256": G.sha256(q), "size_bytes": q.stat().st_size}
        for q in (adapter_xosc, executable, csv_path, wd / "trajectory.parquet", wd / "trajectory.csv") if q.exists()]
    assert G.sha256(job["source_xosc"]) == source_digest, "upstream base changed"
    G.atomic_json(report_path, json_safe(report))
    fid = report.get("fidelity", {})
    print(f"[svd-exec] {job['job_id']}: {report['status']} ({report['total_s']:.2f}s)"
          + (f" ADE post-startup {fid['ade_post_startup_m'] * 100:.2f} cm, max {fid['max_post_startup_m'] * 100:.2f} cm" if fid else ""), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("smoke", "recon", "kde"), required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[20260910, 20260911, 20260912])
    parser.add_argument("--limit", type=int, default=None, help="debug: run only the first N jobs")
    args = parser.parse_args()
    jobs, npz_sources = jobs_for(args.stage, args.seeds)
    if args.limit:
        jobs = jobs[:args.limit]
    batch_name = f"svd_d5_executed_{args.stage}"
    out = PROJECT / "runs" / batch_name
    out.mkdir(parents=True, exist_ok=True)
    sources = {EX_SOURCE, RUNNER_20, BINARY, Path(__file__).resolve(), POPULATION, ANALYTIC_PAIRED, RAW, MAP, *npz_sources}
    sources.update(Path(j["source_xosc"]) for j in jobs)
    for cat in ("Vehicles/VehicleCatalog.xosc", "Pedestrians/PedestrianCatalog.xosc",
                "Controllers/ControllerCatalog.xosc", "Environments/EnvironmentCatalog.xosc"):
        sources.add(Path("/opt/Catalogs") / cat)
    provenance = [{"path": str(p), "sha256": G.sha256(p), "size_bytes": p.stat().st_size} for p in sorted(sources)]
    digest_of = {p["path"]: p["sha256"] for p in provenance}
    protocol = dict(method=METHOD, execution_label=EXECUTION_LABEL, stage=args.stage, batch_name=batch_name,
        seeds=args.seeds if args.stage == "kde" else None,
        jobs=[{k: v for k, v in j.items() if k != "vector"} for j in jobs], sources=provenance,
        subprocess_timeout_s=60, worker_count=1, nice=10, min_free_disk_bytes=MIN_FREE,
        esmini_flags=["--headless", "--csv_logger", "--fixed_timestep 1/fps", "--logfile_path", "--disable_stdout"],
        all_xosc_retained=True, scoring_performed=False, stop_on_failure=False,
        agent_shape="timed Polyline (absolute Timing) of the 50 decoded samples",
        removed_from_base="Agent1 Nurbs FollowTrajectory shape; Agent1_StartSpeedAction; Agent1_SpeedEvent",
        startup_skip_steps=STARTUP_SKIP_STEPS, smoke_gate=dict(ade_m=SMOKE_ADE_GATE_M, max_m=SMOKE_MAX_GATE_M))
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    batch = out / run_id
    batch.mkdir(exist_ok=True)
    local_bin = PROJECT / "bin/esmini"
    local_bin.parent.mkdir(exist_ok=True)
    if not local_bin.exists():
        local_bin.symlink_to(BINARY)
    assert local_bin.resolve() == BINARY.resolve()
    if (local_bin.parent / "config.yml").exists():
        raise RuntimeError("Unexpected local esmini config; will not silently override behavior")
    snapshot = batch / "runner_snapshot.py"
    if not snapshot.exists():
        shutil.copyfile(__file__, snapshot)
    G.atomic_json(batch / "protocol.json", protocol)
    raw_all = pd.read_parquet(RAW, columns=["trackId", "frame", "xCenter", "yCenter", "heading", "xVelocity", "yVelocity", "length", "width"])
    status = dict(run_id=run_id, batch_dir=str(batch), execution_label=EXECUTION_LABEL, status="running", samples=[])
    t0 = time.perf_counter()
    for n, job in enumerate(jobs):
        if shutil.disk_usage(batch).free < MIN_FREE:
            status.update(status="paused_low_disk", remaining_sample=job["job_id"])
            G.atomic_json(batch / "batch_status.json", status)
            print("PAUSED: less than 10 GiB free; no files deleted. Resume after freeing space.", flush=True)
            return 75
        report = run_job(job, batch, local_bin, run_id, raw_all, digest_of[job["source_xosc"]])
        status["samples"].append({k: report.get(k) for k in ("sample_id", "sample_dir", "status", "error", "total_s", "simulate_s", "trajectory_path", "fidelity", "checks", "xosc_surgery")})
        if (n + 1) % 25 == 0 or n + 1 == len(jobs):
            G.atomic_json(batch / "batch_status.json", json_safe(status))
            print(f"[svd-exec] {n + 1}/{len(jobs)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    status["status"] = "completed" if all(r["status"] == "completed" for r in status["samples"]) else "completed_with_failures"
    status["elapsed_s"] = time.perf_counter() - t0
    G.atomic_json(batch / "batch_status.json", json_safe(status))
    rows = []
    for r, job in zip(status["samples"], jobs):
        fid = r.get("fidelity") or {}
        rows.append(dict(sample_id=r["sample_id"], status=r["status"], sample_dir=r["sample_dir"], trajectory_path=r.get("trajectory_path"),
            scenario_id=job["scenario_id"], subset=job["class"], scenario_uid=job["source_context"]["scenario_uid"],
            mode=job["source_context"]["mode"], method=METHOD, execution_label=EXECUTION_LABEL,
            duration_s=job["duration_s"], raw_duration_s=job["source_context"]["raw_duration_s"],
            numeric_ok=job["source_context"]["numeric_ok"], duration_clipped=job["source_context"]["duration_clipped"],
            **{k: fid.get(k) for k in ("ade_all_vertices_m", "max_all_vertices_m", "ade_post_startup_m", "max_post_startup_m",
                                        "first_step_within_1cm_s", "polyline_truncated_by_stop", "vertices_beyond_log")},
            simulate_s=r.get("simulate_s"), total_s=r.get("total_s"), error=r.get("error")))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(batch / "manifest.csv", index=False)
    G.atomic_json(out / "latest_batch.json", dict(batch_dir=str(batch), status=status["status"], run_id=run_id,
        execution_label=EXECUTION_LABEL, manifest=str(batch / "manifest.csv")))
    ok = manifest[manifest.status.eq("completed")]
    summary = dict(status=status["status"], samples=len(jobs), completed=int(len(ok)), batch_dir=str(batch),
                   execution_label=EXECUTION_LABEL, elapsed_s=status["elapsed_s"],
                   ade_post_startup_m=dict(median=float(ok.ade_post_startup_m.median()), max=float(ok.ade_post_startup_m.max())) if len(ok) else None,
                   max_post_startup_m=dict(median=float(ok.max_post_startup_m.median()), max=float(ok.max_post_startup_m.max())) if len(ok) else None,
                   ade_all_vertices_m=dict(median=float(ok.ade_all_vertices_m.median()), max=float(ok.ade_all_vertices_m.max())) if len(ok) else None)
    if args.stage == "smoke":
        gate = bool(len(ok) == len(jobs) and (ok.ade_post_startup_m < SMOKE_ADE_GATE_M).all() and (ok.max_post_startup_m < SMOKE_MAX_GATE_M).all())
        summary.update(smoke_gate_passed=gate, smoke_gate=dict(ade_m=SMOKE_ADE_GATE_M, max_m=SMOKE_MAX_GATE_M, scope="vertices at t >= Agent1_Delay + 5/fps"))
        G.atomic_json(batch / "smoke_gate.json", json_safe(dict(summary, per_sample=ok[["sample_id", "ade_all_vertices_m", "max_all_vertices_m", "ade_post_startup_m", "max_post_startup_m", "first_step_within_1cm_s"]].to_dict("records"))))
        print(json.dumps(json_safe(summary), indent=2), flush=True)
        assert gate, "SMOKE ADE GATE FAILED"
        return 0
    print(json.dumps(json_safe(summary), indent=2), flush=True)
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
