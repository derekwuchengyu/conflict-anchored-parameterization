#!/usr/bin/env python3
"""Execute approved Ours3 exact-rotation jobs through the original headless adapter.

No arguments: 39_180 default + six OAT variants. --jobs-json: reusable job batch.
Only theta1/theta2/EndSpeed vary. All XOSC and logged samples are retained.
No scoring, new collision adapter, endpoint constraint, or stop-at-goal is added.
Upstream functions are AST-loaded to avoid module import filesystem side effects.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
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
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RAW = ROOT / "HetroD-labeler/data/00_tracks.parquet"
CV_SOURCE = ROOT / "exp_coverage_velocity/scripts/cvrender.py"
EX_SOURCE = ROOT / "hetero-param/hetero_param/esmini_exec.py"
ARC_SOURCE = ROOT / "exp_coverage_velocity/scripts/cvlib.py"
MAP = ROOT / "retrieval-scenarios/data/map/tyms.xodr"
BINARY = ROOT / "esmini/bin/esmini"
MIN_FREE = 10 * 1024 ** 3
SPEED_PARAMETER = "Agent1_1_SA_EndSpeed"
PARAMETER_NAMES = ["theta1_deg", "theta2_deg", "interaction_end_speed_kmh"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def ast_functions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError(f"Missing source function: {set(names) - {node.name for node in nodes}}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def literal_constant(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def params(path):
    return {p.get("name"): p.get("value") for p in ET.parse(path).getroot().iter("ParameterDeclaration")}


def signature(element):
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
            tuple(signature(child) for child in element))


def normalize_job(job):
    result = dict(job)
    for key in ("job_id", "scenario_id", "ego", "target", "min_frame", "max_frame", "source_xosc", "anchor_frame"):
        if key not in result:
            raise ValueError(f"Missing job field: {key}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", result["job_id"]) or result["job_id"] in (".", ".."):
        raise ValueError("job_id must be a plain directory name")
    for key in ("ego", "target", "min_frame", "max_frame", "anchor_frame"):
        result[key] = int(result[key])
    result.setdefault("fps", 30.0)
    result.setdefault("dataset", "HetroD")
    result.setdefault("recording", "00")
    result.setdefault("geometry_variant_id", "source_cp3_exact_rotation")
    result.setdefault("raw_tracks_path", str(RAW))
    result.setdefault("map_path", str(MAP))
    result.setdefault("parameters", {})
    result.setdefault("anchor_match_tolerance_m", 0.001)
    result.setdefault("source_files", [])
    for key in ("source_xosc", "raw_tracks_path", "map_path"):
        result[key] = str(Path(result[key]).resolve())
    result["source_files"] = [str(Path(p).resolve()) for p in result["source_files"]]
    if set(result["parameters"]) - set(PARAMETER_NAMES):
        raise ValueError("Only theta1_deg, theta2_deg, interaction_end_speed_kmh may vary")
    assert result["ego"] != result["target"]
    assert result["min_frame"] <= result["anchor_frame"] <= result["max_frame"]
    assert result["max_frame"] > result["min_frame"] and result["fps"] > 0
    return result


def load_gt(job):
    return pd.read_parquet(job["raw_tracks_path"], filters=[("trackId", "in", [job["ego"], job["target"]]),
        ("frame", ">=", job["min_frame"]), ("frame", "<=", job["max_frame"])]).sort_values(["trackId", "frame"])


def trajectory(raw, actor, lower, upper, fps):
    g = raw[(raw.trackId == actor) & raw.frame.between(lower, upper)].drop_duplicates("frame").sort_values("frame")
    if len(g) < 2:
        return None
    return SimpleNamespace(frame=g.frame.to_numpy(float), x=g.xCenter.to_numpy(float), y=g.yCenter.to_numpy(float),
        heading=g.heading.to_numpy(float), speed=np.hypot(g.xVelocity, g.yVelocity).to_numpy(float),
        fps=fps, length=float(g.length.median()), width=float(g.width.median()))


def load_pure_functions(raw, job):
    namespace = {"np": np, "pd": pd, "ET": ET, "Path": Path, "NT": 50}
    ast_functions(ARC_SOURCE, ["arc_resample"], namespace)
    target = raw[raw.trackId == job["target"]].rename(columns={"xCenter": "x", "yCenter": "y"})
    namespace["L"] = SimpleNamespace(real_actor_track=lambda _T, _sid: target,
                                     arc_resample=namespace["arc_resample"])
    namespace["KNOTS_7CP"] = literal_constant(CV_SOURCE, "KNOTS_7CP")
    ast_functions(CV_SOURCE, ["_signed_angle", "_rot", "theta_geometry", "theta_to_cps", "patch_cps"], namespace)
    namespace["paths"] = SimpleNamespace(dataset_of=lambda _dataset: {"fps": job["fps"], "xodr": job["map_path"]})
    namespace["SIMC"] = SimpleNamespace(real_traj=lambda _dataset, actor, lo, hi:
                                         trajectory(raw, actor, lo, hi, job["fps"]))
    namespace["CATALOG_ENTITIES"] = True
    namespace["_CATALOG_DIRS"] = literal_constant(EX_SOURCE, "_CATALOG_DIRS")
    ast_functions(EX_SOURCE, ["_standard_catalogs", "_ego_replay_story", "_sim_time_stop", "to_esmini_replay"], namespace)
    return namespace


def context_from_source(raw, funcs, job):
    root = ET.parse(job["source_xosc"]).getroot()
    nurbs = root.findall(".//Nurbs")
    assert len(nurbs) == 1
    if "cps_xy" in job:
        xyz = np.asarray(job["cps_xy"], float).reshape(3, 2)
    else:
        cp = [cp for cp in nurbs[0].findall("ControlPoint") if cp.get("weight") is not None]
        assert len(cp) == 3
        xyz = np.array([[float(c.find("Position/WorldPosition").get(axis)) for axis in ("x", "y")] for c in cp])
    assert np.isfinite(xyz).all()
    geom = funcs["theta_geometry"](None, job["scenario_id"], xyz.reshape(-1))
    if geom is None:
        raise ValueError("Existing exact geometry rejects L1 or L2 below 1 m")
    anchor = raw[(raw.trackId == job["target"]) & (raw.frame == job["anchor_frame"])]
    assert len(anchor) == 1, "Anchor frame is absent or ambiguous"
    a = anchor.iloc[0]
    distance = float(np.linalg.norm(xyz[1] - [a.xCenter, a.yCenter]))
    assert distance <= job["anchor_match_tolerance_m"], f"Central CP / recorded anchor mismatch {distance} m"
    speed_kmh = float(np.hypot(a.xVelocity, a.yVelocity) * 3.6)
    assert np.isfinite(speed_kmh) and speed_kmh >= 0
    target = raw[raw.trackId == job["target"]]
    if "target_gt_start_frame" in job:
        target = target[target.frame >= int(job["target_gt_start_frame"])]
        assert len(target) >= 2, "target support after target_gt_start_frame is too short"
    p = params(job["source_xosc"])
    context = dict(scenario_id=job["scenario_id"], dataset=job["dataset"], recording=job["recording"],
        subset=job.get("class"), ego=job["ego"], target=job["target"],
        metadata_window_frames=[job["min_frame"], job["max_frame"]],
        target_gt_support_frames=[int(target.frame.min()), int(target.frame.max())], fps=job["fps"],
        geometry_variant_id=job["geometry_variant_id"], q_minus_xy=xyz[0].tolist(),
        q_minus_requested_xy=xyz[0].tolist(), q_minus_written_xy=[float(f"{v:.4f}") for v in xyz[0]],
        q_critical_default_xy=xyz[1].tolist(), q_plus_default_xy=xyz[2].tolist(),
        L1_m=float(geom["L1"]), L2_m=float(geom["L2"]), incoming_unit_xy=geom["u1"].tolist(),
        incoming_source=geom["u1_source"], theta1_default_deg=float(geom["theta1"]),
        theta2_default_deg=float(geom["theta2"]), anchor_frame=job["anchor_frame"],
        anchor_position_match_m=distance, default_end_speed_kmh=speed_kmh,
        speed_source=f"hypot(raw.xVelocity, raw.yVelocity) at frame {job['anchor_frame']}",
        fixed_offset_raw=p["Agent1_Offset"], fixed_duration_raw=p["Agent1_1_SA_DynamicDuration"],
        fixed_start_speed_raw=p["Agent1_Speed"], fixed_parameter_declarations={k: v for k, v in p.items() if k != SPEED_PARAMETER},
        source_end_speed_raw=p[SPEED_PARAMETER], source_xosc=job["source_xosc"],
        no_stop_at_goal_added=True, no_endpoint_constraints=True, catalogue_entities=True,
        original_adapter_stop_margin_s=1.0)
    if "source_context" in job:
        context["source_context"] = job["source_context"]
    if "expected_default_end_speed_kmh" in job:
        assert abs(speed_kmh - job["expected_default_end_speed_kmh"]) < 1e-8
    if "expected_target_gt_end" in job:
        assert int(target.frame.max()) == job["expected_target_gt_end"]
    default = [context["theta1_default_deg"], context["theta2_default_deg"], speed_kmh]
    requested = {name: default[i] if job["parameters"].get(name) is None else float(job["parameters"][name])
                 for i, name in enumerate(PARAMETER_NAMES)}
    if "theta1_delta_deg" in job:
        requested["theta1_deg"] += float(job["theta1_delta_deg"])
    if "theta2_delta_deg" in job:
        requested["theta2_deg"] += float(job["theta2_delta_deg"])
    if "end_speed_factor" in job:
        requested["interaction_end_speed_kmh"] *= float(job["end_speed_factor"])
    assert all(np.isfinite(value) for value in requested.values())
    assert requested["interaction_end_speed_kmh"] >= 0, "Negative EndSpeed requires a reviewed policy; no clipping"
    return context, geom, p, requested


def set_middle_cp_weight(path, weight):
    """Conflict-CP weight override (uturn_859_881 uses 8; patch_cps always writes 5).
    Applied to the already-patched 3 weighted CPs, so the geometry is untouched."""
    tree = ET.parse(path)
    cp = [c for c in tree.getroot().findall(".//Nurbs/ControlPoint") if c.get("weight") is not None]
    assert len(cp) == 3, "middle_cp_weight expects the 3 weighted shape CPs"
    cp[1].set("weight", f"{weight:g}")
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return weight


def verify_xosc(patched, executable, original_params, context):
    assert params(patched) == original_params
    generated = params(executable)
    assert set(generated) == set(original_params)
    for name, value in original_params.items():
        if name != SPEED_PARAMETER:
            assert generated[name] == value, (name, generated[name], value)
    src = ET.parse(context["source_xosc"]).getroot()
    dst = ET.parse(executable).getroot()
    assert signature(src.find("Entities")) == signature(dst.find("Entities")), "CatalogReference changed"
    cp = [c for c in dst.findall(".//Nurbs/ControlPoint") if c.get("weight") is not None]
    assert len(cp) == 3
    first = cp[0].find("Position/WorldPosition")
    written = [float(first.get(k)) for k in ("x", "y")]
    assert written == context["q_minus_written_xy"]
    error = np.abs(np.asarray(written) - context["q_minus_requested_xy"])
    assert max(error) <= 5.00001e-5
    src_stops = {e.get("name") for e in src.iter("Event") if "StopAtEnd" in e.get("name", "") or "StopAtGoal" in e.get("name", "")}
    dst_stops = {e.get("name") for e in dst.iter("Event") if "StopAtEnd" in e.get("name", "") or "StopAtGoal" in e.get("name", "")}
    assert dst_stops == src_stops, "Stop-at-goal event was added or removed"
    compared = []
    for source_event in src.iter("Event"):
        if source_event.find(".//SpeedAction") is None:
            continue
        name = source_event.get("name")
        matches = [e for e in dst.iter("Event") if e.get("name") == name]
        assert len(matches) == 1, f"Original SpeedAction event missing or duplicate: {name}"
        # The reused adapter intentionally rewires FLAG-AV_CONNECTED to t > 0.
        # Normalize that documented existing rewrite before checking the full
        # Event; the SpeedAction and its other timing remain identical.
        expected = ET.fromstring(ET.tostring(source_event))
        for cond in expected.iter("Condition"):
            bv = cond.find("ByValueCondition")
            pc = bv.find("ParameterCondition") if bv is not None else None
            if pc is not None and pc.get("parameterRef") == "FLAG-AV_CONNECTED":
                bv.remove(pc)
                ET.SubElement(bv, "SimulationTimeCondition", value="0.0", rule="greaterThan")
                cond.set("conditionEdge", "none")
        assert signature(expected) == signature(matches[0]), f"SpeedAction event changed beyond existing adapter: {name}"
        compared.append(name)
    assert compared, "No original SpeedAction event found"
    return dict(fixed_declarations_unchanged=True, catalog_reference_entities_unchanged=True,
        q_minus_unchanged=True, q_minus_rounding_error_per_axis_m=error.tolist(),
        original_speed_event_unchanged=True, original_speed_event_names=compared,
        no_stop_at_goal_added=True, no_endpoint_constraints_added=True)


def parse_tidy(csv_path, raw, sample_id, context):
    """Retain every sample; dimensions are recorded metadata, not body replacement."""
    lines = csv_path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("Index"))
    columns = [value.strip() for value in lines[start].split(",")]
    while columns and not columns[-1]:
        columns.pop()
    df = pd.read_csv(csv_path, skiprows=start + 1, names=columns, header=None,
                     usecols=range(len(columns)), skipinitialspace=True)
    records = []
    for column in columns:
        match = re.fullmatch(r"#(\d+) Entity_Name \[-\]", column)
        if not match:
            continue
        n = match.group(1)
        entity = str(df[column].iloc[0]).strip()
        if entity not in ("Ego", "Agent1"):
            raise RuntimeError(f"Unexpected entity {entity}; actor mapping requires review")
        actor, role = (context["ego"], "ego") if entity == "Ego" else (context["target"], "target")
        gt = raw[raw.trackId == actor]
        result = pd.DataFrame(dict(sample_id=sample_id, scenario_id=context["scenario_id"],
            subset=context["subset"], entity_name=entity, actor_id=str(actor), role=role,
            time_s=df["TimeStamp [s]"].astype(float),
            x=df[f"#{n} World_Position_X [m]"].astype(float), y=df[f"#{n} World_Position_Y [m]"].astype(float),
            z=df[f"#{n} World_Position_Z [m]"].astype(float),
            heading_deg=np.degrees(df[f"#{n} World_Heading_Angle [rad]"].astype(float)),
            speed_mps=df[f"#{n} Current_Speed [m/s]"].astype(float),
            length=float(gt.length.median()), width=float(gt.width.median())))
        for output, header in (("vx_mps", "Vel_X [m/s]"), ("vy_mps", "Vel_Y [m/s]"),
                               ("ax_mps2", "Acc_X [m/s2]"), ("ay_mps2", "Acc_Y [m/s2]")):
            result[output] = df[f"#{n} {header}"].astype(float)
        result["frame"] = context["metadata_window_frames"][0] + result.time_s * context["fps"]
        lo, hi = context["metadata_window_frames"]
        result["within_metadata_window"] = result.frame.between(lo - .5, hi + .5)
        lo, hi = context["target_gt_support_frames"]
        result["within_target_gt_support"] = result.frame.between(lo - .5, hi + .5)
        result["within_entity_gt_support"] = result.frame.between(float(gt.frame.min()) - .5, float(gt.frame.max()) + .5)
        records.append(result)
    if len(records) != 2:
        raise RuntimeError(f"Expected 2 entities, saw {len(records)}")
    return pd.concat(records, ignore_index=True)


def run_job(job, batch, local_bin, run_id, source_digest):
    """Sequential reusable job execution. Never remove upstream or run artifacts."""
    wd = batch / job["job_id"]
    wd.mkdir(exist_ok=True)
    report_path = wd / "sample.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old.get("status") == "completed" and all((wd / n).exists() for n in
                ("patched_source.xosc", "run.xosc", "run.csv", "trajectory.parquet")):
            print(f"[ours3] reuse completed {job['job_id']}", flush=True)
            return old
    start = time.perf_counter()
    report = dict(sample_id=job["job_id"], sample_dir=str(wd), job=job, run_id=run_id,
                  status="preparing", all_xosc_retained=True, scoring_performed=False)
    patched, executable, csv_path = wd / "patched_source.xosc", wd / "run.xosc", wd / "run.csv"
    try:
        raw = load_gt(job)
        funcs = load_pure_functions(raw, job)
        context, geom, original_params, requested = context_from_source(raw, funcs, job)
        report.update(context=context, parameters_requested=requested, parameters_applied=requested.copy(),
                      load_context_s=time.perf_counter() - start)
        cp_start = time.perf_counter()
        cps = funcs["theta_to_cps"](geom, requested["theta1_deg"], requested["theta2_deg"])
        funcs["patch_cps"](Path(job["source_xosc"]), cps, patched)
        if job.get("middle_cp_weight") is not None:
            set_middle_cp_weight(patched, float(job["middle_cp_weight"]))
        funcs["to_esmini_replay"](patched, job["dataset"], job["ego"], job["target"],
            job["min_frame"], job["max_frame"], executable,
            {SPEED_PARAMETER: requested["interaction_end_speed_kmh"]}, agent_replay=False)
        report["prepare_s"] = time.perf_counter() - cp_start
        report["checks"] = verify_xosc(patched, executable, original_params, context)
        assert abs(float(params(executable)[SPEED_PARAMETER]) - requested["interaction_end_speed_kmh"]) < 1e-12
        report["cps_requested_xy"] = cps.reshape(3, 2).tolist()
        report["cps_written_xy"] = [[float(f"{v:.4f}") for v in xy] for xy in cps.reshape(3, 2)]
        cmd = ["nice", "-n", "10", str(local_bin), "--osc", str(executable), "--headless",
               "--csv_logger", str(csv_path), "--fixed_timestep", str(1.0 / job["fps"]),
               "--logfile_path", str(wd / "esmini.log"), "--disable_stdout"]
        report.update(command=cmd, subprocess_cwd=str(wd), status="executing")
        atomic_json(report_path, report)
        sim_start = time.perf_counter()
        with (wd / "stdout.txt").open("wb") as stdout, (wd / "stderr.txt").open("wb") as stderr:
            result = subprocess.run(cmd, cwd=wd, stdout=stdout, stderr=stderr, timeout=60)
        report.update(simulate_s=time.perf_counter() - sim_start, returncode=result.returncode)
        if result.returncode != 0:
            raise RuntimeError(f"esmini returned {result.returncode}")
        parse_start = time.perf_counter()
        tidy = parse_tidy(csv_path, raw, job["job_id"], context)
        tidy.to_parquet(wd / "trajectory.parquet", index=False)
        tidy.to_csv(wd / "trajectory.csv", index=False, float_format="%.12g")
        report.update(status="completed", parse_export_s=time.perf_counter() - parse_start,
            trajectory_rows=len(tidy), observed_time_range_s=[float(tidy.time_s.min()), float(tidy.time_s.max())],
            trajectory_path=str(wd / "trajectory.parquet"))
    except subprocess.TimeoutExpired:
        report.update(status="timeout", timeout_s=60, simulate_s=time.perf_counter() - sim_start)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["total_s"] = time.perf_counter() - start
    report["artifacts"] = [{"path": str(p), "sha256": sha256(p), "size_bytes": p.stat().st_size}
        for p in (patched, executable, csv_path, wd / "trajectory.parquet", wd / "trajectory.csv") if p.exists()]
    assert sha256(job["source_xosc"]) == source_digest, "upstream base changed"
    atomic_json(report_path, report)
    print(f"[ours3] {job['job_id']}: {report['status']} ({report['total_s']:.2f}s)", flush=True)
    return report


def default_spec():
    base = dict(scenario_id="39_180", **{"class": "cutinl"}, dataset="HetroD", recording="00", ego=39, target=180,
        min_frame=2424, max_frame=3002, anchor_frame=2641,
        source_xosc=str(ROOT / "exp_cross_coverage/esmini_runs/_xosc_base_anch_px_s10/HetroD-01KEEP_02CUTIN_L_39_180_f2425.xosc"),
        geometry_variant_id="special_pet_cp3d10_exact_rotation", expected_default_end_speed_kmh=8.836849193507831,
        expected_target_gt_end=2923)
    jobs = [dict(base, job_id="default")]
    for axis in (1, 2):
        for label, delta in (("minus", -10), ("plus", 10)):
            jobs.append(dict(base, job_id=f"theta{axis}_{label}10deg", **{f"theta{axis}_delta_deg": delta}))
    for label, factor in (("minus", .8), ("plus", 1.2)):
        jobs.append(dict(base, job_id=f"end_speed_{label}20pct", end_speed_factor=factor))
    return dict(batch_name="ours3_39_180_oat", jobs=jobs)


def run_batch(spec, input_path=None):
    batch_name = spec["batch_name"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch_name) or batch_name in (".", ".."):
        raise ValueError("batch_name must be a plain directory name")
    jobs = [normalize_job(j) for j in spec["jobs"]]
    if not jobs or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Expected nonempty unique job IDs")
    out = PROJECT / "runs" / batch_name
    out.mkdir(parents=True, exist_ok=True)
    sources = {CV_SOURCE, EX_SOURCE, ARC_SOURCE, BINARY, Path(__file__).resolve()}
    for job in jobs:
        sources.update(Path(job[k]) for k in ("source_xosc", "raw_tracks_path", "map_path"))
        sources.update(Path(p) for p in job["source_files"])
    if input_path is not None:
        sources.add(Path(input_path).resolve())
    for cat in ("Vehicles/VehicleCatalog.xosc", "Pedestrians/PedestrianCatalog.xosc",
                "Controllers/ControllerCatalog.xosc", "Environments/EnvironmentCatalog.xosc"):
        sources.add(Path("/opt/Catalogs") / cat)
    provenance = [{"path": str(p), "sha256": sha256(p), "size_bytes": p.stat().st_size} for p in sorted(sources)]
    digest_of = {p["path"]: p["sha256"] for p in provenance}
    protocol = dict(method="ours3_exact", batch_name=batch_name, jobs=jobs, sources=provenance,
        subprocess_timeout_s=60, worker_count=1, nice=10, min_free_disk_bytes=MIN_FREE,
        all_xosc_retained=True, scoring_performed=False, stop_on_failure=spec.get("stop_on_failure", True))
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
    atomic_json(batch / "protocol.json", protocol)
    status = dict(run_id=run_id, batch_dir=str(batch), status="running", samples=[])
    for job in jobs:
        if shutil.disk_usage(batch).free < MIN_FREE:
            status.update(status="paused_low_disk", remaining_sample=job["job_id"])
            atomic_json(batch / "batch_status.json", status)
            print("PAUSED: less than 10 GiB free; no files deleted. Resume after freeing space.", flush=True)
            return 75
        report = run_job(job, batch, local_bin, run_id, digest_of[job["source_xosc"]])
        status["samples"].append(report)
        atomic_json(batch / "batch_status.json", status)
        if report["status"] != "completed" and protocol["stop_on_failure"]:
            status.update(status="paused_failure_requires_review")
            atomic_json(batch / "batch_status.json", status)
            print(f"PAUSED after failure; artifacts retained: {report['sample_dir']}/sample.json", flush=True)
            return 1
    status["status"] = "completed" if all(r["status"] == "completed" for r in status["samples"]) else "completed_with_failures"
    atomic_json(batch / "batch_status.json", status)
    manifest = pd.DataFrame([dict(sample_id=r["sample_id"], status=r["status"], sample_dir=r["sample_dir"],
        trajectory_path=r.get("trajectory_path"), scenario_id=r["job"]["scenario_id"], subset=r["job"].get("class"),
        **r.get("parameters_applied", {}), load_context_s=r.get("load_context_s"), prepare_s=r.get("prepare_s"),
        simulate_s=r.get("simulate_s"), parse_export_s=r.get("parse_export_s"), total_s=r["total_s"],
        error=r.get("error")) for r in status["samples"]])
    manifest.to_csv(batch / "manifest.csv", index=False)
    atomic_json(out / "latest_batch.json", dict(batch_dir=str(batch), status=status["status"], run_id=run_id,
        default_trajectory=str(batch / jobs[0]["job_id"] / "trajectory.parquet"), manifest=str(batch / "manifest.csv")))
    print(json.dumps({"status": status["status"], "samples": len(jobs), "batch_dir": str(batch)}), flush=True)
    return 0 if status["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", type=Path, help="JSON object with batch_name and jobs list; omitted: original seven OAT jobs")
    args = parser.parse_args()
    spec = json.loads(args.jobs_json.read_text()) if args.jobs_json else default_spec()
    return run_batch(spec, args.jobs_json)


if __name__ == "__main__":
    raise SystemExit(main())
