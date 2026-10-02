#!/usr/bin/env python3
"""Execute 192 existing exact-window SAKURA bc templates, retaining artifacts.

The only parameter override is the authorized default EndSpeed: recorded target
endpoint hypot(vx,vy), in km/h. Existing duration, offset, initial speed, geometry
and event timing stay fixed. Existing EX.to_esmini_replay adapter behavior and
CatalogReference entities are reused. Missing sources are never regenerated.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
HELPER = PROJECT / "scripts/20_ours3_generate.py"
spec = importlib.util.spec_from_file_location("sakura_existing_runner_helpers", HELPER)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
POP = PROJECT / "results/sakura_source_population.csv"
CASES = PROJECT / "results/svd_d5_cases.csv"
SPEED = "Agent1_1_SA_EndSpeed"
METHOD = "sakura_bc"


def adapter_functions(raw, job):
    ns = {"np": np, "ET": ET, "Path": Path,
          "paths": SimpleNamespace(dataset_of=lambda _dataset: {"fps": job["fps"], "xodr": job["map_path"]}),
          "SIMC": SimpleNamespace(real_traj=lambda _dataset, actor, lo, hi:
                                   H.trajectory(raw, actor, lo, hi, job["fps"])),
          "CATALOG_ENTITIES": True, "_CATALOG_DIRS": H.literal_constant(H.EX_SOURCE, "_CATALOG_DIRS")}
    H.ast_functions(H.EX_SOURCE, ["_standard_catalogs", "_ego_replay_story", "_sim_time_stop", "to_esmini_replay"], ns)
    return ns


def verify(original, executable, end_speed):
    before, after = H.params(original), H.params(executable)
    assert set(before) == set(after)
    for key, value in before.items():
        if key != SPEED:
            assert value == after[key], f"Fixed declaration changed: {key}"
    assert abs(float(after[SPEED]) - end_speed) < 1e-12
    src, dst = ET.parse(original).getroot(), ET.parse(executable).getroot()
    assert H.signature(src.find("Entities")) == H.signature(dst.find("Entities")), "CatalogReference changed"
    source_nurbs, output_nurbs = src.findall(".//Nurbs"), dst.findall(".//Nurbs")
    assert len(source_nurbs) == len(output_nurbs) == 1
    assert H.signature(source_nurbs[0]) == H.signature(output_nurbs[0]), "SAKURA geometry changed"
    assert len(source_nurbs[0].findall("ControlPoint")) == 4
    assert not [p for p in source_nurbs[0].findall("ControlPoint") if p.get("weight") is not None]
    original_stops = {e.get("name") for e in src.iter("Event") if "StopAtEnd" in e.get("name", "") or "StopAtGoal" in e.get("name", "")}
    output_stops = {e.get("name") for e in dst.iter("Event") if "StopAtEnd" in e.get("name", "") or "StopAtGoal" in e.get("name", "")}
    assert original_stops == output_stops
    checked_events = []
    for event in src.iter("Event"):
        if event.find(".//SpeedAction") is None:
            continue
        name = event.get("name")
        matches = [e for e in dst.iter("Event") if e.get("name") == name]
        assert len(matches) == 1
        expected = ET.fromstring(ET.tostring(event))
        # Existing adapter's documented FLAG trigger rewrite; no new timing rule.
        for cond in expected.iter("Condition"):
            bv = cond.find("ByValueCondition")
            pc = bv.find("ParameterCondition") if bv is not None else None
            if pc is not None and pc.get("parameterRef") == "FLAG-AV_CONNECTED":
                bv.remove(pc)
                ET.SubElement(bv, "SimulationTimeCondition", value="0.0", rule="greaterThan")
                cond.set("conditionEdge", "none")
        assert H.signature(expected) == H.signature(matches[0]), f"Unexpected speed-event change: {name}"
        checked_events.append(name)
    assert set(checked_events) == {"Agent1_StartEvent", "Agent1_SpeedEvent"}
    return {"only_end_speed_declaration_overridden": True,
            "fixed_duration_offset_initial_speed_unchanged": True,
            "catalog_reference_entities_unchanged": True,
            "four_boundary_helper_cp_geometry_unchanged": True,
            "original_speed_events_unchanged_except_existing_flag_adapter": checked_events,
            "no_stop_at_goal_added": True, "no_endpoint_constraint_added": True,
            "no_duration_or_clock_correction": True}


def run_one(job, batch, run_id, binary):
    wd = batch / job["job_id"]
    wd.mkdir(exist_ok=True)
    sample_path = wd / "sample.json"
    if sample_path.exists():
        old = json.loads(sample_path.read_text())
        if old.get("status") == "completed" and all(Path(a["path"]).exists() and H.sha256(a["path"]) == a["sha256"] for a in old["artifacts"]):
            return old
        # Prior artifacts must not be overwritten after a failed interrupted run.
        raise RuntimeError(f"Existing incomplete sample requires a new batch, preserving artifacts: {wd}")
    t0 = time.perf_counter()
    source_copy, executable, csv_path = wd / "source.xosc", wd / "run.xosc", wd / "run.csv"
    report = dict(sample_id=job["job_id"], sample_dir=str(wd), job=job, run_id=run_id,
                  method=METHOD, is_nominal_default=True, status="preparing", all_xosc_retained=True,
                  scoring_performed=False, parameters_requested={}, parameters_applied={})
    try:
        raw = H.load_gt(job)
        target = raw[raw.trackId.eq(job["target"])].sort_values("frame")
        assert len(target) > 1 and target.frame.is_unique
        end = target.iloc[-1]
        end_speed = float(np.hypot(end.xVelocity, end.yVelocity) * 3.6)
        assert np.isfinite(end_speed) and end_speed >= 0
        assert abs(end_speed - job["expected_end_speed_kmh"]) < 1e-8
        original = H.params(job["source_xosc"])
        context = dict(method=METHOD, baseline_family="existing_bc", scenario_id=job["scenario_id"],
            scenario_uid=job["scenario_uid"], dataset=job["dataset"], recording=job["recording"],
            subset=job["class"], ego=job["ego"], target=job["target"], fps=job["fps"],
            source_tracks=job["source_tracks"], raw_tracks_path=job["raw_tracks_path"],
            metadata_window_frames=[job["min_frame"], job["max_frame"]],
            target_gt_support_frames=[int(target.frame.min()), int(target.frame.max())],
            geometry_variant_id="existing_bc_boundary_helpers_no_interior_cp",
            default_end_speed_kmh=end_speed, default_endpoint_end_speed_kmh=end_speed,
            endpoint_speed_frame=int(end.frame),
            speed_source="hypot(raw.xVelocity, raw.yVelocity) at last recorded target frame within metadata window",
            source_end_speed_raw=original[SPEED],
            fixed_offset_raw=original["Agent1_Offset"],
            fixed_duration_raw=original["Agent1_1_SA_DynamicDuration"],
            fixed_start_speed_raw=original["Agent1_Speed"],
            fixed_parameter_declarations={k: v for k, v in original.items() if k != SPEED},
            source_xosc=job["source_xosc"], source_sha256=job["source_sha256"],
            declared_end_speed_is_measured_terminal_speed=False,
            preserved_duration_note="Existing SA duration retained, including metadata-minus-one-frame and truncated target support differences",
            is_nominal_default=True, no_stop_at_goal_added=True, no_endpoint_constraints=True,
            catalogue_entities=True, original_adapter_stop_margin_s=1.0)
        applied = {"endpoint_end_speed_kmh": end_speed}
        report.update(context=context, parameters_requested=applied, parameters_applied=applied.copy(),
                      load_context_s=time.perf_counter() - t0)
        H.atomic_json(wd / "context.json", context)
        H.atomic_json(wd / "parameters.json", dict(method=METHOD, requested=applied,
            fixed_declarations=context["fixed_parameter_declarations"], original_end_speed_raw=original[SPEED]))
        shutil.copyfile(job["source_xosc"], source_copy)
        assert H.sha256(source_copy) == job["source_sha256"]
        t_prepare = time.perf_counter()
        funcs = adapter_functions(raw, job)
        funcs["to_esmini_replay"](source_copy, job["dataset"], job["ego"], job["target"],
            job["min_frame"], job["max_frame"], executable, {SPEED: end_speed}, agent_replay=False)
        report["checks"] = verify(source_copy, executable, end_speed)
        report["prepare_s"] = time.perf_counter() - t_prepare
        cmd = ["nice", "-n", "10", str(binary), "--osc", str(executable), "--headless",
               "--csv_logger", str(csv_path), "--fixed_timestep", str(1 / job["fps"]),
               "--logfile_path", str(wd / "esmini.log"), "--disable_stdout"]
        report.update(command=cmd, subprocess_cwd=str(wd), status="executing")
        H.atomic_json(sample_path, report)
        t_sim = time.perf_counter()
        with (wd / "stdout.txt").open("wb") as stdout, (wd / "stderr.txt").open("wb") as stderr:
            result = subprocess.run(cmd, cwd=wd, stdout=stdout, stderr=stderr, timeout=60)
        report.update(returncode=result.returncode, simulate_s=time.perf_counter() - t_sim)
        if result.returncode != 0:
            raise RuntimeError(f"esmini returned {result.returncode}; no architecture fallback")
        t_parse = time.perf_counter()
        tidy = H.parse_tidy(csv_path, raw, job["job_id"], context)
        tidy.to_parquet(wd / "trajectory.parquet", index=False)
        tidy.to_csv(wd / "trajectory.csv", index=False, float_format="%.12g")
        report.update(status="completed", parse_export_s=time.perf_counter() - t_parse,
                      trajectory_rows=len(tidy), trajectory_path=str(wd / "trajectory.parquet"),
                      observed_time_range_s=[float(tidy.time_s.min()), float(tidy.time_s.max())])
    except subprocess.TimeoutExpired:
        report.update(status="timeout", timeout_s=60, simulate_s=time.perf_counter() - t_sim)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["total_s"] = time.perf_counter() - t0
    report["artifacts"] = [dict(path=str(p), sha256=H.sha256(p), size_bytes=p.stat().st_size)
                           for p in sorted(wd.iterdir()) if p.is_file() and p.name != "sample.json"]
    assert H.sha256(job["source_xosc"]) == job["source_sha256"], "Source XML changed"
    H.atomic_json(sample_path, report)
    print(f"[sakura] {job['job_id']}: {report['status']} ({report['total_s']:.2f}s)", flush=True)
    return report


def main():
    t0 = time.perf_counter()
    population = pd.read_csv(POP)
    cases = pd.read_csv(CASES)
    cases = cases[cases["mode"].eq("fullfit")].set_index(["subset", "scenario_uid"])
    eligible = population[population.status.eq("unique_existing_bc_source")]
    assert len(population) == 513 and len(eligible) == 192
    assert eligible.source_structure_ok.eq(True).all()
    jobs = []
    for r in eligible.itertuples(index=False):
        source = cases.loc[(r.subset, r.scenario_uid)]
        jobs.append(dict(job_id=f"sakura_bc_{r.subset}_{r.scenario_id}", scenario_id=r.scenario_id,
            scenario_uid=r.scenario_uid, **{"class": r.subset}, dataset="HetroD", recording="00",
            ego=int(r.ego), target=int(r.actor), min_frame=int(r.metadata_min_frame), max_frame=int(r.metadata_max_frame),
            fps=30.0, source_xosc=r.source_path, source_sha256=r.source_sha256,
            expected_end_speed_kmh=float(r.gt_end_speed_kmh), source_tracks=source.source_tracks,
            raw_tracks_path=str(H.RAW), map_path=str(H.MAP)))
    paths = {Path(__file__).resolve(), HELPER, POP, CASES, H.EX_SOURCE, H.RAW, H.MAP, H.BINARY}
    paths.update(Path(j["source_xosc"]) for j in jobs)
    paths.update(Path(j["source_tracks"]) for j in jobs)
    for cat in ("Vehicles/VehicleCatalog.xosc", "Pedestrians/PedestrianCatalog.xosc", "Controllers/ControllerCatalog.xosc", "Environments/EnvironmentCatalog.xosc"):
        paths.add(Path("/opt/Catalogs") / cat)
    sources = [dict(path=str(p), sha256=H.sha256(p), size_bytes=p.stat().st_size) for p in sorted(paths)]
    protocol = dict(method=METHOD, source_scope="192 exact-window existing bc templates from full 513-case inventory",
        population_size=513, selected_n=192, missing_or_window_mismatch_n=321,
        selection_by_quality=False, jobs=jobs, sources=sources,
        endpoint_speed_default="hypot(vx,vy)*3.6 at actual target GT endpoint",
        only_overridden_parameter=SPEED, fixed_duration_offset_initial_speed=True,
        geometry="existing four boundary/helper CP NURBS; no interior weighted points",
        adapter="existing EX.to_esmini_replay; CatalogReference preserved; agent_replay=False",
        worker_count=1, fixed_timestep_s=1 / 30, subprocess_timeout_s=60, nice=10,
        new_clock_or_duration_correction=False, new_stop_at_goal=False, background_vehicles_added=False,
        all_artifacts_retained=True, scoring_performed=False)
    import hashlib
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    batch = PROJECT / "runs/sakura_defaults" / run_id
    batch.mkdir(parents=True, exist_ok=True)
    binary = PROJECT / "bin/esmini"
    assert binary.resolve() == H.BINARY.resolve()
    assert not (binary.parent / "config.yml").exists(), "Unexpected local esmini config"
    H.atomic_json(batch / "protocol.json", protocol)
    if not (batch / "runner_snapshot.py").exists():
        shutil.copyfile(__file__, batch / "runner_snapshot.py")
    reports = []
    for job in jobs:
        if shutil.disk_usage(batch).free < H.MIN_FREE:
            raise RuntimeError("Low disk space; all prior artifacts retained")
        reports.append(run_one(job, batch, run_id, binary))
        H.atomic_json(batch / "status.json", dict(run_id=run_id, batch_dir=str(batch), completed_so_far=len(reports),
            total_jobs=len(jobs), status_counts=pd.Series([r["status"] for r in reports]).value_counts().to_dict()))
    indexed = {(r["job"]["class"], r["job"]["scenario_uid"]): r for r in reports}
    rows = []
    for source in population.to_dict("records"):
        row = {k: source[k] for k in ("subset", "case_index", "scenario_id", "scenario_uid", "ego", "actor")}
        row.update(source_status=source["status"], method=METHOD, run_id=run_id, execution_status="not_executed_missing_or_mismatched_source")
        report = indexed.get((source["subset"], source["scenario_uid"]))
        if report:
            row.update(execution_status=report["status"], error=report.get("error", ""),
                sample_id=report["sample_id"], sample_json=str(Path(report["sample_dir"]) / "sample.json"),
                trajectory_path=report.get("trajectory_path", ""),
                endpoint_end_speed_kmh=report["parameters_applied"].get("endpoint_end_speed_kmh"),
                fixed_duration_s=source.get("fixedD_s"), source_path=source.get("source_path"), source_sha256=source.get("source_sha256"))
            row.update({k: report.get(k) for k in ("load_context_s", "prepare_s", "simulate_s", "parse_export_s", "total_s")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(PROJECT / "results/sakura_default_cases.csv", index=False)
    H.atomic_json(PROJECT / "results/sakura_default_sample_list.json", [str(Path(r["sample_dir"]) / "sample.json") for r in reports])
    for source in sources:
        assert H.sha256(source["path"]) == source["sha256"], f"Input changed: {source['path']}"
    status = dict(run_id=run_id, batch_dir=str(batch), population_n=513, executed_n=len(reports),
        source_missing_or_mismatch_n=321, status_counts=pd.Series([r["status"] for r in reports]).value_counts().to_dict(),
        elapsed_s=time.perf_counter() - t0, all_source_hashes_unchanged=True, artifacts_retained=True,
        source_manifest=str(batch / "protocol.json"), scoring_performed=False)
    H.atomic_json(batch / "status.json", status)
    H.atomic_json(PROJECT / "results/sakura_default_manifest.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
