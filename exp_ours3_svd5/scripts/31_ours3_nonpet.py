#!/usr/bin/env python3
"""Score saved ours3 traces with the shared, existing non-PET definitions.

Use --batch-dir to discover sample.json files, or --sample-list for a JSON/TXT
list of sample.json paths. This reads saved traces only; it never simulates,
shifts time, changes a trajectory, computes PET, or scores background traffic.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent

SINGLETON_SUBSETS = {
    "uturn_859_881": ("859_881", "exp_cross_coverage/data/uturn_859_881/real_tracks.parquet"),
    "uturn_859_881_start15371": ("859_881", "exp_cross_coverage/data/uturn_859_881_start15371/real_tracks.parquet"),
}
COMMON_PATH = PROJECT / "scripts/30_interaction_metrics.py"
spec = importlib.util.spec_from_file_location("ours3_shared_nonpet", COMMON_PATH)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
PARAMETERS = ("theta1_deg", "theta2_deg", "interaction_end_speed_kmh")


def sample_paths(args):
    paths = []
    for directory in args.batch_dir:
        paths.extend(sorted(directory.resolve().rglob("sample.json")))
    for path in args.sample_list:
        content = path.read_text()
        items = json.loads(content) if path.suffix.lower() == ".json" else content.splitlines()
        if isinstance(items, dict):
            items = items["samples"]
        for item in items:
            value = item["sample_json"] if isinstance(item, dict) else item
            if not str(value).strip():
                continue
            p = Path(value)
            paths.append((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Supply a nonempty list with no duplicate sample.json paths")
    return paths


def resolve_context(sample, population):
    """Resolve recorded input identity, keeping special geometry separate."""
    job = sample.get("job", {})
    context = sample.get("context", {})
    source_context = context.get("source_context", job.get("source_context", {}))
    if not isinstance(source_context, dict):
        raise ValueError("source_context must be an object when supplied")
    c = {**source_context, **job, **context}
    c["scenario_id"] = str(c["scenario_id"])
    c["subset"] = c.get("subset") or c.get("class") or source_context.get("subset")
    c["target"] = c.get("target", c.get("actor"))
    c["metadata_window_frames"] = c.get("metadata_window_frames", [c.get("min_frame"), c.get("max_frame")])
    lo, hi = c["metadata_window_frames"]
    if lo is None or hi is None or hi < lo:
        raise ValueError("Missing or invalid original metadata window")
    c["geometry_variant_id"] = c.get("geometry_variant_id", c.get("geometry_variant", "unspecified"))
    special = c["geometry_variant_id"] == "special_pet_cp3d10_exact_rotation"
    singleton = SINGLETON_SUBSETS.get(c.get("subset"))
    if special:
        if c["scenario_id"] != "39_180":
            raise ValueError("Unknown special scenario; no source substitution")
        c["source_tracks"] = str(ROOT / "exp_cross_coverage/data/special_39_180/real_tracks.parquet")
        c["comparison_population"] = "special_39_180_only"
        c["subset"] = "special_39_180"
    elif singleton is not None:
        # a singleton rare scenario that belongs to no fitted class: it is its own population,
        # exactly as special_39_180 is, and its recorded tracks live in their own data dir
        if c["scenario_id"] != singleton[0]:
            raise ValueError(f"Singleton subset {c['subset']} expects scenario {singleton[0]}, got {c['scenario_id']}")
        c["source_tracks"] = str(ROOT / singleton[1])
        c["comparison_population"] = f"{c['subset']}_only"
    else:
        c["comparison_population"] = "regular_class_population"
        if not c.get("source_tracks"):
            matching = population[population.scenario_id.astype(str).eq(c["scenario_id"])
                                  & population.min_frame.eq(lo) & population.max_frame.eq(hi)]
            if c.get("subset"):
                matching = matching[matching.subset.eq(c["subset"])]
            if len(matching) != 1:
                raise ValueError(f"Cannot uniquely resolve population source: {len(matching)} rows")
            c["source_tracks"] = str(matching.iloc[0].source_tracks)
            c["subset"] = str(matching.iloc[0].subset)
    c["scenario_uid"] = c.get("scenario_uid") or f"{c['dataset']}/{str(c.get('recording', '00')).zfill(2)}/{c['scenario_id']}/{lo}-{hi}"
    c["fps"] = float(c.get("fps", M.SIMC.paths.dataset_of(c["dataset"])["fps"]))
    expected_fps = float(M.SIMC.paths.dataset_of(c["dataset"])["fps"])
    if c["fps"] != expected_fps:
        raise ValueError(f"Sample FPS {c['fps']} differs from dataset FPS {expected_fps}; no conversion")
    c["source_tracks"] = str(Path(c["source_tracks"]).resolve())
    return c


def replay_diagnostic(recorded, simulated):
    """Same-clock esmini ego versus recorded ego; no best-lag fitting or shift."""
    mask = (simulated.frame >= recorded.frame[0]) & (simulated.frame <= recorded.frame[-1])
    frames = simulated.frame[mask]
    if not len(frames):
        raise ValueError("No same-clock ego samples for replay diagnostic")
    pos = np.hypot(simulated.x[mask] - np.interp(frames, recorded.frame, recorded.x),
                   simulated.y[mask] - np.interp(frames, recorded.frame, recorded.y))
    heading = np.rad2deg(np.unwrap(np.deg2rad(recorded.heading)))
    error_heading = np.abs((simulated.heading[mask] - np.interp(frames, recorded.frame, heading) + 180) % 360 - 180)
    error_speed = np.abs(simulated.speed[mask] - np.interp(frames, recorded.frame, recorded.speed))
    result = {"ego_replay_sameclock_samples": len(frames), "ego_replay_time_shift_applied_s": 0.0}
    for name, values in (("position_m", pos), ("heading_deg", error_heading), ("speed_mps", error_speed)):
        for stat, value in (("mean", np.mean(values)), ("p95", np.percentile(values, 95)), ("max", np.max(values))):
            result[f"ego_replay_abs_{name}_{stat}"] = float(value)
    return result


def signed_difference(value, reference, key, fps):
    if not np.isfinite([value, reference]).all():
        return np.nan
    delta = value - reference
    if key.endswith("heading"):
        delta = (delta + 180) % 360 - 180
    elif M.UNITS[key] == "frame":
        delta /= fps
    return float(delta)


def oat_table(paired):
    """Within-scene changes from the unique nominal parameter configuration."""
    out = []
    keys = ["run_id", "scenario_uid", "source_tracks", "geometry_variant_id"]
    for _, group in paired.groupby(keys, dropna=False, sort=False):
        nominal = group[group.is_nominal_default.eq(True)]
        for record in group.to_dict("records"):
            row = {k: record.get(k) for k in ["sample_json", "sample_id", *keys, "subset", "score_status"]}
            if record.get("mode") != "ours3":
                row.update(oat_status="not_applicable_no_theta_OAT")
                out.append(row)
                continue
            if len(nominal) != 1:
                row.update(oat_status="no_unique_nominal_reference", nominal_reference_count=len(nominal))
                out.append(row)
                continue
            reference = nominal.iloc[0]
            row.update(oat_status="ok" if record["score_status"] == "ok" and reference.score_status == "ok" else "failed_or_partial_score",
                       nominal_sample_id=reference.sample_id)
            changed = []
            for parameter in PARAMETERS:
                delta = float(record.get(parameter, np.nan) - reference.get(parameter, np.nan))
                row[f"delta_{parameter}"] = delta
                if np.isfinite(delta) and abs(delta) > 1e-9:
                    changed.append(parameter)
            row.update(changed_parameter_count=len(changed), changed_parameter=";".join(changed),
                       single_parameter_oat=bool(len(changed) <= 1))
            for key, unit in M.UNITS.items():
                name = f"change_{key}{'_s' if unit == 'frame' else ''}"
                row[name] = signed_difference(record[f"generated_{key}"], reference[f"generated_{key}"], key, record["fps"])
            out.append(row)
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, action="append", default=[])
    parser.add_argument("--sample-list", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--output-prefix", default="ours3_nonpet")
    parser.add_argument("--method", help="Explicit method label; otherwise sample/context method or ours3")
    args = parser.parse_args()
    if Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("output-prefix must be a plain file prefix")
    paths = sample_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, args.output_dir / f"{args.output_prefix}_scorer_snapshot.py")
    population_path = PROJECT / "results/ours3_population.csv"
    population = pd.read_csv(population_path) if population_path.exists() else pd.DataFrame()
    source_files = {Path(__file__), COMMON_PATH, PROJECT / "DECISIONS.md", Path(M.SIMC.paths.__file__), *args.sample_list}
    if population_path.exists():
        source_files.add(population_path)
    for p in ("similarity/interaction_sim.py", "similarity/core.py", "sweep/core.py", "interaction.py"):
        source_files.add(ROOT / "hetero-param/hetero_param" / p)
    source_groups, replay_cache, rows = {}, {}, []
    start = time.monotonic()
    with M.prohibit_pet() as guard:
        for i, path in enumerate(paths):
            row = dict(sample_json=str(path), sample_id=path.parent.name, run_id="unknown", scenario_uid="unknown",
                       source_tracks="unknown", geometry_variant_id="unknown", score_status="pending", score_detail="",
                       subset="unknown", mode="ours3", clock_aligned=False, pet_status="pending_review_not_computed",
                       actor_gt_kinematics="derived_native_recorded_xy",
                       actor_generated_kinematics="esmini_recorded_heading_speed_accel_derived_from_speed",
                       ego_kinematics="recorded_GT_heading_speed_accel_derived_from_speed",
                       criticality_kinematics="both_xy_derived_on_existing_shared_frame_grid",
                       anchor_definition="existing_width_weighted_all_path_nearest_pair_no_PET",
                       comparison_scope="ours3_saved_execution_nonpet_interaction", is_nominal_default=False)
            real, generated = dict.fromkeys(M.KEYS, np.nan), dict.fromkeys(M.KEYS, np.nan)
            fps = np.nan
            try:
                source_files.add(path)
                sample = json.loads(path.read_text())
                row.update(sample_id=sample.get("sample_id", path.parent.name), run_id=sample.get("run_id", "unspecified"),
                           upstream_status=sample.get("status", "unknown"))
                c = resolve_context(sample, population)
                method = args.method or sample.get("method") or c.get("method") or "ours3"
                row.update(method=method, mode=method, comparison_scope=f"{method}_saved_execution_nonpet_interaction")
                fps = c["fps"]
                lo, hi = c["metadata_window_frames"]
                row.update({k: c[k] for k in ("scenario_uid", "scenario_id", "subset", "dataset", "source_tracks", "geometry_variant_id", "comparison_population", "fps")})
                row.update(ego=int(c["ego"]), actor=int(c["target"]), metadata_min_frame=lo,
                           metadata_max_frame=hi, metadata_horizon_s=(hi - lo) / fps,
                           commanded_end_speed_is_measured_arrival_speed=False)
                parameters = sample.get("parameters_applied", sample.get("parameters_requested", {}))
                defaults = [c.get("theta1_default_deg"), c.get("theta2_default_deg"), c.get("default_end_speed_kmh")]
                for k, default in zip(PARAMETERS, defaults):
                    row[k] = float(parameters.get(k, np.nan))
                    row[f"default_{k}"] = float(default) if default is not None else np.nan
                row["is_nominal_default"] = all(np.isfinite(row[k]) and np.isfinite(row[f"default_{k}"])
                                                   and abs(row[k] - row[f"default_{k}"]) < 1e-9 for k in PARAMETERS)
                if method != "ours3":
                    row["is_nominal_default"] = bool(sample.get("is_nominal_default", c.get("is_nominal_default", False)))
                row["endpoint_end_speed_kmh"] = sample.get("endpoint_end_speed_kmh", c.get("endpoint_end_speed_kmh", parameters.get("endpoint_end_speed_kmh", np.nan)))
                source = c["source_tracks"]
                source_files.add(Path(source))
                if source not in source_groups:
                    tracks = pd.read_parquet(source)
                    source_groups[source] = {(str(sid), role): g.sort_values("frame") for (sid, role), g in tracks.groupby(["scenario_id", "role"])}
                cache_key = (source, c["scenario_uid"])
                if cache_key not in replay_cache:
                    a = source_groups[source][(c["scenario_id"], "actor")]
                    e = source_groups[source][(c["scenario_id"], "ego")]
                    if not a.track_id.eq(row["actor"]).all() or not e.track_id.eq(row["ego"]).all():
                        raise ValueError("Recorded actor identities differ from sample context")
                    actor = M.traj_from_tidy(a, fps=fps, kinematics="derived")
                    ego = M.traj_from_tidy(e, fps=fps, kinematics="recorded")
                    desc = M.nonpet_descriptors(actor, ego)
                    replay_cache[cache_key] = (actor, ego, desc, dict(scenario_uid=c["scenario_uid"],
                        scenario_id=c["scenario_id"], source_tracks=source, fps=fps, subset=c["subset"],
                        actor_gt_kinematics=row["actor_gt_kinematics"], ego_kinematics=row["ego_kinematics"], **desc))
                actor, ego, real, _ = replay_cache[cache_key]
                offset = lo - actor.frame[0]
                row.update(actual_actor_min_frame=float(actor.frame[0]), actual_actor_max_frame=float(actor.frame[-1]),
                           actual_ego_min_frame=float(ego.frame[0]), actual_ego_max_frame=float(ego.frame[-1]),
                           real_actor_samples=len(actor.frame), real_actor_horizon_s=(actor.frame[-1] - actor.frame[0]) / fps,
                           real_shared_span_s=M.shared_span(actor, ego), clock_offset_frames=float(offset),
                           clock_offset_s=float(offset / fps), clock_aligned=bool(offset == 0),
                           method_claim_eligible_clock=bool(offset == 0),
                           claim_note="clock aligned; no method advantage claim without matched comparator" if offset == 0 else "起始時鐘疑點，未用於方法優劣主張")
                if sample.get("status") != "completed":
                    raise ValueError(f"Upstream status={sample.get('status')}: {sample.get('error', '')}")
                trajectory_path = Path(sample.get("trajectory_path", path.parent / "trajectory.parquet")).resolve()
                source_files.add(trajectory_path)
                for item in sample.get("artifacts", []):
                    if Path(item["path"]).resolve() == trajectory_path and M.sha256(trajectory_path) != item["sha256"]:
                        raise ValueError("Trajectory hash differs from generation manifest")
                tidy = pd.read_parquet(trajectory_path).rename(columns={"speed_mps": "speed"})
                if not {"ego", "target"}.issubset(set(tidy.role)):
                    raise ValueError("Expected explicit ego/target roles")
                # Same half-frame inclusion convention as the existing exporter.
                # This selects samples only; it never rounds, rebases, or shifts frames.
                in_window = tidy.frame.between(lo - .5, hi + .5)
                target_all = tidy[tidy.role.eq("target")]
                a = tidy[in_window & tidy.role.eq("target")]
                e = tidy[in_window & tidy.role.eq("ego")]
                if not a.actor_id.astype(str).eq(str(row["actor"])).all() or not e.actor_id.astype(str).eq(str(row["ego"])).all():
                    raise ValueError("Generated actor identities differ from sample context")
                predicted = M.traj_from_tidy(a, fps=fps, kinematics="recorded")
                simulated_ego = M.traj_from_tidy(e, fps=fps, kinematics="recorded")
                generated = M.nonpet_descriptors(predicted, ego)
                simulated_pair = M.nonpet_descriptors(predicted, simulated_ego)
                row.update(replay_diagnostic(ego, simulated_ego))
                row.update({f"diagnostic_simego_{k}": simulated_pair[k] for k in M.KEYS})
                row.update({f"diagnostic_simego_change_{k}{'_s' if unit == 'frame' else ''}":
                            signed_difference(simulated_pair[k], generated[k], k, fps) for k, unit in M.UNITS.items()})
                row.update(trajectory_path=str(trajectory_path), generated_samples=len(predicted.frame),
                           generated_all_samples=len(target_all), generated_removed_margin_samples=len(target_all) - len(a),
                           generated_min_frame=float(predicted.frame[0]), generated_max_frame=float(predicted.frame[-1]),
                           generated_duration_s=(predicted.frame[-1] - predicted.frame[0]) / fps,
                           generated_all_min_frame=float(target_all.frame.min()), generated_all_max_frame=float(target_all.frame.max()),
                           generated_shared_span_s=M.shared_span(predicted, ego),
                           shared_span_difference_s=M.shared_span(predicted, ego) - M.shared_span(actor, ego),
                           generated_start_minus_gt_start_s=float((predicted.frame[0] - actor.frame[0]) / fps),
                           generated_end_minus_gt_end_s=float((predicted.frame[-1] - actor.frame[-1]) / fps),
                           generated_before_gt_support_samples=int((a.frame < actor.frame[0] - .5).sum()),
                           generated_after_gt_support_samples=int((a.frame > actor.frame[-1] + .5).sum()),
                           generated_frame_integer_rounding_max_abs=float(np.max(np.abs(predicted.frame - np.rint(predicted.frame)))),
                           frame_selection="metadata_lo_minus_0.5_to_hi_plus_0.5; exporter convention; frames unchanged",
                           generated_frame_origin="saved_esmini_absolute_clock_no_shift")
                errs = " | ".join(value for value in [real.get("estimator_errors", ""), generated.get("estimator_errors", "")] if value)
                row.update(score_status="partial_estimator_error" if errs else "ok", score_detail=errs,
                           simulated_ego_diagnostic_errors=simulated_pair.get("estimator_errors", ""))
            except Exception as exc:
                row.update(score_status="failed", score_detail=f"{type(exc).__name__}: {exc}")
            row.update({f"real_{key}": real.get(key, np.nan) for key in M.KEYS})
            row.update({f"generated_{key}": generated.get(key, np.nan) for key in M.KEYS})
            row.update(M.pair_differences(real, generated, fps))
            row.update(real_pet=np.nan, generated_pet=np.nan)
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"ours3 non-PET {i + 1}/{len(paths)} ({time.monotonic() - start:.1f}s)", flush=True)
    assert guard["blocked_pet_calls"] == 0, guard
    paired = pd.DataFrame(rows)
    assert len(paired) == len(paths)
    summary = M.generate_summary(paired)
    oat = oat_table(paired)
    outputs = {}
    for label, table in (("paired", paired), ("summary", summary), ("oat", oat),
                         ("replay", pd.DataFrame([value[3] for value in replay_cache.values()]))):
        destination = args.output_dir / f"{args.output_prefix}_{label}.csv"
        table.to_csv(destination, index=False)
        outputs[label] = dict(path=str(destination), rows=len(table), sha256=M.sha256(destination))
    source_records = [dict(path=str(p.resolve()), sha256=M.sha256(p)) for p in sorted(source_files) if p.exists()]
    method_names = sorted(paired["mode"].unique())
    audit = dict(scope=f"Saved {'/'.join(method_names)} execution, existing non-PET interaction estimators; no background metrics",
                 methods=method_names,
                 n_input_samples=len(paths), n_output_cases=len(paired), n_replay_cases=len(replay_cache),
                 score_status_counts=paired.score_status.value_counts().to_dict(),
                 clock_aligned_n=int(paired.clock_aligned.sum()), clock_misaligned_n=int((~paired.clock_aligned).sum()),
                 pet_status="pending_review_not_computed", **guard, metrics_units=M.UNITS,
                 main_ego_source="recorded GT, identical to SVD evaluator; simulated ego only diagnostic",
                 source_files=source_records, missing_source_files=[str(p.resolve()) for p in sorted(source_files) if not p.exists()],
                 sample_json_paths=[str(p) for p in paths], outputs=outputs,
                 elapsed_s=time.monotonic() - start)
    (args.output_dir / f"{args.output_prefix}_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    report = [f"# {'/'.join(method_names)} 非PET互動評分", "", f"輸入與輸出各 {len(paired)} 筆；GT場景 {len(replay_cache)} 筆。評分狀態：{audit['score_status_counts']}。",
              "PET保留pending／NaN；執行guard確認0次PET呼叫。本表未計算背景碰撞。",
              "主值以生成target搭配同場recorded GT ego；實際esmini ego及互動差異另列diagnostic欄。沒有移動時間原點。",
              "GT target由原生GT xy導出heading/speed；生成target採esmini記錄heading/speed，兩者arrival acceleration均沿用既有各自速度微分。criticality另依既有函式由共同時間網格的兩車xy導出。",
              "arrival speed是既有conflict-point最近樣本的實測速度，不是commanded EndSpeed；沒有將末端速度相似度列入評分。",
              "生成資料僅去除metadata視窗外的simulator margin，沿用exporter半frame納入容差，未改寫frame；GT保留既有來源支持窗。所有horizon、共享時間及超出target GT窗的樣本數另列。",
              f"起始時鐘一致 {audit['clock_aligned_n']} 筆；不一致 {audit['clock_misaligned_n']} 筆另標，未用於方法優劣主張。",
              "OAT表只表示同一場景、同一幾何來源，相對唯一預設參數的descriptor變化；執行成功不代表有效性或方法優勢。special_39_180與常規cutinl class population分開。",
              "", "| sample | population | status | GT / gen shared s | min dist m | TTC s | angle deg | arrival speed m/s | ego replay max position error m |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in paired.head(30).to_dict("records"):
        f = lambda key: f"{row.get(key, np.nan):.5g}"
        report.append(f"| {row['sample_id']} | {row.get('comparison_population', 'unknown')} | {row['score_status']} | {f('real_shared_span_s')} / {f('generated_shared_span_s')} | {f('generated_min_dist')} | {f('generated_min_ttc')} | {f('generated_conflict_angle')} | {f('generated_agent_arr_speed')} | {f('ego_replay_abs_position_m_max')} |")
    if len(paired) > 30:
        report.append("\n上表僅展示前30筆；完整paired保留全部案例，summary按類別列出有限／非有限分母。")
    (args.output_dir / f"{args.output_prefix}_SUMMARY.md").write_text("\n".join(report) + "\n")
    print(json.dumps({k: audit[k] for k in ("n_input_samples", "n_output_cases", "n_replay_cases", "score_status_counts", "clock_aligned_n", "clock_misaligned_n", "blocked_pet_calls", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
