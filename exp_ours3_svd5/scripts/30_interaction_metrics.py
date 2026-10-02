#!/usr/bin/env python3
"""Existing non-PET interaction estimators, isolated from all PET call paths.

Main reads saved SVD d5 fullfit/LOGO reconstructions; it does not fit, sample,
render, repair endpoints, shift clocks, or compute PET. Shared functions accept
Traj objects or tidy traces for subsequent method outputs. Every input case is
retained, including failures and non-finite descriptors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / "hetero-param"))
from hetero_param.similarity import core as SIMC, interaction_sim as ISIM
from hetero_param.sweep import core as SWC

UNITS = {
    "min_dist": "m", "min_dist_frame": "frame", "min_ttc": "s", "ttc_frame": "frame",
    "conflict_x": "m", "conflict_y": "m", "min_path_dist": "m",
    "conflict_angle": "deg", "closing_speed": "m/s", "drac": "m/s^2",
    **{f"{role}_arr_{key}": unit for role in ("agent", "ego")
       for key, unit in (("frame", "frame"), ("speed", "m/s"), ("accel", "m/s^2"), ("heading", "deg"))},
}
KEYS = tuple(UNITS)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def traj_from_tidy(frame, *, fps, kinematics="recorded", meta=None):
    """Build one actor's Traj; caller selects actor and explicit absolute frame.

    recorded: retain supplied heading_deg and speed (esmini state or recorded
    replay); accel is d(speed)/dt, matching sr/05_descriptors._traj_from_df.
    derived: use SIMC.derive_kinematics from xy and the supplied dataset clock.
    No re-zero, clipping, extrapolation, or anchor change is performed.
    """
    g = frame.sort_values("frame")
    f, x, y = (g[k].to_numpy(float) for k in ("frame", "x", "y"))
    if len(g) < 3 or not np.isfinite(np.column_stack([f, x, y])).all() or np.any(np.diff(f) <= 0):
        raise ValueError("At least three finite, strictly time-ordered actor states required")
    length, width = float(g.length.iloc[0]), float(g.width.iloc[0])
    if not np.isfinite([length, width, fps]).all() or min(length, width, fps) <= 0:
        raise ValueError("Recorded dimensions and dataset FPS must be positive")
    if kinematics == "recorded":
        heading, speed = g.heading_deg.to_numpy(float), g.speed.to_numpy(float)
        accel = np.gradient(speed, f / fps)
    elif kinematics == "derived":
        heading, speed, accel = SIMC.derive_kinematics(f, x, y, fps)
    else:
        raise ValueError("kinematics must be recorded or derived")
    if not np.isfinite(np.column_stack([heading, speed, accel])).all():
        raise ValueError("Non-finite input kinematics")
    return SIMC.Traj(frame=f, x=x, y=y, heading=heading, speed=speed, accel=accel,
                     fps=float(fps), length=length, width=width,
                     meta={**(meta or {}), "kinematics": kinematics})


def nonpet_descriptors(agent, ego):
    """Call only existing PET-free primitives, preserving existing definitions.

    ISIM.conflict_point is the width-weighted all-path nearest-point pair;
    ISIM.arrival_state reads the closest sample to that point, not a PET time.
    SWC._criticality calls IX._pair_interaction, which derives both actors'
    heading/speed on its shared frame grid. No high-level descriptors bundle.
    """
    if agent.fps != ego.fps:
        raise ValueError("Agent and ego must use the same dataset frame clock")
    result = dict.fromkeys(KEYS, np.nan)
    errors = []
    for name, fn in (("min_distance", ISIM.min_distance), ("ttc", ISIM.ttc),
                     ("criticality", SWC._criticality)):
        try:
            result.update(fn(agent, ego))
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        cxy, _i, _j, distance = ISIM.conflict_point(agent, ego)
        result.update(conflict_x=float(cxy[0]), conflict_y=float(cxy[1]), min_path_dist=float(distance))
        for role, trajectory in (("agent", agent), ("ego", ego)):
            result.update({f"{role}_{k}": v for k, v in ISIM.arrival_state(trajectory, cxy).items()})
    except Exception as exc:
        errors.append(f"conflict_arrival: {type(exc).__name__}: {exc}")
    result.update(pet=np.nan, pet_abs=np.nan, pet_status="pending_review_not_computed",
                  estimator_errors=" | ".join(errors))
    return result


def pair_differences(real, generated, fps):
    """Absolute paired differences; frame differences in seconds, headings wrapped.

    A pair containing NaN or infinity retains NaN difference; summaries record
    the non-finite population rather than treating inf-inf as zero.
    """
    result = {}
    for key, unit in UNITS.items():
        a, b = float(real.get(key, np.nan)), float(generated.get(key, np.nan))
        delta = np.nan
        if np.isfinite([a, b]).all():
            delta = abs(a - b)
            if key.endswith("heading"):
                delta = float(ISIM._wrap180(a - b))
            elif unit == "frame":
                delta /= fps
        result[f"absdiff_{key}{'_s' if unit == 'frame' else ''}"] = delta
    result["absdiff_pet"] = np.nan
    return result


@contextmanager
def prohibit_pet():
    """Process-local guard only, restored on exit; never changes source files."""
    count = {"blocked_pet_calls": 0}
    def forbidden(*_args, **_kwargs):
        count["blocked_pet_calls"] += 1
        raise RuntimeError("PET is pending review and must not execute")
    old_pet, old_calculate = ISIM.pet, ISIM.pet_utils.calculate_pet
    ISIM.pet, ISIM.pet_utils.calculate_pet = forbidden, forbidden
    try:
        yield count
    finally:
        ISIM.pet, ISIM.pet_utils.calculate_pet = old_pet, old_calculate


def shared_span(agent, ego):
    return max(0.0, min(agent.frame[-1], ego.frame[-1]) - max(agent.frame[0], ego.frame[0])) / agent.fps


def generate_summary(paired):
    rows = []
    for (subset, mode), group in paired.groupby(["subset", "mode"], sort=False):
        views = {"all_cases_diagnostic": group,
                 "clock_aligned_only": group[group.clock_aligned.eq(True)],
                 "clock_misaligned_no_method_claim": group[group.clock_aligned.eq(False)]}
        for scope, g in views.items():
            for key, unit in UNITS.items():
                real, pred = g[f"real_{key}"].to_numpy(float), g[f"generated_{key}"].to_numpy(float)
                difference = g[f"absdiff_{key}{'_s' if unit == 'frame' else ''}"].to_numpy(float)
                finite = difference[np.isfinite(difference)]
                rows.append({"subset": subset, "mode": mode, "scope": scope, "metric": key,
                             "difference_unit": "s" if unit == "frame" else unit,
                             "n_cases": len(g), "n_scored": int(g.score_status.eq("ok").sum()),
                             "n_failed_or_partial": int((~g.score_status.eq("ok")).sum()),
                             "real_finite_n": int(np.isfinite(real).sum()),
                             "generated_finite_n": int(np.isfinite(pred).sum()),
                             "real_inf_n": int(np.isinf(real).sum()), "generated_inf_n": int(np.isinf(pred).sum()),
                             "paired_finite_n": len(finite), "paired_nonfinite_n": len(g) - len(finite),
                             "mean_abs_difference": float(np.mean(finite)) if len(finite) else np.nan,
                             "median_abs_difference": float(np.median(finite)) if len(finite) else np.nan,
                             "p95_abs_difference": float(np.percentile(finite, 95)) if len(finite) else np.nan})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=PROJECT / "results/svd_d5_cases.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--output-prefix", default="svd_d5_nonpet")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("output-prefix must be a plain file prefix")
    shutil.copyfile(__file__, args.output_dir / f"{args.output_prefix}_scorer_snapshot.py")
    cases = pd.read_csv(args.cases, dtype={"recording": str})
    if cases.duplicated(["subset", "case_index", "mode"]).any():
        raise ValueError("Duplicate input case keys")
    sources, arrays, ground_truth, paired_rows = {}, {}, {}, []
    fps_by_dataset = {name: float(SWC.paths.dataset_of(name)["fps"]) for name in cases.dataset.unique()} if hasattr(SWC, "paths") else {
        name: float(SIMC.paths.dataset_of(name)["fps"]) for name in cases.dataset.unique()}
    start = time.monotonic()
    with prohibit_pet() as guard:
        for i, case in enumerate(cases.to_dict("records")):
            fps = fps_by_dataset[case["dataset"]]
            row = dict(case)
            row.update(score_status="pending", score_detail="", pet_status="pending_review_not_computed",
                       fps=fps, actor_gt_kinematics="derived_native_recorded_xy",
                       actor_generated_kinematics="derived_50_sample_xy",
                       ego_kinematics="recorded_heading_speed_accel_derived_from_speed",
                       criticality_kinematics="both_xy_derived_on_existing_shared_frame_grid",
                       generated_frame_origin="metadata_min_frame_legacy_decoder",
                       anchor_definition="existing_width_weighted_all_path_nearest_pair_no_PET",
                       comparison_scope="official_raw_d5_baseline_available_trajectory")
            offset = case["metadata_min_frame"] - case["actual_actor_min_frame"]
            row.update(clock_offset_frames=offset, clock_offset_s=offset / fps,
                       clock_aligned=bool(offset == 0),
                       method_claim_eligible_clock=bool(offset == 0),
                       claim_note="clock aligned; official raw d5 baseline; method advantage requires matched comparator" if offset == 0 else "起始時鐘疑點，未用於方法優劣主張")
            real, generated = dict.fromkeys(KEYS, np.nan), dict.fromkeys(KEYS, np.nan)
            try:
                source = str(Path(case["source_tracks"]).resolve())
                if source not in sources:
                    tracks = pd.read_parquet(source)
                    sources[source] = {(str(sid), role): g.sort_values("frame")
                                       for (sid, role), g in tracks.groupby(["scenario_id", "role"])}
                cache_key = (source, str(case["scenario_id"]))
                if cache_key not in ground_truth:
                    a = sources[source][(str(case["scenario_id"]), "actor")]
                    e = sources[source][(str(case["scenario_id"]), "ego")]
                    if not a.track_id.eq(int(case["actor"])).all() or not e.track_id.eq(int(case["ego"])).all():
                        raise ValueError("Actor/ego identity mismatch in source traces")
                    actor = traj_from_tidy(a, fps=fps, kinematics="derived")
                    ego = traj_from_tidy(e, fps=fps, kinematics="recorded")
                    desc = nonpet_descriptors(actor, ego)
                    ground_truth[cache_key] = (actor, ego, desc, {
                        "subset": case["subset"], "scenario_uid": case["scenario_uid"],
                        "scenario_id": case["scenario_id"], "source_tracks": source,
                        "actor": case["actor"], "ego": case["ego"], "fps": fps,
                        "actor_kinematics": "derived_native_recorded_xy", "ego_kinematics": "recorded",
                        "shared_span_s": shared_span(actor, ego), **desc})
                actor, ego, real, _gtrow = ground_truth[cache_key]
                row["real_shared_span_s"] = shared_span(actor, ego)
                if case["status"] != "ok":
                    raise ValueError(f"Upstream reconstruction status={case['status']}: {case.get('detail', '')}")
                npz_path = (PROJECT / case["reconstruction_npz"]).resolve()
                if npz_path not in arrays:
                    with np.load(npz_path, allow_pickle=False) as loaded:
                        arrays[npz_path] = {key: loaded[key] for key in loaded.files}
                saved = arrays[npz_path]
                idx = int(case["case_index"])
                if str(saved["scenario_uid"][idx]) != case["scenario_uid"]:
                    raise ValueError("Reconstruction array row identity mismatch")
                vector = saved[case["reconstruction_array_key"]][idx]
                if vector.shape != (101,) or not np.isfinite(vector).all() or vector[-1] <= 0:
                    raise ValueError("Invalid vector or nonpositive duration; no clipping/repair applied")
                xy, duration = vector[:100].reshape(50, 2), float(vector[-1])
                frame = float(case["metadata_min_frame"]) + np.linspace(0, duration, 50) * fps
                tidy = pd.DataFrame({"frame": frame, "x": xy[:, 0], "y": xy[:, 1],
                                     "length": actor.length, "width": actor.width})
                predicted = traj_from_tidy(tidy, fps=fps, kinematics="derived")
                generated = nonpet_descriptors(predicted, ego)
                row.update(generated_shared_span_s=shared_span(predicted, ego),
                           generated_min_frame=float(frame[0]), generated_max_frame=float(frame[-1]),
                           generated_duration_s=duration, generated_samples=50,
                           real_actor_samples=len(actor.frame), ego_samples=len(ego.frame))
                errs = " | ".join(v for v in [real.get("estimator_errors", ""), generated.get("estimator_errors", "")] if v)
                row.update(score_status="partial_estimator_error" if errs else "ok", score_detail=errs)
            except Exception as exc:
                row.update(score_status="failed", score_detail=f"{type(exc).__name__}: {exc}")
            row.update({f"real_{key}": real.get(key, np.nan) for key in KEYS})
            row.update({f"generated_{key}": generated.get(key, np.nan) for key in KEYS})
            row.update(pair_differences(real, generated, fps))
            row.update(real_pet=np.nan, generated_pet=np.nan)
            paired_rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"non-PET {i + 1}/{len(cases)} ({time.monotonic() - start:.1f}s)", flush=True)
    if guard["blocked_pet_calls"]:
        raise AssertionError(f"Unexpected indirect PET dependency: {guard}")
    paired = pd.DataFrame(paired_rows)
    assert len(paired) == len(cases)
    summary = generate_summary(paired)
    outputs = {}
    for name, table in (("paired", paired), ("summary", summary),
                        ("replay", pd.DataFrame([v[3] for v in ground_truth.values()]))):
        path = args.output_dir / f"{args.output_prefix}_{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = {"path": str(path), "rows": len(table), "sha256": sha256(path)}
    source_paths = [args.cases, Path(__file__), ROOT / "hetero-param/hetero_param/similarity/interaction_sim.py",
                    ROOT / "hetero-param/hetero_param/similarity/core.py", ROOT / "hetero-param/hetero_param/sweep/core.py",
                    ROOT / "hetero-param/hetero_param/interaction.py", *map(Path, sources), *arrays]
    audit = {"scope": "Official raw SVD d5 fullfit/LOGO baseline, as confirmed in DECISIONS.md; no new sampling or simulation",
             "n_cases": len(cases), "n_replay_cases": len(ground_truth),
             "score_status_counts": paired.score_status.value_counts().to_dict(),
             "clock_aligned_n": int(paired.clock_aligned.sum()),
             "clock_misaligned_n": int((~paired.clock_aligned).sum()),
             "clock_offset_definition": "metadata_min_frame minus actual_actor_min_frame; no shift applied",
             "pet_status": "pending_review_not_computed", **guard,
             "fps_by_dataset": fps_by_dataset, "metrics_units": UNITS,
             "source_files": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in source_paths],
             "outputs": outputs, "elapsed_s": time.monotonic() - start}
    (args.output_dir / f"{args.output_prefix}_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    report = ["# SVD d5 主 baseline 非PET配對結果", "", f"保留全部 {len(paired)} 筆fullfit/LOGO案例；GT replay {len(ground_truth)}筆。",
              f"評分狀態：{paired.score_status.value_counts().to_dict()}；PET未計算，guard確認0次PET呼叫。",
              f"起始時鐘一致 {int(paired.clock_aligned.sum())} 筆；不一致 {int((~paired.clock_aligned).sum())} 筆，後者標示「起始時鐘疑點，未用於方法優劣主張」。",
              "", "依最新DECISIONS.md，raw d5是正式指定baseline，不固定頭尾。尚無相同場景的ours／SAK配對對照，不能從此表宣布方法優勢。",
              "生成時間沿用metadata_min_frame，不補移時。GT actor與SVD actor均由xy導出kinematics；GT原生30Hz、SVD50點，保留舊estimator各自的取樣解析度。Ego沿用recorded heading/speed，accel為速度微分。",
              "conflict_angle/closing_speed/DRAC的既有函式另在共同frame grid由兩車xy導出，不採上面供arrival用的heading。arrival anchor是既有path-nearest conflict point，不是PET boundary。",
              "既有函式按各自可用的共享時間範圍評分，不因生成duration不同裁短GT baseline；shared_span_s另列。這不是新的同曝險試驗。",
              "", "下表只列時鐘一致案例之median absolute paired error；完整summary同列finite n、nonfinite n及全部／錯位診斷。", "",
              "| subset | mode | metric | finite / cases | median abs error | unit |",
              "|---|---|---|---:|---:|---|"]
    selected = summary[summary.scope.eq("clock_aligned_only") & summary.metric.isin(["min_dist", "min_ttc", "conflict_angle", "agent_arr_speed"])]
    for row in selected.itertuples():
        report.append(f"| {row.subset} | {row.mode} | {row.metric} | {row.paired_finite_n}/{row.n_cases} | {row.median_abs_difference:.5g} | {row.difference_unit} |")
    (args.output_dir / f"{args.output_prefix}_SUMMARY.md").write_text("\n".join(report) + "\n")
    print(json.dumps({k: audit[k] for k in ("n_cases", "n_replay_cases", "score_status_counts", "clock_aligned_n", "clock_misaligned_n", "blocked_pet_calls", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
