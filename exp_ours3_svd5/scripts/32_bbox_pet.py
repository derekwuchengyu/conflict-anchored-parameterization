#!/usr/bin/env python3
"""Approved bbox-PET angle adapter for saved SVD, ours, and SAKURA results.

Pass --paired repeatedly for existing non-PET paired CSVs. Outputs are separate;
inputs are never overwritten. A persistent hash cache reuses PET values for
identical trajectories and metric code when further batches are appended.
"""
from __future__ import annotations

import argparse
import hashlib
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


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = module_at("nonpet32_shared", PROJECT / "scripts/30_interaction_metrics.py")
ADAPTER_PATH = PROJECT / "scripts/bbox_pet_adapter.py"
A = module_at("approved_pet_adapter", ADAPTER_PATH)
PET = M.ISIM.pet_utils
THRESHOLD, GATE = 1.4, "bbox"


def sanity_checks():
    """Actual geometry plus spy checks of gate units and immutable Traj inputs."""
    center = np.array([0., 0.])
    correct = PET.get_rotated_corners(center, 4., 2., np.pi / 2)
    np.testing.assert_allclose(correct.min(axis=0), [-1, -2], atol=1e-12)
    np.testing.assert_allclose(correct.max(axis=0), [1, 2], atol=1e-12)
    wrong = PET.get_rotated_corners(center, 4., 2., 90.)
    assert PET.rect_circle_overlap(correct, np.array([-1.5, 1.5]), .1) == False
    assert PET.rect_circle_overlap(wrong, np.array([-1.5, 1.5]), .1) == True
    frame = np.array([100., 101.00001, 101.99999])
    def trajectory(x, heading):
        return M.SIMC.Traj(frame=frame.copy(), x=np.full(3, x), y=np.arange(3.),
                          heading=np.full(3, heading), speed=np.ones(3), accel=np.zeros(3),
                          fps=30., length=4., width=2.)
    agent, ego = trajectory(0., 90.), trajectory(2., 0.)
    snapshot = {role: {key: getattr(traj, key).copy() for key in ("frame", "x", "y", "heading", "speed", "accel")}
                for role, traj in (("agent", agent), ("ego", ego))}
    seen = {}
    def capture(_xy1, _xy2, frame1, frame2, **kwargs):
        seen.update(kwargs)
        np.testing.assert_array_equal(frame1, frame.astype(int))
        np.testing.assert_array_equal(frame2, frame.astype(int))
        return 1.25, 1, 2, {"type": "no_intersection", "min_distance": 2.}
    result = A.pet_bbox_degrees(agent, ego, calculate_pet_fn=capture,
                               conflict_threshold=THRESHOLD, gate=GATE)
    assert result["pet"] == 1.25 and seen["heading_in_degrees"] is False
    assert seen["fps"] == 30. and seen["gate"] == GATE and seen["conflict_threshold"] == THRESHOLD
    np.testing.assert_allclose(seen["heading1"], np.pi / 2, atol=1e-12)
    assert not np.shares_memory(seen["heading1"], agent.heading)
    gate_angles = []
    original = PET._corners_batch
    def spy(points, length, width, radians):
        gate_angles.append(np.asarray(radians).copy())
        return original(points, length, width, radians)
    try:
        PET._corners_batch = spy
        real_result = A.pet_bbox_degrees(agent, ego, calculate_pet_fn=PET.calculate_pet,
                                        conflict_threshold=THRESHOLD, gate=GATE)
    finally:
        PET._corners_batch = original
    assert len(gate_angles) == 2, "Sanity geometry must exercise the bbox gate"
    np.testing.assert_allclose(gate_angles[0], np.pi / 2, atol=1e-7)
    np.testing.assert_allclose(gate_angles[1], 0., atol=1e-7)
    for role, traj in (("agent", agent), ("ego", ego)):
        for key, values in snapshot[role].items():
            np.testing.assert_array_equal(getattr(traj, key), values)
    reviewed = module_at("reviewed_pet_adapter_check", PROJECT / "review_only/bbox_pet_degree_adapter.py")
    import inspect
    assert inspect.getsource(A.pet_bbox_degrees) == inspect.getsource(reviewed.pet_bbox_degrees)
    return dict(status="passed", orientation_90deg_extents=[[-1, -2], [1, 2]],
                known_circle_correct_overlap=False, known_circle_legacy_raw90_overlap=True,
                actual_bbox_gate_exercised=True, gate_receives_pi_over_2_not_converted_twice=True,
                no_input_arrays_mutated=True, existing_integer_frame_cast_preserved=True,
                approved_function_matches_reviewed_proposal=True, actual_toy_pet=real_result["pet"])


def encoded(result):
    return {k: (str(v) if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
            for k, v in result.items()}


def decoded(result):
    return {k: (float(v) if k in ("pet", "pet_abs", "min_path_dist") else v) for k, v in result.items()}


def frame_diagnostics(traj, prefix):
    integer = traj.frame.astype(int)
    return {f"{prefix}_frame_samples": len(traj.frame),
            f"{prefix}_noninteger_frame_n": int(np.count_nonzero(traj.frame != integer)),
            f"{prefix}_int_cast_duplicate_n": len(integer) - len(np.unique(integer)),
            f"{prefix}_int_cast_max_abs_frames": float(np.max(np.abs(traj.frame - integer))),
            f"{prefix}_frame_origin": float(traj.frame[0]), f"{prefix}_frame_end": float(traj.frame[-1])}


class PetCache:
    def __init__(self, path):
        self.path, self.hits, self.misses = path, 0, 0
        self.algorithm = dict(adapter_sha256=M.sha256(ADAPTER_PATH), legacy_sha256=M.sha256(PET.__file__),
                              threshold=THRESHOLD, gate=GATE, numpy_version=np.__version__,
                              input_heading="degrees", internal_heading="radians", heading_in_degrees=False,
                              frame_policy="legacy_astype_int_no_rounding_or_shift")
        self.algorithm_hash = hashlib.sha256(json.dumps(self.algorithm, sort_keys=True).encode()).hexdigest()
        content = json.loads(path.read_text()) if path.exists() else {}
        self.entries = content.get("entries", {})

    def evaluate(self, agent, ego):
        digest = hashlib.sha256(self.algorithm_hash.encode())
        for traj in (agent, ego):
            digest.update(np.asarray([traj.fps, traj.length, traj.width, len(traj.frame)], dtype="<f8").tobytes())
            for values in (traj.frame, traj.x, traj.y, traj.heading):
                digest.update(np.asarray(values, dtype="<f8").tobytes())
        key = digest.hexdigest()
        if key in self.entries:
            self.hits += 1
            return decoded(self.entries[key]["result"]), key, True
        result = A.pet_bbox_degrees(agent, ego, calculate_pet_fn=PET.calculate_pet,
                                   conflict_threshold=THRESHOLD, gate=GATE)
        self.entries[key] = {"algorithm_hash": self.algorithm_hash, "result": encoded(result)}
        self.misses += 1
        return result, key, False

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(current_algorithm=self.algorithm, current_algorithm_hash=self.algorithm_hash,
                                             entries=self.entries), indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)


def summarize(paired):
    rows = []
    for (source, subset, mode), group in paired.groupby(["input_paired_path", "subset", "mode"], dropna=False, sort=False):
        for scope, g in (("all_cases", group), ("clock_aligned_only", group[group.clock_aligned.eq(True)]),
                         ("clock_misaligned_no_method_claim", group[group.clock_aligned.eq(False)])):
            real, gen, diff = [g[k].to_numpy(float) for k in ("real_pet", "generated_pet", "absdiff_pet")]
            finite = diff[np.isfinite(diff)]
            rows.append(dict(input_paired_path=source, subset=subset, mode=mode, scope=scope,
                             n_cases=len(g), n_score_failed=int(g.pet_score_status.ne("ok").sum()),
                             real_finite_n=int(np.isfinite(real).sum()), generated_finite_n=int(np.isfinite(gen).sum()),
                             real_inf_n=int(np.isinf(real).sum()), generated_inf_n=int(np.isinf(gen).sum()),
                             real_nan_n=int(np.isnan(real).sum()), generated_nan_n=int(np.isnan(gen).sum()),
                             paired_finite_n=len(finite), paired_nonfinite_n=len(g) - len(finite),
                             mean_abs_difference_s=float(np.mean(finite)) if len(finite) else np.nan,
                             median_abs_difference_s=float(np.median(finite)) if len(finite) else np.nan,
                             p95_abs_difference_s=float(np.percentile(finite, 95)) if len(finite) else np.nan))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--output-prefix", default="bbox_pet_corrected")
    parser.add_argument("--cache-path", type=Path, default=PROJECT / "results/bbox_pet_cache.json")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("Output prefix must be a plain file prefix")
    sanity = sanity_checks()
    (args.output_dir / f"{args.output_prefix}_sanity.json").write_text(json.dumps(sanity, indent=2) + "\n")
    if args.sanity_only:
        print(json.dumps(sanity, indent=2))
        return
    if not args.paired or len(args.paired) != len(set(p.resolve() for p in args.paired)):
        raise ValueError("Supply one or more distinct --paired CSVs")
    shutil.copyfile(__file__, args.output_dir / f"{args.output_prefix}_scorer_snapshot.py")
    expected, input_rows, source_hashes = {}, [], {}
    for paired_path in args.paired:
        paired_path = paired_path.resolve()
        source_hashes[str(paired_path)] = M.sha256(paired_path)
        table = pd.read_csv(paired_path)
        audit_path = Path(str(paired_path).replace("_paired.csv", "_audit.json"))
        if audit_path.exists():
            source_hashes[str(audit_path)] = M.sha256(audit_path)
            for item in json.loads(audit_path.read_text()).get("source_files", []):
                expected.setdefault(str(Path(item["path"]).resolve()), set()).add(item["sha256"])
        for index, row in enumerate(table.to_dict("records")):
            input_rows.append(dict(row, input_paired_path=str(paired_path), input_case_index=index))
    def verify_source(path):
        path = Path(path).resolve()
        key = str(path)
        if key not in source_hashes:
            digest = M.sha256(path)
            if key in expected and expected[key] != {digest}:
                raise ValueError(f"Source differs from pre-PET provenance: {path}")
            source_hashes[key] = digest
        return path
    cache = PetCache(args.cache_path)
    sources, replays, models, rows = {}, {}, {}, []
    start = time.monotonic()
    for i, original in enumerate(input_rows):
        keep = ("input_paired_path", "input_case_index", "subset", "mode", "method", "scenario_uid", "scenario_id",
                "case_index", "sample_id", "sample_json", "run_id", "geometry_variant_id", "comparison_population",
                "source_tracks", "dataset", "actor", "ego", "clock_aligned", "clock_offset_frames", "clock_offset_s",
                "claim_note", "real_shared_span_s", "generated_shared_span_s", "metadata_min_frame", "metadata_max_frame")
        row = {key: original.get(key) for key in keep}
        row.update(pet_score_status="pending", pet_score_detail="", pet_version="approved_degree_to_radian_adapter",
                   pet_threshold=THRESHOLD, pet_gate=GATE, frame_policy="legacy_astype_int_no_rounding_or_shift",
                   prepet_score_status=original.get("score_status"), real_pet=np.nan, generated_pet=np.nan,
                   real_pet_abs=np.nan, generated_pet_abs=np.nan, absdiff_pet=np.nan,
                   real_pet_type="not_computed", generated_pet_type="not_computed", absdiff_pet_reason="score_failed")
        try:
            fps = float(original["fps"])
            row["fps"] = fps
            source = verify_source(original["source_tracks"])
            if source not in sources:
                tracks = pd.read_parquet(source)
                sources[source] = {(str(sid), role): g.sort_values("frame") for (sid, role), g in tracks.groupby(["scenario_id", "role"])}
            replay_key = (str(source), original["scenario_uid"])
            if replay_key not in replays:
                a = sources[source][(str(original["scenario_id"]), "actor")]
                e = sources[source][(str(original["scenario_id"]), "ego")]
                if not a.track_id.eq(int(original["actor"])).all() or not e.track_id.eq(int(original["ego"])).all():
                    raise ValueError("GT actor identities differ from pre-PET case")
                actor = M.traj_from_tidy(a, fps=fps, kinematics="derived")
                ego = M.traj_from_tidy(e, fps=fps, kinematics="recorded")
                real, key, hit = cache.evaluate(actor, ego)
                replay_row = dict(source_tracks=str(source), scenario_uid=original["scenario_uid"],
                                  scenario_id=original["scenario_id"], subset=original["subset"], fps=fps,
                                  pet_cache_key=key, **real, **frame_diagnostics(actor, "actor"), **frame_diagnostics(ego, "ego"))
                replays[replay_key] = (actor, ego, real, replay_row)
            actor, ego, real, replay_row = replays[replay_key]
            row.update({f"real_{key}": real[key] for key in ("pet", "pet_abs", "pet_type")})
            row["real_pet_min_path_dist"] = real["min_path_dist"]
            row["real_pet_cache_key"] = replay_row["pet_cache_key"]
            if original.get("score_status") == "failed":
                raise ValueError(f"Pre-PET input failed: {original.get('score_detail')}")
            if pd.notna(original.get("reconstruction_npz")):
                row["method"] = "svd_d5"
                path = verify_source(PROJECT / original["reconstruction_npz"])
                if path not in models:
                    with np.load(path, allow_pickle=False) as data:
                        models[path] = {key: data[key] for key in data.files}
                data = models[path]
                index = int(original["case_index"])
                if str(data["scenario_uid"][index]) != original["scenario_uid"]:
                    raise ValueError("SVD array row identity mismatch")
                vector = data[original["reconstruction_array_key"]][index]
                if vector.shape != (101,) or not np.isfinite(vector).all() or vector[-1] <= 0:
                    raise ValueError("Invalid saved SVD vector; no repair")
                xy = vector[:100].reshape(50, 2)
                frames = float(original["metadata_min_frame"]) + np.linspace(0, vector[-1], 50) * fps
                tidy = pd.DataFrame(dict(frame=frames, x=xy[:, 0], y=xy[:, 1], length=actor.length, width=actor.width))
                predicted = M.traj_from_tidy(tidy, fps=fps, kinematics="derived")
                row["generated_kinematics"] = "derived_50_sample_xy"
            else:
                row["method"] = original.get("method") or original["mode"]
                path = verify_source(original["trajectory_path"])
                tidy = pd.read_parquet(path).rename(columns={"speed_mps": "speed"})
                lo, hi = float(original["metadata_min_frame"]), float(original["metadata_max_frame"])
                target = tidy[tidy.role.eq("target") & tidy.frame.between(lo - .5, hi + .5)]
                if not target.actor_id.astype(str).eq(str(int(original["actor"]))).all():
                    raise ValueError("Saved generated actor identity mismatch")
                predicted = M.traj_from_tidy(target, fps=fps, kinematics="recorded")
                row["generated_kinematics"] = "esmini_recorded_heading_speed"
            row.update(frame_diagnostics(predicted, "generated"))
            generated, key, hit = cache.evaluate(predicted, ego)
            row.update({f"generated_{key}": generated[key] for key in ("pet", "pet_abs", "pet_type")})
            row.update(generated_pet_min_path_dist=generated["min_path_dist"], generated_pet_cache_key=key,
                       generated_pet_cache_hit=hit, pet_score_status="ok")
            if np.isfinite([real["pet"], generated["pet"]]).all():
                row.update(absdiff_pet=abs(real["pet"] - generated["pet"]), absdiff_pet_reason="both_finite")
            elif np.isnan([real["pet"], generated["pet"]]).any():
                row["absdiff_pet_reason"] = "estimator_nan"
            else:
                row["absdiff_pet_reason"] = "both_no_event" if np.isinf([real["pet"], generated["pet"]]).all() else (
                    "real_no_event" if np.isinf(real["pet"]) else "generated_no_event")
        except Exception as exc:
            row.update(pet_score_status="failed", pet_score_detail=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        if (i + 1) % 50 == 0:
            cache.save()
            print(f"bbox-PET {i + 1}/{len(input_rows)}; computed={cache.misses}, cache_hits={cache.hits} ({time.monotonic() - start:.1f}s)", flush=True)
    cache.save()
    paired = pd.DataFrame(rows)
    assert len(paired) == len(input_rows)
    outputs = {}
    for name, table in (("paired", paired), ("summary", summarize(paired)),
                        ("replay", pd.DataFrame([value[3] for value in replays.values()]))):
        destination = args.output_dir / f"{args.output_prefix}_{name}.csv"
        table.to_csv(destination, index=False)
        outputs[name] = dict(path=str(destination), rows=len(table), sha256=M.sha256(destination))
    for path in (Path(__file__), ADAPTER_PATH, Path(PET.__file__), PROJECT / "scripts/30_interaction_metrics.py",
                 PROJECT / "DECISIONS.md", PROJECT / "review_only/bbox_pet_degree_adapter.py"):
        source_hashes[str(path.resolve())] = M.sha256(path)
    audit = dict(scope="Approved bbox-PET angle-interface correction only; saved trajectories and recorded GT ego",
                 n_input_cases=len(input_rows), n_output_cases=len(paired), n_replay_cases=len(replays),
                 status_counts=paired.pet_score_status.value_counts().to_dict(),
                 difference_reason_counts=paired.absdiff_pet_reason.value_counts().to_dict(),
                 algorithm=cache.algorithm, algorithm_hash=cache.algorithm_hash,
                 cache_path=str(args.cache_path.resolve()), computed_pet_calls=cache.misses, cache_hits=cache.hits,
                 input_nonpet_files_unchanged=all(M.sha256(p) == source_hashes[str(p.resolve())] for p in args.paired),
                 source_files=[dict(path=p, sha256=h) for p, h in sorted(source_hashes.items())],
                 outputs=outputs, sanity=sanity, elapsed_s=time.monotonic() - start)
    assert audit["input_nonpet_files_unchanged"]
    (args.output_dir / f"{args.output_prefix}_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    report = ["# bbox-PET 角度接口修正結果", "",
              f"全部 {len(paired)} 筆成對案例保留，GT來源場景 {len(replays)} 筆。狀態：{audit['status_counts']}。",
              f"有限／非有限差值原因：{audit['difference_reason_counts']}。未把inf或NaN當0。",
              "已使用審核過的最小adapter：僅bbox-PET收到heading的弧度副本，heading_in_degrees=False避免gate重複轉換；原Traj仍為度。90°幾何、實際bbox gate及輸入不變檢查皆通過。",
              "FPS、gate=bbox、threshold=1.4、原本frame.astype(int)皆保留。未移動時間原點或更改XOSC／軌跡。生成與GT支持窗沿用非PET評分，各自exposure另列。",
              "esmini輸出小數frame有記錄精度差異，整數截斷可能造成重複frame；generated_*frame*欄完整記錄，本次不修正。起始時鐘疑點案例另標，不用於方法優劣主張。",
              "PET=0表示此估計器的衝突區占用時間重疊，不等同車體碰撞；本表未評估背景車碰撞。",
              "原非PET檔案未修改。以input_paired_path＋input_case_index連回原case，亦保留scenario_uid/mode/sample_id等鍵。",
              f"實際新算 {cache.misses} 次，cache命中 {cache.hits} 次；之後追加batch可重用相同軌跡與評分code hash的結果。", ""]
    (args.output_dir / f"{args.output_prefix}_SUMMARY.md").write_text("\n".join(report))
    print(json.dumps({k: audit[k] for k in ("n_input_cases", "n_output_cases", "n_replay_cases", "status_counts", "difference_reason_counts", "computed_pet_calls", "cache_hits", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
