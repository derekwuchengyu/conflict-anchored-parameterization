#!/usr/bin/env python3
"""E6 (Tables 3/4): write the analytic SVD d5 + matched KDE draws as pseudo-samples.

Source: generated/svd_d5_kde_matched/<class>/seed_2026091{0,1,2}.npz
        (X [1000,101] raw decoded vectors, center_scenario_uid, applied_duration_s,
        duration_clipped, numeric_ok; scripts/11b_svd_d5_kde_matched.py).
Pattern: scripts/50_e9_svd_pseudo_samples.py (one directory per draw with a
sample.json + trajectory.parquet that scripts/31_ours3_nonpet.py and
scripts/32_bbox_pet.py can score with --method svd_d5_kde_matched_analytic).
Decode convention = scripts/30_interaction_metrics.py (~231-251) and the E3
executed decode: xy = vector[:100] as 50 time-major pairs, frame = centre
metadata_min_frame + linspace(0, applied_duration, 50) * fps, kinematics derived
from xy with hetero_param.similarity.core.derive_kinematics. Duration policy is
the existing KDE decode (applied = max(raw, 0.5) s, raw kept, clipped flagged);
nothing is repaired or resampled. Ego rows = the recorded ego track of the
centre scenario; target dims = the centre's recorded target dims. Nothing is
simulated; every sample.json says "analytic": true.

Idempotent: a draw whose sample.json exists and whose trajectory.parquet still
matches the recorded sha256 is skipped (cache per sample).

Output: runs/svd_d5_kde_matched_analytic/<class>_seed<seed>/<attempt_id>/
        results/e6_svd_pseudo_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / "hetero-param"))
from hetero_param.similarity import core as SIMC  # noqa: E402

GENERATED = PROJECT / "generated/svd_d5_kde_matched"
CASES = PROJECT / "results/svd_d5_cases.csv"
OUT = PROJECT / "runs/svd_d5_kde_matched_analytic"
MANIFEST = PROJECT / "results/e6_svd_pseudo_manifest.json"
CLASSES = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr"]
SEEDS = [20260910, 20260911, 20260912]
NT, DUR_MIN, FPS = 50, 0.5, 30.0
METHOD = "svd_d5_kde_matched_analytic"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def centre_table():
    cases = pd.read_csv(CASES, dtype={"recording": str, "scenario_id": str})
    full = cases[cases["mode"].eq("fullfit")].drop_duplicates("scenario_uid").set_index("scenario_uid")
    return full


def gt_rows(source_tracks, scenario_id, cache):
    key = (source_tracks, scenario_id)
    if key not in cache:
        tracks = pd.read_parquet(source_tracks)
        g = tracks[tracks.scenario_id.astype(str).eq(str(scenario_id))]
        e = g[g.role.eq("ego")].sort_values("frame")
        a = g[g.role.eq("actor")].sort_values("frame")
        h = e.heading_deg.to_numpy(float)
        sp = e.speed.to_numpy(float)
        fr = e.frame.to_numpy(float)
        cache[key] = dict(ego_frame=fr, ego_x=e.x.to_numpy(float), ego_y=e.y.to_numpy(float), ego_heading=h, ego_speed=sp,
                          ego_length=float(e.length.median()), ego_width=float(e.width.median()),
                          ego_track=int(e.track_id.iloc[0]), actor_track=int(a.track_id.iloc[0]),
                          target_length=float(a.length.median()), target_width=float(a.width.median()))
    return cache[key]


def tidy_from_vector(vector, sample_id, subset, scenario_id, ego_id, target_id, lo, hi, gt, duration_applied):
    xy = np.asarray(vector[:NT * 2], float).reshape(NT, 2)
    frame = lo + np.linspace(0.0, duration_applied, NT) * FPS
    heading, speed, accel = SIMC.derive_kinematics(frame, xy[:, 0], xy[:, 1], FPS)
    vx, vy = speed * np.cos(np.radians(heading)), speed * np.sin(np.radians(heading))
    t = (frame - lo) / FPS
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    target = pd.DataFrame(dict(sample_id=sample_id, scenario_id=str(scenario_id), subset=subset,
        entity_name="Agent1", actor_id=str(target_id), role="target", time_s=t,
        x=xy[:, 0], y=xy[:, 1], z=0.0, heading_deg=heading, speed_mps=speed,
        length=gt["target_length"], width=gt["target_width"],
        vx_mps=vx, vy_mps=vy, ax_mps2=ax, ay_mps2=ay, frame=frame))
    te = (gt["ego_frame"] - lo) / FPS
    evx, evy = gt["ego_speed"] * np.cos(np.radians(gt["ego_heading"])), gt["ego_speed"] * np.sin(np.radians(gt["ego_heading"]))
    ego = pd.DataFrame(dict(sample_id=sample_id, scenario_id=str(scenario_id), subset=subset,
        entity_name="Ego", actor_id=str(ego_id), role="ego", time_s=te,
        x=gt["ego_x"], y=gt["ego_y"], z=0.0, heading_deg=gt["ego_heading"], speed_mps=gt["ego_speed"],
        length=gt["ego_length"], width=gt["ego_width"],
        vx_mps=evx, vy_mps=evy, ax_mps2=np.gradient(evx, te), ay_mps2=np.gradient(evy, te), frame=gt["ego_frame"]))
    out = pd.concat([target, ego[target.columns]], ignore_index=True)
    out["within_metadata_window"] = out.frame.between(lo - .5, hi + .5)
    out["within_target_gt_support"] = out["within_metadata_window"]
    out["within_entity_gt_support"] = out["within_metadata_window"]
    return out


def process(task):
    subset, seed, force = task
    npz_path = GENERATED / subset / f"seed_{seed}.npz"
    npz_sha = sha256(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    full = centre_table()
    batch = OUT / f"{subset}_seed{seed}"
    batch.mkdir(parents=True, exist_ok=True)
    cache, rows, n_written, n_skipped = {}, [], 0, 0
    n = len(d["X"])
    for i in range(n):
        sid = str(d["attempt_id"][i])
        vec_raw = np.asarray(d["X"][i], float)
        raw_dur, app_dur = float(d["raw_duration_s"][i]), float(d["applied_duration_s"][i])
        clipped, num_ok = bool(d["duration_clipped"][i]), bool(d["numeric_ok"][i])
        if abs(vec_raw[-1] - raw_dur) > 1e-9 or abs(max(raw_dur, DUR_MIN) - app_dur) > 1e-9:
            raise RuntimeError(f"{sid}: duration bookkeeping mismatch")
        uid = str(d["center_scenario_uid"][i])
        c = full.loc[uid]
        lo, hi = int(c.metadata_min_frame), int(c.metadata_max_frame)
        source = str(Path(c.source_tracks).resolve())
        gt = gt_rows(source, c.scenario_id, cache)
        if gt["ego_track"] != int(c.ego) or gt["actor_track"] != int(c.actor):
            raise RuntimeError(f"{uid}: ego/actor identity mismatch")
        wd = batch / sid
        traj = wd / "trajectory.parquet"
        sj = wd / "sample.json"
        row = dict(sample_id=sid, subset=subset, seed=seed, draw_index=i, center_scenario_uid=uid, center_case_index=int(d["center_case_index"][i]),
                   raw_duration_s=raw_dur, applied_duration_s=app_dur, duration_clipped=clipped, numeric_ok=num_ok,
                   numeric_status=str(d["numeric_status"][i]), finite_vector=bool(np.isfinite(vec_raw).all()))
        if not force and sj.exists() and traj.exists():
            try:
                prev = json.loads(sj.read_text())
                if prev.get("artifacts") and prev["artifacts"][0]["sha256"] == sha256(traj) and prev.get("decode", {}).get("npz_sha256") == npz_sha:
                    n_skipped += 1
                    row["status"] = prev["status"]
                    rows.append(row)
                    continue
            except Exception:
                pass
        if not np.isfinite(vec_raw).all():
            row["status"] = "failed_nonfinite_vector"
            rows.append(row)
            wd.mkdir(parents=True, exist_ok=True)
            write_json(sj, dict(sample_id=sid, sample_dir=str(wd), status="failed", error="non-finite decoded vector",
                                analytic=True, simulated=False, method=METHOD))
            continue
        applied = vec_raw.copy()
        applied[-1] = app_dur
        tidy = tidy_from_vector(applied, sid, subset, c.scenario_id, int(c.ego), int(c.actor), lo, hi, gt, app_dur)
        wd.mkdir(parents=True, exist_ok=True)
        tidy.to_parquet(traj, index=False)
        source_context = dict(subset=subset, source_tracks=source, scenario_uid=uid, mode="kde", seed=seed, draw_index=i,
                              attempt_id=sid, center_case_index=int(d["center_case_index"][i]), center_scenario_uid=uid,
                              kernel_center_scenario_uid=uid, kde_npz=str(npz_path.relative_to(PROJECT)), kde_npz_sha256=npz_sha,
                              kde_array_key="X", raw_duration_s=raw_dur, applied_duration_s=app_dur, duration_clipped=clipped,
                              numeric_ok=num_ok, numeric_status=str(d["numeric_status"][i]), h_loo=float(d["h_loo"]),
                              sampling_design=str(d["sampling_design"]),
                              decode_convention="center metadata_min_frame + linspace(0, applied_duration, 50) * fps; kinematics derived from xy",
                              method=METHOD)
        job = dict(job_id=sid, scenario_id=str(c.scenario_id), **{"class": subset}, dataset=str(c.dataset), recording=str(c.recording),
                   ego=int(c.ego), target=int(c.actor), min_frame=lo, max_frame=hi,
                   geometry_variant_id="svd_d5_kde_matched_analytic_decode", source_context=source_context)
        context = dict(scenario_id=str(c.scenario_id), dataset=str(c.dataset), recording=str(c.recording), subset=subset,
                       ego=int(c.ego), target=int(c.actor), metadata_window_frames=[lo, hi], target_gt_support_frames=[lo, hi],
                       fps=FPS, geometry_variant_id=job["geometry_variant_id"], source_context=source_context, method=METHOD)
        report = dict(sample_id=sid, sample_dir=str(wd), job=job, context=context, status="completed", analytic=True, simulated=False,
                      method=METHOD, arm="svd_d5_kde_matched_analytic", execution_label="SVD d5 + matched KDE (analytic decode)",
                      parameters_requested={}, parameters_applied={}, is_nominal_default=False,
                      decode=dict(convention=source_context["decode_convention"], duration_raw_s=raw_dur, duration_applied_s=app_dur,
                                  duration_clipped=clipped, duration_rule=f"max(raw, {DUR_MIN})", npz_sha256=npz_sha,
                                  ego_rows="recorded ego track of the centre scenario", target_dims="recorded target dims of the centre scenario"),
                      vector_raw=vec_raw.tolist(), trajectory_path=str(traj), trajectory_rows=len(tidy))
        report["artifacts"] = [dict(path=str(traj), sha256=sha256(traj), size_bytes=traj.stat().st_size)]
        write_json(sj, report)
        row["status"] = "completed"
        rows.append(row)
        n_written += 1
    table = pd.DataFrame(rows)
    table.to_csv(batch / "draws.csv", index=False)
    write_json(batch / "protocol.json", dict(arm="svd_d5_kde_matched_analytic", method=METHOD, analytic=True, subset=subset, seed=seed,
                                              n=len(rows), source_npz=str(npz_path), source_sha256=npz_sha,
                                              n_duration_clipped=int(table.duration_clipped.sum()), n_numeric_not_ok=int((~table.numeric_ok).sum()),
                                              n_written=n_written, n_skipped_cached=n_skipped))
    return dict(subset=subset, seed=seed, batch_dir=str(batch), n=len(rows), n_written=n_written, n_skipped_cached=n_skipped,
                n_duration_clipped=int(table.duration_clipped.sum()), n_numeric_not_ok=int((~table.numeric_ok).sum()),
                n_completed=int(table.status.eq("completed").sum()), source_npz=str(npz_path), source_sha256=npz_sha)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--classes", nargs="*", default=CLASSES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()
    t0 = time.monotonic()
    tasks = [(c, s, args.force) for c in args.classes for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(process, tasks))
    manifest = dict(script=str(Path(__file__)), script_sha256=sha256(__file__), method=METHOD, decode_reference="scripts/30_interaction_metrics.py; scripts/50_e9_svd_pseudo_samples.py",
                    cases_csv=dict(path=str(CASES), sha256=sha256(CASES)), batches=results, elapsed_s=time.monotonic() - t0)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST, manifest)
    print(json.dumps([{k: r[k] for k in ("subset", "seed", "n", "n_written", "n_skipped_cached", "n_duration_clipped", "n_numeric_not_ok")} for r in results], indent=1))
    print(f"elapsed {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
