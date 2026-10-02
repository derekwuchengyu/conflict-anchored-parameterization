#!/usr/bin/env python3
"""E3 like-for-like copy: derive a decoded-polyline-support-truncated batch.

Label: "SVD executed (timed Polyline, E3)".

After the timed Polyline's last vertex the Agent1 FollowTrajectoryAction
completes and esmini's default behaviour takes over (the entity re-aligns to
its tracked road position and continues at its last speed); the algebraic SVD
decode has no samples after the 50th vertex.  Scorer 31 selects the metadata
window, so the as-executed scores include that post-polyline tail.  This script
mirrors scripts/40_truncate_to_gt_support.py: for every completed sample of
runs/svd_d5_executed_<stage>/<run_id>/ it writes
runs/svd_d5_executed_<stage>_polytrunc/<run_id>/<sample_id>/ whose
trajectory.parquet keeps only target rows with
within_decoded_polyline_support == True (all ego rows kept), re-hashes the
trajectory artifacts, symlinks the other files and records the derivation.
Nothing under the original batch is modified.  Scoring is left to the caller
(31 per batch; 32 exactly once over all paired CSVs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
NOTE = "target rows limited to the decoded polyline support (time <= last vertex + 0.5/fps); E3 like-for-like with the algebraic 50-sample decode"
LINKED = ("run.xosc", "patched_source.xosc", "run.csv", "esmini.log", "stdout.txt", "stderr.txt")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_sample(src_dir: Path, dst_dir: Path):
    sample = json.loads((src_dir / "sample.json").read_text())
    if sample.get("status") != "completed":
        return dict(sample_id=src_dir.name, status="skipped_not_completed")
    src_traj = Path(sample.get("trajectory_path", src_dir / "trajectory.parquet"))
    tidy = pd.read_parquet(src_traj)
    target = tidy.role.eq("target")
    keep = (~target) | (target & tidy.within_decoded_polyline_support.astype(bool))
    trimmed = tidy[keep].reset_index(drop=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_traj, dst_csv = dst_dir / "trajectory.parquet", dst_dir / "trajectory.csv"
    trimmed.to_parquet(dst_traj, index=False)
    trimmed.to_csv(dst_csv, index=False, float_format="%.12g")
    for name in LINKED:
        src, dst = src_dir / name, dst_dir / name
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
    derived = dict(sample)
    derived.update(sample_dir=str(dst_dir), trajectory_path=str(dst_traj), trajectory_rows=int(len(trimmed)),
                   derived_truncation=NOTE, derived_from_sample_json=str(src_dir / "sample.json"),
                   derived_from_sample_json_sha256=sha256(src_dir / "sample.json"),
                   derived_original_trajectory_sha256=sha256(src_traj),
                   derived_target_rows_original=int(target.sum()), derived_target_rows_kept=int(trimmed.role.eq("target").sum()),
                   derived_target_rows_removed=int(target.sum() - trimmed.role.eq("target").sum()),
                   derived_ego_rows=int((~target).sum()),
                   derived_polyline_end_s=(sample.get("fidelity") or {}).get("polyline_end_s"))
    derived["geometry_variant_id"] = sample.get("job", {}).get("geometry_variant_id")
    artifacts = []
    for item in sample.get("artifacts", []):
        p = Path(item["path"])
        if p.name in ("trajectory.parquet", "trajectory.csv"):
            q = dst_traj if p.name == "trajectory.parquet" else dst_csv
            artifacts.append(dict(path=str(q), sha256=sha256(q), size_bytes=q.stat().st_size,
                                  derived_from=item["path"], original_sha256=item["sha256"]))
        else:
            artifacts.append(item)
    derived["artifacts"] = artifacts
    (dst_dir / "sample.json").write_text(json.dumps(derived, indent=2, ensure_ascii=False) + "\n")
    return dict(sample_id=src_dir.name, status="derived", target_rows_original=int(target.sum()),
                target_rows_kept=derived["derived_target_rows_kept"], target_rows_removed=derived["derived_target_rows_removed"],
                polyline_end_s=derived["derived_polyline_end_s"],
                metadata_window_frames=sample.get("context", {}).get("metadata_window_frames"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("smoke", "recon", "kde"), required=True)
    args = parser.parse_args()
    source_batch = PROJECT / f"runs/svd_d5_executed_{args.stage}"
    latest = json.loads((source_batch / "latest_batch.json").read_text())
    run_dir = Path(latest["batch_dir"])
    derived_run = PROJECT / f"runs/svd_d5_executed_{args.stage}_polytrunc" / run_dir.name
    derived_run.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    records = [derive_sample(src, derived_run / src.name)
               for src in sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "sample.json").exists())]
    table = pd.DataFrame(records)
    table.to_csv(derived_run / "truncation_summary.csv", index=False)
    status = dict(execution_label="SVD executed (timed Polyline, E3)", run_id=run_dir.name, source_batch=str(run_dir),
                  derived_batch=str(derived_run), derivation=NOTE, n_samples=len(table),
                  status_counts=table.status.value_counts().to_dict(),
                  target_rows_removed_total=int(table.target_rows_removed.fillna(0).sum()),
                  samples_with_removed_rows=int(table.target_rows_removed.fillna(0).gt(0).sum()),
                  elapsed_s=time.monotonic() - start, script=str(Path(__file__).resolve()), script_sha256=sha256(__file__))
    (derived_run / "batch_status.json").write_text(json.dumps(status, indent=2) + "\n")
    (PROJECT / f"runs/svd_d5_executed_{args.stage}_polytrunc/latest_batch.json").write_text(
        json.dumps(dict(batch_dir=str(derived_run), run_id=run_dir.name, source_batch=str(run_dir)), indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
