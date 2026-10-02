#!/usr/bin/env python3
"""E4-S sensitivity: derive a GT-support-truncated copy of the ours3 disk batch.

For every completed sample in runs/ours3_disk_population_defaults/<run_id>/,
build runs/ours3_disk_population_defaults_gttrunc/<run_id>/<sample_id>/ where
trajectory.parquet keeps only target rows with within_target_gt_support == True
(all ego rows are kept), then score the derived batch with the unchanged
scripts/31_ours3_nonpet.py and scripts/32_bbox_pet.py.

The derived sample.json is a copy of the original with: sample_dir and
trajectory_path pointing at the derived directory, the trajectory.parquet /
trajectory.csv artifact entries re-hashed (31 verifies the trajectory hash),
and a "derived_truncation" field. run.xosc, patched_source.xosc, run.csv and the
esmini log are symlinked to the original files. Nothing under the original
batch is modified.

Usage (nps env, from anywhere):
  python -B scripts/40_truncate_to_gt_support.py            # build + score
  python -B scripts/40_truncate_to_gt_support.py --no-score # build only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
SOURCE_BATCH = PROJECT / "runs/ours3_disk_population_defaults"
DERIVED_BATCH = PROJECT / "runs/ours3_disk_population_defaults_gttrunc"
NONPET_PREFIX = "ours3_disk_population_gttrunc_nonpet"
PET_PREFIX = "bbox_pet_disk_gttrunc"
NOTE = "target rows limited to GT target support; E4-S sensitivity"
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
    keep = (~target) | (target & tidy.within_target_gt_support.astype(bool))
    trimmed = tidy[keep].reset_index(drop=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_traj = dst_dir / "trajectory.parquet"
    trimmed.to_parquet(dst_traj, index=False)
    dst_csv = dst_dir / "trajectory.csv"
    trimmed.to_csv(dst_csv, index=False)
    for name in LINKED:
        src = src_dir / name
        dst = dst_dir / name
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
    derived = dict(sample)
    derived["sample_dir"] = str(dst_dir)
    derived["trajectory_path"] = str(dst_traj)
    derived["trajectory_rows"] = int(len(trimmed))
    derived["derived_truncation"] = NOTE
    derived["derived_from_sample_json"] = str(src_dir / "sample.json")
    derived["derived_from_sample_json_sha256"] = sha256(src_dir / "sample.json")
    derived["derived_original_trajectory_sha256"] = sha256(src_traj)
    derived["derived_target_rows_original"] = int(target.sum())
    derived["derived_target_rows_kept"] = int(trimmed.role.eq("target").sum())
    derived["derived_target_rows_removed"] = int(target.sum() - trimmed.role.eq("target").sum())
    derived["derived_ego_rows"] = int((~target).sum())
    derived["derived_target_gt_support_frames"] = sample.get("context", {}).get("target_gt_support_frames")
    artifacts = []
    for item in sample.get("artifacts", []):
        p = Path(item["path"])
        if p.name == "trajectory.parquet":
            artifacts.append(dict(path=str(dst_traj), sha256=sha256(dst_traj), size_bytes=dst_traj.stat().st_size,
                                  derived_from=item["path"], original_sha256=item["sha256"]))
        elif p.name == "trajectory.csv":
            artifacts.append(dict(path=str(dst_csv), sha256=sha256(dst_csv), size_bytes=dst_csv.stat().st_size,
                                  derived_from=item["path"], original_sha256=item["sha256"]))
        else:
            artifacts.append(item)
    derived["artifacts"] = artifacts
    (dst_dir / "sample.json").write_text(json.dumps(derived, indent=2, ensure_ascii=False) + "\n")
    return dict(sample_id=src_dir.name, status="derived", target_rows_original=int(target.sum()),
                target_rows_kept=derived["derived_target_rows_kept"],
                target_rows_removed=derived["derived_target_rows_removed"],
                gt_support_frames=derived["derived_target_gt_support_frames"],
                metadata_window_frames=sample.get("context", {}).get("metadata_window_frames"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-batch", type=Path, default=SOURCE_BATCH)
    parser.add_argument("--derived-batch", type=Path, default=DERIVED_BATCH)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--run-id", help="Select a source run explicitly when historical runs coexist")
    args = parser.parse_args()
    run_dirs = ([args.source_batch / args.run_id] if args.run_id else
                [p for p in sorted(args.source_batch.iterdir()) if p.is_dir()])
    if args.run_id and (Path(args.run_id).name != args.run_id or not run_dirs[0].is_dir()):
        raise SystemExit(f"Invalid source run id: {args.run_id}")
    if len(run_dirs) != 1:
        raise SystemExit(f"Expected exactly one run id under {args.source_batch}, found {[p.name for p in run_dirs]}")
    run_dir = run_dirs[0]
    derived_run = args.derived_batch / run_dir.name
    derived_run.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    records = []
    for src_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "sample.json").exists()):
        records.append(derive_sample(src_dir, derived_run / src_dir.name))
    table = pd.DataFrame(records)
    table.to_csv(derived_run / "truncation_summary.csv", index=False)
    status = dict(run_id=run_dir.name, source_batch=str(run_dir), derived_batch=str(derived_run),
                  derivation=NOTE, n_samples=len(table),
                  status_counts=table.status.value_counts().to_dict(),
                  target_rows_removed_total=int(table.target_rows_removed.fillna(0).sum()),
                  samples_with_removed_rows=int(table.target_rows_removed.fillna(0).gt(0).sum()),
                  elapsed_s=time.monotonic() - start, script=str(Path(__file__).resolve()),
                  script_sha256=sha256(__file__))
    (derived_run / "batch_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    if args.no_score:
        return
    env = dict(os.environ, MPLCONFIGDIR=os.environ.get("MPLCONFIGDIR", "/tmp/ours3_svd5_mpl"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    results = PROJECT / "results"
    cmd31 = [str(PY), "-B", str(PROJECT / "scripts/31_ours3_nonpet.py"), "--batch-dir", str(derived_run),
             "--output-prefix", NONPET_PREFIX]
    print("RUN:", " ".join(cmd31), flush=True)
    with (results / f"{NONPET_PREFIX}_stdout.txt").open("w") as out, (results / f"{NONPET_PREFIX}_stderr.txt").open("w") as err:
        subprocess.run(cmd31, check=True, env=env, cwd=str(PROJECT), stdout=out, stderr=err)
    print((results / f"{NONPET_PREFIX}_stdout.txt").read_text(), flush=True)
    paired = results / f"{NONPET_PREFIX}_paired.csv"
    cmd32 = [str(PY), "-B", str(PROJECT / "scripts/32_bbox_pet.py"), "--paired", str(paired),
             "--output-prefix", PET_PREFIX]
    print("RUN:", " ".join(cmd32), flush=True)
    with (results / f"{PET_PREFIX}_stdout.txt").open("w") as out, (results / f"{PET_PREFIX}_stderr.txt").open("w") as err:
        subprocess.run(cmd32, check=True, env=env, cwd=str(PROJECT), stdout=out, stderr=err)
    print((results / f"{PET_PREFIX}_stdout.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
