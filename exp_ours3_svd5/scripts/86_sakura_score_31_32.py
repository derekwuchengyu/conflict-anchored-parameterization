#!/usr/bin/env python3
"""Run the unchanged non-PET scorer (31) and bbox-PET scorer (32) on every SAKURA arm batch of scripts/85.

31: one call per batch (--method <arm>, --output-prefix sak_<arm>_nonpet)
32: one call per 31 paired file, strictly one process at a time, --cache-path results/bbox_pet_cache_sak.json
    (results/bbox_pet_cache.json is never touched).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
PY = sys.executable
BATCHES = [  # (batch dir glob, 31 method label, output tag)
    ("sakura_plain_defaults_extra", "sakura_plain_extra", "sakura_plain_extra"),
    ("sakura_route_defaults", "sakura_route", "sakura_route"),
    ("sakura_route_kde_s20260910", "sakura_route_kde", "sakura_route_kde_s20260910"),
    ("sakura_route_kde_s20260911", "sakura_route_kde", "sakura_route_kde_s20260911"),
    ("sakura_route_kde_s20260912", "sakura_route_kde", "sakura_route_kde_s20260912"),
    ("sakura_route_kde_noclip_pilot", "sakura_route_kde_noclip", "sakura_route_kde_noclip"),
    ("sakura_route_kde_39_180_cond", "sakura_route_kde_cond_39_180", "sakura_route_kde_cond_39_180"),
]


def batch_dir(name):
    lb = PROJECT / "runs" / name / "latest_batch.json"
    return Path(json.loads(lb.read_text())["batch_dir"])


def run(cmd, log_prefix):
    t0 = time.time()
    with (PROJECT / f"results/{log_prefix}_stdout.txt").open("wb") as so, (PROJECT / f"results/{log_prefix}_stderr.txt").open("wb") as se:
        r = subprocess.run(cmd, cwd=PROJECT, stdout=so, stderr=se)
    print(f"[{log_prefix}] rc={r.returncode} ({time.time() - t0:.0f}s)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"{log_prefix} failed; see results/{log_prefix}_stderr.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="output tags to (re)score")
    ap.add_argument("--skip-31", action="store_true")
    ap.add_argument("--skip-32", action="store_true")
    args = ap.parse_args()
    todo = [b for b in BATCHES if not args.only or b[2] in args.only]
    if not args.skip_31:
        for name, method, tag in todo:
            bd = batch_dir(name)
            run([PY, "-B", "scripts/31_ours3_nonpet.py", "--batch-dir", str(bd), "--method", method,
                 "--output-prefix", f"sak_{tag}_nonpet"], f"sak_{tag}_31")
    if not args.skip_32:
        for name, method, tag in todo:
            paired = PROJECT / f"results/sak_{tag}_nonpet_paired.csv"
            run([PY, "-B", "scripts/32_bbox_pet.py", "--paired", str(paired), "--output-prefix", f"bbox_pet_sak_{tag}",
                 "--cache-path", str(PROJECT / "results/bbox_pet_cache_sak.json")], f"bbox_pet_sak_{tag}_32")


if __name__ == "__main__":
    main()
