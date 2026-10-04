#!/usr/bin/env python3
"""Step 2: sample new parameter vectors with a per-class KDE.

Fits a Gaussian KDE on (theta1, theta2, v_end) of each class and draws new
vectors by dependent sampling. Each draw keeps the geometry (q_in, u1, L1, L2)
and base scenario of its kernel center. v_end is clipped at 0.

    python sample_params.py --params output/params.csv --n 1000 --seed 2026 \
        --out output/samples.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.kde import ParamKDE

COLS = ["theta1_deg", "theta2_deg", "end_speed_kmh"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True, help="output of extract_params.py")
    ap.add_argument("--n", type=int, default=1000, help="draws per class")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--classes", nargs="*", help="classes to sample (default: all)")
    ap.add_argument("--out", default="output/samples.csv")
    args = ap.parse_args()

    params = pd.read_csv(args.params, dtype={"scenario_id": str}, float_precision="round_trip")
    params = params[params.status == "ok"]
    classes = args.classes or list(dict.fromkeys(params["class"]))
    out = []
    for name in classes:
        group = params[params["class"] == name].reset_index(drop=True)
        if len(group) < 2:
            print(f"[skip] {name}: needs at least 2 scenarios")
            continue
        model = ParamKDE(group[COLS].to_numpy(float), COLS)
        rng = np.random.default_rng(args.seed)          # restarted per class
        draws, centers = model.dependent(args.n, rng)
        samples = group.iloc[centers].reset_index(drop=True)
        samples.insert(0, "sample_id", [f"{name}__seed{args.seed}__draw{k:06d}" for k in range(1, args.n + 1)])
        for j, col in enumerate(COLS):
            samples["center_" + col] = samples[col]
            samples[col] = draws[:, j]
        samples["end_speed_kmh"] = np.maximum(0.0, samples["end_speed_kmh"])
        samples["kde_bandwidth"] = model.h
        out.append(samples)
        print(f"{name}: N={len(group)} h={model.h:.4f} -> {args.n} draws")
    df = pd.concat(out, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} samples -> {args.out}")


if __name__ == "__main__":
    main()
