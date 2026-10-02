#!/usr/bin/env python3
"""Step 3: write executable OpenSCENARIO files (and optionally run esmini).

Input is params.csv (reconstruction of the recorded interactions) or
samples.csv (KDE variants). Each row patches its base logical xosc with the
control points [q_in, q_c, q_out] rebuilt from (theta1, theta2) and sets the
end speed v_end.

    python generate_scenarios.py --params output/samples.csv --out output/scenarios
    # also execute in esmini with the ego replaying its recorded trajectory:
    python generate_scenarios.py --params output/samples.csv --out output/scenarios \
        --run --tracks 00_tracks.parquet --xodr tyms.xodr --esmini esmini/bin/esmini
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from core import esmini, io, theta, xosc


def geometry(row):
    return dict(pm=[row["qm_x"], row["qm_y"]], u1=[row["u1_x"], row["u1_y"]], L1=row["L1"], L2=row["L2"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True, help="params.csv or samples.csv")
    ap.add_argument("--out", default="output/scenarios")
    ap.add_argument("--base-dir", default=".", help="directory that relative base_xosc paths refer to")
    ap.add_argument("--limit", type=int, help="only the first N rows")
    ap.add_argument("--run", action="store_true", help="execute each scenario in esmini")
    ap.add_argument("--tracks", help="tracks file (needed with --run for the ego replay)")
    ap.add_argument("--xodr", help="OpenDRIVE map (needed with --run)")
    ap.add_argument("--esmini", default="esmini/bin/esmini")
    ap.add_argument("--catalogs", default="/opt/Catalogs", help="esmini catalog root")
    ap.add_argument("--config", default="configs/hetrod.yaml")
    args = ap.parse_args()

    df = pd.read_csv(args.params, dtype={"scenario_id": str}, float_precision="round_trip")
    df = df[df.status == "ok"].head(args.limit) if args.limit else df[df.status == "ok"]
    if "base_xosc" not in df:
        raise SystemExit("params need a base_xosc column (add it to the scenario list)")
    fps = yaml.safe_load(Path(args.config).read_text())["fps"]
    tracks = io.load_tracks(args.tracks, set(df.ego)) if args.run else None
    out_dir = Path(args.out)
    n_ok = 0
    for row in df.to_dict("records"):
        name = row.get("sample_id") or row["scenario_id"]
        cps = theta.theta_to_cps(geometry(row), row["theta1_deg"], row["theta2_deg"])
        base = Path(row["base_xosc"])
        base = base if base.is_absolute() else Path(args.base_dir) / base
        weight = row.get("middle_cp_weight")
        logical = xosc.write_scenario(base, out_dir / name / "scenario.xosc", cps, row["end_speed_kmh"],
                                      None if pd.isna(weight) else weight)
        if args.run:
            ego = io.window(tracks, row["ego"], row["min_frame"], row["max_frame"])
            run_xosc = esmini.to_esmini_replay(logical, out_dir / name / "run.xosc", ego, Path(args.xodr).resolve(),
                                               fps, row["min_frame"], row["max_frame"], args.catalogs)
            try:
                csv_path = esmini.run(run_xosc, out_dir / name, Path(args.esmini).resolve(), fps)
                esmini.read_csv(csv_path, row["min_frame"], fps).to_csv(out_dir / name / "trajectory.csv", index=False)
            except Exception as exc:                    # keep going; report at the end
                print(f"[fail] {name}: {exc}")
                continue
        n_ok += 1
    print(f"{n_ok}/{len(df)} scenarios written to {out_dir}")


if __name__ == "__main__":
    main()
