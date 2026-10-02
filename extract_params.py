#!/usr/bin/env python3
"""Step 1: extract conflict-anchored parameters from recorded interactions.

For each ego-target pair: conflict anchor q_c on the target track (or the
optional anchor_frame column of the scenario list) -> circular
window entry/exit q_in, q_out -> deflection angles theta1, theta2 and end
speed v_end (recorded target speed at the anchor frame).

    python extract_params.py --tracks 00_tracks.parquet --scenarios scenarios.csv \
        --config configs/hetrod.yaml --out output/params.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core import anchor, io, theta, window


def class_config(cfg, name):
    c = dict(cfg.get("default", {}))
    c.update(cfg.get("classes", {}).get(name, {}))
    return c


def extract_one(row, tracks, cfg):
    c = class_config(cfg, row["class"])
    ego = io.window(tracks, row["ego"], row["min_frame"], row["max_frame"])
    target = io.window(tracks, row["target"], row["min_frame"], row["max_frame"])
    out = dict(anchor_method=c["anchor"], L_m=float(c["L"]))
    if len(ego) < 2 or len(target) < 2:
        return dict(out, status="no_track")
    if pd.notna(row.get("anchor_frame", np.nan)):          # anchor given in the scenario list
        frame, kind = int(row["anchor_frame"]), "given"
        out["anchor_method"] = "given"
    else:
        frame, kind = anchor.find_anchor(c["anchor"], ego, target, fps=cfg["fps"],
                                         conflict_threshold=cfg.get("conflict_threshold", 2.0))
    if frame is None:
        return dict(out, status="no_anchor")
    out.update(anchor_kind=kind, anchor_frame=frame)
    disk = window.disk_points(target, frame, float(c["L"]))
    out.update({k: v for k, v in disk.items() if k != "anchor_index"})
    if disk["status"] != "ok":
        return out
    cps = [[disk["qm_x"], disk["qm_y"]], [disk["qc_x"], disk["qc_y"]], [disk["qp_x"], disk["qp_y"]]]
    geom = theta.theta_geometry(cps, target)
    if geom is None:
        return dict(out, status="degenerate_geometry")
    speed = float(target.loc[target.frame == frame, "speed"].iloc[0])
    out.update(u1_x=float(geom["u1"][0]), u1_y=float(geom["u1"][1]), u1_source=geom["u1_source"],
               L1=geom["L1"], L2=geom["L2"], theta1_deg=geom["theta1"], theta2_deg=geom["theta2"],
               end_speed_kmh=speed * 3.6)
    if "middle_cp_weight" in c:
        out["middle_cp_weight"] = float(c["middle_cp_weight"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True, help="tracks file (.parquet/.csv, levelXdata columns)")
    ap.add_argument("--scenarios", required=True, help="scenario list csv")
    ap.add_argument("--config", default="configs/hetrod.yaml")
    ap.add_argument("--out", default="output/params.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    scenarios = io.load_scenarios(args.scenarios)
    tracks = io.load_tracks(args.tracks, set(scenarios.ego) | set(scenarios.target))
    rows = []
    for row in scenarios.to_dict("records"):
        rows.append({**row, **extract_one(row, tracks, cfg)})
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ok = df.status.eq("ok")
    print(f"{ok.sum()}/{len(df)} scenarios parameterized -> {args.out}")
    if (~ok).any():
        print(df.loc[~ok, "status"].value_counts().to_string())


if __name__ == "__main__":
    main()
