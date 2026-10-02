"""E7: physical feasibility — off-road (off-drivable-lane) rate per method.

Drivable area = union of all 'driving' lane polygons of tyms.xodr, buffered
+0.5 m (vehicle-centre tolerance), with ALL interior rings filled
(FILL_HOLES=True — junction-interior holes are tyms.xodr map artifacts, not
real non-drivable area; user decision 2026-08-11). Per generated agent path:
fraction of
points outside; a path is flagged off-road if > 5 % of its points are outside
(same spirit as the trajectory_plots lane classifier).

Sets scored: real actors (sanity), svd_d3 recon, svd_kde cm samples,
sakura/ours rendered base, and each method's variety set.

Usage: python 33_e7_offroad.py --name tlkeep
Output: results/<name>_e7_offroad.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
EXP = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, "/home/hcis-s19/Documents/ChengYu/trajectory_plots")
import xodr_lanes  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
from shapely import vectorized  # noqa: E402
from run_label_lib import trim_lead_still  # noqa: E402

XODR = "/home/hcis-s19/Documents/ChengYu/tyms.xodr"
BUF = 0.5
OUT_FRAC = 0.05
# USER DECISION (2026-08-11): the driving-lane union of tyms.xodr contains two
# interior holes in the middle of the junction (visible as triangular notches
# in figs/offroad_trajs_*.png).  They are map artifacts of the lane
# tessellation, NOT real non-drivable area, so the off-road gate must not
# count them.  FILL_HOLES=True rebuilds every union component from its
# exterior ring only (Polygon(p.exterior)) after union+buffer, removing ALL
# interior rings while leaving the exterior boundary untouched.  Set to False
# (or pass fill_holes=False) to reproduce the old, holed union.
FILL_HOLES = True


def _fill_interior_rings(area):
    """Rebuild each polygon component from its exterior only (drops all
    interior rings).  Exterior rings of a valid (Multi)Polygon are disjoint,
    so unary_union just reassembles the components."""
    geoms = area.geoms if hasattr(area, "geoms") else [area]
    return unary_union([Polygon(p.exterior) for p in geoms])


def drivable_union(fill_holes: bool | None = None):
    if fill_holes is None:
        fill_holes = FILL_HOLES
    roads, _ = xodr_lanes.parse(XODR)
    polys = []
    for rid, road in roads.items():
        for lane_id, (inner, outer) in road.lane_edges(ds=0.3,
                                                       types=("driving",)).items():
            ring = np.vstack([inner, outer[::-1]])
            try:
                p = Polygon(ring).buffer(0)
                if p.is_valid and p.area > 0:
                    polys.append(p)
            except Exception:  # noqa: BLE001
                continue
    area = unary_union(polys).buffer(BUF)
    if fill_holes:
        area = _fill_interior_rings(area)
    return area


def score(paths, area):
    """paths: list of (x, y) arrays -> (mean point-out frac, offroad flag rate)."""
    fracs = []
    for x, y in paths:
        inside = vectorized.contains(area, np.asarray(x), np.asarray(y))
        fracs.append(1.0 - float(inside.mean()))
    fracs = np.asarray(fracs)
    return float(fracs.mean()), float((fracs > OUT_FRAC).mean()), len(fracs)


def main(name: str):
    ddir, gdir, rdir = EXP / "data" / name, EXP / "generated" / name, EXP / "results"
    area = drivable_union()
    rows = []

    real = pd.read_parquet(ddir / "real_tracks.parquet")
    paths = [(g.sort_values("frame").x.values, g.sort_values("frame").y.values)
             for _, g in real[real.role == "actor"].groupby("scenario_id")]
    rows.append(("real", *score(paths, area)))

    for m, fname, tags in (("svd_d3", "trajectories.parquet", None),
                           ("svd_kde", "trajectories_fid_cm.parquet", None),
                           ("svd_kde/vc", "trajectories_vc.parquet", None)):
        folder = m.split("/")[0]
        df = pd.read_parquet(gdir / folder / fname)
        paths = [(g.sort_values("frame").x.values, g.sort_values("frame").y.values)
                 for _, g in df.groupby("scenario_id")]
        rows.append((m, *score(paths, area)))

    for m in ("sakura", "ours"):
        df = pd.read_parquet(gdir / m / "trajectories.parquet")
        df = df[df.role == "agent"]
        for setname, tags in ((f"{m}/base", ("base",)),
                              (f"{m}/variants", ("of-", "of+", "en-", "en+"))):
            paths = []
            for _, g in df[df.tag.isin(tags)].groupby(["scenario_id", "tag"]):
                g = trim_lead_still(g.sort_values("frame"))
                paths.append((g.x.values, g.y.values))
            rows.append((setname, *score(paths, area)))

    out = pd.DataFrame(rows, columns=["set", "mean_point_out_frac",
                                      "offroad_path_rate", "n"])
    out.to_csv(rdir / f"{name}_e7_offroad.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    main(ap.parse_args().name)
