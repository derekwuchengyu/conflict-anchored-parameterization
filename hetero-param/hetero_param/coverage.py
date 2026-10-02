"""
Coverage / variance — proving a parameterization does MORE than replay (slide 26 + item 4).

Replay reproduces exactly ONE trajectory (the real one): zero variance, a single point in
scenario space. A parameterization exposes a RANGE (here: lateral end-offset and interaction
timing) and therefore covers a REGION — it can be faithful to the real scenario AND span a
controllable neighbourhood of criticality that replay can never reach. This module sweeps the
range, quantifies the spread (min-distance range, convex-hull area in a criticality plane),
checks whether the real scenario lies inside the covered region (representativeness), and
returns the path fan + criticality cloud for visualization.
"""
from __future__ import annotations
import numpy as np

from . import paths, parampath as PP, interaction as IX

# Default parameter ranges (match the pipeline's Agent1_Offset ±0.5..1.5 m and a ±0.5 s delay)
DEFAULT_OFFSETS = np.linspace(-1.5, 1.5, 7)     # metres, lateral
DEFAULT_SHIFTS_S = np.linspace(-0.5, 0.5, 7)    # seconds, timing


# The three real CARLA-swept parameters (from param.xosc) sampled at {min, default, max}:
#   Agent1_Offset          -> lateral start offset  (real range ±0.5 m, step 0.25)
#   Agent1_1_TA_Offset     -> lateral end offset    (trajectory offset)
#   Agent1_1_SA_EndSpeed   -> speed  -> longitudinal extent reached in the window (±~15%)
GRID_OFFSETS = (-0.5, 0.0, 0.5)          # m
GRID_SPEED_FACTORS = (0.85, 1.0, 1.15)   # x real speed


def _taper_shift(curve, start_off, end_off):
    """Shift each point perpendicular to the local tangent, tapering start_off@s=0 -> end_off@s=1
    (analog of Agent1_Offset at the start and Agent1_1_TA_Offset at the end)."""
    tang = np.gradient(curve, axis=0)
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    perp = np.stack([-tang[:, 1], tang[:, 0]], axis=1) / nrm
    t = np.linspace(0, 1, len(curve))[:, None]
    return curve + (start_off * (1 - t) + end_off * t) * perp


def grid_trajectories(dataset, ego, actor, min_frame, max_frame, cfg,
                      offsets=GRID_OFFSETS, speed_factors=GRID_SPEED_FACTORS):
    """The 3-param grid (offset_start × offset_end × speed) at {min,default,max} = 27 concrete
    trajectories (the parameterization's covered set), plus the real (GT) trajectory. Each
    trajectory is the agent's world path over the scenario window at that (offset, speed)."""
    a = IX._series(dataset, actor, min_frame, max_frame)
    if a is None:
        return [], None
    real = np.column_stack([a["x"], a["y"]])
    rs = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(a["x"]), np.diff(a["y"])))])
    total = rs[-1]
    base = PP.parameterized_path(dataset, ego, actor, min_frame, max_frame, cfg, n_samples=300)
    n = len(a["x"])
    trajs = []
    for so in offsets:
        for eo in offsets:
            curve = _taper_shift(base, so, eo)
            cs = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(curve[:, 0]), np.diff(curve[:, 1])))])
            for sf in speed_factors:
                reach = min(total * sf, cs[-1])
                arc = np.linspace(0, reach, n)
                xy = np.column_stack([np.interp(arc, cs, curve[:, 0]), np.interp(arc, cs, curve[:, 1])])
                trajs.append({"start_off": so, "end_off": eo, "speed_factor": sf, "xy": xy})
    return trajs, real


def grid_coverage(trajs, real):
    """Does the 27-trajectory envelope cover the GT? Report envelope width + GT-inside fraction."""
    if not trajs or real is None or len(real) < 2:
        return {}
    # sample the covered set laterally: at each of K longitudinal stations along GT, the spread
    # of the trajectories' nearest points around GT
    n = 20
    idx = np.linspace(0, len(real) - 1, n).astype(int)
    gt = real[idx]
    inside = 0
    widths = []
    for g in gt:
        near = [np.min(np.linalg.norm(t["xy"] - g, axis=1)) for t in trajs]
        widths.append(max(near))
        if min(near) < 0.5:   # some trajectory passes within 0.5 m of this GT station
            inside += 1
    return {"n_traj": len(trajs), "envelope_width": float(np.median(widths)),
            "gt_covered_frac": inside / n}


def grid_interaction_coverage(dataset, ego, actor, min_frame, max_frame, trajs):
    """Across the 27 param-grid trajectories, the RANGE of each interaction descriptor
    (conflict angle / closing speed / min-distance) and whether the REAL value falls inside
    — i.e. does the parameter range span the real interaction in angle/speed too, not just space."""
    a = IX._series(dataset, actor, min_frame, max_frame)
    e = IX._series(dataset, ego, min_frame, max_frame)
    if a is None or e is None or not trajs:
        return {}
    fps = paths.dataset_of(dataset)["fps"]
    rint = IX._pair_interaction(a["frame"], a["x"], a["y"], e["frame"], e["x"], e["y"], fps)
    acc = {"conflict_angle": [], "closing_speed": [], "min_dist": []}
    for t in trajs:
        xy = t["xy"]
        if len(xy) != len(a["frame"]):
            continue
        par = IX._pair_interaction(a["frame"], xy[:, 0], xy[:, 1], e["frame"], e["x"], e["y"], fps)
        if par:
            for k in acc:
                acc[k].append(par[k])
    out = {}
    for k, v in acc.items():
        if not v:
            continue
        v = np.array(v)
        rv = float(rint[k]) if rint else float("nan")
        out[k] = {"min": float(v.min()), "max": float(v.max()), "real": rv,
                  "covers": bool(v.min() <= rv <= v.max())}
    return out


def path_fan(dataset, ego, actor, min_frame, max_frame, cfg, offsets=DEFAULT_OFFSETS):
    """The spatial fan of paths produced by sweeping the lateral offset (for overlay viz)."""
    return [(float(o), PP.parameterized_path(dataset, ego, actor, min_frame, max_frame,
                                             cfg, end_offset=float(o))) for o in offsets]


def coverage_cloud(dataset, ego, actor, min_frame, max_frame, cfg,
                   offsets=DEFAULT_OFFSETS, shifts_s=DEFAULT_SHIFTS_S):
    """Interaction descriptors for each (lateral offset, timing shift) — the covered cloud."""
    fps = paths.dataset_of(dataset)["fps"]
    shifts = [int(round(s * fps)) for s in shifts_s]
    cloud = []
    for o in offsets:
        curve = PP.parameterized_path(dataset, ego, actor, min_frame, max_frame, cfg, end_offset=float(o))
        for s in shifts:
            par = IX.interaction_of_path(dataset, ego, actor, min_frame, max_frame, curve, frame_shift=s)
            if par is None:
                continue
            cloud.append({"offset": float(o), "shift": int(s),
                          "min_dist": par["min_dist"], "closing_speed": par["closing_speed"],
                          "conflict_angle": par["conflict_angle"], "drac": par["drac"]})
    return cloud


def replay_point(dataset, ego, actor, min_frame, max_frame):
    """The single real interaction — exactly what replay covers."""
    a = IX._series(dataset, actor, min_frame, max_frame)
    e = IX._series(dataset, ego, min_frame, max_frame)
    if a is None or e is None:
        return None
    fps = paths.dataset_of(dataset)["fps"]
    r = IX._pair_interaction(a["frame"], a["x"], a["y"], e["frame"], e["x"], e["y"], fps)
    if r is None:
        return None
    return {"min_dist": r["min_dist"], "closing_speed": r["closing_speed"],
            "conflict_angle": r["conflict_angle"], "drac": r["drac"]}


def _hull_area(pts):
    if len(pts) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)   # for 2-D input, .volume is the polygon area
    except Exception:
        return 0.0


def coverage_metrics(cloud, replay=None):
    """Spread of the covered region + whether replay lies inside it."""
    if not cloud:
        return {}
    md = np.array([c["min_dist"] for c in cloud])
    cs = np.array([c["closing_speed"] for c in cloud])
    ca = np.array([c["conflict_angle"] for c in cloud])
    m = {"n": len(cloud),
         "min_dist_range": float(md.max() - md.min()), "min_dist_std": float(md.std()),
         "closing_range": float(cs.max() - cs.min()),
         "conflict_angle_range": float(ca.max() - ca.min()),
         "hull_area": _hull_area(np.column_stack([md, cs]))}
    if replay:
        m["replay_min_dist"] = replay["min_dist"]
        m["replay_closing"] = replay["closing_speed"]
        m["covers_replay"] = bool(md.min() <= replay["min_dist"] <= md.max()
                                  and cs.min() <= replay["closing_speed"] <= cs.max())
    return m
