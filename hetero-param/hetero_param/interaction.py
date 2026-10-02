"""
Two-vehicle INTERACTION descriptors — quantify the critical interaction between the
parameterized challenge agent and the real ego, and how faithfully it matches the real
agent↔ego interaction. This complements path-only fidelity (path_dev) with the thing that
actually matters for AV testing: the moment and window where the two vehicles interact.

Descriptors at the critical (closest-approach) frame — standard surrogate-safety /
conflict quantities from the AV-safety literature (TTC/PET family + conflict geometry):
  min_dist        closest inter-vehicle distance (m)
  conflict_angle  angle between the two headings at closest approach (deg, 0..180)
  closing_speed   rate of approach at closest approach (m/s; + = closing)
  rel_speed       |v_agent - v_ego| at closest approach (m/s)
  crit_frame      the frame of closest approach
Interaction WINDOW: the inter-vehicle distance-vs-time profile in a window around the
critical frame — its shape similarity (correlation) captures the interaction *period*,
not just the instant.

The parameterized agent path is spatial; we time-parameterize it by the REAL agent's
arc-length-vs-time (it keeps the real speed profile — i.e. strategy C timing) so the
temporal descriptors are well defined, then compare (param agent ↔ real ego) against the
ground-truth (real agent ↔ real ego).
"""
from __future__ import annotations
import math
import numpy as np

from . import paths, parampath as PP


def _series(dataset, tid, min_frame, max_frame):
    traj = PP._traj(dataset)
    try:
        d = traj.xs(int(tid), level="track_id").sort_index()
    except KeyError:
        return None
    d = d[(d.index >= min_frame) & (d.index <= max_frame)]
    if len(d) < 2:
        return None
    return {"frame": d.index.values.astype(float),
            "x": d["x"].values.astype(float),
            "y": d["y"].values.astype(float)}


def _heading_speed(frame, x, y, fps):
    dt = np.gradient(frame) / fps
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    dx, dy = np.gradient(x), np.gradient(y)
    speed = np.hypot(dx, dy) / dt
    heading = np.degrees(np.arctan2(dy, dx))
    return heading, speed


def time_parameterize(param_curve, real_x, real_y):
    """Place the param path in time by matching the REAL agent's arc-length-vs-frame,
    so it inherits the real speed profile. Returns param (x, y) sampled at each real frame."""
    rs = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(real_x), np.diff(real_y)))])
    rs = rs / rs[-1] if rs[-1] > 0 else rs
    px, py = param_curve[:, 0], param_curve[:, 1]
    ps = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    ps = ps / ps[-1] if ps[-1] > 0 else ps
    return np.interp(rs, ps, px), np.interp(rs, ps, py)


def _pair_interaction(af, ax, ay, ef, ex, ey, fps):
    """Interaction descriptors between agent (af,ax,ay) and ego (ef,ex,ey) over shared frames."""
    lo, hi = max(af.min(), ef.min()), min(af.max(), ef.max())
    if hi - lo < 2:
        return None
    grid = np.arange(lo, hi + 1)
    axg, ayg = np.interp(grid, af, ax), np.interp(grid, af, ay)
    exg, eyg = np.interp(grid, ef, ex), np.interp(grid, ef, ey)
    dist = np.hypot(axg - exg, ayg - eyg)
    ci = int(np.argmin(dist))
    ah, asp = _heading_speed(grid, axg, ayg, fps)
    eh, esp = _heading_speed(grid, exg, eyg, fps)
    da = abs(ah[ci] - eh[ci]) % 360
    conflict = da if da <= 180 else 360 - da
    signed = math.degrees(math.atan2(math.sin(math.radians(ah[ci] - eh[ci])),
                                     math.cos(math.radians(ah[ci] - eh[ci]))))  # which side
    closing = -float(np.gradient(dist)[ci] * fps)   # range-rate, + = approaching
    D = float(dist[ci])
    drac = (max(closing, 0.0) ** 2) / (2 * D) if D > 1e-6 else float("nan")  # evasive-severity
    return {"crit_frame": float(grid[ci]), "min_dist": D,
            "conflict_angle": float(conflict), "signed_dpsi": float(signed),
            "closing_speed": closing, "drac": float(drac),
            "rel_speed": float(abs(asp[ci] - esp[ci])), "grid": grid, "dist": dist, "ci": ci}


def _dtw1d(a, b):
    """Length-normalized 1-D DTW distance between two sequences."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[na, nb] / (na + nb))


def _gap_window(o, half):
    ci, d = o["ci"], o["dist"]
    return d[max(0, ci - half):min(len(d), ci + half + 1)]


def _profile_similarity(real, par, half):
    """Shape correlation of the inter-vehicle distance profile in a window around each
    interaction's critical frame (the interaction *period* similarity)."""
    def win(o):
        ci, d = o["ci"], o["dist"]
        return d[max(0, ci - half):min(len(d), ci + half + 1)]
    r, p = win(real), win(par)
    n = min(len(r), len(p))
    if n < 3:
        return float("nan")
    r, p = r[:n], p[:n]
    if np.std(r) < 1e-6 or np.std(p) < 1e-6:
        return float("nan")
    return float(np.corrcoef(r, p)[0, 1])


def compare(dataset, ego, actor, min_frame, max_frame, sample_cfg, window_s=2.0):
    """Real vs parameterized interaction. Returns the ground-truth descriptors, the
    parameterized descriptors, their deltas, and the interaction-window similarity."""
    fps = paths.dataset_of(dataset)["fps"]
    a = _series(dataset, actor, min_frame, max_frame)
    e = _series(dataset, ego, min_frame, max_frame)
    if a is None or e is None:
        return None
    real = _pair_interaction(a["frame"], a["x"], a["y"], e["frame"], e["x"], e["y"], fps)
    curve = PP.parameterized_path(dataset, ego, actor, min_frame, max_frame, sample_cfg)
    px, py = time_parameterize(curve, a["x"], a["y"])
    par = _pair_interaction(a["frame"], px, py, e["frame"], e["x"], e["y"], fps)
    if real is None or par is None:
        return None
    half = int(round(window_s * fps))
    return {
        "real_min_dist": real["min_dist"], "param_min_dist": par["min_dist"],
        "d_min_dist": abs(real["min_dist"] - par["min_dist"]),                 # ~ ΔDCPA
        "real_conflict_angle": real["conflict_angle"], "param_conflict_angle": par["conflict_angle"],
        "d_conflict_angle": abs(real["conflict_angle"] - par["conflict_angle"]),
        "side_agree": 1.0 if real["signed_dpsi"] * par["signed_dpsi"] >= 0 else 0.0,
        "real_closing_speed": real["closing_speed"], "param_closing_speed": par["closing_speed"],
        "d_closing_speed": abs(real["closing_speed"] - par["closing_speed"]),  # range-rate
        "real_drac": real["drac"], "param_drac": par["drac"],
        "d_drac": abs(real["drac"] - par["drac"]),
        "real_rel_speed": real["rel_speed"], "param_rel_speed": par["rel_speed"],
        "d_rel_speed": abs(real["rel_speed"] - par["rel_speed"]),
        "d_crit_frame_s": abs(real["crit_frame"] - par["crit_frame"]) / fps,   # ~ ΔPET timing
        "profile_sim": _profile_similarity(real, par, half),                   # window shape corr
        "gap_dtw": _dtw1d(_gap_window(real, half), _gap_window(par, half)),     # window shape DTW
    }


def local_fidelity(dataset, ego, actor, min_frame, max_frame, sample_cfg, window_s=1.5):
    """Similarity DURING the fast interaction: mean displacement between the parameterized
    agent and the real agent in a ±window around the real closest-approach frame (local ADE),
    plus the min-distance error. This is what discriminates sampling METHODS — global path_dev
    is minimized by uniform, and conflict-angle is timing-saturated; local fidelity rewards the
    method whose points land where the interaction actually happens."""
    fps = paths.dataset_of(dataset)["fps"]
    a = _series(dataset, actor, min_frame, max_frame)
    e = _series(dataset, ego, min_frame, max_frame)
    if a is None or e is None:
        return None
    real = _pair_interaction(a["frame"], a["x"], a["y"], e["frame"], e["x"], e["y"], fps)
    if real is None:
        return None
    center = real["crit_frame"]
    curve = PP.parameterized_path(dataset, ego, actor, min_frame, max_frame, sample_cfg)
    px, py = time_parameterize(curve, a["x"], a["y"])
    mask = np.abs(a["frame"] - center) <= window_s * fps
    if int(mask.sum()) < 2:
        return None
    d = np.hypot(px[mask] - a["x"][mask], py[mask] - a["y"][mask])
    par = _pair_interaction(a["frame"], px, py, e["frame"], e["x"], e["y"], fps)
    dd = (lambda k: abs(real[k] - par[k]) if par else float("nan"))
    return {"local_ade": float(np.mean(d)), "local_max": float(np.max(d)),
            "d_min_dist": dd("min_dist"),               # spatial closeness (≈ΔDCPA)
            "d_conflict_angle": dd("conflict_angle"),   # ANGLE (deg)
            "d_closing_speed": dd("closing_speed"),     # SPEED / range-rate (m/s)
            "d_drac": dd("drac"),                       # ACCEL / evasive severity (m/s^2)
            "real_conflict_angle": real["conflict_angle"], "real_closing_speed": real["closing_speed"],
            "real_min_dist": real["min_dist"], "center_frame": float(center)}


def interaction_of_path(dataset, ego, actor, min_frame, max_frame, param_curve, frame_shift=0):
    """Interaction descriptors for a GIVEN param path (optionally time-shifted by frame_shift)
    vs the real ego. Used by the coverage/variance experiment to sweep offset & timing."""
    fps = paths.dataset_of(dataset)["fps"]
    a = _series(dataset, actor, min_frame, max_frame)
    e = _series(dataset, ego, min_frame, max_frame)
    if a is None or e is None:
        return None
    px, py = time_parameterize(param_curve, a["x"], a["y"])
    return _pair_interaction(a["frame"] + frame_shift, px, py, e["frame"], e["x"], e["y"], fps)
