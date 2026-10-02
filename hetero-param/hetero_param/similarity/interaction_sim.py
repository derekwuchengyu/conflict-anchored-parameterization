"""
互動相似 (INTERACTION similarity) — does the parameterized agent reproduce the real
agent↔ego interaction? Descriptors per pair, then real-vs-param deltas.

DESCRIPTORS (agent A, ego E, common integer-frame grid; fps from paths.DATASETS)
  PET        post-encroachment time at the conflict zone.
             estimator="bbox"  → Hetro_PET.pet_utils.calculate_pet (rotated-bbox
               conflict-zone entry/exit; the SAME estimator that produced the GT
               *_pet_optimized.parquet — headings are passed in the parquet's raw
               units to stay bit-consistent with that table).
             estimator="point" → circle of radius r around the conflict point:
               PET = (t_second_enter − t_first_exit); sign: >0 when A clears first.
  TTC        d(t) = ||p_A(t) − p_E(t)||,  ṙ(t) = −dd/dt (closing rate)
             TTC(t) = d(t)/ṙ(t) for ṙ(t) > eps;  min_ttc = min_t TTC(t)   (s)
  min_dist   min_t d(t) over the shared timeline (行進中最小距離) + its frame
  conflict point  spatial argmin over all path point pairs (crossing of the two
             PATHS, alignment-free; width-weighted like pet_utils)
  arrival state   at each vehicle's own closest sample to the conflict point:
             {arr_speed (m/s), arr_accel (m/s², signed longitudinal; decel<0),
              arr_heading (deg), arr_frame}

SIMILARITY = |Δ| of each descriptor between the REAL pair (real agent, real ego)
and the PARAM pair (parameterized agent, real ego); headings compared on the wrapped
circle (0..180°). PET/TTC deltas are NaN unless both sides have a finite value
(flags say which side had no conflict).

INPUT   two core.Traj — or a scenario key via compare()
OUTPUT  descriptors() → dict;  compare() → {real_*, param_*, d_*} flat dict (CSV-ready)
"""
from __future__ import annotations
import numpy as np

from .. import paths, config as C
from . import core

paths.add_import_paths()
import pet_utils  # noqa: E402  (Hetro_PET — read-only)


# ── alignment ────────────────────────────────────────────────────────────────

def _align(a: core.Traj, e: core.Traj):
    """Interpolate both trajectories onto the shared integer-frame grid.
    Returns (grid, ax, ay, ex, ey) or None if the overlap is < 3 frames."""
    lo, hi = max(a.frame.min(), e.frame.min()), min(a.frame.max(), e.frame.max())
    if hi - lo < 2:
        return None
    grid = np.arange(lo, hi + 1)
    return (grid,
            np.interp(grid, a.frame, a.x), np.interp(grid, a.frame, a.y),
            np.interp(grid, e.frame, e.x), np.interp(grid, e.frame, e.y))


# ── individual descriptors ───────────────────────────────────────────────────

def min_distance(agent: core.Traj, ego: core.Traj) -> dict:
    """min_t ||p_A(t) − p_E(t)|| over the shared timeline, and the frame it occurs."""
    al = _align(agent, ego)
    if al is None:
        return {"min_dist": float("nan"), "min_dist_frame": float("nan")}
    grid, ax, ay, ex, ey = al
    d = np.hypot(ax - ex, ay - ey)
    i = int(np.argmin(d))
    return {"min_dist": float(d[i]), "min_dist_frame": float(grid[i])}


def ttc(agent: core.Traj, ego: core.Traj, eps: float = 0.1) -> dict:
    """Point-mass time-to-collision: TTC(t) = d(t) / ṙ(t) where ṙ = −dd/dt > eps.
    min_ttc = inf when the pair is never closing faster than eps (no TTC event)."""
    al = _align(agent, ego)
    if al is None:
        return {"min_ttc": float("nan"), "ttc_frame": float("nan")}
    grid, ax, ay, ex, ey = al
    d = np.hypot(ax - ex, ay - ey)
    closing = -np.gradient(d) * agent.fps          # + = approaching (m/s)
    valid = closing > eps
    if not np.any(valid):
        return {"min_ttc": float("inf"), "ttc_frame": float("nan")}
    t = np.full_like(d, np.inf)
    t[valid] = d[valid] / closing[valid]
    i = int(np.argmin(t))
    return {"min_ttc": float(t[i]), "ttc_frame": float(grid[i])}


def conflict_point(agent: core.Traj, ego: core.Traj):
    """Spatial crossing of the two PATHS: argmin over all point pairs of
    ||a_i − e_j||. Conflict position is width-weighted between the two closest
    samples (same convention as pet_utils.calculate_pet). Returns
    (cxy (2,), i_agent, j_ego, min_path_dist)."""
    A, E = agent.xy, ego.xy
    d = np.linalg.norm(A[:, None, :] - E[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    wa = agent.width if np.isfinite(agent.width) and agent.width > 0 else 1.0
    we = ego.width if np.isfinite(ego.width) and ego.width > 0 else 1.0
    cxy = (A[i] * we + E[j] * wa) / (wa + we)
    return cxy, int(i), int(j), float(d[i, j])


def arrival_state(traj: core.Traj, cxy: np.ndarray) -> dict:
    """Kinematic state at the vehicle's own closest sample to the conflict point:
    speed (m/s), signed longitudinal accel (m/s², decel<0), heading (deg), frame."""
    i = int(np.argmin(np.hypot(traj.x - cxy[0], traj.y - cxy[1])))
    return {"arr_frame": float(traj.frame[i]), "arr_speed": float(traj.speed[i]),
            "arr_accel": float(traj.accel[i]), "arr_heading": float(traj.heading[i])}


def pet(agent: core.Traj, ego: core.Traj, estimator: str = "bbox",
        conflict_threshold: float = 1.4, point_radius: float | None = None,
        gate: str = "bbox") -> dict:
    """Post-encroachment time between agent and ego.

    bbox  — wraps Hetro_PET.pet_utils.calculate_pet: conflict zone = circle (radius =
            other vehicle's width) at each one's closest sample; entry/exit judged by
            the rotated vehicle bbox. Signed: >0 ⇔ agent clears before ego enters.
            NOTE headings are fed in the parquet's raw units (unconverted), exactly as
            the labeler did when producing *_pet_optimized.parquet, so param-side PET
            is estimator-identical to the GT table.
    point — enter/exit of a circle of radius r around each one's closest sample
            (r = other's width, or `point_radius`); PET from the same exit/enter logic.
    Returns {pet (signed, s; inf = no conflict), pet_abs, pet_type, min_path_dist}."""
    if estimator == "bbox":
        # gate="bbox" (v2, 2026-08-14): rescues center-distance-gate rejections
        # via swept-body-rectangle overlap (length/width/off-tracking aware);
        # previously-finite values are untouched. gate="center" = legacy v1.
        v, _f1, _f2, meta = pet_utils.calculate_pet(
            agent.xy, ego.xy, agent.frame.astype(int), ego.frame.astype(int),
            length1=agent.length, width1=agent.width, heading1=agent.heading,
            length2=ego.length, width2=ego.width, heading2=ego.heading,
            conflict_threshold=conflict_threshold, fps=agent.fps, gate=gate)
        v = float(v)
        return {"pet": v, "pet_abs": abs(v), "pet_type": meta.get("type") or meta.get("reason"),
                "min_path_dist": float(meta.get("min_distance", float("nan")))}

    if estimator != "point":
        raise ValueError(f"unknown PET estimator {estimator!r} (use 'bbox' or 'point')")
    cxy, i, j, dmin = conflict_point(agent, ego)
    if dmin > conflict_threshold:
        return {"pet": float("inf"), "pet_abs": float("inf"),
                "pet_type": "no_conflict_zone_entry", "min_path_dist": dmin}
    ra = point_radius or (ego.width if np.isfinite(ego.width) and ego.width > 0 else 2.0)
    re = point_radius or (agent.width if np.isfinite(agent.width) and agent.width > 0 else 2.0)
    za = np.hypot(agent.x - agent.x[i], agent.y - agent.y[i]) <= ra
    ze = np.hypot(ego.x - ego.x[j], ego.y - ego.y[j]) <= re
    if not za.any() or not ze.any():
        return {"pet": float("inf"), "pet_abs": float("inf"),
                "pet_type": "no_conflict_zone_entry", "min_path_dist": dmin}
    a_in, a_out = agent.frame[za].min(), agent.frame[za].max()
    e_in, e_out = ego.frame[ze].min(), ego.frame[ze].max()
    if a_out < e_in:                                   # agent clears first
        v, typ = (e_in - a_out) / agent.fps, "no_intersection"
    elif e_out < a_in:                                 # ego clears first
        v, typ = -(a_in - e_out) / agent.fps, "no_intersection"
    else:
        v, typ = 0.0, "time_overlap"                   # simultaneous occupancy
    return {"pet": float(v), "pet_abs": abs(float(v)), "pet_type": typ,
            "min_path_dist": dmin}


# ── bundle + similarity ──────────────────────────────────────────────────────

def descriptors(agent: core.Traj, ego: core.Traj, estimator: str = "bbox",
                ttc_eps: float = 0.1) -> dict:
    """All interaction descriptors for one (agent, ego) pair in one flat dict."""
    cxy, _i, _j, dmin = conflict_point(agent, ego)
    out = {"conflict_x": float(cxy[0]), "conflict_y": float(cxy[1]),
           "min_path_dist": dmin}
    out.update(min_distance(agent, ego))
    out.update(ttc(agent, ego, ttc_eps))
    out.update(pet(agent, ego, estimator))
    for k, v in arrival_state(agent, cxy).items():
        out[f"agent_{k}"] = v
    for k, v in arrival_state(ego, cxy).items():
        out[f"ego_{k}"] = v
    return out


def _wrap180(dd: float) -> float:
    dd = abs(dd) % 360.0
    return dd if dd <= 180.0 else 360.0 - dd


def _finite_delta(a: float, b: float) -> float:
    return abs(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def similarity(real_d: dict, param_d: dict, fps: float) -> dict:
    """Deltas between the real-pair and param-pair descriptors (lower = more similar)."""
    return {
        "d_pet": _finite_delta(real_d["pet_abs"], param_d["pet_abs"]),
        "pet_conflict_real": bool(np.isfinite(real_d["pet"])),
        "pet_conflict_param": bool(np.isfinite(param_d["pet"])),
        "d_min_ttc": _finite_delta(real_d["min_ttc"], param_d["min_ttc"]),
        "d_min_dist": _finite_delta(real_d["min_dist"], param_d["min_dist"]),
        "d_min_dist_frame_s": _finite_delta(real_d["min_dist_frame"],
                                            param_d["min_dist_frame"]) / fps,
        "d_arr_speed": _finite_delta(real_d["agent_arr_speed"], param_d["agent_arr_speed"]),
        "d_arr_accel": _finite_delta(real_d["agent_arr_accel"], param_d["agent_arr_accel"]),
        "d_arr_heading": _wrap180(real_d["agent_arr_heading"] - param_d["agent_arr_heading"])
        if np.isfinite(real_d["agent_arr_heading"]) and np.isfinite(param_d["agent_arr_heading"])
        else float("nan"),
        "d_arr_time_s": _finite_delta(real_d["agent_arr_frame"],
                                      param_d["agent_arr_frame"]) / fps,
    }


def compare(dataset: str, ego: int, actor: int, min_frame: int, max_frame: int,
            sample_cfg: C.SampleConfig | None = None, estimator: str = "bbox",
            end_offset: float = 0.0) -> dict | None:
    """Interaction similarity for one labeled scenario:
    REAL pair  = (real actor, real ego);  PARAM pair = (param actor, real ego).
    Returns {real_*, param_*, d_*} or None if a trajectory is unavailable."""
    a = core.real_traj(dataset, actor, min_frame, max_frame)
    e = core.real_traj(dataset, ego, min_frame, max_frame)
    p = core.param_traj(dataset, ego, actor, min_frame, max_frame,
                        sample_cfg, end_offset=end_offset)
    if a is None or e is None or p is None:
        return None
    rd = descriptors(a, e, estimator)
    pd_ = descriptors(p, e, estimator)
    out = {f"real_{k}": v for k, v in rd.items()}
    out.update({f"param_{k}": v for k, v in pd_.items()})
    out.update(similarity(rd, pd_, a.fps))
    return out
