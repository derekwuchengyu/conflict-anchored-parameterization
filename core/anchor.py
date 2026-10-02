"""Conflict anchor on the target trajectory: PET, trajectory crossing, or minimum distance.

Each function takes the ego and target window tracks (core.io.window) and
returns (anchor_frame, kind), where anchor_frame is a frame of the target track.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pet as PET

METHODS = ("pet", "cross", "min_dist")


def _vehicle(g: pd.DataFrame):
    return (g.frame.to_numpy(), g[["x", "y"]].to_numpy(float), g.heading.to_numpy(float),
            float(g.length.iloc[0]), float(g.width.iloc[0]))


def _target_frame_near(target, frame):
    """Frame of the target sample nearest to `frame`."""
    return int(target.frame.iloc[int(np.abs(target.frame.to_numpy(float) - frame).argmin())])


def pet_anchor(ego, target, fps=PET.FPS, conflict_threshold=PET.CONFLICT_THRESHOLD):
    """Target position at the PET boundary frame (pet_frame1).

    Falls back to the minimum simultaneous distance when PET is undefined."""
    res = PET.pet_window(*_vehicle(ego), *_vehicle(target),
                         conflict_threshold=conflict_threshold, fps=fps)
    anc = PET.actor_anchor(res, actor_is=2)
    if anc["anchor"] == "none":
        return None, "none"
    return _target_frame_near(target, anc["crit_frame"]), anc["anchor"]


def cross_anchor(ego, target):
    """Trajectory crossing: the target sample closest to the ego path (any time)."""
    if len(ego) < 2 or len(target) < 2:
        return None, "none"
    axy, exy = target[["x", "y"]].to_numpy(float), ego[["x", "y"]].to_numpy(float)
    dmin = np.linalg.norm(axy[:, None, :] - exy[None, :, :], axis=2).min(axis=1)
    return int(target.frame.iloc[int(np.argmin(dmin))]), "cross"


def min_dist_anchor(ego, target):
    """Minimum simultaneous distance between ego and target."""
    m = target.merge(ego[["frame", "x", "y"]], on="frame", suffixes=("", "_ego"))
    if m.empty:
        return None, "none"
    d = np.hypot(m.x - m.x_ego, m.y - m.y_ego)
    return int(m.frame.iloc[int(np.argmin(d.to_numpy()))]), "min_dist"


def find_anchor(method, ego, target, fps=PET.FPS, conflict_threshold=PET.CONFLICT_THRESHOLD):
    if method == "pet":
        return pet_anchor(ego, target, fps, conflict_threshold)
    if method == "cross":
        return cross_anchor(ego, target)
    if method == "min_dist":
        return min_dist_anchor(ego, target)
    raise ValueError(f"unknown anchor method {method!r}; choose from {METHODS}")
