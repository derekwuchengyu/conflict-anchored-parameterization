"""Two path-deflection angles (theta1, theta2) around the conflict anchor.

Construction (frozen per scenario): q_in, the incoming direction u1, and the
chord lengths L1 = |q_c - q_in|, L2 = |q_out - q_c|.
    theta1 = signed angle from u1 to (q_c - q_in)
    theta2 = signed angle from (q_c - q_in) to (q_out - q_c)
Inverse: q_c = q_in + L1 R(theta1) u1,  q_out = q_c + L2 R(theta1 + theta2) u1.
"""
from __future__ import annotations

import numpy as np


def signed_angle(u, v):
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)))


def rot(u, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def arc_resample(x, y, k=50):
    dd = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(dd)])
    if s[-1] <= 1e-9:
        return np.stack([np.full(k, x[0]), np.full(k, y[0])], 1)
    sq = np.linspace(0.0, s[-1], k)
    return np.stack([np.interp(sq, s, x), np.interp(sq, s, y)], 1)


def theta_geometry(cps, target):
    """cps: (3, 2) [q_in, q_c, q_out]; target: window track (columns x, y).

    The incoming direction u1 points from the first target sample to q_in, or
    follows the start heading when q_in is within 1 m of the start.
    Returns dict(pm, u1, L1, L2, theta1, theta2, u1_source), or None if a chord < 1 m."""
    xy = np.asarray(cps, float).reshape(3, 2)
    init = np.array([target.x.values[0], target.y.values[0]])
    v1 = xy[0] - init
    L0 = float(np.hypot(*v1))
    if L0 >= 1.0:
        u1, src = v1 / L0, "init_pt"
    else:
        arc = arc_resample(target.x.values, target.y.values, 200)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(arc, axis=0).T))])
        k = int(np.searchsorted(s, min(2.0, s[-1] * 0.5)))
        d = arc[max(k, 1)] - arc[0]
        u1, src = d / (np.hypot(*d) + 1e-9), "start_heading"
    v2, v3 = xy[1] - xy[0], xy[2] - xy[1]
    L1, L2 = float(np.hypot(*v2)), float(np.hypot(*v3))
    if min(L1, L2) < 1.0:
        return None
    u2, u3 = v2 / L1, v3 / L2
    return dict(pm=xy[0], u1=u1, L1=L1, L2=L2, theta1=signed_angle(u1, u2),
                theta2=signed_angle(u2, u3), u1_source=src)


def theta_to_cps(geom, theta1, theta2):
    """Control points [q_in, q_c, q_out] (flat, length 6) for given angles."""
    pm, u1 = np.asarray(geom["pm"], float), np.asarray(geom["u1"], float)
    pc = pm + geom["L1"] * rot(u1, theta1)
    pp = pc + geom["L2"] * rot(u1, theta1 + theta2)
    return np.concatenate([pm, pc, pp])
