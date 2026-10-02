"""Circular conflict window of radius L centered on the spatial anchor q_c.

Walking the target track backward and forward in time from the anchor, the
entry point q_in and exit point q_out are the first points at Euclidean
distance L from q_c, linearly interpolated on the crossing segment so that
both chords equal L exactly.
"""
from __future__ import annotations

import numpy as np


def crossing(xy, frames, i_anchor, direction, L):
    """First point at distance L from xy[i_anchor] walking in `direction` (-1 / +1).

    Returns (point, interpolated_frame, outside_index), or None if never reached."""
    qc = xy[i_anchor]
    d = np.hypot(*(xy - qc).T)
    order = range(i_anchor - 1, -1, -1) if direction < 0 else range(i_anchor + 1, len(xy))
    prev = i_anchor
    for i in order:
        if d[i] >= L:
            a, b = xy[prev], xy[i]
            v, w = b - a, a - qc
            A, B, C = float(v @ v), float(2 * (v @ w)), float(w @ w - L * L)
            if A <= 0:
                prev = i
                continue
            disc = max(B * B - 4 * A * C, 0.0)
            s = (-B + np.sqrt(disc)) / (2 * A)
            s = float(min(max(s, 0.0), 1.0))
            return a + s * v, float(frames[prev] + s * (frames[i] - frames[prev])), int(i)
        prev = i
    return None


def disk_points(target, anchor_frame, L):
    """target: window track with columns frame, x, y. Returns a dict with status
    'ok' and q_in (qm_*), q_c (qc_*), q_out (qp_*), or a failure status."""
    frames = target.frame.to_numpy(float)
    xy = target[["x", "y"]].to_numpy(float)
    hit = np.flatnonzero(frames == anchor_frame)
    if len(hit) != 1:
        return dict(status="no_anchor_sample")
    i = int(hit[0])
    back = crossing(xy, frames, i, -1, L)
    fwd = crossing(xy, frames, i, +1, L)
    res = dict(qc_x=float(xy[i, 0]), qc_y=float(xy[i, 1]), anchor_index=i)
    if back is None and fwd is None:
        res["status"] = "both_unavailable"
    elif back is None:
        res["status"] = "q_in_unavailable"
    elif fwd is None:
        res["status"] = "q_out_unavailable"
    else:
        res["status"] = "ok"
    if back is not None:
        res.update(qm_x=float(back[0][0]), qm_y=float(back[0][1]), t_in_frame=back[1])
    if fwd is not None:
        res.update(qp_x=float(fwd[0][0]), qp_y=float(fwd[0][1]), t_out_frame=fwd[1])
    return res
