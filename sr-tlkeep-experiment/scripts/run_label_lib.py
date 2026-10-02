"""Shared helpers for the hypothesis-experiment scripts (30_*+).
Kept in sync with scripts/20_run_label.py (same conventions)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FPS = 30.0
NT, NY, NTH = 50, 2, 1
TRIM_SPEED, TRIM_MAX_S = 0.05, 0.6


def circ_mean_deg(a):
    r = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))) % 360.0


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def trim_lead_still(g: pd.DataFrame) -> pd.DataFrame:
    sp = g.speed.values
    mv = np.flatnonzero(sp > TRIM_SPEED)
    if len(mv) == 0 or mv[0] == 0:
        return g
    t = (g.frame.values - g.frame.values[0]) / FPS
    return g if t[mv[0]] > TRIM_MAX_S else g.iloc[mv[0]:]


def arc_resample(x, y, k=50):
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-9:
        return np.stack([np.full(k, x[0]), np.full(k, y[0])], 1)
    sq = np.linspace(0.0, s[-1], k)
    return np.stack([np.interp(sq, s, x), np.interp(sq, s, y)], 1)


def dtw_row(Ri: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Length-normalised 2-D DTW of one path (K,2) vs a batch (Ng,K,2)."""
    Ng, K, _ = G.shape
    C = np.linalg.norm(Ri[None, :, None, :] - G[:, None, :, :], axis=3)
    D = np.full((Ng, K + 1, K + 1), np.inf)
    D[:, 0, 0] = 0.0
    for s in range(2, 2 * K + 1):
        a0, a1 = max(1, s - K), min(K, s - 1)
        A = np.arange(a0, a1 + 1)
        B = s - A
        prev = np.minimum(np.minimum(D[:, A - 1, B - 1], D[:, A - 1, B]),
                          D[:, A, B - 1])
        D[:, A, B] = C[:, A - 1, B - 1] + prev
    return D[:, K, K] / (2.0 * K)
