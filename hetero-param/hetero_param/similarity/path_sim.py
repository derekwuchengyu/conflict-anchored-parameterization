"""
路徑相似 (PATH similarity) — how close is the parameterized path to the real one, as a
curve in space. Wraps parameterization_fidelity.py (read-only) so the numbers are
identical to the rest of the repo.

FORMULAS (real path R = {r_i}, param path P = {p_j}, both (·,2) in metres)
  ADE   arc-length parameterized: P is re-sampled at R's cumulative arc-length
        fractions, then ADE = mean_i || r_i − P(s_i) ||          (PF.compute_ade_fde)
  FDE   || r_N − P(s_N) || — same parameterization, final point
  DTW   2-D dynamic time warping, Euclidean point cost, optimal monotone alignment,
        total cost normalized by (n+m)                            (PF.compute_dtw)
  path_dev  max(OWD(R→P), OWD(P→R)) — symmetric mean nearest-point deviation;
        alignment-free and speed-independent (repo headline metric)

INPUT   two (N,2)/(M,2) xy arrays — or a scenario key via compare()
OUTPUT  dict {ade, fde, dtw, path_dev, n_real, n_param}  (all metres; lower = more similar)
"""
from __future__ import annotations
import numpy as np

from .. import paths
from . import core

paths.add_import_paths()
import parameterization_fidelity as PF  # noqa: E402


def ade(real_xy: np.ndarray, param_xy: np.ndarray) -> float:
    """Arc-length-parameterized Average Displacement Error (m)."""
    return float(PF.compute_ade_fde(np.asarray(real_xy), np.asarray(param_xy))[0])


def fde(real_xy: np.ndarray, param_xy: np.ndarray) -> float:
    """Final Displacement Error (m), same arc-length parameterization as ade()."""
    return float(PF.compute_ade_fde(np.asarray(real_xy), np.asarray(param_xy))[1])


def dtw(real_xy: np.ndarray, param_xy: np.ndarray, subsample: int = 1) -> float:
    """Length-normalized 2-D DTW distance (m). `subsample` thins the param path
    (dense 250-sample NURBS) for speed — 5 matches parampath.geom_fidelity."""
    p = np.asarray(param_xy)[::max(1, subsample)]
    return float(PF.compute_dtw(np.asarray(real_xy), p))


def path_dev(real_xy: np.ndarray, param_xy: np.ndarray) -> float:
    """Symmetric mean nearest-point deviation max(OWD(a→b), OWD(b→a)) (m)."""
    a, b = np.asarray(real_xy), np.asarray(param_xy)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(max(PF.compute_owd(a, b), PF.compute_owd(b, a)))


def similarity(real_xy: np.ndarray, param_xy: np.ndarray, dtw_subsample: int = 5) -> dict:
    """All path-similarity metrics for one pair of curves."""
    real_xy, param_xy = np.asarray(real_xy), np.asarray(param_xy)
    if len(real_xy) < 2 or len(param_xy) < 2:
        return {"ade": float("nan"), "fde": float("nan"), "dtw": float("nan"),
                "path_dev": float("nan"), "n_real": int(len(real_xy)),
                "n_param": int(len(param_xy))}
    a, f = PF.compute_ade_fde(real_xy, param_xy)
    return {"ade": float(a), "fde": float(f),
            "dtw": dtw(real_xy, param_xy, dtw_subsample),
            "path_dev": path_dev(real_xy, param_xy),
            "n_real": int(len(real_xy)), "n_param": int(len(param_xy))}


def compare(dataset: str, ego: int, actor: int, min_frame: int, max_frame: int,
            sample_cfg=None, end_offset: float = 0.0) -> dict | None:
    """Path similarity of (real actor path) vs (parameterized NURBS path) for one
    labeled scenario. None if either trajectory is unavailable."""
    real = core.real_traj(dataset, actor, min_frame, max_frame)
    par = core.param_traj(dataset, ego, actor, min_frame, max_frame,
                          sample_cfg, end_offset=end_offset)
    if real is None or par is None:
        return None
    return similarity(real.xy, par.xy)
