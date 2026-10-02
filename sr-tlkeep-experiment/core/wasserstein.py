"""Empirical Wasserstein metric (Eq. 14-21) and the Scenario
Representativeness (SR) metric (Eq. 22) of de Gelder et al. 2022.

W~_p(Z, W) = ( inf_T sum_ij Delta(z_i, w_j)^p T_ij )^{1/p}
with uniform marginals 1/Nz, 1/Nw and
Delta(z, w) = || (alpha o z) - (alpha o w) ||_2  (Eq. 21).

SR metric: M_p(W, Z, X) = W~_p(Z,W) + beta * ( W~_p(Z,W) - W~_p(X,W) ).
Following the paper's case study we use p = 1, beta = 0.25, and report the
median over `n_repeats` random (1 - test_frac)/test_frac partitions.
"""
from __future__ import annotations

import numpy as np
import ot


def empirical_wasserstein(Zw: np.ndarray, Ww: np.ndarray, p: int = 1) -> float:
    """W~_p between two sets of ALREADY alpha-weighted vectors.

    Zw : (Nz, nx), Ww : (Nw, nx)
    """
    Zw = np.ascontiguousarray(Zw, dtype=np.float64)
    Ww = np.ascontiguousarray(Ww, dtype=np.float64)
    M = ot.dist(Zw, Ww, metric="euclidean")
    if p != 1:
        M = M ** p
    a = np.full(Zw.shape[0], 1.0 / Zw.shape[0])
    b = np.full(Ww.shape[0], 1.0 / Ww.shape[0])
    cost = ot.emd2(a, b, M, numItermax=1_000_000)
    return float(cost ** (1.0 / p))


def sr_metric(Xw_all: np.ndarray, Ww: np.ndarray, beta: float = 0.25,
              p: int = 1, n_repeats: int = 200, test_frac: float = 0.2,
              seed: int = 0):
    """SR metric M_p with a FIXED generated set Ww.

    Xw_all : (N, nx) alpha-weighted real scenario vectors; each repeat is a
             random split into training X (80%) and test Z (20%).
    Returns dict with medians and the per-repeat arrays.
    """
    rng = np.random.default_rng(seed)
    N = Xw_all.shape[0]
    n_test = max(1, int(round(N * test_frac)))
    m_list, wzw_list, wxw_list = [], [], []
    for _ in range(n_repeats):
        perm = rng.permutation(N)
        Z = Xw_all[perm[:n_test]]
        X = Xw_all[perm[n_test:]]
        wzw = empirical_wasserstein(Z, Ww, p)
        wxw = empirical_wasserstein(X, Ww, p)
        m_list.append(wzw + beta * (wzw - wxw))
        wzw_list.append(wzw)
        wxw_list.append(wxw)
    m = np.array(m_list)
    return {
        "M_p_median": float(np.median(m)),
        "W_ZW_median": float(np.median(wzw_list)),
        "W_XW_median": float(np.median(wxw_list)),
        "penalty_median": float(np.median(np.array(wzw_list) - np.array(wxw_list))),
        "M_p_all": m,
        "W_ZW_all": np.array(wzw_list),
        "W_XW_all": np.array(wxw_list),
    }
