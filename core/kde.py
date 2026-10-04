"""Gaussian KDE over the parameter vectors (theta1, theta2, v_end).

Per-column standardization, scalar bandwidth from leave-one-out
cross-validation, and dependent sampling (uniform kernel center + N(0, h^2 I)),
following de Gelder et al. (2022).
"""
from __future__ import annotations

import numpy as np


def loo_log_likelihood(V: np.ndarray, h: float) -> float:
    """Leave-one-out log-likelihood of the KDE at bandwidth h."""
    N, d = V.shape
    D2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    K = np.exp(-D2 / (2.0 * h * h))
    np.fill_diagonal(K, 0.0)
    dens = K.sum(axis=1) / ((N - 1) * (2.0 * np.pi) ** (d / 2.0) * h ** d)
    return float(np.log(np.maximum(dens, 1e-300)).sum())


def loo_bandwidth(V: np.ndarray, n_grid: int = 60) -> float:
    """Grid search + golden-section refinement of the LOO-CV bandwidth."""
    N, d = V.shape
    sig = V.std(axis=0).mean()
    h0 = sig * (4.0 / ((d + 2) * N)) ** (1.0 / (d + 4))
    hs = np.geomspace(h0 / 20.0, h0 * 20.0, n_grid)
    lls = np.array([loo_log_likelihood(V, h) for h in hs])
    i = int(np.argmax(lls))
    lo, hi = hs[max(i - 1, 0)], hs[min(i + 1, n_grid - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)
    c, dd = b - gr * (b - a), a + gr * (b - a)
    fc, fd = loo_log_likelihood(V, np.exp(c)), loo_log_likelihood(V, np.exp(dd))
    for _ in range(40):
        if fc > fd:
            b, dd, fd = dd, c, fc
            c = b - gr * (b - a)
            fc = loo_log_likelihood(V, np.exp(c))
        else:
            a, c, fc = c, dd, fd
            dd = a + gr * (b - a)
            fd = loo_log_likelihood(V, np.exp(dd))
    return float(np.exp((a + b) / 2.0))


class ParamKDE:
    def __init__(self, params, cols):
        self.raw = np.asarray(params, float)
        self.cols = list(cols)
        self.m = self.raw.mean(0)
        self.sd = np.where(self.raw.std(0) < 1e-12, 1.0, self.raw.std(0))
        self.Z = (self.raw - self.m) / self.sd
        self.N, self.p = self.Z.shape
        self.h = loo_bandwidth(self.Z)

    def to_raw(self, Z):
        return np.atleast_2d(Z) * self.sd + self.m

    def dependent(self, M, rng, c=1.0):
        """M draws; returns (samples (M, p), kernel-center indices (M,))."""
        idx = rng.integers(0, self.N, size=M)
        return self.to_raw(self.Z[idx] + c * self.h * rng.standard_normal((M, self.p))), idx

    def conditional(self, i, M, rng, c=1.0):
        """M draws around one kernel center i."""
        Z = self.Z[i] + c * self.h * rng.standard_normal((M, self.p))
        return self.to_raw(Z), np.full(M, i)
