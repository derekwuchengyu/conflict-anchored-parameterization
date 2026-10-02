"""Gaussian KDE with scalar bandwidth (H = h^2 I_d) and *dependent* sampling,
following de Gelder et al. 2022, Sec. III-C.

- Bandwidth h selected by leave-one-out cross-validation (maximising the LOO
  log-likelihood, which minimises the KL divergence to the true pdf).
- Sampling: draw i ~ Uniform{1..N}, then v ~ N(v~_i, h^2 I_d).  Because the
  kernel centres are the joint reduced parameter vectors, correlations between
  the d parameters are retained ("dependent" sampling).
"""
from __future__ import annotations

import numpy as np


def loo_log_likelihood(V: np.ndarray, h: float) -> float:
    """Leave-one-out log-likelihood of the KDE at bandwidth h."""
    N, d = V.shape
    D2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)  # (N, N)
    K = np.exp(-D2 / (2.0 * h * h))
    np.fill_diagonal(K, 0.0)
    dens = K.sum(axis=1) / ((N - 1) * (2.0 * np.pi) ** (d / 2.0) * h ** d)
    dens = np.maximum(dens, 1e-300)
    return float(np.log(dens).sum())


def loo_bandwidth(V: np.ndarray, n_grid: int = 60) -> float:
    """Grid + golden-section refinement of the LOO-CV bandwidth."""
    N, d = V.shape
    # Silverman-style bracket around a plug-in estimate
    sig = V.std(axis=0).mean()
    h0 = sig * (4.0 / ((d + 2) * N)) ** (1.0 / (d + 4))
    hs = np.geomspace(h0 / 20.0, h0 * 20.0, n_grid)
    lls = np.array([loo_log_likelihood(V, h) for h in hs])
    i = int(np.argmax(lls))
    lo = hs[max(i - 1, 0)]
    hi = hs[min(i + 1, n_grid - 1)]
    # golden-section on log h
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


def sample_dependent(V: np.ndarray, h: float, n_samples: int,
                     rng: np.random.Generator):
    """Dependent KDE sampling. Returns (samples (n,d), centre_idx (n,))."""
    N, d = V.shape
    idx = rng.integers(0, N, size=n_samples)
    samples = V[idx] + h * rng.standard_normal((n_samples, d))
    return samples, idx
