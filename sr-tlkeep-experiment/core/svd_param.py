"""SVD-based scenario parameterization, following de Gelder et al. 2022
(arXiv:2202.12025), Sec. III.

Scenario vector (Eq. 1-2):
    x_i = [y_i ; theta_i] in R^{nx},  nx = nt*ny + ntheta
where y_i is the time series (here: agent x,y positions) resampled to nt
uniformly spaced time instants over [t0, t1], and theta_i holds additional
scalar parameters (here: scenario duration).

Weighting (Eq. 23): alpha_k = beta_k / std_k with
    beta_k = 1/sqrt(nt) for the nt*ny time-series entries,
    beta_k = 1          for the ntheta extra parameters.

Data matrix (Eq. 3-4):  X = [(alpha o x_1) - mu, ...] in R^{nx x N}
SVD (Eq. 5):            X = U S V^T
Reduced parameters (Eq. 10): v_i = first d entries of row i of V.
Reconstruction (Eq. 7): alpha o x ~= mu + sum_j sigma_j v_j u_j
"""
from __future__ import annotations

import numpy as np


def resample_traj(t: np.ndarray, ys: np.ndarray, nt: int) -> np.ndarray:
    """Resample a time series to nt uniformly spaced instants over [t0, t1].

    t  : (m,) monotonically increasing timestamps (seconds)
    ys : (m, ny) values (e.g. columns x, y)
    returns (nt, ny)
    """
    t = np.asarray(t, dtype=float)
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    if ys.shape[0] != t.shape[0]:
        ys = ys.T
    tq = np.linspace(t[0], t[-1], nt)
    out = np.empty((nt, ys.shape[1]))
    for k in range(ys.shape[1]):
        out[:, k] = np.interp(tq, t, ys[:, k])
    return out


def build_scenario_vectors(trajs: list, durations: list, nt: int) -> np.ndarray:
    """Stack scenario vectors x_i = [y_i(flattened time-major); duration].

    trajs     : list of (m_i, ny) arrays already in a common world frame
    durations : list of scenario durations in seconds (theta, ntheta=1)
    returns (N, nx) with nx = nt*ny + 1; time-major flattening, i.e.
    [x(t0), y(t0), x(t1), y(t1), ..., duration] matching Eq. 1 stacking of
    y(t0)...y(t1).
    """
    rows = []
    for tr, dur in zip(trajs, durations):
        rows.append(np.concatenate([np.asarray(tr, dtype=float).reshape(-1),
                                    [float(dur)]]))
    return np.asarray(rows)


class SvdParameterization:
    """Fit Eq. 3-7 on raw scenario vectors (N, nx)."""

    def __init__(self, nt: int, ny: int, ntheta: int):
        self.nt, self.ny, self.ntheta = nt, ny, ntheta

    def fit(self, X_raw: np.ndarray) -> "SvdParameterization":
        X_raw = np.asarray(X_raw, dtype=float)
        n_series = self.nt * self.ny
        assert X_raw.shape[1] == n_series + self.ntheta
        beta = np.concatenate([np.full(n_series, 1.0 / np.sqrt(self.nt)),
                               np.ones(self.ntheta)])
        std = X_raw.std(axis=0, ddof=0)
        std = np.where(std < 1e-12, 1.0, std)  # guard constant coordinates
        self.alpha = beta / std
        Xw = X_raw * self.alpha  # (N, nx), rows are alpha o x_i
        self.mu = Xw.mean(axis=0)
        Xc = (Xw - self.mu).T  # (nx, N) columns as in Eq. 3
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.U, self.s, self.V = U, s, Vt.T  # V rows are v_i (Eq. 10 source)
        self.N = X_raw.shape[0]
        return self

    def explained_variance(self, d: int) -> float:
        s2 = self.s ** 2
        return float(s2[:d].sum() / s2.sum())

    def reduced(self, d: int) -> np.ndarray:
        """(N, d) reduced parameters v~_i (Eq. 10)."""
        return self.V[:, :d].copy()

    def reconstruct(self, v: np.ndarray, d: int) -> np.ndarray:
        """Map reduced parameters back to raw scenario vectors (Eq. 7).

        v : (M, d) reduced parameter vectors
        returns (M, nx) raw (unweighted) scenario vectors
        """
        v = np.atleast_2d(np.asarray(v, dtype=float))
        Xw = self.mu + (v * self.s[:d]) @ self.U[:, :d].T
        return Xw / self.alpha

    def weighted(self, X_raw: np.ndarray) -> np.ndarray:
        """alpha o x for distance computations (Eq. 21 uses these)."""
        return np.asarray(X_raw, dtype=float) * self.alpha


def split_vector(x: np.ndarray, nt: int, ny: int, ntheta: int):
    """Inverse of build_scenario_vectors for a single raw vector."""
    y = np.asarray(x[: nt * ny]).reshape(nt, ny)
    theta = np.asarray(x[nt * ny:])
    return y, theta
