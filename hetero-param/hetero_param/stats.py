"""
Non-parametric statistics for thin, non-normal per-super-class samples:
bootstrap confidence intervals and permutation tests. Used for every headline number
(the paper reports CIs + permutation p-values, never a t-test on N=20).
"""
from __future__ import annotations
import numpy as np

_RNG = np.random.default_rng(20260718)


def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def bootstrap_ci(values, stat=np.median, n_boot: int = 10000, alpha: float = 0.05):
    """Percentile bootstrap CI for a statistic. Returns (point, lo, hi, n)."""
    a = _clean(values)
    if len(a) == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    idx = _RNG.integers(0, len(a), size=(n_boot, len(a)))
    boot = stat(a[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(stat(a)), float(lo), float(hi), int(len(a)))


def permutation_test(a, b, stat=np.median, n_perm: int = 10000, two_sided: bool = True):
    """Permutation test for a difference in `stat` between two independent samples.
    Returns (observed_diff, p_value)."""
    a, b = _clean(a), _clean(b)
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    obs = float(stat(a) - stat(b))
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        _RNG.shuffle(pooled)
        d = stat(pooled[:na]) - stat(pooled[na:])
        if (abs(d) >= abs(obs)) if two_sided else (d >= obs):
            count += 1
    return (obs, (count + 1) / (n_perm + 1))


def paired_permutation_test(diffs, n_perm: int = 10000, two_sided: bool = True):
    """Sign-flip permutation test on per-scenario paired differences.

    H0: the paired differences are symmetric about 0 (no systematic effect).
    Statistic: mean of the differences; each permutation flips the sign of every
    scenario's difference independently (the exact null for paired designs).
    Returns (observed_mean_diff, p_value) with the add-one correction, matching
    `permutation_test`'s conventions (module RNG, n_perm=10000 default).
    """
    d = _clean(diffs)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    obs = float(d.mean())
    signs = _RNG.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    perm = (signs * d).mean(axis=1)
    if two_sided:
        count = int(np.sum(np.abs(perm) >= abs(obs)))
    else:
        count = int(np.sum(perm >= obs))
    return (obs, (count + 1) / (n_perm + 1))


def spread(values) -> dict:
    """Summary spread used for the inD-vs-HetroD sensitivity contrast (Exp 3 main figure):
    how much an outcome swings across parameterization choices."""
    a = _clean(values)
    if len(a) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "iqr": float("nan"), "range": float("nan")}
    q25, q75 = np.percentile(a, [25, 75])
    return {"n": int(len(a)), "mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0),
            "iqr": float(q75 - q25), "range": float(a.max() - a.min())}
