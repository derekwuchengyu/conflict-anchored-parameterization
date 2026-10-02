"""
hetero_param.sweep — set-level properties of a parameterization: render the parameter
grid once (a SET of concrete scenarios), then measure what the set is worth.

  core.py        SweepAxes (start/end lateral offset × speed factor × time shift,
                 defaults = the real CARLA param.xosc knobs at {min,default,max}),
                 sweep() → SweptSet of rendered Variants + descriptors + replay point
  coverage.py    the covered REGION per descriptor + spatially, and whether the real
                 scenario lies inside it (replay covers a single point)
  variance.py    spread of every descriptor across the set + η² controllability
                 (which knob moves which outcome; replay: zero variance)
  redundancy.py  near-duplicate structure: diversity, nearest-neighbour distances,
                 greedy ε-cover effective count, redundancy = 1 − n_eff/n

Quick use (from hetero-param/, nps env):
  from hetero_param import sweep as SW
  s = SW.sweep("HetroD", ego=301, actor=303, min_frame=4509, max_frame=4836)
  SW.coverage(s)      # or SW.coverage("HetroD", 301, 303, 4509, 4836)
  SW.variance(s)
  SW.redundancy(s)
  SW.properties("HetroD", 301, 303, 4509, 4836)   # all three, one render
"""
from __future__ import annotations

from . import core, coverage as _coverage, variance as _variance, redundancy as _redundancy
from .core import SweepAxes, SweptSet, Variant, DESCRIPTOR_KEYS, sweep  # noqa: F401

# NOTE the package attributes `coverage` / `variance` / `redundancy` are the facade
# FUNCTIONS below (the documented API). The submodules stay importable via these
# explicit aliases (attribute access hetero_param.sweep.coverage.report would
# otherwise hit the function):
coverage_mod, variance_mod, redundancy_mod = _coverage, _variance, _redundancy


def _as_swept(arg, *rest, **kw) -> core.SweptSet | None:
    """Accept either a prebuilt SweptSet or (dataset, ego, actor, min_frame, max_frame)."""
    if isinstance(arg, core.SweptSet):
        return arg
    return sweep(arg, *rest, **kw)


def coverage(swept_or_dataset, *args, keys=None, **kw) -> dict | None:
    s = _as_swept(swept_or_dataset, *args, **kw)
    return _coverage.report(s, keys=keys) if s else None


def variance(swept_or_dataset, *args, keys=None, **kw) -> dict | None:
    s = _as_swept(swept_or_dataset, *args, **kw)
    return _variance.report(s, keys=keys) if s else None


def redundancy(swept_or_dataset, *args, mode: str = "descriptor",
               eps: float | None = None, keys=None, **kw) -> dict | None:
    s = _as_swept(swept_or_dataset, *args, **kw)
    return _redundancy.report(s, mode=mode, eps=eps, keys=keys) if s else None


def properties(swept_or_dataset, *args, keys=None, redundancy_mode: str = "descriptor",
               redundancy_eps: float | None = None, **kw) -> dict | None:
    """Coverage + variance + redundancy from ONE rendered grid."""
    s = _as_swept(swept_or_dataset, *args, **kw)
    if s is None:
        return None
    return {"coverage": _coverage.report(s, keys=keys),
            "variance": _variance.report(s, keys=keys),
            "redundancy": _redundancy.report(s, mode=redundancy_mode,
                                             eps=redundancy_eps, keys=keys),
            "swept": s}
