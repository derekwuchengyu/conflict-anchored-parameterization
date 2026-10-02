"""
sweep.variance — how much the OUTCOME moves across the swept set, and which knob
moves it. Replay is a single point (zero variance everywhere); a useful
parameterization has controllable, non-degenerate spread.

SPREAD per descriptor (finite members only — inf PET/TTC = no-conflict sentinels are
excluded and reported via frac_finite; headings linearized first)
  n, mean, std, iqr, range      — via stats.spread (the repo's Exp-3 spread summary)
  cv = std / |mean|             — dimensionless spread (nan when |mean| ≈ 0, and nan
                                  for heading keys: linearization centres the mean
                                  near 0, so cv would be convention noise)

CONTROLLABILITY (main effects) per (parameter, descriptor)
  one-way variance decomposition on the grid — the share of descriptor variance
  explained by sweeping that parameter alone (grid analog of a Sobol first-order
  index / ANOVA η²):
      η²(p → y) = Var_over_levels( E[y | p = level] ) / Var(y)
  computed with population variances, level means weighted by level counts.
  η² ≈ 1: that knob alone drives the outcome; η² ≈ 0: the knob does nothing.
  GUARD: non-finite members (destroyed conflicts) are dropped per grid CELL, which
  can confound the one-way decomposition (a null-effect knob inherits another
  knob's variance through the correlated filtering). η² is therefore reported only
  when the surviving members still form a BALANCED design across the other swept
  axes — otherwise nan. The destruction pattern itself is surfaced as
  eta2_finite[param][key] = η²(param → 1{finite}) (which knob creates/destroys
  the conflict), reported for every key with frac_finite < 1.

INPUT   SweptSet (from sweep.core.sweep)
OUTPUT  report() → {spread: {key: {n, mean, std, iqr, range, cv, frac_finite}},
                    eta2: {param: {key: η²}}, eta2_finite: {param: {key: η²}}}
"""
from __future__ import annotations
import numpy as np

from ..stats import spread as _spread
from . import core

PARAM_NAMES = ("start_off", "end_off", "speed_factor", "shift_s")


def descriptor_spread(values: np.ndarray, heading: bool = False) -> dict:
    """stats.spread + cv + frac_finite for one descriptor over the swept set.
    Only finite members enter the spread (inf = no-conflict sentinel, not a value);
    cv is undefined (nan) for heading keys — linearization centres their mean at ~0."""
    v = np.asarray(values, float)
    if heading:
        v, _ = core.linearize_headings(v)
    finite = v[np.isfinite(v)]
    s = _spread(finite)
    s["cv"] = float(s["std"] / abs(s["mean"])) \
        if not heading and np.isfinite(s["std"]) and abs(s["mean"]) > 1e-9 else float("nan")
    s["frac_finite"] = float(len(finite) / len(v)) if len(v) else 0.0
    return s


def eta_squared(param_values: np.ndarray, y: np.ndarray) -> float:
    """η² = Var_levels(E[y|level]) / Var(y), population variances, finite members only."""
    p = np.asarray(param_values, float)
    v = np.asarray(y, float)
    m = np.isfinite(v)
    p, v = p[m], v[m]
    if len(v) < 2:
        return float("nan")
    tot = float(np.var(v))
    if tot < 1e-12:
        return 0.0                                  # degenerate outcome: nothing varies
    levels = np.unique(p)
    if len(levels) < 2:
        return float("nan")                         # parameter not actually swept
    between = sum(len(v[p == lv]) * (v[p == lv].mean() - v.mean()) ** 2 for lv in levels)
    return float(between / (len(v) * tot))


def _balanced_after_filter(pv: np.ndarray, others: list[np.ndarray],
                           mask: np.ndarray) -> bool:
    """True iff the finite members form a balanced design: every level of the target
    parameter retains the SAME multiset of other-parameter combinations. Cell-
    structured destruction (e.g. PET=inf exactly where end_off<0) breaks this and
    would let a null-effect knob absorb another knob's variance."""
    if not others:
        return True                                 # single swept factor: no confound
    combos_by_level = {}
    for lv in np.unique(pv):
        sel = mask & (pv == lv)
        combos_by_level[lv] = sorted(zip(*(o[sel] for o in others))) if sel.any() else []
    ref = next(iter(combos_by_level.values()))
    return all(c == ref for c in combos_by_level.values())


def controllability(swept: core.SweptSet, keys=None, params=None) -> tuple[dict, dict]:
    """(eta2, eta2_finite) for every swept parameter × descriptor (headings
    linearized). Parameter names default to whatever the variants carry (analytic:
    start_off/end_off/speed_factor/shift_s; esmini pipeline grids: the actual xosc
    parameter names). eta2[p][k] is nan when destroyed-conflict filtering leaves an
    unbalanced design (see _balanced_after_filter); eta2_finite[p][k] = η² of the
    finiteness indicator, reported for keys with any non-finite member."""
    keys = keys or swept.keys
    if params is None:
        params = swept.param_names()
    swept_params = [p for p in params
                    if len(np.unique(swept.param_values(p)[
                        np.isfinite(swept.param_values(p))])) >= 2]
    pv_all = {p: swept.param_values(p) for p in swept_params}
    out: dict = {}
    out_fin: dict = {}
    for pname in swept_params:
        pv = pv_all[pname]
        others = [pv_all[o] for o in swept_params if o != pname]
        row, row_fin = {}, {}
        for k in keys:
            y = swept.values(k)
            if core.is_heading_key(k):
                y, _ = core.linearize_headings(y)
            m = np.isfinite(y)
            if m.all():
                row[k] = eta_squared(pv, y)
            else:
                row_fin[k] = eta_squared(pv, m.astype(float))
                row[k] = eta_squared(pv[m], y[m]) \
                    if _balanced_after_filter(pv, others, m) else float("nan")
        out[pname] = row
        if row_fin:
            out_fin[pname] = row_fin
    return out, out_fin


def report(swept: core.SweptSet, keys=None) -> dict:
    """Full variance report for one swept set."""
    keys = keys or swept.keys
    eta2, eta2_finite = controllability(swept, keys)
    return {"spread": {k: descriptor_spread(swept.values(k),
                                            heading=core.is_heading_key(k))
                       for k in keys},
            "eta2": eta2, "eta2_finite": eta2_finite}
