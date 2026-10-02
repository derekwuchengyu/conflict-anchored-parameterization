"""
sweep.coverage — the REGION the swept set covers, and whether the real scenario lies
inside it. Region-based coverage in the SUNRISE taxonomy (a covered region + a
real-point-inside test; cf. formulas #38/#41 and the de Gelder SCDB coverage family),
NOT a probability-mass coverage — no real-world parameter distribution is consumed.

PER-DESCRIPTOR AXIS COVERAGE (values y_i over the swept set, real value y*)
  range        [min_i y_i, max_i y_i]                    (finite members only)
  covers_real  min ≤ y* ≤ max
  position q   (y* − min) / (max − min) ∈ [0,1] when covered — WHERE the real value
               sits in the covered range (0.5 = centered; needs range > 0)
  margin       min(y* − min, max − y*) — distance to the nearest edge (signed:
               negative = how far outside)
  frac_finite  share of variants with a finite value (PET/TTC can be destroyed)
Headings are linearized around the set's circular mean before any of the above.

SPATIAL ENVELOPE (same construction as the legacy flat coverage.py grid_coverage)
  at K stations along the real agent path: envelope_width = median over stations of
  max_i(nearest distance of variant i); gt_covered_frac = share of stations where
  some variant passes within `tol` (default 0.5 m).

2-D REGION  hull_area = convex-hull area of the (min_dist, closing_speed) cloud —
  the legacy criticality-plane spread; 0 for replay (a single point).

INPUT   SweptSet (from sweep.core.sweep)
OUTPUT  report() → {axes: {key: {...}}, covers_real_all, n_covered_axes,
                    hull_area, envelope_width, gt_covered_frac, n_variants}
"""
from __future__ import annotations
import numpy as np

from . import core


def axis_coverage(values: np.ndarray, real: float, heading: bool = False) -> dict:
    """Coverage of ONE descriptor axis by the swept set vs the real value."""
    v = np.asarray(values, float)
    if heading:
        v, real = core.linearize_headings(v, real)
    f = v[np.isfinite(v)]
    out = {"n": int(len(v)), "frac_finite": float(len(f) / len(v)) if len(v) else 0.0,
           "min": float("nan"), "max": float("nan"), "range": float("nan"),
           "real": float(real) if np.isfinite(real) else float("nan"),
           "covers_real": float("nan"), "position": float("nan"), "margin": float("nan")}
    if len(f) == 0:
        return out
    lo, hi = float(f.min()), float(f.max())
    out.update(min=lo, max=hi, range=hi - lo)
    if not np.isfinite(real):
        return out
    out["covers_real"] = float(lo <= real <= hi)
    out["margin"] = float(min(real - lo, hi - real))
    if hi - lo > 1e-12:
        out["position"] = float((real - lo) / (hi - lo))
    return out


def hull_area(swept: core.SweptSet, kx: str = "min_dist", ky: str = "closing_speed") -> float:
    """Convex-hull area of the swept cloud in the (kx, ky) criticality plane."""
    x, y = swept.values(kx), swept.values(ky)
    m = np.isfinite(x) & np.isfinite(y)
    pts = np.column_stack([x[m], y[m]])
    if len(pts) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)      # 2-D input: .volume = polygon area
    except Exception:
        return 0.0


def spatial_envelope(swept: core.SweptSet, n_stations: int = 20, tol: float = 0.5) -> dict:
    """Does the fan of variant paths spatially cover the real agent path?"""
    real = swept.agent.xy if swept.agent is not None else None
    if real is None or len(real) < 2 or not swept.variants:
        return {"envelope_width": float("nan"), "gt_covered_frac": float("nan")}
    idx = np.linspace(0, len(real) - 1, n_stations).astype(int)
    widths, inside = [], 0
    for g in real[idx]:
        near = [float(np.min(np.linalg.norm(v.traj.xy - g, axis=1)))
                for v in swept.variants]
        widths.append(max(near))
        if min(near) < tol:
            inside += 1
    return {"envelope_width": float(np.median(widths)),
            "gt_covered_frac": inside / n_stations}


def report(swept: core.SweptSet, keys=None) -> dict:
    """Full coverage report for one swept set."""
    keys = keys or swept.keys
    axes = {k: axis_coverage(swept.values(k), swept.real.get(k, float("nan")),
                             heading=core.is_heading_key(k)) for k in keys}
    covered = [a["covers_real"] for a in axes.values() if np.isfinite(a["covers_real"])]
    return {"axes": axes,
            "n_variants": len(swept.variants),
            "n_covered_axes": int(sum(covered)),
            "n_scored_axes": len(covered),
            "covers_real_all": bool(covered) and all(c == 1.0 for c in covered),
            "hull_area": hull_area(swept),
            **spatial_envelope(swept)}
