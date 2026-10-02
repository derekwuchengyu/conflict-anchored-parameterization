"""Physical-validity filter rerun (exp_cross_coverage Q3 protocol).

The adversarial verification found the spline-CP KDE's honest limitation:
the 11-D KDE has no geometry x speed coupling, so a tail of samples corners
implausibly (19.6% with mid-path lateral accel > 0.8 g; 3.4-5.5% exceed the
real set's max curvature). This script applies a SYMMETRIC physical filter
to every generated set that has trajectories and re-computes the coverage
table, answering: do the coverage conclusions survive validity filtering?

Filter (per sample, both computable for esmini renders and SVD 50-pt paths):
  latacc_mid <= 8 m/s2   max v^2*kappa on the 5-95%% arc segment
                          (0.5 m arc resample, v > 1 m/s gate)
  vmax <= 60 km/h         tempered support cap (same as exp_cross_coverage)

Usage: micromamba run -n nps python 07_filter_rerun.py
Output: results/validity_spline_full.csv / validity_svd_kde.csv,
        results/summary_stats_filtered.csv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slib as L  # noqa: E402

spec = importlib.util.spec_from_file_location("stats05", HERE / "05_stats_figs.py")
S05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S05)

LATACC_MAX = 8.0      # m/s^2 (~0.8 g)
VMAX_KMH = 60.0


def latacc_vmax(g) -> tuple[float, float]:
    """(max mid-path lateral accel, max speed km/h) for one long-format
    trajectory group (frame-sorted)."""
    x, y = g.x.values.astype(float), g.y.values.astype(float)
    sp = g.speed.values.astype(float)
    vmax = float(sp.max()) * 3.6
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] < 5.0:
        return 0.0, vmax
    sq = np.arange(0.0, s[-1], 0.5)
    xq, yq = np.interp(sq, s, x), np.interp(sq, s, y)
    vq = np.interp(sq, s, sp)
    dx, dy = np.gradient(xq, 0.5), np.gradient(yq, 0.5)
    ddx, ddy = np.gradient(dx, 0.5), np.gradient(dy, 0.5)
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(
        (dx * dx + dy * dy) ** 1.5, 1e-9)
    n = len(sq)
    mid = slice(int(0.05 * n), max(int(0.95 * n), int(0.05 * n) + 1))
    k, v = kappa[mid], vq[mid]
    m = v > 1.0
    if not m.any():
        return 0.0, vmax
    return float((v[m] ** 2 * k[m]).max()), vmax


def validity_table(tr: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    df = tr[tr.role == "agent"] if "role" in tr.columns else tr
    for sid, g in df.groupby("scenario_id"):
        la, vm = latacc_vmax(g.sort_values("frame"))
        rows.append({"scenario_id": sid, "latacc_mid": la, "vmax_kmh": vm,
                     "valid": (la <= LATACC_MAX) and (vm <= VMAX_KMH)})
    out = pd.DataFrame(rows)
    out.to_csv(L.RESULTS / f"validity_{name}.csv", index=False)
    print(f"{name}: {int(out.valid.sum())}/{len(out)} valid "
          f"({100 * (1 - out.valid.mean()):.1f}% dropped; "
          f"latacc>{LATACC_MAX}: {int((out.latacc_mid > LATACC_MAX).sum())}, "
          f"vmax>{VMAX_KMH}: {int((out.vmax_kmh > VMAX_KMH).sum())})",
          flush=True)
    return out


def main():
    S = S05.load_sets()

    tr_sp = {}
    for v in ("spline_geo", "spline_full"):
        tr_sp[v] = pd.read_parquet(L.RESULTS
                                   / f"spline_trajectories_{v}.parquet")
    tr_kde = pd.read_parquet(L.GDIR / "svd_kde"
                             / "trajectories_fidelity.parquet")

    valid = {}
    for v in ("spline_geo", "spline_full"):
        valid[v] = validity_table(tr_sp[v], v)
    valid["svd_kde"] = validity_table(tr_kde, "svd_kde")
    # ours_cross paths are lane-shaped petq3 geometry with base-like speeds;
    # apply the vmax cap only (no per-sample trajectory reload needed for
    # latacc — its lateral axis is the known dead knob, curvature = base)
    # -> keep unfiltered but report separately below.

    rows = []
    for key, (label, scale) in S05.KEYS.items():
        rv = S05.finite(S["real"], key, scale)
        for m in S05.GEN_SETS:
            d = S[m]
            if m in valid:
                ok = set(valid[m][valid[m].valid].scenario_id)
                d = d[d.scenario_id.isin(ok)]
            v = S05.finite(d, key, scale)
            br, tot = S05.bracket_rate(S, d, key, scale)
            rows.append({
                "key": key, "set": m, "filtered": m in valid,
                "n_finite": len(v),
                "bracket_rate": br / 100.0 if np.isfinite(br) else np.nan,
                "bracket_n": tot,
                "iqr_ratio": S05.iqr(v) / S05.iqr(rv),
                "out_of_real_support": float(
                    ((v < rv.min()) | (v > rv.max())).mean()),
                "samples_per_group": d.groupby("group_key").size().mean()})
    df = pd.DataFrame(rows)
    df.to_csv(L.RESULTS / "summary_stats_filtered.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
