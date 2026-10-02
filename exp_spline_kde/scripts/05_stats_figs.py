"""Coverage / variance comparison at the interaction moment:
real vs SVD+KDE vs ours-cross vs spline_geo vs spline_full.

Baselines are read read-only from sr-tlkeep (real, svd_kde vc) and
exp_cross_coverage (svd_kde_10k, ours_cross, teleport flags, param_space).
Metrics follow exp_cross_coverage 30_figures exactly: per-scenario bracket
rate (own generated span ∋ real value, >=2 finite samples, KDE-style sets
attributed via kernel centre), IQR ratio, global cover fraction,
out-of-real-support. Extra: count-matched bracket (<=27 samples/centre,
fixed rng) so the 10k sets can be compared fairly against ours-cross' 27.

Usage: micromamba run -n nps python 05_stats_figs.py
Output: results/summary_stats.csv, results/param_space_spline.parquet,
        figs/fig1..fig5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slib as L  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

INK, GRAY = "#33322e", "#9a9a94"
GEN_SETS = ("svd_kde", "ours_cross", "spline_geo", "spline_full")
COL = {"real": INK, "svd_kde": "#0f8a60", "ours_cross": "#d99000",
       "spline_geo": "#7aa6d9", "spline_full": "#3557b0"}
LAB = {"real": "real (n=298)", "svd_kde": "SVD+KDE (Nw=10000)",
       "ours_cross": "ours minPET × {min,mid,max}³",
       "spline_geo": "spline-CP KDE geo (8D, Nw=10000)",
       "spline_full": "spline-CP KDE full (11D, Nw=10000)"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6f6e68", "ytick.color": "#6f6e68", "axes.grid": True,
    "grid.color": "#eceae5", "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 11})

KEYS = {
    "agent_arr_speed": ("agent arrival speed [km/h]", 3.6),
    "closing_speed": ("closing speed @ conflict [km/h]", 3.6),
    "pet": ("PET (signed) [s]", 1.0),
    "min_dist": ("min bbox distance [m]", 1.0),
    "conflict_angle": ("conflict angle [deg]", 1.0),
    "drac": ("DRAC [m/s²]", 1.0),
}
SPEED_KEYS = ["agent_arr_speed", "closing_speed"]
CM_N = 27          # count-matched samples per centre (= ours-cross grid size)
CM_SEED = 7


def load_sets():
    sr = pd.read_parquet(L.RDESC)
    if "error" in sr.columns:
        sr = sr[sr.error.isna()]
    xc = pd.read_parquet(L.XRES / "cross_descriptors.parquet")
    if "error" in xc.columns:
        xc = xc[xc.error.isna()]
    real = sr[sr.set == "real"].copy()
    ours5 = sr[sr.set == "ours"].copy()
    tp = pd.read_csv(L.XRES / "teleport_excluded.csv")
    bad = set(zip(tp[tp.set == "ours_sr"].scenario_id,
                  tp[tp.set == "ours_sr"].tag))
    if bad:
        ours5 = ours5[~ours5.apply(
            lambda r: (r.scenario_id, r.tag) in bad, axis=1)]
    ours = pd.concat([ours5, xc[xc.set == "ours_cross"]], ignore_index=True)
    kde10k = xc[xc.set == "svd_kde_10k"].copy()
    sp = pd.read_parquet(L.RESULTS / "spline_descriptors.parquet")
    if "error" in sp.columns:
        sp = sp[sp.error.isna()]
    out = {"real": real.assign(group_key=real.scenario_id),
           "svd_kde": kde10k.assign(group_key=kde10k.center_key),
           "ours_cross": ours.assign(group_key=ours.scenario_id)}
    for v in ("spline_geo", "spline_full"):
        d = sp[sp.set == f"{v}_10k"].copy()
        out[v] = d.assign(group_key=d.center_key)
    return out


def finite(df, key, scale):
    v = df[key].values.astype(float) * scale
    return v[np.isfinite(v)]


def iqr(v):
    return float(np.subtract(*np.percentile(v, [75, 25])))


def cover_frac(rv, gv):
    return float(((rv >= gv.min()) & (rv <= gv.max())).mean())


def count_matched(df, n=CM_N, seed=CM_SEED):
    """<= n samples per group_key, deterministic."""
    rng = np.random.default_rng(seed)
    idx = []
    for _, g in df.groupby("group_key"):
        take = g.index.values
        if len(take) > n:
            take = rng.choice(take, n, replace=False)
        idx.extend(take)
    return df.loc[idx]


def bracket_rate(S, df, key, scale):
    g = df[["group_key", key]].copy()
    g = g[np.isfinite(g[key].astype(float))]
    g[key] = g[key].astype(float) * scale
    spans = g.groupby("group_key")[key].agg(["min", "max", "count"])
    spans = spans[spans["count"] >= 2]
    rr = S["real"].set_index("group_key")[key].astype(float) * scale
    hits = tot = 0
    for sid, r in spans.iterrows():
        if sid in rr.index and np.isfinite(rr[sid]):
            tot += 1
            hits += int(r["min"] <= rr[sid] <= r["max"])
    return (100.0 * hits / tot if tot else np.nan), tot


# ── fig1: density curves + range strips ─────────────────────────────────────

def fig1(S):
    order = ("real",) + GEN_SETS
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
    for ax, (key, (label, scale)) in zip(axes.flat, KEYS.items()):
        vals = {m: finite(S[m], key, scale) for m in order}
        allv = np.concatenate(list(vals.values()))
        pct = [0.5, 97.0] if key == "drac" else [0.5, 99.5]
        lo, hi = np.percentile(allv, pct)
        pad = 0.06 * (hi - lo)
        xs = np.linspace(lo - pad, hi + pad, 400)
        ymax = 0.0
        for m in order:
            v = vals[m]
            v = v[(v >= lo - pad) & (v <= hi + pad)]
            if len(v) < 3 or v.std() < 1e-9:
                continue
            dens = gaussian_kde(v)(xs)
            ymax = max(ymax, dens.max())
            if m == "real":
                ax.fill_between(xs, dens, color=GRAY, alpha=0.25, lw=0)
                ax.plot(xs, dens, color=INK, lw=1.8, label=LAB[m])
            else:
                ax.plot(xs, dens, color=COL[m], lw=1.8, label=LAB[m])
        y0 = -0.09 * ymax
        for i, m in enumerate(order):
            v = vals[m]
            y = y0 * (1 + 0.5 * i)
            ax.plot([v.min(), v.max()], [y, y], color=COL[m], lw=2.8,
                    solid_capstyle="butt", alpha=0.9)
            ax.plot([v.min(), v.max()], [y, y], "|", color=COL[m], ms=6)
        ax.axhline(0, color="#d9d8d3", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(y0 * 3.3, ymax * 1.12)
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.set_title(key, loc="left", fontsize=10)
        if key == "agent_arr_speed":
            ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    fig.suptitle("Interaction-moment value distributions — real vs SVD+KDE vs "
                 "ours-cross vs spline-CP KDE (tlkeep, label-2 westbound)",
                 y=0.995)
    fig.text(0.01, 0.005, "curves: Gaussian KDE inside window (per-set "
             "normalized); bars: min–max; x clipped to joint 0.5–99.5 pct "
             "(drac: 97)", fontsize=8, color="#6f6e68")
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(L.FIGS / "fig1_distributions.png", dpi=160)
    plt.close(fig)


# ── fig2: bracket rate + IQR ratio ──────────────────────────────────────────

def fig2(S):
    keys = list(KEYS)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(keys))
    wd = 0.8 / len(GEN_SETS)
    for ax, metric in zip(axes, ("bracket", "iqr")):
        for j, m in enumerate(GEN_SETS):
            ys = []
            for key in keys:
                scale = KEYS[key][1]
                if metric == "bracket":
                    ys.append(bracket_rate(S, S[m], key, scale)[0])
                else:
                    ys.append(iqr(finite(S[m], key, scale))
                              / iqr(finite(S["real"], key, scale)))
            bars = ax.bar(x + (j - (len(GEN_SETS) - 1) / 2) * wd, ys,
                          wd * 0.92, color=COL[m], label=LAB[m])
            fmt = "{:.0f}" if metric == "bracket" else "{:.2f}"
            for b, v in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                        ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(x, [k.replace("_", "\n") for k in keys], fontsize=8.5)
        ax.grid(axis="x", visible=False)
        if metric == "bracket":
            ax.set_ylabel("% scenarios: own generated span ∋ real value")
            ax.set_ylim(0, 118)
            ax.set_title("(a) per-scenario bracket rate")
            ax.legend(frameon=False, fontsize=7.6, loc="upper left", ncol=2)
        else:
            ax.axhline(1.0, color=INK, lw=1.2, ls="--")
            ax.set_ylabel("IQR ratio (generated / real)")
            ax.set_title("(b) robust spread vs real (1 = same)")
    fig.suptitle("Coverage & spread — generated sets vs real (tlkeep)", y=1.0)
    fig.tight_layout()
    fig.savefig(L.FIGS / "fig2_coverage.png", dpi=160)
    plt.close(fig)


# ── fig3: per-scenario spans vs real value ──────────────────────────────────

def fig3(S):
    fig, axes = plt.subplots(len(GEN_SETS), 2, figsize=(15, 3.1 * len(GEN_SETS)),
                             sharex="col")
    for col, key in enumerate(SPEED_KEYS):
        label, scale = KEYS[key]
        rv = S["real"][["group_key", key]].dropna()
        rv = rv[np.isfinite(rv[key])].set_index("group_key")[key] * scale
        order = rv.sort_values().index.to_list()
        pos = {sid: i for i, sid in enumerate(order)}
        for row, m in enumerate(GEN_SETS):
            ax = axes[row, col]
            g = S[m][["group_key", key]].copy()
            g = g[np.isfinite(g[key])]
            g[key] *= scale
            spans = g.groupby("group_key")[key].agg(["min", "max", "count"])
            spans = spans[spans["count"] >= 2]
            hit = n_tot = 0
            xs_, lo_, hi_ = [], [], []
            for sid in order:
                if sid not in spans.index:
                    continue
                lo, hi = spans.loc[sid, "min"], spans.loc[sid, "max"]
                xs_.append(pos[sid]); lo_.append(lo); hi_.append(hi)
                n_tot += 1
                hit += int(lo <= rv[sid] <= hi)
            ax.vlines(xs_, lo_, hi_, color=COL[m], lw=1.1, alpha=0.55)
            ax.plot([pos[s] for s in order], rv[order].values, ".",
                    color=INK, ms=3.0, label="real value")
            ylo = min(np.percentile(lo_, 1), rv.min())
            yhi = max(np.percentile(hi_, 99), rv.max())
            padY = 0.05 * (yhi - ylo)
            ax.set_ylim(ylo - padY, yhi + padY)
            rate = 100.0 * hit / max(n_tot, 1)
            ax.set_title(f"{LAB[m]} — brackets real: {rate:.0f}% "
                         f"({hit}/{n_tot})", fontsize=9, loc="left")
            ax.set_ylabel(label, fontsize=8.5)
            ax.grid(axis="x", visible=False)
            if row == len(GEN_SETS) - 1:
                ax.set_xlabel("scenarios sorted by real value")
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Per-scenario generated span vs real interaction value "
                 "(KDE sets grouped by kernel centre)", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(L.FIGS / "fig3_per_scenario.png", dpi=160)
    plt.close(fig)


# ── fig4: parameter-space projection ────────────────────────────────────────

def signed_lateral(px, py, ref_xy):
    d = np.hypot(ref_xy[:, 0] - px, ref_xy[:, 1] - py)
    i = int(np.argmin(d))
    j = min(i + 1, len(ref_xy) - 1)
    if j == i:
        i, j = i - 1, i
    tx, ty = ref_xy[j] - ref_xy[i]
    nrm = np.hypot(tx, ty)
    if nrm < 1e-9:
        return float(d.min())
    cross = tx * (py - ref_xy[i][1]) - ty * (px - ref_xy[i][0])
    return float(np.sign(cross) * d.min())


def spline_param_rows(S):
    """Project spline samples onto the exp_cross_coverage parameter axes."""
    cross_tr = pd.read_parquet(L.XRES / "cross_trajectories.parquet")
    sr_tr = pd.read_parquet(L.GDIR / L.METHOD / "trajectories.parquet")
    base_path = {}
    for sid, g in cross_tr[(cross_tr.tag == "o0e0d0")
                           & (cross_tr.role == "agent")].groupby("scenario_id"):
        g = g.sort_values("frame")
        base_path[sid] = np.stack([g.x.values, g.y.values], 1)
    for sid, g in sr_tr[(sr_tr.tag == "base")
                        & (sr_tr.role == "agent")].groupby("scenario_id"):
        if sid not in base_path:
            g = g.sort_values("frame")
            base_path[sid] = np.stack([g.x.values, g.y.values], 1)
    rows = []
    for v in ("spline_geo", "spline_full"):
        tr = pd.read_parquet(L.RESULTS / f"spline_trajectories_{v}.parquet")
        look = {sid: g.sort_values("frame")
                for (sid, role), g in tr.groupby(["scenario_id", "role"])
                if role == "agent"}
        for drow in S[v].itertuples():
            g = look.get(drow.scenario_id)
            ref = base_path.get(drow.center_key)
            if g is None or ref is None:
                continue
            cx, cy = float(drow.conflict_x), float(drow.conflict_y)
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            xs, ys = g.x.values, g.y.values
            i = int(np.argmin(np.hypot(xs - cx, ys - cy)))
            arrf = float(drow.agent_arr_frame)
            rows.append({"set": v, "scenario_id": drow.scenario_id,
                         "tag": "gen",
                         "endspeed_kmh": float(drow.agent_arr_speed) * 3.6,
                         "lat_offset_m": signed_lateral(xs[i], ys[i], ref),
                         "t_conflict_s": (arrf - float(g.frame.min())) / L.FPS
                         if np.isfinite(arrf) else np.nan})
    return pd.DataFrame(rows)


def fig4(S):
    base = pd.read_parquet(L.XRES / "param_space.parquet")  # real/svd/ours
    spl = spline_param_rows(S)
    spl.to_parquet(L.RESULTS / "param_space_spline.parquet")
    df = pd.concat([base, spl], ignore_index=True)
    order = ("real", "svd_kde", "ours_cross", "spline_geo", "spline_full")
    PKEYS = {"endspeed_kmh": "realized EndSpeed @ conflict [km/h]",
             "lat_offset_m": "lateral offset vs lane-centre base path [m]",
             "t_conflict_s": "time to conflict [s]"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    srows = []
    for ax, (key, label) in zip(axes, PKEYS.items()):
        vals = {m: df[df.set == m][key].dropna().values for m in order}
        vals = {m: v[np.isfinite(v)] for m, v in vals.items()}
        allv = np.concatenate(list(vals.values()))
        lo, hi = np.percentile(allv, [0.5, 99.5])
        pad = 0.06 * (hi - lo)
        xs = np.linspace(lo - pad, hi + pad, 400)
        ymax = 0
        for m in order:
            v = vals[m]
            vw = v[(v >= lo - pad) & (v <= hi + pad)]
            if len(vw) < 3 or vw.std() < 1e-9:
                continue
            dens = gaussian_kde(vw)(xs)
            ymax = max(ymax, dens.max())
            if m == "real":
                ax.fill_between(xs, dens, color=GRAY, alpha=0.25, lw=0)
                ax.plot(xs, dens, color=INK, lw=1.8, label=LAB[m])
            else:
                ax.plot(xs, dens, color=COL[m], lw=1.8, label=LAB[m])
            p01, p99 = np.percentile(v, [1, 99])
            srows.append({"key": key, "set": m, "n": len(v), "min": v.min(),
                          "max": v.max(), "p01": p01, "p99": p99,
                          "iqr": iqr(v), "std": v.std(ddof=1)})
        y0 = -0.09 * ymax
        for i, m in enumerate(order):
            v = vals[m]
            y = y0 * (1 + 0.5 * i)
            ax.plot([v.min(), v.max()], [y, y], color=COL[m], lw=2.8,
                    solid_capstyle="butt")
            ax.plot([v.min(), v.max()], [y, y], "|", color=COL[m], ms=6)
        ax.axhline(0, color="#d9d8d3", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(y0 * 3.3, ymax * 1.15)
        ax.set_yticks([])
        ax.set_xlabel(label)
        if key == "endspeed_kmh":
            ax.legend(frameon=False, fontsize=7.4)
    fig.suptitle("Parameter-space projection — every trajectory measured on "
                 "the SAME axes (KDE-style sets referenced to their kernel-"
                 "centre base path)", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(L.FIGS / "fig4_param_space.png", dpi=160)
    plt.close(fig)
    s = pd.DataFrame(srows)
    s.to_csv(L.RESULTS / "param_space_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


# ── fig5: example trajectory overlays ───────────────────────────────────────

def fig5(S):
    real_tracks = pd.read_parquet(L.DDIR / "real_tracks.parquet")
    kde_tr = pd.read_parquet(L.GDIR / "svd_kde"
                             / "trajectories_fidelity.parquet")
    sp_tr = pd.read_parquet(L.RESULTS / "spline_trajectories_spline_full.parquet")
    counts = sp_tr.groupby("center_key").scenario_id.nunique()
    picks = counts.sort_values(ascending=False).index[:3]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col", sharey="col")
    for col, ck in enumerate(picks):
        real = real_tracks[(real_tracks.scenario_id == ck)
                           & (real_tracks.role == "actor")].sort_values("frame")
        for row, (src, cname, lab) in enumerate((
                (sp_tr, "spline_full", "spline-CP KDE samples"),
                (kde_tr, "svd_kde", "SVD+KDE samples"))):
            ax = axes[row, col]
            if src is kde_tr:
                gg = src[src.center_key == ck]
            else:
                gg = src[(src.center_key == ck) & (src.role == "agent")]
            n = 0
            for sid, g in gg.groupby("scenario_id"):
                g = g.sort_values("frame")
                ax.plot(g.x, g.y, color=COL[cname], alpha=0.28, lw=0.9)
                n += 1
                if n >= 40:
                    break
            ax.plot(real.x, real.y, color=INK, lw=2.2, label="real actor")
            ax.set_title(f"{ck} — {lab} (first {n})", fontsize=9.5, loc="left")
            ax.set_aspect("equal")
            if row == 1:
                ax.set_xlabel("x [m]")
            if col == 0:
                ax.set_ylabel("y [m]")
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Sampled trajectories vs real — spline-CP KDE renders stay "
                 "smooth lane-shaped curves; SVD+KDE paths shown for the same "
                 "kernel-centre scenarios", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(L.FIGS / "fig5_examples.png", dpi=160)
    plt.close(fig)


# ── summary csv ─────────────────────────────────────────────────────────────

def summary(S):
    rows = []
    for key, (label, scale) in KEYS.items():
        rv = finite(S["real"], key, scale)
        for m in ("real",) + GEN_SETS:
            v = finite(S[m], key, scale)
            p01, p99 = np.percentile(v, [1, 99])
            row = {"key": key, "set": m, "n_finite": len(v),
                   "frac_finite": len(v) / max(len(S[m]), 1),
                   "min": v.min(), "max": v.max(), "range": v.max() - v.min(),
                   "p01": p01, "p99": p99, "std": v.std(ddof=1), "iqr": iqr(v)}
            if m != "real":
                rp01, rp99 = np.percentile(rv, [1, 99])
                row["std_ratio"] = row["std"] / rv.std(ddof=1)
                row["iqr_ratio"] = row["iqr"] / iqr(rv)
                row["range_ratio"] = row["range"] / (rv.max() - rv.min())
                row["robust_range_ratio"] = (p99 - p01) / (rp99 - rp01)
                row["covers_real_frac"] = cover_frac(rv, v)
                lo = max(v.min(), rv.min()); hi = min(v.max(), rv.max())
                row["range_overlap"] = max(0.0, hi - lo) / (rv.max() - rv.min())
                row["out_of_real_support"] = float(
                    ((v < rv.min()) | (v > rv.max())).mean())
                br, tot = bracket_rate(S, S[m], key, scale)
                row["bracket_rate"] = br / 100.0 if np.isfinite(br) else np.nan
                row["bracket_n"] = tot
                brc, totc = bracket_rate(S, count_matched(S[m]), key, scale)
                row["bracket_rate_cm27"] = (brc / 100.0 if np.isfinite(brc)
                                            else np.nan)
                row["samples_per_group"] = (
                    S[m].groupby("group_key").size().mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(L.RESULTS / "summary_stats.csv", index=False)
    cols = ["key", "set", "bracket_rate", "bracket_rate_cm27", "iqr_ratio",
            "out_of_real_support", "samples_per_group"]
    print(df[df.set != "real"][cols].round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    S = load_sets()
    for m, d in S.items():
        print(f"{m}: {len(d)} rows, {d.group_key.nunique()} groups")
    L.FIGS.mkdir(parents=True, exist_ok=True)
    fig1(S)
    fig2(S)
    fig3(S)
    fig4(S)
    fig5(S)
    summary(S)
    print("figures saved to", L.FIGS)
