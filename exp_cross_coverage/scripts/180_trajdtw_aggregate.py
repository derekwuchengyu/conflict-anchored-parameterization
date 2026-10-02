"""Key-set v5 (2026-08-15): gap_dtw → traj_dtw。
composite = pet / min_dist / conflict_angle / conflict_point /
            conflict_speed(=agent_arr_speed) / traj_dtw

traj_dtw = render agent 軌跡 vs GT agent 軌跡的直接 2D DTW(弧長重取樣
100 點,長度正規化,公尺)。**render 停住/截斷**(render 弧長 < 0.9×GT
弧長)時,GT 只取到「render 終點在 GT 上的最近點」的前段來比
(搜尋限制在 1.5×render 弧長的 GT 前綴內,防止誤配到遠端)— 早停的
render 只就它跑過的部分評分(時間性懲罰由 PET 承擔)。逐場景計算,
納入共同集合規則。資料 = gate-v2 _g2 descriptors + 軌跡 parquets。

輸出: results/best_d_summary_v5.csv, best_anchor_summary_v5.csv,
      figs/fig17_best_anchor.png(v5 重劃), 終端 arm×key 分數表。
1786_1797 的 anchor artifacts 依 best-d 自動解析(d5=results/,
d3=results/_bak_anchor_s3/)。
Usage: micromamba run -n nps python scripts/180_trajdtw_aggregate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

EXP = Path("/home/hcis-s19/Documents/ChengYu/exp_cross_coverage")
SR = Path("/home/hcis-s19/Documents/ChengYu/sr-tlkeep-experiment")
R = EXP / "results"
BAK3 = R / "_bak_anchor_s3"
STEPS = [10, 5, 3, 2]
COMBOS = [P + F for P in "pmx" for F in "pmx"]
KEYS = ["pet", "min_dist", "conflict_angle", "conflict_point",
        "conflict_speed", "traj_dtw"]
ANCH_LAB = {"p": "minPET", "m": "minDist", "x": "crossTraj"}
SUBSETS = [
    ("cutinl", "", "cutin (agent 左切入, n=61)"),
    ("keeptl", "", "TL N→E (agent 左轉, n=50)"),
    ("keeptl_sw", "", "TL S→W (agent 左轉, n=82)"),
    ("special_39_180", "", "special_39_180 (單場景)"),
    ("special_1786_1797", "", "special ego1786×機車1797 (單場景)"),
    ("uturn_859_881", "_w8", "special_Uturn_859_881 (W=8, 單場景)"),
]
INK = "#33322e"
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6f6e68", "ytick.color": "#6f6e68", "axes.grid": False,
    "font.size": 10, "axes.titlesize": 11})
CMAP = LinearSegmentedColormap.from_list("err", ["#f2f6fc", "#2a4d8f"])


# ── traj_dtw ────────────────────────────────────────────────────────────────
def arc_resample(x, y, n=100):
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-6:
        return None
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, x), np.interp(t, s, y)])


def dtw2(a, b):
    C = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        Ci = C[i - 1]
        Di, Dp = D[i], D[i - 1]
        for j in range(1, nb + 1):
            Di[j] = Ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    return float(D[na, nb] / (na + nb))


def traj_dtw(rg, gg):
    """render agent path vs GT agent path; truncated-render → GT prefix.
    Returns (dtw_m, coverage_fraction)."""
    rx, ry = rg.x.values.astype(float), rg.y.values.astype(float)
    gx, gy = gg.x.values.astype(float), gg.y.values.astype(float)
    s_g = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(gx), np.diff(gy)))])
    s_r = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rx), np.diff(ry)))])
    Sg, Srr = s_g[-1], s_r[-1]
    if Sg <= 1e-6 or Srr <= 1e-6:
        return np.nan, np.nan
    cov = 1.0
    if Srr < 0.9 * Sg:                       # render 停住/截斷
        mask = s_g <= min(Sg, Srr * 1.5)
        idx = np.nonzero(mask)[0]
        if len(idx) < 5:
            idx = np.arange(min(5, len(gx)))
        d_end = np.hypot(gx[idx] - rx[-1], gy[idx] - ry[-1])
        cut = int(idx[int(np.argmin(d_end))])
        cut = max(cut, 4)
        gx, gy = gx[:cut + 1], gy[:cut + 1]
        cov = float(s_g[cut] / Sg)
    A, B = arc_resample(rx, ry), arc_resample(gx, gy)
    if A is None or B is None:
        return np.nan, np.nan
    return dtw2(A, B), cov


# ── artifact resolution ─────────────────────────────────────────────────────
def load_desc(p: Path):
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if "error" in d.columns:
        d = d[d.error.isna()]
    if not len(d):
        return pd.DataFrame(index=pd.Index([], name="scenario_id"))
    if "set" in d.columns:
        d = d[d.set == "real"]
    return d.set_index("scenario_id")


def real_g2(name):
    return load_desc(R / f"real_desc_{name}_g2.parquet")


def data_dir(name):
    p = EXP / "data" / name
    return p if p.exists() else SR / "data" / name


def step_files(name, wsuf, s):
    d = R / f"cp3s{s}_descriptors_{name}{wsuf}_g2.parquet"
    if s == 10 and not wsuf:
        t = R / f"cp3_trajectories_{name}.parquet"
    else:
        t = R / f"cp3s{s}_trajectories_{name}{wsuf}.parquet"
    return d, t


def combo_files(name, wsuf, c, astep):
    base = BAK3 if (name == "special_1786_1797" and astep == 3) else R
    return (base / f"anch{c}_descriptors_{name}{wsuf}_g2.parquet",
            base / f"anch{c}_trajectories_{name}{wsuf}.parquet")


def paired_delta(g, r, key):
    if key == "conflict_point":
        if np.isfinite(g.conflict_x) and np.isfinite(r.conflict_x):
            return float(np.hypot(g.conflict_x - r.conflict_x,
                                  g.conflict_y - r.conflict_y))
        return np.nan
    # v5 (2026-08-15 user): conflict_speed = agent 進衝突區當下自身速度
    # (agent_arr_speed = speed at the agent's closest sample to the conflict
    # point), NOT the relative closing speed.
    col = "agent_arr_speed" if key == "conflict_speed" else key
    gv, rv = float(g[col]), float(r[col])
    if not (np.isfinite(gv) and np.isfinite(rv)):
        return np.nan
    return abs(gv - rv)


def aggregate(name, arms):
    """arms: {arm: (desc_df|None, tdtw_series, cov_series)} → piv table."""
    rd = real_g2(name)
    valid = {k for k, (d, _t, _c) in arms.items() if d is not None and len(d)}
    med, ncom = {}, {}
    for key in KEYS:
        tab = {}
        for k in valid:
            dd, td, _ = arms[k]
            if key == "traj_dtw":
                tab[k] = td.reindex([s for s in dd.index if s in rd.index])
            else:
                tab[k] = pd.Series({sid: paired_delta(dd.loc[sid],
                                                      rd.loc[sid], key)
                                    for sid in dd.index if sid in rd.index})
        t = pd.DataFrame(tab).dropna()
        med[key] = {k: (float(t[k].median()) if k in t.columns and len(t)
                        else np.nan) for k in arms}
        ncom[key] = len(t)
    piv = pd.DataFrame({key: {k: med[key][k] for k in arms} for key in KEYS})
    norm = piv.div(piv.max(axis=0), axis=1)
    piv["composite"] = norm.mean(axis=1).where(
        [k in valid for k in piv.index], np.nan)
    piv["cov_med"] = [float(arms[k][2].median()) if arms[k][2] is not None
                      and len(arms[k][2].dropna()) else np.nan
                      for k in piv.index]
    piv.attrs["n"] = ncom
    return piv


def build_arm(dfile, tfile, gt_tracks):
    dd = load_desc(dfile)
    if dd is None or not tfile.exists():
        return (dd, pd.Series(dtype=float), pd.Series(dtype=float))
    tr = pd.read_parquet(tfile)
    td, cv = {}, {}
    ag = {sid: g.sort_values("frame") for (sid, role), g in
          tr.groupby(["scenario_id", "role"]) if role == "agent"}
    for sid in dd.index:
        gg = gt_tracks.get(sid)
        rg = ag.get(sid)
        if gg is None or rg is None:
            continue
        v, c = traj_dtw(rg, gg)
        td[sid], cv[sid] = v, c
    return dd, pd.Series(td, dtype=float), pd.Series(cv, dtype=float)


def main():
    out_d, out_a, best_step, pivs = [], [], {}, {}
    for name, wsuf, label in SUBSETS:
        rt = pd.read_parquet(data_dir(name) / "real_tracks.parquet")
        gt = {sid: g.sort_values("frame") for (sid, role), g in
              rt.groupby(["scenario_id", "role"]) if role == "actor"}
        # best d (v5)
        arms = {f"±{s}": build_arm(*step_files(name, wsuf, s), gt)
                for s in STEPS}
        piv = aggregate(name, arms)
        ok = piv.composite.dropna()
        bd = ok.idxmin() if len(ok) else None
        best_step[name] = int(bd.strip("±")) if bd else 10
        # anchor renders exist only at the v4 steps; a best-d flip within a
        # 0.01 tie-band (e.g. keeptl_sw ±5 0.714 vs ±10 0.715) is noise —
        # pin to the rendered step instead of comparing arms across d.
        RENDERED = {"cutinl": 10, "keeptl": 10, "keeptl_sw": 10,
                    "special_39_180": 10, "special_1786_1797": (3, 5),
                    "uturn_859_881": 5}
        rend = RENDERED[name]
        rend = rend if isinstance(rend, tuple) else (rend,)
        if best_step[name] not in rend:
            pin = min(rend, key=lambda s: abs(
                ok.get(f"±{s}", np.inf) - ok.min()))
            if abs(ok.get(f"±{pin}", np.inf) - ok.min()) < 0.01:
                print(f"    [pin] {name}: best d {bd} ties ±{pin} "
                      f"(Δ<0.01) — anchors evaluated at ±{pin}", flush=True)
                best_step[name] = pin
            else:
                print(f"    !! {name}: best d {bd} has NO anchor renders — "
                      f"evaluating at ±{pin} (re-render to confirm)",
                      flush=True)
                best_step[name] = pin
        for arm, r in piv.iterrows():
            out_d.append({"subset": name, "arm": arm,
                          **{k: r[k] for k in KEYS},
                          "composite": r.composite, "cov_med": r.cov_med})
        print(f"[d] {name:20s} best d = {bd} "
              + " ".join(f"{k}={v:.3f}" for k, v in ok.items()), flush=True)

        # anchors at best d
        astep = best_step[name]
        arms = {c: build_arm(*combo_files(name, wsuf, c, astep), gt)
                for c in COMBOS}
        arms["legacy"] = build_arm(*step_files(name, wsuf, astep), gt)
        n_missing = [c for c, (d, _t, _c) in arms.items() if d is None]
        if n_missing:
            print(f"    !! {name}: anchor artifacts missing at d{astep}: "
                  f"{n_missing} (need re-render at this d)", flush=True)
        piv = aggregate(name, arms)
        pivs[name] = (piv, astep, label)
        for arm, r in piv.iterrows():
            out_a.append({"subset": name, "astep": astep, "arm": arm,
                          **{k: r[k] for k in KEYS},
                          "composite": r.composite, "cov_med": r.cov_med})
    pd.DataFrame(out_d).to_csv(R / "best_d_summary_v5.csv", index=False)
    pd.DataFrame(out_a).to_csv(R / "best_anchor_summary_v5.csv", index=False)

    print("\n== per-subset arm×key (v5) ==")
    for name, _w, _l in SUBSETS:
        piv, astep, label = pivs[name]
        print(f"\n===== {label}  [d=±{astep}, n_pet={piv.attrs['n']['pet']}, "
              f"n_dtw={piv.attrs['n']['traj_dtw']}]")
        print(piv.round(3).to_string())
    print("\n== best pos×frm (v5) ==")
    for name, _w, _l in SUBSETS:
        piv, astep, label = pivs[name]
        ok = piv.composite.dropna()
        b = ok.idxmin()
        print(f"  {label:44s} d=±{astep}  best = {b} ({ok[b]:.3f})  "
              f"legacy = {piv.loc['legacy', 'composite']:.3f}")

    # fig17 v4
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))
    for ax, (name, _w, _l) in zip(axes.flat, SUBSETS):
        piv, astep, label = pivs[name]
        M = np.full((3, 3), np.nan)
        for i, P in enumerate("pmx"):
            for j, F in enumerate("pmx"):
                if P + F in piv.index:
                    M[i, j] = piv.loc[P + F, "composite"]
        vmax, vmin = np.nanmax(M), np.nanmin(M)
        ax.imshow(M, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal")
        best = np.unravel_index(np.nanargmin(M), M.shape)
        for i in range(3):
            for j in range(3):
                if np.isfinite(M[i, j]):
                    dark = (M[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.55
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                            fontsize=10,
                            fontweight="bold" if best == (i, j) else "normal",
                            color="white" if dark else INK)
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1,
                                   fill=False, ec="#c22e2e", lw=2.4))
        ax.set_xticks(range(3), [ANCH_LAB[k] for k in "pmx"], fontsize=8.5)
        ax.set_yticks(range(3), [ANCH_LAB[k] for k in "pmx"], fontsize=8.5)
        ax.set_xlabel("critical FRAME (速度事件)", fontsize=8.5)
        ax.set_ylabel("critical POS (CP 幾何)", fontsize=8.5)
        ax.set_title(f"{label}  d=±{astep}\ncomposite (低=好)   legacy = "
                     f"{piv.loc['legacy', 'composite']:.3f}",
                     fontsize=9.2, loc="left")
    fig.suptitle("critical-pos × critical-frame — v5 composite "
                 "(pet/min_dist/conflict_angle/conflict_point/conflict_speed/"
                 "traj_dtw[截斷感知];紅框 = best;PET gate v2)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(EXP / "figs" / "fig17_best_anchor.png", dpi=150)
    print("\nsaved figs/fig17_best_anchor.png (v5)")


if __name__ == "__main__":
    main()
