"""Stage 9d:θ 實驗的比較、KDE 分佈匹配、validity 證明圖。

fig12_theta_coverage   old sobol4(Crit_Lat/Long)vs tsobol_pool / tsobol_cond
                       / tkde_cond:bracket 6 鍵 + trajspace cover(需先跑
                       COV_RESULTS_DIR=results_theta 的 30/40)
fig13_theta_kde_match  real θ 分佈 vs tkde_cond 取樣值 vs render 實現值 θ-hat
                       (KS / W1)+ validity 率(wrongway / win-dev>5 m /
                       teleport)含 tuni_wild 病態對照
fig14_wild_example     tuni_wild 最嚴重逆向案例 vs 同場景 tkde_cond fan
Usage: micromamba run -n nps python scripts/95_theta_figs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cov_lib as L  # noqa: E402

RT = L.EXP / "results_theta"
RO = L.EXP / "results"
INK, MUT, GRID = "#33322e", "#6f6e68", "#eceae5"
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})
BKEYS = ["pet", "min_dist", "conflict_angle", "agent_arr_speed",
         "closing_speed", "drac"]
KLAB = {"pet": "PET", "min_dist": "min dist", "conflict_angle": "angle",
        "agent_arr_speed": "arr speed", "closing_speed": "closing",
        "drac": "DRAC"}
TSTRATS = ["sobol4_old", "tsobol_pool", "tsobol_cond", "tkde_cond"]
TCOL = {"sobol4_old": "#7a4fbf", "tsobol_pool": "#a1541c",
        "tsobol_cond": "#2a78d6", "tkde_cond": "#0f8a60",
        "tuni_wild": "#c23b3b"}
TLAB = {"sobol4_old": "Sobol27 Crit_Lat/Long(舊)",
        "tsobol_pool": "Sobol27 θ pooled range",
        "tsobol_cond": "Sobol27 θ cluster range",
        "tkde_cond": "θ cluster-KDE 27",
        "tuni_wild": "uniform θ pooled min-max(病態)"}


def savefig(fig, name):
    fig.savefig(L.FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig12():
    bro = pd.read_csv(RO / "bracket_by_key.csv")
    brt = pd.read_csv(RT / "bracket_by_key.csv")
    tso = pd.read_csv(RO / "trajspace_summary.csv")
    tst = pd.read_csv(RT / "trajspace_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.2))
    for c, name in enumerate(L.MULTI):
        ax = axes[0, c]
        xs = np.arange(len(BKEYS))
        W = 0.19
        for i, strat in enumerate(TSTRATS):
            if strat == "sobol4_old":
                d = bro[(bro.subset == name) & (bro.strategy == "sobol4")]
            else:
                d = brt[(brt.subset == name) & (brt.strategy == strat)]
            v = [d[d.key == k].bracket.values for k in BKEYS]
            v = [x[0] if len(x) else np.nan for x in v]
            b = ax.bar(xs + (i - 1.5) * W, v, W * 0.9, color=TCOL[strat],
                       label=TLAB[strat] if c == 0 else None)
            for rect, val in zip(b, v):
                if np.isfinite(val):
                    ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5,
                            f"{val:.0f}", ha="center", fontsize=5.4,
                            color=MUT)
        ax.set_xticks(xs, [KLAB[k] for k in BKEYS], fontsize=7.5)
        ax.set_ylim(0, 112)
        ax.set_title(f"{name} — bracket rate [%]", loc="left", fontsize=9.5)
        if c == 0:
            ax.legend(fontsize=6.5, loc="lower right")
        ax = axes[1, c]
        rows_lab = ["cover≤0.5m full", "cover≤0.5m win", "GT站點cov full",
                    "n_eff/27 [%]"]
        for i, strat in enumerate(TSTRATS):
            if strat == "sobol4_old":
                d = tso[(tso.subset == name) & (tso.strategy == "sobol4")]
            else:
                d = tst[(tst.subset == name) & (tst.strategy == strat)]
            df_, dw = (d[d.scope == "full"], d[d.scope == "win"])
            if not len(df_) or not len(dw):
                continue
            df_, dw = df_.iloc[0], dw.iloc[0]
            vals = [df_.cover_rate_05, dw.cover_rate_05, df_.gt_station_cov,
                    100 * df_.neff_med / 27]
            b = ax.bar(np.arange(4) + (i - 1.5) * 0.19, vals, 0.17,
                       color=TCOL[strat])
            for rect, val in zip(b, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5,
                        f"{val:.0f}", ha="center", fontsize=5.4, color=MUT)
            ax.text(0.99, 0.98 - i * 0.075,
                    f"minDTW {df_.cover_dtw_med:.2f} m",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color=TCOL[strat])
        ax.set_xticks(np.arange(4), rows_lab, fontsize=7)
        ax.set_ylim(0, 112)
        ax.set_title("trajspace cover / 預算利用", loc="left", fontsize=9.5)
    fig.suptitle("θ 參數化 coverage — 舊 Crit 軸 vs θ pooled vs θ cluster "
                 "vs θ cluster-KDE(同 27 預算)", x=0.01, ha="left")
    fig.text(0.01, -0.045,
             "Sobol = scrambled 低差異序列(deterministic 準亂數;任意前綴都近似均勻鋪滿取樣盒,"
             "比純亂數均勻、比網格不規則)。紫 = 舊法對照:每場景以自身路徑為中心的 ±2 m 位移軸"
             "(scenario-centered);棕→藍→綠 = θ 軸依序加上條件:class pooled range(無條件)→ "
             "-d cluster range(條件化)→ cluster-KDE(條件化+真實機率密度)。\n"
             "bracket = 該場景 fan 的 min ≤ GT 值 ≤ max 的場景比率;cover≤0.5m = fan 中最佳 render "
             "對自己 GT 的 DTW ≤0.5 m 的場景比率;GT站點cov = GT 路徑 20 站點中被任一 render 貼近 "
             "0.5 m 的比例;n_eff = fan 在路徑空間 greedy 0.5 m ε-cover 的等效點數(重複點會現形)。",
             fontsize=7, color="#6f6e68")
    fig.tight_layout()
    savefig(fig, "fig12_theta_coverage.png")


def realized_theta(path, geo_row):
    init = path[0]
    A = np.array([geo_row.mdx, geo_row.mdy])
    i_a = int(np.argmin(np.hypot(path[:, 0] - A[0], path[:, 1] - A[1])))
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]

    def at(arc):
        return np.array([np.interp(arc, s, path[:, 0]),
                         np.interp(arc, s, path[:, 1])])
    if s[-1] < s[i_a] + geo_row.L1 + geo_row.L2:
        return (np.nan, np.nan)
    Ap = path[i_a]
    B = at(s[i_a] + geo_row.L1)
    C = at(s[i_a] + geo_row.L1 + geo_row.L2)
    v1, v2, v3 = Ap - init, B - Ap, C - B

    def ang(u, v):
        return np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v))
    if min(np.hypot(*v1), np.hypot(*v2), np.hypot(*v3)) < 0.5:
        return (np.nan, np.nan)
    return (ang(v1, v2), ang(v2, v3))


def win_dev(gt, rd, cxy, half=20.0):
    x, y = rd[:, 0], rd[:, 1]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    i = int(np.argmin(np.hypot(x - cxy[0], y - cxy[1])))
    m = (s >= s[i] - half) & (s <= s[i] + half)
    if m.sum() < 2:
        return np.nan
    d, _ = cKDTree(gt).query(rd[m])
    return float(d.max())


def exit_angle(gt, rd, k_m=3.0):
    i = int(np.argmin(np.hypot(rd[:, 0] - gt[-1, 0], rd[:, 1] - gt[-1, 1])))
    if i < 3:
        return np.nan

    def chord(p, j):
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(p[:j + 1], axis=0).T))]
        j0 = int(np.searchsorted(s, max(0.0, s[-1] - k_m)))
        return p[j] - p[j0] if j > j0 else None
    u = chord(rd, i)
    v = chord(gt, len(gt) - 1)
    if u is None or v is None:
        return np.nan
    nu, nv = np.hypot(*u), np.hypot(*v)
    if min(nu, nv) < 0.3:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(u @ v / nu / nv, -1, 1))))


def validity_and_match():
    stats, vrows = [], []
    for name in L.MULTI:
        man = pd.read_parquet(RT / f"manifest_{name}.parquet")
        traj = pd.read_parquet(RT / f"fan_trajectories_{name}.parquet")
        geo = pd.read_csv(RT / f"theta_geom_{name}.csv").set_index(
            "scenario_id")
        try:
            bad = pd.read_csv(RT / f"validity_{name}.csv")
            tp = set(map(tuple, bad[bad.kind == "teleport"]
                         [["scenario_id", "tag"]].values))
        except Exception:  # noqa: BLE001
            tp = set()
        rt = L.real_tracks(name)
        gts = {sid: g.sort_values("frame")[["x", "y"]].values
               for sid, g in rt[rt.role == "actor"].groupby("scenario_id")}
        agent = {k: g.sort_values("frame")[["x", "y"]].values
                 for k, g in traj[traj.role == "agent"].groupby(
                     ["scenario_id", "tag"])}
        for strat, gm in man.groupby("strategy"):
            if strat == "base":
                continue
            n = ww = wd = ntp = 0
            th_real = []
            for r in gm.itertuples():
                sid = r.scenario_id
                if sid not in geo.index or not geo.loc[sid].theta_ok:
                    continue
                key = (sid, r.tag)
                n += 1
                if key in tp:
                    ntp += 1
                    continue
                if key not in agent or sid not in gts:
                    continue
                p = agent[key]
                ea = exit_angle(gts[sid], p)
                if np.isfinite(ea) and ea > 120:
                    ww += 1
                    vrows.append(dict(subset=name, strategy=strat,
                                      scenario_id=sid, tag=r.tag,
                                      exit_angle=round(ea, 1)))
                wdv = win_dev(gts[sid], p,
                              (geo.loc[sid].critx, geo.loc[sid].crity),
                              half=float(L.SUBSETS[name]["d"]))
                if np.isfinite(wdv) and wdv > 5.0:
                    wd += 1
                if strat == "tkde_cond":
                    th_real.append(realized_theta(p, geo.loc[sid]))
            row = dict(subset=name, strategy=strat, n=n,
                       teleport_pct=round(100 * ntp / n, 2) if n else np.nan,
                       wrongway_pct=round(100 * ww / n, 2) if n else np.nan,
                       windev5_pct=round(100 * wd / n, 2) if n else np.nan)
            if strat == "tkde_cond" and th_real:
                th_real = np.array([t for t in th_real
                                    if np.isfinite(t[0])])
                row["real_th_n"] = len(th_real)
                np.save(RT / f"realized_theta_{name}.npy", th_real)
            stats.append(row)
    vdf = pd.DataFrame(stats)
    vdf.to_csv(RT / "theta_validity.csv", index=False)
    pd.DataFrame(vrows).to_csv(RT / "theta_wrongway_list.csv", index=False)
    return vdf


def fig13(vdf):
    fig, axes = plt.subplots(len(L.MULTI), 3, figsize=(14.5, 3.9 * 3))
    for r, name in enumerate(L.MULTI):
        geo = pd.read_csv(RT / f"theta_geom_{name}.csv")
        geo = geo[geo.theta_ok]
        man = pd.read_parquet(RT / f"manifest_{name}.parquet")
        kde_m = man[(man.strategy == "tkde_cond") &
                    (man.scenario_id.isin(set(geo.scenario_id)))]
        try:
            th_hat = np.load(RT / f"realized_theta_{name}.npy")
        except Exception:  # noqa: BLE001
            th_hat = np.empty((0, 2))
        for c_i, (key, real, samp, hat) in enumerate((
                ("θ1", geo.theta1_deg.values, kde_m.t1.values,
                 th_hat[:, 0] if len(th_hat) else []),
                ("θ2", geo.theta2_deg.values, kde_m.t2.values,
                 th_hat[:, 1] if len(th_hat) else []))):
            ax = axes[r, c_i]
            lo = np.floor((min(np.percentile(real, 1),
                               np.percentile(samp, 1)) - 5) / 4) * 4
            hi = np.ceil((max(np.percentile(real, 99),
                              np.percentile(samp, 99)) + 5) / 4) * 4
            bins = np.arange(lo, hi + 4, 4.0)
            ax.hist(real, bins=bins, density=True, color="#c9c8c2",
                    edgecolor="white", lw=0.8,
                    label=f"real 離散 (n={len(real)})")
            xs = np.linspace(lo, hi, 400)
            sfin = samp[np.isfinite(samp)]
            ax.plot(xs, gaussian_kde(sfin)(xs), color=TCOL["tkde_cond"],
                    lw=2.0, label=f"KDE 取樣 pdf (n={len(sfin)})")
            hat = np.asarray(hat, float)
            hat = hat[np.isfinite(hat)] if len(hat) else hat
            if len(hat) > 5:
                ax.plot(xs, gaussian_kde(hat)(xs), color="#7a4fbf",
                        lw=1.7, ls="--",
                        label=f"render 實現 θ-hat pdf (n={len(hat)})")
            ks = ks_2samp(real, samp)
            w1 = wasserstein_distance(real, samp)
            txt = f"取樣 vs real: KS {ks.statistic:.2f}  W1 {w1:.1f}°"
            if len(hat):
                ksh = ks_2samp(real, hat)
                w1h = wasserstein_distance(real, hat)
                txt += f"\n實現 vs real: KS {ksh.statistic:.2f}  W1 {w1h:.1f}°"
            ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right",
                    va="top", fontsize=7.5)
            ax.set_title(f"{name} — {key} 分佈", loc="left", fontsize=9)
            if r == 0 and c_i == 0:
                ax.legend(fontsize=6.5, loc="upper left")
            ax.set_xlabel("deg")
        ax = axes[r, 2]
        d = vdf[vdf.subset == name]
        strats = ["tsobol_pool", "tsobol_cond", "tkde_cond", "tuni_wild"]
        for j, met in enumerate(("wrongway_pct", "windev5_pct",
                                 "teleport_pct")):
            for i, s in enumerate(strats):
                row = d[d.strategy == s]
                v = float(row[met].iloc[0]) if len(row) else np.nan
                b = ax.bar(j + (i - 1.5) * 0.2, v, 0.18, color=TCOL[s],
                           label=TLAB[s] if (r == 0 and j == 0) else None)
                if np.isfinite(v):
                    ax.text(j + (i - 1.5) * 0.2, v + 0.08, f"{v:.1f}",
                            ha="center", fontsize=5.6, color=MUT)
        ax.set_xticks(range(3), ["逆向 >120°", "win-dev(±d) >5 m", "teleport"],
                      fontsize=8)
        ax.set_title("validity 失效率 [%]", loc="left", fontsize=9)
        if r == 0:
            ax.legend(fontsize=6, ncols=2)
    fig.suptitle("θ cluster-KDE 取樣 vs 真實分佈(左/中)與 validity(右)"
                 " — 病態對照 = pooled min-max 均勻", x=0.01, ha="left")
    fig.text(0.01, -0.035,
             "θ-hat(render 實現值)= 對每條 render 反算:A = 軌跡上最接近 -d 點處,"
             "B/C = 由 A 沿弧長 +L1 / +L1+L2 處;θ-hat1 = ∠(init→A, A→B)、"
             "θ-hat2 = ∠(A→B, B→C)(與 GT 的 θ 定義完全相同)。"
             "KS = max_x |F_real(x) − F_sample(x)|(兩累積分佈的最大垂直差;0 = 同分佈、1 = 完全分離)。"
             "W1 = ∫|F_real − F_sample| dx(把一個分佈搬成另一個所需的平均搬運距離,單位 deg)。\n"
             "win-dev = max{ 到 GT 軌跡最近距離 : render 上距 crit 點 ±d 弧長窗內的每個點 }(d = ours 方法的 crit±d,本三類 = 10 m);>5 m 計一次失效。"
             "teleport = 逐步位置差速度(略前 10 步)max >30 m/s 且超過 esmini 狀態速度 +10 m/s(route 失效瞬移的 fingerprint)。"
             "逆向 = render 在最接近 GT 終點處的行進方向(3 m 弦)與 GT 終點行進方向夾角 >120°。",
             fontsize=7, color="#6f6e68")
    fig.tight_layout()
    savefig(fig, "fig13_theta_kde_match.png")


def fig14():
    try:
        ww = pd.read_csv(RT / "theta_wrongway_list.csv")
    except Exception:  # noqa: BLE001
        return
    ww = ww[ww.strategy == "tuni_wild"]
    if not len(ww):
        print("no wild wrongway case found — fig14 skipped")
        return
    w = ww.sort_values("exit_angle").iloc[-1]
    name, sid = w.subset, w.scenario_id
    man = pd.read_parquet(RT / f"manifest_{name}.parquet")
    traj = pd.read_parquet(RT / f"fan_trajectories_{name}.parquet")
    geo = pd.read_csv(RT / f"theta_geom_{name}.csv").set_index("scenario_id")
    rt = L.real_tracks(name)
    gt = rt[(rt.scenario_id == sid) & (rt.role == "actor")].sort_values(
        "frame")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    kt = set(man[(man.strategy == "tkde_cond") &
                 (man.scenario_id == sid)].tag)
    for t in kt:
        g = traj[(traj.scenario_id == sid) & (traj.tag == t) &
                 (traj.role == "agent")].sort_values("frame")
        if len(g):
            ax.plot(g.x, g.y, color=TCOL["tkde_cond"], lw=0.8, alpha=0.45,
                    zorder=2)
    g = traj[(traj.scenario_id == sid) & (traj.tag == w.tag) &
             (traj.role == "agent")].sort_values("frame")
    ax.plot(g.x, g.y, color=TCOL["tuni_wild"], lw=2.0, zorder=4,
            label=f"wild 取樣(exit ∠ {w.exit_angle:.0f}°)")
    ax.plot(gt.x, gt.y, color=INK, lw=2.0, zorder=3, label="GT")
    ge = geo.loc[sid]
    wm = man[(man.scenario_id == sid) & (man.tag == w.tag)].iloc[0]
    ax.plot(ge.mdx, ge.mdy, "o", ms=7, mfc="white", mec=INK, zorder=5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"{name} {sid} — 無條件 pooled min-max 取樣的病態案例"
                 f"(θ1 {wm.t1:+.1f}°, θ2 {wm.t2:+.1f}° vs 該場景原始 "
                 f"{ge.theta1_deg:+.1f}/{ge.theta2_deg:+.1f}°);"
                 f"綠 = 同場景 cluster-KDE fan", loc="left", fontsize=9)
    savefig(fig, "fig14_wild_example.png")


if __name__ == "__main__":
    if (RT / "trajspace_summary.csv").exists():
        fig12()
    if (RT / "theta_validity.csv").exists():
        vdf = pd.read_csv(RT / "theta_validity.csv")
    else:
        vdf = validity_and_match()
    print(vdf.to_string(index=False))
    fig13(vdf)
    fig14()
