"""PET gate v2 (swept-bbox rescue) 全量重算 — GT + d-sweep + anchor-sweep 的
descriptors 以 _g2 後綴重算(舊檔保留做 A/B),再重跑兩層聚合:
  (1) best-d(6-key composite)      → results/best_d_composite_g2.csv
  (2) best pos×frm anchor combo     → results/best_anchor_{summary,composite}_g2.csv
並報告: 每子集 GT 被救回的場景、PET n_common 前後、verdict 前後。
Window keys (profile_dissim/gap_dtw) 與 PET 無關,沿用既有 CSV。
Usage: micromamba run -n nps python scripts/170_pet_gate_recompute.py
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib as L  # noqa: E402
from run_label_lib import trim_lead_still, wrap180  # noqa: E402
from hetero_param.sweep import core as SWC  # noqa: E402
from hetero_param.similarity.core import Traj  # noqa: E402

EXP = Path("/home/hcis-s19/Documents/ChengYu/exp_cross_coverage")
SR = Path("/home/hcis-s19/Documents/ChengYu/sr-tlkeep-experiment")
R = EXP / "results"
FPS = 30.0
STEPS = [10, 5, 3, 2]
COMBOS = [P + F for P in "pmx" for F in "pmx"]
DKEYS = ["pet", "min_dist", "agent_arr_speed", "conflict_angle",
         "agent_arr_heading", "conflict_point"]
COMP_KEYS = ["pet", "min_dist", "conflict_angle", "conflict_point",
             "profile_dissim", "gap_dtw"]
SUBSETS = [  # (name, wsuf, anchor_step)
    ("cutinl", "", 10), ("keeptl", "", 10), ("keeptl_sw", "", 10),
    ("special_39_180", "", 10), ("special_1786_1797", "", 3),
    ("uturn_859_881", "_w8", 5),
]


def data_dir(name):
    p = EXP / "data" / name
    return p if p.exists() else SR / "data" / name


def _traj(g, derived=False):
    t = g.frame.values.astype(float) / FPS
    sp = g.speed.values.astype(float)
    T = Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
             y=g.y.values.astype(float),
             heading=g.heading_deg.values.astype(float), speed=sp,
             accel=np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp),
             fps=FPS, length=float(g.length.iloc[0]),
             width=float(g.width.iloc[0]), meta={})
    return SWC._derived_kinematics_copy(T) if derived else T


G_JOB = {}


def _desc_real(sid):
    try:
        a, e = G_JOB[sid]
        d = SWC.descriptors(_traj(a, derived=True), _traj(e), estimator="bbox")
        d.update({"set": "real", "scenario_id": sid, "tag": "real"})
        return d
    except Exception as ex:  # noqa: BLE001
        return {"scenario_id": sid, "error": f"{type(ex).__name__}: {ex}"}


def _desc_gen(sid):
    try:
        a, e = G_JOB[sid]
        d = SWC.descriptors(_traj(a), _traj(e), estimator="bbox")
        d.update({"scenario_id": sid})
        return d
    except Exception as ex:  # noqa: BLE001
        return {"scenario_id": sid, "error": f"{type(ex).__name__}: {ex}"}


def compute_desc(tracks: pd.DataFrame, roles=("agent", "ego"), real=False,
                 screen_teleport=True):
    global G_JOB
    G_JOB = {}
    ar, er = roles
    for (sid, role), g in tracks.groupby(["scenario_id", "role"]):
        if role != ar:
            continue
        g = g.sort_values("frame")
        e = tracks[(tracks.scenario_id == sid) &
                   (tracks.role == er)].sort_values("frame")
        if not len(e):
            continue
        if not real and screen_teleport and L.is_teleport(g):
            continue
        G_JOB[sid] = ((g if real else trim_lead_still(g)), e)
    fn = _desc_real if real else _desc_gen
    with Pool(8) as pool:
        out = [r for r in pool.imap_unordered(fn, list(G_JOB), chunksize=8)
               if r]
    dd = pd.DataFrame(out)
    if "error" in dd.columns:
        dd = dd[dd.error.isna()]
    return dd


def g2(path: Path) -> Path:
    return path.with_name(path.stem + "_g2.parquet")


def dsweep_tfile(name, wsuf, step):
    if step == 10 and not wsuf:
        return R / f"cp3_trajectories_{name}.parquet"
    return R / f"cp3s{step}_trajectories_{name}{wsuf}.parquet"


def recompute_all():
    made = 0
    for name, wsuf, _astep in SUBSETS:
        # GT
        out = R / f"real_desc_{name}_g2.parquet"
        if not out.exists():
            tr = pd.read_parquet(data_dir(name) / "real_tracks.parquet")
            dd = compute_desc(tr, roles=("actor", "ego"), real=True)
            dd.to_parquet(out)
            made += 1
            print(f"[GT ] {name}: {len(dd)} rows", flush=True)
        # d-sweep renders
        for step in STEPS:
            tf = dsweep_tfile(name, wsuf, step)
            if not tf.exists():
                continue
            out = g2(R / f"cp3s{step}_descriptors_{name}{wsuf}.parquet")
            if out.exists():
                continue
            dd = compute_desc(pd.read_parquet(tf))
            dd.to_parquet(out)
            made += 1
            print(f"[d  ] {name} s{step}: {len(dd)} rows", flush=True)
        # anchor renders
        for c in COMBOS:
            tf = R / f"anch{c}_trajectories_{name}{wsuf}.parquet"
            if not tf.exists():
                continue
            out = g2(R / f"anch{c}_descriptors_{name}{wsuf}.parquet")
            if out.exists():
                continue
            dd = compute_desc(pd.read_parquet(tf))
            dd.to_parquet(out)
            made += 1
            print(f"[anc] {name} {c}: {len(dd)} rows", flush=True)
    print(f"recomputed {made} descriptor files", flush=True)


def load_g2(name, wsuf, kind, key):
    """kind: 'real' | ('step', s) | ('combo', c) — returns indexed df or None."""
    if kind == "real":
        p = R / f"real_desc_{name}_g2.parquet"
    elif kind[0] == "step":
        p = g2(R / f"cp3s{kind[1]}_descriptors_{name}{wsuf}.parquet")
    else:
        p = g2(R / f"anch{kind[1]}_descriptors_{name}{wsuf}.parquet")
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if "error" in d.columns:
        d = d[d.error.isna()]
    if not len(d):
        return pd.DataFrame(index=pd.Index([], name="scenario_id"))
    return d.set_index("scenario_id")


def paired_delta(g, r, key):
    if key == "conflict_point":
        if np.isfinite(g.conflict_x) and np.isfinite(r.conflict_x):
            return float(np.hypot(g.conflict_x - r.conflict_x,
                                  g.conflict_y - r.conflict_y))
        return np.nan
    gv, rv = float(g[key]), float(r[key])
    if not (np.isfinite(gv) and np.isfinite(rv)):
        return np.nan
    if key == "agent_arr_heading":
        return abs(float(wrap180(gv % 360 - rv % 360)))
    return abs(gv - rv)


def composite_table(name, wsuf, arms, wtab_getter):
    """arms: {label: desc_df}. Returns (summary_rows, composite dict)."""
    rd = load_g2(name, wsuf, "real", None)
    valid = {k: v for k, v in arms.items() if v is not None and len(v)}
    delta = {}
    for key in DKEYS:
        tab = {}
        for k, dd in valid.items():
            col = {sid: paired_delta(dd.loc[sid], rd.loc[sid], key)
                   for sid in dd.index if sid in rd.index}
            tab[k] = pd.Series(col)
        delta[key] = pd.DataFrame(tab)
    rows = []
    for key in DKEYS:
        t = delta[key].dropna()
        for k in arms:
            rows.append({"subset": name, "arm": k, "key": key,
                         "n_common": len(t) if k in valid else 0,
                         "median_common": (float(t[k].median())
                                           if k in valid and len(t) else np.nan)})
    for key in ("profile_dissim", "gap_dtw"):
        for k in arms:
            v, n = wtab_getter(k, key)
            rows.append({"subset": name, "arm": k, "key": key,
                         "n_common": n, "median_common": v})
    sub = pd.DataFrame(rows)
    piv = sub[sub.key.isin(COMP_KEYS)].pivot_table(
        index="key", columns="arm", values="median_common")
    norm = piv.div(piv.max(axis=1), axis=0)
    comp = {k: (float(norm[k].mean()) if k in norm.columns and
                norm[k].notna().any() and k in valid else np.nan)
            for k in arms}
    return rows, comp


def main():
    t0 = time.time()
    recompute_all()

    # rescued GT scenarios
    print("\n== GT rescued (old pet=inf → new finite)")
    for name, wsuf, _ in SUBSETS:
        p_old = SR / "results" / f"{name}_descriptors.parquet"
        if not p_old.exists():
            p_old = R / f"real_desc_{name}.parquet"
        old = pd.read_parquet(p_old)
        if "set" in old.columns:
            old = old[old.set == "real"]
        old = old.set_index("scenario_id")
        new = load_g2(name, wsuf, "real", None)
        resc = [s for s in new.index if s in old.index
                and not np.isfinite(old.loc[s, "pet"])
                and np.isfinite(new.loc[s, "pet"])]
        if resc:
            print(f"  {name}: {len(resc)} rescued -> "
                  + ", ".join(f"{s}(pet {new.loc[s,'pet']:.2f})" for s in resc))

    # ---- (1) best-d, g2 ----------------------------------------------------
    print("\n== best d (g2, 6-key)")
    alld, allc = [], []
    for name, wsuf, _ in SUBSETS:
        arms = {f"±{s}": load_g2(name, wsuf, ("step", s), None) for s in STEPS}
        wt = pd.read_csv(R / f"stepm_sweep_{name}{wsuf}.csv") \
            if (R / f"stepm_sweep_{name}{wsuf}.csv").exists() else None

        def wg(k, key):
            if wt is None:
                return np.nan, 0
            m = wt[(wt.step == int(k.strip("±"))) & (wt.key == key)]
            return ((float(m.mae_median.iloc[0]), int(m.n.iloc[0]))
                    if len(m) else (np.nan, 0))
        rows, comp = composite_table(name, wsuf, arms, wg)
        alld += rows
        best = min((v, k) for k, v in comp.items() if np.isfinite(v))
        allc += [{"subset": name, "arm": k, "composite": v}
                 for k, v in comp.items()]
        print(f"  {name:20s} best d = {best[1]} ({best[0]:.3f})   "
              + " ".join(f"{k}={v:.3f}" for k, v in comp.items()
                         if np.isfinite(v)))
    pd.DataFrame(alld).to_csv(R / "best_d_summary_g2.csv", index=False)
    pd.DataFrame(allc).to_csv(R / "best_d_composite_g2.csv", index=False)

    # ---- (2) best anchor combo, g2 ----------------------------------------
    print("\n== best pos×frm (g2, 6-key; legacy = d-sweep arm at anchor step)")
    alla, allac = [], []
    for name, wsuf, astep in SUBSETS:
        arms = {c: load_g2(name, wsuf, ("combo", c), None) for c in COMBOS}
        arms["legacy"] = load_g2(name, wsuf, ("step", astep), None)
        wt = pd.read_csv(R / f"anchor_sweep_{name}{wsuf}.csv") \
            if (R / f"anchor_sweep_{name}{wsuf}.csv").exists() else None

        def wg(k, key):
            if wt is None:
                return np.nan, 0
            pos, frm = ("legacy", "legacy") if k == "legacy" else (k[0], k[1])
            m = wt[(wt.pos == pos) & (wt.frm == frm) & (wt.key == key)]
            return ((float(m.mae_median.iloc[0]), int(m.n.iloc[0]))
                    if len(m) else (np.nan, 0))
        rows, comp = composite_table(name, wsuf, arms, wg)
        alla += rows
        allac += [{"subset": name, "arm": k, "composite": v}
                  for k, v in comp.items()]
        best = min((v, k) for k, v in comp.items() if np.isfinite(v))
        leg = comp.get("legacy", np.nan)
        print(f"  {name:20s} best = {best[1]} ({best[0]:.3f})   "
              f"legacy = {leg:.3f}")
    pd.DataFrame(alla).to_csv(R / "best_anchor_summary_g2.csv", index=False)
    pd.DataFrame(allac).to_csv(R / "best_anchor_composite_g2.csv", index=False)

    # PET n_common before/after
    print("\n== PET n_common (anchor sweep): v1 -> g2")
    old = pd.read_csv(R / "best_anchor_summary.csv")
    new = pd.DataFrame(alla)
    for name, wsuf, _ in SUBSETS:
        o = old[(old.subset == name) & (old.key == "pet") & old.valid]
        n = new[(new.subset == name) & (new.key == "pet") & (new.n_common > 0)]
        if len(o) and len(n):
            print(f"  {name:20s} {int(o.n_common.max())} -> "
                  f"{int(n.n_common.max())}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
