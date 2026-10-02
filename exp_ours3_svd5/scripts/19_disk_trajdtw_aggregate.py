#!/usr/bin/env python3
"""E10 stage 3: v5 aggregation (exp_cross_coverage/scripts/180_trajdtw_aggregate.py, key set
pet / min_dist / conflict_angle / conflict_point / conflict_speed(=agent_arr_speed) / traj_dtw,
gate-v2 _g2 descriptors, paired |delta| medians on the per-key common set, composite = mean
over keys of median / worst-arm median) applied to the Euclidean-DISK renders of
18_disk_anchor_sweep.py, side by side with the cached ARC-LENGTH arms.

The arc side re-runs the same code on the exp_cross_coverage artifacts (must reproduce
best_d_summary_v5.csv / best_anchor_summary_v5.csv; the max deviation is printed and stored
in the report). The v5 pinning rule (anchor arms evaluated at the RENDERED step) is kept.

Outputs (results/disk_sweep/): best_d_summary_v5_disk.csv, best_anchor_summary_v5_disk.csv
(same columns as the v5 files), arc_vs_disk_sweep.csv (subset, table, arm, key, arc_value,
disk_value, n_common_arc, n_common_disk), DISK_SWEEP_REPORT.md.
Usage: python -B scripts/19_disk_trajdtw_aggregate.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
EXP = ROOT / "exp_cross_coverage"
SR = ROOT / "sr-tlkeep-experiment"
R = EXP / "results"                      # arc artifacts + real_desc_{name}_g2
BAK3 = R / "_bak_anchor_s3"
DS = PROJECT / "results" / "disk_sweep"  # disk artifacts (18)
DS_BAK3 = DS / "_anchor_s3"
SRC_180 = EXP / "scripts" / "180_trajdtw_aggregate.py"
# 2026-09-14: cutinl scenarios relabelled away from cut-in-left (same list as scripts/10 EXCLUDED_SCENARIOS).
# Renders are deterministic per scenario, so they are dropped from the real reference at aggregation time
# (every per-key common set intersects with it) instead of re-rendering the unchanged scenarios.
EXCLUDED_SCENARIOS = {"cutinl": {"39_180", "39_189", "1669_1657"}}


def drop_excluded(name, df):
    drop = EXCLUDED_SCENARIOS.get(name, set())
    return df if df is None or not drop else df[~df.index.astype(str).isin(drop)]
STEPS = [10, 5, 3, 2]
COMBOS = [P + F for P in "pmx" for F in "pmx"]
KEYS = ["pet", "min_dist", "conflict_angle", "conflict_point", "conflict_speed", "traj_dtw"]
ANCH_LAB = {"p": "minPET", "m": "minDist", "x": "crossTraj"}
SUBSETS = [
    ("cutinl", "", "cutin (agent left cut-in, n=61)"),
    ("keeptl", "", "TL N->E (agent left turn, n=50)"),
    ("keeptl_sw", "", "TL S->W (agent left turn, n=82)"),
    ("special_39_180", "", "special_39_180 (single scenario)"),
    ("special_1786_1797", "", "special ego1786 x moto1797 (single scenario)"),
    ("uturn_859_881", "_w8", "special_Uturn_859_881 (W=8, single scenario)"),
]
# 180.main RENDERED (anchor renders exist only at these steps; pin within a 0.01 tie band)
RENDERED = {"cutinl": 10, "keeptl": 10, "keeptl_sw": 10, "special_39_180": 10,
            "special_1786_1797": (3, 5), "uturn_859_881": 5}
# 05_sc.tex quoted v5 (arc) statements, for the "do conclusions change" check
PAPER = {
    "cutinl": dict(best="xx", best_val=0.421, legacy=0.610, best_d=10),
    "keeptl": dict(best="xm", best_val=0.541, legacy=0.619, best_d=10),
    "keeptl_sw": dict(best="pm", best_val=0.553, legacy=0.491, best_d=5, d_note="L=5 0.7141 vs L=10 0.7147"),
}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_180_functions() -> dict:
    """AST-load the v5 scorer pieces of 180 verbatim (no re-derivation)."""
    names = ["arc_resample", "dtw2", "traj_dtw", "load_desc", "paired_delta", "aggregate", "build_arm"]
    ns = {"np": np, "pd": pd, "Path": Path, "KEYS": KEYS}
    parsed = ast.parse(SRC_180.read_text())
    body = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), {n.name for n in body}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC_180), "exec"), ns)
    ns["R"] = R
    ns["real_g2"] = lambda name: drop_excluded(name, ns["load_desc"](R / f"real_desc_{name}_g2.parquet"))   # 180.real_g2
    return ns


def real_g2(F, name):
    return drop_excluded(name, F["load_desc"](R / f"real_desc_{name}_g2.parquet"))


def data_dir(name):
    p = EXP / "data" / name
    return p if p.exists() else SR / "data" / name


def step_files(name, wsuf, s, source):
    if source == "arc":                      # 180.step_files
        d = R / f"cp3s{s}_descriptors_{name}{wsuf}_g2.parquet"
        t = (R / f"cp3_trajectories_{name}.parquet" if (s == 10 and not wsuf)
             else R / f"cp3s{s}_trajectories_{name}{wsuf}.parquet")
        return d, t
    return (DS / f"cp3s{s}_descriptors_{name}{wsuf}_disk_g2.parquet",
            DS / f"cp3s{s}_trajectories_{name}{wsuf}_disk.parquet")


def combo_files(name, wsuf, c, astep, source):
    s3 = name == "special_1786_1797" and astep == 3
    if source == "arc":                      # 180.combo_files
        base = BAK3 if s3 else R
        return (base / f"anch{c}_descriptors_{name}{wsuf}_g2.parquet",
                base / f"anch{c}_trajectories_{name}{wsuf}.parquet")
    base = DS_BAK3 if s3 else DS
    return (base / f"anch{c}_descriptors_{name}{wsuf}_disk_g2.parquet",
            base / f"anch{c}_trajectories_{name}{wsuf}_disk.parquet")


def run_source(F, source, log):
    """180.main aggregation for one artifact source ('arc' | 'disk'). Returns
    (best_d rows, best_anchor rows, {name: (piv_d, piv_a, astep)})."""
    out_d, out_a, pivs = [], [], {}
    for name, wsuf, label in SUBSETS:
        rt = pd.read_parquet(data_dir(name) / "real_tracks.parquet")
        gt = {sid: g.sort_values("frame") for (sid, role), g in rt.groupby(["scenario_id", "role"]) if role == "actor"}
        rd = real_g2(F, name)
        arms = {f"±{s}": F["build_arm"](*step_files(name, wsuf, s, source), gt) for s in STEPS}
        piv = F["aggregate"](name, arms)
        ok = piv.composite.dropna()
        bd = ok.idxmin() if len(ok) else None
        best_step = int(bd.strip("±")) if bd else 10
        rend = RENDERED[name]
        rend = rend if isinstance(rend, tuple) else (rend,)
        pin_note = ""
        if best_step not in rend:
            pin = min(rend, key=lambda s: abs(ok.get(f"±{s}", np.inf) - ok.min()))
            if abs(ok.get(f"±{pin}", np.inf) - ok.min()) < 0.01:
                pin_note = f"best d {bd} ties ±{pin} (Δ<0.01) — anchors evaluated at ±{pin}"
            else:
                pin_note = f"best d {bd} has NO anchor renders — evaluating at ±{pin}"
            best_step = pin
            log(f"    [{source}] {name}: {pin_note}")
        for arm, r in piv.iterrows():
            out_d.append({"subset": name, "arm": arm, **{k: r[k] for k in KEYS},
                          "composite": r.composite, "cov_med": r.cov_med})
        log(f"[{source}][d] {name:20s} best d = {bd} " + " ".join(f"{k}={v:.3f}" for k, v in ok.items()))
        astep = best_step
        arms_a = {c: F["build_arm"](*combo_files(name, wsuf, c, astep, source), gt) for c in COMBOS}
        arms_a["legacy"] = F["build_arm"](*step_files(name, wsuf, astep, source), gt)
        missing = [c for c, (d, _t, _c) in arms_a.items() if d is None]
        if missing:
            log(f"    [{source}] {name}: anchor artifacts missing at d{astep}: {missing}")
        piv_a = F["aggregate"](name, arms_a)
        for arm, r in piv_a.iterrows():
            out_a.append({"subset": name, "astep": astep, "arm": arm, **{k: r[k] for k in KEYS},
                          "composite": r.composite, "cov_med": r.cov_med})
        alt = {}
        for other in rend:                      # secondary anchor tables at the other RENDERED steps
            if other == astep:
                continue
            arms_o = {c: F["build_arm"](*combo_files(name, wsuf, c, other, source), gt) for c in COMBOS}
            arms_o["legacy"] = F["build_arm"](*step_files(name, wsuf, other, source), gt)
            if all(d is None for d, _t, _c in arms_o.values()):
                continue
            piv_o = F["aggregate"](name, arms_o)
            alt[other] = piv_o
            for arm, r in piv_o.iterrows():
                out_a.append({"subset": name, "astep": other, "arm": arm, **{k: r[k] for k in KEYS},
                              "composite": r.composite, "cov_med": r.cov_med})
        pivs[name] = dict(piv_d=piv, piv_a=piv_a, astep=astep, best_d=bd, pin_note=pin_note,
                          missing=missing, n_real=len(rd), label=label, alt=alt)
    return pd.DataFrame(out_d), pd.DataFrame(out_a), pivs


def status_counts(name):
    p = DS / f"disk_sweep_status_{name}.csv"
    if not p.exists():
        return None
    st = pd.read_csv(p)
    st = st[~st.scenario.astype(str).isin(EXCLUDED_SCENARIOS.get(name, set()))].copy()
    st["group"] = np.select(
        [st.status.eq("ok"), st.status.eq("teleport"), st.status.str.startswith("unavailable"),
         st.status.str.startswith("not_reproducible"), st.status.str.startswith("render_fail")],
        ["ok", "teleport", "unavailable", "not_reproducible", "render_fail"], default="other")
    tab = st.groupby(["arm", "group"]).size().unstack(fill_value=0)
    for c in ["ok", "teleport", "unavailable", "not_reproducible", "render_fail", "other"]:
        if c not in tab.columns:
            tab[c] = 0
    tab["n"] = tab[["ok", "teleport", "unavailable", "not_reproducible", "render_fail", "other"]].sum(axis=1)
    tab["base_3cp"] = st[st.base_n_weighted == 3].groupby("arm").size().reindex(tab.index).fillna(0).astype(int)
    tab["base_lt3cp"] = st[st.base_n_weighted.notna() & (st.base_n_weighted < 3)].groupby("arm").size() \
        .reindex(tab.index).fillna(0).astype(int)
    det = st[st.status.str.startswith("unavailable")].groupby(["arm", "status"]).size()
    return tab, det, st


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def main():
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    F = load_180_functions()
    d_arc, a_arc, p_arc = run_source(F, "arc", log)
    d_disk, a_disk, p_disk = run_source(F, "disk", log)
    DS.mkdir(parents=True, exist_ok=True)
    d_disk.to_csv(DS / "best_d_summary_v5_disk.csv", index=False)
    a_disk.to_csv(DS / "best_anchor_summary_v5_disk.csv", index=False)

    # arc reproduction check vs the frozen v5 tables
    v5d = pd.read_csv(R / "best_d_summary_v5.csv")
    v5a = pd.read_csv(R / "best_anchor_summary_v5.csv")
    md = d_arc.merge(v5d, on=["subset", "arm"], suffixes=("", "_v5"))
    a_arc_primary = a_arc[[r.astep == p_arc[r.subset]["astep"] for r in a_arc.itertuples()]]   # v5 has primary astep only
    ma = a_arc_primary.merge(v5a, on=["subset", "arm"], suffixes=("", "_v5"))
    dev_d = max(float(np.nanmax(np.abs(md[k] - md[f"{k}_v5"]))) if md[k].notna().any() else 0.0
                for k in KEYS + ["composite"])
    dev_a = max(float(np.nanmax(np.abs(ma[k] - ma[f"{k}_v5"]))) if ma[k].notna().any() else 0.0
                for k in KEYS + ["composite"])
    astep_same = bool((ma.astep == ma.astep_v5).all())
    log(f"arc reproduction vs v5 csv: max|Δ| best_d={dev_d:.2e} best_anchor={dev_a:.2e} astep identical={astep_same}")

    # side-by-side table
    rows = []
    for table, A, D, PA, PD in (("best_d", d_arc, d_disk, p_arc, p_disk), ("best_anchor", a_arc, a_disk, p_arc, p_disk)):
        for name, *_ in SUBSETS:
            pa = PA[name]["piv_d" if table == "best_d" else "piv_a"]
            pdk = PD[name]["piv_d" if table == "best_d" else "piv_a"]
            na, nd_ = pa.attrs["n"], pdk.attrs["n"]
            arms = list(pa.index)
            for arm in arms:
                for key in KEYS + ["composite", "cov_med"]:
                    rows.append({"subset": name, "table": table, "arm": arm, "key": key,
                                 "astep_arc": PA[name]["astep"] if table == "best_anchor" else np.nan,
                                 "astep_disk": PD[name]["astep"] if table == "best_anchor" else np.nan,
                                 "arc_value": float(pa.loc[arm, key]) if arm in pa.index else np.nan,
                                 "disk_value": float(pdk.loc[arm, key]) if arm in pdk.index else np.nan,
                                 "n_common_arc": na.get(key, np.nan) if key in KEYS else np.nan,
                                 "n_common_disk": nd_.get(key, np.nan) if key in KEYS else np.nan})
    side = pd.DataFrame(rows)
    side.to_csv(DS / "arc_vs_disk_sweep.csv", index=False)

    # ---- report -------------------------------------------------------------
    rep = ["# Disk-window sweep report (E10 stage 3)", "",
           "Scorer: v5 (180_trajdtw_aggregate, AST-loaded verbatim) — keys pet / min_dist / conflict_angle / "
           "conflict_point / conflict_speed(=agent_arr_speed) / traj_dtw, gate-v2 `_g2` descriptors, paired |Δ| "
           "medians on the per-key common set, composite = mean over keys of median / worst-arm median (lower = better). "
           "Anchor arms are pinned to the RENDERED step exactly as in v5.", "",
           f"Arc side re-run through the same code reproduces the frozen v5 csv files to max|Δ| "
           f"{dev_d:.1e} (best_d) / {dev_a:.1e} (best_anchor); astep identical: {astep_same}.", "",
           "Window geometry: **disk** = q-/q+ at Euclidean distance L from q_c (chords == L, "
           "16_disk_window_extract.disk_points), q_c = target sample at the arm's own anchor frame; **arc** = the "
           "cached ±step arc-length arms of exp_cross_coverage (v5). Both sides use the same base xosc files, "
           "esmini, FPS, teleport screen, trim and descriptor chain.", ""]
    conclusions = []
    for name, wsuf, label in SUBSETS:
        A, D = p_arc[name], p_disk[name]
        rep.append(f"## {name} — {label}")
        rep.append("")
        # best d
        da, dd_ = A["piv_d"].composite.dropna(), D["piv_d"].composite.dropna()
        best_da = da.idxmin() if len(da) else None
        best_dd = dd_.idxmin() if len(dd_) else None
        rep.append(f"**Best L (d-sweep)** — disk: {best_dd} ({fmt(dd_.min() if len(dd_) else None)}) ; "
                   f"arc/v5: {best_da} ({fmt(da.min() if len(da) else None)}).")
        rep.append("")
        rep.append("| L | arc composite | disk composite | n_pet arc | n_pet disk | n_dtw arc | n_dtw disk |")
        rep.append("|---|---|---|---|---|---|---|")
        for s in STEPS:
            k = f"±{s}"
            rep.append(f"| {k} | {fmt(A['piv_d'].composite.get(k))} | {fmt(D['piv_d'].composite.get(k))} | "
                       f"{A['piv_d'].attrs['n']['pet']} | {D['piv_d'].attrs['n']['pet']} | "
                       f"{A['piv_d'].attrs['n']['traj_dtw']} | {D['piv_d'].attrs['n']['traj_dtw']} |")
        rep.append("")
        if D["pin_note"]:
            rep.append(f"Pinning (disk): {D['pin_note']}.")
        if A["pin_note"]:
            rep.append(f"Pinning (arc): {A['pin_note']}.")
        # anchors
        aa, ad = A["piv_a"].composite.dropna(), D["piv_a"].composite.dropna()
        aa9, ad9 = aa.drop("legacy", errors="ignore"), ad.drop("legacy", errors="ignore")
        best_aa = aa9.idxmin() if len(aa9) else None
        best_ad = ad9.idxmin() if len(ad9) else None
        leg_a, leg_d = aa.get("legacy", np.nan), ad.get("legacy", np.nan)
        rep.append("")
        rep.append(f"**Best (h_p, h_t) at d=±{D['astep']}** — disk: {best_ad} ({fmt(ad9.min() if len(ad9) else None)}), "
                   f"legacy {fmt(leg_d)} ; arc/v5 (d=±{A['astep']}): {best_aa} ({fmt(aa9.min() if len(aa9) else None)}), "
                   f"legacy {fmt(leg_a)}.")
        if D["missing"]:
            rep.append(f"Disk anchor arms missing / not reproducible at d=±{D['astep']}: {D['missing']}.")
        rep.append("")
        rep.append("| arm | arc composite | disk composite | disk pet | disk min_dist | disk angle | disk cpoint | disk speed | disk traj_dtw |")
        rep.append("|---|---|---|---|---|---|---|---|---|")
        for arm in list(COMBOS) + ["legacy"]:
            ra = A["piv_a"].loc[arm] if arm in A["piv_a"].index else None
            rd_ = D["piv_a"].loc[arm] if arm in D["piv_a"].index else None
            lab = arm if arm == "legacy" else f"{arm} ({ANCH_LAB[arm[0]]}/{ANCH_LAB[arm[1]]})"
            rep.append(f"| {lab} | {fmt(ra.composite if ra is not None else None)} | {fmt(rd_.composite if rd_ is not None else None)} | "
                       + " | ".join(fmt(rd_[k] if rd_ is not None else None) for k in KEYS) + " |")
        rep.append("")
        rep.append(f"Common-set sizes (anchor table): pet arc {A['piv_a'].attrs['n']['pet']} / disk {D['piv_a'].attrs['n']['pet']}; "
                   f"traj_dtw arc {A['piv_a'].attrs['n']['traj_dtw']} / disk {D['piv_a'].attrs['n']['traj_dtw']}.")
        nd_min = min(D['piv_a'].attrs['n'].values())
        na_min = min(A['piv_a'].attrs['n'].values())
        if A['n_real'] > 1 and (nd_min < 20 or nd_min < 0.5 * na_min):
            rep.append(f"**Caveat: the disk anchor composite rests on a common set of only {nd_min} scenarios "
                       f"(arc: {na_min}) because availability differs per anchor (see status table); the arm ranking "
                       f"is NOT decision-grade at this n.**")
        for other, piv_o in sorted(D.get("alt", {}).items()):
            oo = piv_o.composite.dropna()
            oo9 = oo.drop("legacy", errors="ignore")
            rep.append("")
            rep.append(f"Secondary disk anchor table at d=±{other} (also rendered): best {oo9.idxmin() if len(oo9) else None} "
                       f"({fmt(oo9.min() if len(oo9) else None)}), legacy {fmt(oo.get('legacy', np.nan))}; "
                       + ", ".join(f"{k}={fmt(v)}" for k, v in oo.items()) + ".")
        # status counts
        sc = status_counts(name)
        if sc is not None:
            tab, det, st = sc
            rep.append("")
            rep.append("Disk render status per arm (n = scenarios in real_meta; base_3cp / base_lt3cp = base xosc kept 3 / fewer weighted CPs):")
            rep.append("")
            rep.append("| arm | n | ok | teleport | unavailable | render_fail | not_reproducible | other | base_3cp | base_lt3cp |")
            rep.append("|---|---|---|---|---|---|---|---|---|---|")
            for arm, r in tab.iterrows():
                rep.append(f"| {arm} | {r.n} | {r.ok} | {r.teleport} | {r.unavailable} | {r.render_fail} | "
                           f"{r.not_reproducible} | {r.other} | {r.base_3cp} | {r.base_lt3cp} |")
            if len(det):
                rep.append("")
                rep.append("Unavailable breakdown: " + "; ".join(f"{a}: {s.replace('unavailable:', '')}={n}"
                                                                for (a, s), n in det.items()))
            oth = st[~st.status.isin(["ok", "teleport"]) & ~st.status.str.startswith("unavailable")
                     & ~st.status.str.startswith("not_reproducible")]
            if len(oth):
                rep.append("")
                rep.append("Other statuses: " + "; ".join(f"{a}/{sid}: {s}" for a, sid, s in
                                                          oth[["arm", "scenario", "status"]].itertuples(index=False)))
        # conclusion check
        pp = PAPER.get(name)
        if pp:
            changed = []
            if best_ad != pp["best"]:
                changed.append(f"best (h_p,h_t) {pp['best']} -> {best_ad}")
            beats_arc = np.isfinite(leg_a) and aa9.min() < leg_a
            beats_disk = np.isfinite(leg_d) and len(ad9) and ad9.min() < leg_d
            if bool(beats_arc) != bool(beats_disk):
                changed.append(f"best-combo-vs-legacy verdict flips (arc: {'beats' if beats_arc else 'loses to'} legacy; "
                               f"disk: {'beats' if beats_disk else 'loses to'} legacy)")
            if best_dd is not None and int(best_dd.strip('±')) != pp["best_d"]:
                changed.append(f"best L ±{pp['best_d']} -> {best_dd}")
            verdict = ("UNCHANGED" if not changed else "CHANGED: " + "; ".join(changed))
            conclusions.append((name, verdict, pp, best_ad, ad9.min() if len(ad9) else np.nan, leg_d, best_dd,
                                dd_.min() if len(dd_) else np.nan))
            rep.append("")
            rep.append(f"05_sc.tex statement (arc/v5): best {pp['best']} {pp['best_val']:.3f} vs legacy {pp['legacy']:.3f}, "
                       f"best L ±{pp['best_d']}{(' (' + pp['d_note'] + ')') if 'd_note' in pp else ''}. "
                       f"Disk rule: best {best_ad} {fmt(ad9.min() if len(ad9) else None)} vs legacy {fmt(leg_d)}, "
                       f"best L {best_dd} {fmt(dd_.min() if len(dd_) else None)}. **Conclusion {verdict}.**")
        rep.append("")

    rep.append("## Summary vs 05_sc.tex")
    rep.append("")
    rep.append("| subset | paper (arc) | disk | verdict |")
    rep.append("|---|---|---|---|")
    for name, verdict, pp, best_ad, bv, leg_d, best_dd, dv in conclusions:
        rep.append(f"| {name} | best {pp['best']} {pp['best_val']:.3f} / legacy {pp['legacy']:.3f} / L ±{pp['best_d']} | "
                   f"best {best_ad} {fmt(bv)} / legacy {fmt(leg_d)} / L {best_dd} {fmt(dv)} | {verdict} |")
    rep.append("")
    rep.append("Caveats: (1) the disk arms re-derive q_c from the generator's own anchor rule and patch the full 3-point "
               "window even where the legacy sampler had dropped interior CPs (base_lt3cp), so the disk common set is not "
               "the arc common set — n_common columns in arc_vs_disk_sweep.csv give both sizes; (2) anchor frames "
               "outside the annotated window (pipeline PET-window bug) and windows that end inside the disk are "
               "'unavailable' under the paper rule and are not rendered; (3) special_1786_1797 s3 p-anchored bases "
               "have no _pf2 namespace on disk (pre-2026-08-15 PET-frame fix), so those 5 arms are not reproducible "
               "at astep 3 — only relevant if the disk pinning selects ±3.")
    rep.append("")
    rep.append("Log of this aggregation run:")
    rep.append("")
    rep.extend(f"    {l}" for l in lines)
    (DS / "DISK_SWEEP_REPORT.md").write_text("\n".join(rep) + "\n")
    manifest = dict(scorer_source=str(SRC_180), scorer_sha256=sha256(SRC_180), script_sha256=sha256(Path(__file__)),
                    arc_reproduction_max_abs_dev=dict(best_d=dev_d, best_anchor=dev_a, astep_identical=astep_same),
                    real_desc=[{"path": str(R / f"real_desc_{n}_g2.parquet"), "sha256": sha256(R / f"real_desc_{n}_g2.parquet")}
                               for n, *_ in SUBSETS])
    (DS / "19_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("\n".join(rep[rep.index("## Summary vs 05_sc.tex"):]))


if __name__ == "__main__":
    main()
