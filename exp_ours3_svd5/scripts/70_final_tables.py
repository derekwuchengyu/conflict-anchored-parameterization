#!/usr/bin/env python3
"""70 — final table assembly (E11 assembly half): T1 … T9 as tidy CSVs + booktabs LaTeX + manifest.

Reads ONLY existing verified result files (no science is recomputed; every number is copied
or trivially aggregated — sums over classes, seed means, min/max of stored per-seed values).
Writes ONLY under results/final/ (T1.csv … T9.csv, latex/T1.tex … T9.tex, manifest.json).

Caveat flag columns on every table (0/1):
  tautology        — anchor instant coincides with the descriptor instant (Table 2 flag), or a
                     quantity that is precise partly by construction (u_c sampled on the EndSpeed
                     plateau, Table 6a Block C / v_end axis)
  in_sample        — the arm was fitted on the same scenes it is scored against (SVD fullfit
                     basis / both KDEs) or the sweep selects and compares on the same subset
  single_seed      — only seed 20260910 of the ours3 KDE is scored (seeds 20260911/12 pending)
  analytic         — SVD d5 rows are algebraic 50-point decodes, not esmini executions
  exposure_unequal — arms differ in sample count, centre count, class mix or scene set
  pending_decision — user decision still open (SVD executed window vs polytrunc as Table 5/7 primary)

Run:  python3 scripts/70_final_tables.py        (from anywhere; paths are absolute)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/hcis-s19/Documents/ChengYu/exp_ours3_svd5")
RES = ROOT / "results"
OUT = RES / "final"
TEX = OUT / "latex"
OUT.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)

CLASSES = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr"]
CLASS_TEX = {"tlkeep": "tlkeep", "keeptl": "keeptl", "keeptl_sw": "keeptl\\_sw", "cutinl": "cutinl",
             "cutinr": "cutinr", "pooled": "pooled"}
MEASURE_TEX = {"pet": "$|\\Delta\\mathrm{PET}|$", "dmin": "$|\\Delta d_{\\min}|$", "alpha": "$|\\Delta\\alpha|$",
               "cpoint": "$\\|\\Delta\\mathbf c\\|$", "uc": "$|\\Delta u_c|$", "dtw": "DTW(target)"}
FLAG_COLS = ["tautology", "in_sample", "single_seed", "analytic", "exposure_unequal", "pending_decision"]

ARM_LABEL = {
    "real": "real replay (recorded target)",
    "ours3_disk": "ours3 disk defaults (executed)",
    "ours3_arc": "ours3 arc defaults (executed, sensitivity)",
    "ours3_disk_kde": "ours3 disk + KDE (executed, seed 20260910)",
    "ours3_disk_defaults": "ours3 disk defaults (executed)",
    "ours3_disk_gttrunc": "ours3 disk defaults, GT-support truncated (executed, sensitivity E4-S)",
    "sakura_bc": "SAKURA bc (executed)",
    "svd_d5_fullfit": "SVD d5 fullfit (analytic decode, in-sample)",
    "svd_d5_logo": "SVD d5 LOGO (analytic decode, held-out group)",
    "svd_d5_fullfit_recon": "SVD d5 fullfit reconstruction (analytic, in-sample ceiling)",
    "svd_d5_logo_recon": "SVD d5 LOGO reconstruction (analytic, held-out-group ceiling)",
    "svd_d5_kde": "SVD d5 + matched KDE (analytic decode, 3 seeds)",
    "svd_d5_kde_matched_analytic": "SVD d5 + matched KDE (analytic decode, 3 seeds)",
    "svd_d5_kde_executed_E3": "SVD executed (timed Polyline, E3) KDE, first 100/class/seed",
    "svd_exec_E3_kde_kde": "SVD executed (timed Polyline, E3) KDE (window, batch partial at scoring)",
    "svd_exec_E3_recon_fullfit": "SVD executed (timed Polyline, E3) fullfit recon (window)",
    "svd_exec_E3_recon_logo": "SVD executed (timed Polyline, E3) LOGO recon (window)",
    "svd_exec_E3_smoke_fullfit": "SVD executed (timed Polyline, E3) smoke fullfit (window, n=25)",
    "svd_exec_E3_smoke_polytrunc_fullfit": "SVD executed (timed Polyline, E3) smoke fullfit (polytrunc, n=25)",
}
SVD_ANALYTIC = {"svd_d5_fullfit", "svd_d5_logo", "svd_d5_fullfit_recon", "svd_d5_logo_recon", "svd_d5_kde",
                "svd_d5_kde_matched_analytic", "svd_e7_oat", "svd_e7_target"}
E3_ARMS = {"svd_d5_kde_executed_E3", "svd_exec_E3_kde_kde", "svd_exec_E3_recon_fullfit", "svd_exec_E3_recon_logo",
           "svd_exec_E3_smoke_fullfit", "svd_exec_E3_smoke_polytrunc_fullfit"}

SOURCES: dict[str, dict] = {}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reg(p: Path, rows=None) -> Path:
    p = Path(p)
    if str(p) not in SOURCES:
        SOURCES[str(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size, "rows": rows}
    elif rows is not None and SOURCES[str(p)]["rows"] is None:
        SOURCES[str(p)]["rows"] = rows
    return p


def rcsv(p, **kw) -> pd.DataFrame:
    p = Path(p)
    d = pd.read_csv(p, **kw)
    reg(p, rows=int(len(d)))
    return d


def rjson(p):
    p = Path(p)
    with open(p) as f:
        d = json.load(f)
    reg(p, rows=(len(d) if isinstance(d, (list, dict)) else None))
    return d


def flags(**kw) -> dict:
    return {c: int(bool(kw.get(c, 0))) for c in FLAG_COLS}


def fmt(x, nd=3, dash="--"):
    if x is None:
        return dash
    try:
        if isinstance(x, str):
            return x
        if x != x or (isinstance(x, float) and math.isinf(x)):
            return dash
    except TypeError:
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def texesc(s) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write_tex(name: str, caption: str, label: str, colspec: str, header: list[str], rows: list[list[str]],
              notes: list[str], wide: bool = True, size: str = "\\footnotesize", midrules: set[int] | None = None,
              colsep_pt: float | None = None):
    env = "table*" if wide else "table"
    lines = [f"% auto-generated by scripts/70_final_tables.py — do not edit by hand",
             f"\\begin{{{env}}}[t]", "\\centering", size]
    if colsep_pt is not None:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{colsep_pt}pt}}")
    lines += [
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if midrules and i in midrules and i > 0:
            lines.append("\\midrule")
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append("\\begin{flushleft}\\scriptsize")
        for n in notes:
            lines.append(n + "\\\\")
        lines.append("\\end{flushleft}")
    lines.append(f"\\end{{{env}}}")
    (TEX / name).write_text("\n".join(lines) + "\n")


def flag_marks(f: dict) -> str:
    m = {"tautology": "t", "in_sample": "i", "single_seed": "s", "analytic": "a", "exposure_unequal": "e",
         "pending_decision": "p"}
    s = "".join(m[k] for k in FLAG_COLS if f.get(k))
    return f"\\textsuperscript{{{s}}}" if s else ""


FLAG_LEGEND = ("Flags: t = anchor/descriptor tautology or precision by construction; i = in-sample fit; "
               "s = single seed (ours3 KDE seed 20260910 only); a = analytic SVD decode (no simulator); "
               "e = unequal exposure (n, centres, class mix or scene set differ); p = pending user decision "
               "(SVD executed window vs polytrunc).")


def save_csv(name: str, df: pd.DataFrame):
    for c in FLAG_COLS:
        if c not in df.columns:
            df[c] = 0
    df.to_csv(OUT / name, index=False)
    print(f"  wrote {name}: {len(df)} rows")


# ----------------------------------------------------------------------------------------------
# T1 — data / protocol / applicability
# ----------------------------------------------------------------------------------------------
def build_t1():
    pop = rcsv(RES / "ours3_population_summary.csv").set_index("subset")
    disk = rcsv(RES / "disk_window_L10_summary.csv").set_index("subset")
    svd = rcsv(RES / "svd_d5_summary.csv").set_index("subset")
    svdm = rcsv(RES / "svd_d5_matched_summary.csv").set_index("subset")
    sak = rcsv(RES / "sakura_source_summary.csv")
    cases = rcsv(RES / "svd_d5_cases.csv")
    ctx_arc = rjson(RES / "ours3_contexts.json")
    ctx_disk = rjson(RES / "ours3_disk_contexts.json")
    diskpop = rcsv(RES / "ours3_disk_population.csv")
    fit = rjson(ROOT / "plans/ours3_disk_kde_dependent/7e0593f38aafa843/fit.json")
    bw = rcsv(RES / "svd_d5_kde_matched_bandwidth.csv").set_index("subset")

    n_groups_global = int(cases["group_id"].nunique())
    cd = pd.DataFrame(ctx_disk)
    ca = pd.DataFrame(ctx_arc)
    rows = []
    for c in CLASSES + ["pooled"]:
        if c == "pooled":
            sel = cd
            sela = ca
            g = lambda tab, col: int(tab[col].sum())
        else:
            sel = cd[cd.subset == c]
            sela = ca[ca.subset == c]
            g = lambda tab, col: int(tab.loc[c, col])
        sak_n = sak[(sak.subset == c) & (sak.status == "unique_existing_bc_source")]["n"].sum() if c != "pooled" else \
            sak[sak.status == "unique_existing_bc_source"]["n"].sum()
        r = {
            "class": c,
            "n_population_513": g(svd, "n_cases"),
            "n_arc_cohort_489_eligible": g(pop, "eligible"),
            "n_arc_unavailable": g(pop, "unavailable"),
            "n_disk_ok_geometry": g(disk, "ok"),
            "n_disk_qminus_unavailable": g(disk, "q_minus_unavailable"),
            "n_disk_qplus_unavailable": g(disk, "q_plus_unavailable"),
            "n_disk_both_unavailable": g(disk, "both_unavailable"),
            "n_disk_rescued_by_full_support_diag": g(disk, "rescued_by_full_support"),
            "n_disk_cohort_469": g(disk, "in_arc_cohort_and_ok"),
            "n_disk_render_jobs": g(disk, "render_jobs_cohort"),
            "n_svd_d5_matched_eligible": g(svdm, "n_eligible"),
            "n_svd_d5_fullfit_ok": g(svdm, "fullfit_cases_ok"),
            "n_svd_d5_logo_ok": g(svdm, "logo_cases_ok"),
            "n_global_groups_in_class": (int(svdm.loc[c, "n_global_groups_represented_in_class"]) if c != "pooled"
                                         else n_groups_global),
            "n_sakura_bc_available": int(sak_n),
            "C_L_disk_ok_over_population": (g(disk, "ok") / g(svd, "n_cases")),
            "theta1_deg_median": float(sel["theta1_deg"].median()),
            "theta1_deg_min": float(sel["theta1_deg"].min()),
            "theta1_deg_max": float(sel["theta1_deg"].max()),
            "theta2_deg_median": float(sel["theta2_deg"].median()),
            "theta2_deg_min": float(sel["theta2_deg"].min()),
            "theta2_deg_max": float(sel["theta2_deg"].max()),
            "end_speed_kmh_median": float(sel["end_speed_kmh"].median()),
            "end_speed_kmh_min": float(sel["end_speed_kmh"].min()),
            "end_speed_kmh_max": float(sel["end_speed_kmh"].max()),
            "arc_theta1_deg_median_sensitivity": float(sela["theta1_deg"].median()),
            "ours_kde_n_fit": (fit["fits"][c]["n_fit"] if c != "pooled" else sum(v["n_fit"] for v in fit["fits"].values())),
            "ours_kde_h_loo_standardised": (fit["fits"][c]["h"] if c != "pooled" else np.nan),
            "svd_kde_h_loo_latent": (float(bw.loc[c, "h_loo"]) if c != "pooled" else np.nan),
            "svd_kde_n_train": (int(bw.loc[c, "n_train"]) if c != "pooled" else int(bw["n_train"].sum())),
            "window_rule": "Euclidean disk L=10 m (04_impl eq. conflict-window-disk), frozen E0 item 7",
            "svd_arm_kind": "analytic decode (d=5, matched 489 cohort)",
        }
        r.update(flags(analytic=1, exposure_unequal=(r["n_disk_cohort_469"] != r["n_svd_d5_matched_eligible"])))
        rows.append(r)
    df = pd.DataFrame(rows)
    save_csv("T1.csv", df)

    hdr = ["class", "$N$", "arc 489", "disk 469", "$q_-$/$q_+$ n/a", "SVD d5", "groups", "SAKURA",
           "$C_L$", "$\\theta_1$ [$^\\circ$]", "$\\theta_2$ [$^\\circ$]", "$v_{\\rm end}$ [km/h]", "$h$ ours", "$h$ SVD"]
    trs = []
    for _, r in df.iterrows():
        trs.append([CLASS_TEX[r["class"]], fmt(r.n_population_513), fmt(r.n_arc_cohort_489_eligible),
                    fmt(r.n_disk_cohort_469),
                    f"{fmt(r.n_disk_qminus_unavailable)}/{fmt(r.n_disk_qplus_unavailable)}",
                    fmt(r.n_svd_d5_matched_eligible) + "\\textsuperscript{a}", fmt(r.n_global_groups_in_class),
                    fmt(r.n_sakura_bc_available) if r.n_sakura_bc_available else "--",
                    fmt(r.C_L_disk_ok_over_population, 3),
                    f"{r.theta1_deg_median:.1f} [{r.theta1_deg_min:.0f},{r.theta1_deg_max:.0f}]",
                    f"{r.theta2_deg_median:.1f} [{r.theta2_deg_min:.0f},{r.theta2_deg_max:.0f}]",
                    f"{r.end_speed_kmh_median:.1f} [{r.end_speed_kmh_min:.0f},{r.end_speed_kmh_max:.0f}]",
                    fmt(r.ours_kde_h_loo_standardised, 3), fmt(r.svd_kde_h_loo_latent, 3)])
    write_tex("T1.tex", "Data, protocol and applicability per class. $N$ = labelled population (513); arc 489 = "
              "historical arc-length cohort; disk 469 = Euclidean $L=10$\\,m cohort (frozen geometry, E0 item 7); "
              "$q_-$/$q_+$ n/a = scenes whose backward/forward boundary point is unavailable inside the annotated "
              "window; SVD d5 = matched-cohort cases; groups = global ego/target connected components (85 in total, "
              "the CI cluster unit); $C_L$ = disk-geometry availability over the population (485/513 = 0.945); "
              "parameter columns give the median [min, max] of the disk contexts; $h$ = LOO bandwidth of the ours3 "
              "3-D KDE (standardised units) and of the SVD latent KDE.",
              "tab:t1-data", "l" + "r" * 13, hdr, trs,
              ["\\textsuperscript{a} SVD d5 rows are analytic decodes fitted on the 489-scene matched cohort; "
               "ours3 rows are esmini executions on the 469-scene disk cohort (exposure unequal by 20 scenes: "
               "keeptl 11, cutinr 7, tlkeep 2).", FLAG_LEGEND], midrules={5}, size="\\scriptsize", colsep_pt=3)


# ----------------------------------------------------------------------------------------------
# T2 — six measures + D_m
# ----------------------------------------------------------------------------------------------
def build_t2():
    sets = {"primary": "table2_six_measures_primary.csv", "extended": "table2_six_measures_extended.csv",
            "sensitivity_arc": "table2_six_measures_sensitivity_arc.csv",
            "sensitivity_gttrunc": "table2_six_measures_sensitivity_gttrunc.csv"}
    reg(RES / "TABLE2_SIX_MEASURES_REPORT.md")
    man = rjson(RES / "table2_manifest.json")
    frames = []
    for role, fn in sets.items():
        d = rcsv(RES / fn)
        d.insert(0, "row_role", role)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    keep = ["row_role", "comparison_set", "scope", "row_type", "arm", "measure", "paper_measure", "unit", "n_cohort",
            "n_eligible", "n_arm_finite", "n_common_Sk", "n_generated_no_event", "n_real_no_event", "n_pet_sign_flip",
            "median_err", "b_k", "normalized", "D_m", "n_measures_in_Dm", "tautology_flag", "contrast", "ours_arm",
            "ref_arm", "n_paired", "n_groups_global", "n_groups_within_class", "ours_median", "ref_median",
            "median_paired_diff", "ci95_low_global_group", "ci95_high_global_group", "ci95_low_within_class_group",
            "ci95_high_within_class_group", "p_signflip_raw", "p_holm", "holm_significant_005", "favours",
            "ci_global_excludes_0", "primary_contrast"]
    df = raw[[c for c in keep if c in raw.columns]].copy()
    df["arm_label"] = df["arm"].map(lambda a: ARM_LABEL.get(a, a) if isinstance(a, str) else a)
    df["stat_unit"] = "median paired diff; 95% cluster bootstrap CI on 85 global groups; cluster sign-flip p; Holm 5x6 per set"
    for c in FLAG_COLS:
        df[c] = 0
    df["tautology"] = df["tautology_flag"].fillna("").astype(str).str.contains("tautology").astype(int)
    df["analytic"] = df.apply(lambda r: int(str(r.get("arm", "")).startswith("svd") or str(r.get("ref_arm", "")).startswith("svd")), axis=1)
    df["in_sample"] = df.apply(lambda r: int("fullfit" in str(r.get("arm", "")) or "fullfit" in str(r.get("ref_arm", ""))), axis=1)
    df["exposure_unequal"] = (df["row_role"] == "extended").astype(int)  # SAKURA subset of classes
    save_csv("T2.csv", df)

    # LaTeX: primary set, per class: ours / fullfit / LOGO medians, D_m, contrast vs LOGO
    prim = raw[raw.row_role == "primary"]
    am = prim[prim.row_type == "arm_measure"]
    ct = prim[(prim.row_type != "arm_measure") & (prim.contrast == "ours3_disk_minus_svd_d5_logo")]
    order = ["pet", "dmin", "alpha", "cpoint", "uc", "dtw"]
    hdr = ["class ($n$)", "measure", "ours3 disk", "SVD fullfit\\textsuperscript{ai}", "SVD LOGO\\textsuperscript{a}",
           "$\\Delta$ ours$-$LOGO [95\\% CI]", "$p_{\\rm Holm}$"]
    trs, mids = [], set()
    for sc in CLASSES + ["pooled"]:
        s = am[am.scope == sc]
        if s.empty:
            continue
        mids.add(len(trs))
        n_el = int(s["n_eligible"].iloc[0])
        first = True
        for m in order:
            sm = s[s.measure == m]
            if sm.empty:
                continue
            vals = {a: float(sm[sm.arm == a]["median_err"].iloc[0]) for a in ["ours3_disk", "svd_d5_fullfit", "svd_d5_logo"]}
            best = min(vals, key=vals.get)
            cells = [("\\textbf{%s}" % fmt(v)) if a == best else fmt(v) for a, v in vals.items()]
            taut = "\\textsuperscript{t}" if "tautology" in str(sm[sm.arm == "ours3_disk"]["tautology_flag"].iloc[0]) else ""
            c = ct[(ct.scope == sc) & (ct.measure == m)]
            if len(c):
                c = c.iloc[0]
                star = "*" if bool(c.holm_significant_005) else ""
                dtxt = f"{c.median_paired_diff:+.3f} [{c.ci95_low_global_group:.2f}, {c.ci95_high_global_group:.2f}]"
                ptxt = f"{c.p_holm:.3f}{star}"
            else:
                dtxt, ptxt = "--", "--"
            label = f"{CLASS_TEX[sc]} ({n_el})" if first else ""
            first = False
            unit = str(sm["unit"].iloc[0])
            trs.append([label, f"{MEASURE_TEX[m]} [{texesc(unit)}]{taut}"] + cells + [dtxt, ptxt])
        dm = {a: float(s[s.arm == a]["D_m"].iloc[0]) for a in ["ours3_disk", "svd_d5_fullfit", "svd_d5_logo"]}
        bestd = min(dm, key=dm.get)
        trs.append(["", "$D_m$"] + [("\\textbf{%s}" % fmt(v)) if a == bestd else fmt(v) for a, v in dm.items()] + ["", ""])
        ne = s[s.measure == "pet"]
        trs.append(["", "PET no-event gen/real"] + [f"{int(ne[ne.arm == a]['n_generated_no_event'].iloc[0])}/{int(ne[ne.arm == a]['n_real_no_event'].iloc[0])}" for a in ["ours3_disk", "svd_d5_fullfit", "svd_d5_logo"]] + ["", ""])
    write_tex("T2.tex", "Six interaction discrepancy measures (05\\_sc.tex subsec:metrics) and composite $D_m$ on the "
              "469-scene disk cohort: ours3 Euclidean-disk defaults (esmini executed) vs matched SVD d5 (analytic "
              "decode; fullfit = in-sample basis, LOGO = scene's global group held out). Medians on the per-measure "
              "common finite set $S_k$; bold = lowest median; $\\Delta$ = median paired difference (negative favours "
              "ours) with 95\\% cluster-bootstrap CI over the 85 global ego/target groups; $p_{\\rm Holm}$ over "
              "5 classes $\\times$ 6 measures (pooled: own 6-cell family); * = Holm-significant at 0.05. Signed "
              "bbox-PET (E0 item 1); min\\_ttc excluded (E0 item 3).",
              "tab:t2-six-measures", "llrrrlr", hdr, trs,
              ["\\textsuperscript{t} anchor tautology: the ours anchor instant coincides with the descriptor instant "
               "(PET anchor = $u_c$ instant in 89\\% of pooled scenes; crossTraj anchor = conflict point and $u_c$ "
               "instant in cutinl). \\textsuperscript{a} analytic decode; \\textsuperscript{i} in-sample. "
               "Sensitivity sets (arc geometry, GT-support truncation, +SAKURA) are in T2.csv; all five non-DTW "
               "measures are scene-wise identical under GT truncation."], midrules=mids, size="\\scriptsize")


# ----------------------------------------------------------------------------------------------
# T3 — coverage / first-hit / variation ; T4 — similarity
# ----------------------------------------------------------------------------------------------
def build_t34():
    reg(RES / "TABLE34_REPORT.md")
    man = rjson(RES / "table34_manifest.json")
    cov = rcsv(RES / "table3_coverage.csv")
    fh = rcsv(RES / "table3_first_hit.csv")
    var = rcsv(RES / "table3_variation.csv")
    sim = rcsv(RES / "table4_similarity.csv")
    seeds_used = man.get("seeds_used", {})

    def fl(arm):
        return flags(in_sample=arm in ("ours3_disk_kde", "svd_d5_kde_matched_analytic", "svd_d5_kde_executed_E3",
                                       "svd_d5_fullfit_recon"),
                     single_seed=(arm == "ours3_disk_kde"), analytic=arm in SVD_ANALYTIC,
                     exposure_unequal=arm in ("ours3_disk_kde", "svd_d5_kde_matched_analytic", "svd_d5_kde_executed_E3"),
                     pending_decision=arm in E3_ARMS)

    rows = []
    # coverage: matched489 + disk469, pool all/valid, eps 1.0 (others kept in the source), seed-aggregated
    c = cov[cov.eps_mult == 1.0]
    grp = ["subset", "real_set", "arm", "pool", "space", "budget"]
    for key, g in c.groupby(grp):
        r = dict(zip(grp, key))
        r.update({"block": "coverage", "n_real": int(g.n_real.iloc[0]), "n_seeds": int(g.seed.nunique()),
                  "seeds": ";".join(str(s) for s in sorted(g.seed.unique())),
                  "n_samples_per_seed": int(g.n_samples.iloc[0]),
                  "value": float(g.coverage_mean.mean()), "value_seed_min": float(g.coverage_mean.min()),
                  "value_seed_max": float(g.coverage_mean.max()),
                  "ordering_min": float(g.coverage_min.mean()), "ordering_max": float(g.coverage_max.mean()),
                  "arm_label": ARM_LABEL.get(key[2], key[2])})
        r.update(fl(key[2]))
        rows.append(r)
    for key, g in fh.groupby(["subset", "real_set", "arm", "pool", "space"]):
        r = dict(zip(["subset", "real_set", "arm", "pool", "space"], key))
        r.update({"block": "first_hit", "budget": int(g.pool_size.iloc[0]), "n_real": int(g.n_real.iloc[0]),
                  "n_seeds": int(g.seed.nunique()), "seeds": ";".join(str(s) for s in sorted(g.seed.unique())),
                  "n_samples_per_seed": int(g.pool_size.iloc[0]),
                  "value": float(np.median(g.first_hit_median_all_censored)),
                  "value_seed_min": float(g.first_hit_median_all_censored.min()),
                  "value_seed_max": float(g.first_hit_median_all_censored.max()),
                  "n_hit_seed_mean": float(g.n_hit.mean()), "censored_fraction_mean": float(g.censored_fraction.mean()),
                  "first_hit_median_hits_only_seed_median": float(np.median(g.first_hit_median_hits_only)),
                  "arm_label": ARM_LABEL.get(key[2], key[2])})
        r.update(fl(key[2]))
        rows.append(r)
    v = var[var.scope == "all_seeds"]
    for _, g in v.iterrows():
        r = {"block": "variation", "subset": g.subset, "real_set": "own_centres", "arm": g.arm, "pool": "all",
             "space": "per_centre", "budget": int(g.n_samples), "n_seeds": np.nan, "seeds": "all_seeds",
             "n_samples_per_seed": np.nan, "value": np.nan,
             "n_centres": int(g.n_centres), "n_centres_ge2": int(g.n_centres_ge2),
             "samples_per_centre_median": float(g.samples_per_centre_median),
             "sampler_invalid_share": float(g.sampler_invalid_share),
             "iqr_pet_s": float(g.iqr_median_pet), "iqr_d_min_m": float(g.iqr_median_d_min),
             "iqr_alpha_deg": float(g.iqr_median_alpha), "iqr_conflict_x_m": float(g.iqr_median_conflict_x),
             "iqr_conflict_y_m": float(g.iqr_median_conflict_y), "iqr_u_c_mps": float(g.iqr_median_u_c),
             "iqr_dtw_m": float(g.iqr_median_dtw), "recovery_interaction_pooled": float(g.recovery_interaction_pooled),
             "recovery_path_pooled": float(g.recovery_path_pooled), "recovery_joint_pooled": float(g.recovery_joint_pooled),
             "arm_label": ARM_LABEL.get(g.arm, g.arm)}
        r.update(fl(g.arm))
        rows.append(r)
    t3 = pd.DataFrame(rows)
    t3["ours_kde_seed_status"] = "seed 20260910 scored; 20260911 rendered 5000/5000 not scored; 20260912 rendering"
    save_csv("T3.csv", t3)

    # LaTeX T3: coverage@1000 (interaction/path/joint) ours vs SVD analytic, coverage@100 incl. E3, first-hit medians
    q = t3[(t3.block == "coverage") & (t3.real_set == "matched489") & (t3.pool == "all")]
    f = t3[(t3.block == "first_hit") & (t3.real_set == "matched489") & (t3.pool == "all")]
    hdr = ["class", "space", "ours@100\\textsuperscript{is}", "SVD@100\\textsuperscript{ia}", "E3@100\\textsuperscript{iep}",
           "ours@1000\\textsuperscript{is}", "SVD@1000\\textsuperscript{ia}", "first-hit ours", "first-hit SVD"]
    trs, mids = [], set()
    for sc in CLASSES:
        mids.add(len(trs))
        for i, sp in enumerate(["interaction", "path", "joint"]):
            def cv(arm, b):
                x = q[(q.subset == sc) & (q.space == sp) & (q.arm == arm) & (q.budget == b)]
                if x.empty:
                    return "--"
                x = x.iloc[0]
                return fmt(x.value, 2) if x.n_seeds == 1 else f"{x.value:.2f} [{x.value_seed_min:.2f},{x.value_seed_max:.2f}]"

            def fhv(arm):
                x = f[(f.subset == sc) & (f.space == sp) & (f.arm == arm)]
                if x.empty:
                    return "--"
                x = x.iloc[0]
                return "$>$1000" if math.isinf(x.value) else fmt(x.value, 0)
            trs.append([CLASS_TEX[sc] if i == 0 else "", sp, cv("ours3_disk_kde", 100), cv("svd_d5_kde_matched_analytic", 100),
                        cv("svd_d5_kde_executed_E3", 100), cv("ours3_disk_kde", 1000), cv("svd_d5_kde_matched_analytic", 1000),
                        fhv("ours3_disk_kde"), fhv("svd_d5_kde_matched_analytic")])
    write_tex("T3.tex", "Coverage of the matched-489 real set at budget $b$ (fraction of real scenarios with $\\ge 1$ "
              "sample within $\\varepsilon$ = real-to-real median NN distance; interaction = 6-D z-scored descriptor "
              "space, path = DTW to the kernel-centre GT path, joint = both on one sample; mean over 20 draw orderings; "
              "SVD values: seed mean [min, max] over 3 seeds; ours: seed 20260910 only) and first-hit median draw "
              "index (natural order, censored at the pool size; $>$1000 = fewer than half of the real scenarios hit).",
              "tab:t3-coverage", "llrrrrrrr", hdr, trs,
              ["ours3 KDE draws kernel centres from the 469 executable disk contexts only (keeptl 39/50, cutinr 13/20), "
               "so path coverage is capped by centre availability; the E3 executed KDE pool is 100 draws/class/seed "
               "(coverage@1000 undefined). Both dependent samplers are in-sample; per-centre variation and the "
               "valid-only pool are in T3.csv.", FLAG_LEGEND], midrules=mids)

    # T4
    s = sim[sim.scope.isin(["all_seeds", "all_seeds_sampler_valid"])].copy()
    keep = ["subset", "arm", "scope", "descriptor", "unit", "n_real", "n_arm", "n_arm_finite", "n_arm_nonfinite",
            "sampler_invalid_share", "real_median", "arm_median", "w1", "w1_seed_min", "w1_seed_max", "w1_seed_std",
            "variance_ratio", "iqr_ratio", "out_of_real_support", "real_range_covered", "bracket_rate_all",
            "bracket_n_centres_all", "bracket_rate_match27", "bracket_rate_match100", "per_centre_span_median"]
    t4 = s[keep].copy()
    t4["arm_label"] = t4["arm"].map(lambda a: ARM_LABEL.get(a, a))
    t4["n_seeds"] = t4["arm"].map(lambda a: len(seeds_used.get(a, [])) or np.nan)
    for c in FLAG_COLS:
        t4[c] = 0
    for i, r in t4.iterrows():
        for k, vv in fl(r.arm).items():
            t4.at[i, k] = vv
    t4["stat_unit"] = "W1 in native units vs real matched489 distribution; bracket = share of centres whose draws bracket the real value"
    save_csv("T4.csv", t4)
    hdr = ["class", "arm"] + ["PET [s]", "$d_{\\min}$ [m]", "$\\alpha$ [$^\\circ$]", "$c_x$ [m]", "$c_y$ [m]", "$u_c$ [m/s]"]
    trs, mids = [], set()
    desc = ["pet", "d_min", "alpha", "conflict_x", "conflict_y", "u_c"]
    short = {"ours3_disk_kde": "ours3 KDE\\textsuperscript{is}", "svd_d5_kde_matched_analytic": "SVD KDE\\textsuperscript{ia}",
             "svd_d5_kde_executed_E3": "SVD E3 KDE\\textsuperscript{iep}", "sakura_bc": "SAKURA bc"}
    for sc in CLASSES:
        mids.add(len(trs))
        for j, arm in enumerate(short):
            x = t4[(t4.subset == sc) & (t4.arm == arm) & (t4.scope == "all_seeds")]
            if x.empty:
                continue
            cells = []
            for d in desc:
                y = x[x.descriptor == d]
                cells.append(fmt(float(y.w1.iloc[0]), 2) if len(y) else "--")
            # bold lowest among ours vs SVD analytic
            trs.append([CLASS_TEX[sc] if j == 0 else "", short[arm]] + cells)
    write_tex("T4.tex", "Distribution similarity to the real matched-489 set: 1-Wasserstein distance $W_1$ per interaction "
              "descriptor (native units, all seeds pooled; lower = closer). Out-of-support share, variance ratio, "
              "bracket rates and per-seed spread are in T4.csv.", "tab:t4-similarity", "llrrrrrr", hdr, trs,
              ["Real per-class std used for z-scoring and $\\varepsilon$: see TABLE34\\_REPORT.md. keeptl/cutinr rows of "
               "ours partly reflect missing kernel centres (39/50, 13/20).", FLAG_LEGEND], midrules=mids)


# ----------------------------------------------------------------------------------------------
# T5 — validity / background
# ----------------------------------------------------------------------------------------------
def build_t5():
    reg(RES / "TABLE5_VALIDITY_REPORT.md")
    man = rjson(RES / "table5_manifest.json")
    summ = rcsv(RES / "table5_validity_summary.csv")
    stats = rcsv(RES / "table5_validity_stats.csv")

    def pval(horizon, metric, arm, cls, ref="ours3_disk"):
        x = stats[(stats.horizon == horizon) & (stats.metric == metric) & (stats.arm == arm) & (stats.scope == cls)
                  & (stats.reference == ref)]
        if x.empty:
            return (np.nan, np.nan, np.nan, np.nan)
        x = x.iloc[0]
        return (float(x.mean_paired_diff), float(x.p_signflip), float(x.p_holm_family), int(x.n_scenarios))

    rows = []
    full = summ[summ.horizon == "full"].set_index(["scope", "arm"])
    for _, r in summ[summ.horizon == "chmed"].iterrows():
        fr = full.loc[(r.scope, r.arm)]
        d_s, p_s, ph_s, n_s = pval("chmed", "any_solid", r.arm, r.scope)
        d_o, p_o, ph_o, n_o = pval("full", "offroad_vl", r.arm, r.scope)
        d_g, p_g, ph_g, n_g = pval("full", "phys_gate1", r.arm, r.scope)
        row = {
            "class": r.scope, "arm": r.arm, "arm_label": ARM_LABEL.get(r.arm, r.arm_label), "kind": r.kind,
            "execution_variant": ("window (as executed)" if r.arm in E3_ARMS and "polytrunc" not in r.arm else
                                  ("polytrunc (up to last vertex)" if "polytrunc" in r.arm else
                                   ("esmini executed" if r.kind == "executed" else r.kind))),
            "n_samples": int(r.n), "n_attempted": int(r.n_attempted), "n_scenarios": int(r.n_scenarios),
            "mean_driven_s_chmed": float(r.mean_driven_s),
            "solid_hit_rate_chmed": float(r.any_solid_rate), "solid_hit_wilson_lo": float(r.any_solid_wilson_lo),
            "solid_hit_wilson_hi": float(r.any_solid_wilson_hi),
            "solid_hit_delta_above_real": (float(r.any_solid_delta_real) if r.arm != "real" else np.nan),
            "solid_hit_delta_boot_lo": (float(r.any_solid_delta_real_boot_lo) if r.arm != "real" else np.nan),
            "solid_hit_delta_boot_hi": (float(r.any_solid_delta_real_boot_hi) if r.arm != "real" else np.nan),
            "near_interaction_solid_rate_chmed": float(r.any_near_solid_rate),
            "solid_hits_per_s_chmed": float(r.solid_hits_per_s),
            "ped_solid_rate_chmed": float(r.ped_any_solid_rate),
            "solid_vs_ours3_disk_paired_diff": d_s, "solid_vs_ours3_disk_p": p_s, "solid_vs_ours3_disk_p_holm": ph_s,
            "solid_vs_ours3_disk_n_scenes": n_s,
            "offroad_vl_rate_full": float(fr.offroad_vl_rate), "offroad_vl_wilson_lo": float(fr.offroad_vl_wilson_lo),
            "offroad_vl_wilson_hi": float(fr.offroad_vl_wilson_hi),
            "offroad_e7_rate_full": float(fr.offroad_e7_rate),
            "offroad_vs_ours3_disk_paired_diff": d_o, "offroad_vs_ours3_disk_p": p_o, "offroad_vs_ours3_disk_p_holm": ph_o,
            "teleport_rate_full": float(fr.teleport_rate), "wrongway_rate_full": float(fr.wrongway_rate),
            "phys_gate1_rate_full": float(fr.phys_gate1_rate), "phys_gate1_vs_ours3_disk_p_holm": ph_g,
            "sampler_invalid_rate": float(fr.sampler_invalid_rate), "valid_all_rate_full": float(fr.valid_all_rate),
            "ego_critical_rate_full": float(fr.ego_critical_rate),
            "shift_m25_solid_full": float(fr.shift_m25_any_solid_rate), "shift_p25_solid_full": float(fr.shift_p25_any_solid_rate),
            "primary_metric": "chmed Δ-above-real solid hit (H4 pre-registered primary) + off-road VL (full)",
        }
        row.update(flags(analytic=r.kind == "analytic",
                         single_seed=(r.arm == "ours3_disk_kde"),
                         exposure_unequal=(r.arm in ("ours3_disk_kde", "svd_d5_kde", "svd_exec_E3_kde_kde", "sakura_bc")
                                           or r.arm in E3_ARMS and "smoke" in r.arm),
                         pending_decision=(r.arm in E3_ARMS), in_sample=("fullfit" in r.arm or "kde" in r.arm)))
        rows.append(row)
    t5 = pd.DataFrame(rows)
    save_csv("T5.csv", t5)

    arms = ["real", "ours3_disk", "ours3_disk_kde", "svd_d5_fullfit", "svd_d5_logo", "svd_d5_kde",
            "svd_exec_E3_recon_fullfit", "svd_exec_E3_kde_kde", "sakura_bc"]
    short = {"real": "real", "ours3_disk": "ours3", "ours3_disk_kde": "ours3 KDE\\textsuperscript{s}",
             "svd_d5_fullfit": "SVD ff\\textsuperscript{ai}", "svd_d5_logo": "SVD LOGO\\textsuperscript{a}",
             "svd_d5_kde": "SVD KDE\\textsuperscript{a}", "svd_exec_E3_recon_fullfit": "E3 ff\\textsuperscript{p}",
             "svd_exec_E3_kde_kde": "E3 KDE\\textsuperscript{ep}", "sakura_bc": "SAKURA"}
    hdr = ["class", "layer"] + [short[a] for a in arms]
    trs, mids = [], set()
    for sc in ["pooled"] + CLASSES:
        mids.add(len(trs))
        sub = t5[t5["class"] == sc].set_index("arm")
        r1, r2, r3 = [CLASS_TEX[sc], "solid hit (chmed)"], ["", "$\\Delta$ above real"], ["", "off-road VL (full)"]
        for a in arms:
            if a not in sub.index:
                r1.append("--"); r2.append("--"); r3.append("--")
                continue
            x = sub.loc[a]
            star = "*" if (x.solid_vs_ours3_disk_p_holm == x.solid_vs_ours3_disk_p_holm and x.solid_vs_ours3_disk_p_holm < 0.05) else ""
            r1.append(f"{x.solid_hit_rate_chmed:.2f}{star}")
            z = lambda v: f"{v:.2f}".replace("0.", ".")
            r2.append("--" if a == "real" else f"{x.solid_hit_delta_above_real:+.2f} [{z(x.solid_hit_delta_boot_lo)},{z(x.solid_hit_delta_boot_hi)}]")
            r3.append(f"{x.offroad_vl_rate_full:.3f}")
        trs += [r1, r2, r3]
    write_tex("T5.tex", "Background-collision and map validity (H4). Solid hit = OBB SAT overlap $>2$ frames and depth "
              "$\\ge 0.1$\\,m with any recorded background vehicle at the chmed horizon (min over arms of the arm-median "
              "driven time); $\\Delta$ above real = paired excess over the recorded target in the same scenes with a "
              "10\\,000-draw group-bootstrap CI; * = scenario-paired sign-flip vs ours3 disk survives Holm; off-road VL "
              "= share of samples with $>5$\\,\\% of window-clipped 30\\,Hz points outside the drivable union (full "
              "horizon). SVD executed (timed Polyline, E3): analytic-vs-executed post-startup ADE median 0.09\\,cm "
              "(recon), 0.10\\,cm (KDE); window variant shown, polytrunc pending user decision.",
              "tab:t5-validity", "ll" + "r" * len(arms), hdr, trs,
              ["Pooled KDE rows mix class sets (ours3 KDE: tlkeep/keeptl/keeptl\\_sw 1000 each + cutinl 634 at scoring "
               "time, no cutinr; SVD KDE: 5 classes $\\times$ 3 seeds $\\times$ 100; E3 KDE: 843 completed at scoring) "
               "and must be read per class. Every generated arm sits above the real replay (all $\\Delta$ CIs exclude 0).",
               FLAG_LEGEND], midrules=mids, size="\\scriptsize", colsep_pt=2.5)


# ----------------------------------------------------------------------------------------------
# T6a — controllability ; T6b — few-shot
# ----------------------------------------------------------------------------------------------
def build_t6a():
    reg(RES / "e7/TABLE6A_REPORT.md")
    man = rjson(RES / "e7/manifest.json")
    oat = rcsv(RES / "e7/table6a_oat_summary.csv")
    jac = rcsv(RES / "e7/table6a_jacobian_summary.csv")
    tgt = rcsv(RES / "e7/table6a_target_summary.csv")
    con = rcsv(RES / "e7/table6a_target_contrasts.csv")
    spd = rcsv(RES / "e7/table6a_speed_attainment.csv")
    rows = []
    for _, r in oat.iterrows():
        row = {"block": "A_oat", "class": r.subset, "arm": r.arm, "axis": r.axis, "n_scenes": int(r.n_scenes),
               "intended": r.intended, "crosstalk_median": r.crosstalk_median, "dead_axis_n": int(r.dead_axis_n),
               "max_sensitivity_median_iqr_per_sd": r.max_sensitivity_median, "share_entropy_median": r.share_entropy_median,
               "max_share_descriptor": r.max_share_descriptor_mode, "n_pet_no_event": r.n_pet_no_event_total}
        for d in ["pet", "d_min", "alpha", "cpoint", "u_c"]:
            row[f"mono_strict_{d}"] = r[f"mono_strict_frac_{d}"]
            row[f"mono_lenient_{d}"] = r[f"mono_lenient_frac_{d}"]
            row[f"sens_{d}"] = r[f"sens_median_{d}"]
        row.update(flags(analytic=r.arm.startswith("svd"), tautology=(r.axis == "v_end"),
                         exposure_unequal=(r.subset == "special_39_180")))
        row["note"] = ("u_c sampled on the EndSpeed plateau: v_end->u_c precision partly by construction of the speed profile"
                       if r.axis == "v_end" else ("levels in expert units (10 deg / 20%), cutinl normalisers borrowed"
                                                  if r.subset == "special_39_180" else ""))
        rows.append(row)
    for _, r in jac.iterrows():
        row = {"block": "A_jacobian", "class": r.subset, "arm": r.arm, "axis": f"all_{int(r.n_axes)}", "n_scenes": int(r.n_scenes),
               "jac_rank_median": r.rank_6_median, "jac_rank_min": r.rank_6_min, "jac_cond_median": r.cond_6_median,
               "jac_rank5_median": r.rank_5_median, "jac_cond5_median": r.cond_5_median, "full_rank_frac": r.full_rank_6_frac}
        row.update(flags(analytic=r.arm.startswith("svd")))
        rows.append(row)
    for _, r in tgt.iterrows():
        row = {"block": "B_target", "class": r.scope, "arm": r.arm, "axis": r.family, "n_scenes": int(r.n_scenes),
               "n_targets": int(r.n_targets), "n_pairs": int(r.n_pairs), "hit_27_raw": r.hit_27_raw_rate,
               "hit_100_raw": r.hit_100_raw_rate, "hit_100_raw_ci_low": r.hit_100_raw_ci_low, "hit_100_raw_ci_high": r.hit_100_raw_ci_high,
               "hit_100_identity_gated": r.hit_100_identity_rate, "hit_100_valid_gated": r.hit_100_valid_rate,
               "first_hit_median_censored101": r.first_hit_median_censored101, "first_hit_censored": bool(r.first_hit_median_is_censored)}
        c = con[(con.scope == r.scope) & (con.family == r.family) & (con.quantity == "hit_100") & (con.gate == "raw")]
        if len(c):
            c = c.iloc[0]
            row.update({"contrast_ours_minus_svd_hit100_raw": c.mean_paired_diff, "ci_low": c.ci95_low_cluster,
                        "ci_high": c.ci95_high_cluster, "p_raw": c.p_signflip_raw, "p_holm": c.p_holm,
                        "min_attainable_p": c.min_attainable_p, "favours": c.favours})
        ci = con[(con.scope == r.scope) & (con.family == r.family) & (con.quantity == "hit_100") & (con.gate == "identity")]
        if len(ci):
            ci = ci.iloc[0]
            row.update({"contrast_identity_gated": ci.mean_paired_diff, "p_identity_raw": ci.p_signflip_raw, "favours_identity": ci.favours})
        row.update(flags(analytic=r.arm.startswith("svd"), exposure_unequal=(r.scope == "special_39_180")))
        row["note"] = ("class-level Holm p=1.0 is structural: exact sign-flip with <=5 clusters cannot go below 2^-(G-1)"
                       if r.scope in CLASSES else "")
        rows.append(row)
    for _, r in spd.iterrows():
        row = {"block": "C_speed", "class": r.scope, "arm": r.arm, "axis": r.quantity, "n_scenes": r.n_scenes,
               "n": int(r.n), "abs_err_median_mps": r.abs_err_median_mps, "abs_err_iqr_mps": r.abs_err_iqr_mps,
               "spearman_rho": r.spearman_rho, "ols_slope": r.ols_slope, "per_scene_slope_median": r.per_scene_slope_median,
               "n_levels_clipped": r.n_levels_clipped}
        row.update(flags(tautology=1))
        row["note"] = "u_c sampled on the EndSpeed plateau: v_end->u_c precision partly by construction (verifier caveat)"
        rows.append(row)
    t6a = pd.DataFrame(rows)
    save_csv("T6a.csv", t6a)

    # LaTeX: (a) OAT per class ; (b) attainment
    hdr = ["class", "$\\theta_1$ mono/cross/dead", "$\\theta_2$ mono/cross/dead", "$v_{\\rm end}$ mono $u_c$/cross\\textsuperscript{t}",
           "ours $J$ rank/cond", "SVD entropy\\textsuperscript{a}", "SVD $J$ rank/cond\\textsuperscript{a}"]
    trs = []
    for sc in CLASSES + ["special_39_180"]:
        o = oat[(oat.subset == sc)]
        j = jac[jac.subset == sc]

        def ax(axis, desc):
            x = o[(o.arm == "ours3_e7_oat") | (o.arm == "ours3_e9_oat_39_180_a2641")]
            x = x[x.axis == axis]
            if x.empty:
                return "--"
            x = x.iloc[0]
            return f"{x[f'mono_strict_frac_{desc}']:.1f}/{x.crosstalk_median:.2f}/{int(x.dead_axis_n)}"
        svdo = o[o.arm == "svd_e7_oat"]
        ent = "--" if svdo.empty else f"{svdo.share_entropy_median.min():.2f}--{svdo.share_entropy_median.max():.2f}"
        jo = j[j.arm.str.startswith("ours")]
        js = j[j.arm.str.startswith("svd")]
        trs.append([CLASS_TEX.get(sc, "39\\_180"), ax("theta1", "d_min"), ax("theta2", "d_min"), ax("v_end", "u_c"),
                    "--" if jo.empty else f"{jo.rank_6_median.iloc[0]:.0f}/3, {jo.cond_6_median.iloc[0]:.0f}", ent,
                    "--" if js.empty else f"{js.rank_6_median.iloc[0]:.0f}/5, {js.cond_6_median.iloc[0]:.0f}"])
    write_tex("T6a_oat.tex", "Controllability, Block A (OAT, levels $\\pm1/\\pm2$ class SD, 5 scenes per class; 39\\_180: "
              "$\\pm10/20^\\circ$, $\\times0.6$--$1.4$): strict-monotone fraction of the intended descriptor "
              "($d_{\\min}$ for $\\theta_1,\\theta_2$; $u_c$ for $v_{\\rm end}$), median crosstalk share, dead-axis count; "
              "Jacobian rank/condition at nominal (IQR-normalised, 6 descriptors); SVD latent axes: share-entropy "
              "range over $v_1..v_5$ (1 = fully entangled).", "tab:t6a-oat", "lllllll", hdr, trs,
              ["\\textsuperscript{t} $u_c$ is sampled on the EndSpeed plateau, so the $v_{\\rm end}\\to u_c$ precision is "
               "partly by construction of the speed profile. \\textsuperscript{a} SVD levels are analytic decodes "
               "(E3: analytic $\\approx$ executed to mm).", FLAG_LEGEND])
    hdr = ["scope", "target", "ours hit@100", "SVD hit@100\\textsuperscript{a}", "$\\Delta$ [CI] $p$", "ours id-gated",
           "SVD id-gated", "first-hit ours/SVD"]
    trs, mids = [], set()
    for sc in CLASSES + ["special_39_180", "pooled_25"]:
        mids.add(len(trs))
        for fam in ["pet", "d_min"]:
            o = tgt[(tgt.scope == sc) & (tgt.family == fam) & (tgt.arm == "ours3_e7_target")]
            s = tgt[(tgt.scope == sc) & (tgt.family == fam) & (tgt.arm == "svd_e7_target")]
            c = con[(con.scope == sc) & (con.family == fam) & (con.quantity == "hit_100") & (con.gate == "raw")]
            if o.empty or s.empty:
                continue
            o, s = o.iloc[0], s.iloc[0]
            if len(c) and c.iloc[0].ci95_low_cluster == c.iloc[0].ci95_low_cluster:
                c = c.iloc[0]
                ptx = "$<$0.001" if c.p_signflip_raw < 0.001 else f"{c.p_signflip_raw:.3f}"
                ctxt = f"{c.mean_paired_diff:+.2f} [{c.ci95_low_cluster:.2f},{c.ci95_high_cluster:.2f}] {ptx}"
                if sc in CLASSES:
                    ctxt += "\\textsuperscript{h}"
            elif len(c):
                ctxt = f"{c.iloc[0].mean_paired_diff:+.2f} (single scene, no CI)"
            else:
                ctxt = "--"

            def fhx(x):
                return ("$>$100" if bool(x.first_hit_median_is_censored) else fmt(x.first_hit_median_censored101, 0))
            trs.append([(CLASS_TEX.get(sc, sc.replace("_", "\\_")) if fam == "pet" else ""), fam.replace("_", "\\_"),
                        fmt(o.hit_100_raw_rate, 2), fmt(s.hit_100_raw_rate, 2), ctxt, fmt(o.hit_100_identity_rate, 2),
                        fmt(s.hit_100_identity_rate, 2), f"{fhx(o)}/{fhx(s)}"])
    write_tex("T6a_target.tex", "Controllability, Block B (equal-budget target attainment, $N=100$ conditional draws per "
              "scene and arm, seed 20260910): share of scene$\\times$target pairs hit within 100 draws, raw and "
              "identity-gated (DTW $\\le 1$\\,m to the recorded target path); $\\Delta$ = ours$-$SVD with cluster "
              "bootstrap CI and exact sign-flip $p$; first-hit median draw index.", "tab:t6a-target", "llrrlrrr",
              hdr, trs, ["\\textsuperscript{h} class-level Holm $p=1.0$ is structural with $\\le5$ clusters "
                         "($p_{\\min}=2^{-(G-1)}\\ge0.0625$); only the pooled 25-scene row carries inferential weight "
                         "(PET $-0.17$ [$-0.30,-0.06$], $p=0.023$; $d_{\\min}$ $-0.28$ [$-0.41,-0.16$], $p=0.0004$ favour "
                         "SVD on raw hits; identity-gated 0.71 vs 0.68 / 0.67 vs 0.68). tlkeep PET targets (0.2--2.3\\,s) "
                         "lie below every selected scene's recorded $|$PET$|$ (4.2--9.9\\,s).", FLAG_LEGEND], midrules=mids)
    (TEX / "T6a.tex").write_text("% T6a = two tabulars\n\\input{T6a_oat.tex}\n\\input{T6a_target.tex}\n")


def build_t6b():
    reg(RES / "TABLE6B_FEWSHOT_REPORT.md")
    man = rjson(RES / "table6b_manifest.json")
    d = rcsv(RES / "table6b_fewshot_summary.csv")
    keep = ["subset", "scenario_id", "scenario_uid", "role", "arm", "N_train", "n_fits_attempted", "n_fits_ok",
            "n_rank_failures", "n_pool", "status", "D_m", "D_m_subset_median", "D_m_subset_q1", "D_m_subset_q3",
            "pet_median", "dmin_median", "alpha_median", "cpoint_median", "uc_median", "dtw_median",
            "explained_variance_median", "h_loo_median", "h_loo_min", "h_loo_max", "n_kde_definable",
            "n_generated_no_event", "n_pet_flip"]
    t = d[keep].copy()
    t["held_out"] = t["arm"].str.startswith("svd").map({True: "yes (LOGO pool, scene never in subset)", False: "training-free default"})
    for c in FLAG_COLS:
        t[c] = 0
    t["analytic"] = t["arm"].str.startswith("svd").astype(int)
    t["exposure_unequal"] = 1  # one deterministic render vs 10 subsets
    t["in_sample"] = 0
    t["note"] = np.where(t.N_train.astype(str) == "6", "N=6=d+1: LOO bandwidth = sqrt(2/5) regardless of data (regular simplex)", "")
    t.loc[t.scenario_id == "230_179", "note"] = "230_179 unavailable under the disk rule; arc default is sensitivity only"
    save_csv("T6b.csv", t)
    Ns = ["6", "8", "16", "32", "full"]
    hdr = ["class / scene", "ours3 disk"] + [f"SVD $N={n}$\\textsuperscript{{a}}" for n in Ns]
    trs = []
    for (sc, sid), g in d.groupby(["subset", "scenario_id"], sort=False):
        ours = g[g.arm == "ours3_disk_default"]
        oursv = fmt(float(ours.D_m.iloc[0]), 2) if len(ours) and ours.D_m.notna().iloc[0] else \
            ("arc " + fmt(float(g[g.arm == 'ours3_arc_default'].D_m.iloc[0]), 2) + "\\textsuperscript{e}")
        cells = []
        for n in Ns:
            x = g[g.N_train.astype(str) == n]
            if x.empty or x.status.iloc[0] != "ok":
                cells.append("n/f")
            else:
                cells.append(fmt(float(x.D_m_subset_median.iloc[0]), 2))
        trs.append([f"{CLASS_TEX[sc]} {texesc(sid)}", oursv] + cells)
    write_tex("T6b.tex", "Few-shot: held-out SVD d5 reconstruction of one scene from $N$ training scenes of its LOGO pool "
              "(median scene-wise $D_m$ over 10 random subsets; the held-out scene is never in the subset) against the "
              "training-free ours3 disk default of the same scene (one render). $N\\in\\{1,2,4\\}$ cannot support $d=5$ "
              "(rank $N-1$); n/f = not feasible (pool smaller than $N$). Scene-wise $D_m$ is not comparable across scenes.",
              "tab:t6b-fewshot", "llrrrrr", hdr, trs,
              ["\\textsuperscript{a} analytic decode. \\textsuperscript{e} 230\\_179 has no disk render (unavailable under "
               "the paper window rule); its arc-window value is sensitivity only. At $N=6=d+1$ the LOO bandwidth is "
               "$\\sqrt{2/5}$ for every subset (regular simplex), i.e. the KDE is definable but not data-dependent.",
               FLAG_LEGEND])


# ----------------------------------------------------------------------------------------------
# T7 — efficiency (partial)
# ----------------------------------------------------------------------------------------------
def parse_runner_log(p: Path):
    """runner stdout lines: '[ours3] <id>: completed (0.28s)' -> per-sample wall seconds."""
    reg(p)
    ts = []
    pat = re.compile(r"completed \(([\d.]+)s\)")
    with open(p, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                ts.append(float(m.group(1)))
    return ts


def build_t7():
    svdm = rcsv(RES / "svd_d5_matched_summary.csv")
    bw = rcsv(RES / "svd_d5_kde_matched_bandwidth.csv")
    tim = rcsv(RES / "svd_d5_kde_matched_timing.csv")
    fit = rjson(ROOT / "plans/ours3_disk_kde_dependent/7e0593f38aafa843/fit.json")
    ex = rjson(RES / "ours3_population_execution_audit.json")
    e3 = rjson(RES / "svd_d5_executed_manifest.json")
    e7 = rjson(RES / "e7/manifest.json")
    e9 = rjson(RES / "e9_39_180/manifest.json")
    rows = []

    def add(stage, arm, n, unit, total_s, per_unit_s, source, status="available", **fl):
        r = {"stage": stage, "arm": arm, "arm_label": ARM_LABEL.get(arm, arm), "n": n, "unit": unit,
             "wall_s_total": total_s, "wall_s_per_unit": per_unit_s, "source": str(source), "status": status}
        r.update(flags(**fl))
        rows.append(r)

    add("model fit (basis)", "svd_d5_matched", int(svdm.fits_ok.sum()), "fits (5 classes x (1 fullfit + LOGO groups))",
        float(svdm.total_fit_wall_s.sum()), float(svdm.total_fit_wall_s.sum() / svdm.fits_ok.sum()),
        RES / "svd_d5_matched_summary.csv", analytic=1)
    add("encode/decode (analytic)", "svd_d5_matched", int(svdm.n_eligible.sum()), "scenes",
        float(svdm.total_encode_decode_wall_s.sum()), float(svdm.total_encode_decode_wall_s.sum() / svdm.n_eligible.sum()),
        RES / "svd_d5_matched_summary.csv", analytic=1)
    add("KDE bandwidth (LOO grid 60)", "svd_d5_kde", int(len(bw)), "classes", float(bw.bandwidth_wall_s.sum()),
        float(bw.bandwidth_wall_s.mean()), RES / "svd_d5_kde_matched_bandwidth.csv", analytic=1)
    add("KDE sampling + decode (numeric)", "svd_d5_kde", int(tim.n_attempts.sum()), "draws",
        float((tim.sample_wall_s + tim.decode_wall_s).sum()),
        float(((tim.sample_ms_per_attempt + tim.decode_ms_per_attempt) / 1000.0).median()),
        RES / "svd_d5_kde_matched_timing.csv", analytic=1)
    add("KDE fit (3-D, LOO h)", "ours3_disk_kde", int(sum(v["n_fit"] for v in fit["fits"].values())), "contexts",
        float(sum(v["fit_s"] for v in fit["fits"].values())), float(sum(v["fit_s"] for v in fit["fits"].values()) / 5),
        ROOT / "plans/ours3_disk_kde_dependent/7e0593f38aafa843/fit.json", in_sample=1)
    add("esmini execution (defaults, arc cohort)", "ours3_arc", int(ex["n_completed"]), "renders",
        float(ex["simulation_wall_s_sum"]), float(ex["simulation_wall_s_median"]),
        RES / "ours3_population_execution_audit.json")
    add("esmini job total incl. xosc write (defaults, arc)", "ours3_arc", int(ex["n_completed"]), "renders",
        float(ex["job_total_s_sum"]), float(ex["job_total_s_median"]), RES / "ours3_population_execution_audit.json")
    for tag, arm, logf in [("esmini execution (defaults, disk cohort) [runner log]", "ours3_disk", RES / "ours3_disk_population_runner_stdout.log"),
                           ("esmini execution (KDE draws, seed 20260910) [runner log]", "ours3_disk_kde", RES / "ours3_disk_kde_s20260910_runner_stdout.log")]:
        if logf.exists():
            ts = parse_runner_log(logf)
            add(tag, arm, len(ts), "renders (job wall incl. write)", float(sum(ts)), float(np.median(ts)) if ts else np.nan, logf,
                single_seed=(arm == "ours3_disk_kde"))
        else:
            add(tag, arm, np.nan, "renders", np.nan, np.nan, logf, status="missing")
    for b in ["smoke", "recon", "kde"]:
        v = e3["batches"][b]
        add(f"esmini execution (E3 timed Polyline, {b})", "svd_exec_E3_" + b, int(v["n"]), "renders", float(v["simulate_s_total"]),
            float(v["simulate_s_median"]), RES / "svd_d5_executed_manifest.json", pending_decision=1)
    for b in ["recon_polytrunc", "kde_polytrunc"]:
        add(f"polytrunc variant ({b})", "svd_exec_E3_" + b, int(e3["batches"][b]["n"]), "derived truncations (no new simulation)",
            np.nan, np.nan, RES / "svd_d5_executed_manifest.json", status="derived: no timing recorded", pending_decision=1)
    for k, v in e7["target_manifest"]["phases"]["execute"].items():
        add("esmini execution (E7 target draws, 100/scene x 5 scenes)", k, int(v["n_jobs"]), "renders", float(v["wall_s"]),
            float(v["wall_s"] / v["n_jobs"]), RES / "e7/manifest.json")
    for k, v in e7["oat_manifest"]["phases"]["execute"].items():
        if isinstance(v, dict) and "wall_s" in v:
            add("esmini execution (E7 OAT, 12 levels x 5 scenes)", k, int(v.get("n_jobs", 60)), "renders", float(v["wall_s"]),
                float(v["wall_s"] / v.get("n_jobs", 60)), RES / "e7/manifest.json")
    add("SVD OAT levels + target draws (analytic decode)", "svd_e7", int(e7["oat_manifest"]["phases"]["plan"]["n_svd_decodes"]) +
        int(e7["target_manifest"]["phases"]["plan"]["n_svd"]), "decodes", np.nan, np.nan, RES / "e7/manifest.json",
        status="no per-decode timing recorded (numeric decode; see KDE decode ms/attempt)", analytic=1)
    add("E9 39_180 esmini jobs (OAT/Sobol/condKDE + bg collision)", "e9_39_180", int(e9["counts"]["n_esmini_jobs_new"]), "renders",
        np.nan, np.nan, RES / "e9_39_180/manifest.json", status="no wall time in manifest")
    add("E9 39_180 SVD pseudo-samples", "e9_39_180_svd", int(e9["counts"]["n_svd_pseudo_samples"]), "decodes", np.nan, np.nan,
        RES / "e9_39_180/manifest.json", status="no wall time in manifest", analytic=1)
    add("post-processing (E3 analytic-vs-executed table)", "svd_exec_E3", 978 + 1500, "pairs", float(e3["elapsed_s"]), np.nan,
        RES / "svd_d5_executed_manifest.json", pending_decision=1)
    for stage in ["xosc generation from parameters (ours3, per render)", "scorer 31 (non-PET) per arm", "scorer 32 (bbox-PET) per arm",
                  "ours3 KDE draws seeds 20260911/20260912 execution", "ours3 KDE conditional sampling (per draw)"]:
        add(stage, "n/a", np.nan, "", np.nan, np.nan, "", status="missing: not recorded in any listed source")
    t7 = pd.DataFrame(rows)
    save_csv("T7.csv", t7)
    hdr = ["stage", "arm", "$n$", "total [s]", "per unit [s]", "status"]
    trs = []
    for _, r in t7.iterrows():
        trs.append([texesc(r.stage[:46]), texesc(r.arm) + flag_marks({k: r[k] for k in FLAG_COLS}), fmt(r.n, 0),
                    fmt(r.wall_s_total, 1), fmt(r.wall_s_per_unit, 3), texesc(r.status[:34])])
    write_tex("T7.tex", "Efficiency (partial): wall-clock per pipeline stage as recorded in the manifests. SVD d5 stages "
              "are algebraic (no simulator); ours3 stages include esmini execution (fixed step 1/30 s, headless). "
              "Missing stages are listed explicitly.", "tab:t7-efficiency", "llrrrl", hdr, trs,
              ["Machine and load differ between batches; per-render medians are comparable only within a batch. "
               "The E3 executed SVD arm costs the same esmini time per render as ours3 (0.21--0.26 s median).",
               FLAG_LEGEND], size="\\scriptsize", colsep_pt=3)


# ----------------------------------------------------------------------------------------------
# T8 — 39_180 singleton
# ----------------------------------------------------------------------------------------------
def build_t8():
    reg(RES / "e9_39_180/T8_REPORT.md")
    man = rjson(RES / "e9_39_180/manifest.json")
    d = rcsv(RES / "e9_39_180/T8_singleton_table.csv")
    keep = ["arm", "class_data_used", "kind", "definable", "n_requested", "n_executed", "n_valid", "teleport_n", "offroad_n",
            "wrongway_n", "alat_n", "vmax_n", "offroad_driving_only_n", "teleport_after_conflict_n",
            "bg_solid_rate_all_gtsupport", "bg_solid_rate_label0_gtsupport", "bg_solid_rate_all_valid_gtsupport",
            "pet_median", "dmin_median", "uc_median", "err_pet_median", "err_dmin_median", "err_alpha_median",
            "err_cpoint_median", "err_uc_median", "err_dtw_median", "err_dtw_nearest", "D_m", "pet_iqr", "dmin_iqr",
            "bracket_n_of_6", "duration_clipped_n", "end_speed_clipped_n", "pet_no_event_n", "pet_flip_n"]
    t = d[keep].copy()
    t["used_class_data"] = t["class_data_used"].map(lambda s: "no" if s == "none" else f"yes ({s})")
    for c in FLAG_COLS:
        t[c] = 0
    t["analytic"] = (t["kind"] == "analytic").astype(int)
    t["in_sample"] = t["class_data_used"].isin(["basis", "basis+bandwidth", "bandwidth"]).astype(int)
    t["exposure_unequal"] = 1
    t["pending_decision"] = 0
    t["note"] = np.where(t.arm.str.startswith("svd_extbasis") | (t.arm == "ours3_condkde_a2641"),
                         "externally supplied basis/bandwidth (E0 item 5): NOT learned from one scenario", "")
    t.loc[t.arm == "svd_d5_singleton_N1", "note"] = "N=1: centred rank 0, LOO bandwidth undefined; zero samples"
    t["executed_svd_rows"] = man.get("executed_svd_rows", "")
    save_csv("T8.csv", t)
    hdr = ["arm", "class data", "valid/req.", "bg solid (GT supp.)", "$|\\Delta$PET$|$ [s]", "$|\\Delta d_{\\min}|$ [m]",
           "DTW [m]", "$D_m$", "PET IQR [s]", "bracket"]
    trs = []
    for _, r in d.iterrows():
        nv = f"{int(r.n_valid)}/{int(r.n_requested)}" if r.n_requested else "0/0"
        trs.append([texesc(r.arm) + ("\\textsuperscript{a}" if r.kind == "analytic" else ""), texesc(r.class_data_used), nv,
                    fmt(r.bg_solid_rate_all_gtsupport, 2), fmt(r.err_pet_median, 2), fmt(r.err_dmin_median, 2),
                    fmt(r.err_dtw_median, 2), fmt(r.D_m, 3), fmt(r.pet_iqr, 2), fmt(r.bracket_n_of_6, 0) + "/6"])
    write_tex("T8.tex", "39\\_180 singleton (cut-in-left motorcycle, replay PET 9.03\\,s, $d_{\\min}$ 3.81\\,m): eight arms. "
              "valid = no teleport, no wrong-way, $a_{\\rm lat}\\le5$\\,m/s$^2$, $v\\le25$\\,m/s, $\\le5$\\,\\% off the "
              "driving+shoulder union; bg solid = analytic OBB solid hit with any of the 82 rec00 vehicle tracks within the "
              "GT support; medians over valid samples; $D_m$ over the seven arms with $\\ge1$ valid sample. "
              "Arms with class data (basis / bandwidth) are externally supplied and are not learned from one scenario "
              "(E0 item 5).", "tab:t8-singleton", "llrrrrrrrr", hdr, trs,
              ["\\textsuperscript{a} analytic decode; executed SVD rows were pending E3 when T8 was built (E3 has since "
               "completed but T8 was not re-run). ours arms trade PET-event retention ($>95$\\,\\%) and 6/6 bracketing "
               "for post-conflict route-end teleports and 48--74\\,\\% background solid hits within GT support "
               "(lane-sharing motorcycles 179/166/162).", FLAG_LEGEND], size="\\scriptsize")


# ----------------------------------------------------------------------------------------------
# T9 — anchor / L sweep under the disk rule + E3 analytic vs executed
# ----------------------------------------------------------------------------------------------
def build_t9():
    reg(RES / "disk_sweep/DISK_SWEEP_REPORT.md")
    reg(RES / "E3_SVD_EXECUTION_REPORT.md")
    bd = rcsv(RES / "disk_sweep/best_d_summary_v5_disk.csv")
    ba = rcsv(RES / "disk_sweep/best_anchor_summary_v5_disk.csv")
    ad = rcsv(RES / "disk_sweep/arc_vs_disk_sweep.csv")
    ave = rcsv(RES / "svd_d5_executed_analytic_vs_executed.csv")
    rows = []
    adc = ad[ad.key == "composite"]
    for _, r in bd.iterrows():
        a = adc[(adc.table == "best_d") & (adc.subset == r.subset) & (adc.arm == r.arm)]
        row = {"block": "sweep_L", "subset": r.subset, "arm": r.arm, "astep": np.nan, "composite_disk": r.composite,
               "composite_arc_paper": (float(a.arc_value.iloc[0]) if len(a) else np.nan), "cov_med": r.cov_med,
               "pet": r.pet, "min_dist": r.min_dist, "conflict_angle": r.conflict_angle, "conflict_point": r.conflict_point,
               "conflict_speed": r.conflict_speed, "traj_dtw": r.traj_dtw, "scorer": "v5 composite (sr-tlkeep 180)"}
        row.update(flags(in_sample=1, exposure_unequal=r.subset.startswith(("special", "uturn"))))
        rows.append(row)
    for _, r in ba.iterrows():
        a = adc[(adc.table == "best_anchor") & (adc.subset == r.subset) & (adc.arm == r.arm)]
        row = {"block": "sweep_anchor", "subset": r.subset, "arm": r.arm, "astep": r.astep, "composite_disk": r.composite,
               "composite_arc_paper": (float(a.arc_value.iloc[0]) if len(a) else np.nan), "cov_med": r.cov_med,
               "n_common_disk": (float(a.n_common_disk.iloc[0]) if len(a) else np.nan),
               "n_common_arc": (float(a.n_common_arc.iloc[0]) if len(a) else np.nan),
               "pet": r.pet, "min_dist": r.min_dist, "conflict_angle": r.conflict_angle, "conflict_point": r.conflict_point,
               "conflict_speed": r.conflict_speed, "traj_dtw": r.traj_dtw, "scorer": "v5 composite (sr-tlkeep 180)"}
        row.update(flags(in_sample=1, exposure_unequal=r.subset.startswith(("special", "uturn"))))
        rows.append(row)
    for _, r in ave[(ave["mode"] == "all")].iterrows():
        row = {"block": "e3_analytic_vs_executed", "subset": r.scope, "arm": r.execution_label, "variant": r.variant,
               "metric": r.metric, "unit": r.unit, "n_pairs": r.n_pairs, "n_both_finite": r.n_both_finite,
               "median_abs_diff": r.median_abs_diff, "p90_abs_diff": r.p90_abs_diff, "max_abs_diff": r.max_abs_diff,
               "frac_identical_1e-3": r["frac_identical_1e-3"], "analytic_no_event": r.analytic_no_event,
               "executed_no_event": r.executed_no_event, "pet_sign_flips": r.pet_sign_flips,
               "ade_all_vertices_median": r.ade_all_vertices_median, "teleport_window": r.teleport_window,
               "n_clock_misaligned": r.n_clock_misaligned}
        row.update(flags(analytic=1, pending_decision=1))
        rows.append(row)
    t9 = pd.DataFrame(rows)
    save_csv("T9.csv", t9)

    # LaTeX (a): sweep summary vs 05_sc.tex
    hdr = ["subset", "$L$: $\\pm10$ / $\\pm5$ / $\\pm3$ / $\\pm2$ (disk)", "$\\pm10$ / $\\pm5$ (arc)", "best anchor (disk)",
           "legacy (disk)", "$n$ common disk/arc", "best anchor (arc)", "legacy (arc)"]
    trs = []
    for sub in ["cutinl", "keeptl", "keeptl_sw", "special_39_180", "special_1786_1797", "uturn_859_881"]:
        b = bd[bd.subset == sub].set_index("arm")
        Lc = " / ".join(fmt(b.loc[a, "composite"], 3) if a in b.index else "--" for a in ["±10", "±5", "±3", "±2"])
        aa = adc[(adc.table == "best_d") & (adc.subset == sub)].set_index("arm")
        La = " / ".join(fmt(aa.loc[a, "arc_value"], 3) if a in aa.index else "--" for a in ["±10", "±5"])
        an = ba[ba.subset == sub]
        if an.empty:
            trs.append([texesc(sub), Lc, La, "--", "--", "--", "--", "--"]); continue
        st = an.astep.max()
        an = an[an.astep == st]
        nl = an[an.arm != "legacy"].sort_values("composite")
        leg = an[an.arm == "legacy"]
        best = nl.iloc[0]
        ac = adc[(adc.table == "best_anchor") & (adc.subset == sub)]
        arcbest = ac[ac.arm != "legacy"].sort_values("arc_value")
        arcleg = ac[ac.arm == "legacy"]
        ncom = ad[(ad.table == "best_anchor") & (ad.subset == sub)].n_common_disk.dropna()
        ncoma = ad[(ad.table == "best_anchor") & (ad.subset == sub)].n_common_arc.dropna()
        trs.append([texesc(sub), Lc, La, f"{texesc(best.arm)} {best.composite:.3f} (d$\\pm${int(st)})",
                    fmt(leg.composite.iloc[0], 3) if len(leg) else "--",
                    (f"{int(ncom.min())}" if len(ncom) else "--") + "/" + (f"{int(ncoma.min())}" if len(ncoma) else "--"),
                    f"{texesc(arcbest.arm.iloc[0])} {arcbest.arc_value.iloc[0]:.3f}" if len(arcbest) else "--",
                    fmt(arcleg.arc_value.iloc[0], 3) if len(arcleg) else "--"])
    write_tex("T9_sweep.tex", "Window-radius and anchor sweep under the Euclidean-disk rule (E10 stage 2; v5 composite, "
              "lower is better; 1\\,928 renders) beside the historical arc-length values quoted in the paper before the "
              "rewrite. Anchor codes: first letter = spatial anchor ($p$ = PET, $x$ = crossTraj, $m$ = min-dist), second = "
              "temporal anchor; legacy = frozen configuration. Selection and comparison use the same subsets (descriptive, "
              "not held-out).", "tab:t9-sweep", "llllllll", hdr, trs,
              ["05\\_sc.tex now states: keeptl best pm 0.620 vs legacy 0.718 (27 episodes); keeptl\\_sw legacy 0.540 best "
               "(best explicit 0.573); cutinl px 0.383 vs 0.489 on 9 common episodes (PET event outside the annotated "
               "interval in 21--23/61); $L=5$ vs 10: 0.805/0.812, 0.692/0.697, keeptl 0.665 vs 0.719.", FLAG_LEGEND],
              size="\\scriptsize", colsep_pt=3)
    # LaTeX (b): E3 analytic vs executed pooled
    hdr = ["metric", "unit", "window: median / p90 / max $|\\Delta|$", "identical $\\le10^{-3}$", "polytrunc: median / p90 / max $|\\Delta|$", "identical"]
    trs = []
    pw = ave[(ave.variant == "window") & (ave.scope == "pooled") & (ave["mode"] == "all")].set_index("metric")
    pp = ave[(ave.variant == "polytrunc") & (ave.scope == "pooled") & (ave["mode"] == "all")].set_index("metric")
    cnt = []
    for m in pw.index:
        w, p = pw.loc[m], pp.loc[m] if m in pp.index else None
        if w.median_abs_diff != w.median_abs_diff:
            if m == "signed_pet_no_event":
                cnt.append(f"PET no-event analytic/executed: window {int(w.analytic_no_event)}/{int(w.executed_no_event)}, "
                           f"polytrunc {int(p.analytic_no_event)}/{int(p.executed_no_event)}; sign flips {int(w.pet_sign_flips)}")
            elif m == "teleport_screen":
                cnt.append(f"teleport flags in window: {int(w.teleport_window)} (window variant) vs {int(p.teleport_window)} (polytrunc), "
                           f"{int(w.n_clock_misaligned)} clock-misaligned rows")
            continue
        trs.append([texesc(m), texesc(w.unit), f"{w.median_abs_diff:.3g} / {w.p90_abs_diff:.3g} / {w.max_abs_diff:.3g}",
                    fmt(w["frac_identical_1e-3"], 2),
                    (f"{p.median_abs_diff:.3g} / {p.p90_abs_diff:.3g} / {p.max_abs_diff:.3g}" if p is not None else "--"),
                    fmt(p["frac_identical_1e-3"], 2) if p is not None else "--"])
    write_tex("T9_e3.tex", "SVD executed (timed Polyline, E3): analytic 50-point decode vs esmini execution of the same "
              "decode on the 978 reconstructions (489 $\\times$ fullfit/LOGO). window = scored on the metadata window as "
              "executed (includes the post-polyline re-alignment tail); polytrunc = target rows up to the last vertex "
              "(like-for-like). Post-startup ADE median 0.09\\,cm, max 0.87\\,cm; no passing-order flip.",
              "tab:t9-e3", "llrrrr", hdr, trs,
              cnt + ["Which variant is primary for Tables 5 and 7 is a pending user decision (E3 manifest names polytrunc; the "
               "audit recommends window as primary with polytrunc as sensitivity).", FLAG_LEGEND], size="\\scriptsize")
    (TEX / "T9.tex").write_text("% T9 = two tabulars\n\\input{T9_sweep.tex}\n\\input{T9_e3.tex}\n")


# ----------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("70_final_tables: assembling T1..T9 ->", OUT)
    build_t1()
    build_t2()
    build_t34()
    build_t5()
    build_t6a()
    build_t6b()
    build_t7()
    build_t8()
    build_t9()
    # register paper text sources (read for wording only)
    for p in ["/home/hcis-s19/Documents/ChengYu/Conflict-anchored Param/sections/05_sc.tex",
              "/home/hcis-s19/Documents/ChengYu/Conflict-anchored Param/sections/04_impl.tex",
              str(RES / "E10_STAGE1_DISK_WINDOW_REPORT.md"), str(ROOT / "DECISIONS_20260910_E0.md"),
              "/home/hcis-s19/Documents/ChengYu/OURS_VS_SVD_EVIDENCE_AUDIT_20260910.md"]:
        if Path(p).exists():
            reg(Path(p), rows=sum(1 for _ in open(p, errors="replace")))
    outputs = {}
    for p in sorted(list(OUT.glob("T*.csv")) + list(TEX.glob("*.tex"))):
        d = {"sha256": sha256(p), "bytes": p.stat().st_size}
        if p.suffix == ".csv":
            d["rows"] = int(len(pd.read_csv(p)))
        outputs[str(p)] = d
    manifest = {
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
        "scope": "assembly only: numbers copied from verified result files; aggregation limited to sums over classes, seed mean/min/max, medians of stored per-seed values, per-sample wall-time medians parsed from runner logs",
        "flag_columns": FLAG_COLS,
        "primaries": {
            "H1": "T2 composite D_m and signed |dPET| (E0 item 1), ours3_disk vs svd_d5_logo primary contrast, 469 disk cohort, Holm 5x6",
            "H2": "T6a Block A OAT monotonicity/crosstalk + Block B identity-gated attainment (pooled 25-scene row)",
            "H3": "T6b definability (rank N-1 >= d) + nearest-sample D_m; T8 singleton arms",
            "H4": "T5 chmed Δ-above-real solid hit + off-road VL (full horizon), scenario-paired sign-flip vs ours3_disk, Holm within family",
        },
        "pending_user_decisions": [
            "SVD executed (timed Polyline, E3): window (as executed, same window semantics as ours3) vs polytrunc (like-for-like) as the Table 5 / Table 7 primary — E3 manifest names polytrunc, audit recommends window; both shown, flagged pending_decision",
            "whether to add a stop-at-end action to both arms (ours3 NURBS end and E3 polyline end both continue at last speed in esmini)",
        ],
        "ours3_kde_seed_status": {"20260910": "5000/5000 rendered, scored (T3/T4/T5)",
                                  "20260911": "5000/5000 rendered (runs/ours3_disk_kde_s20260911_*/6245e6c41f9e0bb7), NOT scored — re-run scripts/46_coverage_similarity.py and 45_validity_bg.py",
                                  "20260912": "rendering in progress (176 samples at batch_status read)"},
        "sources": SOURCES, "outputs": outputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"manifest: {len(SOURCES)} sources, {len(outputs)} outputs, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
