#!/usr/bin/env python3
"""Compile results/SAKURA_ARMS_REPORT.md and results/sakura_arms_manifest.json from the SAKURA arm outputs
(85 render batches, 31/32 scores, 87 Table 2, 45b Table 5, 46b Tables 3/4)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
XC = PROJECT.parent / "exp_cross_coverage"
CLASSES = ["keeptl", "keeptl_sw", "cutinl", "cutinr", "special_39_180"]
SEEDS = [20260910, 20260911, 20260912]
BATCHES = ["sakura_plain_defaults_extra", "sakura_route_defaults", "sakura_route_kde_s20260910", "sakura_route_kde_s20260911",
           "sakura_route_kde_s20260912", "sakura_route_kde_noclip_pilot", "sakura_route_kde_39_180_cond"]
MEAS = [("pet", "|dPET| (s)"), ("dmin", "|d d_min| (m)"), ("alpha", "|d alpha| (deg)"), ("cpoint", "||d c|| (m)"), ("uc", "|d u_c| (m/s)"), ("dtw", "DTW (m)")]
T2_ARMS = ["sakura_plain", "sakura_route", "svd_d5_fullfit", "svd_d5_logo", "ours3_disk"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def f3(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def latest(name):
    lb = PROJECT / "runs" / name / "latest_batch.json"
    return json.loads(lb.read_text()) if lb.exists() else None


def main():
    L = ["# SAKURA arms — plain chord, SAKURA-route, SAKURA-route + KDE (build, render, score)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by scripts/88_sakura_arms_report.py. Arms rendered by scripts/85_sakura_arms.py, "
         "scored with the unchanged scripts/31 (non-PET) / 32 (bbox-PET, cache results/bbox_pet_cache_sak.json) / verbatim traj_dtw, "
         "Table 2 format by scripts/87_table2_sakura_arms.py, Table 5 substrate by scripts/45b_validity_extra_arms.py (45 imported, not edited), "
         "Tables 3/4 by scripts/46b_coverage_similarity_extra.py (46 imported, not edited). Every number below is labelled with its arm name.", ""]
    manifest = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), script="scripts/88_sakura_arms_report.py", seeds=SEEDS)

    # ── 1. scenes / bases / routes ──
    sc = pd.read_csv(RES / "sakura_arms_scenes.csv")
    L += ["## 1. Scenes, bases, routes", "",
          "Scene list = results/svd_d5_cases.csv fullfit rows of keeptl / keeptl_sw / cutinl / cutinr (all scenes of each class that have a plain "
          "SAKURA base in exp_cross_coverage/esmini_runs/_xosc_base_sakura, `*_<ego>_<actor>_f<min_frame+1>.xosc`) + special_39_180 "
          "(ego 39, actor 180, window 2424-3002, exp_cross_coverage/data/special_39_180 tracks). Route plans = exp_cross_coverage/results/route_plan_<cls>.json "
          "(CARLA GlobalRoutePlanner, cached; the planner itself was not re-run).", "",
          "| class | scenes | bases found | no base | route found (of bases) | routed fraction | sakura_bc existing (192) | sakura_plain_extra rendered |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    bc_cases = pd.read_csv(RES / "sakura_default_cases.csv")
    bc_n = bc_cases[bc_cases.execution_status.eq("completed")].subset.value_counts().to_dict()
    plain_man = pd.read_csv(Path(latest("sakura_plain_defaults_extra")["manifest"]))
    scene_summary = {}
    for cls in CLASSES:
        g = sc[sc.subset.eq(cls)]
        gb = g[g.base_found]
        nb = ", ".join(g[~g.base_found].scenario_id) or "-"
        pe = int(plain_man[plain_man.subset.eq(cls) & plain_man.status.eq("completed")].shape[0])
        scene_summary[cls] = dict(n_scenes=int(len(g)), n_bases=int(len(gb)), no_base=g[~g.base_found].scenario_id.tolist(),
                                  n_route=int(gb.route_found.sum()), routed_fraction=float(gb.route_found.mean()) if len(gb) else np.nan,
                                  no_route=gb[~gb.route_found].scenario_id.tolist(), sakura_bc_existing=int(bc_n.get(cls, 0)), sakura_plain_extra=pe)
        L.append(f"| {cls} | {len(g)} | {len(gb)} | {nb} | {int(gb.route_found.sum())} | {f3(scene_summary[cls]['routed_fraction'])} | {bc_n.get(cls, 0)} | {pe} |")
    manifest["scenes"] = scene_summary
    L += ["", "No-route scenes (rendered as the plain chord in the route arms; 210 fallback): " +
          "; ".join(f"{c}: {', '.join(scene_summary[c]['no_route']) or '-'}" for c in CLASSES), ""]

    # ── 2. render batches ──
    L += ["## 2. Render batches (esmini headless, flags identical to scripts/20)", "",
          "| batch | run_id | jobs | completed | failed/timeout | wall s |", "|---|---|---:|---:|---:|---:|"]
    batches = {}
    for b in BATCHES:
        lb = latest(b)
        if lb is None:
            L.append(f"| {b} | (not rendered) | | | | |")
            continue
        bs = json.loads((Path(lb["batch_dir"]) / "batch_status.json").read_text())
        man = pd.read_csv(lb["manifest"])
        cnt = man.status.value_counts().to_dict()
        batches[b] = dict(run_id=lb["run_id"], batch_dir=lb["batch_dir"], n_jobs=int(len(man)), status_counts=cnt, elapsed_s=bs.get("elapsed_s"),
                          per_class=man.groupby("subset").status.value_counts().unstack(fill_value=0).to_dict("index"))
        L.append(f"| {b} | {lb['run_id']} | {len(man)} | {cnt.get('completed', 0)} | {len(man) - cnt.get('completed', 0)} | {f3(bs.get('elapsed_s'), 0)} |")
    manifest["batches"] = batches

    # ── 3. KDE fits + clip counts ──
    plan = json.loads((RES / "sakura_arms_plan.json").read_text())
    L += ["", "## 3. SAKURA-route KDE: fits and clip counts", "",
          "Parameters per scene = (Agent1_Offset of the base [m], v_avg [km/h] = route length / recorded duration, or GT path length / duration where no route). "
          "Fit = cvlib.ParamKDE (AST-loaded exactly as scripts/23b): per-column standardisation, LOO scalar bandwidth h (kde_sampling.loo_bandwidth), dependent sampling "
          "(uniform centre + N(0, h^2 I) in z-space). Executed offset = clip(requested, centre base offset +- 0.5 m), speed = max(0, requested); both recorded per draw.", "",
          "| class | N fit | h (z) | mean offset m / v km/h | sd offset m / v km/h | " + " | ".join(f"offset clipped s{s} (of 1000)" for s in SEEDS) + " | " + " | ".join(f"speed<0 s{s}" for s in SEEDS) + " |",
          "|---|---:|---:|---|---|" + "---:|" * 6]
    clip = {}
    for cls in ["keeptl", "keeptl_sw", "cutinl", "cutinr"]:
        f = plan["fits"][cls]
        oc = [plan["pools"][f"{cls}_seed{s}"]["n_offset_clipped"] for s in SEEDS]
        spc = [plan["pools"][f"{cls}_seed{s}"]["n_speed_clipped"] for s in SEEDS]
        clip[cls] = dict(offset_clipped=dict(zip(map(str, SEEDS), oc)), speed_clipped=dict(zip(map(str, SEEDS), spc)), h=f["h"], n_fit=f["n_fit"], mean=f["mean"], sd=f["sd"])
        L.append(f"| {cls} | {f['n_fit']} | {f['h']:.4f} | {f['mean'][0]:.3f} / {f['mean'][1]:.2f} | {f['sd'][0]:.3f} / {f['sd'][1]:.2f} | " + " | ".join(map(str, oc)) + " | " + " | ".join(map(str, spc)) + " |")
    cond = plan["cond"]
    if cond:
        L.append(f"| special_39_180 (cond, cutinl h) | 1 (+cutinl 60 for h/mean/sd) | {plan['fits']['cutinl']['h']:.4f} | centre {cond['seed20260910']['centre']['offset_m']:.3f} / {cond['seed20260910']['centre']['v_avg_kmh']:.2f} | (cutinl) | "
                 + " | ".join(str(cond[f'seed{s}']['n_offset_clipped']) + " (of 100)" for s in SEEDS) + " | " + " | ".join(str(cond[f'seed{s}']['n_speed_clipped']) for s in SEEDS) + " |")
    else:
        L.append("| special_39_180 (conditional, frozen) | excluded from v3 | – | – | – | – | – | – | – | – |")
    manifest["kde"] = dict(fits=plan["fits"], clip_counts=clip, cond=cond, plan_root=plan["plan_root"], source_key=plan["source_key"])
    L += ["", "The recorded lane offsets of a class span several lanes (sd 1.4-3.6 m) while the executable window is +-0.5 m around the centre's own offset, so most draws "
          "hit the clip bound: the executed offset distribution is the centre offsets +- 0.5 m, not the fitted class distribution (sampler-invalid = offset_clipped or speed_clipped).", ""]

    # ── 4. Table 5: validity / background, incl. unclipped pilot teleport rate ──
    t5s_path = RES / "table5_validity_summary_sakura.csv"
    if t5s_path.exists():
        t5s = pd.read_csv(t5s_path)
        t5x = pd.read_csv(RES / "table5_validity_samples_sakura.csv", low_memory=False)
        L += ["## 4. Validity / background (Table 5 substrate; horizons fixed to Table 5's per-scene ch/chmed cutoffs)", ""]
        # unclipped pilot vs clipped counterpart (same first 100 draws of seed 20260910)
        full = t5x[t5x.horizon.eq("full")]
        nc = full[full.arm.eq("sakura_route_kde_noclip")].copy()
        nc["draw_key"] = nc.sample_id.str.replace("__noclip__", "__", regex=False)
        cl = full[full.arm.eq("sakura_route_kde") & full.sample_id.isin(nc.draw_key)].set_index("sample_id")
        L += ["### 4a. Unclipped-offset pilot (seed 20260910, first 100 draws per class) vs the same draws executed with the +-0.5 m clip", "",
              "| class | n pairs | teleport rate unclipped | teleport rate clipped | off-road VL unclipped | off-road VL clipped | solid bg hit (full) unclipped | clipped | mean |requested offset - centre| m |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        pilot = {}
        for cls in ["keeptl", "keeptl_sw", "cutinl", "cutinr", "pooled"]:
            g = nc if cls == "pooled" else nc[nc.cls.eq(cls)]
            c = cl.reindex(g.draw_key)
            if not len(g):
                continue
            pool_dev = []
            for c2 in (["keeptl", "keeptl_sw", "cutinl", "cutinr"] if cls == "pooled" else [cls]):
                pt = pd.read_parquet(Path(plan["plan_root"]) / f"pool_{c2}_seed20260910_n1000.parquet").head(100)
                pool_dev += (pt.requested_offset_m - pt.center_offset_m).abs().tolist()
            row = dict(mean_abs_requested_offset_minus_centre_m=float(np.mean(pool_dev)),n=int(len(g)), teleport_unclipped=float(g.teleport.astype(float).mean()), teleport_clipped=float(c.teleport.astype(float).mean()),
                       offroad_unclipped=float(g.offroad_vl.astype(float).mean()), offroad_clipped=float(c.offroad_vl.astype(float).mean()),
                       solid_unclipped=float(g.any_solid.astype(float).mean()), solid_clipped=float(c.any_solid.astype(float).mean()))
            pilot[cls] = row
            L.append(f"| {cls} | {row['n']} | {f3(row['teleport_unclipped'])} | {f3(row['teleport_clipped'])} | {f3(row['offroad_unclipped'])} | {f3(row['offroad_clipped'])} | {f3(row['solid_unclipped'])} | {f3(row['solid_clipped'])} | {f3(row['mean_abs_requested_offset_minus_centre_m'])} |")
        manifest["unclipped_pilot"] = pilot
        L += ["", "### 4b. chmed horizon (primary): background solid hit, off-road, physics, sampler-invalid, valid-and-critical yield", "",
              "| scope | arm | n | solid hit [boot CI] | Δ-above-real [CI] | near-int. solid | off-road VL | off-road E7 | teleport | wrong-way | gate1 | sampler-invalid | valid∧critical |",
              "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        order = ["real", "sakura_bc", "sakura_plain_extra", "sakura_plain", "sakura_route", "sakura_route_kde", "sakura_route_kde_noclip", "sakura_route_kde_cond_39_180", "ours3_disk"]
        t5rows = {}
        for scope in ["pooled"] + CLASSES:
            s = t5s[t5s.horizon.eq("chmed") & t5s.scope.eq(scope)].set_index("arm")
            for a in order:
                if a not in s.index:
                    continue
                r = s.loc[a]
                t5rows[f"{scope}/{a}"] = {k: (float(r[k]) if pd.notna(r[k]) else None) for k in ("n", "any_solid_rate", "any_solid_delta_real", "any_near_solid_rate", "offroad_vl_rate", "offroad_e7_rate", "teleport_rate", "wrongway_rate", "phys_gate1_rate", "sampler_invalid_rate", "valid_and_critical_rate")}
                delta = "—" if a == "real" else f"{r.any_solid_delta_real:+.3f} [{f3(r.get('any_solid_delta_real_boot_lo'))}, {f3(r.get('any_solid_delta_real_boot_hi'))}]"
                L.append(f"| {scope} | {a} | {int(r.n)} | {f3(r.any_solid_rate)} [{f3(r.any_solid_boot_lo)}, {f3(r.any_solid_boot_hi)}] | {delta} | {f3(r.any_near_solid_rate)} | "
                         f"{f3(r.offroad_vl_rate)} | {f3(r.offroad_e7_rate)} | {'n/a' if a == 'real' else f3(r.teleport_rate)} | {f3(r.wrongway_rate)} | {f3(r.phys_gate1_rate)} | {f3(r.sampler_invalid_rate)} | {f3(r.valid_and_critical_rate)} |")
        manifest["table5_chmed"] = t5rows
        st = pd.read_csv(RES / "table5_validity_stats_sakura.csv")
        st = st[st.horizon.eq("chmed") & st.metric.eq("any_solid") & st.scope.eq("pooled")]
        if len(st):
            L += ["", "Scenario-paired sign-flip tests (chmed, pooled, any_solid; Holm within family): " +
                  "; ".join(f"{r.arm} vs {r.reference}: diff {r.mean_paired_diff:+.3f}, p={r.p_signflip:.4f} (Holm {r.p_holm_family:.4f}, n={int(r.n_scenarios)})" for r in st.itertuples())]
        L.append("")

    # ── 5. Table 2 ──
    t2p = RES / "table2_sakura_arms.csv"
    if t2p.exists():
        t2 = pd.read_csv(t2p)
        L += ["## 5. Six-measure fidelity (Table 2 format; medians over S_k = scenes finite for all five arms; b_k = worst arm; D_m = mean median/b_k)", ""]
        t2rows = {}
        for scope in CLASSES[:4] + ["pooled", "special_39_180"]:
            t = t2[t2.scope.eq(scope)]
            if t.empty:
                continue
            L += [f"### {scope} (common cohort {int(t.n_common_cohort.iloc[0])}, eligible {int(t.n_eligible_common.iloc[0])})", "",
                  "| measure | n S_k | " + " | ".join(T2_ARMS) + " | b_k |", "|---|---:|" + "---:|" * (len(T2_ARMS) + 1)]
            for m, lab in MEAS:
                tm = t[t.measure.eq(m)].set_index("arm")
                best = tm.median_err_common_Sk.idxmin() if tm.median_err_common_Sk.notna().any() else None
                cells = [(f"**{f3(tm.median_err_common_Sk.get(a, np.nan))}**" if a == best else f3(tm.median_err_common_Sk.get(a, np.nan))) for a in T2_ARMS]
                L.append(f"| {lab} | {int(tm.n_common_Sk.iloc[0])} | " + " | ".join(cells) + f" | {f3(tm.b_k.iloc[0])} |")
                t2rows[f"{scope}/{m}"] = {a: (float(tm.median_err_common_Sk.get(a, np.nan))) for a in T2_ARMS}
            dm = t.drop_duplicates("arm").set_index("arm").D_m
            L.append("| **D_m** | | " + " | ".join(f"**{f3(dm.get(a, np.nan))}**" for a in T2_ARMS) + " | |")
            t2rows[f"{scope}/D_m"] = {a: float(dm.get(a, np.nan)) for a in T2_ARMS}
            ne = t[t.measure.eq("pet")].set_index("arm")
            L.append("| no-event gen / PET flips | | " + " | ".join(f"{int(ne.n_generated_no_event.get(a, 0))} / {int(ne.n_pet_sign_flip.get(a, 0))}" for a in T2_ARMS) + " | |")
            L.append("| own-scene median (all finite, n) | | " + " | ".join(f"{f3(t[t.measure.eq('dtw') & t.arm.eq(a)].median_err_arm_all_finite.iloc[0])} DTW (n={int(t[t.measure.eq('dtw') & t.arm.eq(a)].n_arm_all_finite.iloc[0])})" for a in T2_ARMS) + " | |")
            L.append("")
        manifest["table2"] = t2rows
        ct = pd.read_csv(RES / "table2_sakura_arms_contrasts.csv")
        if len(ct):
            L += ["Paired contrasts (median paired difference on S_k, 34 cluster bootstrap CI over global groups, cluster sign-flip p, Holm over the 5 classes x 6 measures per contrast; * = Holm < 0.05):", "",
                  "| scope | contrast | measure | n | median diff | 95% CI | p raw | p Holm | favours |", "|---|---|---|---:|---:|---|---:|---:|---|"]
            for r in ct[ct.contrast.isin(["sakura_route_minus_sakura_plain", "ours3_disk_minus_sakura_route"])].itertuples():
                L.append(f"| {r.scope} | {r.contrast} | {r.paper_measure} | {r.n_paired} | {f3(r.median_paired_diff)} | [{f3(r.ci95_low_global_group)}, {f3(r.ci95_high_global_group)}] | {f3(r.p_signflip_raw, 4)} | {f3(r.p_holm, 4)}{'*' if r.holm_significant_005 else ''} | {r.favours} |")
            L.append("")

    # ── 6. Tables 3/4 ──
    t3p = RES / "table3_coverage_sakura.csv"
    if t3p.exists():
        t3 = pd.read_csv(t3p)
        t3r = json.loads((RES / "table34_manifest.json").read_text())
        L += ["## 6. Coverage of the real set (Table 3 format; matched489, eps x1; mean over 20 orderings, mean over 3 seeds)", "",
              "| class | arm | space | budget 10 | budget 100 | budget 1000 (pool) | valid-only @10 | valid-only @100 | valid-only @1000 | ours3_disk_kde @100 / @1000 (Table 3) | svd_d5_kde analytic @100 / @1000 (Table 3) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
        ref = pd.read_csv(RES / "table3_coverage.csv")
        ref = ref[ref.real_set.eq("matched489") & ref.eps_mult.eq(1.0) & ref.pool.eq("all")]
        cov_rows = {}

        def cov_at(tbl, arm, cls, space, b, pool="all"):
            g = tbl[tbl.arm.eq(arm) & tbl.subset.eq(cls) & tbl.space.eq(space) & tbl.budget.eq(b) & tbl.pool.eq(pool) & tbl.real_set.eq("matched489") & tbl.eps_mult.eq(1.0)]
            return float(g.coverage_mean.mean()) if len(g) else np.nan
        for cls in CLASSES[:4]:
            for space in ("interaction", "path", "joint"):
                c10, c100, c1000 = [cov_at(t3, "sakura_route_kde", cls, space, b) for b in (10, 100, 1000)]
                v10, v100, v1000 = [cov_at(t3, "sakura_route_kde", cls, space, b, "valid") for b in (10, 100, 1000)]
                o100, o1000 = cov_at(ref, "ours3_disk_kde", cls, space, 100), cov_at(ref, "ours3_disk_kde", cls, space, 1000)
                s100, s1000 = cov_at(ref, "svd_d5_kde_matched_analytic", cls, space, 100), cov_at(ref, "svd_d5_kde_matched_analytic", cls, space, 1000)
                cov_rows[f"{cls}/{space}"] = dict(b10=c10, b100=c100, b1000=c1000, valid_b10=v10, valid_b100=v100, valid_b1000=v1000, ours3_disk_kde_b100=o100, ours3_disk_kde_b1000=o1000, svd_kde_b100=s100, svd_kde_b1000=s1000)
                L.append(f"| {cls} | sakura_route_kde | {space} | {f3(c10)} | {f3(c100)} | {f3(c1000)} | {f3(v10)} | {f3(v100)} | {f3(v1000)} | {f3(o100)} / {f3(o1000)} | {f3(s100)} / {f3(s1000)} |")
            for arm in ("sakura_route_defaults", "sakura_plain"):
                g = t3[t3.arm.eq(arm) & t3.subset.eq(cls) & t3.real_set.eq("matched489") & t3.eps_mult.eq(1.0)]
                for space in ("interaction", "path", "joint"):
                    ga, gv = g[g.space.eq(space) & g.pool.eq("all")], g[g.space.eq(space) & g.pool.eq("valid")]
                    if len(ga):
                        cov_rows[f"{cls}/{space}/{arm}"] = dict(pool=float(ga.coverage_mean.iloc[0]), valid=float(gv.coverage_mean.iloc[0]) if len(gv) else np.nan, n=int(ga.n_samples.iloc[0]))
                        L.append(f"| {cls} | {arm} (nominal, 1 per scene) | {space} | | | {f3(float(ga.coverage_mean.iloc[0]))} (n={int(ga.n_samples.iloc[0])}) | | | {f3(float(gv.coverage_mean.iloc[0])) if len(gv) else 'n/a'} | | |")
        manifest["table3_coverage"] = cov_rows
        t4 = pd.read_csv(RES / "table4_similarity_sakura.csv")
        t4r = pd.read_csv(RES / "table4_similarity.csv")
        L += ["", "### Table 4 (all seeds pooled): W1 to the real set per descriptor; joint 6-D z-scored W1 (POT emd2)", "",
              "| class | arm | PET s | d_min m | alpha deg | cx m | cy m | u_c m/s | DTW m (mean) | joint6 z | out-of-support (PET/d_min) | bracket rate all (PET/d_min) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        sim_rows = {}
        for cls in CLASSES[:4]:
            for tbl, arm in ((t4, "sakura_route_kde"), (t4, "sakura_route_defaults"), (t4, "sakura_plain"), (t4r, "ours3_disk_kde"), (t4r, "svd_d5_kde_matched_analytic")):
                g = tbl[tbl.arm.eq(arm) & tbl.subset.eq(cls) & tbl.scope.eq("all_seeds")].set_index("descriptor")
                if g.empty:
                    continue
                w = {k: float(g.w1.get(k, np.nan)) for k in ("pet", "d_min", "alpha", "conflict_x", "conflict_y", "u_c", "dtw", "joint6_zscored")}
                sim_rows[f"{cls}/{arm}"] = w
                oos = f"{f3(g.out_of_real_support.get('pet', np.nan))}/{f3(g.out_of_real_support.get('d_min', np.nan))}"
                br = f"{f3(g.bracket_rate_all.get('pet', np.nan))}/{f3(g.bracket_rate_all.get('d_min', np.nan))}" if "bracket_rate_all" in g else "n/a"
                L.append(f"| {cls} | {arm} | " + " | ".join(f3(w[k]) for k in ("pet", "d_min", "alpha", "conflict_x", "conflict_y", "u_c", "dtw", "joint6_zscored")) + f" | {oos} | {br} |")
        manifest["table4_w1"] = sim_rows
        cdp = RES / "table3_sakura_39_180_cond.csv"
        if cdp.exists():
            cd = pd.read_csv(cdp)
            L += ["", "### special_39_180 — frozen pre-v3 conditional results; excluded from this rerun", "",
                  "| scope | n | valid (gate) | recovered interaction | path | joint | PET median (real 9.03) | d_min median (real 3.81) | DTW median | bracket PET/d_min |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            for r in cd.itertuples():
                L.append(f"| {r.scope} | {r.n} | {r.n_valid_gate} | {f3(r.recovered_interaction_rate)} | {f3(r.recovered_path_rate)} | {f3(r.recovered_joint_rate)} | {f3(r.pet_median)} | {f3(r.d_min_median)} | {f3(r.dtw_median)} | {r.pet_bracket}/{r.d_min_bracket} |")
            manifest["cond_39_180"] = cd.to_dict("records")
        L.append("")

    # ── 7. deviations ──
    L += ["## 7. Deviations / notes", "",
          "- Bases: 2 cutinr scenes (752_899, 926_905) and 1 cutinl scene (1669_1657) have no exact-window plain SAKURA base and are absent from every SAKURA arm (denominators are bases found, not class size).",
          "- sakura_plain_extra also renders the cutinl population copy of 39_180 (same scene/window as the special, cutinl source tracks); it is not among the 192 sakura_bc samples. The _xosc_base_sakura bases equal the _xosc_base_bc bodies of the 192 (checked on 101_84: body identical; the base EndSpeed differs by construction and is overridden; the SA duration differs by 0.1 s).",
          "- sakura_route no-route fallback follows 210 verbatim: the base is rendered untouched (constant start speed, plain chord); the KDE centres of the same scenes use v_avg = GT path length / duration, so a no-route KDE centre and its sakura_route default are not the same render.",
          "- special_39_180 has no route: sakura_route = plain chord at the base constant speed (11.118 km/h); sakura_plain = chord with EndSpeed = recorded endpoint speed 8.448 km/h.",
          "- Offset clip: recorded per draw (requested vs applied, offset_clipped, offset_clip_bounds); the unclipped pilot re-executes the same first 100 draws of seed 20260910 per class without the clip.",
          "- 45b horizons: ch / chmed cutoffs are Table 5's per-scene values (all SAKURA scenes have a Table 5 row); the values 45's rule would derive from the new arms alone are kept as ch_dur_own_s / chmed_dur_own_s. The real reference was re-scored through the same loop and compared with Table 5's real rows (see table5_sakura_manifest.json real_reference_consistency).",
          "- 46b: eps, scaling and real reference identical to Table 3/4 (asserted against table34_manifest.json); the valid-only pool uses the 45b flags (all draws flagged, so valid rows exist at every budget); DTW cache separate (e6_dtw_cache_sakura.csv).",
          f"- The v3 cutinl KDE fit uses {plan['fits']['cutinl']['n_fit']} class scenes with a base. The 39_180 conditional arm and its earlier fit are frozen pre-v3 references; no v3 conditional draws were made.",
          "- Superseded (retained, not scored) batches: the first render of the KDE / noclip arms carried the special_39_180 label and source tracks on the cutinl draws centred on 39_180 (a uid-keyed lookup collision in scenes_by_uid, fixed in 85 before the second render; trajectories identical): runs/sakura_route_kde_s20260910/d171f50a7af7775f, s20260911/64269e6bd15dc61f, s20260912/e3e816bb629f46e8, sakura_route_kde_noclip_pilot/b94488dfeef5af24. Scored batches = latest_batch.json of each arm (section 2).",
          "- Table 5 'pooled' rows here pool only keeptl / keeptl_sw / cutinl / cutinr (+ frozen special rows for its own arms); the copied ours3_disk / real / sakura_bc rows are restricted to the same classes, so they differ from Table 5's 5-class pooled rows.",
          "- Smoke batches with --limit 3 (run_ids fbce695b68a1b64c, 92a62f8b563ec74a, 9c44b198055452f7, 78710e2931fe7617, a33ce0f9b92aef95) are retained under runs/ but are not the latest_batch of any arm and are not scored.", ""]

    # ── manifest: sources ──
    src = {}
    for p in [PROJECT / "scripts/85_sakura_arms.py", PROJECT / "scripts/86_sakura_score_31_32.py", PROJECT / "scripts/87_table2_sakura_arms.py",
              PROJECT / "scripts/45b_validity_extra_arms.py", PROJECT / "scripts/46b_coverage_similarity_extra.py", PROJECT / "scripts/88_sakura_arms_report.py",
              PROJECT / "scripts/20_ours3_generate.py", PROJECT / "scripts/25_sakura_generate.py", PROJECT / "scripts/31_ours3_nonpet.py", PROJECT / "scripts/32_bbox_pet.py",
              PROJECT / "scripts/35_six_measure_table.py", PROJECT / "scripts/45_validity_bg.py", PROJECT / "scripts/46_coverage_similarity.py",
              XC / "scripts/sakura_route.py", XC / "scripts/210_sakura_arms.py", XC / "scripts/180_trajdtw_aggregate.py",
              PROJECT.parent / "exp_coverage_velocity/scripts/cvlib.py", PROJECT.parent / "sr-tlkeep-experiment/core/kde_sampling.py",
              PROJECT.parent / "hetero-param/hetero_param/esmini_exec.py", RES / "svd_d5_cases.csv", RES / "sakura_arms_scenes.csv", RES / "sakura_arms_plan.json",
              PROJECT.parent / "HetroD-labeler/data/00_tracks.parquet", PROJECT.parent / "retrieval-scenarios/data/map/tyms.xodr", PROJECT.parent / "esmini/bin/esmini"] \
             + sorted((XC / "results").glob("route_plan_*.json")) + sorted(set(Path(p) for p in sc.source_tracks.dropna())):
        if Path(p).exists():
            src[str(p)] = sha(p)
    bases = {r.scenario_uid + "@" + r.subset: dict(base=r.base, sha256=r.base_sha256) for r in sc[sc.base_found].itertuples()}
    manifest["sources_sha256"] = src
    manifest["bases"] = bases
    outs = ["sakura_arms_scenes.csv", "sakura_arms_plan.json", "table2_sakura_arms.csv", "table2_sakura_arms_cases.csv", "table2_sakura_arms_contrasts.csv",
            "table5_validity_summary_sakura.csv", "table5_validity_samples_sakura.csv", "table5_validity_stats_sakura.csv", "table5_validity_hits_sakura.csv",
            "table3_coverage_sakura.csv", "table3_first_hit_sakura.csv", "table3_variation_sakura.csv", "table4_similarity_sakura.csv", "table3_sakura_39_180_cond.csv",
            "bbox_pet_cache_sak.json"] + [f"sak_{t}_nonpet_paired.csv" for t in ["sakura_plain_extra", "sakura_route", "sakura_route_kde_s20260910", "sakura_route_kde_s20260911", "sakura_route_kde_s20260912", "sakura_route_kde_noclip", "sakura_route_kde_cond_39_180"]] \
        + [f"bbox_pet_sak_{t}_paired.csv" for t in ["sakura_plain_extra", "sakura_route", "sakura_route_kde_s20260910", "sakura_route_kde_s20260911", "sakura_route_kde_s20260912", "sakura_route_kde_noclip", "sakura_route_kde_cond_39_180"]]
    manifest["outputs_sha256"] = {o: sha(RES / o) for o in outs if (RES / o).exists()}
    (RES / "sakura_arms_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (RES / "SAKURA_ARMS_REPORT.md").write_text("\n".join(L))
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
