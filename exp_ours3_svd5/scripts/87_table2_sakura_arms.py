#!/usr/bin/env python3
"""Table 2 format for the SAKURA arms: six interaction discrepancy measures per class.

Arms: sakura_plain (= the 192 sakura_bc rows of results/table2_six_measures_cases.csv + the new
sakura_plain_extra samples: cutinr, special_39_180, cutinl copy of 39_180), sakura_route (new),
and ours3_disk / svd_d5_fullfit / svd_d5_logo copied from results/table2_six_measures_cases.csv on the
common scenes.  Errors for the new arms are computed exactly as scripts/35_six_measure_table.py
score_arm does for trace arms (five non-PET quantities from the 31 outputs, PET from the 32 outputs,
DTW with the verbatim exp_cross_coverage/scripts/180_trajdtw_aggregate.py traj_dtw loaded by 35);
aggregation = 35's rule: S_k = scenes finite for every arm (clock-aligned, score ok, PET ok),
median over S_k, b_k = worst arm median, D_m = mean_k median/b_k over the arm set
{sakura_plain, sakura_route, svd_d5_fullfit, svd_d5_logo, ours3_disk}.  A second block gives each
arm's median over ALL its own finite scenes (n differs per arm).  special_39_180 is a separate scope:
its ours3_disk / SVD rows are the cutinl-population rows of the same scene/window (anchor 2609).

Outputs: results/table2_sakura_arms.csv, table2_sakura_arms_cases.csv, table2_sakura_arms_contrasts.csv,
table2_sakura_manifest.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
_spec = importlib.util.spec_from_file_location("six35", PROJECT / "scripts/35_six_measure_table.py")
M35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M35)          # traj_dtw (AST verbatim), GroundTruth, build_cases, signflip_p, bootstrap, MEASURES, PAPER, UNITS, holm
MEASURES, PAPER, UNITS = M35.MEASURES, M35.PAPER, M35.UNITS
SPECIAL_UID, SPECIAL_CLS = "HetroD/00/39_180/2424-3002", "special_39_180"
CLASSES = ["keeptl", "keeptl_sw", "cutinl", "cutinr"]
ARMS = ["sakura_plain", "sakura_route", "svd_d5_fullfit", "svd_d5_logo", "ours3_disk"]
NEW = {"sakura_plain_extra": (RES / "sak_sakura_plain_extra_nonpet_paired.csv", RES / "bbox_pet_sak_sakura_plain_extra_paired.csv"),
       "sakura_route": (RES / "sak_sakura_route_nonpet_paired.csv", RES / "bbox_pet_sak_sakura_route_paired.csv")}
CONTRASTS = [("sakura_route", "sakura_plain"), ("ours3_disk", "sakura_plain"), ("ours3_disk", "sakura_route"),
             ("svd_d5_logo", "sakura_route"), ("svd_d5_fullfit", "sakura_route")]


def err_rows(arm, paired_path, pet_path, gt, fullfit):
    """35.score_arm trace branch, keyed by (subset, scenario_uid) so the special scene may coexist with its cutinl copy."""
    paired = pd.read_csv(paired_path, low_memory=False)
    paired["input_case_index"] = np.arange(len(paired))
    pet = pd.read_csv(pet_path, low_memory=False)
    pet = pet[pet.input_paired_path.map(lambda p: str(Path(p).resolve())) == str(paired_path.resolve())].set_index("input_case_index")
    assert len(pet) == len(paired), (arm, len(pet), len(paired))
    rows = []
    for r in paired.itertuples():
        p = pet.loc[r.input_case_index]
        assert str(p.scenario_uid) == str(r.scenario_uid) and str(p.subset) == str(r.subset)
        real_pet, gen_pet = float(p.real_pet), float(p.generated_pet)
        both = np.isfinite([real_pet, gen_pet]).all()
        d = dict(err_pet=abs(real_pet - gen_pet) if both else np.nan,
                 err_dmin=abs(r.generated_min_dist - r.real_min_dist), err_alpha=abs(r.generated_conflict_angle - r.real_conflict_angle),
                 err_cpoint=float(np.hypot(r.generated_conflict_x - r.real_conflict_x, r.generated_conflict_y - r.real_conflict_y)),
                 err_uc=abs(r.generated_agent_arr_speed - r.real_agent_arr_speed))
        gg = gt.actor_path(r.source_tracks, r.scenario_id)
        if isinstance(r.trajectory_path, str) and Path(r.trajectory_path).exists():
            tidy = pd.read_parquet(r.trajectory_path, columns=["role", "frame", "x", "y"])
            lo, hi = float(r.metadata_min_frame), float(r.metadata_max_frame)
            tg = tidy[tidy.role.eq("target") & tidy.frame.between(lo - .5, hi + .5)].sort_values("frame")
            n_gen = len(tg)
            dtw, cov = M35.traj_dtw(tg[["x", "y"]], gg) if n_gen >= 2 else (np.nan, np.nan)
        else:
            n_gen, dtw, cov = 0, np.nan, np.nan
        uid = r.scenario_uid
        rows.append(dict(arm=arm, subset=r.subset, scenario_uid=uid, scenario_id=str(r.scenario_id), sample_id=r.sample_id,
                         group_id=fullfit.group_id.get(uid, np.nan), group_id_within_class=fullfit.group_id_within_class.get(uid, np.nan),
                         clock_aligned=bool(r.clock_aligned), score_status=r.score_status, pet_score_status=p.pet_score_status,
                         eligible=bool(r.clock_aligned) and r.score_status == "ok" and p.pet_score_status == "ok",
                         real_pet=real_pet, generated_pet=gen_pet, generated_no_event=bool(np.isinf(gen_pet)), real_no_event=bool(np.isinf(real_pet)),
                         pet_flip=bool(both and np.sign(real_pet) != np.sign(gen_pet)), pet_reason=p.absdiff_pet_reason,
                         real_min_dist=r.real_min_dist, generated_min_dist=r.generated_min_dist, real_conflict_angle=r.real_conflict_angle,
                         generated_conflict_angle=r.generated_conflict_angle, real_conflict_x=r.real_conflict_x, real_conflict_y=r.real_conflict_y,
                         generated_conflict_x=r.generated_conflict_x, generated_conflict_y=r.generated_conflict_y,
                         real_agent_arr_speed=r.real_agent_arr_speed, generated_agent_arr_speed=r.generated_agent_arr_speed,
                         **d, err_dtw=dtw, dtw_coverage=cov, generated_path_samples=n_gen, gt_path_samples=len(gg),
                         geometry_variant_id=r.geometry_variant_id, input_paired_path=str(paired_path), input_case_index=int(r.input_case_index)))
    return pd.DataFrame(rows)


def evaluate(cases, arms, scope_name, uids_all):
    wide = {a: cases[cases.arm.eq(a)].set_index("scenario_uid") for a in arms}
    uids = sorted(set(uids_all).intersection(*[set(w.index) for w in wide.values()]))
    elig = [u for u in uids if all(bool(wide[a].loc[u, "eligible"]) for a in arms)]
    rows = []
    med, Sk = {}, {}
    for m in MEASURES:
        col = f"err_{m}"
        S = [u for u in elig if all(np.isfinite(float(wide[a].loc[u, col])) for a in arms)]
        Sk[m] = S
        med[m] = {a: float(np.median(wide[a].loc[S, col].astype(float))) if S else np.nan for a in arms}
    bk = {m: (np.nanmax(list(med[m].values())) if Sk[m] else np.nan) for m in MEASURES}
    Dm = {}
    for a in arms:
        ratios = [med[m][a] / bk[m] for m in MEASURES if Sk[m] and np.isfinite(bk[m]) and bk[m] > 0]
        Dm[a] = float(np.mean(ratios)) if ratios else np.nan
    for a in arms:
        A = wide[a]
        Ael = A.loc[elig]
        for m in MEASURES:
            col = f"err_{m}"
            own = A[A.eligible.astype(bool)][col].astype(float)
            own = own[np.isfinite(own)]
            rows.append(dict(scope=scope_name, arm=a, measure=m, paper_measure=PAPER[m], unit=UNITS[m], n_scenes_arm=int(len(A)),
                             n_common_cohort=len(uids), n_eligible_common=len(elig), n_common_Sk=len(Sk[m]),
                             median_err_common_Sk=med[m][a], b_k=bk[m],
                             normalized=med[m][a] / bk[m] if np.isfinite(bk[m]) and bk[m] > 0 else np.nan, D_m=Dm[a],
                             n_arm_all_finite=int(len(own)), median_err_arm_all_finite=float(own.median()) if len(own) else np.nan,
                             n_generated_no_event=int(Ael.generated_no_event.sum()), n_real_no_event=int(Ael.real_no_event.sum()),
                             n_pet_sign_flip=int(Ael.pet_flip.sum()),
                             dtw_coverage_median=float(Ael.loc[Sk[m], "dtw_coverage"].median()) if m == "dtw" and Sk[m] else np.nan))
    contrasts = []
    for x_arm, y_arm in CONTRASTS:
        if x_arm not in arms or y_arm not in arms:
            continue
        for m in MEASURES:
            S, col = Sk[m], f"err_{m}"
            x = wide[x_arm].loc[S, col].astype(float).to_numpy()
            y = wide[y_arm].loc[S, col].astype(float).to_numpy()
            gg = wide[x_arm].loc[S, "group_id"].astype(str).to_numpy()
            lo, hi = M35.bootstrap(x, y, gg) if len(S) else (np.nan, np.nan)
            p, G, ptype = M35.signflip_p(x - y, gg) if len(S) else (np.nan, 0, "n/a")
            contrasts.append(dict(scope=scope_name, contrast=f"{x_arm}_minus_{y_arm}", measure=m, paper_measure=PAPER[m], n_paired=len(S),
                                  n_groups_global=G, x_median=med[m][x_arm], y_median=med[m][y_arm],
                                  median_paired_diff=float(np.median(x - y)) if len(S) else np.nan, ci95_low_global_group=lo, ci95_high_global_group=hi,
                                  x_lower_fraction=float(np.mean(x < y)) if len(S) else np.nan, p_signflip_raw=p, signflip_type=ptype,
                                  favours=(x_arm if np.median(x - y) < 0 else y_arm if np.median(x - y) > 0 else "tie") if len(S) else ""))
    return rows, contrasts


def main():
    t0 = time.monotonic()
    fullfit, _anchor = M35.build_cases()
    gt = M35.GroundTruth()
    t2 = pd.read_csv(RES / "table2_six_measures_cases.csv", low_memory=False)
    t2["scenario_id"] = t2.scenario_id.astype(str)
    new = pd.concat([err_rows(a, p, q, gt, fullfit) for a, (p, q) in NEW.items()], ignore_index=True)
    bc_rows = t2[t2.arm.eq("sakura_bc")].assign(arm="sakura_plain", source_rows="table2_six_measures_cases.csv sakura_bc")
    extra_rows = new[new.arm.eq("sakura_plain_extra")].assign(arm="sakura_plain", source_rows="sakura_plain_extra (85)")
    dup = extra_rows.set_index(["subset", "scenario_uid"]).index.isin(bc_rows.set_index(["subset", "scenario_uid"]).index)
    print(f"[sakura_plain] extra rows already in the 192 sakura_bc (dropped from sakura_plain, kept in the batch): {extra_rows[dup].sample_id.tolist()}", flush=True)
    plain = pd.concat([bc_rows, extra_rows[~dup]], ignore_index=True)
    route = new[new.arm.eq("sakura_route")].assign(source_rows="sakura_route_defaults (85)")
    ref = t2[t2.arm.isin(["ours3_disk", "svd_d5_fullfit", "svd_d5_logo"])].assign(source_rows="table2_six_measures_cases.csv")
    cases = pd.concat([plain, route, ref], ignore_index=True)
    cases["scope_class"] = cases.subset
    cases.to_csv(RES / "table2_sakura_arms_cases.csv", index=False)
    rows, contrasts = [], []
    for cls in CLASSES + ["pooled"]:
        sub = cases[~cases.subset.eq(SPECIAL_CLS)] if cls == "pooled" else cases[cases.subset.eq(cls)]
        r, c = evaluate(sub, ARMS, cls, sub.scenario_uid.unique())
        rows += r
        contrasts += c
    # special scene: sakura rows of the special class + the cutinl-population rows of the same scene for the reference arms
    ref_special = cases[cases.scenario_uid.eq(SPECIAL_UID) & cases.arm.isin(["ours3_disk", "svd_d5_fullfit", "svd_d5_logo"])]
    if ref_special.arm.nunique() == 3:
        special_status = "recomputed"
        sp = pd.concat([cases[cases.subset.eq(SPECIAL_CLS)], ref_special.assign(subset=SPECIAL_CLS)], ignore_index=True)
        r, c = evaluate(sp, ARMS, SPECIAL_CLS, [SPECIAL_UID])
        rows += r
        contrasts += c
    else:
        # v3 rerun 2026-09-14: 39_180 left the cutinl cohort, so scripts/35 has no ours3_disk / SVD reference rows for it
        # and the special scope's common set S_k would be empty. The rarecase / Table III pipeline is frozen by user
        # decision: keep the pre-rerun special_39_180 rows (extracted from _snapshot_20260914.tar).
        special_status = ("special_39_180 scope frozen: rows and contrasts taken from results/special_39_180_table2_frozen_"
                          "{arms,contrasts}.csv (pre-rerun); p_holm of those contrasts is re-adjusted together with the new pooled scope")
        rows += pd.read_csv(RES / "special_39_180_table2_frozen_arms.csv", low_memory=False).to_dict("records")
        contrasts += pd.read_csv(RES / "special_39_180_table2_frozen_contrasts.csv", low_memory=False).to_dict("records")
        print(f"[special_39_180] {special_status}", flush=True)
    table = pd.DataFrame(rows)
    ctr = pd.DataFrame(contrasts)
    if len(ctr):
        ctr["p_holm"] = np.nan
        for (ctrn, pooled), g in ctr.groupby(["contrast", ctr.scope.eq("pooled") | ctr.scope.eq(SPECIAL_CLS)]):
            idx = g.index[np.isfinite(g.p_signflip_raw)]
            if len(idx):
                ctr.loc[idx, "p_holm"] = M35.holm(ctr.loc[idx, "p_signflip_raw"].tolist())
        ctr["holm_significant_005"] = ctr.p_holm < 0.05
    table.to_csv(RES / "table2_sakura_arms.csv", index=False)
    ctr.to_csv(RES / "table2_sakura_arms_contrasts.csv", index=False)
    man = dict(script="scripts/87_table2_sakura_arms.py", generated=time.strftime("%Y-%m-%d %H:%M:%S"), arms=ARMS, classes=CLASSES + ["pooled", SPECIAL_CLS],
               inputs={str(p): M35.sha256(p) for p in [RES / "table2_six_measures_cases.csv", *[q for pq in NEW.values() for q in pq]]},
               verbatim=dict(dtw=M35.DTW_HASHES), n_rows=dict(cases=len(cases), table=len(table), contrasts=len(ctr)),
               counts=cases.groupby(["arm", "subset"]).size().unstack(fill_value=0).to_dict(),
               note=["sakura_plain = 192 sakura_bc rows (Table 2 cases) + sakura_plain_extra (cutinr 20, special_39_180 1, cutinl copy of 39_180 1)",
                     "special_39_180 scope: ours3_disk / SVD rows are the cutinl-population rows of the same scene/window (ours anchor 2609)",
                     "medians on S_k (finite for all five arms) are the Table 2 rule; median_err_arm_all_finite is each arm's own eligible finite set"],
               elapsed_s=time.monotonic() - t0)
    man["special_39_180_status"] = special_status
    (RES / "table2_sakura_manifest.json").write_text(json.dumps(man, indent=2, default=str) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.3f}"):
        piv = table.pivot_table(index=["scope", "measure"], columns="arm", values="median_err_common_Sk", sort=False)
        print(piv.to_string())
        print(table.drop_duplicates(["scope", "arm"]).pivot(index="scope", columns="arm", values="D_m").to_string())
        print(table[table.measure.eq("pet")].pivot(index="scope", columns="arm", values="n_common_Sk").to_string())


if __name__ == "__main__":
    main()
