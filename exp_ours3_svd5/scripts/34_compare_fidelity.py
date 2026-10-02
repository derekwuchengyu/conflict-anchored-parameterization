#!/usr/bin/env python3
"""Paired, scene-matched interaction fidelity with actor-group bootstrap.

No new trajectories, estimator changes or outcome-based scene selection.
All cases and finite/undefined counts are saved. Clock-misaligned cases remain
diagnostic and are never used to make a method superiority claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results"
METRICS = {"min_dist": "m", "conflict_angle": "deg", "agent_arr_speed": "m/s",
           "conflict_point_distance": "m", "min_ttc": "s"}
SEED, REPEATS = 20260910, 2000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors(frame):
    d = frame.copy()
    d["absdiff_conflict_point_distance"] = np.hypot(
        d.generated_conflict_x - d.real_conflict_x,
        d.generated_conflict_y - d.real_conflict_y)
    return d


def bootstrap(ours, baseline, groups):
    delta = ours - baseline
    unique = np.unique(groups)
    if len(unique) < 2:
        return np.nan, np.nan
    indices = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(SEED)
    values = np.empty(REPEATS)
    for i in range(REPEATS):
        ix = np.concatenate([indices[j] for j in rng.integers(len(unique), size=len(unique))])
        values[i] = np.median(delta[ix])
    return tuple(np.quantile(values, [.025, .975]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=OUT / "ours3_population_nonpet_paired.csv")
    parser.add_argument("--svd", type=Path, default=OUT / "svd_d5_nonpet_paired.csv")
    parser.add_argument("--output-prefix", default="fidelity_nonpet")
    args = parser.parse_args()
    ours = errors(pd.read_csv(args.ours))
    svd = errors(pd.read_csv(args.svd))
    assert ours.scenario_uid.is_unique
    assert ours.is_nominal_default.eq(True).all(), "Only the default, no best-of-N selection"
    detail, rows, counts = [], [], []
    for mode, source in svd.groupby("mode", sort=False):
        if not source.scenario_uid.is_unique:
            raise ValueError("SVD source keys are duplicated")
        joint = source.merge(ours, on=["scenario_uid", "subset"], how="left",
                              suffixes=("_svd", "_ours"), validate="one_to_one", indicator=True)
        for subset, g in joint.groupby("subset", sort=False):
            counts.append(dict(subset=subset, svd_mode=mode, source_cases=len(g),
                               ours_source_available=int(g._merge.eq("both").sum()),
                               ours_source_unavailable=int(g._merge.eq("left_only").sum()),
                               both_clock_aligned=int((g.clock_aligned_svd.eq(True)&g.clock_aligned_ours.eq(True)).sum())))
        for _, r in joint.iterrows():
            common = r["_merge"] == "both"
            clock = bool(r.clock_aligned_svd == True and r.clock_aligned_ours == True) if common else False
            good = bool(common and r.score_status_svd == "ok" and r.score_status_ours == "ok")
            for metric, unit in METRICS.items():
                key = f"absdiff_{metric}"
                x, y = float(r.get(key+"_ours", np.nan)), float(r.get(key+"_svd", np.nan))
                finite = bool(np.isfinite([x, y]).all())
                item = dict(subset=r.subset, scenario_uid=r.scenario_uid,
                            scenario_id=r.scenario_id_svd, group_id=r.group_id,
                            svd_mode=mode, metric=metric, unit=unit, ours_available=common,
                            both_scored=good, clock_aligned=clock, paired_finite=finite,
                            ours_error=x, svd_error=y, paired_error_difference=x-y if finite else np.nan,
                            comparison_eligible=bool(good and clock and finite))
                detail.append(item)
    detailed = pd.DataFrame(detail)
    for (subset, mode, metric), g in detailed.groupby(["subset", "svd_mode", "metric"], sort=False):
        for scope, candidates in (("clock_aligned_primary", g[g.clock_aligned]),
                                  ("all_cases_diagnostic", g)):
            eligible = candidates[candidates.both_scored & candidates.paired_finite]
            x, y = eligible.ours_error.to_numpy(), eligible.svd_error.to_numpy()
            groups = eligible.group_id.to_numpy(str)
            lo, hi = bootstrap(x, y, groups) if len(x) else (np.nan, np.nan)
            rows.append(dict(subset=subset, svd_mode=mode, scope=scope, metric=metric,
                unit=METRICS[metric], source_cases=len(g), scope_cases=len(candidates),
                ours_available=int(candidates.ours_available.sum()),
                both_scored=int(candidates.both_scored.sum()), paired_finite_n=len(eligible),
                paired_nonfinite_or_unscored_n=len(candidates)-len(eligible), n_actor_groups=len(np.unique(groups)),
                ours_median_abs_error=float(np.median(x)) if len(x) else np.nan,
                svd_median_abs_error=float(np.median(y)) if len(y) else np.nan,
                median_paired_error_difference=float(np.median(x-y)) if len(x) else np.nan,
                paired_cluster_bootstrap_ci_low=lo, paired_cluster_bootstrap_ci_high=hi,
                ours_lower_error_fraction=float(np.mean(x<y)) if len(x) else np.nan,
                exploratory_ci_favors="ours" if hi<0 else "svd" if lo>0 else "uncertain",
                note="Per-metric exploratory CI; not a multiplicity-adjusted confirmatory claim"))
    summary = pd.DataFrame(rows)
    detailed.to_csv(OUT / f"{args.output_prefix}_cases.csv", index=False)
    summary.to_csv(OUT / f"{args.output_prefix}_summary.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / f"{args.output_prefix}_availability.csv", index=False)
    report = ["# 同場互動相似度：ours3 對原始 SVD d5", "",
        "各方法使用自己的default reconstruction，沒有KDE best-of-N或按結果更换來源。負的成對差值表示ours誤差較小。",
        "主列限定起始時鐘一致；所有來源、缺模板、非有限值及時鐘疑點均在明細保留。各自GT／生成支持窗與取樣解析度沿用既有評分器，並非固定頭尾試驗。",
        "95%區間由共享ego／target群組重抽2000次得到。這些是多指標探索區間，沒有多重比較校正，不能單憑一格便作確認性顯著主張。PET另表，不能以本表替代PET。", "",
        "| 類別 | SVD訓練 | 指標 | 配對n / groups | ours誤差中位 | SVD誤差中位 | 成對差中位 [95%區間] |",
        "|---|---|---|---:|---:|---:|---|" ]
    for r in summary[summary.scope.eq("clock_aligned_primary")].itertuples():
        report.append(f"| {r.subset} | {r.svd_mode} | {r.metric} ({r.unit}) | {r.paired_finite_n} / {r.n_actor_groups} | {r.ours_median_abs_error:.4g} | {r.svd_median_abs_error:.4g} | {r.median_paired_error_difference:.4g} [{r.paired_cluster_bootstrap_ci_low:.4g}, {r.paired_cluster_bootstrap_ci_high:.4g}] |")
    (OUT / f"{args.output_prefix}_REPORT.md").write_text("\n".join(report)+"\n")
    (OUT / f"{args.output_prefix}_manifest.json").write_text(json.dumps({
        "sources": [{"path":str(p),"sha256":sha(p)} for p in [args.ours,args.svd,Path(__file__)]],
        "metrics":METRICS,"bootstrap_repeats":REPEATS,"bootstrap_seed":SEED,
        "bootstrap_unit":"global shared ego/target connected component within each class",
        "counterpart":"same reference scenario default; no sample search",
        "predeclared_clock_policy":"exclude misaligned clocks from primary claim, retain in diagnostics",
        "dimensions":"ours3 variable knobs plus scene context; SVD5 coefficients; context information differs",
        "multiple_comparison_adjusted":False,"PET_included":False
    },indent=2)+"\n")
    print(summary[summary.scope.eq("clock_aligned_primary")][[
        "subset","svd_mode","metric","paired_finite_n","ours_median_abs_error",
        "svd_median_abs_error","exploratory_ci_favors"]].to_string(index=False))


if __name__ == "__main__":
    main()
