#!/usr/bin/env python3
"""Raw d5 control using the pre-scoring ours3-eligible cohort as training data.

Original 513-case models, raw representation and global identity groups remain
unchanged. Arrays retain original case indices; excluded cases are all-NaN and
explicitly not evaluated. Existing non-PET scoring is run on the 489 eligible
cases only, so exclusions are not counted as reconstruction failures.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT / "results"
MODELS = PROJECT / "models/svd_d5_matched"
SCORING_OUT = OUT / "matched_cohort"
HELPER = PROJECT / "scripts/10_svd_d5_models.py"
SCORER = PROJECT / "scripts/30_interaction_metrics.py"
ELIGIBILITY = OUT / "ours3_population.csv"
ORIGINAL_CASES = OUT / "svd_d5_cases.csv"
ORIGINAL_MANIFEST = OUT / "svd_d5_manifest.json"
# cutinl 48 -> 45 on 2026-09-14: 39_180, 39_189 and 1669_1657 were eligible and are now excluded at the cohort root
# (scripts/10 EXCLUDED_SCENARIOS, relabelled away from cut-in-left).
EXPECTED = {"tlkeep": 294, "keeptl": 50, "keeptl_sw": 77, "cutinl": 45, "cutinr": 20}
spec = importlib.util.spec_from_file_location("matched_existing_raw_d5", HELPER)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def digest(path):
    return H.source_entry(Path(path))["sha256"]


def manifest_write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    started = time.perf_counter()
    original = pd.read_csv(ORIGINAL_CASES, dtype={"recording": str})
    cases = original[original["mode"].eq("fullfit")].copy()
    assert not cases[["subset", "scenario_uid"]].duplicated().any()
    eligible_source = pd.read_csv(ELIGIBILITY)
    assert len(eligible_source) == len(cases) and not eligible_source[["subset", "scenario_uid"]].duplicated().any()
    eligibility = eligible_source.set_index(["subset", "scenario_uid"])
    assert set(map(tuple, cases[["subset", "scenario_uid"]].to_numpy())) == set(eligibility.index)
    cohort_hash = digest(ELIGIBILITY)
    grouping = json.loads(ORIGINAL_MANIFEST.read_text())
    group_members = grouping["group_members"]
    assert len(group_members) == cases.group_id.nunique(), (len(group_members), cases.group_id.nunique())
    sources = {Path(__file__).resolve(), HELPER, SCORER, ELIGIBILITY,
               OUT / "ours3_population_manifest.json", PROJECT / "scripts/21_ours3_population.py",
               ORIGINAL_CASES, ORIGINAL_MANIFEST, H.CORE / "core/svd_param.py"}
    sources.update(PROJECT.glob("models/svd_d5/**/*.npz"))
    sources.update(Path(p) for p in cases.source_tracks)
    sources.update(Path(p) for p in cases.source_meta)
    before_hashes = {str(p): digest(p) for p in sorted(sources)}
    statuses, fits, summaries = [], [], []
    for subset, class_cases in cases.groupby("subset", sort=False):
        class_cases = class_cases.sort_values("case_index").reset_index(drop=True)
        n = len(class_cases)
        assert class_cases.case_index.tolist() == list(range(n))
        old_raw = PROJECT / "models/svd_d5" / subset / "raw_features.npz"
        with np.load(old_raw, allow_pickle=False) as f:
            X = f["X_raw"].copy()
            assert class_cases.scenario_uid.tolist() == f["scenario_uid"].tolist()
            assert class_cases.group_id.tolist() == f["group_id"].tolist()
        class_cases["source_input_status"] = class_cases.input_status
        class_cases["cohort_eligible"] = [eligibility.loc[(subset, uid), "status"] == "eligible"
                                          for uid in class_cases.scenario_uid]
        class_cases["cohort_selection_status"] = [eligibility.loc[(subset, uid), "status"] for uid in class_cases.scenario_uid]
        class_cases["cohort_exclusion_detail"] = [str(eligibility.loc[(subset, uid), "detail"]) if not ok else ""
                                                  for uid, ok in zip(class_cases.scenario_uid, class_cases.cohort_eligible)]
        class_cases["training_scope"] = "matched_cohort"
        class_cases["cohort_source"] = str(ELIGIBILITY)
        class_cases["cohort_source_sha256"] = cohort_hash
        class_cases["global_group_source"] = str(ORIGINAL_MANIFEST)
        excluded = ~class_cases.cohort_eligible
        class_cases.loc[excluded, "input_status"] = "outside_prespecified_ours3_cohort"
        class_cases.loc[excluded, "input_detail"] = "Not eligible in frozen pre-scoring 21 population; neither fitted nor reconstructed"
        assert int(class_cases.cohort_eligible.sum()) == EXPECTED[subset]
        assert class_cases.loc[~excluded, "input_status"].eq("ok").all()
        for row in class_cases.itertuples(index=False):
            assert f"HetroD/00/{int(row.ego)}" in group_members[row.group_id]
            assert f"HetroD/00/{int(row.actor)}" in group_members[row.group_id]
        model_dir = MODELS / subset
        model_dir.mkdir(parents=True, exist_ok=True)
        string_arrays = {key: class_cases[key].to_numpy(dtype=str)
                         for key in ("scenario_id", "scenario_uid", "group_id", "input_status")}
        np.savez_compressed(model_dir / "raw_features.npz", X_raw=X, **string_arrays,
            cohort_eligible=class_cases.cohort_eligible.to_numpy(bool),
            nt=H.NT, ny=H.NY, ntheta=H.NTH, fps=H.FPS, training_scope="matched_cohort")
        output = {f"X_rec_{mode}": np.full((n, H.NX), np.nan) for mode in ("fullfit", "logo")}
        output.update({f"Z_{mode}": np.full((n, H.D), np.nan) for mode in ("fullfit", "logo")})
        pop = {"X": X, "cases": class_cases}
        class_fits, class_statuses = [], []
        for mode, group in [("fullfit", None)] + [("logo", g) for g in sorted(class_cases.group_id.unique())]:
            fit, rows = H.fit_and_decode(subset, pop, mode, group, model_dir, output)
            fit.update(training_scope="matched_cohort", n_original_class=n,
                n_prespecified_eligible=EXPECTED[subset], cohort_source_sha256=cohort_hash,
                model_id="matched_cohort__" + fit["model_id"])
            evaluated = class_cases.cohort_eligible & (True if mode == "fullfit" else class_cases.group_id.eq(group))
            fit["n_evaluate_original_indices"] = fit["n_evaluate"]
            fit["n_evaluate"] = int(evaluated.sum())
            if fit["model_npz"]:
                path = PROJECT / fit["model_npz"]
                with np.load(path, allow_pickle=False) as saved:
                    content = {k: saved[k].copy() for k in saved.files}
                training_indices = content["training_case_indices"]
                assert class_cases.iloc[training_indices].cohort_eligible.all()
                if mode == "logo":
                    assert not class_cases.iloc[training_indices].group_id.eq(group).any()
                content.update(evaluation_uids=class_cases.loc[evaluated, "scenario_uid"].to_numpy(dtype=str),
                    training_scope=np.array("matched_cohort"), cohort_source_sha256=np.array(cohort_hash),
                    global_groups_preserved_from_original_513=True)
                np.savez_compressed(path, **content)
            for row in rows:
                row["model_id"] = "matched_cohort__" + row["model_id"]
                row["evaluated_in_matched_cohort"] = bool(row["cohort_eligible"])
                if not row["cohort_eligible"]:
                    row.update(status="not_evaluated_outside_prespecified_cohort",
                        detail="Excluded by pre-scoring ours3 source eligibility; not a fit/generation failure",
                        model_id="", model_npz="", reconstructed_duration_s=None)
                class_statuses.append(row)
            class_fits.append(fit)
        np.savez_compressed(model_dir / "reconstructions.npz", **output, **string_arrays, X_raw=X,
            requested_d=H.D, cohort_eligible=class_cases.cohort_eligible.to_numpy(bool),
            training_scope="matched_cohort", raw_duration_clipping_applied=False, endpoint_correction_applied=False)
        result = pd.DataFrame(class_statuses)
        for mode in ("fullfit", "logo"):
            table = result[result["mode"].eq(mode)].sort_values("case_index")
            assert table.case_index.tolist() == list(range(n))
            assert np.isnan(output[f"X_rec_{mode}"][excluded]).all()
            assert np.isnan(output[f"Z_{mode}"][excluded]).all()
            ok = table.status.eq("ok").to_numpy()
            assert np.isfinite(output[f"X_rec_{mode}"][ok]).all()
            # Independently decode each retained row from the saved d5 model.
            for model_path, group_rows in table[table.status.eq("ok")].groupby("model_npz"):
                with np.load(PROJECT / model_path, allow_pickle=False) as saved:
                    ids = group_rows.case_index.to_numpy(int)
                    decoded = (saved["mu_weighted"] + (output[f"Z_{mode}"][ids] * saved["singular_values_d"]) @ saved["U_d"].T) / saved["alpha"]
                np.testing.assert_allclose(decoded, output[f"X_rec_{mode}"][ids], rtol=0, atol=1e-12)
        summary = dict(subset=subset, training_scope="matched_cohort", n_original_cases=n,
            n_eligible=EXPECTED[subset], n_excluded=n - EXPECTED[subset],
            n_global_groups_represented_in_class=int(class_cases.group_id.nunique()),
            fullfit_cases_ok=int((result["mode"].eq("fullfit") & result.status.eq("ok")).sum()),
            logo_cases_ok=int((result["mode"].eq("logo") & result.status.eq("ok")).sum()),
            fits_attempted=len(class_fits), fits_ok=sum(f["fit_status"] == "ok" for f in class_fits),
            total_fit_wall_s=sum(f["fit_wall_s"] for f in class_fits),
            total_encode_decode_wall_s=sum(f["encode_decode_wall_s"] for f in class_fits))
        print(json.dumps(summary), flush=True)
        summaries.append(summary)
        statuses.extend(class_statuses)
        fits.extend(class_fits)
    all_cases = pd.DataFrame(statuses)
    n_cases, n_elig = len(cases), sum(EXPECTED.values())      # 513/489 before 2026-09-14, 510/486 after
    assert len(all_cases) == 2 * n_cases and int(all_cases.cohort_eligible.sum()) == 2 * n_elig
    case_path = OUT / "svd_d5_matched_cases.csv"
    score_case_path = OUT / "svd_d5_matched_scoring_cases.csv"
    all_cases.to_csv(case_path, index=False)
    all_cases[all_cases.cohort_eligible].to_csv(score_case_path, index=False)
    pd.DataFrame(fits).to_csv(OUT / "svd_d5_matched_fits.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "svd_d5_matched_summary.csv", index=False)
    manifest = dict(training_scope="matched_cohort", status="fitted_scoring_pending",
        cohort_source=str(ELIGIBILITY), cohort_source_sha256=cohort_hash,
        selection_rule="Only status=eligible in already-selected scripts/21_ours3_population.py output; no score, fidelity, or generation outcome used for selection",
        n_original_cases=n_cases, n_eligible=n_elig, n_excluded=n_cases - n_elig, case_status_rows=2 * n_cases, scoring_rows=2 * n_elig,
        feature_definition=grouping["feature_definition"], model_definition=grouping["model_definition"],
        group_definition=grouping["group_definition"], group_members=group_members, n_global_components=len(group_members),
        grouping_note=f"Keep all original {n_cases}-case graph connections, including bridges through excluded cases; only eligible same-class references train each fit",
        fullfit_note="Each class fits and reconstructs its own eligible references; still in-sample",
        logo_note="Each target connected group is excluded from training; target reference used only for encoding/reconstruction",
        no_operations=["basis or alpha from held-out data", "endpoint constraints", "duration clipping", "KDE", "sampling", "XOSC", "simulation", "metric changes", "PET"],
        array_schema="Original full class row count and case_index retained; excluded rows in X_rec_* and Z_* are NaN. Source X_raw remains available for all original rows.",
        complete_case_status_csv=str(case_path), scoring_case_csv=str(score_case_path),
        scoring_exclusion_note=f"Excluded cohort members are not passed to scorer; complete {2 * n_cases}-row audit distinguishes planned exclusions from model failures",
        classes=summaries, sources=[dict(path=p, sha256=v) for p, v in before_hashes.items()],
        model_generation_elapsed_s=time.perf_counter() - started,
        original_513_models_preserved=True, process_count=1, numerical_threads_requested=1)
    manifest_path = OUT / "svd_d5_matched_manifest.json"
    manifest_write(manifest_path, manifest)
    SCORING_OUT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", str(SCORER), "--cases", str(score_case_path), "--output-dir", str(SCORING_OUT)]
    before_scoring = time.perf_counter()
    with (SCORING_OUT / "scorer_stdout.txt").open("w") as stdout, (SCORING_OUT / "scorer_stderr.txt").open("w") as stderr:
        process = subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    manifest.update(scoring_command=cmd, scoring_returncode=process.returncode,
                    scoring_wall_s=time.perf_counter() - before_scoring)
    if process.returncode != 0:
        manifest.update(status="scoring_failed", original_sources_unchanged=all(digest(p) == v for p, v in before_hashes.items()))
        manifest_write(manifest_path, manifest)
        raise RuntimeError(f"Existing scorer failed with return code {process.returncode}; see {SCORING_OUT}")
    for path, value in before_hashes.items():
        assert digest(path) == value, f"Original source/model changed: {path}"
    score_audit = json.loads((SCORING_OUT / "svd_d5_nonpet_audit.json").read_text())
    assert score_audit["n_cases"] == 2 * n_elig and score_audit["blocked_pet_calls"] == 0
    manifest.update(status="matched_cohort_fit_and_existing_nonpet_scoring_completed",
        original_sources_unchanged=True, scoring_audit=score_audit,
        elapsed_s=time.perf_counter() - started,
        outputs=[dict(path=str(p), sha256=digest(p)) for p in sorted(
            [*MODELS.glob("**/*.npz"), case_path, score_case_path, OUT / "svd_d5_matched_fits.csv",
             OUT / "svd_d5_matched_summary.csv", *SCORING_OUT.glob("*")]) if p.is_file()])
    manifest_write(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "n_eligible", "n_excluded", "scoring_wall_s", "elapsed_s", "original_sources_unchanged")}, indent=2))
    print((SCORING_OUT / "scorer_stdout.txt").read_text())


if __name__ == "__main__":
    main()
