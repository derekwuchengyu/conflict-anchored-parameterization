#!/usr/bin/env python3
"""[E2 / Task C] Matched-cohort variant of 11_svd_d5_kde.py: same pipeline on models/svd_d5_matched
(489-scene cohort fullfit models), plus applied_duration_s = max(raw, 0.5) and duration_clipped flags
(HANDOFF_BASELINE.md matched-KDE spec items 1-6). Saved-model decode is checked against
reconstructions.npz before sampling. Derived from 11 (diff in results/svd_d5_kde_matched_diff_vs_11.txt).

Generate raw, unconditioned class SVD d5 + KDE samples numerically.

Uses the saved full-fit models; no new SVD fit, boundary changes, clipping,
XOSC, simulator, or downstream metrics. This is class/full-fit distribution
exploration, never held-out evaluation or the singleton 39_180 experiment.
Run with nps Python -B. Existing XOSC and upstream artifacts are never touched.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
CORE = ROOT / "sr-tlkeep-experiment"
sys.path.insert(0, str(CORE))
from core.kde_sampling import loo_bandwidth, sample_dependent
from core.svd_param import SvdParameterization

D, NT, NY, NTH = 5, 50, 2, 1
SEEDS = (20260910, 20260911, 20260912)
N_DRAWS, LOO_GRID = 1000, 60
CLASSES = ("tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr")
RESULTS = PROJECT / "results"
GENERATED = PROJECT / "generated/svd_d5_kde_matched"
DURATION_FLOOR_S = 0.5  # sr-tlkeep-experiment/scripts/04_generate_svd.py:52 rule, recorded not applied to X


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_source(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def rel(path):
    return str(path.relative_to(PROJECT))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    upstream_manifest_path = RESULTS / "svd_d5_matched_manifest.json"
    upstream = json.loads(upstream_manifest_path.read_text())
    assert upstream["model_definition"]["d"] == D
    # Refuse to silently use models after their recorded source inputs changed.
    for source in upstream["sources"]:
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError(f"SVD source changed: {source['path']}")
    sources = [artifact_source(Path(__file__).resolve()),
               artifact_source(CORE / "core/svd_param.py"),
               artifact_source(CORE / "core/kde_sampling.py"),
               artifact_source(upstream_manifest_path)]
    attempt_frames, bandwidth_rows, timing_rows, summaries, output_files = [], [], [], [], []
    for subset in CLASSES:
        model_path = PROJECT / f"models/svd_d5_matched/{subset}/{subset}__fullfit.npz"
        model_source = artifact_source(model_path)
        sources.append(model_source)
        saved = np.load(model_path, allow_pickle=False)
        assert int(saved["requested_d"]) == D
        V = np.asarray(saved["V_train_d"], float)
        assert V.shape[1] == D and V.shape[0] >= D + 1
        assert saved["U_d"].shape == (NT * NY + NTH, D)
        assert np.isfinite(V).all() and np.all(saved["singular_values_d"] > 0)
        P = SvdParameterization(NT, NY, NTH)
        P.alpha, P.mu = saved["alpha"], saved["mu_weighted"]
        P.U, P.s = saved["U_d"], saved["singular_values_d"]
        train_keys = saved["training_keys"].astype(str)
        train_uids = saved["training_uids"].astype(str)
        train_case_indices = saved["training_case_indices"].astype(int)
        assert len(train_keys) == len(train_uids) == len(train_case_indices) == len(V)
        # Spec item 2/5: saved-model decode must reproduce the stored matched reconstructions.
        recon = np.load(PROJECT / f"models/svd_d5_matched/{subset}/reconstructions.npz", allow_pickle=False)
        redecoded = P.reconstruct(V, D)
        stored = np.asarray(recon["X_rec_fullfit"], float)[train_case_indices]
        assert np.all(np.abs(redecoded - stored) < 1e-12), f"{subset}: saved-model decode mismatch"
        before = time.perf_counter()
        h = float(loo_bandwidth(V, n_grid=LOO_GRID))
        bandwidth_wall = time.perf_counter() - before
        if not np.isfinite(h) or h <= 0:
            raise ValueError(f"{subset}: non-positive or non-finite LOO bandwidth")
        bandwidth_rows.append({"subset": subset, "d": D, "fit_scope": "fullfit_matched_cohort",
                               "n_train": len(V), "h_loo": h, "grid_points": LOO_GRID,
                               "bandwidth_wall_s": bandwidth_wall,
                               "model_npz": rel(model_path), "model_sha256": model_source["sha256"]})

        subset_dir = GENERATED / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_valid, subset_nonpositive, subset_nonfinite = 0, 0, 0
        for batch_index, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed)
            before = time.perf_counter()
            Z, center_idx = sample_dependent(V, h, N_DRAWS, rng)
            sample_wall = time.perf_counter() - before
            before = time.perf_counter()
            X = P.reconstruct(Z, D)
            decode_wall = time.perf_counter() - before
            assert Z.shape == (N_DRAWS, D) and X.shape == (N_DRAWS, NT * NY + NTH)
            finite = np.isfinite(Z).all(axis=1) & np.isfinite(X).all(axis=1)
            raw_duration = X[:, -1].copy()
            applied_duration = np.maximum(raw_duration, DURATION_FLOOR_S)
            duration_clipped = raw_duration < DURATION_FLOOR_S
            positive_duration = raw_duration > 0
            numeric_ok = finite & positive_duration
            status = np.where(~finite, "non_finite", np.where(~positive_duration, "non_positive_duration", "ok"))
            attempt_ids = np.array([f"svd5kde_{subset}_s{seed}_{i:06d}" for i in range(N_DRAWS)])
            output = subset_dir / f"seed_{seed}.npz"
            np.savez_compressed(
                output, X=X, Z=Z, center_idx=center_idx,
                center_case_index=train_case_indices[center_idx],
                center_keys=train_keys[center_idx], center_scenario_uid=train_uids[center_idx],
                attempt_id=attempt_ids, raw_duration_s=raw_duration,
                applied_duration_s=applied_duration, duration_clipped=duration_clipped,
                numeric_ok=numeric_ok, numeric_status=status,
                h_loo=h, d=D, nt=NT, ny=NY, ntheta=NTH, seed=seed,
                model_sha256=np.array(model_source["sha256"]),
                sampling_design=np.array("unconditional_class_fullfit_dependent_kde"),
                duration_clipping_applied=False, endpoint_correction_applied=False,
            )
            attempt_frames.append(pd.DataFrame({
                "subset": subset, "method": "svd_d5_kde_matched", "fit_scope": "fullfit_matched_cohort",
                "sampling_design": "unconditional_class_dependent_kde",
                "attempt_id": attempt_ids, "seed": seed, "array_row": np.arange(N_DRAWS),
                "d": D, "center_idx": center_idx,
                "center_case_index": train_case_indices[center_idx],
                "center_scenario_id": train_keys[center_idx],
                "center_scenario_uid": train_uids[center_idx],
                "h_loo": h, "raw_duration_s": raw_duration,
                "applied_duration_s": applied_duration, "duration_clipped": duration_clipped,
                "numeric_ok": numeric_ok, "numeric_status": status,
                "generated_npz": rel(output), "vector_array_key": "X", "latent_array_key": "Z",
                "model_npz": rel(model_path), "model_sha256": model_source["sha256"],
                "duration_clipping_applied": False, "applied_duration_rule": "max(raw_duration_s, 0.5) recorded for decoding only", "endpoint_correction_applied": False,
                "executed": False, "singleton_claim_eligible": False,
            }))
            timing_rows.append({"subset": subset, "seed": seed, "batch_index": batch_index,
                                "is_first_measured_batch_in_class": batch_index == 0,
                                "n_attempts": N_DRAWS, "sample_wall_s": sample_wall,
                                "decode_wall_s": decode_wall,
                                "sample_ms_per_attempt": sample_wall * 1000 / N_DRAWS,
                                "decode_ms_per_attempt": decode_wall * 1000 / N_DRAWS,
                                "sampling_design": "unconditional_class_fullfit_dependent_kde"})
            subset_valid += int(numeric_ok.sum())
            subset_nonpositive += int((finite & ~positive_duration).sum())
            subset_nonfinite += int((~finite).sum())
            output_files.append({"path": rel(output), "sha256": digest(output), "n_attempts": N_DRAWS})
        summaries.append({"subset": subset, "d": D, "n_train": len(V), "n_seeds": len(SEEDS),
                          "n_attempts": N_DRAWS * len(SEEDS), "h_loo": h,
                          "numeric_ok": subset_valid, "non_positive_duration": subset_nonpositive,
                          "non_finite": subset_nonfinite})
        print(f"{subset}: d5 h={h:.8f}, attempts={N_DRAWS * len(SEEDS)}, "
              f"numeric-ok={subset_valid}, duration<=0={subset_nonpositive}", flush=True)

    attempts = pd.concat(attempt_frames, ignore_index=True)
    assert len(attempts) == len(CLASSES) * len(SEEDS) * N_DRAWS
    assert attempts.attempt_id.is_unique
    attempts.to_csv(RESULTS / "svd_d5_kde_matched_attempts.csv", index=False)
    pd.DataFrame(bandwidth_rows).to_csv(RESULTS / "svd_d5_kde_matched_bandwidth.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(RESULTS / "svd_d5_kde_matched_timing.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULTS / "svd_d5_kde_matched_summary.csv", index=False)
    for source in sources:
        assert digest(ROOT / source["path"]) == source["sha256"], "Source/model changed during generation"
    manifest = {
        "status": "NUMERIC_MATCHED_COHORT_FULLFIT_D5_KDE_GENERATION_COMPLETE_NO_SIMULATION",
        "cohort": "489-scene matched cohort (models/svd_d5_matched); class N 294/50/77/48/20",
        "applied_duration_rule": "applied_duration_s = max(raw_duration_s, 0.5) and duration_clipped recorded per attempt; X keeps raw values",
        "prefix_rule": "budget comparisons must use the first rows of these 1000-row draws; no independent 100-row redraw",
        "d": D, "nt": NT, "ny": NY, "ntheta": NTH, "fps": 30.0,
        "classes": list(CLASSES), "seeds": list(SEEDS), "draws_per_class_per_seed": N_DRAWS,
        "total_attempts": len(attempts), "training_scope": "Saved matched-cohort class full-fit SVD models (489 scenes), not held-out",
        "sampling": "Existing sample_dependent: uniform training centre, then V[centre] + h*N(0,I5)",
        "bandwidth": "Fresh per-class d5 LOO-CV using existing core, 60-point grid and 40 golden-section iterations; no d3 bandwidth reuse",
        "boundary_policy": "Raw decoded xy and duration; no endpoint fixing, duration clipping, filtering, resampling, or rejection",
        "numeric_status_definition": "ok only means all latent/output entries finite and raw duration > 0. This is not physical, road, background-collision or simulator validity.",
        "timing_scope": "Bandwidth fit measured once per class; sampling and decoding measured separately for each of the three actual seed batches. Batch 0 is first measured, not a claimed process-cold benchmark. No extra benchmark samples. NPZ/CSV I/O, source validation, export, simulation and metrics excluded.",
        "singleton_separation": "A 39_180 kernel centre, if drawn from CUTIN_L, belongs to the full 61-observation class model. It is not the N=1 singleton experiment and provides no singleton or rare-generation evidence.",
        "array_schema": "X[1000,101] raw decoded time-major xy plus duration; Z[1000,5]; center_idx indexes model V_train_d; center_case_index indexes class raw_features; center_keys/center_scenario_uid preserve source identity; attempt_id and numeric_status align by row",
        "attempt_csv_schema": "Each row is one attempted numeric sample. generated_npz and model_npz are relative to project. array_row addresses X/Z; center_scenario_uid joins source inventory. No attempt is dropped.",
        "sources": sources, "source_model_manifest_sha256": digest(upstream_manifest_path),
        "outputs": output_files, "summary": summaries,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "processes": 1, "numerical_threads_requested": 1, "dont_write_bytecode": sys.dont_write_bytecode},
    }
    (RESULTS / "svd_d5_kde_matched_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                              indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
