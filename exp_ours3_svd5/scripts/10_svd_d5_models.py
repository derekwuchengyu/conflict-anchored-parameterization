#!/usr/bin/env python3
"""Fit raw SVD d5 for five classes and reconstruct full-fit / group-held-out.

Run using nps Python with -B. Upstream inputs are read-only. The grouping graph
uses shared ego/target identities across all five classes, but each SVD learns
from its own class only. No KDE, clipping, boundary correction, metrics, XOSC,
or simulator execution occurs. The singleton 39_180 diagnostic stays separate.
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
from core.svd_param import SvdParameterization

D, NT, NY, NTH, FPS = 5, 50, 2, 1, 30.0
NX = NT * NY + NTH
DATASET, RECORDING = "HetroD", "00"
CLASS_DIRS = {
    "tlkeep": CORE / "data/tlkeep",
    "keeptl": CORE / "data/keeptl",
    "keeptl_sw": CORE / "data/keeptl_sw",
    "cutinl": ROOT / "exp_cross_coverage/data/cutinl",
    "cutinr": CORE / "data/cutinr",
}
# 2026-09-14: scenarios whose current label in HetroD-labeler/data/00_labeled_scenarios.json is no longer
# "被左側 cut-in" (label 8): 39_180 -> 77, 39_189 -> 29, 1669_1657 -> 16. The upstream real_meta.csv was frozen
# on 2026-08-11, before these relabels, so they are removed here at the cohort root.
EXCLUDED_SCENARIOS = {"cutinl": {"39_180", "39_189", "1669_1657"}}
MODEL_ROOT = PROJECT / "models/svd_d5"
RESULTS = PROJECT / "results"


def source_entry(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path.relative_to(ROOT)), "sha256": h.hexdigest(),
            "size_bytes": path.stat().st_size}


def project_path(path):
    return str(path.relative_to(PROJECT))


def load_class(subset, directory):
    meta = pd.read_csv(directory / "real_meta.csv")
    drop = EXCLUDED_SCENARIOS.get(subset, set())
    if drop:
        missing = drop - set(meta.scenario_id.astype(str))
        if missing:
            raise ValueError(f"{subset}: excluded scenarios not in real_meta.csv: {sorted(missing)}")
        meta = meta[~meta.scenario_id.astype(str).isin(drop)].reset_index(drop=True)
    tracks = pd.read_parquet(directory / "real_tracks.parquet")
    if not meta.scenario_id.is_unique:
        raise ValueError(f"{subset}: scenario_id is not unique")
    sources = [source_entry(directory / name)
               for name in ("real_meta.csv", "real_tracks.parquet")]
    existing = None
    source_vector_path = directory / "real_vectors.npz"
    if source_vector_path.exists():
        sources.append(source_entry(source_vector_path))
        # The checked-in upstream NPZ may have object-string keys. It is a
        # trusted local input; every newly written NPZ uses Unicode arrays.
        source_vectors = np.load(source_vector_path, allow_pickle=True)
        source_keys = [str(k) for k in source_vectors["keys"]]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError(f"{subset}: duplicate source vector keys")
        existing = dict(zip(source_keys, source_vectors["X_raw"]))

    X, rows, max_input_difference = np.full((len(meta), NX), np.nan), [], 0.0
    actor_groups = {str(key): g.sort_values("frame")
                    for key, g in tracks[tracks.role == "actor"].groupby("scenario_id")}
    for i, row in enumerate(meta.itertuples(index=False)):
        sid = str(row.scenario_id)
        item = {"subset": subset, "case_index": i, "scenario_id": sid,
                "scenario_uid": f"{DATASET}/{RECORDING}/{sid}/{int(row.min_frame)}-{int(row.max_frame)}",
                "dataset": DATASET, "recording": RECORDING,
                "ego": int(row.ego), "actor": int(row.actor), "label": int(row.label),
                "metadata_min_frame": int(row.min_frame), "metadata_max_frame": int(row.max_frame),
                "source_meta": str(directory / "real_meta.csv"),
                "source_tracks": str(directory / "real_tracks.parquet"),
                "feature_source": "upstream_real_vectors_npz" if existing is not None else "actor_track_resampling",
                "input_status": "ok", "input_detail": ""}
        try:
            actor = actor_groups[sid]
            if len(actor) < 10 or not actor.frame.is_unique:
                raise ValueError("Insufficient actor observations or duplicate frame")
            if not actor.track_id.eq(int(row.actor)).all():
                raise ValueError("Actor track_id does not match metadata")
            t = (actor.frame.to_numpy(float) - actor.frame.iloc[0]) / FPS
            if t[-1] <= 0.5 or not np.all(np.diff(t) > 0):
                raise ValueError("Non-positive/short duration or non-increasing timestamps")
            q = np.linspace(0, t[-1], NT)
            xy = np.column_stack([np.interp(q, t, actor.x), np.interp(q, t, actor.y)])
            built = np.r_[xy.ravel(), t[-1]]
            vector = np.asarray(existing[sid] if existing is not None else built, dtype=float)
            if vector.shape != (NX,) or not np.isfinite(vector).all():
                raise ValueError("Invalid raw feature vector")
            difference = float(np.max(abs(vector - built)))
            if difference > 1e-8:
                raise ValueError(f"Upstream vector differs from documented track resampling by {difference}")
            max_input_difference = max(max_input_difference, difference)
            X[i] = vector
            item.update(actual_actor_min_frame=int(actor.frame.min()),
                        actual_actor_max_frame=int(actor.frame.max()),
                        actor_points=len(actor), raw_duration_s=float(vector[-1]))
        except (KeyError, ValueError, TypeError) as exc:
            item.update(input_status="invalid_source", input_detail=f"{type(exc).__name__}: {exc}")
        rows.append(item)
    return {"X": X, "cases": pd.DataFrame(rows), "sources": sources,
            "upstream_vector_resampling_max_abs_difference": max_input_difference}


def assign_global_groups(populations):
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    def key(tid):
        return f"{DATASET}/{RECORDING}/{int(tid)}"

    for pop in populations.values():
        for row in pop["cases"].itertuples(index=False):
            parents[find(key(row.ego))] = find(key(row.actor))
    members = {}
    for actor in list(parents):
        members.setdefault(find(actor), set()).add(actor)
    ids = {root: "g_" + hashlib.sha256("|".join(sorted(actors)).encode()).hexdigest()[:16]
           for root, actors in members.items()}
    group_members = {ids[root]: sorted(actors) for root, actors in members.items()}
    for pop in populations.values():
        cases = pop["cases"]
        cases["group_id"] = [ids[find(key(actor))] for actor in cases.actor]
        cases["group_actor_count"] = cases.group_id.map(lambda group: len(group_members[group]))
    # Verify that no track ever belongs to more than one connected component.
    seen = {}
    for pop in populations.values():
        for row in pop["cases"].itertuples(index=False):
            for actor in (row.ego, row.actor):
                previous = seen.setdefault(key(actor), row.group_id)
                assert previous == row.group_id
    return group_members


def fit_and_decode(subset, pop, mode, group_id, model_dir, outputs):
    X, cases = pop["X"], pop["cases"]
    eligible = cases.input_status.eq("ok").to_numpy()
    evaluate = np.ones(len(cases), bool) if mode == "fullfit" else cases.group_id.eq(group_id).to_numpy()
    train = eligible.copy() if mode == "fullfit" else eligible & ~evaluate
    test = eligible & evaluate
    model_id = f"{subset}__{mode}" + (f"__{group_id}" if group_id else "")
    destination = model_dir / f"{model_id}.npz"
    record = {"subset": subset, "mode": mode, "group_id_excluded": group_id or "",
              "model_id": model_id, "requested_d": D, "n_train": int(train.sum()),
              "n_evaluate": int(evaluate.sum()), "n_valid_evaluate": int(test.sum()),
              "fit_status": "ok", "detail": "", "fit_wall_s": 0.0,
              "encode_decode_wall_s": 0.0, "numerical_rank": None, "model_npz": ""}
    P, train_rank = None, None
    if train.sum() < D + 1:
        record.update(fit_status="insufficient_training_count",
                      detail=f"Need at least {D + 1} centered observations for d={D}")
    else:
        try:
            before = time.perf_counter()
            P = SvdParameterization(NT, NY, NTH).fit(X[train])
            record["fit_wall_s"] = time.perf_counter() - before
            tolerance = np.finfo(float).eps * max(NX, int(train.sum())) * P.s[0]
            train_rank = int(np.count_nonzero(P.s > tolerance))
            record["numerical_rank"] = train_rank
            if train_rank < D:
                record.update(fit_status="insufficient_numeric_rank",
                              detail=f"Centered numerical rank {train_rank} < requested d={D}")
            else:
                train_keys = cases.loc[train, "scenario_id"].to_numpy(dtype=str)
                train_uids = cases.loc[train, "scenario_uid"].to_numpy(dtype=str)
                evaluate_uids = cases.loc[evaluate, "scenario_uid"].to_numpy(dtype=str)
                np.savez_compressed(
                    destination, requested_d=D, nt=NT, ny=NY, ntheta=NTH, fps=FPS,
                    alpha=P.alpha, mu_weighted=P.mu, U_d=P.U[:, :D],
                    singular_values_d=P.s[:D], V_train_d=P.reduced(D),
                    training_case_indices=np.flatnonzero(train), training_keys=train_keys,
                    training_uids=train_uids, evaluation_uids=evaluate_uids,
                    excluded_group=np.array(group_id or ""), numerical_rank=train_rank,
                )
                record["model_npz"] = project_path(destination)
                before = time.perf_counter()
                Z = ((X[test] * P.alpha - P.mu) @ P.U[:, :D]) / P.s[:D]
                reconstructed = P.reconstruct(Z, D)
                if not np.isfinite(reconstructed).all():
                    raise ValueError("Decode produced non-finite output")
                outputs[f"X_rec_{mode}"][test] = reconstructed
                outputs[f"Z_{mode}"][test] = Z
                record["encode_decode_wall_s"] = time.perf_counter() - before
                if mode == "fullfit":
                    np.testing.assert_allclose(Z, P.reduced(D), rtol=1e-8, atol=1e-10)
                else:
                    assert not set(cases.loc[train, "group_id"]) & {group_id}
        except (ValueError, np.linalg.LinAlgError, AssertionError) as exc:
            record.update(fit_status="fit_or_decode_failure", detail=f"{type(exc).__name__}: {exc}")

    status_rows = []
    for i in np.flatnonzero(evaluate):
        info = cases.iloc[i].to_dict()
        status = "invalid_source" if not eligible[i] else record["fit_status"]
        info.update(mode=mode, model_id=model_id, status=status,
                    detail=info["input_detail"] if not eligible[i] else record["detail"],
                    requested_d=D, numerical_rank=train_rank, n_train=int(train.sum()),
                    excluded_component_size_in_class=int(evaluate.sum()) if mode == "logo" else 0,
                    model_npz=record["model_npz"],
                    raw_features_npz=project_path(model_dir / "raw_features.npz"),
                    reconstruction_npz=project_path(model_dir / "reconstructions.npz"),
                    reconstruction_array_key=f"X_rec_{mode}", coefficient_array_key=f"Z_{mode}",
                    reconstructed_duration_s=float(outputs[f"X_rec_{mode}"][i, -1]) if status == "ok" else None,
                    singleton_claim_eligible=False)
        status_rows.append(info)
    return record, status_rows


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    populations = {name: load_class(name, directory) for name, directory in CLASS_DIRS.items()}
    groups = assign_global_groups(populations)
    all_fits, all_statuses, summaries = [], [], []
    for subset, pop in populations.items():
        X, cases = pop["X"], pop["cases"]
        n = len(cases)
        model_dir = MODEL_ROOT / subset
        model_dir.mkdir(parents=True, exist_ok=True)
        unicode_columns = {key: cases[key].to_numpy(dtype=str)
                           for key in ("scenario_id", "scenario_uid", "group_id", "input_status")}
        np.savez_compressed(model_dir / "raw_features.npz", X_raw=X, **unicode_columns,
                            nt=NT, ny=NY, ntheta=NTH, fps=FPS)
        outputs = {f"X_rec_{mode}": np.full((n, NX), np.nan) for mode in ("fullfit", "logo")}
        outputs.update({f"Z_{mode}": np.full((n, D), np.nan) for mode in ("fullfit", "logo")})
        class_fits, class_statuses = [], []
        jobs = [("fullfit", None)] + [("logo", group) for group in sorted(cases.group_id.unique())]
        for mode, group in jobs:
            fit, statuses = fit_and_decode(subset, pop, mode, group, model_dir, outputs)
            class_fits.append(fit)
            class_statuses.extend(statuses)
        np.savez_compressed(model_dir / "reconstructions.npz", **outputs, **unicode_columns,
                            requested_d=D, X_raw=X, raw_duration_clipping_applied=False)
        status_frame = pd.DataFrame(class_statuses)
        for mode in ("fullfit", "logo"):
            rows = status_frame[status_frame["mode"].eq(mode)].sort_values("case_index")
            assert rows.case_index.to_list() == list(range(n))
            ok = rows.status.eq("ok").to_numpy()
            assert np.isfinite(outputs[f"X_rec_{mode}"][ok]).all()
            if not ok.all():
                assert np.isnan(outputs[f"X_rec_{mode}"][~ok]).all()
        all_fits.extend(class_fits)
        all_statuses.extend(class_statuses)
        summaries.append({"subset": subset, "n_cases": n,
                          "n_source_valid": int(cases.input_status.eq("ok").sum()),
                          "n_connected_groups_in_class": cases.group_id.nunique(),
                          "fullfit_cases_ok": int((status_frame["mode"].eq("fullfit") & status_frame.status.eq("ok")).sum()),
                          "logo_cases_ok": int((status_frame["mode"].eq("logo") & status_frame.status.eq("ok")).sum()),
                          "fits_attempted": len(class_fits),
                          "fits_ok": sum(r["fit_status"] == "ok" for r in class_fits),
                          "total_fit_wall_s": sum(r["fit_wall_s"] for r in class_fits),
                          "upstream_vector_resampling_max_abs_difference": pop["upstream_vector_resampling_max_abs_difference"]})
        print(f"{subset}: cases={n}, groups={cases.group_id.nunique()}, "
              f"LOGO ok={summaries[-1]['logo_cases_ok']}/{n}", flush=True)

    pd.DataFrame(all_statuses).to_csv(RESULTS / "svd_d5_cases.csv", index=False)
    pd.DataFrame(all_fits).to_csv(RESULTS / "svd_d5_fits.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULTS / "svd_d5_summary.csv", index=False)
    source_paths = [source_entry(CORE / "core/svd_param.py"), source_entry(Path(__file__).resolve())]
    for pop in populations.values():
        source_paths.extend(pop["sources"])
    for source in source_paths:
        assert source_entry(ROOT / source["path"])["sha256"] == source["sha256"], "Upstream input changed during run"
    manifest = {
        "status": "RAW_SVD_D5_MODELS_AND_DETERMINISTIC_RECONSTRUCTIONS_ONLY",
        "feature_definition": {"nt": NT, "ny": NY, "ntheta": NTH, "fps": FPS, "nx": NX,
                               "layout": "50 xy pairs in time-major order, then actor duration seconds",
                               "resampling": "uniform normalized actor time from first to last actual actor observation",
                               "duration_clipping": False, "endpoint_correction": False},
        "model_definition": {"d": D, "implementation": "sr-tlkeep-experiment/core/svd_param.py",
                             "centering_scaling": "Each training population fits its own beta/std and weighted mean",
                             "minimum_training_count": D + 1, "minimum_numeric_rank": D,
                             "fit_modes": ["fullfit", "logo"]},
        "group_definition": "Connected components of shared ego OR target track identities across all five classes, scoped by HetroD/00; each LOGO fit trains only on same-class cases outside the target component",
        "group_members": groups, "n_global_components": len(groups),
        "group_limitations": ["No background-track connections", "No time-overlap exclusion beyond shared ego/target connectivity", "Single recording only; not independent-recording evaluation", "Full-fit is in-sample; LOGO encodes the held-out reference but does not fit on it"],
        "singleton_separation": "39_180 remains an original CUTIN_L class observation here. These class-population results are not a single-scenario training experiment and must not be used as singleton or rare-generation evidence. See separate singleton_diagnostic for N=1.",
        "no_operations": ["KDE fit or sampling", "added noise or priors", "clipping generated duration", "boundary fixes", "metrics", "XOSC export", "simulation"],
        "case_status_schema": "svd_d5_cases.csv: one row per subset/case_index/mode; join source on scenario_uid. Array row is case_index in reconstruction_npz. status==ok indicates finite saved output; failures retain NaN and explicit status.",
        "model_npz_schema": "alpha[101], mu_weighted[101], U_d[101,5], singular_values_d[5], V_train_d[n_train,5], training_case_indices, training_keys, training_uids, evaluation_uids",
        "reconstruction_npz_schema": "scenario_id, scenario_uid, group_id, input_status; X_raw[N,101]; X_rec_fullfit and X_rec_logo[N,101]; Z_fullfit and Z_logo[N,5]",
        "decode_equation": "X_raw = (mu_weighted + (Z * singular_values_d) @ U_d.T) / alpha",
        "sources": source_paths, "classes": summaries,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "processes": 1, "numerical_threads_requested": 1,
                        "dont_write_bytecode": sys.dont_write_bytecode},
    }
    (RESULTS / "svd_d5_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                          indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
