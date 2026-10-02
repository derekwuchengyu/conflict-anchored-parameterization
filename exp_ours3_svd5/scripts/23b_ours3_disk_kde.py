#!/usr/bin/env python3
"""[E1 on the Euclidean-disk geometry] Variant of 23_ours3_kde.py that fits/samples the 3-D KDE on
results/ours3_disk_contexts.json (469 scenes, DECISIONS_20260910_E0.md item 7) and plans/executes under
plans/ours3_disk_kde_dependent and runs/ours3_disk_kde_*. Derived from 23 (diff in
results/ours3_disk_kde_diff_vs_23.txt); sampler, bandwidth and clipping rule unchanged.

Plan and optionally execute Ours3 dependent-KDE draws with matched budgets.

Uses the unchanged upstream ParamKDE and LOO bandwidth functions on precisely
three columns. Fixed geometry and XOSC context come from each sampled kernel
center. EndSpeed max(0, requested) follows the existing renderer and is recorded
explicitly; theta is never clipped, wrapped, rejected, or resampled.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
CONTEXTS = PROJECT / "results/ours3_disk_contexts.json"
GEOMETRY_SOURCE_DEFAULT = str(PROJECT / "results/disk_window_L10_points.csv")
RUNNER = PROJECT / "scripts/20_ours3_generate.py"
CV_SOURCE = ROOT / "exp_coverage_velocity/scripts/cvlib.py"
KDE_SOURCE = ROOT / "sr-tlkeep-experiment/core/kde_sampling.py"
CLIP_SOURCE = ROOT / "exp_coverage_velocity/scripts/cvrender.py"
CLASSES = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr"]
COLS = ["theta1_deg", "theta2_deg", "interaction_end_speed_kmh"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_kde():
    namespace = {"np": np}
    for path, names in ((KDE_SOURCE, {"loo_log_likelihood", "loo_bandwidth"}),
                        (CV_SOURCE, {"ParamKDE"})):
        parsed = ast.parse(path.read_text())
        nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in names]
        assert {node.name for node in nodes} == names
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace["ParamKDE"]


def make_job(row, context, plan_path, source_paths):
    applied = dict(zip(COLS, row["applied_params"]))
    sampling = dict(method="ours3_kde_dependent", seed=row["seed"], draw=row["draw"],
        planned_pool_size=row["planned_pool_size"], kernel_center_index=row["center_index"],
        kernel_center_scenario_id=context["scenario_id"], kernel_center_scenario_uid=context["scenario_uid"],
        h=row["h"], sampler_parameters_requested=dict(zip(COLS, row["requested_params"])),
        sampler_parameters_applied=applied, end_speed_clipped=row["end_speed_clipped"],
        end_speed_clip_rule="max(0.0, requested_end_speed_kmh)", theta_clipped=False,
        theta_wrapped=False, rejected_or_resampled=False, sampling_plan=str(plan_path),
        fit_scope="all available Euclidean-disk (L=10 m) contexts within each class; exploratory in-sample fit",
        geometry_source=context.get("geometry_source", GEOMETRY_SOURCE_DEFAULT),
        center_parameters=dict(zip(COLS, context["params"])))
    return dict(job_id=row["job_id"], scenario_id=context["scenario_id"],
        **{"class": context["subset"]}, dataset="HetroD", recording="00", ego=context["ego"],
        target=context["actor"], min_frame=context["min_frame"], max_frame=context["max_frame"],
        source_xosc=context["base"], cps_xy=np.asarray(context["cps6"]).reshape(3, 2).tolist(),
        anchor_frame=context["anchor_frame"], geometry_variant_id=context["geometry_variant"],
        parameters=applied, expected_default_end_speed_kmh=context["end_speed_kmh"],
        source_context=sampling,
        source_files=[str(p) for p in source_paths] + [context.get("geometry_source", GEOMETRY_SOURCE_DEFAULT)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260910])
    parser.add_argument("--budget", type=int, default=100,
                        help="Last 1-based draw in the fixed pool to execute")
    parser.add_argument("--start-draw", type=int, default=1,
                        help="First 1-based draw; use 101 to extend a completed 100-draw run")
    parser.add_argument("--pool-size", type=int, default=1000,
                        help="Draw the complete parameter pool once so cumulative budgets use stable prefixes")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--contexts", type=Path, default=CONTEXTS)
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=CLASSES)
    parser.add_argument("--batch-prefix", default="ours3_disk_kde")
    args = parser.parse_args()
    assert 1 <= args.start_draw <= args.budget <= args.pool_size
    assert len(set(args.seeds)) == len(args.seeds)
    contexts_path = args.contexts.resolve()
    source_paths = [contexts_path, Path(__file__).resolve(), CV_SOURCE, KDE_SOURCE, CLIP_SOURCE, RUNNER]
    sources = [{"path": str(p), "sha256": digest(p)} for p in source_paths]
    source_key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    plan_root = PROJECT / "plans/ours3_disk_kde_dependent" / source_key
    plan_root.mkdir(parents=True, exist_ok=True)
    source_snapshot = plan_root / "planner_snapshot.py"
    if not source_snapshot.exists():
        source_snapshot.write_text(Path(__file__).read_text())
    contexts = json.loads(contexts_path.read_text())
    by_class = {name: [context for context in contexts if context["subset"] == name] for name in args.classes}
    assert sum(map(len, by_class.values())) == len(contexts)      # 469 before 2026-09-14 (per-class L + cutinl exclusion)
    assert all(len(group) >= 2 for group in by_class.values())
    ParamKDE = load_kde()
    models, fits = {}, {}
    fit_path = plan_root / "fit.json"
    saved_fit = json.loads(fit_path.read_text()) if fit_path.exists() else None
    if saved_fit is not None:
        assert saved_fit["sources"] == sources
    for name in args.classes:
        group = by_class[name]
        raw = np.asarray([context["params"] for context in group], float)
        assert raw.shape == (len(group), 3) and np.isfinite(raw).all()
        fit_npz = plan_root / f"fit_{name}.npz"
        if saved_fit is not None:
            # Restore the exact earlier fitted model; do not replace its fit
            # timing/provenance when later executing another budget prefix.
            arrays = np.load(fit_npz)
            model = ParamKDE.__new__(ParamKDE)
            model.raw, model.cols = arrays["raw"], COLS
            model.m, model.sd, model.Z = arrays["mean"], arrays["sd"], arrays["standardized"]
            model.h = float(arrays["h"])
            model.N, model.p = model.Z.shape
            assert np.array_equal(model.raw, raw)
            elapsed = saved_fit["fits"][name]["fit_s"]
        else:
            start = time.perf_counter()
            model = ParamKDE(raw, COLS)
            elapsed = time.perf_counter() - start
        models[name] = model
        fits[name] = dict(n_fit=len(group), dimensions=3, columns=COLS, mean=model.m.tolist(),
            sd=model.sd.tolist(), h=float(model.h), fit_s=elapsed,
            center_scenario_ids=[context["scenario_id"] for context in group],
            center_scenario_uids=[context["scenario_uid"] for context in group],
            standardization="per-column mean and population std; near-zero std replaced with 1",
            bandwidth="unchanged upstream LOO log-likelihood grid + golden-section refinement",
            excluded_for_offset=[], held_out_evaluation=False)
        if saved_fit is None:
            np.savez_compressed(fit_npz, raw=model.raw, mean=model.m, sd=model.sd,
                                standardized=model.Z, h=model.h)
        print(f"[ours3 KDE fit] {name}: N={len(group)}, p=3, h={model.h:.9g}, fit={elapsed:.3f}s", flush=True)
    if saved_fit is None:
        write_json(fit_path, dict(fits=fits, sources=sources,
            variable_columns=COLS, fixed_context="kernel center's q_minus, incoming direction, L1, L2, base and fixed declarations",
            old_offset_fit_exclusions_applied=False, fit_scope="in-sample exploratory fit; no new train/test split"))
    jobs, pool_refs, execution_plan_rows = [], [], []
    for seed in args.seeds:
        for name in args.classes:
            pool_path = plan_root / f"pool_{name}_seed{seed}_n{args.pool_size}.parquet"
            timing_path = pool_path.with_suffix(".json")
            model, group = models[name], by_class[name]
            if pool_path.exists() and timing_path.exists():
                table = pd.read_parquet(pool_path)
                timing = json.loads(timing_path.read_text())
                assert len(table) == args.pool_size
                print(f"[ours3 KDE plan] reusing {pool_path.name}", flush=True)
            else:
                rng = np.random.default_rng(seed)
                start = time.perf_counter()
                requested, centers = model.dependent(args.pool_size, rng)
                sampling_s = time.perf_counter() - start
                applied = requested.copy()
                applied[:, 2] = np.maximum(0.0, applied[:, 2])
                rows = []
                for k, (p, q, index) in enumerate(zip(requested, applied, centers), 1):
                    center = group[int(index)]
                    rows.append(dict(job_id=f"{name}__seed{seed}__draw{k:06d}", subset=name,
                        seed=seed, draw=k, planned_pool_size=args.pool_size, center_index=int(index),
                        center_scenario_id=center["scenario_id"], center_scenario_uid=center["scenario_uid"],
                        requested_params=p.tolist(), applied_params=q.tolist(),
                        requested_theta1_deg=float(p[0]), requested_theta2_deg=float(p[1]),
                        requested_interaction_end_speed_kmh=float(p[2]),
                        applied_theta1_deg=float(q[0]), applied_theta2_deg=float(q[1]),
                        applied_interaction_end_speed_kmh=float(q[2]),
                        end_speed_clipped=bool(p[2] < 0), theta_clipped=False,
                        rejected_or_resampled=False, h=float(model.h),
                        source_xosc=center["base"], source_xosc_sha256=center["base_sha256"],
                        geometry_source=center.get("geometry_source", GEOMETRY_SOURCE_DEFAULT), geometry_variant=center["geometry_variant"]))
                table = pd.DataFrame(rows)
                table.to_parquet(pool_path, index=False)
                table.to_csv(pool_path.with_suffix(".csv"), index=False)
                timing = dict(subset=name, seed=seed, n_generated_parameter_vectors=args.pool_size,
                    n_rejected=0, n_resampled=0, n_end_speed_clipped=int(table.end_speed_clipped.sum()),
                    sampling_s=sampling_s, sampling_s_per_parameter_vector=sampling_s / args.pool_size,
                    pool_sha256=digest(pool_path), rng="numpy.random.default_rng(seed), restarted per class",
                    vectorized_draw_order="upstream ParamKDE.dependent(M): all uniform centers then all Gaussian noises",
                    fit_path=str(fit_path), fit_h=float(model.h))
                write_json(timing_path, timing)
            pool_refs.append(dict(path=str(pool_path), metadata=str(timing_path), **timing))
            selected = table[table.draw.between(args.start_draw, args.budget)]
            assert len(selected) == args.budget - args.start_draw + 1
            for row in selected.to_dict("records"):
                row["requested_params"] = list(row["requested_params"])
                row["applied_params"] = list(row["applied_params"])
                center = group[int(row["center_index"])]
                job = make_job(row, center, pool_path, source_paths + [fit_path, pool_path, timing_path])
                jobs.append(job)
                execution_plan_rows.append({k: v for k, v in row.items() if k not in ("requested_params", "applied_params")})
    seed_tag = "_".join(map(str, args.seeds))
    batch_name = f"{args.batch_prefix}_s{seed_tag}_draw{args.start_draw:06d}_{args.budget:06d}_pool{args.pool_size}"
    jobs_path = plan_root / f"{batch_name}_jobs.json"
    write_json(jobs_path, dict(batch_name=batch_name, stop_on_failure=False, jobs=jobs))
    pd.DataFrame(execution_plan_rows).to_csv(plan_root / f"{batch_name}_execution_plan.csv", index=False)
    report_path = plan_root / f"{batch_name}_plan.json"
    report = dict(batch_name=batch_name, jobs_path=str(jobs_path), start_draw=args.start_draw,
        cumulative_budget_end=args.budget, draws_selected_per_class_per_seed=args.budget - args.start_draw + 1,
        planned_pool_size=args.pool_size, seeds=args.seeds, n_jobs=len(jobs), n_classes=len(args.classes),
        fits=fits, pools=pool_refs, selected_end_speed_clipped=sum(j["source_context"]["end_speed_clipped"] for j in jobs),
        no_offset_or_duration_dimensions=True, no_theta_clipping=True, no_draw_filtering_or_resampling=True,
        retained_context="center provides q_minus/L1/L2/incoming vector/base/fixed Offset/Duration/start speed",
        interpretation="exploratory in-sample KDE budget result; does not complete all experiments 1-7",
        status="planned", sources=sources)
    write_json(report_path, report)
    print(json.dumps({"n_jobs": len(jobs), "jobs_path": str(jobs_path),
                      "end_speed_clipped": report["selected_end_speed_clipped"]}), flush=True)
    if not args.execute:
        return 0
    spec = importlib.util.spec_from_file_location("ours3_execution_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = time.perf_counter()
    result = module.run_batch(json.loads(jobs_path.read_text()), jobs_path)
    report.update(status="execution_completed" if result == 0 else "execution_returned_nonzero",
                  runner_returncode=result, runner_wall_s=time.perf_counter() - start,
                  runner_latest_batch=str(PROJECT / "runs" / batch_name / "latest_batch.json"))
    write_json(report_path, report)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
