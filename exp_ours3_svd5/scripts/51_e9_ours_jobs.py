#!/usr/bin/env python3
"""E9 (39_180 singleton table T8): ours3 arms, planned and executed through
scripts/20_ours3_generate.py run_batch (single worker, esmini).

  (vi)   expert OAT, class_data_used = none. Disk geometry (L = 10 m), anchor
         2641 (special, primary) and 2609 (population, secondary), 5 levels per
         axis: theta1/theta2 deltas {-20,-10,0,+10,+20} deg, v_end factors
         {0.6,0.8,1.0,1.2,1.4}. The default, +/-10 deg and +/-20 % levels are
         reused from runs/ours3_disk_39_180_oat/db3d7f5cbcdc61f3 (identical
         parameters, not re-rendered); only the 6 new levels per anchor run here.
  (vii)  expert-range Sobol, class_data_used = none. theta1, theta2 uniform in
         nominal +/- 15 deg, v_end uniform in [0, 30] km/h, anchor 2641;
         scrambled Sobol (scipy.stats.qmc), 128 points drawn, first 100 used,
         seeds {20260910, 20260911, 20260912}.
  (viii) conditional KDE, class_data_used = bandwidth. cvlib.ParamKDE restored
         from plans/ours3_disk_kde_dependent/7a54a1c472f1327d/fit_cutinl.npz
         (48 cutinl disk contexts, h = 0.5665 in standardized space; not refit),
         conditional draws centred on the SPECIAL (anchor 2641) vector (28.644,
         -2.327, 8.837) standardized with the class mean/sd, z_c + h N(0, I3).
         The fit's own 39_180 row is the population-anchor (2609) vector, so the
         centre is not a fit point. EndSpeed < 0 -> 0 with the clip flag
         (23b rule). 100 x 3 seeds, anchor 2641.

The Sobol/OAT arms use only this scenario's geometry and expert ranges; the
conditional-KDE arm borrows the class bandwidth and is labelled accordingly.
"""
from __future__ import annotations

import argparse
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
from scipy.stats import qmc

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RUNNER = PROJECT / "scripts/20_ours3_generate.py"
DISK_CSV = PROJECT / "results/disk_window_L10_special_39_180.csv"
OAT_JOBS = PROJECT / "results/ours3_disk_39_180_oat_jobs.json"
OAT_RUN = PROJECT / "runs/ours3_disk_39_180_oat/db3d7f5cbcdc61f3"
FIT_NPZ = PROJECT / "plans/ours3_disk_kde_dependent/7a54a1c472f1327d/fit_cutinl.npz"
FIT_JSON = PROJECT / "plans/ours3_disk_kde_dependent/7a54a1c472f1327d/fit.json"
REAL_TRACKS = ROOT / "exp_cross_coverage/data/special_39_180/real_tracks.parquet"
PLANS = PROJECT / "plans/e9_39_180"
SEEDS = [20260910, 20260911, 20260912]
N = 100
COLS = ["theta1_deg", "theta2_deg", "interaction_end_speed_kmh"]
SPECIAL = np.array([28.644383184776665, -2.3266750584164044, 8.836849193507831])
THETA_HALF = 15.0
V_RANGE = (0.0, 30.0)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def base_job(anchor):
    row = pd.read_csv(DISK_CSV)
    row = row[row.anchor_frame.eq(anchor)].iloc[0]
    template = [j for j in json.loads(OAT_JOBS.read_text())["jobs"] if j["anchor_frame"] == anchor and j["job_id"].endswith("__default")][0]
    cps = [[float(row.qm_x), float(row.qm_y)], [float(row.qc_x), float(row.qc_y)], [float(row.qp_x), float(row.qp_y)]]
    assert np.allclose(cps, template["cps_xy"]) and template["source_xosc"] == row.base
    job = dict(scenario_id="39_180", **{"class": "special_39_180"}, dataset="HetroD", recording="00", ego=39, target=180,
               min_frame=2424, max_frame=3002, anchor_frame=int(anchor), source_xosc=str(row.base), cps_xy=cps,
               geometry_variant_id="special_pet_euclid_L10_exact_rotation", expected_target_gt_end=2923,
               source_context=dict(template["source_context"]), source_files=[str(DISK_CSV)])
    if anchor == 2641:
        job["expected_default_end_speed_kmh"] = float(SPECIAL[2])
    return job, dict(theta1=float(row.theta1_deg), theta2=float(row.theta2_deg))


def oat_jobs():
    jobs, reused = [], []
    existing = {p.name: p for p in OAT_RUN.iterdir() if (p / "sample.json").exists()}
    for anchor in (2641, 2609):
        base, _ = base_job(anchor)
        levels = [("default", {}), ("theta1_minus10deg", dict(theta1_delta_deg=-10)), ("theta1_plus10deg", dict(theta1_delta_deg=10)),
                  ("theta2_minus10deg", dict(theta2_delta_deg=-10)), ("theta2_plus10deg", dict(theta2_delta_deg=10)),
                  ("end_speed_minus20pct", dict(end_speed_factor=0.8)), ("end_speed_plus20pct", dict(end_speed_factor=1.2)),
                  ("theta1_minus20deg", dict(theta1_delta_deg=-20)), ("theta1_plus20deg", dict(theta1_delta_deg=20)),
                  ("theta2_minus20deg", dict(theta2_delta_deg=-20)), ("theta2_plus20deg", dict(theta2_delta_deg=20)),
                  ("end_speed_minus40pct", dict(end_speed_factor=0.6)), ("end_speed_plus40pct", dict(end_speed_factor=1.4))]
        for name, extra in levels:
            job_id = f"a{anchor}__{name}"
            if job_id in existing:
                old = json.loads((existing[job_id] / "sample.json").read_text())
                oj = old["job"]
                assert oj["anchor_frame"] == anchor and np.allclose(oj["cps_xy"], base["cps_xy"]) and oj["source_xosc"] == base["source_xosc"]
                assert all(abs(float(oj.get(k, np.nan)) - float(v)) < 1e-12 for k, v in extra.items()) and old["status"] == "completed"
                reused.append(dict(job_id=job_id, anchor=anchor, sample_json=str(existing[job_id] / "sample.json"), level=name,
                                   source_run=str(OAT_RUN), **extra))
                continue
            jobs.append(dict(base, job_id=job_id, source_context=dict(base["source_context"], arm="ours3_oat", level=name,
                                                                        class_data_used="none"), **extra))
    return dict(batch_name="ours3_39_180_e9_oat", stop_on_failure=False, jobs=jobs), reused


def sobol_jobs(seed):
    base, nominal = base_job(2641)
    assert abs(nominal["theta1"] - SPECIAL[0]) < 1e-9 and abs(nominal["theta2"] - SPECIAL[1]) < 1e-9
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    unit = sampler.random(128)[:N]
    lo = np.array([SPECIAL[0] - THETA_HALF, SPECIAL[1] - THETA_HALF, V_RANGE[0]])
    hi = np.array([SPECIAL[0] + THETA_HALF, SPECIAL[1] + THETA_HALF, V_RANGE[1]])
    P = qmc.scale(unit, lo, hi)
    jobs, rows = [], []
    for d, p in enumerate(P, 1):
        params = dict(zip(COLS, map(float, p)))
        jid = f"sobol__seed{seed}__draw{d:03d}"
        jobs.append(dict(base, job_id=jid, parameters=params,
                         source_context=dict(base["source_context"], arm="ours3_sobol", seed=seed, draw=d, class_data_used="none",
                                             sampler="scipy.stats.qmc.Sobol(d=3, scramble=True, seed=seed).random(128)[:100]",
                                             ranges=dict(theta1_deg=[lo[0], hi[0]], theta2_deg=[lo[1], hi[1]], interaction_end_speed_kmh=list(V_RANGE)),
                                             end_speed_clipped=False)))
        rows.append(dict(job_id=jid, seed=seed, draw=d, **params, unit_u1=unit[d - 1][0], unit_u2=unit[d - 1][1], unit_u3=unit[d - 1][2]))
    return dict(batch_name=f"ours3_39_180_e9_sobol_{seed}", stop_on_failure=False, jobs=jobs), pd.DataFrame(rows)


def condkde_jobs(seed):
    base, _ = base_job(2641)
    with np.load(FIT_NPZ) as z:
        raw, mean, sd, Z, h = z["raw"], z["mean"], z["sd"], z["standardized"], float(z["h"])
    fit = json.loads(FIT_JSON.read_text())["fits"]["cutinl"]
    assert fit["center_scenario_ids"][0] == "39_180" and abs(fit["h"] - h) < 1e-12 and raw.shape == (48, 3)
    # The fit's own 39_180 row is the POPULATION-anchor (2609) context vector; the
    # kernel centre here is the SPECIAL (2641) vector, standardized with the class
    # mean/sd. It is not one of the 48 fit points.
    assert np.allclose(raw[0], [18.319860931768776, 6.214904457407825, 8.77913011], atol=1e-6), raw[0]
    assert np.allclose((raw - mean) / sd, Z)
    z_center = (SPECIAL - mean) / sd
    rng = np.random.default_rng(seed)
    Zs = z_center + h * rng.standard_normal((N, 3))
    requested = Zs * sd + mean
    applied = requested.copy()
    applied[:, 2] = np.maximum(0.0, applied[:, 2])
    jobs, rows = [], []
    for d in range(N):
        jid = f"condkde__seed{seed}__draw{d + 1:03d}"
        params = dict(zip(COLS, map(float, applied[d])))
        clipped = bool(requested[d, 2] < 0)
        jobs.append(dict(base, job_id=jid, parameters=params,
                         source_context=dict(base["source_context"], arm="ours3_condkde", seed=seed, draw=d + 1, class_data_used="bandwidth",
                                             kde_fit=str(FIT_NPZ), kde_fit_sha256=sha256(FIT_NPZ), kde_n_fit=48, kde_h_standardized=h,
                                             kernel_center="special_2641_vector_standardized_with_class_mean_sd (not a fit point; fit row 0 is the 2609 population vector)",
                                             kernel_center_raw=SPECIAL.tolist(), kernel_center_standardized=z_center.tolist(),
                                             sampler_parameters_requested=dict(zip(COLS, map(float, requested[d]))),
                                             end_speed_clipped=clipped, end_speed_clip_rule="max(0.0, requested_end_speed_kmh)",
                                             theta_clipped=False, rejected_or_resampled=False)))
        rows.append(dict(job_id=jid, seed=seed, draw=d + 1, **{f"requested_{c}": float(v) for c, v in zip(COLS, requested[d])},
                         **{f"applied_{c}": float(v) for c, v in zip(COLS, applied[d])}, end_speed_clipped=clipped, h=h))
    return dict(batch_name=f"ours3_39_180_e9_condkde_{seed}", stop_on_failure=False, jobs=jobs), pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="subset of batch names to execute")
    args = parser.parse_args()
    PLANS.mkdir(parents=True, exist_ok=True)
    specs, plan = [], dict(sources={p.name: dict(path=str(p), sha256=sha256(p)) for p in (DISK_CSV, OAT_JOBS, FIT_NPZ, FIT_JSON, RUNNER, Path(__file__))},
                           seeds=SEEDS, n_per_seed=N, batches=[])
    oat, reused = oat_jobs()
    pd.DataFrame(reused).to_csv(PLANS / "oat_reused_from_db3d7f5cbcdc61f3.csv", index=False)
    specs.append(oat)
    for seed in SEEDS:
        spec, table = sobol_jobs(seed)
        table.to_csv(PLANS / f"{spec['batch_name']}_plan.csv", index=False)
        specs.append(spec)
    for seed in SEEDS:
        spec, table = condkde_jobs(seed)
        table.to_csv(PLANS / f"{spec['batch_name']}_plan.csv", index=False)
        specs.append(spec)
        plan.setdefault("condkde_end_speed_clipped", {})[str(seed)] = int(table.end_speed_clipped.sum())
    for spec in specs:
        path = PLANS / f"{spec['batch_name']}_jobs.json"
        write_json(path, spec)
        plan["batches"].append(dict(batch_name=spec["batch_name"], jobs_json=str(path), n_jobs=len(spec["jobs"])))
    plan["oat_reused"] = len(reused)
    write_json(PLANS / "plan.json", plan)
    print(json.dumps(dict(batches=[(b["batch_name"], b["n_jobs"]) for b in plan["batches"]], oat_reused=len(reused),
                          condkde_clipped=plan.get("condkde_end_speed_clipped"))), flush=True)
    if not args.execute:
        return 0
    spec_mod = importlib.util.spec_from_file_location("ours3_runner_e9", RUNNER)
    module = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(module)
    results = {}
    for spec in specs:
        if args.only and spec["batch_name"] not in args.only:
            continue
        path = PLANS / f"{spec['batch_name']}_jobs.json"
        start = time.perf_counter()
        rc = module.run_batch(json.loads(path.read_text()), path)
        results[spec["batch_name"]] = dict(returncode=rc, wall_s=time.perf_counter() - start)
        print(f"[e9 ours] {spec['batch_name']}: rc={rc} {results[spec['batch_name']]['wall_s']:.0f}s", flush=True)
        write_json(PLANS / "execution.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
