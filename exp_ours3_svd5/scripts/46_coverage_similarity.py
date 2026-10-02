#!/usr/bin/env python3
"""Tables 3 and 4 (EXPERIMENT_PLAN.md): variation / coverage / first-hit and
distribution similarity of the generated pools against the real set.

Arms (pool, kind, kernel centre = the scenario the sample was rendered for):
  ours3_disk_kde              executed ours3 (theta1, theta2, EndSpeed) dependent KDE draws,
                              runs/ours3_disk_kde_s<seed>_*/ (1000 per class per seed)
  svd_d5_kde_matched_analytic analytic decode of generated/svd_d5_kde_matched (scripts/44),
                              1000 per class x 3 seeds
  svd_d5_kde_executed_E3      "SVD executed (timed Polyline, E3)": the first 100 per class per
                              seed executed in esmini (runs/svd_d5_executed_kde/a4960569a62fea3e),
                              scores reused read-only from results/svd_d5_executed_kde_nonpet_paired.csv
                              and results/bbox_pet_svd_executed_paired.csv
  sakura_bc                   1 per scene (192), nominal
  ours3_disk_defaults         1 per scene (469), nominal
  svd_d5_fullfit_recon / svd_d5_logo_recon   1 per scene analytic reconstructions of the real
                              set (in-sample rank-5 fullfit and LOGO); reported only as the
                              reachable ceiling of the SVD representation, not as a sampler.
Real reference = the real_* columns of the matched-cohort scorer outputs (489 scenes, one row
per scenario; PET from bbox_pet_matched_paired.csv, the rest from
matched_cohort/svd_d5_nonpet_paired.csv) and the GT target path from the per-subset
real_tracks.parquet. Two real-set denominators: matched489 (primary, the SVD-KDE training
set) and disk469 (the ours3 disk cohort, the ours-KDE training set).

Descriptor spaces:
  interaction  6-D (signed PET, d_min, alpha, conflict x, conflict y, u_c), z-scored with the
               REAL per-class std (train-frozen), eps = median real-to-real NN distance
  path         DTW of the generated target path to the GT target path of the SAME scenario
               (exp_cross_coverage/scripts/180_trajdtw_aggregate.py traj_dtw, AST-loaded verbatim
               as scripts/35 does); eps = median real-to-real NN dtw2 between arc-resampled GT paths
               of the class. A real scenario is path-covered only by samples whose kernel centre
               is that scenario (the descriptor is relative to the own centre).
  joint        both on the same sample.
Coverage / first-hit / budget curves follow exp_coverage_velocity/scripts/cvlib.py and
50_curves.py (running minimum, 20 orderings: order 0 = natural draw order, others
default_rng(60000 + o) permutations, eps x {1, 1.5, 2}; invalid draws cost budget but
cannot cover). Validity = Table 5 flags (scripts/45) joined by sample id: teleport, off-road (VL
windowing), physics gate 1, background solid hit at horizon chmed, sampler-invalid; samples
absent from Table 5 are "unflagged" and cannot cover in the valid-only pool.
Similarity follows exp_hexshift/scripts/03_stats_figs.py (W1, out-of-real-support,
variance ratio, count-matched bracket, per-centre span) and sr-tlkeep-experiment/core/
wasserstein.py (POT emd2 for the joint W1).

Outputs: results/e6_descriptors.parquet, e6_real_reference.csv, table3_coverage.csv,
table3_first_hit.csv, table3_variation.csv, table4_similarity.csv, TABLE34_REPORT.md,
table34_manifest.json, figures/table3_budget_curves.png.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
from scipy.stats import wasserstein_distance

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
DTW_SOURCE = ROOT / "exp_cross_coverage/scripts/180_trajdtw_aggregate.py"
CV_SOURCE = ROOT / "exp_coverage_velocity/scripts/cvlib.py"
CASES = RESULTS / "svd_d5_cases.csv"
DISK_CONTEXTS = RESULTS / "ours3_disk_contexts.json"   # 469 executable disk-window contexts (E10)
TABLE5 = RESULTS / "table5_validity_samples.csv"
TABLE2_CASES = RESULTS / "table2_six_measures_cases.csv"
E3_MANIFEST = PROJECT / "runs/svd_d5_executed_kde/a4960569a62fea3e/manifest.csv"
# ours3 disk defaults run id, read from the batch pointer (was the hard-coded pre-v3 id 024eaeeb650a5f60)
OURS_DISK_RUN = json.loads((PROJECT / "runs/ours3_disk_population_defaults/latest_batch.json").read_text())["run_id"]
CLASSES = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr"]
DESC = ["pet", "d_min", "alpha", "conflict_x", "conflict_y", "u_c"]
DESC_SRC = dict(pet="pet", d_min="min_dist", alpha="conflict_angle", conflict_x="conflict_x", conflict_y="conflict_y", u_c="agent_arr_speed")
UNITS = dict(pet="s", d_min="m", alpha="deg", conflict_x="m", conflict_y="m", u_c="m/s", dtw="m")
SPACES = ["interaction", "path", "joint"]
BUDGETS = [10, 100, 1000]
EMULT = [1.0, 1.5, 2.0]
NORD = 20
MATCH_N = [27, 100]
REAL_SETS = ["matched489", "disk469"]
SEED_ORDER = 60000

ARMS = {
    "ours3_disk_kde": dict(label="ours3 disk + KDE (executed)", kind="dependent", seeded=True),
    "svd_d5_kde_matched_analytic": dict(label="SVD d5 + matched KDE (analytic decode)", kind="dependent", seeded=True),
    "svd_d5_kde_executed_E3": dict(label="SVD executed (timed Polyline, E3) KDE, first 100/class/seed", kind="dependent", seeded=True),
    "sakura_bc": dict(label="SAKURA bc nominal (1 per scene)", kind="nominal", seeded=False),
    "ours3_disk_defaults": dict(label="ours3 disk defaults nominal (1 per scene)", kind="nominal", seeded=False),
    "svd_d5_fullfit_recon": dict(label="SVD d5 fullfit reconstruction (in-sample ceiling, 1 per scene)", kind="ceiling", seeded=False),
    "svd_d5_logo_recon": dict(label="SVD d5 LOGO reconstruction (held-out-group ceiling, 1 per scene)", kind="ceiling", seeded=False),
}
TABLE5_ARM = {"ours3_disk_kde": "ours3_disk_kde", "svd_d5_kde_matched_analytic": "svd_d5_kde",
              "svd_d5_kde_executed_E3": "svd_exec_E3_kde_kde", "sakura_bc": "sakura_bc", "ours3_disk_defaults": "ours3_disk",
              "svd_d5_fullfit_recon": "svd_d5_fullfit", "svd_d5_logo_recon": "svd_d5_logo"}
COLORS = {"ours3_disk_kde": "#2a78d6", "svd_d5_kde_matched_analytic": "#eb6834", "svd_d5_kde_executed_E3": "#1baf7a",
          "sakura_bc": "#eda100", "ours3_disk_defaults": "#e87ba4", "svd_d5_fullfit_recon": "#7a7a7a", "svd_d5_logo_recon": "#3d3d3d"}


# ── helpers ────────────────────────────────────────────────────────────────
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ast_load(path, names, namespace):
    src = Path(path).read_text()
    tree = ast.parse(src)
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in picked} != set(names):
        raise RuntimeError(f"{path}: expected {names}")
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), namespace)
    return {n.name: hashlib.sha256(ast.get_source_segment(src, n).encode()).hexdigest() for n in picked}


NS = {"np": np, "pd": pd}
DTW_HASHES = ast_load(DTW_SOURCE, ["arc_resample", "dtw2", "traj_dtw"], NS)
traj_dtw, dtw2, arc_resample = NS["traj_dtw"], NS["dtw2"], NS["arc_resample"]
CV_HASHES = ast_load(CV_SOURCE, ["nn_dists", "coverage", "running_min_dist", "first_hit", "_eps"], NS)
nn_dists, running_min_dist, first_hit, cv_eps = NS["nn_dists"], NS["running_min_dist"], NS["first_hit"], NS["_eps"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)) + "\n")


# ── real reference ─────────────────────────────────────────────────────────
class GroundTruth:
    def __init__(self):
        self.cache = {}

    def actor_path(self, source_tracks, scenario_id):
        source = str(Path(source_tracks).resolve())
        if source not in self.cache:
            tracks = pd.read_parquet(source)
            self.cache[source] = {(str(sid), role): g.sort_values("frame")[["frame", "x", "y"]].reset_index(drop=True)
                                  for (sid, role), g in tracks.groupby(["scenario_id", "role"])}
        return self.cache[source][(str(scenario_id), "actor")]


def load_real():
    nonpet = pd.read_csv(RESULTS / "matched_cohort/svd_d5_nonpet_paired.csv", low_memory=False)
    pet = pd.read_csv(RESULTS / "bbox_pet_matched_paired.csv", low_memory=False)
    nonpet = nonpet[nonpet["mode"].eq("fullfit")].reset_index(drop=True)
    pet = pet[pet["mode"].eq("fullfit")].drop_duplicates("scenario_uid").set_index("scenario_uid")
    cases = pd.read_csv(CASES, dtype={"recording": str, "scenario_id": str})
    full = cases[cases["mode"].eq("fullfit")].drop_duplicates("scenario_uid").set_index("scenario_uid")
    disk_ctx = json.loads(DISK_CONTEXTS.read_text())
    disk = set(c["scenario_uid"] for c in disk_ctx)
    assert len(disk) == len(disk_ctx), len(disk)      # 469 before 2026-09-14 (per-class L + cutinl exclusion)
    rows = []
    for r in nonpet.itertuples():
        p = pet.loc[r.scenario_uid]
        rows.append(dict(scenario_uid=r.scenario_uid, scenario_id=str(r.scenario_id), subset=r.subset,
                         group_id=full.group_id.get(r.scenario_uid), source_tracks=str(Path(r.source_tracks).resolve()),
                         ego=int(r.ego), actor=int(r.actor), metadata_min_frame=int(r.metadata_min_frame),
                         clock_aligned=bool(r.clock_aligned), in_matched489=True, in_disk469=r.scenario_uid in disk,
                         pet=float(p.real_pet), d_min=float(r.real_min_dist), alpha=float(r.real_conflict_angle),
                         conflict_x=float(r.real_conflict_x), conflict_y=float(r.real_conflict_y), u_c=float(r.real_agent_arr_speed),
                         real_pet_type=p.real_pet_type))
    real = pd.DataFrame(rows)
    assert real.scenario_uid.is_unique and len(real) == len(full[full.index.isin(real.scenario_uid)]), len(real)      # 489 -> 486 on 2026-09-14
    real["pet_finite"] = np.isfinite(real.pet)
    real["int_finite"] = np.isfinite(real[DESC].astype(float)).all(axis=1)
    return real


def real_scaling(real):
    """Train-frozen per-class std of the six descriptors over the matched489 real set (finite rows)."""
    out = {}
    for cls, g in real.groupby("subset"):
        g = g[g.int_finite]
        sd = g[DESC].astype(float).std(ddof=1)
        sd = sd.where(sd > 1e-9, 1.0)
        out[cls] = dict(mean=g[DESC].astype(float).mean().to_dict(), sd=sd.to_dict(), n=int(len(g)))
    return out


def zscore(df, scaling):
    Z = np.full((len(df), len(DESC)), np.nan)
    for i, (cls, idx) in enumerate(df.groupby("subset").groups.items()):
        s = scaling[cls]
        pos = df.index.get_indexer(idx)
        Z[pos] = (df.loc[idx, DESC].astype(float).to_numpy() - np.array([s["mean"][k] for k in DESC])) / np.array([s["sd"][k] for k in DESC])
    return Z


# ── real-to-real path eps ──────────────────────────────────────────────────
def _pair_dtw(args):
    i, j, A, B = args
    return i, j, dtw2(A, B)


def real_path_eps(real, gt, workers):
    cache_path = RESULTS / "e6_real_path_nn.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if set(cached.scenario_uid) == set(real.scenario_uid):
            return cached
    rows = []
    for cls, g in real.groupby("subset"):
        uids = list(g.scenario_uid)
        paths = []
        for r in g.itertuples():
            p = gt.actor_path(r.source_tracks, r.scenario_id)
            paths.append(arc_resample(p.x.to_numpy(float), p.y.to_numpy(float)))
        n = len(uids)
        tasks = [(i, j, paths[i], paths[j]) for i in range(n) for j in range(i + 1, n) if paths[i] is not None and paths[j] is not None]
        D = np.full((n, n), np.inf)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, j, d in ex.map(_pair_dtw, tasks, chunksize=256):
                D[i, j] = D[j, i] = d
        nn = D.min(axis=1)
        for k, uid in enumerate(uids):
            rows.append(dict(subset=cls, scenario_uid=uid, nn_dtw_m=float(nn[k]), nn_uid=uids[int(np.argmin(D[k]))] if np.isfinite(nn[k]) else ""))
        print(f"[eps path] {cls}: n={n}, pairs={len(tasks)}, median NN dtw = {np.median(nn[np.isfinite(nn)]):.3f} m", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False)
    return out


# ── arm loading ────────────────────────────────────────────────────────────
def join_pet(paired, pet, arm):
    pet = pet.drop_duplicates("sample_id").set_index("sample_id")
    missing = set(paired.sample_id) - set(pet.index)
    if missing:
        raise RuntimeError(f"{arm}: {len(missing)} samples without PET rows, e.g. {sorted(missing)[:3]}")
    p = pet.loc[paired.sample_id]
    if not (p.scenario_uid.to_numpy() == paired.scenario_uid.to_numpy()).all():
        raise RuntimeError(f"{arm}: PET join scenario mismatch")
    return p.reset_index()


def base_rows(arm, paired, pet, seed, draw, center, extra):
    """One descriptor row per sample from a 31-style paired table and a 32-style PET table."""
    p = join_pet(paired, pet, arm)
    out = pd.DataFrame(dict(
        arm=arm, arm_label=ARMS[arm]["label"], arm_kind=ARMS[arm]["kind"], subset=paired.subset.to_numpy(),
        seed=np.asarray(seed), draw=np.asarray(draw), sample_id=paired.sample_id.to_numpy(),
        scenario_uid=paired.scenario_uid.to_numpy(), kernel_center_uid=np.asarray(center),
        score_status=paired.score_status.to_numpy(), pet_score_status=p.pet_score_status.to_numpy(),
        clock_aligned=paired.clock_aligned.astype(bool).to_numpy(),
        pet=p.generated_pet.astype(float).to_numpy(), pet_type=p.generated_pet_type.to_numpy(),
        d_min=paired.generated_min_dist.astype(float).to_numpy(), alpha=paired.generated_conflict_angle.astype(float).to_numpy(),
        conflict_x=paired.generated_conflict_x.astype(float).to_numpy(), conflict_y=paired.generated_conflict_y.astype(float).to_numpy(),
        u_c=paired.generated_agent_arr_speed.astype(float).to_numpy(),
        trajectory_path=paired.trajectory_path.to_numpy() if "trajectory_path" in paired else "",
        source_tracks=paired.source_tracks.map(lambda s: str(Path(s).resolve())).to_numpy(),
        scenario_id=paired.scenario_id.astype(str).to_numpy(),
        metadata_min_frame=paired.metadata_min_frame.astype(float).to_numpy(), metadata_max_frame=paired.metadata_max_frame.astype(float).to_numpy(),
    ))
    for k, v in extra.items():
        out[k] = v
    return out


def load_ours_kde(gt_paths):
    frames = []
    for paired_path in sorted(RESULTS.glob("e6_ours3_disk_kde_s*_nonpet_paired.csv")) + [RESULTS / "e6_ours3_disk_kde_nonpet_paired.csv"]:
        if not paired_path.exists():
            continue
        pet_path = Path(str(paired_path).replace("e6_ours3_disk_kde", "bbox_pet_e6_ours3_disk_kde").replace("_nonpet_paired.csv", "_paired.csv"))
        if not pet_path.exists():
            print(f"[ours3_disk_kde] no PET file for {paired_path.name}; skipped", flush=True)
            continue
        paired = pd.read_csv(paired_path, low_memory=False)
        pet = pd.read_csv(pet_path, low_memory=False)
        meta = []
        for sj in paired.sample_json:
            s = json.loads(Path(sj).read_text())
            sc = s["job"]["source_context"]
            req, app = sc["sampler_parameters_requested"], sc["sampler_parameters_applied"]
            meta.append(dict(seed=int(sc["seed"]), draw=int(sc["draw"]), center=sc["kernel_center_scenario_uid"],
                             end_speed_clipped=bool(sc["end_speed_clipped"]), h=float(sc["h"]),
                             req_theta1=req["theta1_deg"], req_theta2=req["theta2_deg"], req_end_speed=req["interaction_end_speed_kmh"],
                             app_theta1=app["theta1_deg"], app_theta2=app["theta2_deg"], app_end_speed=app["interaction_end_speed_kmh"],
                             upstream_status=s.get("status"), run_id=s.get("run_id")))
        m = pd.DataFrame(meta)
        if not (m.center.to_numpy() == paired.scenario_uid.to_numpy()).all():
            raise RuntimeError("ours3_disk_kde: kernel centre differs from the rendered scenario")
        rows = base_rows("ours3_disk_kde", paired, pet, m.seed, m.draw, m.center,
                         dict(sampler_invalid=m.end_speed_clipped.to_numpy(), sampler_invalid_reason=np.where(m.end_speed_clipped, "end_speed_clipped", ""),
                              kde_h=m.h.to_numpy(), requested_theta1_deg=m.req_theta1.to_numpy(), requested_theta2_deg=m.req_theta2.to_numpy(),
                              requested_end_speed_kmh=m.req_end_speed.to_numpy(), applied_theta1_deg=m.app_theta1.to_numpy(),
                              applied_theta2_deg=m.app_theta2.to_numpy(), applied_end_speed_kmh=m.app_end_speed.to_numpy(),
                              upstream_status=m.upstream_status.to_numpy(), run_id=m.run_id.to_numpy(), input_paired_path=str(paired_path)))
        frames.append(rows)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    assert out.sample_id.is_unique
    return out


def load_svd_analytic():
    paired_path = RESULTS / "e6_svd_d5_kde_matched_analytic_nonpet_paired.csv"
    pet_path = RESULTS / "bbox_pet_e6_svd_d5_kde_matched_analytic_paired.csv"
    if not paired_path.exists() or not pet_path.exists():
        return None, None
    paired = pd.read_csv(paired_path, low_memory=False)
    pet = pd.read_csv(pet_path, low_memory=False)
    draws = pd.concat([pd.read_csv(p) for p in sorted((PROJECT / "runs/svd_d5_kde_matched_analytic").glob("*_seed*/draws.csv"))], ignore_index=True)
    draws = draws.set_index("sample_id")
    d = draws.loc[paired.sample_id]
    if not (d.center_scenario_uid.to_numpy() == paired.scenario_uid.to_numpy()).all():
        raise RuntimeError("svd analytic: centre mismatch")
    inv = d.duration_clipped.to_numpy() | ~d.numeric_ok.to_numpy()
    reason = np.where(d.duration_clipped, "duration_clipped", np.where(~d.numeric_ok, "numeric_not_ok", ""))
    rows = base_rows("svd_d5_kde_matched_analytic", paired, pet, d.seed.to_numpy(), d.draw_index.to_numpy(), d.center_scenario_uid.to_numpy(),
                     dict(sampler_invalid=inv, sampler_invalid_reason=reason, raw_duration_s=d.raw_duration_s.to_numpy(),
                          applied_duration_s=d.applied_duration_s.to_numpy(), upstream_status="analytic", run_id="analytic",
                          input_paired_path=str(paired_path)))
    return rows, draws


def load_svd_executed(draws):
    paired_path = RESULTS / "svd_d5_executed_kde_nonpet_paired.csv"
    pet_all = pd.read_csv(RESULTS / "bbox_pet_svd_executed_paired.csv", low_memory=False)
    pet = pet_all[pet_all.input_paired_path.map(lambda s: Path(s).resolve()) == paired_path.resolve()]
    paired = pd.read_csv(paired_path, low_memory=False)
    # run id taken from the E3 manifest folder (was the hard-coded pre-v3 id 18ddc200291162ed) so both stay in sync
    paired = paired[paired.run_id.astype(str).eq(E3_MANIFEST.parent.name)].reset_index(drop=True)
    assert len(paired), f"no executed-SVD KDE rows for run {E3_MANIFEST.parent.name} in {paired_path}"
    parsed = paired.sample_id.str.extract(r"svd5kde_(?P<cls>\w+?)_s(?P<seed>\d+)_(?P<draw>\d+)$")
    seed, draw = parsed.seed.astype(int).to_numpy(), parsed.draw.astype(int).to_numpy()
    if draws is not None:
        d = draws.reindex(paired.sample_id)
        inv = (d.duration_clipped.fillna(False).astype(bool) | ~d.numeric_ok.fillna(True).astype(bool)).to_numpy()
        reason = np.where(d.duration_clipped.fillna(False).astype(bool), "duration_clipped", np.where(~d.numeric_ok.fillna(True).astype(bool), "numeric_not_ok", ""))
    else:
        inv, reason = np.zeros(len(paired), bool), np.full(len(paired), "")
    man = pd.read_csv(E3_MANIFEST).set_index("sample_id")
    ade = man.reindex(paired.sample_id)
    rows = base_rows("svd_d5_kde_executed_E3", paired, pet, seed, draw, paired.scenario_uid.to_numpy(),
                     dict(sampler_invalid=inv, sampler_invalid_reason=reason, upstream_status=paired.upstream_status.to_numpy(),
                          run_id=paired.run_id.to_numpy(), input_paired_path=str(paired_path),
                          e3_ade_post_startup_m=ade.ade_post_startup_m.to_numpy(), e3_ade_all_vertices_m=ade.ade_all_vertices_m.to_numpy(),
                          e3_duration_clipped=ade.duration_clipped.to_numpy()))
    return rows


def load_nominal(arm, paired_path, pet, filt=None):
    paired = pd.read_csv(paired_path, low_memory=False)
    if filt is not None:
        paired = paired[filt(paired)].reset_index(drop=True)
    rows = base_rows(arm, paired, pet, 0, 1, paired.scenario_uid.to_numpy(),
                     dict(sampler_invalid=False, sampler_invalid_reason="", upstream_status=paired.upstream_status.to_numpy() if "upstream_status" in paired else "completed",
                          run_id=paired.run_id.to_numpy() if "run_id" in paired else "", input_paired_path=str(paired_path)))
    assert rows.sample_id.is_unique, arm
    return rows


def load_recon(mode):
    """1-per-scene SVD reconstructions (fullfit / LOGO) with the Table 2 DTW."""
    nonpet = pd.read_csv(RESULTS / "matched_cohort/svd_d5_nonpet_paired.csv", low_memory=False)
    pet = pd.read_csv(RESULTS / "bbox_pet_matched_paired.csv", low_memory=False)
    nonpet = nonpet[nonpet["mode"].eq(mode)].reset_index(drop=True)
    nonpet["sample_id"] = nonpet.subset + "__" + nonpet.scenario_id.astype(str) + "__" + mode
    pet = pet[pet["mode"].eq(mode)].copy()
    pet["sample_id"] = pet.subset + "__" + pet.scenario_id.astype(str) + "__" + mode
    nonpet["trajectory_path"] = ""
    arm = f"svd_d5_{mode}_recon"
    rows = base_rows(arm, nonpet, pet, 0, 1, nonpet.scenario_uid.to_numpy(),
                     dict(sampler_invalid=False, sampler_invalid_reason="", upstream_status="analytic", run_id="analytic",
                          input_paired_path=str(RESULTS / "matched_cohort/svd_d5_nonpet_paired.csv")))
    t2 = pd.read_csv(TABLE2_CASES, low_memory=False)
    t2 = t2[t2.arm.eq(f"svd_d5_{mode}")].set_index("scenario_uid")
    rows["dtw"] = t2.err_dtw.reindex(rows.scenario_uid).to_numpy()
    rows["dtw_coverage"] = t2.dtw_coverage.reindex(rows.scenario_uid).to_numpy()
    rows["dtw_source"] = "table2_six_measures_cases.csv"
    return rows


# ── DTW per sample ─────────────────────────────────────────────────────────
def _dtw_task(args):
    key, traj_path, lo, hi, source, scenario_id = args
    try:
        tidy = pd.read_parquet(traj_path, columns=["role", "frame", "x", "y"])
        tg = tidy[tidy.role.eq("target") & tidy.frame.between(lo - .5, hi + .5)].sort_values("frame")
        gg = _GT.actor_path(source, scenario_id)
        if len(tg) < 2:
            return key, np.nan, np.nan, len(tg)
        d, cov = traj_dtw(tg[["x", "y"]], gg)
        return key, float(d), float(cov), len(tg)
    except Exception as exc:  # keep the row, record the failure
        return key, np.nan, np.nan, -1


_GT = GroundTruth()


def compute_dtw(df, workers):
    cache_path = RESULTS / "e6_dtw_cache.csv"
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["arm", "sample_id", "dtw", "dtw_coverage", "n_gen", "trajectory_path"])
    cache_key = set(zip(cache.arm, cache.sample_id))
    todo = df[df.trajectory_path.astype(str).ne("") & df.dtw.isna() & ~pd.Series(list(zip(df.arm, df.sample_id)), index=df.index).isin(cache_key)]
    tasks = [((r.arm, r.sample_id), r.trajectory_path, r.metadata_min_frame, r.metadata_max_frame, r.source_tracks, r.scenario_id)
             for r in todo.itertuples()]
    print(f"[dtw] {len(tasks)} samples to compute ({len(cache)} cached)", flush=True)
    new = []
    t0 = time.monotonic()
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for n, (key, d, cov, ng) in enumerate(ex.map(_dtw_task, tasks, chunksize=64)):
                new.append(dict(arm=key[0], sample_id=key[1], dtw=d, dtw_coverage=cov, n_gen=ng))
                if (n + 1) % 2000 == 0:
                    print(f"[dtw] {n + 1}/{len(tasks)} ({time.monotonic() - t0:.0f}s)", flush=True)
        newdf = pd.DataFrame(new)
        newdf["trajectory_path"] = [t[1] for t in tasks]
        cache = pd.concat([cache, newdf], ignore_index=True).drop_duplicates(["arm", "sample_id"], keep="last")
        cache.to_csv(cache_path, index=False)
    c = cache.set_index(["arm", "sample_id"])
    idx = pd.MultiIndex.from_arrays([df.arm, df.sample_id])
    hit = idx.isin(c.index)
    df.loc[hit, "dtw"] = c.dtw.reindex(idx[hit]).to_numpy()
    df.loc[hit, "dtw_coverage"] = c.dtw_coverage.reindex(idx[hit]).to_numpy()
    df.loc[hit, "dtw_source"] = "traj_dtw(180_trajdtw_aggregate.py) on the saved trace"
    return df


# ── Table 5 join ───────────────────────────────────────────────────────────
def join_table5(df):
    cols = ["arm", "sample_id", "horizon", "scored", "sampler_invalid", "teleport", "offroad_vl", "offroad_e7", "wrongway",
            "phys_gate1", "any_solid", "ped_any_solid", "valid_all"]
    t5 = pd.read_csv(TABLE5, usecols=cols, low_memory=False)
    t5 = t5[t5.horizon.eq("chmed")].drop_duplicates(["arm", "sample_id"]).set_index(["arm", "sample_id"])
    keys = pd.MultiIndex.from_arrays([df.arm.map(TABLE5_ARM), df.sample_id])
    present = keys.isin(t5.index)
    sub = t5.reindex(keys[present])
    df["table5_available"] = present
    for c in ("teleport", "offroad_vl", "offroad_e7", "wrongway", "phys_gate1", "valid_all"):
        df[f"t5_{c}"] = np.nan
        df.loc[present, f"t5_{c}"] = sub[c].to_numpy()
    df["t5_bg_solid_chmed"] = np.nan
    df.loc[present, "t5_bg_solid_chmed"] = sub.any_solid.to_numpy()
    df["t5_ped_solid_chmed"] = np.nan
    df.loc[present, "t5_ped_solid_chmed"] = sub.ped_any_solid.to_numpy()
    df["t5_scored"] = np.nan
    df.loc[present, "t5_scored"] = sub.scored.to_numpy()

    def gate(r):
        if not r.table5_available or not bool(r.t5_scored):
            return np.nan
        bad = any(bool(x) for x in (r.t5_teleport, r.t5_offroad_vl, r.t5_phys_gate1, r.t5_bg_solid_chmed) if pd.notna(x))
        return not bad and not bool(r.sampler_invalid)
    df["valid_gate"] = df.apply(gate, axis=1)
    return df


# ── coverage machinery ─────────────────────────────────────────────────────
def hit_matrices(G, R, eps_int, eps_path, m, valid_mask=None):
    """Boolean (N_real, M) hit matrices for interaction / path / joint; invalid columns never hit."""
    ZG = G[[f"z_{k}" for k in DESC]].to_numpy(float)
    ZR = R[[f"z_{k}" for k in DESC]].to_numpy(float)
    okG = np.isfinite(ZG).all(axis=1)
    D = np.full((len(R), len(G)), np.inf)
    if okG.any():
        Gz = ZG[okG]
        for i in range(len(R)):
            D[i, okG] = np.linalg.norm(Gz - ZR[i], axis=1)
    H_int = D <= eps_int * m
    centre = G.kernel_center_uid.to_numpy()
    dtw = G.dtw.to_numpy(float)
    H_path = (centre[None, :] == R.scenario_uid.to_numpy()[:, None]) & (dtw[None, :] <= eps_path * m)
    if valid_mask is not None:
        H_int &= valid_mask[None, :]
        H_path &= valid_mask[None, :]
    return dict(interaction=H_int, path=H_path, joint=H_int & H_path), D


def budget_curve(H, orders):
    """coverage after b draws for each ordering: (n_orders, M)."""
    M = H.shape[1]
    out = np.empty((len(orders), M))
    for o, perm in enumerate(orders):
        out[o] = np.maximum.accumulate(H[:, perm], axis=1).mean(axis=0) if H.shape[0] else np.nan
    return out


def orderings(M, n_ord):
    return [np.arange(M)] + [np.random.default_rng(SEED_ORDER + o).permutation(M) for o in range(1, n_ord)]


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-orderings", type=int, default=NORD)
    args = ap.parse_args()
    t_start = time.monotonic()
    gt = _GT
    real = load_real()
    scaling = real_scaling(real)
    Zr = zscore(real, scaling)
    for j, k in enumerate(DESC):
        real[f"z_{k}"] = Zr[:, j]
    real_nn = real_path_eps(real, gt, args.workers)
    real = real.merge(real_nn[["scenario_uid", "nn_dtw_m"]], on="scenario_uid", how="left")
    eps = {}
    for cls, g in real.groupby("subset"):
        gi = g[g.int_finite]
        eps[cls] = dict(interaction=cv_eps(gi[[f"z_{k}" for k in DESC]].to_numpy(float)),
                        path=float(np.nanmedian(g.nn_dtw_m)), n_real_int=int(len(gi)), n_real=int(len(g)),
                        n_real_pet_nonfinite=int((~g.pet_finite).sum()))
    print("[eps]", json.dumps(eps, indent=1), flush=True)
    real.to_csv(RESULTS / "e6_real_reference.csv", index=False)

    # ── arms ──
    arms = {}
    ours = load_ours_kde(gt)
    if ours is not None:
        arms["ours3_disk_kde"] = ours
    svd_an, draws = load_svd_analytic()
    if svd_an is not None:
        arms["svd_d5_kde_matched_analytic"] = svd_an
    arms["svd_d5_kde_executed_E3"] = load_svd_executed(draws)
    pet_all = pd.read_csv(RESULTS / "bbox_pet_all_methods_paired.csv", low_memory=False)
    arms["sakura_bc"] = load_nominal("sakura_bc", RESULTS / "sakura_population_nonpet_paired.csv", pet_all[pet_all.method.eq("sakura_bc")])
    pet_disk = pd.read_csv(RESULTS / "bbox_pet_disk_paired.csv", low_memory=False)
    arms["ours3_disk_defaults"] = load_nominal("ours3_disk_defaults", RESULTS / "ours3_disk_population_nonpet_paired.csv",
                                               pet_disk[pet_disk.run_id.astype(str).eq(OURS_DISK_RUN)],
                                               filt=lambda p: p.run_id.astype(str).eq(OURS_DISK_RUN))
    arms["svd_d5_fullfit_recon"] = load_recon("fullfit")
    arms["svd_d5_logo_recon"] = load_recon("logo")
    df = pd.concat(arms.values(), ignore_index=True)
    for c in ("dtw", "dtw_coverage"):
        if c not in df:
            df[c] = np.nan
    if "dtw_source" not in df:
        df["dtw_source"] = ""
    df["dtw"] = df.dtw.astype(float)
    df = compute_dtw(df, args.workers)
    df = join_table5(df)
    # real reference of the centre + z-scores
    rr = real.set_index("scenario_uid")
    for k in DESC:
        df[f"real_{k}"] = rr[k].reindex(df.kernel_center_uid).to_numpy()
    df["centre_in_matched489"] = df.kernel_center_uid.isin(rr.index)
    df["centre_in_disk469"] = df.kernel_center_uid.map(rr.in_disk469).fillna(False).astype(bool)
    Z = zscore(df, scaling)
    for j, k in enumerate(DESC):
        df[f"z_{k}"] = Z[:, j]
    df["int_finite"] = np.isfinite(Z).all(axis=1)
    df["eps_interaction"] = df.subset.map(lambda c: eps[c]["interaction"])
    df["eps_path"] = df.subset.map(lambda c: eps[c]["path"])
    df["dist_to_centre_interaction"] = np.sqrt(sum((df[f"z_{k}"] - (df[f"real_{k}"] - df.subset.map(lambda c, k=k: scaling[c]["mean"][k])) / df.subset.map(lambda c, k=k: scaling[c]["sd"][k])) ** 2 for k in DESC))
    df["recovered_interaction"] = df.dist_to_centre_interaction <= df.eps_interaction
    df["recovered_path"] = df.dtw <= df.eps_path
    df["recovered_joint"] = df.recovered_interaction & df.recovered_path
    df.to_parquet(RESULTS / "e6_descriptors.parquet", index=False)
    print(f"[descriptors] {len(df)} rows; per arm: {df.arm.value_counts().to_dict()}", flush=True)

    # ── Table 3: coverage / budget curves / first hit ──
    cov_rows, curve_rows, fh_rows = [], [], []
    for cls in CLASSES:
        for real_set in REAL_SETS:
            Rall = real[real.subset.eq(cls) & (real.in_disk469 if real_set == "disk469" else real.in_matched489)]
            Rsets = dict(interaction=Rall[Rall.int_finite], path=Rall, joint=Rall[Rall.int_finite])
            for arm, spec in ARMS.items():
                if arm not in df.arm.unique():
                    continue
                A = df[df.arm.eq(arm) & df.subset.eq(cls)]
                if not len(A):
                    continue
                seeds = sorted(A.seed.unique()) if spec["seeded"] else [0]
                for seed in seeds:
                    G = A[A.seed.eq(seed)].sort_values("draw").reset_index(drop=True)
                    M = len(G)
                    Gfull = G
                    for pool in ("all", "valid"):
                        G = Gfull
                        if pool == "valid":
                            # Table 5 flags exist only for a prefix of each pool (completed samples at scoring time;
                            # SVD arms: first 100 per class per seed). The valid-only pool is that contiguous flagged
                            # prefix, so budgets beyond it are not reported rather than reported as artificially low.
                            flagged = G.table5_available.to_numpy()
                            n_prefix = int(np.argmin(flagged)) if not flagged.all() else len(flagged)
                            if n_prefix == 0:
                                continue
                            G = Gfull.iloc[:n_prefix].reset_index(drop=True)
                        M = len(G)
                        vmask = G.valid_gate.to_numpy()
                        valid_mask = np.array([bool(v) if pd.notna(v) else False for v in vmask])
                        vm = None if pool == "all" else valid_mask
                        for m in EMULT:
                            H_all, D = hit_matrices(G, Rall, eps[cls]["interaction"], eps[cls]["path"], m, vm)
                            for space in SPACES:
                                Rs = Rsets[space]
                                sel = Rall.scenario_uid.isin(Rs.scenario_uid).to_numpy()
                                H = H_all[space][sel]
                                nR = int(sel.sum())
                                if nR == 0:
                                    continue
                                if spec["kind"] == "dependent":
                                    ords = orderings(M, args.n_orderings)
                                    curve = budget_curve(H, ords)
                                    for b in sorted(set(BUDGETS + [M])):
                                        if b <= M:
                                            c = curve[:, b - 1]
                                            cov_rows.append(dict(subset=cls, real_set=real_set, arm=arm, arm_label=spec["label"], seed=seed, pool=pool, eps_mult=m,
                                                                 space=space, budget=b, budget_is_pool_size=b == M, n_real=nR, n_samples=M,
                                                                 n_samples_valid=int(valid_mask.sum()), n_samples_flagged=int(G.table5_available.sum()),
                                                                 coverage_mean=float(c.mean()), coverage_min=float(c.min()), coverage_max=float(c.max()),
                                                                 coverage_natural_order=float(c[0]), n_orderings=len(ords)))
                                    if m == 1.0 and real_set == "matched489":
                                        for b in sorted(set([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, M])):
                                            if b <= M:
                                                curve_rows.append(dict(subset=cls, arm=arm, seed=seed, pool=pool, space=space, budget=b,
                                                                       coverage_mean=float(curve[:, b - 1].mean()), coverage_min=float(curve[:, b - 1].min()),
                                                                       coverage_max=float(curve[:, b - 1].max())))
                                    if m == 1.0:
                                        # first hit = 1-based position of the first hitting draw in the natural draw order
                                        # (cvlib.first_hit convention on the running minimum; censored = inf)
                                        fh = np.full(H.shape[0], np.inf)
                                        for i in range(H.shape[0]):
                                            j = np.flatnonzero(H[i])
                                            if len(j):
                                                fh[i] = j[0] + 1
                                        fin = np.isfinite(fh)
                                        fh_rows.append(dict(subset=cls, real_set=real_set, arm=arm, arm_label=spec["label"], seed=seed, pool=pool, space=space,
                                                            n_real=nR, pool_size=M, n_hit=int(fin.sum()), censored_fraction=float(1 - fin.mean()) if nR else np.nan,
                                                            first_hit_median_all_censored=float(np.median(fh)) if nR else np.nan,
                                                            first_hit_median_hits_only=float(np.median(fh[fin])) if fin.any() else np.inf,
                                                            first_hit_p90_hits_only=float(np.percentile(fh[fin], 90)) if fin.any() else np.inf,
                                                            hit_at_1_fraction=float((fh == 1).mean()) if nR else np.nan))
                                else:  # nominal / ceiling: budget = number of scenes, no orderings
                                    c = float(H.any(axis=1).mean()) if nR else np.nan
                                    cov_rows.append(dict(subset=cls, real_set=real_set, arm=arm, arm_label=spec["label"], seed=seed, pool=pool, eps_mult=m,
                                                         space=space, budget=M, budget_is_pool_size=True, n_real=nR, n_samples=M,
                                                         n_samples_valid=int(valid_mask.sum()), n_samples_flagged=int(G.table5_available.sum()),
                                                         coverage_mean=c, coverage_min=c, coverage_max=c, coverage_natural_order=c, n_orderings=0))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(RESULTS / "table3_coverage.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(RESULTS / "table3_budget_curve_points.csv", index=False)
    fh_df = pd.DataFrame(fh_rows)
    fh_df.to_csv(RESULTS / "table3_first_hit.csv", index=False)

    # ── Table 3: per-centre variation / recovery ──
    var_rows = []
    for cls in CLASSES:
        for arm, spec in ARMS.items():
            A = df[df.arm.eq(arm) & df.subset.eq(cls)]
            if not len(A) or spec["kind"] != "dependent":
                continue
            for scope, sub in [("all_seeds", A)] + [(f"seed_{s}", A[A.seed.eq(s)]) for s in sorted(A.seed.unique())]:
                g = sub.groupby("kernel_center_uid")
                counts = g.size()
                multi = counts[counts >= 2].index
                row = dict(subset=cls, arm=arm, arm_label=spec["label"], scope=scope, n_samples=len(sub), n_centres=int(counts.size),
                           n_centres_ge2=int(len(multi)), samples_per_centre_median=float(counts.median()),
                           samples_per_centre_min=int(counts.min()), samples_per_centre_max=int(counts.max()),
                           sampler_invalid_share=float(sub.sampler_invalid.mean()))
                for k in DESC + ["dtw"]:
                    iq = g[k].agg(lambda v: (lambda f: np.subtract(*np.percentile(f, [75, 25])) if len(f) >= 2 else np.nan)(v[np.isfinite(v.astype(float))].astype(float)))
                    row[f"iqr_median_{k}"] = float(np.nanmedian(iq.loc[multi])) if len(multi) else np.nan
                    sp = g[k].agg(lambda v: (lambda f: f.max() - f.min() if len(f) >= 2 else np.nan)(v[np.isfinite(v.astype(float))].astype(float)))
                    row[f"span_median_{k}"] = float(np.nanmedian(sp.loc[multi])) if len(multi) else np.nan
                for space in SPACES:
                    rec = g[f"recovered_{space}"].mean()
                    row[f"recovery_{space}_median_over_centres"] = float(rec.median())
                    row[f"recovery_{space}_pooled"] = float(sub[f"recovered_{space}"].mean())
                    row[f"centres_with_any_recovery_{space}"] = float(g[f"recovered_{space}"].any().mean())
                var_rows.append(row)
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(RESULTS / "table3_variation.csv", index=False)

    # ── Table 4: distribution similarity ──
    import ot
    sim_rows = []
    rng = np.random.default_rng(20260910)
    for cls in CLASSES:
        R = real[real.subset.eq(cls)]
        for arm, spec in ARMS.items():
            A = df[df.arm.eq(arm) & df.subset.eq(cls)]
            if not len(A):
                continue
            scopes = [("all_seeds", A)] + ([(f"seed_{s}", A[A.seed.eq(s)]) for s in sorted(A.seed.unique())] if spec["seeded"] and A.seed.nunique() > 1 else [])
            if spec["kind"] == "dependent" and A.sampler_invalid.any():
                scopes.append(("all_seeds_sampler_valid", A[~A.sampler_invalid.astype(bool)]))
            for scope, sub in scopes:
                for k in DESC + ["dtw"]:
                    rv = R[k].astype(float).to_numpy() if k != "dtw" else np.zeros(len(R))
                    rv = rv[np.isfinite(rv)]
                    gv = sub[k].astype(float).to_numpy()
                    gv = gv[np.isfinite(gv)]
                    d = dict(subset=cls, arm=arm, arm_label=spec["label"], scope=scope, descriptor=k, unit=UNITS[k],
                             n_real=len(rv), n_arm_finite=len(gv), n_arm=len(sub), n_arm_nonfinite=int(len(sub) - len(gv)),
                             sampler_invalid_share=float(sub.sampler_invalid.mean()), real_median=float(np.median(rv)) if len(rv) else np.nan,
                             arm_median=float(np.median(gv)) if len(gv) else np.nan)
                    if k == "dtw":
                        # the real DTW to its own path is 0 by definition: W1 to that point mass = mean DTW; no spread ratios
                        d.update(w1=float(np.mean(gv)) if len(gv) else np.nan, variance_ratio=np.nan, iqr_ratio=np.nan,
                                 out_of_real_support=np.nan, real_range_covered=np.nan, arm_p90=float(np.percentile(gv, 90)) if len(gv) else np.nan)
                    elif len(rv) >= 2 and len(gv) >= 2:
                        lo, hi = rv.min(), rv.max()
                        d.update(w1=float(wasserstein_distance(rv, gv)), variance_ratio=float(np.var(gv, ddof=1) / max(np.var(rv, ddof=1), 1e-12)),
                                 iqr_ratio=float((np.percentile(gv, 75) - np.percentile(gv, 25)) / max(np.percentile(rv, 75) - np.percentile(rv, 25), 1e-9)),
                                 out_of_real_support=float(np.mean((gv < lo) | (gv > hi))),
                                 real_range_covered=float(np.mean((rv >= gv.min()) & (rv <= gv.max()))))
                    else:
                        d.update(w1=np.nan, variance_ratio=np.nan, iqr_ratio=np.nan, out_of_real_support=np.nan, real_range_covered=np.nan)
                    # count-matched bracket + per-centre span (dependent samplers only)
                    if spec["kind"] == "dependent" and k != "dtw":
                        rr_ = R.set_index("scenario_uid")[k].astype(float)
                        for mn in [None] + MATCH_N:
                            hits = tot = 0
                            for ck, gg in sub.groupby("kernel_center_uid"):
                                if ck not in rr_.index or not np.isfinite(rr_[ck]):
                                    continue
                                v = gg[k].astype(float).to_numpy()
                                v = v[np.isfinite(v)]
                                if mn is not None and len(v) > mn:
                                    v = rng.choice(v, mn, replace=False)
                                if len(v) < 2:
                                    continue
                                tot += 1
                                hits += int(v.min() <= rr_[ck] <= v.max())
                            tag = "all" if mn is None else f"match{mn}"
                            d[f"bracket_rate_{tag}"] = hits / tot if tot else np.nan
                            d[f"bracket_n_centres_{tag}"] = tot
                        sp = sub.groupby("kernel_center_uid")[k].agg(lambda v: (lambda f: f.max() - f.min() if len(f) >= 2 else np.nan)(v[np.isfinite(v.astype(float))].astype(float)))
                        d["per_centre_span_median"] = float(np.nanmedian(sp)) if sp.notna().any() else np.nan
                    sim_rows.append(d)
                # joint z-scored W1 (POT emd2, uniform weights) on finite 6-D rows
                ZR = R[R.int_finite][[f"z_{k}" for k in DESC]].to_numpy(float)
                ZG = sub[sub.int_finite][[f"z_{k}" for k in DESC]].to_numpy(float)
                if len(ZR) >= 2 and len(ZG) >= 2:
                    Mx = ot.dist(ZR, ZG, metric="euclidean")
                    w1j = float(ot.emd2(np.full(len(ZR), 1 / len(ZR)), np.full(len(ZG), 1 / len(ZG)), Mx, numItermax=1_000_000))
                else:
                    w1j = np.nan
                sim_rows.append(dict(subset=cls, arm=arm, arm_label=spec["label"], scope=scope, descriptor="joint6_zscored", unit="z",
                                     n_real=len(ZR), n_arm_finite=len(ZG), n_arm=len(sub), n_arm_nonfinite=int(len(sub) - len(ZG)),
                                     sampler_invalid_share=float(sub.sampler_invalid.mean()), w1=w1j))
    sim = pd.DataFrame(sim_rows)
    # seed spread
    seeded = sim[sim.scope.str.startswith("seed_")]
    if len(seeded):
        sp = seeded.groupby(["subset", "arm", "descriptor"]).w1.agg(["min", "max", "std"]).rename(columns=dict(min="w1_seed_min", max="w1_seed_max", std="w1_seed_std"))
        sim = sim.merge(sp.reset_index(), on=["subset", "arm", "descriptor"], how="left")
    sim.to_csv(RESULTS / "table4_similarity.csv", index=False)

    make_figure(pd.DataFrame(curve_rows), eps)
    write_report(df, real, eps, cov, fh_df, var_df, sim, scaling)
    manifest(df, eps, scaling, t_start)
    print(f"[done] {time.monotonic() - t_start:.0f}s", flush=True)


# ── figure ─────────────────────────────────────────────────────────────────
def make_figure(curve, eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(CLASSES), len(SPACES), figsize=(13, 15), sharex=True, sharey=True)
    for i, cls in enumerate(CLASSES):
        for j, space in enumerate(SPACES):
            ax = axes[i, j]
            for arm in ("ours3_disk_kde", "svd_d5_kde_matched_analytic", "svd_d5_kde_executed_E3"):
                for pool, ls in (("all", "-"), ("valid", "--")):
                    g = curve[curve.subset.eq(cls) & curve.space.eq(space) & curve.arm.eq(arm) & curve.pool.eq(pool)]
                    if not len(g):
                        continue
                    agg = g.groupby("budget").agg(mean=("coverage_mean", "mean"), lo=("coverage_min", "min"), hi=("coverage_max", "max")).reset_index()
                    ax.plot(agg.budget, agg["mean"], ls, color=COLORS[arm], lw=2)
                    if pool == "all":
                        ax.fill_between(agg.budget, agg.lo, agg.hi, color=COLORS[arm], alpha=0.15, lw=0)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.grid(True, color="#e6e6e6", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(space, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{cls}\ncoverage of real set", fontsize=10)
            if i == len(CLASSES) - 1:
                ax.set_xlabel("draws per class (budget)", fontsize=10)
            e = eps[cls]
            ax.text(0.02, 0.96, f"eps_int={e['interaction']:.2f}z  eps_path={e['path']:.2f}m", transform=ax.transAxes, fontsize=7.5, va="top", color="#52514e")
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for arm in ("ours3_disk_kde", "svd_d5_kde_matched_analytic", "svd_d5_kde_executed_E3"):
        for pool, ls in (("all", "-"), ("valid", "--")):
            handles.append(Line2D([0], [0], color=COLORS[arm], ls=ls, lw=2)); labels.append(f"{ARMS[arm]['label']} ({pool})")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Table 3 budget curves: fraction of real scenarios (matched489) with >= 1 sample within eps (x1); band = min-max over 20 orderings x seeds; dashed = valid-only pool", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(FIGURES / "table3_budget_curves.png", dpi=150)
    plt.close(fig)


# ── report ─────────────────────────────────────────────────────────────────
def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "inf" if isinstance(v, float) and np.isinf(v) else "n/a"
    return f"{v:.{nd}f}"


def write_report(df, real, eps, cov, fh, var, sim, scaling):
    L = ["# Tables 3 and 4 — variation, coverage, first-hit and distribution similarity", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by scripts/46_coverage_similarity.py. Real reference = matched489 (primary; the SVD-KDE training set); disk469 rows (the ours-KDE training set) in the CSVs. Interaction space = 6-D (signed PET, d_min, alpha, conflict x/y, u_c) z-scored with the REAL per-class std; eps = median real-to-real nearest-neighbour distance (x1 unless stated); path = DTW of the generated target path to the GT target path of its own kernel-centre scenario (a real scenario is path-covered only by samples rendered for it); joint = both on the same sample. Coverage at budget b = fraction of real scenarios with >= 1 sample within eps among the first b draws (mean and min-max over 20 orderings; per seed). Valid-only pool: samples failing Table 5 flags (teleport / off-road VL / physics gate 1 / background solid hit at chmed) or sampler-invalid cannot cover; samples not present in Table 5 are unflagged and also cannot cover in that pool (their counts are listed).", "",
         "**SVD in-sample caveat.** The SVD d5 basis and the KDE bandwidth of the svd_d5_kde arms were fitted on the same 489 scenes that form the real set; the ours3 KDE was fitted on the 469-scene disk cohort of the same scenes. Both dependent samplers are therefore in-sample; the svd_d5_fullfit_recon / svd_d5_logo_recon rows are the reachable ceiling of a single reconstruction per scene (LOGO = the scene's global ego/target group held out of the basis), not samplers.", ""]
    e3 = df[df.arm.eq("svd_d5_kde_executed_E3")]
    if len(e3):
        L += [f"**SVD executed (timed Polyline, E3)** analytic-vs-executed ADE (runs/svd_d5_executed_kde/a4960569a62fea3e/manifest.csv): post-startup median {np.nanmedian(e3.e3_ade_post_startup_m):.4f} m (p90 {np.nanpercentile(e3.e3_ade_post_startup_m.dropna(), 90):.4f} m, max {np.nanmax(e3.e3_ade_post_startup_m):.3f} m); all-vertex median {np.nanmedian(e3.e3_ade_all_vertices_m):.3f} m (max {np.nanmax(e3.e3_ade_all_vertices_m):.1f} m, startup transient of the timed Polyline). Its scores are reused read-only from the E3 outputs (first 100 draws per class per seed; batch 45dfe7516aaece5c was still running and is not included).", ""]
    L += ["**Valid-only pool.** Table 5 flags exist only for a contiguous prefix of each pool (ours3_disk_kde: all 1000 draws for tlkeep/keeptl/keeptl_sw, 634 for cutinl, none for cutinr; SVD analytic: first 100 per seed; SVD executed: first 100 per seed where scored). The valid-only pool is that flagged prefix; budgets beyond it are not reported. Valid-only comparisons between arms are therefore fair at budget 10 and 100 only; the budget-1000 valid rows exist for ours only.", ""]
    L += ["**Centre availability.** ours3_disk_kde draws kernel centres only from the 469 executable disk-window contexts (keeptl 39/50, cutinr 13/20, tlkeep 292/294 of the matched489 real set); the SVD-KDE draws centres from all 489. Real scenarios without an ours centre can still be interaction-covered by other centres' draws but never path-covered; the keeptl and cutinr similarity rows (notably conflict x/y W1 in cutinr) partly reflect this missing-centre mixture. The disk469 rows of table3_coverage.csv / table3_first_hit.csv restrict the real set to the 469 scenes both samplers were fitted on.", ""]
    arms_present = [a for a in ARMS if a in df.arm.unique()]
    L += ["## Pools", "", "| arm | n samples | classes x seeds | sampler-invalid share | Table 5 flagged | valid (gate) | non-finite interaction (= PET no-event, inf) | median DTW to own GT path (m) |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for a in arms_present:
        A = df[df.arm.eq(a)]
        L.append(f"| {a} — {ARMS[a]['label']} | {len(A)} | {A.subset.nunique()} x {A.seed.nunique()} (seeds {sorted(A.seed.unique())}) | {A.sampler_invalid.mean():.3f} | {int(A.table5_available.sum())} | {int(A.valid_gate.fillna(False).astype(bool).sum())} | {int((~A.int_finite).sum())} ({int(np.isinf(A.pet).sum())} PET inf) | {fmt(float(np.nanmedian(A.dtw)))} |")
    L += ["", "A sample with no bbox-PET event (PET = inf) has no finite 6-D interaction vector and can only cover in path space; such samples are kept in every pool and counted above (they cost budget)."]
    L += ["", "## eps per class (real-to-real median nearest-neighbour distance)", "", "| class | n real (matched489) | n real finite PET | eps interaction (z units) | eps path (m) | real std used for z: PET s / d_min m / alpha deg / cx m / cy m / u_c m/s |", "|---|---:|---:|---:|---:|---|"]
    for cls in CLASSES:
        e, s = eps[cls], scaling[cls]["sd"]
        L.append(f"| {cls} | {e['n_real']} | {e['n_real_int']} | {e['interaction']:.3f} | {e['path']:.3f} | {' / '.join(f'{s[k]:.2f}' for k in DESC)} |")
    L += ["", "## Table 3a — coverage of the real set (matched489, eps x1)", ""]
    for cls in CLASSES:
        L += [f"### {cls}", "", "| arm | seed | pool | space | budget | n real | coverage mean [min, max over orderings] | natural order |", "|---|---|---|---|---:|---:|---|---:|"]
        c = cov[cov.subset.eq(cls) & cov.real_set.eq("matched489") & cov.eps_mult.eq(1.0)]
        for r in c.sort_values(["arm", "seed", "pool", "space", "budget"]).itertuples():
            if r.budget not in (10, 100, 1000) and not r.budget_is_pool_size:
                continue
            L.append(f"| {r.arm} | {r.seed if ARMS[r.arm]['seeded'] else '-'} | {r.pool} | {r.space} | {r.budget}{' (pool)' if r.budget_is_pool_size else ''} | {r.n_real} | {r.coverage_mean:.3f} [{r.coverage_min:.3f}, {r.coverage_max:.3f}] | {r.coverage_natural_order:.3f} |")
        L.append("")
        # plain statements
        st = []
        def cov_at(arm, space, b, pool="all"):
            g = c[c.arm.eq(arm) & c.space.eq(space) & c.budget.eq(b) & c.pool.eq(pool)]
            return float(g.coverage_mean.mean()) if len(g) else np.nan
        for space in SPACES:
            parts = []
            for arm in ("ours3_disk_kde", "svd_d5_kde_matched_analytic", "svd_d5_kde_executed_E3"):
                for b in (100, 1000):
                    v = cov_at(arm, space, b)
                    if np.isfinite(v):
                        parts.append(f"{arm}@{b}={v:.2f}")
            nom = []
            for arm in ("sakura_bc", "ours3_disk_defaults", "svd_d5_fullfit_recon", "svd_d5_logo_recon"):
                g = c[c.arm.eq(arm) & c.space.eq(space) & c.pool.eq("all")]
                if len(g):
                    nom.append(f"{arm}={float(g.coverage_mean.iloc[0]):.2f} (n={int(g.n_samples.iloc[0])})")
            st.append(f"- {space}: " + ", ".join(parts) + ("; nominal/ceiling: " + ", ".join(nom) if nom else ""))
        f = fh[fh.subset.eq(cls) & fh.real_set.eq("matched489") & fh.pool.eq("all")]
        for space in SPACES:
            fs = f[f.space.eq(space)]
            if len(fs):
                st.append(f"- first hit ({space}): " + ", ".join(f"{r.arm}{'/s' + str(r.seed) if ARMS[r.arm]['seeded'] else ''}: median {fmt(r.first_hit_median_all_censored, 0)} draws (censored {r.censored_fraction:.2f}, pool {r.pool_size})" for r in fs.itertuples()))
        v = var[var.subset.eq(cls) & var.scope.eq("all_seeds")]
        for r in v.itertuples():
            st.append(f"- per-centre ({r.arm}, {r.n_samples} samples over {r.n_centres} centres, median {r.samples_per_centre_median:.0f}/centre, {r.n_centres_ge2} with >= 2): median IQR " + ", ".join(f"{k}={fmt(getattr(r, f'iqr_median_{k}'))}{UNITS[k]}" for k in DESC + ['dtw']) + f"; recovery (share of a centre's own samples within eps of its real descriptors) interaction {r.recovery_interaction_pooled:.2f}, path {r.recovery_path_pooled:.2f}, joint {r.recovery_joint_pooled:.2f}")
        L += st + [""]
    L += ["## Table 3b — first hit per real scenario (matched489, eps x1, natural draw order, censored at pool size)", "", "| class | arm | seed | pool | space | n real | pool size | hit | censored frac | median first hit (censored = inf counted) | median (hits only) | p90 (hits only) | hit at draw 1 |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fh[fh.real_set.eq("matched489")].sort_values(["subset", "arm", "seed", "pool", "space"]).itertuples():
        L.append(f"| {r.subset} | {r.arm} | {r.seed} | {r.pool} | {r.space} | {r.n_real} | {r.pool_size} | {r.n_hit} | {r.censored_fraction:.3f} | {fmt(r.first_hit_median_all_censored, 1)} | {fmt(r.first_hit_median_hits_only, 1)} | {fmt(r.first_hit_p90_hits_only, 1)} | {r.hit_at_1_fraction:.3f} |")
    L += ["", "## Table 3c — per-centre variation (dependent samplers, all seeds)", "", "| class | arm | n samples | centres | median n/centre | " + " | ".join(f"IQR {k} ({UNITS[k]})" for k in DESC + ["dtw"]) + " | recovery int / path / joint (pooled) | sampler-invalid |", "|---|---|---:|---:|---:|" + "---:|" * 7 + "---|---:|"]
    for r in var[var.scope.eq("all_seeds")].itertuples():
        L.append(f"| {r.subset} | {r.arm} | {r.n_samples} | {r.n_centres} ({r.n_centres_ge2} >= 2) | {r.samples_per_centre_median:.0f} | " + " | ".join(fmt(getattr(r, f"iqr_median_{k}")) for k in DESC + ["dtw"]) + f" | {r.recovery_interaction_pooled:.2f} / {r.recovery_path_pooled:.2f} / {r.recovery_joint_pooled:.2f} | {r.sampler_invalid_share:.3f} |")
    L += ["", "## Table 4 — distribution similarity to the real set (all seeds pooled; per-seed rows and spread in table4_similarity.csv)", ""]
    for cls in CLASSES:
        L += [f"### {cls}", "", "| arm | descriptor | n real | n arm finite | W1 (real units) | W1 seed range | var ratio | out-of-support | bracket all (n centres) | bracket match27 (n) | bracket match100 (n) | per-centre span median | sampler-invalid |", "|---|---|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|"]
        s = sim[sim.subset.eq(cls) & sim.scope.eq("all_seeds")]
        for r in s.itertuples():
            seedr = f"[{fmt(getattr(r, 'w1_seed_min', np.nan))}, {fmt(getattr(r, 'w1_seed_max', np.nan))}]" if hasattr(r, "w1_seed_min") and np.isfinite(getattr(r, "w1_seed_min", np.nan)) else "-"
            def br(tag):
                v, n = getattr(r, f"bracket_rate_{tag}", np.nan), getattr(r, f"bracket_n_centres_{tag}", np.nan)
                return f"{fmt(v)} ({int(n)})" if np.isfinite(v) else "-"
            L.append(f"| {r.arm} | {r.descriptor} | {r.n_real} | {r.n_arm_finite} | {fmt(r.w1)} | {seedr} | {fmt(getattr(r, 'variance_ratio', np.nan))} | {fmt(getattr(r, 'out_of_real_support', np.nan))} | {br('all')} | {br('match27')} | {br('match100')} | {fmt(getattr(r, 'per_centre_span_median', np.nan))} | {r.sampler_invalid_share:.3f} |")
        L.append("")
        # plain statements: who is closer per descriptor
        st = []
        for k in DESC + ["joint6_zscored"]:
            g = s[s.descriptor.eq(k) & s.arm.isin(["ours3_disk_kde", "svd_d5_kde_matched_analytic", "svd_d5_kde_executed_E3", "sakura_bc", "ours3_disk_defaults"])].dropna(subset=["w1"])
            if len(g):
                best = g.sort_values("w1").iloc[0]
                st.append(f"- {k}: lowest W1 = {best.arm} ({best.w1:.3f}); " + ", ".join(f"{r.arm} {r.w1:.3f}" for r in g.itertuples()))
        L += st + [""]
    L += ["## What ours loses / gains (plain statements)", ""]
    for cls in CLASSES:
        c = cov[cov.subset.eq(cls) & cov.real_set.eq("matched489") & cov.eps_mult.eq(1.0) & cov.pool.eq("all")]
        s = sim[sim.subset.eq(cls) & sim.scope.eq("all_seeds")]
        o = c[c.arm.eq("ours3_disk_kde") & c.budget.eq(1000)]
        v = c[c.arm.eq("svd_d5_kde_matched_analytic") & c.budget.eq(1000)]
        lines = []
        for space in SPACES:
            oo, vv = o[o.space.eq(space)], v[v.space.eq(space)]
            if len(oo) and len(vv):
                lines.append(f"{space}@1000: ours {oo.coverage_mean.mean():.2f} vs SVD-KDE analytic {vv.coverage_mean.mean():.2f} (seed mean)")
        w = []
        for k in DESC:
            so, sv = s[s.arm.eq("ours3_disk_kde") & s.descriptor.eq(k)], s[s.arm.eq("svd_d5_kde_matched_analytic") & s.descriptor.eq(k)]
            if len(so) and len(sv):
                w.append(f"{k}: W1 ours {so.w1.iloc[0]:.2f} vs SVD {sv.w1.iloc[0]:.2f}; out-of-support ours {so.out_of_real_support.iloc[0]:.2f} vs SVD {sv.out_of_real_support.iloc[0]:.2f}; var ratio ours {so.variance_ratio.iloc[0]:.2f} vs SVD {sv.variance_ratio.iloc[0]:.2f}")
        L.append(f"- **{cls}**: " + "; ".join(lines) + ". " + " | ".join(w))
    L += ["", "Files: table3_coverage.csv (all eps multipliers, both real sets, all/valid pools), table3_first_hit.csv, table3_variation.csv, table4_similarity.csv, e6_descriptors.parquet, e6_real_reference.csv, figures/table3_budget_curves.png, table34_manifest.json."]
    (RESULTS / "TABLE34_REPORT.md").write_text("\n".join(L) + "\n")


def manifest(df, eps, scaling, t_start):
    sources = [CASES, DISK_CONTEXTS, TABLE5, TABLE2_CASES, E3_MANIFEST, DTW_SOURCE, CV_SOURCE, Path(__file__),
               RESULTS / "matched_cohort/svd_d5_nonpet_paired.csv", RESULTS / "bbox_pet_matched_paired.csv",
               RESULTS / "bbox_pet_all_methods_paired.csv", RESULTS / "bbox_pet_disk_paired.csv",
               RESULTS / "sakura_population_nonpet_paired.csv", RESULTS / "ours3_disk_population_nonpet_paired.csv",
               RESULTS / "svd_d5_executed_kde_nonpet_paired.csv", RESULTS / "bbox_pet_svd_executed_paired.csv"]
    sources += sorted(RESULTS.glob("e6_*_nonpet_paired.csv")) + sorted(RESULTS.glob("bbox_pet_e6_*_paired.csv"))
    sources += sorted(set(Path(p) for p in df.source_tracks.unique()))
    sources += sorted((PROJECT / "generated/svd_d5_kde_matched").glob("*/seed_*.npz"))
    outputs = ["e6_descriptors.parquet", "e6_real_reference.csv", "e6_real_path_nn.csv", "e6_dtw_cache.csv", "table3_coverage.csv",
               "table3_first_hit.csv", "table3_variation.csv", "table3_budget_curve_points.csv", "table4_similarity.csv", "TABLE34_REPORT.md"]
    m = dict(script=str(Path(__file__)), generated=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=time.monotonic() - t_start,
             sources=[dict(path=str(p), sha256=sha256(p)) for p in sources if Path(p).exists()],
             verbatim_functions=dict(dtw=DTW_HASHES, cvlib=CV_HASHES),
             outputs=[dict(path=str(RESULTS / o), sha256=sha256(RESULTS / o)) for o in outputs if (RESULTS / o).exists()] + [dict(path=str(FIGURES / "table3_budget_curves.png"), sha256=sha256(FIGURES / "table3_budget_curves.png"))],
             counts=df.groupby(["arm", "subset"]).size().unstack(fill_value=0).to_dict(),
             seeds_used={a: sorted(int(s) for s in df[df.arm.eq(a)].seed.unique()) for a in df.arm.unique()},
             eps=eps, real_scaling=scaling, budgets=BUDGETS, eps_multipliers=EMULT, n_orderings=NORD, ordering_rng="numpy default_rng(60000 + o), o >= 1; order 0 = natural draw order",
             match_n=MATCH_N, real_sets=REAL_SETS,
             notes=["SVD executed arm scores reused read-only from results/svd_d5_executed_kde_nonpet_paired.csv and results/bbox_pet_svd_executed_paired.csv (run 18ddc200291162ed, first 100 per class per seed).",
                    "svd_d5_fullfit_recon / svd_d5_logo_recon are 1-per-scene reconstructions (ceilings), not samplers; their DTW comes from table2_six_measures_cases.csv.",
                    "ours3_disk_kde seeds 20260911/20260912 are included only if their e6 scoring files exist at run time."])
    write_json(RESULTS / "table34_manifest.json", m)


if __name__ == "__main__":
    main()
