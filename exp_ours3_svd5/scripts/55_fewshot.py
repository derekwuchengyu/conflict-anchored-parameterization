#!/usr/bin/env python3
"""Table 6b: few-shot SVD d5 held-out reconstruction vs the training-free ours3 default.

SELECTION RULE (written before any fit was run, 2026-09-10)
-----------------------------------------------------------
Held-out targets (one per class + two named cases):
  * per class: among the ours3 Euclidean-disk cohort scenes that are eligible in
    results/table2_six_measures_cases.csv (arm ours3_disk: clock-aligned, non-PET
    score ok, PET score ok) and have all six REAL descriptors finite
    (real_pet, real_min_dist, real_conflict_angle, real_conflict_x, real_conflict_y,
    real_agent_arr_speed), the scene whose robust z-vector is closest to the class
    median: z_k = (x_k - median_k) / (1.4826 * MAD_k) (MAD_k == 0 -> unit scale),
    distance = ||z||_2, ties broken by scenario_uid; a class-median scene that is
    also one of the named cases is kept once and flagged.
  * cutinl 39_180 (HetroD/00/39_180/2424-3002): the singleton case.
  * keeptl 230_179 (HetroD/00/230_179/3210-3620): the corner case. It is NOT in the
    disk cohort (q- unavailable under the paper L = 10 m disk rule), so the ours3
    disk line is "unavailable under the paper window rule" and the arc-window value
    is reported as sensitivity only.
Training pool for a held-out scene = the class's matched-cohort eligible scenes
(results/svd_d5_matched_scoring_cases.csv, mode fullfit, cohort_eligible, status ok)
minus every scene in the held-out scene's GLOBAL shared ego/target connected
component (the LOGO rule of scripts/10_svd_d5_models.py; group_id from
results/svd_d5_cases.csv, 85 components).
N_train in {6, 8, 16, 32, full}: 10 random subsets per finite N drawn without
replacement with numpy default_rng(20260910 + k), k = 0..9; "full" = the whole
pool, fitted once. N larger than the pool is reported as "not feasible (pool < N)".
N in {1, 2, 4} are listed as "rank-insufficient for d=5 (rank = N-1)" with no fit.
Per fit: raw SVD d5 (sr-tlkeep-experiment/core/svd_param.py, 50 xy + duration, no
endpoint fixing) on the subset; numerical rank = count(s > eps*max(101,N)*s[0]) as
in scripts/10 fit_and_decode; rank < 5 is a reported rank failure (no retry, no
re-draw). Otherwise the held-out raw vector is encoded in the subset basis and
decoded (LOGO-style reconstruction), the decode is written as a pseudo-sample with
the scripts/50 convention (50 time-major xy, frame = metadata_min_frame +
linspace(0, duration, 50) * 30, applied duration = max(raw, 0.5) with raw kept and
the clip flagged, kinematics derived from xy, Ego rows = recorded GT ego) and
scored by the unchanged scripts/31 (--method svd_d5_fewshot) and scripts/32
(cache results/bbox_pet_cache_e8.json; one 32 run per stage). DTW = traj_dtw of
exp_cross_coverage/scripts/180_trajdtw_aggregate.py loaded verbatim by AST, decoded
50-point path vs GT target path (same as scripts/35). Explained variance = s^2
share of the first 5 singular values on the subset; KDE definability = loo_bandwidth
(core/kde_sampling.py) on V_train_d finite and > 0, h reported.
Aggregation per held-out scene x N: median and IQR over the subsets of each of the
six discrepancies; b_k = max over the arm set {SVD few-shot N rows (median over
subsets), ours3 disk default, ours3 arc default} of the per-measure value; D_m =
mean_k value_k / b_k (measures with b_k > 0); per-subset D_m uses the same b_k.
ours3 values are the single-render errors of the same scene from
results/table2_six_measures_cases.csv (arms ours3_disk / ours3_arc; training-free,
hence flat in N). fullfit numbers in Table 2 are in-sample; every number here is
held-out (the scene is never in the subset). The optional KDE stage (N in {8, 32,
full}): 100 dependent-KDE draws per fit (sample_dependent, h = h_loo of that fit,
rng default_rng(20260910 + k)), scored the same way; the nearest-draw D_m to the
held-out scene is an ORACLE best-of-100 number, labelled as such, never a
single-reconstruction column.

Usage: python -B scripts/55_fewshot.py [--phase all|fit|score|aggregate|kde|kde_score|kde_aggregate]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
RUNS = PROJECT / "runs/table6b_fewshot"
RUNS_KDE = PROJECT / "runs/table6b_fewshot_kde"
PY = sys.executable
sys.path.insert(0, str(ROOT / "sr-tlkeep-experiment"))
sys.path.insert(0, str(ROOT / "hetero-param"))
from core.svd_param import SvdParameterization  # noqa: E402
from core.kde_sampling import loo_bandwidth, sample_dependent  # noqa: E402
from hetero_param.similarity import core as SIMC  # noqa: E402

CASES_TABLE2 = RESULTS / "table2_six_measures_cases.csv"
MATCHED_CASES = RESULTS / "svd_d5_matched_scoring_cases.csv"
CASES_513 = RESULTS / "svd_d5_cases.csv"
DTW_SOURCE = ROOT / "exp_cross_coverage/scripts/180_trajdtw_aggregate.py"
SVD_SOURCE = ROOT / "sr-tlkeep-experiment/core/svd_param.py"
KDE_SOURCE = ROOT / "sr-tlkeep-experiment/core/kde_sampling.py"
HASH_LIST = PROJECT / "HANDOFF_FILE_HASHES.sha256"
CLASSES = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr"]
SPECIALS = [("cutinl", "HetroD/00/39_180/2424-3002", "singleton case"),
            ("keeptl", "HetroD/00/230_179/3210-3620", "corner case; not in disk cohort")]
REAL_KEYS = ["real_pet", "real_min_dist", "real_conflict_angle", "real_conflict_x", "real_conflict_y", "real_agent_arr_speed"]
MEASURES = ["pet", "dmin", "alpha", "cpoint", "uc", "dtw"]
PAPER = dict(pet="|dPET| (s)", dmin="|d d_min| (m)", alpha="|d alpha| (deg)", cpoint="||d c|| (m)", uc="|d u_c| (m/s)", dtw="DTW(target) (m)")
N_FINITE = [6, 8, 16, 32]
N_RANK_INSUFFICIENT = [1, 2, 4]
N_KDE = ["8", "32", "full"]
N_SUBSETS, SEED0, D, NT, NX, FPS, DUR_MIN = 10, 20260910, 5, 50, 101, 30.0, 0.5
KDE_DRAWS, LOO_GRID = 100, 60
METHOD = "svd_d5_fewshot"
N_ORDER = ["1", "2", "4", "6", "8", "16", "32", "full"]
CACHE_E8 = RESULTS / "bbox_pet_cache_e8.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def ast_load(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in picked} == set(names), (path, names)
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), namespace)
    src = Path(path).read_text()
    return {n.name: hashlib.sha256(ast.get_source_segment(src, n).encode()).hexdigest() for n in picked}


DTW_NS = {"np": np, "pd": pd}
DTW_HASHES = ast_load(DTW_SOURCE, ["arc_resample", "dtw2", "traj_dtw"], DTW_NS)
traj_dtw = DTW_NS["traj_dtw"]


def hash_check():
    proc = subprocess.run(["sha256sum", "-c", str(HASH_LIST.name)], cwd=PROJECT, capture_output=True, text=True)
    bad = [l for l in proc.stdout.splitlines() if not l.endswith(": OK")]
    return dict(returncode=proc.returncode, n_lines=len(proc.stdout.splitlines()), failures=bad)


def nan_to_none(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


# ── selection ───────────────────────────────────────────────────────────────
def select_targets():
    t2 = pd.read_csv(CASES_TABLE2)
    disk = t2[t2.arm.eq("ours3_disk")].set_index("scenario_uid")
    arc = t2[t2.arm.eq("ours3_arc")].set_index("scenario_uid")
    matched = pd.read_csv(MATCHED_CASES, dtype={"recording": str})
    matched = matched[matched["mode"].eq("fullfit") & matched.cohort_eligible & matched.status.eq("ok")]
    assert len(matched) == 489, len(matched)
    matched = matched.set_index("scenario_uid")
    groups = pd.read_csv(CASES_513, dtype={"recording": str})
    groups = groups[groups["mode"].eq("fullfit")].drop_duplicates("scenario_uid").set_index("scenario_uid")
    assert groups.group_id.nunique() == 85
    targets, sel_rows = [], []
    for cls in CLASSES:
        cand = disk[disk.subset.eq(cls) & disk.eligible.astype(bool)]
        cand = cand[np.isfinite(cand[REAL_KEYS].astype(float)).all(axis=1)]
        cand = cand[cand.index.isin(matched.index)]
        X = cand[REAL_KEYS].astype(float)
        med = X.median()
        mad = (X - med).abs().median() * 1.4826
        scale = mad.where(mad > 0, 1.0)
        z = (X - med) / scale
        dist = np.sqrt((z ** 2).sum(axis=1))
        order = pd.DataFrame(dict(dist=dist, uid=cand.index)).sort_values(["dist", "uid"])
        uid = order.uid.iloc[0]
        for rank, r in enumerate(order.head(5).itertuples()):
            sel_rows.append(dict(subset=cls, candidate_rank=rank, scenario_uid=r.uid, robust_z_distance=r.dist,
                                 selected=r.uid == uid, n_candidates=len(cand), **{k: float(cand.loc[r.uid, k]) for k in REAL_KEYS},
                                 **{f"class_median_{k}": float(med[k]) for k in REAL_KEYS}))
        targets.append(dict(subset=cls, scenario_uid=uid, role="class_median", note=f"closest to class median (robust z dist {order.dist.iloc[0]:.3f}, {len(cand)} candidates)"))
    for cls, uid, note in SPECIALS:
        dup = [t for t in targets if t["scenario_uid"] == uid]
        if dup:
            dup[0]["role"] += "+named_case"
            dup[0]["note"] += f"; also the named case ({note})"
        else:
            targets.append(dict(subset=cls, scenario_uid=uid, role="named_case", note=note))
    out = []
    for t in targets:
        uid, cls = t["scenario_uid"], t["subset"]
        m = matched.loc[uid]
        assert m.subset == cls
        g = groups.loc[uid, "group_id"]
        assert g == m.group_id
        pool = matched[matched.subset.eq(cls) & ~matched.group_id.eq(g)]
        in_disk = uid in disk.index
        t.update(scenario_id=str(m.scenario_id), group_id=g, case_index=int(m.case_index), ego=int(m.ego), actor=int(m.actor),
                 metadata_min_frame=int(m.metadata_min_frame), metadata_max_frame=int(m.metadata_max_frame),
                 source_tracks=str(m.source_tracks), raw_features_npz=str(m.raw_features_npz),
                 n_group_in_class=int((matched.subset.eq(cls) & matched.group_id.eq(g)).sum()),
                 pool_uids=pool.index.tolist(), pool_case_indices=pool.case_index.astype(int).tolist(), n_pool=len(pool),
                 disk_available=bool(in_disk), disk_eligible=bool(in_disk and disk.loc[uid, "eligible"]),
                 arc_available=uid in arc.index, arc_eligible=bool(uid in arc.index and arc.loc[uid, "eligible"]),
                 ours3_disk={m_: (float(disk.loc[uid, f"err_{m_}"]) if in_disk else np.nan) for m_ in MEASURES},
                 ours3_arc={m_: (float(arc.loc[uid, f"err_{m_}"]) if uid in arc.index else np.nan) for m_ in MEASURES},
                 real={k: float((disk if in_disk else arc).loc[uid, k]) for k in REAL_KEYS})
        out.append(t)
    return out, pd.DataFrame(sel_rows)


# ── pseudo-sample writer (scripts/50 convention, generalised to any scene) ──
class ClassSource:
    def __init__(self):
        self.tracks, self.raw = {}, {}

    def ref(self, target):
        src = target["source_tracks"]
        if src not in self.tracks:
            self.tracks[src] = pd.read_parquet(src)
        tr = self.tracks[src]
        sid = target["scenario_id"]
        e = tr[tr.role.eq("ego") & tr.scenario_id.astype(str).eq(sid)].sort_values("frame")
        a = tr[tr.role.eq("actor") & tr.scenario_id.astype(str).eq(sid)].sort_values("frame")
        assert e.track_id.eq(target["ego"]).all() and a.track_id.eq(target["actor"]).all()
        h, sp = e.heading_deg.to_numpy(float), e.speed.to_numpy(float)
        lo = target["metadata_min_frame"]
        t = (e.frame.to_numpy(float) - lo) / FPS
        vx, vy = sp * np.cos(np.radians(h)), sp * np.sin(np.radians(h))
        ego = pd.DataFrame(dict(scenario_id=sid, entity_name="Ego", actor_id=str(target["ego"]), role="ego", time_s=t,
                                x=e.x.to_numpy(float), y=e.y.to_numpy(float), z=0.0, heading_deg=h, speed_mps=sp,
                                length=float(e.length.median()), width=float(e.width.median()), vx_mps=vx, vy_mps=vy,
                                ax_mps2=np.gradient(vx, t), ay_mps2=np.gradient(vy, t), frame=e.frame.to_numpy(float)))
        return dict(ego=ego, target_length=float(a.length.median()), target_width=float(a.width.median()),
                    actor=a, actor_min_frame=int(a.frame.min()), actor_max_frame=int(a.frame.max()))

    def raw_features(self, target):
        p = PROJECT / target["raw_features_npz"]
        if p not in self.raw:
            with np.load(p, allow_pickle=False) as z:
                self.raw[p] = dict(X=z["X_raw"].copy(), uid=z["scenario_uid"].astype(str), sha256=sha256(p), path=str(p))
        return self.raw[p]


def tidy_from_vector(vector_applied, sample_id, target, ref):
    xy = np.asarray(vector_applied[:NT * 2], float).reshape(NT, 2)
    lo, hi = target["metadata_min_frame"], target["metadata_max_frame"]
    frame = lo + np.linspace(0.0, float(vector_applied[-1]), NT) * FPS
    heading, speed, accel = SIMC.derive_kinematics(frame, xy[:, 0], xy[:, 1], FPS)
    vx, vy = speed * np.cos(np.radians(heading)), speed * np.sin(np.radians(heading))
    t = (frame - lo) / FPS
    tgt = pd.DataFrame(dict(sample_id=sample_id, scenario_id=target["scenario_id"], subset=target["subset"],
                            entity_name="Agent1", actor_id=str(target["actor"]), role="target", time_s=t,
                            x=xy[:, 0], y=xy[:, 1], z=0.0, heading_deg=heading, speed_mps=speed,
                            length=ref["target_length"], width=ref["target_width"], vx_mps=vx, vy_mps=vy,
                            ax_mps2=np.gradient(vx, t), ay_mps2=np.gradient(vy, t), frame=frame))
    ego = ref["ego"].copy()
    ego.insert(0, "sample_id", sample_id)
    ego["subset"] = target["subset"]
    out = pd.concat([tgt, ego[tgt.columns]], ignore_index=True)
    gt_end = ref["actor_max_frame"]
    out["within_metadata_window"] = out.frame.between(lo - .5, hi + .5)
    out["within_target_gt_support"] = out.frame.between(lo - .5, gt_end + .5)
    ent_hi = np.where(out.role.eq("target"), gt_end, hi)
    out["within_entity_gt_support"] = (out.frame >= lo - .5) & (out.frame <= ent_hi + .5)
    return out


def write_sample(wd, sample_id, target, ref, vector_raw, meta, method=METHOD):
    wd.mkdir(parents=True, exist_ok=True)
    applied = np.asarray(vector_raw, float).copy()
    applied[-1] = max(float(vector_raw[-1]), DUR_MIN)
    tidy = tidy_from_vector(applied, sample_id, target, ref)
    traj = wd / "trajectory.parquet"
    tidy.to_parquet(traj, index=False)
    lo, hi = target["metadata_min_frame"], target["metadata_max_frame"]
    src_ctx = dict(subset=target["subset"], source_tracks=target["source_tracks"], scenario_uid=target["scenario_uid"])
    job = dict(job_id=sample_id, scenario_id=target["scenario_id"], **{"class": target["subset"]}, dataset="HetroD", recording="00",
               ego=target["ego"], target=target["actor"], min_frame=lo, max_frame=hi,
               geometry_variant_id=f"{method}_analytic_decode", source_context=src_ctx)
    context = dict(scenario_id=target["scenario_id"], dataset="HetroD", recording="00", subset=target["subset"],
                   ego=target["ego"], target=target["actor"], metadata_window_frames=[lo, hi],
                   target_gt_support_frames=[ref["actor_min_frame"], ref["actor_max_frame"]], fps=FPS,
                   geometry_variant_id=job["geometry_variant_id"], source_context=src_ctx, scenario_uid=target["scenario_uid"])
    report = dict(sample_id=sample_id, sample_dir=str(wd), job=job, context=context, status="completed", analytic=True, simulated=False,
                  method=method, is_nominal_default=True, parameters_applied={}, held_out=True,
                  label_sentence="Analytic SVD d5 decode of the held-out scene in a few-shot subset basis; the scene is not in the training subset",
                  decode=dict(convention="scripts/30_interaction_metrics.py: 50 time-major xy pairs, frame = metadata_min_frame + linspace(0, duration, 50) * 30, kinematics derived from xy",
                              duration_raw_s=float(vector_raw[-1]), duration_applied_s=float(applied[-1]),
                              duration_clipped=bool(vector_raw[-1] < DUR_MIN), duration_rule=f"max(raw, {DUR_MIN})"),
                  vector_raw=np.asarray(vector_raw, float).tolist(), trajectory_path=str(traj), trajectory_rows=len(tidy), **meta)
    report["artifacts"] = [dict(path=str(traj), sha256=sha256(traj), size_bytes=traj.stat().st_size)]
    write_json(wd / "sample.json", report)
    return report


# ── fits ────────────────────────────────────────────────────────────────────
def numerical_rank(P, n_train):
    tol = np.finfo(float).eps * max(NX, n_train) * P.s[0]
    return int(np.count_nonzero(P.s > tol))


def fit_subset(X_train, x_held):
    t0 = time.perf_counter()
    P = SvdParameterization(NT, 2, 1).fit(X_train)
    fit_wall = time.perf_counter() - t0
    rank = numerical_rank(P, len(X_train))
    rec = dict(numerical_rank=rank, fit_wall_s=fit_wall, explained_variance_d5=float(P.explained_variance(D)) if rank >= 1 else np.nan)
    if rank < D:
        rec.update(fit_status="insufficient_numeric_rank", detail=f"centered numerical rank {rank} < d={D}")
        return P, rec, None, None
    z = ((x_held * P.alpha - P.mu) @ P.U[:, :D]) / P.s[:D]
    x_rec = P.reconstruct(z, D)[0]
    if not np.isfinite(x_rec).all():
        rec.update(fit_status="fit_or_decode_failure", detail="non-finite decode")
        return P, rec, None, None
    V = P.reduced(D)
    t0 = time.perf_counter()
    try:
        h = float(loo_bandwidth(V, n_grid=LOO_GRID))
        h_err = ""
    except Exception as exc:  # report, never retry
        h, h_err = np.nan, f"{type(exc).__name__}: {exc}"
    rec.update(fit_status="ok", detail="", z=z.tolist(), h_loo=h, h_loo_error=h_err, bandwidth_wall_s=time.perf_counter() - t0,
               kde_definable=bool(np.isfinite(h) and h > 0), duration_raw_s=float(x_rec[-1]),
               duration_applied_s=float(max(x_rec[-1], DUR_MIN)), duration_clipped=bool(x_rec[-1] < DUR_MIN))
    return P, rec, x_rec, V


def phase_fit(targets, src):
    RUNS.mkdir(parents=True, exist_ok=True)
    fit_rows, sample_paths = [], []
    for t in targets:
        ref = src.ref(t)
        rf = src.raw_features(t)
        assert rf["uid"][t["case_index"]] == t["scenario_uid"]
        x_held = rf["X"][t["case_index"]]
        pool_idx = np.asarray(t["pool_case_indices"], int)
        assert t["case_index"] not in pool_idx and len(pool_idx) == t["n_pool"]
        assert all(rf["uid"][i] != t["scenario_uid"] for i in pool_idx)
        scene_dir = RUNS / f"{t['subset']}__{t['scenario_id']}"
        jobs = []
        for N in N_FINITE:
            if N > len(pool_idx):
                fit_rows.append(dict(subset=t["subset"], scenario_uid=t["scenario_uid"], scenario_id=t["scenario_id"], role=t["role"],
                                     N_train=str(N), N_int=N, subset_k=np.nan, seed=np.nan, n_train=0, n_pool=len(pool_idx),
                                     fit_status="not_feasible_pool_smaller_than_N", detail=f"pool {len(pool_idx)} < N {N}"))
                continue
            for k in range(N_SUBSETS):
                seed = SEED0 + k
                rng = np.random.default_rng(seed)
                idx = np.sort(rng.choice(pool_idx, size=N, replace=False))
                jobs.append((str(N), N, k, seed, idx))
        jobs.append(("full", len(pool_idx), 0, np.nan, pool_idx.copy()))
        for N_label, N, k, seed, idx in jobs:
            sid = f"fewshot__{t['subset']}__{t['scenario_id']}__N{N_label}__k{k}"
            wd = scene_dir / f"N{N_label}_k{k}"
            P, rec, x_rec, V = fit_subset(rf["X"][idx], x_held)
            row = dict(subset=t["subset"], scenario_uid=t["scenario_uid"], scenario_id=t["scenario_id"], role=t["role"],
                       N_train=N_label, N_int=N, subset_k=k, seed=seed, n_train=len(idx), n_pool=len(pool_idx),
                       held_out_group=t["group_id"], training_uids_sha256=hashlib.sha256("|".join(rf["uid"][idx]).encode()).hexdigest(),
                       sample_id=sid, sample_dir=str(wd), **{kk: vv for kk, vv in rec.items() if kk != "z"})
            wd.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(wd / "model.npz", alpha=P.alpha, mu_weighted=P.mu, U_d=P.U[:, :D], singular_values=P.s,
                                singular_values_d=P.s[:D], V_train_d=(V if V is not None else np.full((len(idx), D), np.nan)),
                                training_case_indices=idx, training_uids=rf["uid"][idx], held_out_uid=np.array(t["scenario_uid"]),
                                held_out_case_index=t["case_index"], z=np.asarray(rec.get("z", [np.nan] * D), float),
                                x_held_raw=x_held, x_rec_raw=(x_rec if x_rec is not None else np.full(NX, np.nan)),
                                numerical_rank=rec["numerical_rank"], h_loo=rec.get("h_loo", np.nan), seed=(seed if np.isfinite(seed) else -1),
                                raw_features_sha256=np.array(rf["sha256"]))
            if x_rec is None:
                write_json(wd / "fit_failure.json", {k_: nan_to_none(v_) for k_, v_ in row.items()})
                fit_rows.append(row)
                continue
            meta = dict(N_train=N_label, n_train=int(len(idx)), subset_k=k, seed=(int(seed) if np.isfinite(seed) else None),
                        numerical_rank=rec["numerical_rank"], explained_variance_d5=rec["explained_variance_d5"], h_loo=nan_to_none(rec["h_loo"]),
                        kde_definable=rec["kde_definable"], z=rec["z"], held_out_group=t["group_id"], training_uids=rf["uid"][idx].tolist(),
                        model_npz=str(wd / "model.npz"), raw_features_npz=rf["path"], raw_features_sha256=rf["sha256"])
            rep = write_sample(wd, sid, t, ref, x_rec, meta)
            row.update(sample_json=str(wd / "sample.json"), trajectory_path=rep["trajectory_path"])
            sample_paths.append(str(wd / "sample.json"))
            fit_rows.append(row)
        ok = sum(1 for r in fit_rows if r["scenario_uid"] == t["scenario_uid"] and r.get("fit_status") == "ok")
        print(f"[fit] {t['subset']} {t['scenario_id']} ({t['role']}): pool {len(pool_idx)}, ok fits {ok}", flush=True)
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(RESULTS / "table6b_fewshot_fits.csv", index=False)
    write_json(RESULTS / "table6b_fewshot_sample_list.json", sample_paths)
    return fits


def run_scorers(sample_list, prefix, log_tag):
    cmd31 = [PY, "-B", str(PROJECT / "scripts/31_ours3_nonpet.py"), "--sample-list", str(sample_list),
             "--output-dir", str(RESULTS), "--output-prefix", f"{prefix}_nonpet", "--method", METHOD]
    env = dict(os.environ, MPLCONFIGDIR="/tmp/ours3_svd5_mpl", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with (RESULTS / f"{prefix}_31_stdout.txt").open("w") as so, (RESULTS / f"{prefix}_31_stderr.txt").open("w") as se:
        rc = subprocess.run(cmd31, stdout=so, stderr=se, cwd=PROJECT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"scripts/31 failed rc={rc}; see {prefix}_31_stderr.txt")
    if subprocess.run(["pgrep", "-f", "32_bbox_pet.py"], capture_output=True).returncode == 0:
        raise RuntimeError("another scripts/32 process is running; refusing to run PET concurrently")
    cmd32 = [PY, "-B", str(PROJECT / "scripts/32_bbox_pet.py"), "--paired", str(RESULTS / f"{prefix}_nonpet_paired.csv"),
             "--output-dir", str(RESULTS), "--output-prefix", f"{prefix}_bbox_pet", "--cache-path", str(CACHE_E8)]
    with (RESULTS / f"{prefix}_32_stdout.txt").open("w") as so, (RESULTS / f"{prefix}_32_stderr.txt").open("w") as se:
        rc = subprocess.run(cmd32, stdout=so, stderr=se, cwd=PROJECT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"scripts/32 failed rc={rc}; see {prefix}_32_stderr.txt")
    return dict(cmd31=cmd31, cmd32=cmd32)


# ── join scorer outputs → six errors per sample ─────────────────────────────
def six_errors(prefix, src, targets):
    paired = pd.read_csv(RESULTS / f"{prefix}_nonpet_paired.csv")
    paired["input_case_index"] = np.arange(len(paired))
    pet = pd.read_csv(RESULTS / f"{prefix}_bbox_pet_paired.csv")
    pet = pet[pet.input_paired_path.map(lambda p: str(Path(p).resolve())) == str((RESULTS / f"{prefix}_nonpet_paired.csv").resolve())]
    pet = pet.set_index("input_case_index")
    assert len(pet) == len(paired), (len(pet), len(paired))
    by_uid = {t["scenario_uid"]: t for t in targets}
    gt_cache = {}
    rows = []
    for r in paired.itertuples():
        p = pet.loc[r.input_case_index]
        assert str(p.scenario_uid) == str(r.scenario_uid) and str(p.sample_id) == str(r.sample_id)
        real_pet, gen_pet = float(p.real_pet), float(p.generated_pet)
        both = np.isfinite([real_pet, gen_pet]).all()
        d = dict(err_pet=abs(real_pet - gen_pet) if both else np.nan,
                 err_dmin=abs(r.generated_min_dist - r.real_min_dist),
                 err_alpha=abs(r.generated_conflict_angle - r.real_conflict_angle),
                 err_cpoint=float(np.hypot(r.generated_conflict_x - r.real_conflict_x, r.generated_conflict_y - r.real_conflict_y)),
                 err_uc=abs(r.generated_agent_arr_speed - r.real_agent_arr_speed))
        t = by_uid[r.scenario_uid]
        if r.scenario_uid not in gt_cache:
            gt_cache[r.scenario_uid] = src.ref(t)["actor"]
        gg = gt_cache[r.scenario_uid]
        if r.score_status == "ok" or r.score_status == "partial_estimator_error":
            tidy = pd.read_parquet(r.trajectory_path)
            lo, hi = float(r.metadata_min_frame), float(r.metadata_max_frame)
            tg = tidy[tidy.role.eq("target") & tidy.frame.between(lo - .5, hi + .5)].sort_values("frame")
            dtw, cov = traj_dtw(tg[["x", "y"]], gg) if len(tg) >= 2 else (np.nan, np.nan)
        else:
            dtw, cov = np.nan, np.nan
        rows.append(dict(sample_id=r.sample_id, sample_json=r.sample_json, subset=r.subset, scenario_uid=r.scenario_uid, scenario_id=str(r.scenario_id),
                         score_status=r.score_status, score_detail=r.score_detail, pet_score_status=p.pet_score_status, pet_reason=p.absdiff_pet_reason,
                         clock_aligned=bool(r.clock_aligned), real_pet=real_pet, generated_pet=gen_pet,
                         generated_no_event=bool(np.isinf(gen_pet)), pet_flip=bool(both and np.sign(real_pet) != np.sign(gen_pet)),
                         **d, err_dtw=dtw, dtw_coverage=cov, input_paired_path=str(RESULTS / f"{prefix}_nonpet_paired.csv"), input_case_index=int(r.input_case_index)))
    return pd.DataFrame(rows)


def robust_stats(values):
    v = np.asarray(values, float)
    f = v[np.isfinite(v)]
    if len(f) == 0:
        return np.nan, np.nan, np.nan, 0
    return float(np.median(f)), float(np.percentile(f, 25)), float(np.percentile(f, 75)), int(len(f))


# ── aggregation ─────────────────────────────────────────────────────────────
def phase_aggregate(targets, fits, errors):
    fits = fits.copy()
    errors = errors.copy()
    cases = fits.merge(errors.drop(columns=["subset", "scenario_uid", "scenario_id"]), on="sample_id", how="left")
    for m in MEASURES:
        cases[f"err_{m}"] = cases[f"err_{m}"].astype(float)
    cases["N_train"] = cases["N_train"].astype(str)
    # rank-insufficient rows (no fit)
    extra = []
    for t in targets:
        for N in N_RANK_INSUFFICIENT:
            extra.append(dict(subset=t["subset"], scenario_uid=t["scenario_uid"], scenario_id=t["scenario_id"], role=t["role"], N_train=str(N), N_int=N,
                              subset_k=np.nan, seed=np.nan, n_train=0, n_pool=t["n_pool"], fit_status="rank_insufficient_no_fit",
                              detail=f"rank-insufficient for d=5 (rank = N-1 = {N - 1})"))
    cases = pd.concat([pd.DataFrame(extra), cases], ignore_index=True)
    cases["N_order"] = cases.N_train.map(N_ORDER.index)
    cases = cases.sort_values(["subset", "scenario_uid", "N_order", "subset_k"]).drop(columns="N_order")
    cases.to_csv(RESULTS / "table6b_fewshot.csv", index=False)

    summary, scene_bk = [], {}
    for t in targets:
        uid = t["scenario_uid"]
        sc = cases[cases.scenario_uid.eq(uid)]
        arms = {}
        for N in N_ORDER:
            g = sc[sc.N_train.eq(N)]
            ok = g[g.fit_status.eq("ok")]
            arms[N] = {m: robust_stats(ok[f"err_{m}"])[0] for m in MEASURES}
        arms["ours3_disk"] = t["ours3_disk"]
        arms["ours3_arc"] = t["ours3_arc"]
        bk = {}
        for m in MEASURES:
            vals = [arms[a][m] for a in arms if np.isfinite(arms[a][m])]
            bk[m] = max(vals) if vals else np.nan
        scene_bk[uid] = bk

        def dm(vals):
            ratios = [vals[m] / bk[m] for m in MEASURES if np.isfinite(vals[m]) and np.isfinite(bk[m]) and bk[m] > 0]
            return float(np.mean(ratios)) if ratios else np.nan, len(ratios)

        for N in N_ORDER:
            g = sc[sc.N_train.eq(N)]
            ok = g[g.fit_status.eq("ok")]
            row = dict(subset=t["subset"], scenario_uid=uid, scenario_id=t["scenario_id"], role=t["role"], arm=f"svd_d5_fewshot_N{N}", N_train=N,
                       n_fits_attempted=int(g.fit_status.isin(["ok", "insufficient_numeric_rank", "fit_or_decode_failure"]).sum()),
                       n_fits_ok=len(ok), n_rank_failures=int(g.fit_status.eq("insufficient_numeric_rank").sum()),
                       n_pool=t["n_pool"], status=(g.fit_status.iloc[0] if len(g) and len(ok) == 0 else ("ok" if len(ok) else "missing")),
                       detail=(g.detail.iloc[0] if len(g) and len(ok) == 0 else ""))
            for m in MEASURES:
                med, q1, q3, nf = robust_stats(ok[f"err_{m}"]) if len(ok) else (np.nan, np.nan, np.nan, 0)
                row.update({f"{m}_median": med, f"{m}_q1": q1, f"{m}_q3": q3, f"{m}_n_finite": nf, f"{m}_b_k": bk[m]})
            if len(ok):
                dms = [dm({m: r[f"err_{m}"] for m in MEASURES})[0] for r in ok.to_dict("records")]
                row.update(D_m=dm(arms[N])[0], D_m_n_measures=dm(arms[N])[1], D_m_subset_median=robust_stats(dms)[0], D_m_subset_q1=robust_stats(dms)[1], D_m_subset_q3=robust_stats(dms)[2])
                row.update(explained_variance_median=robust_stats(ok.explained_variance_d5)[0], explained_variance_min=float(np.nanmin(ok.explained_variance_d5)),
                           h_loo_median=robust_stats(ok.h_loo)[0], h_loo_min=float(np.nanmin(ok.h_loo)) if ok.h_loo.notna().any() else np.nan,
                           h_loo_max=float(np.nanmax(ok.h_loo)) if ok.h_loo.notna().any() else np.nan,
                           n_kde_definable=int(ok.kde_definable.astype(bool).sum()), n_duration_clipped=int(ok.duration_clipped.astype(bool).sum()),
                           n_generated_no_event=int(ok.generated_no_event.astype(bool).sum()), n_pet_flip=int(ok.pet_flip.astype(bool).sum()),
                           n_score_failed=int((~ok.score_status.eq("ok")).sum()), n_pet_failed=int((~ok.pet_score_status.eq("ok")).sum()))
            else:
                row.update(D_m=np.nan, D_m_n_measures=0, D_m_subset_median=np.nan, D_m_subset_q1=np.nan, D_m_subset_q3=np.nan,
                           explained_variance_median=np.nan, explained_variance_min=np.nan, h_loo_median=np.nan, h_loo_min=np.nan, h_loo_max=np.nan,
                           n_kde_definable=0, n_duration_clipped=0, n_generated_no_event=0, n_pet_flip=0, n_score_failed=0, n_pet_failed=0)
            summary.append(row)
        for arm in ("ours3_disk", "ours3_arc"):
            avail = t["disk_available"] if arm == "ours3_disk" else t["arc_available"]
            row = dict(subset=t["subset"], scenario_uid=uid, scenario_id=t["scenario_id"], role=t["role"], arm=f"{arm}_default", N_train="training-free",
                       n_fits_attempted=0, n_fits_ok=int(avail), n_rank_failures=0, n_pool=t["n_pool"],
                       status="ok" if avail else "unavailable under the paper window rule (q- not reachable in the L = 10 m disk)",
                       detail="single default render, in-sample-free (no training); flat in N" + ("" if arm == "ours3_disk" else "; arc-length window = sensitivity only"))
            for m in MEASURES:
                row.update({f"{m}_median": arms[arm][m], f"{m}_q1": np.nan, f"{m}_q3": np.nan, f"{m}_n_finite": int(np.isfinite(arms[arm][m])), f"{m}_b_k": bk[m]})
            row.update(D_m=dm(arms[arm])[0], D_m_n_measures=dm(arms[arm])[1])
            summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "table6b_fewshot_summary.csv", index=False)
    return cases, summary, scene_bk


# ── optional KDE stage ──────────────────────────────────────────────────────
def phase_kde(targets, fits, src):
    RUNS_KDE.mkdir(parents=True, exist_ok=True)
    rows, sample_paths = [], []
    for t in targets:
        ref = src.ref(t)
        rf = src.raw_features(t)
        f = fits[fits.scenario_uid.eq(t["scenario_uid"]) & fits.N_train.astype(str).isin(N_KDE) & fits.fit_status.eq("ok")]
        for fr in f.itertuples():
            with np.load(Path(fr.sample_dir) / "model.npz", allow_pickle=False) as z:
                m = {k: z[k] for k in z.files}
            if not (np.isfinite(fr.h_loo) and fr.h_loo > 0):
                rows.append(dict(fit_sample_id=fr.sample_id, status="kde_undefined_no_draws", h_loo=fr.h_loo))
                continue
            P = SvdParameterization(NT, 2, 1)
            P.alpha, P.mu, P.U, P.s = m["alpha"], m["mu_weighted"], m["U_d"], m["singular_values_d"]
            seed = SEED0 + int(fr.subset_k)
            rng = np.random.default_rng(seed)
            Z, centre = sample_dependent(m["V_train_d"], float(fr.h_loo), KDE_DRAWS, rng)
            X = P.reconstruct(Z, D)
            fit_dir = RUNS_KDE / f"{t['subset']}__{t['scenario_id']}" / f"N{fr.N_train}_k{fr.subset_k}"
            fit_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fit_dir / "draws.npz", Z=Z, X=X, center_idx=centre, center_uids=m["training_uids"][centre], h_loo=fr.h_loo, seed=seed,
                                model_npz=np.array(str(Path(fr.sample_dir) / "model.npz")))
            for d in range(KDE_DRAWS):
                sid = f"{fr.sample_id}__kde{d + 1:03d}"
                meta = dict(fit_sample_id=fr.sample_id, N_train=str(fr.N_train), n_train=int(fr.n_train), subset_k=int(fr.subset_k), draw=d + 1, kde_seed=seed,
                            h_loo=float(fr.h_loo), z=Z[d].tolist(), center_idx=int(centre[d]), center_uid=str(m["training_uids"][centre[d]]),
                            held_out_group=t["group_id"], model_npz=str(Path(fr.sample_dir) / "model.npz"),
                            sampling="dependent KDE: uniform training centre + h_loo * N(0, I5) (core/kde_sampling.sample_dependent)")
                rep = write_sample(fit_dir / f"draw{d + 1:03d}", sid, t, ref, X[d], meta, method="svd_d5_fewshot_kde100")
                sample_paths.append(rep["sample_dir"] + "/sample.json")
                rows.append(dict(fit_sample_id=fr.sample_id, sample_id=sid, subset=t["subset"], scenario_uid=t["scenario_uid"], scenario_id=t["scenario_id"],
                                 N_train=str(fr.N_train), subset_k=int(fr.subset_k), draw=d + 1, kde_seed=seed, h_loo=float(fr.h_loo), status="written",
                                 duration_raw_s=float(X[d, -1]), duration_clipped=bool(X[d, -1] < DUR_MIN), center_uid=str(m["training_uids"][centre[d]])))
        print(f"[kde] {t['subset']} {t['scenario_id']}: {len(sample_paths)} draws so far", flush=True)
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "table6b_fewshot_kde_draws.csv", index=False)
    write_json(RESULTS / "table6b_fewshot_kde_sample_list.json", sample_paths)
    return draws


def phase_kde_aggregate(targets, scene_bk, src):
    draws = pd.read_csv(RESULTS / "table6b_fewshot_kde_draws.csv")
    errors = six_errors("table6b_fewshot_kde", src, targets)
    e = draws.merge(errors.drop(columns=["subset", "scenario_uid", "scenario_id"]), on="sample_id", how="left")
    out_rows = []
    for m in MEASURES:
        e[f"ratio_{m}"] = e.apply(lambda r: r[f"err_{m}"] / scene_bk[r.scenario_uid][m] if np.isfinite(r[f"err_{m}"]) and scene_bk[r.scenario_uid][m] > 0 else np.nan, axis=1)
    e["D_m_draw"] = e[[f"ratio_{m}" for m in MEASURES]].mean(axis=1, skipna=True)
    e["D_m_draw_n_measures"] = e[[f"ratio_{m}" for m in MEASURES]].notna().sum(axis=1)
    e.loc[e.D_m_draw_n_measures < len(MEASURES), "D_m_draw"] = np.nan   # oracle D_m requires all six finite
    e.to_csv(RESULTS / "table6b_fewshot_kde_cases.csv", index=False)
    per_fit = []
    for fid, g in e.groupby("fit_sample_id"):
        best = g.D_m_draw.idxmin() if g.D_m_draw.notna().any() else None
        row = dict(fit_sample_id=fid, subset=g.subset.iloc[0], scenario_uid=g.scenario_uid.iloc[0], scenario_id=g.scenario_id.iloc[0], N_train=str(g.N_train.iloc[0]),
                   subset_k=int(g.subset_k.iloc[0]), n_draws=len(g), n_draws_all_six_finite=int(g.D_m_draw.notna().sum()),
                   n_generated_no_event=int(g.generated_no_event.fillna(False).astype(bool).sum()), n_score_failed=int((~g.score_status.eq("ok")).sum()),
                   oracle_nearest_D_m=float(g.D_m_draw.min()) if best is not None else np.nan, oracle_nearest_draw=int(g.loc[best, "draw"]) if best is not None else -1,
                   D_m_draw_median=float(g.D_m_draw.median()) if g.D_m_draw.notna().any() else np.nan)
        for m in MEASURES:
            row[f"oracle_min_{m}"] = float(g[f"err_{m}"].min()) if g[f"err_{m}"].notna().any() else np.nan
            row[f"draw_median_{m}"] = float(g[f"err_{m}"].median()) if g[f"err_{m}"].notna().any() else np.nan
        per_fit.append(row)
    per_fit = pd.DataFrame(per_fit)
    per_fit.to_csv(RESULTS / "table6b_fewshot_kde_per_fit.csv", index=False)
    summ = []
    for (uid, N), g in per_fit.groupby(["scenario_uid", "N_train"]):
        t = next(t for t in targets if t["scenario_uid"] == uid)
        row = dict(subset=t["subset"], scenario_uid=uid, scenario_id=t["scenario_id"], role=t["role"], N_train=N, n_fits_with_draws=len(g),
                   n_draws=int(g.n_draws.sum()), n_draws_all_six_finite=int(g.n_draws_all_six_finite.sum()),
                   label="ORACLE best-of-100 dependent-KDE draws per fit (nearest draw to the held-out scene); not a reconstruction")
        row.update(oracle_nearest_D_m_median=robust_stats(g.oracle_nearest_D_m)[0], oracle_nearest_D_m_q1=robust_stats(g.oracle_nearest_D_m)[1], oracle_nearest_D_m_q3=robust_stats(g.oracle_nearest_D_m)[2],
                   D_m_draw_median_of_medians=robust_stats(g.D_m_draw_median)[0])
        for m in MEASURES:
            row[f"oracle_min_{m}_median"] = robust_stats(g[f"oracle_min_{m}"])[0]
            row[f"draw_median_{m}_median"] = robust_stats(g[f"draw_median_{m}"])[0]
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ["N_order"] = summ.N_train.map(N_ORDER.index)
    summ = summ.sort_values(["subset", "scenario_uid", "N_order"]).drop(columns="N_order")
    summ.to_csv(RESULTS / "table6b_fewshot_kde_summary.csv", index=False)
    return e, per_fit, summ


# ── figure ──────────────────────────────────────────────────────────────────
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]  # dataviz default categorical, validated (light)
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def phase_figure(targets, summary, kde_summary=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    finite_N = [n for n in N_ORDER if n not in map(str, N_RANK_INSUFFICIENT)]
    xpos = {n: (int(n) if n != "full" else 64) for n in finite_N}
    panels = MEASURES + ["D_m"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), constrained_layout=True)
    axes = axes.ravel()
    for pi, meas in enumerate(panels):
        ax = axes[pi]
        for ti, t in enumerate(targets):
            uid = t["scenario_uid"]
            s = summary[summary.scenario_uid.eq(uid)]
            col = f"{meas}_median" if meas != "D_m" else "D_m"
            q1c, q3c = (f"{meas}_q1", f"{meas}_q3") if meas != "D_m" else ("D_m_subset_q1", "D_m_subset_q3")
            xs, ys, lo, hi = [], [], [], []
            for n in finite_N:
                r = s[s.arm.eq(f"svd_d5_fewshot_N{n}")]
                if len(r) and np.isfinite(r[col].iloc[0]):
                    xs.append(xpos[n]); ys.append(float(r[col].iloc[0]))
                    lo.append(float(r[q1c].iloc[0]) if np.isfinite(r[q1c].iloc[0]) else float(r[col].iloc[0]))
                    hi.append(float(r[q3c].iloc[0]) if np.isfinite(r[q3c].iloc[0]) else float(r[col].iloc[0]))
            c = PALETTE[ti % len(PALETTE)]
            label = f"{t['subset']} {t['scenario_id']}" + (" (named)" if "named" in t["role"] else "")
            if xs:
                ax.fill_between(xs, lo, hi, color=c, alpha=0.12, linewidth=0)
                ax.plot(xs, ys, color=c, marker=MARKERS[ti % len(MARKERS)], ms=5, lw=1.8, label=label if pi == 0 else None)
            dk = s[s.arm.eq("ours3_disk_default")]
            if len(dk) and np.isfinite(dk[col].iloc[0]):
                ax.axhline(float(dk[col].iloc[0]), color=c, lw=1.2, ls=(0, (4, 2)))
            ar = s[s.arm.eq("ours3_arc_default")]
            if len(ar) and np.isfinite(ar[col].iloc[0]) and not (len(dk) and np.isfinite(dk[col].iloc[0])):
                ax.axhline(float(ar[col].iloc[0]), color=c, lw=1.0, ls=(0, (1, 2)))
            if kde_summary is not None and meas == "D_m":
                k = kde_summary[kde_summary.scenario_uid.eq(uid)]
                kx = [xpos[str(n)] for n in k.N_train if str(n) in xpos]
                ky = [float(v) for n, v in zip(k.N_train, k.oracle_nearest_D_m_median) if str(n) in xpos]
                if kx:
                    ax.plot(kx, ky, color=c, marker=MARKERS[ti % len(MARKERS)], ms=4, lw=0.8, ls=":", alpha=0.7, mfc="white")
        ax.set_xscale("log", base=2)
        ax.set_xticks([xpos[n] for n in finite_N])
        ax.set_xticklabels([n if n != "full" else "full" for n in finite_N])
        ax.set_title(PAPER.get(meas, "D_m (composite; lower is better)"), fontsize=10.5)
        ax.set_xlabel("N_train (held-out, LOGO pool)", fontsize=9)
        ax.grid(True, color="#e6e6e3", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.5)
    ax = axes[-1]
    ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color="#555", lw=1.2, ls=(0, (4, 2))), Line2D([], [], color="#555", lw=1.0, ls=(0, (1, 2)))]
    labels += ["ours3 disk default (training-free, flat)", "ours3 arc default (only where disk unavailable)"]
    if kde_summary is not None:
        handles.append(Line2D([], [], color="#555", lw=0.8, ls=":", marker="o", mfc="white", ms=4))
        labels.append("oracle best-of-100 KDE draws (D_m only)")
    ax.legend(handles, labels, loc="center", fontsize=8.5, frameon=False, title="held-out scene (one per class + named cases)\nline = SVD d5 few-shot median, band = IQR over 10 subsets", title_fontsize=8.5)
    fig.suptitle("Table 6b: few-shot SVD d5 held-out reconstruction vs training-free ours3 default (N = 1, 2, 4 rank-insufficient, not plotted)", fontsize=11)
    out = FIGURES / "table6b_fewshot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── report ──────────────────────────────────────────────────────────────────
def fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def phase_report(targets, selection, fits, cases, summary, kde_summary, hashes, cmds):
    L = ["# Table 6b — few-shot SVD d5 held-out reconstruction vs the training-free ours3 default", "",
         "Selection rule, training pool (LOGO), subset seeds, scoring and aggregation are in the docstring of scripts/55_fewshot.py and were fixed before any fit ran.",
         "Every SVD number here is HELD-OUT (the scene is never in the subset); Table 2 fullfit numbers are in-sample. ours3 disk default = single training-free render of the same scene (flat in N); it is the reference line, not a fit.",
         "N in {1, 2, 4} cannot support d = 5 (centred rank = N-1) and are listed without a fit. N > pool is 'not feasible'. Rank failures are reported, never retried.",
         f"Scoring: unchanged scripts/31 (--method {METHOD}) and scripts/32 (cache results/bbox_pet_cache_e8.json, one run per stage); DTW = traj_dtw of exp_cross_coverage/scripts/180 (AST-loaded; sha {DTW_HASHES['traj_dtw'][:12]}).", ""]
    L += ["## Held-out scenes", "", "| class | scene | role | group | pool (LOGO) | group size in class | disk default | arc default |", "|---|---|---|---|---:|---:|---|---|"]
    for t in targets:
        L.append(f"| {t['subset']} | {t['scenario_id']} | {t['role']} — {t['note']} | {t['group_id']} | {t['n_pool']} | {t['n_group_in_class']} | "
                 f"{'available' if t['disk_available'] else 'UNAVAILABLE under the paper window rule'} | {'available (sensitivity)' if t['arc_available'] else 'n/a'} |")
    L += ["", "Real descriptors of the selected scenes and class medians are in results/table6b_fewshot_selection.csv (top-5 candidates per class).", ""]
    L += ["## Fits", ""]
    fs = fits.groupby(["subset", "scenario_id", "N_train"]).fit_status.value_counts().unstack(fill_value=0)
    L += ["```", fs.to_string(), "```", ""]
    L += ["## Per scene: six measures and D_m vs N (median over subsets; IQR in the CSV)", ""]
    for t in targets:
        uid = t["scenario_uid"]
        s = summary[summary.scenario_uid.eq(uid)]
        L += [f"### {t['subset']} {t['scenario_id']} ({t['role']}) — pool {t['n_pool']}", "",
              "| arm | fits ok / rank fail | " + " | ".join(PAPER[m] for m in MEASURES) + " | D_m | D_m subset IQR | EV(d5) med | h_loo med [min, max] | KDE definable | gen no-event |",
              "|---|---|" + "---:|" * (len(MEASURES) + 6)]
        for r in s.to_dict("records"):
            if r["arm"].startswith("svd") and r["status"] not in ("ok",):
                L.append(f"| N={r['N_train']} | — | {r['detail'] or r['status']} |" + " |" * (len(MEASURES) + 5))
                continue
            cells = " | ".join(fmt(r[f"{m}_median"]) for m in MEASURES)
            if r["arm"].startswith("svd"):
                L.append(f"| N={r['N_train']} | {r['n_fits_ok']} / {r['n_rank_failures']} | {cells} | {fmt(r['D_m'])} | [{fmt(r['D_m_subset_q1'])}, {fmt(r['D_m_subset_q3'])}] | "
                         f"{fmt(r['explained_variance_median'])} | {fmt(r['h_loo_median'], 4)} [{fmt(r['h_loo_min'], 4)}, {fmt(r['h_loo_max'], 4)}] | {r['n_kde_definable']}/{r['n_fits_ok']} | {r['n_generated_no_event']} |")
            else:
                L.append(f"| {r['arm']} | {r['status'] if r['status'] != 'ok' else 'training-free'} | {cells} | {fmt(r['D_m'])} | — | — | — | — | — |")
        L.append("")
        # plain statements
        disk = s[s.arm.eq("ours3_disk_default")].iloc[0]
        arc = s[s.arm.eq("ours3_arc_default")].iloc[0]
        ref, refname = (disk, "ours3 disk") if t["disk_available"] else (arc, "ours3 arc (sensitivity; disk unavailable)")
        stmts = []
        for m in MEASURES + ["D_m"]:
            col = f"{m}_median" if m != "D_m" else "D_m"
            rv = ref[col]
            reached = [r["N_train"] for r in s[s.arm.str.startswith("svd") & s.status.eq("ok")].to_dict("records") if np.isfinite(r[col]) and np.isfinite(rv) and r[col] <= rv]
            name = PAPER.get(m, "D_m")
            if not np.isfinite(rv):
                stmts.append(f"- {name}: {refname} value not finite (no-event/unavailable); no comparison.")
            elif reached:
                stmts.append(f"- {name}: SVD d5 held-out median reaches the {refname} value ({fmt(rv)}) at N = {', '.join(reached)}.")
            else:
                stmts.append(f"- {name}: SVD d5 held-out median never reaches the {refname} value ({fmt(rv)}) at any N (incl. full pool {t['n_pool']}).")
        L += stmts + [""]
        if kde_summary is not None:
            k = kde_summary[kde_summary.scenario_uid.eq(uid)]
            if len(k):
                L += ["ORACLE best-of-100 dependent-KDE draws (nearest draw to the held-out scene; NOT a reconstruction, selection uses the held-out answer):", ""]
                for r in k.to_dict("records"):
                    L.append(f"- N={r['N_train']}: nearest-draw D_m median over fits {fmt(r['oracle_nearest_D_m_median'])} [IQR {fmt(r['oracle_nearest_D_m_q1'])}, {fmt(r['oracle_nearest_D_m_q3'])}]; "
                             f"median-draw D_m {fmt(r['D_m_draw_median_of_medians'])}; draws with all six finite {r['n_draws_all_six_finite']}/{r['n_draws']}")
                L.append("")
    L += ["## KDE definability vs N", ""]
    kd = summary[summary.arm.str.startswith("svd") & summary.status.eq("ok")]
    L += ["| N | fits ok | KDE definable (h finite, > 0) | h_loo median (min–max over scenes) |", "|---|---:|---:|---|"]
    for N in N_ORDER:
        g = kd[kd.N_train.eq(N)]
        if not len(g):
            L.append(f"| {N} | 0 | — | rank-insufficient / not feasible |")
            continue
        L.append(f"| {N} | {int(g.n_fits_ok.sum())} | {int(g.n_kde_definable.sum())} | {fmt(g.h_loo_median.median(), 4)} ({fmt(g.h_loo_min.min(), 4)}–{fmt(g.h_loo_max.max(), 4)}) |")
    L += ["", "Note: loo_bandwidth always returns a positive number on any N >= 2 grid; definability here is the existence of a finite, positive LOO-CV bandwidth on V_train_d, not evidence that the KDE is a useful density at that N.",
          "Finding: at N = 6 = d + 1 the five reduced coordinates of the six training scenes always form a regular simplex (V has orthonormal, zero-mean columns), so every N = 6 subset returns the same LOO bandwidth h = sqrt(2/5) = 0.632456 regardless of which scenes were drawn, and the explained variance is trivially 1.0; the N = 6 KDE is 'definable' but carries no data-dependent bandwidth. At N = 8 (rank 7, d = 5) h is already data-dependent (0.44-0.47).", ""]
    L += ["## Deviations / notes", "",
          "- Pseudo-samples follow scripts/50: the decoded 50-point path enters scripts/31 as a trace with derived kinematics stored as the recorded heading/speed (identical values to the scripts/30 SVD branch); PET in scripts/32 therefore takes the trace branch, same points, same heading — not the reconstruction_npz branch.",
          "- 'full' is one deterministic fit on the whole LOGO pool (no subsets); its IQR is empty by construction.",
          "- cutinr pool is smaller than 32, so N = 32 is 'not feasible' there and 'full' is the whole pool.",
          "- 230_179 (keeptl corner case) has no disk render; its reference line is the arc-window ours3 value and is sensitivity only.",
          "- D_m is scene-wise (b_k over the arm set of that scene) and is not comparable across scenes or with the Table 2 class D_m.",
          "- Held-out PET no-event decodes give a NaN |dPET| for that subset; medians use the finite subsets only (count in *_n_finite).",
          "- Oracle KDE rows: D_m of a draw is defined only when all six errors are finite; 4228 of 13700 draws are PET no-event and are excluded from the nearest-draw search (counts per row in n_draws_all_six_finite). The nearest draw is chosen with the held-out answer, so it is an upper bound on what any sampler could reach, not a method result.",
          "- tlkeep 90_110: |d alpha| jumps to ~78 deg for N >= 16 while N = 8 gives ~5 deg; the larger-N decodes place the closest approach on a differently oriented segment. Reported as is, not investigated here.",
          f"- HANDOFF hash check before: {hashes['before']['n_lines']} lines, failures {hashes['before']['failures']}; after: {hashes['after']['n_lines']} lines, failures {hashes['after']['failures']}.",
          "- results/bbox_pet_cache.json was not used or touched; PET cache for this table is results/bbox_pet_cache_e8.json.", ""]
    (RESULTS / "TABLE6B_FEWSHOT_REPORT.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="all", choices=["all", "fit", "score", "aggregate", "kde", "kde_score", "kde_aggregate", "report"])
    ap.add_argument("--skip-kde", action="store_true")
    args = ap.parse_args()
    t_start = time.monotonic()
    hashes = dict(before=hash_check())
    if hashes["before"]["failures"]:
        raise RuntimeError(f"HANDOFF hash check failed before run: {hashes['before']['failures']}")
    targets, selection = select_targets()
    selection.to_csv(RESULTS / "table6b_fewshot_selection.csv", index=False)
    write_json(RESULTS / "table6b_fewshot_targets.json", [{k: (v if k not in ("ours3_disk", "ours3_arc", "real") else {kk: nan_to_none(vv) for kk, vv in v.items()}) for k, v in t.items()} for t in targets])
    src = ClassSource()
    cmds = {}
    phases = [args.phase] if args.phase != "all" else ["fit", "score", "aggregate"] + ([] if args.skip_kde else ["kde", "kde_score", "kde_aggregate"]) + ["report"]
    fits = phase_fit(targets, src) if "fit" in phases else pd.read_csv(RESULTS / "table6b_fewshot_fits.csv", dtype={"N_train": str})
    if "score" in phases:
        cmds["main"] = run_scorers(RESULTS / "table6b_fewshot_sample_list.json", "table6b_fewshot", "main")
    if "kde" in phases:
        phase_kde(targets, fits, src)
    if not (RESULTS / "table6b_fewshot_bbox_pet_paired.csv").exists():
        print(f"[fit] {len(fits)} fit rows; main-stage scorer outputs absent, stopping before aggregation", flush=True)
        return
    errors = six_errors("table6b_fewshot", src, targets)
    cases, summary, scene_bk = phase_aggregate(targets, fits, errors)
    kde_summary = None
    if "kde_score" in phases:
        cmds["kde"] = run_scorers(RESULTS / "table6b_fewshot_kde_sample_list.json", "table6b_fewshot_kde", "kde")
    if ("kde_aggregate" in phases or "report" in phases) and (RESULTS / "table6b_fewshot_kde_bbox_pet_paired.csv").exists():
        _, _, kde_summary = phase_kde_aggregate(targets, scene_bk, src)
    fig = phase_figure(targets, summary, kde_summary)
    hashes["after"] = hash_check()
    phase_report(targets, selection, fits, cases, summary, kde_summary, hashes, cmds)
    outputs = [RESULTS / n for n in ("table6b_fewshot.csv", "table6b_fewshot_summary.csv", "table6b_fewshot_fits.csv", "table6b_fewshot_selection.csv",
                                     "table6b_fewshot_targets.json", "table6b_fewshot_nonpet_paired.csv", "table6b_fewshot_bbox_pet_paired.csv",
                                     "TABLE6B_FEWSHOT_REPORT.md", "table6b_fewshot_kde_summary.csv", "table6b_fewshot_kde_per_fit.csv", "table6b_fewshot_kde_cases.csv")] + [fig]
    manifest = dict(status="TABLE6B_FEWSHOT_COMPLETE" if kde_summary is not None else "TABLE6B_FEWSHOT_MAIN_COMPLETE_KDE_STAGE_ABSENT",
                    selection_rule="see scripts/55_fewshot.py docstring (fixed before fits)", targets=[{k: v for k, v in t.items() if k not in ("pool_uids", "pool_case_indices", "ours3_disk", "ours3_arc", "real")} for t in targets],
                    N_train=dict(rank_insufficient_no_fit=N_RANK_INSUFFICIENT, finite=N_FINITE, full="whole LOGO pool", subsets_per_N=N_SUBSETS, seeds=[SEED0 + k for k in range(N_SUBSETS)]),
                    d=D, nt=NT, nx=NX, fps=FPS, duration_rule=f"applied = max(raw, {DUR_MIN}); raw kept; clipped flagged", method_label=METHOD,
                    kde_stage=dict(N=N_KDE, draws_per_fit=KDE_DRAWS, sampler="core/kde_sampling.sample_dependent", h="h_loo of the fit", rng="default_rng(20260910 + k)", label="oracle best-of-100; not a reconstruction") if kde_summary is not None else "not run",
                    scoring_commands=cmds, pet_cache=str(CACHE_E8), pet_cache_main_untouched=True,
                    sources={str(p): sha256(p) for p in (CASES_TABLE2, MATCHED_CASES, CASES_513, DTW_SOURCE, SVD_SOURCE, KDE_SOURCE, Path(__file__).resolve(),
                                                          PROJECT / "scripts/31_ours3_nonpet.py", PROJECT / "scripts/32_bbox_pet.py", PROJECT / "scripts/30_interaction_metrics.py")},
                    raw_features={p: v["sha256"] for p, v in ((str(k), v) for k, v in src.raw.items())},
                    dtw_function_hashes=DTW_HASHES, hash_check=hashes,
                    outputs=[dict(path=str(p), sha256=sha256(p), rows=(sum(1 for _ in open(p)) - 1 if p.suffix == ".csv" else None)) for p in outputs if p.exists()],
                    environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, threads=1),
                    elapsed_s=time.monotonic() - t_start)
    write_json(RESULTS / "table6b_manifest.json", manifest)
    print(summary[summary.arm.str.startswith("svd") | summary.arm.eq("ours3_disk_default")][["subset", "scenario_id", "arm", "n_fits_ok", "n_rank_failures", "D_m", "status"]].to_string(index=False))
    print(f"elapsed {time.monotonic() - t_start:.0f}s; hash failures after: {hashes['after']['failures']}")


if __name__ == "__main__":
    main()
