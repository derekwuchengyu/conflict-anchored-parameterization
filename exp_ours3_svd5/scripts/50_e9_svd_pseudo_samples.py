#!/usr/bin/env python3
"""E9 (39_180 singleton table T8): analytic SVD-side arms written as pseudo-samples.

Arms (every sample.json carries "class_data_used"):
  (iii) svd_d5 class-basis decodes of 39_180           class_data_used = basis
        513-pool fullfit, matched fullfit, matched LOGO (the LOGO model excludes
        39_180's global ego/target group g_a892cf18c80191f4).
  (iv)  APPROVED external-basis Gaussian (E0 item 5)   class_data_used = basis+bandwidth
        z0 = encode(39_180 raw 101-D alpha-weighted vector) into the matched
        cutinl LOGO basis that EXCLUDES its group; draws z0 + h_ext N(0, I5),
        h_ext in {loo_bandwidth(V_train_d of that LOGO model), 2x}; decoded with
        the same basis. 100 draws x seeds {20260910, 20260911, 20260912} per h.
  (v)   APPROVED raw-vector Gaussian jitter null       class_data_used = none
        raw 101-D vector (50 xy pairs + duration) + N(0, sigma_xy^2) i.i.d. per
        coordinate, sigma_xy in {0.5, 1.0, 2.0} m, and N(0, 0.5^2) s on the
        duration. 100 x 3 seeds per sigma. No basis, no class data.

Decode convention = scripts/30_interaction_metrics.py: xy = vector[:100] as 50
time-major pairs, frame = 2424 + linspace(0, duration, 50) * 30, kinematics
derived from xy. Duration policy = the existing SVD KDE decode: applied =
max(raw, 0.5) s, raw kept, duration_clipped flagged. Nothing is simulated; every
sample.json says "analytic": true. Ego rows are the recorded ego 39 track.

These arms are NOT learned from the one scenario: (iii)/(iv) use the cutinl
class basis (and (iv) a class-derived bandwidth); (v) uses an externally chosen
sigma. The singleton fit itself (N=1) is rank 0 and is reported separately from
singleton_diagnostic/results/diagnostic.json.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / "hetero-param"))
from hetero_param.similarity import core as SIMC  # noqa: E402

KDE_SOURCE = ROOT / "sr-tlkeep-experiment/core/kde_sampling.py"
SVD_SOURCE = ROOT / "sr-tlkeep-experiment/core/svd_param.py"
REAL_TRACKS = ROOT / "exp_cross_coverage/data/special_39_180/real_tracks.parquet"
RAW_FEATURES = PROJECT / "models/svd_d5/cutinl/raw_features.npz"
REC_513 = PROJECT / "models/svd_d5/cutinl/reconstructions.npz"
REC_MATCHED = PROJECT / "models/svd_d5_matched/cutinl/reconstructions.npz"
LOGO_MODEL = PROJECT / "models/svd_d5_matched/cutinl/cutinl__logo__g_a892cf18c80191f4.npz"
FULL_MATCHED_MODEL = PROJECT / "models/svd_d5_matched/cutinl/cutinl__fullfit.npz"
FULL_513_MODEL = PROJECT / "models/svd_d5/cutinl/cutinl__fullfit.npz"
DIAG = PROJECT / "singleton_diagnostic/results/diagnostic.json"
OUT = PROJECT / "runs"
SCENARIO = dict(scenario_id="39_180", dataset="HetroD", recording="00", ego=39, target=180,
                min_frame=2424, max_frame=3002, target_gt_end=2923, fps=30.0,
                scenario_uid="HetroD/00/39_180/2424-3002", subset="special_39_180",
                actor_start_frame=2424, ego_end_frame=3002, basis_class="cutinl")
SEEDS = [20260910, 20260911, 20260912]
N_DRAWS = 100
NT = 50
DUR_MIN = 0.5
SIGMA_DUR = 0.5


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def ast_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), (path, names)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        m = {k: z[k] for k in z.files}
    m["path"] = str(path)
    m["sha256"] = sha256(path)
    return m


def encode(model, x_raw):
    """Least-squares coefficients in the model's d-dim basis (U_d orthonormal)."""
    xw = np.asarray(x_raw, float) * model["alpha"]
    return ((xw - model["mu_weighted"]) @ model["U_d"]) / model["singular_values_d"]


def decode(model, z):
    z = np.atleast_2d(np.asarray(z, float))
    xw = model["mu_weighted"] + (z * model["singular_values_d"]) @ model["U_d"].T
    return xw / model["alpha"]


def tidy_from_vector(vector, sample_id, subset, ego_rows, duration_applied):
    xy = np.asarray(vector[:NT * 2], float).reshape(NT, 2)
    fps, lo = SCENARIO["fps"], SCENARIO["min_frame"]
    a0 = SCENARIO.get("actor_start_frame", lo)
    frame = a0 + np.linspace(0.0, duration_applied, NT) * fps
    heading, speed, accel = SIMC.derive_kinematics(frame, xy[:, 0], xy[:, 1], fps)
    vx, vy = speed * np.cos(np.radians(heading)), speed * np.sin(np.radians(heading))
    t = (frame - lo) / fps
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    length, width = float(ego_rows["target_length"]), float(ego_rows["target_width"])
    target = pd.DataFrame(dict(sample_id=sample_id, scenario_id=SCENARIO["scenario_id"], subset=subset,
        entity_name="Agent1", actor_id=str(SCENARIO["target"]), role="target", time_s=t,
        x=xy[:, 0], y=xy[:, 1], z=0.0, heading_deg=heading, speed_mps=speed, length=length, width=width,
        vx_mps=vx, vy_mps=vy, ax_mps2=ax, ay_mps2=ay, frame=frame))
    ego = ego_rows["ego"].copy()
    ego.insert(0, "sample_id", sample_id)
    ego["subset"] = subset
    out = pd.concat([target, ego[target.columns]], ignore_index=True)
    hi, gt_end = SCENARIO["max_frame"], SCENARIO["target_gt_end"]
    out["within_metadata_window"] = out.frame.between(lo - .5, hi + .5)
    out["within_target_gt_support"] = out.frame.between(lo - .5, gt_end + .5)
    ent_hi = np.where(out.role.eq("target"), gt_end, hi)
    out["within_entity_gt_support"] = (out.frame >= lo - .5) & (out.frame <= ent_hi + .5)
    return out


def ego_reference():
    tracks = pd.read_parquet(REAL_TRACKS)
    sid = SCENARIO["scenario_id"]
    e = tracks[tracks.role.eq("ego") & tracks.scenario_id.eq(sid)].sort_values("frame")
    a = tracks[tracks.role.eq("actor") & tracks.scenario_id.eq(sid)].sort_values("frame")
    assert e.track_id.eq(SCENARIO["ego"]).all() and a.track_id.eq(SCENARIO["target"]).all()
    assert int(e.frame.min()) == SCENARIO["min_frame"] and int(e.frame.max()) == SCENARIO["ego_end_frame"]
    assert int(a.frame.min()) == SCENARIO.get("actor_start_frame", SCENARIO["min_frame"])
    assert int(a.frame.max()) == SCENARIO["target_gt_end"]
    fps, lo = SCENARIO["fps"], SCENARIO["min_frame"]
    h = e.heading_deg.to_numpy(float)
    sp = e.speed.to_numpy(float)
    t = (e.frame.to_numpy(float) - lo) / fps
    vx, vy = sp * np.cos(np.radians(h)), sp * np.sin(np.radians(h))
    ego = pd.DataFrame(dict(scenario_id=sid, entity_name="Ego", actor_id=str(SCENARIO["ego"]), role="ego", time_s=t,
        x=e.x.to_numpy(float), y=e.y.to_numpy(float), z=0.0, heading_deg=h, speed_mps=sp,
        length=float(e.length.median()), width=float(e.width.median()), vx_mps=vx, vy_mps=vy,
        ax_mps2=np.gradient(vx, t), ay_mps2=np.gradient(vy, t), frame=e.frame.to_numpy(float)))
    return dict(ego=ego, target_length=float(a.length.median()), target_width=float(a.width.median()),
                actor_xy=a[["x", "y"]].to_numpy(float), actor_frame=a.frame.to_numpy(float))


def write_sample(batch_dir, sample_id, vector_raw, vector_applied, tidy, meta, arm, label, method):
    wd = batch_dir / sample_id
    wd.mkdir(parents=True, exist_ok=True)
    traj = wd / "trajectory.parquet"
    tidy.to_parquet(traj, index=False)
    sub = SCENARIO["subset"]
    job = dict(job_id=sample_id, scenario_id=SCENARIO["scenario_id"], **{"class": sub},
               dataset=SCENARIO["dataset"], recording=SCENARIO["recording"], ego=SCENARIO["ego"],
               target=SCENARIO["target"], min_frame=SCENARIO["min_frame"], max_frame=SCENARIO["max_frame"],
               geometry_variant_id=f"{sub}_svd_{arm}",
               source_context=dict(subset=sub, source_tracks=str(REAL_TRACKS),
                                   scenario_uid=SCENARIO["scenario_uid"]))
    context = dict(scenario_id=SCENARIO["scenario_id"], dataset=SCENARIO["dataset"], recording=SCENARIO["recording"],
                   subset=sub, ego=SCENARIO["ego"], target=SCENARIO["target"],
                   metadata_window_frames=[SCENARIO["min_frame"], SCENARIO["max_frame"]],
                   target_gt_support_frames=[SCENARIO.get("actor_start_frame", SCENARIO["min_frame"]),
                                             SCENARIO["target_gt_end"]],
                   fps=SCENARIO["fps"], geometry_variant_id=job["geometry_variant_id"],
                   source_context=job["source_context"])
    report = dict(sample_id=sample_id, sample_dir=str(wd), job=job, context=context, status="completed",
                  analytic=True, simulated=False, method=method, arm=arm, class_data_used=label,
                  learned_from_one_scenario=False,
                  label_sentence=(f"Analytic SVD decode; uses the {SCENARIO['basis_class']} class basis" if label != "none"
                                  else "Raw-vector jitter null; no basis and no class data, sigma chosen externally"),
                  parameters_applied={}, is_nominal_default=bool(meta.pop("is_nominal_default", False)),
                  decode=dict(convention=f"scripts/30_interaction_metrics.py: 50 time-major xy pairs, frame = "
                                         f"{SCENARIO.get('actor_start_frame', SCENARIO['min_frame'])} + "
                                         f"linspace(0, duration, 50) * {SCENARIO['fps']:g}, kinematics derived from xy",
                              duration_raw_s=float(vector_raw[-1]), duration_applied_s=float(vector_applied[-1]),
                              duration_clipped=bool(vector_raw[-1] < DUR_MIN), duration_rule=f"max(raw, {DUR_MIN})"),
                  vector_raw=np.asarray(vector_raw, float).tolist(),
                  trajectory_path=str(traj), trajectory_rows=len(tidy), **meta)
    report["artifacts"] = [dict(path=str(traj), sha256=sha256(traj), size_bytes=traj.stat().st_size)]
    write_json(wd / "sample.json", report)
    return report


def main():
    ns = {"np": np}
    ast_functions(KDE_SOURCE, ["loo_log_likelihood", "loo_bandwidth"], ns)
    loo_bandwidth = ns["loo_bandwidth"]
    diag = json.loads(DIAG.read_text())
    ref = ego_reference()
    rf = load_model(RAW_FEATURES)
    i = list(map(str, rf["scenario_id"])).index("39_180")
    x_raw = np.asarray(rf["X_raw"][i], float)
    assert x_raw.shape == (101,)
    # the raw vector must be the actor track resampled over its own support
    tq = np.linspace(0, ref["actor_frame"][-1] - ref["actor_frame"][0], NT) / SCENARIO["fps"]
    ta = (ref["actor_frame"] - ref["actor_frame"][0]) / SCENARIO["fps"]
    rebuilt = np.column_stack([np.interp(tq, ta, ref["actor_xy"][:, 0]), np.interp(tq, ta, ref["actor_xy"][:, 1])]).reshape(-1)
    assert np.allclose(rebuilt, x_raw[:100], atol=1e-6) and abs(x_raw[-1] - ta[-1]) < 1e-9
    rec513 = load_model(REC_513)
    recm = load_model(REC_MATCHED)
    j = list(map(str, rec513["scenario_id"])).index("39_180")
    k = list(map(str, recm["scenario_id"])).index("39_180")
    assert np.array_equal(rec513["X_raw"][j], x_raw) and np.array_equal(recm["X_raw"][k], x_raw)
    assert str(recm["group_id"][k]) == "g_a892cf18c80191f4"
    logo = load_model(LOGO_MODEL)
    assert "HetroD/00/39_180/2424-3002" not in set(map(str, logo["training_uids"]))
    assert "HetroD/00/39_180/2424-3002" in set(map(str, logo["evaluation_uids"]))
    assert str(logo["excluded_group"]) == "g_a892cf18c80191f4" and int(logo["numerical_rank"]) >= 5
    fullm, full513 = load_model(FULL_MATCHED_MODEL), load_model(FULL_513_MODEL)
    z0 = encode(logo, x_raw)
    assert np.allclose(decode(logo, z0)[0], recm["X_rec_logo"][k], atol=1e-8), "LOGO encode/decode does not reproduce the saved LOGO row"
    assert np.allclose(decode(fullm, encode(fullm, x_raw))[0], recm["X_rec_fullfit"][k], atol=1e-8)
    assert np.allclose(decode(full513, encode(full513, x_raw))[0], rec513["X_rec_fullfit"][j], atol=1e-8)
    V = np.asarray(logo["V_train_d"], float)
    h_loo = float(loo_bandwidth(V, n_grid=60))
    assert np.isfinite(h_loo) and h_loo > 0
    print(f"[e9 svd] z0 = {np.round(z0, 5).tolist()}  h_loo(LOGO V_train_d, N={len(V)}) = {h_loo:.6g}", flush=True)
    manifest = dict(scenario=SCENARIO, seeds=SEEDS, n_draws_per_seed=N_DRAWS, z0=z0.tolist(), h_loo_logo=h_loo,
                    logo_model=dict(path=logo["path"], sha256=logo["sha256"], n_train=int(len(V)),
                                    excluded_group=str(logo["excluded_group"]), numerical_rank=int(logo["numerical_rank"])),
                    sources={p.name: dict(path=str(p), sha256=sha256(p)) for p in
                             (RAW_FEATURES, REC_513, REC_MATCHED, LOGO_MODEL, FULL_MATCHED_MODEL, FULL_513_MODEL,
                              REAL_TRACKS, KDE_SOURCE, SVD_SOURCE, DIAG, Path(__file__))},
                    singleton_fit=dict(status=diag["status"], centered_matrix_rank=diag["centered_matrix_rank"],
                                       kde_bandwidth=diag["kde_bandwidth"], kde_bandwidth_error=diag["kde_bandwidth_error"],
                                       n_samples_generated=0, class_data_used="none"),
                    batches={})
    # (iii) plain decodes
    decodes = [("pool513_fullfit", rec513["X_rec_fullfit"][j], "basis", REC_513, "X_rec_fullfit", j, 61),
               ("matched_fullfit", recm["X_rec_fullfit"][k], "basis", REC_MATCHED, "X_rec_fullfit", k, 48),
               ("matched_logo", recm["X_rec_logo"][k], "basis", REC_MATCHED, "X_rec_logo", k, 44)]
    for arm, vec, label, npz, key, idx, ntrain in decodes:
        batch = OUT / f"svd_39_180_e9_{arm}"
        batch.mkdir(parents=True, exist_ok=True)
        applied = vec.copy()
        applied[-1] = max(float(vec[-1]), DUR_MIN)
        tidy = tidy_from_vector(applied, f"svd_{arm}", "special_39_180", ref, float(applied[-1]))
        meta = dict(reconstruction_npz=str(npz), reconstruction_array_key=key, case_index=int(idx), n_train=ntrain,
                    is_nominal_default=True, seed=None, draw=None, sigma_or_h=None)
        write_sample(batch, f"svd_{arm}", vec, applied, tidy, meta, arm, label, f"svd_d5_{arm}")
        write_json(batch / "protocol.json", dict(arm=arm, class_data_used=label, analytic=True, n=1,
                                                  source_npz=str(npz), source_sha256=sha256(npz), array_key=key, row=int(idx)))
        manifest["batches"][arm] = dict(batch_dir=str(batch), n=1, class_data_used=label)
    # (iv) external-basis Gaussian in the LOGO basis
    for h_name, h in (("h1", h_loo), ("h2", 2 * h_loo)):
        arm = f"extbasis_gauss_{h_name}"
        batch = OUT / f"svd_39_180_e9_{arm}"
        batch.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            noise = rng.standard_normal((N_DRAWS, 5))
            Z = z0 + h * noise
            X = decode(logo, Z)
            for d in range(N_DRAWS):
                vec = X[d]
                applied = vec.copy()
                applied[-1] = max(float(vec[-1]), DUR_MIN)
                sid = f"{arm}__seed{seed}__draw{d + 1:03d}"
                tidy = tidy_from_vector(applied, sid, "special_39_180", ref, float(applied[-1]))
                meta = dict(seed=seed, draw=d + 1, sigma_or_h=float(h), h_ext=float(h), h_loo=h_loo, h_multiple=1 if h_name == "h1" else 2,
                            z=Z[d].tolist(), z0=z0.tolist(), basis_model=logo["path"], basis_sha256=logo["sha256"],
                            basis_excludes_39_180_group=True, basis_n_train=int(len(V)),
                            rng="numpy default_rng(seed); the h1 and h2 arms share the same standard-normal draws per seed")
                write_sample(batch, sid, vec, applied, tidy, meta, arm, "basis+bandwidth", f"svd_extbasis_gauss_{h_name}")
                rows.append(dict(sample_id=sid, seed=seed, draw=d + 1, duration_raw=float(vec[-1]), duration_applied=float(applied[-1]),
                                 duration_clipped=bool(vec[-1] < DUR_MIN)))
        pd.DataFrame(rows).to_csv(batch / "draws.csv", index=False)
        write_json(batch / "protocol.json", dict(arm=arm, class_data_used="basis+bandwidth", analytic=True, n=len(rows),
                                                  h_ext=h, h_loo=h_loo, seeds=SEEDS, basis=logo["path"], basis_sha256=logo["sha256"],
                                                  duration_clipped_n=int(sum(r["duration_clipped"] for r in rows))))
        manifest["batches"][arm] = dict(batch_dir=str(batch), n=len(rows), class_data_used="basis+bandwidth", h_ext=h,
                                        duration_clipped_n=int(sum(r["duration_clipped"] for r in rows)))
        print(f"[e9 svd] {arm}: {len(rows)} draws, h={h:.5g}, clipped={manifest['batches'][arm]['duration_clipped_n']}", flush=True)
    # (v) raw-vector jitter null
    for sigma in (0.5, 1.0, 2.0):
        arm = f"rawjitter_s{sigma:g}"
        batch = OUT / f"svd_39_180_e9_{arm}"
        batch.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            noise = rng.standard_normal((N_DRAWS, 101))
            X = x_raw + noise * np.r_[np.full(100, sigma), SIGMA_DUR]
            for d in range(N_DRAWS):
                vec = X[d]
                applied = vec.copy()
                applied[-1] = max(float(vec[-1]), DUR_MIN)
                sid = f"{arm}__seed{seed}__draw{d + 1:03d}"
                tidy = tidy_from_vector(applied, sid, "special_39_180", ref, float(applied[-1]))
                meta = dict(seed=seed, draw=d + 1, sigma_or_h=float(sigma), sigma_xy_m=float(sigma), sigma_duration_s=SIGMA_DUR,
                            rng="numpy default_rng(seed); the three sigma arms share the same standard-normal draws per seed")
                write_sample(batch, sid, vec, applied, tidy, meta, arm, "none", f"svd_rawjitter_s{sigma:g}")
                rows.append(dict(sample_id=sid, seed=seed, draw=d + 1, duration_raw=float(vec[-1]), duration_applied=float(applied[-1]),
                                 duration_clipped=bool(vec[-1] < DUR_MIN)))
        pd.DataFrame(rows).to_csv(batch / "draws.csv", index=False)
        write_json(batch / "protocol.json", dict(arm=arm, class_data_used="none", analytic=True, n=len(rows), sigma_xy_m=sigma,
                                                  sigma_duration_s=SIGMA_DUR, seeds=SEEDS,
                                                  duration_clipped_n=int(sum(r["duration_clipped"] for r in rows))))
        manifest["batches"][arm] = dict(batch_dir=str(batch), n=len(rows), class_data_used="none", sigma_xy_m=sigma,
                                        duration_clipped_n=int(sum(r["duration_clipped"] for r in rows)))
        print(f"[e9 svd] {arm}: {len(rows)} draws, clipped={manifest['batches'][arm]['duration_clipped_n']}", flush=True)
    out = PROJECT / "results/e9_39_180"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "svd_arms_manifest.json", manifest)
    print(json.dumps({k: v["n"] for k, v in manifest["batches"].items()}))


if __name__ == "__main__":
    main()
