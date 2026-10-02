#!/usr/bin/env python3
"""Uturn 859_881 singleton table (T8-uturn): SVD-side arms as analytic pseudo-samples.

Same recipe as scripts/50_e9_svd_pseudo_samples.py, with the one difference the
scenario forces: 859_881 belongs to NO fitted class, so there is no in-sample
("matched fullfit") decode to print.  Every SVD row is therefore an EXTERNAL
basis, and the donor is the matched keeptl cohort (50 scenes, rank 5); the
recording's own uid is absent from its training set by construction, which is
verified here.

  svd_extbasis_recon        decode of the recorded 101-D vector in the keeptl
                            basis -> the row's similarity cells.
  svd_extbasis_gauss_h1/h2  z0 + h N(0, I5) decoded in the same basis,
                            h = loo_bandwidth(V_train_d) and 2x, 100 draws x
                            seeds {20260910, 20260911, 20260912}.
  svd_d5_singleton_N1       stated, not produced: N = 1 -> centred rank 0.

The helper functions (encode/decode/tidy/write_sample) are AST-loaded from
scripts/50 so both scenarios share one code path.
"""
from __future__ import annotations

import argparse
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

SRC50 = PROJECT / "scripts/50_e9_svd_pseudo_samples.py"
KDE_SOURCE = ROOT / "sr-tlkeep-experiment/core/kde_sampling.py"
SVD_SOURCE = ROOT / "sr-tlkeep-experiment/core/svd_param.py"
REAL_TRACKS = ROOT / "exp_cross_coverage/data/uturn_859_881/real_tracks.parquet"
BASIS_MODEL = PROJECT / "models/svd_d5_matched/keeptl/keeptl__fullfit.npz"
OUT = PROJECT / "runs"
RESULTS = PROJECT / "results/e9_uturn_859_881"
SCENARIO = dict(scenario_id="859_881", dataset="HetroD", recording="00", ego=859, target=881,
                min_frame=14435, max_frame=15711, target_gt_end=15630, fps=30.0,
                scenario_uid="HetroD/00/859_881/14435-15711", subset="uturn_859_881",
                actor_start_frame=14783, ego_end_frame=15711, basis_class="keeptl")
SEEDS = [20260910, 20260911, 20260912]
N_DRAWS = 100
NT = 50
DUR_MIN = 0.5


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ast_functions(path, names, namespace):
    src = Path(path).read_text()
    tree = ast.parse(src)
    picked = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in picked} != set(names):
        raise RuntimeError(f"{path}: expected {names}, found {[n.name for n in picked]}")
    exec(compile(ast.fix_missing_locations(ast.Module(body=picked, type_ignores=[])), str(path), "exec"), namespace)
    return {n.name: hashlib.sha256(ast.get_source_segment(src, n).encode()).hexdigest() for n in picked}


def helpers():
    ns = dict(np=np, pd=pd, Path=Path, json=json, hashlib=hashlib, SIMC=SIMC,
              NT=NT, DUR_MIN=DUR_MIN, SCENARIO=SCENARIO, REAL_TRACKS=REAL_TRACKS, sha256=sha256)
    digests = ast_functions(SRC50, ["load_model", "encode", "decode", "tidy_from_vector",
                                    "ego_reference", "write_sample", "write_json"], ns)
    return ns, digests


def raw_vector(ref):
    """101-D = 50 interleaved xy of the actor resampled uniformly in time over its own
    support + the support duration (scripts/50 convention, asserted there against
    models/svd_d5/*/raw_features.npz)."""
    ta = (ref["actor_frame"] - ref["actor_frame"][0]) / SCENARIO["fps"]
    tq = np.linspace(0.0, ta[-1], NT)
    xy = np.column_stack([np.interp(tq, ta, ref["actor_xy"][:, 0]), np.interp(tq, ta, ref["actor_xy"][:, 1])])
    return np.r_[xy.reshape(-1), ta[-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--label-v2", action="store_true", help="updated label in separate run batches")
    ap.add_argument("--actor-start-frame", type=int, default=None,
                    help="start the target support at this frame while retaining the label-v2 window")
    args = ap.parse_args()
    global SCENARIO, RESULTS, REAL_TRACKS
    if args.label_v2:
        label = json.loads((ROOT / "HetroD-labeler/data/00_labeled_scenarios.json").read_text())["859_881"]
        lo, hi = int(label["min_frame"]), int(label["max_frame"])
        SCENARIO = dict(SCENARIO, min_frame=lo, max_frame=hi,
                        scenario_uid=f"HetroD/00/859_881/{lo}-{hi}", actor_start_frame=lo,
                        ego_end_frame=hi)
        RESULTS = PROJECT / "results/e9_uturn_859_881_l3_labelv2"
    if args.actor_start_frame is not None:
        if not args.label_v2:
            ap.error("--actor-start-frame requires --label-v2")
        start = args.actor_start_frame
        SCENARIO = dict(SCENARIO, actor_start_frame=start,
                        subset=f"uturn_859_881_start{start}")
        REAL_TRACKS = ROOT / f"exp_cross_coverage/data/uturn_859_881_start{start}/real_tracks.parquet"
        RESULTS = PROJECT / f"results/e9_uturn_859_881_l3_labelv2_start{start}"
    ns, digests = helpers()
    load_model, encode, decode = ns["load_model"], ns["encode"], ns["decode"]
    tidy_from_vector, write_sample, write_json = ns["tidy_from_vector"], ns["write_sample"], ns["write_json"]
    kde = {"np": np}
    ast_functions(KDE_SOURCE, ["loo_log_likelihood", "loo_bandwidth"], kde)

    ref = ns["ego_reference"]()
    x_raw = raw_vector(ref)
    assert x_raw.shape == (101,) and np.isfinite(x_raw).all()
    basis = load_model(BASIS_MODEL)
    assert SCENARIO["scenario_uid"] not in set(map(str, basis["training_uids"])), "donor basis must exclude the scenario"
    assert int(basis["numerical_rank"]) >= 5
    V = np.asarray(basis["V_train_d"], float)
    z0 = encode(basis, x_raw)
    h_loo = float(kde["loo_bandwidth"](V, n_grid=60))
    assert np.isfinite(h_loo) and h_loo > 0
    print(f"[uturn svd] z0 = {np.round(z0, 5).tolist()}  h_loo(keeptl V_train_d, N={len(V)}) = {h_loo:.6g}", flush=True)

    manifest = dict(scenario=SCENARIO, seeds=args.seeds, n_draws_per_seed=N_DRAWS, z0=z0.tolist(), h_loo=h_loo,
                    basis_model=dict(path=basis["path"], sha256=basis["sha256"], n_train=int(len(V)),
                                     donor_class=SCENARIO["basis_class"], numerical_rank=int(basis["numerical_rank"]),
                                     scenario_in_training=False),
                    helper_digests=digests,
                    sources={p.name: dict(path=str(p), sha256=sha256(p)) for p in
                             (BASIS_MODEL, REAL_TRACKS, KDE_SOURCE, SVD_SOURCE, SRC50, Path(__file__))},
                    singleton_fit=dict(status="undefined", centered_matrix_rank=0, n_samples_generated=0,
                                       class_data_used="none",
                                       reason="N = 1: the centred design matrix has rank N-1 = 0, so no basis and no "
                                              "LOO bandwidth exist and zero samples can be drawn"),
                    batches={})

    # (a) external-basis reconstruction of the recording
    start_tag = f"start{args.actor_start_frame}_" if args.actor_start_frame is not None else ""
    arm = "extbasis_recon"
    batch = OUT / f"svd_uturn_859_881_e9_{'labelv2_' if args.label_v2 else ''}{start_tag}{arm}"
    batch.mkdir(parents=True, exist_ok=True)
    vec = decode(basis, z0)[0]
    applied = vec.copy()
    applied[-1] = max(float(vec[-1]), DUR_MIN)
    sid = f"svd_{arm}"
    tidy = tidy_from_vector(applied, sid, SCENARIO["subset"], ref, float(applied[-1]))
    write_sample(batch, sid, vec, applied, tidy, dict(is_nominal_default=True, seed=None, draw=None, sigma_or_h=None,
                 basis_model=basis["path"], basis_sha256=basis["sha256"], basis_n_train=int(len(V)),
                 z=z0.tolist(), z0=z0.tolist(), donor_class=SCENARIO["basis_class"], scenario_in_training=False),
                 arm, "basis", f"svd_d5_{arm}")
    write_json(batch / "protocol.json", dict(arm=arm, class_data_used="basis", analytic=True, n=1,
                                             basis=basis["path"], basis_sha256=basis["sha256"]))
    manifest["batches"][arm] = dict(batch_dir=str(batch), n=1, class_data_used="basis")
    print(f"[uturn svd] {arm}: 1 decode", flush=True)

    # (b) external-basis Gaussian draws
    for h_name, h in (("h1", h_loo), ("h2", 2 * h_loo)):
        arm = f"extbasis_gauss_{h_name}"
        batch = OUT / f"svd_uturn_859_881_e9_{'labelv2_' if args.label_v2 else ''}{start_tag}{arm}"
        batch.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            Z = z0 + h * rng.standard_normal((N_DRAWS, 5))
            X = decode(basis, Z)
            for d in range(N_DRAWS):
                v = X[d]
                ap_ = v.copy()
                ap_[-1] = max(float(v[-1]), DUR_MIN)
                sid = f"{arm}__seed{seed}__draw{d + 1:03d}"
                tidy = tidy_from_vector(ap_, sid, SCENARIO["subset"], ref, float(ap_[-1]))
                write_sample(batch, sid, v, ap_, tidy,
                             dict(seed=seed, draw=d + 1, sigma_or_h=float(h), h_ext=float(h), h_loo=h_loo,
                                  h_multiple=1 if h_name == "h1" else 2, z=Z[d].tolist(), z0=z0.tolist(),
                                  basis_model=basis["path"], basis_sha256=basis["sha256"],
                                  basis_n_train=int(len(V)), donor_class=SCENARIO["basis_class"],
                                  scenario_in_training=False,
                                  rng="numpy default_rng(seed); the h1 and h2 arms share the same draws per seed"),
                             arm, "basis+bandwidth", f"svd_extbasis_gauss_{h_name}")
                rows.append(dict(sample_id=sid, seed=seed, draw=d + 1, duration_raw=float(v[-1]),
                                 duration_applied=float(ap_[-1]), duration_clipped=bool(v[-1] < DUR_MIN)))
        pd.DataFrame(rows).to_csv(batch / "draws.csv", index=False)
        clipped = int(sum(r["duration_clipped"] for r in rows))
        write_json(batch / "protocol.json", dict(arm=arm, class_data_used="basis+bandwidth", analytic=True,
                                                 n=len(rows), h_ext=h, h_loo=h_loo, seeds=args.seeds,
                                                 basis=basis["path"], basis_sha256=basis["sha256"],
                                                 duration_clipped_n=clipped))
        manifest["batches"][arm] = dict(batch_dir=str(batch), n=len(rows), class_data_used="basis+bandwidth",
                                        h_ext=h, duration_clipped_n=clipped)
        print(f"[uturn svd] {arm}: {len(rows)} draws, h={h:.5g}, clipped={clipped}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "svd_arms_manifest.json", manifest)
    print(json.dumps({k: v["n"] for k, v in manifest["batches"].items()}))


if __name__ == "__main__":
    main()
