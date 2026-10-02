#!/usr/bin/env python3
"""Uturn 859_881 singleton table: ours3 arms, executed through scripts/20_ours3_generate.py.

  ours3_recon      the recording's own (theta1, theta2, EndSpeed) decode -> the
                   row's similarity cells.
  ours3_condkde    class_data_used = bandwidth.  Draws centred on the scenario's
                   own standardized vector with the KEEPTL class bandwidth
                   (h = 0.49718 over 39 disk contexts, not refit -- 859_881
                   belongs to no fitted class, so the donor is the same left-turn
                   cohort that donates the SVD basis), 100 draws x seeds
                   {20260910, 20260911, 20260912}.

Geometry = the canonical best config for this scenario: 3 CPs at the crossTraj
anchor +/- 5 m arc length (base _xosc_base_anch_xx_s5) with the conflict CP
weight raised to 8.
--radius3 instead uses the original conflict anchor and L=3 base
(cp3s3w8base.xosc), with separate plans and run directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RUNNER = PROJECT / "scripts/20_ours3_generate.py"
FIT_NPZ = PROJECT / "plans/ours3_disk_kde_dependent/7a54a1c472f1327d/fit_keeptl.npz"
FIT_JSON = PROJECT / "plans/ours3_disk_kde_dependent/7a54a1c472f1327d/fit.json"
BASE_XOSC = ROOT / "exp_cross_coverage/esmini_runs/_xosc_base_anch_xx_s5/HetroD-01KEEP_02TL_859_881_f14436.xosc"
PLANS = PROJECT / "plans/e9_uturn_859_881"
SEEDS = [20260910, 20260911, 20260912]
N = 100
COLS = ["theta1_deg", "theta2_deg", "interaction_end_speed_kmh"]
# the recording's own decode, read back from the first generated sample.json
SPECIAL = np.array([21.530957473300063, 121.40108427690005, 5.755447801170644])
DONOR = "keeptl"
RADIUS3 = False
LABEL_V2 = False
ACTOR_START_FRAME = None
CUSTOM_RAW = None
LABELS = ROOT / "HetroD-labeler/data/00_labeled_scenarios.json"


def window():
    if LABEL_V2:
        entry = json.loads(LABELS.read_text())["859_881"]
        return int(entry["min_frame"]), int(entry["max_frame"])
    return 14435, 15711


def batch_prefix():
    start = f"start{ACTOR_START_FRAME}_" if ACTOR_START_FRAME is not None else ""
    return f"ours3_uturn_859_881_e9_{'l3_' if RADIUS3 else ''}{'labelv2_' if LABEL_V2 else ''}{start}"


def prepare_actor_start_base(start_frame: int):
    """Keep the 14986 scenario clock, but activate 881 at its action start."""
    lo, _ = window()
    end_frame, anchor_frame, fps = 15630, 15451, 30.0
    if not lo < start_frame < anchor_frame < end_frame:
        raise ValueError("actor start must lie between the window start and conflict anchor")
    raw = pd.read_parquet(ROOT / "HetroD-labeler/data/00_tracks.parquet")
    actor = raw[(raw.trackId == 881) & raw.frame.between(start_frame, end_frame)].sort_values("frame")
    ego = raw[(raw.trackId == 859) & raw.frame.between(lo, window()[1])].sort_values("frame")
    if actor.frame.iloc[0] != start_frame or actor.frame.iloc[-1] != end_frame:
        raise ValueError("requested actor support is absent from raw tracks")

    data_dir = ROOT / f"exp_cross_coverage/data/uturn_859_881_start{start_frame}"
    data_dir.mkdir(parents=True, exist_ok=True)
    custom_raw = data_dir / "raw_tracks.parquet"
    pd.concat([ego, actor]).sort_values(["trackId", "frame"]).to_parquet(custom_raw, index=False)
    real = pd.read_parquet(ROOT / "exp_cross_coverage/data/uturn_859_881/real_tracks.parquet")
    real = real[(real.role.eq("ego")) | (real.role.eq("actor") & (real.frame >= start_frame))]
    real.sort_values(["role", "frame"]).to_parquet(data_dir / "real_tracks.parquet", index=False)

    xy = actor[["xCenter", "yCenter"]].to_numpy(float)
    arc = np.r_[0.0, np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).cumsum()]
    points = [(float(np.interp(s, arc, xy[:, 0])), float(np.interp(s, arc, xy[:, 1])))
              for s in (0.0, 0.8, 1.0)]
    heading = float(np.deg2rad(actor.heading.iloc[0]))

    tree = ET.parse(BASE_XOSC)
    root = tree.getroot()
    params = {p.get("name"): p for p in root.iter("ParameterDeclaration")}
    values = {
        "Agent1_Delay": (start_frame - lo) / fps,
        "Agent1_Speed": float(np.hypot(actor.xVelocity.iloc[0], actor.yVelocity.iloc[0]) * 3.6),
        "Agent1_1_TA_DynamicDuration": (end_frame - start_frame) / fps,
        "Agent1_1_SA_DynamicDuration": (anchor_frame - start_frame) / fps,
    }
    for name, value in values.items():
        params[name].set("value", f"{value:.12g}")

    parent = {child: node for node in root.iter() for child in node}
    replacements = {"25.4": points[0], "$Agent1_S": points[0], "${$Agent1_S + 0.8}": points[1],
                    "${$Agent1_S + 1}": points[2]}
    counts = {key: 0 for key in replacements}
    for lane in list(root.iter("LanePosition")):
        key = lane.get("s")
        if key not in replacements:
            continue
        x, y = replacements[key]
        world = ET.Element("WorldPosition", x=f"{x:.8f}", y=f"{y:.8f}", h=f"{heading:.12g}")
        par = parent[lane]
        par.remove(lane)
        par.append(world)
        counts[key] += 1
    if counts != {"25.4": 1, "$Agent1_S": 2, "${$Agent1_S + 0.8}": 1,
                  "${$Agent1_S + 1}": 1}:
        raise RuntimeError(f"unexpected Agent1 start-control layout: {counts}")

    base_dir = ROOT / "exp_cross_coverage/esmini_runs/_xosc_base_cp3s3_start15371"
    base_dir.mkdir(parents=True, exist_ok=True)
    base = base_dir / "HetroD-01KEEP_02TL_859_881_f14987_start15371.xosc"
    tree.write(base, encoding="utf-8", xml_declaration=True)
    (data_dir / "README.md").write_text(
        f"881 action support starts at frame {start_frame}; scenario/ego/background clock remains at {lo}.\n"
        f"Delay={(start_frame-lo)/fps:.6f} s, trajectory duration={(end_frame-start_frame)/fps:.6f} s.\n")
    return base, custom_raw


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_job():
    lo, hi = window()
    subset = f"uturn_859_881_start{ACTOR_START_FRAME}" if ACTOR_START_FRAME is not None else "uturn_859_881"
    return dict(scenario_id="859_881", **{"class": subset}, dataset="HetroD", recording="00",
                ego=859, target=881, min_frame=lo, max_frame=hi,
                anchor_frame=15451 if RADIUS3 else 15406,
                source_xosc=str(BASE_XOSC), middle_cp_weight=8.0,
                geometry_variant_id="uturn_cp3d3_w8_exact_rotation" if RADIUS3 else "uturn_crossTraj_cp3d5_w8_exact_rotation",
                expected_default_end_speed_kmh=float(SPECIAL[2]), expected_target_gt_end=15630,
                **({"raw_tracks_path": str(CUSTOM_RAW), "target_gt_start_frame": ACTOR_START_FRAME}
                   if ACTOR_START_FRAME is not None else {}),
                source_context=dict(subset=subset, anchor_combo="original_cp3" if RADIUS3 else "xx",
                                    step_m=3.0 if RADIUS3 else 5.0, conflict_cp_weight=8.0))


def recon_spec():
    b = base_job()
    job = dict(b, job_id="recon", source_context=dict(b["source_context"], arm="ours3_recon", class_data_used="none"))
    return dict(batch_name=batch_prefix() + "recon", stop_on_failure=True, jobs=[job])


def condkde_spec(seed):
    b = base_job()
    with np.load(FIT_NPZ) as z:
        raw, mean, sd, Z, h = z["raw"], z["mean"], z["sd"], z["standardized"], float(z["h"])
    fit = json.loads(FIT_JSON.read_text())["fits"][DONOR]
    assert abs(fit["h"] - h) < 1e-12 and raw.shape == (39, 3) and fit["columns"] == COLS
    assert np.allclose((raw - mean) / sd, Z)
    assert "859_881" not in set(map(str, fit["center_scenario_ids"])), "donor cohort must exclude the scenario"
    z_center = (SPECIAL - mean) / sd
    rng = np.random.default_rng(seed)
    requested = (z_center + h * rng.standard_normal((N, 3))) * sd + mean
    applied = requested.copy()
    applied[:, 2] = np.maximum(0.0, applied[:, 2])
    jobs, rows = [], []
    for d in range(N):
        jid = f"condkde__seed{seed}__draw{d + 1:03d}"
        clipped = bool(requested[d, 2] < 0)
        jobs.append(dict(b, job_id=jid, parameters=dict(zip(COLS, map(float, applied[d]))),
                         source_context=dict(b["source_context"], arm="ours3_condkde", seed=seed, draw=d + 1,
                                             class_data_used="bandwidth", kde_fit=str(FIT_NPZ),
                                             kde_fit_sha256=sha256(FIT_NPZ), kde_donor_class=DONOR, kde_n_fit=39,
                                             kde_h_standardized=h,
                                             kernel_center="the recording's own (theta1, theta2, EndSpeed) vector, "
                                                           "standardized with the donor class mean/sd; not a fit point",
                                             kernel_center_raw=SPECIAL.tolist(),
                                             kernel_center_standardized=z_center.tolist(),
                                             sampler_parameters_requested=dict(zip(COLS, map(float, requested[d]))),
                                             end_speed_clipped=clipped,
                                             end_speed_clip_rule="max(0.0, requested_end_speed_kmh)",
                                             theta_clipped=False, rejected_or_resampled=False)))
        rows.append(dict(job_id=jid, seed=seed, draw=d + 1,
                         **{f"requested_{c}": float(v) for c, v in zip(COLS, requested[d])},
                         **{f"applied_{c}": float(v) for c, v in zip(COLS, applied[d])},
                         end_speed_clipped=clipped, h=h))
    return dict(batch_name=batch_prefix() + f"condkde_{seed}", stop_on_failure=False, jobs=jobs), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--radius3", action="store_true", help="independent L=3 original-anchor run")
    ap.add_argument("--label-v2", action="store_true", help="updated 14986-15651 label, isolated outputs")
    ap.add_argument("--actor-start-frame", type=int, default=None,
                    help="keep the label-v2 clock but delay target motion to this recorded frame")
    args = ap.parse_args()
    global LABEL_V2
    LABEL_V2 = args.label_v2
    if args.radius3:
        global BASE_XOSC, PLANS, SPECIAL, RADIUS3
        RADIUS3 = True
        BASE_XOSC = ROOT / "exp_cross_coverage/esmini_runs/HetroD-01KEEP_02TL_859_881_f14436/cp3s3w8base.xosc"
        PLANS = PROJECT / "plans/e9_uturn_859_881_l3"
        SPECIAL = np.array([125.18710880941325, 41.00908062595026, 9.250848936957084])
    if LABEL_V2:
        if not RADIUS3:
            ap.error("--label-v2 currently requires --radius3")
        BASE_XOSC = ROOT / "exp_cross_coverage/esmini_runs/_xosc_base_cp3s3/HetroD-01KEEP_02TL_859_881_f14987.xosc"
        PLANS = PROJECT / "plans/e9_uturn_859_881_l3_labelv2"
    if args.actor_start_frame is not None:
        if not (LABEL_V2 and RADIUS3):
            ap.error("--actor-start-frame requires --radius3 --label-v2")
        global ACTOR_START_FRAME, CUSTOM_RAW
        ACTOR_START_FRAME = args.actor_start_frame
        BASE_XOSC, CUSTOM_RAW = prepare_actor_start_base(ACTOR_START_FRAME)
        PLANS = PROJECT / f"plans/e9_uturn_859_881_l3_labelv2_start{ACTOR_START_FRAME}"
    PLANS.mkdir(parents=True, exist_ok=True)
    specs = {"recon": recon_spec()}
    for seed in SEEDS:
        spec, draws = condkde_spec(seed)
        draws.to_csv(PLANS / f"condkde_draws_{seed}.csv", index=False)
        specs[f"condkde_{seed}"] = spec
    paths = {}
    for name, spec in specs.items():
        p = PLANS / f"{name}_jobs.json"
        p.write_text(json.dumps(spec, indent=1))
        paths[name] = p
        print(f"[uturn ours] {name}: {len(spec['jobs'])} jobs -> {p}")
    if not args.execute:
        return 0
    for name, p in paths.items():
        if args.only and name not in args.only:
            continue
        print(f"[uturn ours] executing {name}", flush=True)
        r = subprocess.run([sys.executable, str(RUNNER), "--jobs-json", str(p)], capture_output=True, text=True)
        tail = "\n".join(r.stdout.strip().splitlines()[-2:])
        print(f"[uturn ours] {name} rc={r.returncode} {tail}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
