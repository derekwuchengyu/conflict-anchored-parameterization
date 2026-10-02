#!/usr/bin/env python3
"""E10 stage 1: Euclidean conflict-window extraction (paper 04_impl.tex, L = 10 m disk).

For every source case the spatial anchor q_c is the FROZEN anchor of the arc-length arm
(results/ours3_contexts.json; PET / legacy PET / crossTraj per class, unchanged). Walking the
recorded target track backward and forward in time from the anchor, q- and q+ are the first
points where the Euclidean distance to q_c reaches L, linearly interpolated on the crossing
segment so that both chords equal L exactly (eq. fixed-window-distances). The observation
interval is the annotated episode window; a boundary that is not reached inside the window
makes the case unavailable (paper: "provided that both sets are nonempty"). Whether the full
recorded support would rescue it is recorded as a diagnostic only.

Outputs (results/): disk_window_L10_points.csv (513-case census), disk_window_L10_summary.csv,
ours3_disk_population.csv, ours3_disk_contexts.json, ours3_disk_default_jobs.json (489 cohort),
ours3_disk_rescued_jobs.json (cases outside the 489 cohort that the disk rule makes available),
ours3_disk_39_180_oat_jobs.json (two anchor frames x 7 OAT jobs), ours3_disk_manifest.json.
Read-only upstream; no simulation, scoring, or source mutation. New geometry variant only.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT / "results"
VELOCITY = ROOT / "exp_coverage_velocity/scripts"
RAW = ROOT / "HetroD-labeler/data/00_tracks.parquet"
L_DISK = 10.0
# 2026-09-14 per-class radius, chosen on the primary metric D_int after the fair L=5 vs L=10 comparison
# (review_only/bestL_check_20260914/REPORT.md): cut-in left/right use 5 m, every other class keeps 10 m.
# Output file names keep their historical "L10" spelling; the L_m / window_L_m columns record the real radius.
# 2026-09-14 (later): user reverted cut-in right to 10 m. At 5 m it admitted 4 short-window scenes on the narrow arm
# (review_only/v3_decisions_20260914/cutinr_new_scenes.png) that made its off-road rate and variance ratio much worse.
L_BY_CLASS = {"cutinl": 5.0}
VARIANT = "euclid_L10_exact_rotation"
SPECIAL_VARIANT = "special_pet_euclid_L10_exact_rotation"
CP_LABELS = {"tlkeep": "leftturn", "keeptl": "keeplt", "cutinr": "cutin"}
SPECIAL_TRACKS = ROOT / "exp_cross_coverage/data/special_39_180/real_tracks.parquet"
SPECIAL_BASES = {
    2641: ROOT / "exp_cross_coverage/esmini_runs/_xosc_base_anch_px_s10/HetroD-01KEEP_02CUTIN_L_39_180_f2425.xosc",
    2609: ROOT / "exp_cross_coverage/esmini_runs/_xosc_base_anch_xx_s10/HetroD-01KEEP_02CUTIN_L_39_180_f2425.xosc",
}


def L_for(subset):
    return L_BY_CLASS.get(subset, L_DISK)


def variant_for(subset):
    return VARIANT.replace("L10", f"L{L_for(subset):g}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def existing_functions(path, names, namespace):
    parsed = ast.parse(path.read_text())
    body = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != set(names):
        raise ValueError(f"Missing requested existing function in {path}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def crossing(xy, frames, i_anchor, direction, L):
    """First point at Euclidean distance L from xy[i_anchor] walking in `direction`.
    Returns (point, interpolated_frame, outside_index) or None when never reached."""
    qc = xy[i_anchor]
    d = np.hypot(*(xy - qc).T)
    order = range(i_anchor - 1, -1, -1) if direction < 0 else range(i_anchor + 1, len(xy))
    prev = i_anchor
    for i in order:
        if d[i] >= L:
            a, b = xy[prev], xy[i]
            v, w = b - a, a - qc
            A, B, C = float(v @ v), float(2 * (v @ w)), float(w @ w - L * L)
            if A <= 0:
                prev = i
                continue
            disc = max(B * B - 4 * A * C, 0.0)
            s = (-B + np.sqrt(disc)) / (2 * A)
            s = float(min(max(s, 0.0), 1.0))
            p = a + s * v
            f = float(frames[prev] + s * (frames[i] - frames[prev]))
            return p, f, int(i)
        prev = i
    return None


def disk_points(g, anchor_frame, L):
    """g: sorted actor samples with columns frame, x, y. Returns dict."""
    frames = g.frame.to_numpy(float)
    xy = g[["x", "y"]].to_numpy(float)
    hit = np.flatnonzero(frames == anchor_frame)
    if len(hit) != 1:
        return dict(status="no_anchor_sample")
    i = int(hit[0])
    back = crossing(xy, frames, i, -1, L)
    fwd = crossing(xy, frames, i, +1, L)
    res = dict(qc_x=float(xy[i, 0]), qc_y=float(xy[i, 1]), anchor_index=i,
               d_window_start_to_qc=float(np.hypot(*(xy[0] - xy[i]))),
               d_window_end_to_qc=float(np.hypot(*(xy[-1] - xy[i]))))
    if back is None and fwd is None:
        res["status"] = "both_unavailable"
    elif back is None:
        res["status"] = "q_minus_unavailable"
    elif fwd is None:
        res["status"] = "q_plus_unavailable"
    else:
        res["status"] = "ok"
    if back is not None:
        res.update(qm_x=float(back[0][0]), qm_y=float(back[0][1]), t_minus_frame=back[1],
                   chord1=float(np.hypot(*(back[0] - xy[i]))))
    if fwd is not None:
        res.update(qp_x=float(fwd[0][0]), qp_y=float(fwd[0][1]), t_plus_frame=fwd[1],
                   chord2=float(np.hypot(*(fwd[0] - xy[i]))))
    return res


def fallback_anchor_frames():
    """Anchor frames for cases outside the 489 cohort, using the SAME criteria as the arc arm:
    window-scoped PET (exp_cp3_points pet_window) for tlkeep/keeptl/cutinr; hetero-param
    parampath traj_cross for cutinl ('xx') and pet for keeptl_sw (legacy)."""
    result = {}
    pw_sources = []
    for subset, label in CP_LABELS.items():
        path = ROOT / f"exp_cp3_points/results/pet_window_{label}.csv"
        pw_sources.append(path)
        pw = pd.read_csv(path)
        for r in pw.itertuples(index=False):
            if pd.notna(r.new_crit_x):
                result[(subset, str(r.scenario_id))] = dict(crit_xy=(float(r.new_crit_x), float(r.new_crit_y)),
                                                            source=f"pet_window:{r.new_anchor}")
    return result, pw_sources


def parampath_anchor(subset, ego, actor, mf, xf):
    cwd = os.getcwd()
    try:
        sys.path.insert(0, str(ROOT / "exp_cross_coverage/scripts"))
        import lib as CL  # noqa: F401  (chdirs to hetero-param)
        from hetero_param import parampath as PP
        pet, md, tc = PP.anchor_frames("HetroD", int(ego), int(actor), int(mf), int(xf))
    finally:
        os.chdir(cwd)
    if subset == "cutinl":
        return tc, "parampath:traj_cross"
    if subset == "keeptl_sw":
        return pet, "parampath:pet"
    return None, "none"


def main():
    cases = pd.read_csv(OUT / "svd_d5_cases.csv")
    cases = cases[cases["mode"] == "fullfit"].copy()
    assert cases.scenario_uid.is_unique
    population = pd.read_csv(OUT / "ours3_population.csv")
    pop_by_uid = {r.scenario_uid: r for r in population.itertuples(index=False)}
    contexts = json.loads((OUT / "ours3_contexts.json").read_text())
    ctx_by_uid = {c["scenario_uid"]: c for c in contexts}
    assert len(ctx_by_uid) == len(contexts)
    ns = {"np": np, "NT": 50}
    existing_functions(VELOCITY / "cvlib.py", ["arc_resample"], ns)
    library = SimpleNamespace(arc_resample=ns["arc_resample"])
    ns["L"] = library
    existing_functions(VELOCITY / "cvrender.py", ["_signed_angle", "_rot", "theta_geometry", "theta_to_cps"], ns)
    fallback, pw_sources = fallback_anchor_frames()
    source_paths = {OUT / "svd_d5_cases.csv", OUT / "ours3_population.csv", OUT / "ours3_contexts.json",
                    VELOCITY / "cvlib.py", VELOCITY / "cvrender.py", Path(__file__), *pw_sources}
    raw_ids = sorted(set(cases.actor.astype(int)))
    raw = pd.read_parquet(RAW, filters=[("trackId", "in", raw_ids)], columns=["trackId", "frame", "xCenter", "yCenter"])
    source_paths.add(RAW)
    rows, disk_contexts, jobs_cohort, jobs_rescued = [], [], [], []
    for subset, subset_cases in cases.groupby("subset", sort=False):
        track_path = Path(subset_cases.source_tracks.iloc[0])
        source_paths.add(track_path)
        tracks = pd.read_parquet(track_path)
        by_actor = {str(s): g.sort_values("frame").reset_index(drop=True) for s, g in
                    tracks[tracks.role == "actor"].groupby("scenario_id")}
        library.real_actor_track = lambda _t, sid: by_actor[str(sid)]
        for case in subset_cases.itertuples(index=False):
            sid = str(case.scenario_id)
            row = dict(subset=subset, scenario_id=sid, scenario_uid=case.scenario_uid, case_index=case.case_index,
                       ego=int(case.ego), actor=int(case.actor), min_frame=int(case.metadata_min_frame),
                       max_frame=int(case.metadata_max_frame), in_arc_cohort=case.scenario_uid in ctx_by_uid,
                       group_id=case.group_id, L_m=L_for(subset))
            g = by_actor.get(sid)
            if g is None:
                row.update(status="no_track", anchor_source="none")
                rows.append(row)
                continue
            ctx = ctx_by_uid.get(case.scenario_uid)
            if ctx is not None:
                anchor_frame, anchor_source = int(ctx["anchor_frame"]), f"frozen_context:{ctx['geometry_variant']}"
            elif (subset, sid) in fallback:
                cx, cy = fallback[(subset, sid)]["crit_xy"]
                dist = np.hypot(g.x.to_numpy() - cx, g.y.to_numpy() - cy)
                k = int(dist.argmin())
                if dist[k] > 0.001:
                    row.update(status="no_anchor_sample", anchor_source=fallback[(subset, sid)]["source"])
                    rows.append(row)
                    continue
                anchor_frame, anchor_source = int(g.frame.iloc[k]), fallback[(subset, sid)]["source"]
            else:
                frame, anchor_source = parampath_anchor(subset, case.ego, case.actor, case.metadata_min_frame, case.metadata_max_frame)
                if frame is None or frame not in set(g.frame.astype(int)):
                    row.update(status="no_anchor", anchor_source=anchor_source)
                    rows.append(row)
                    continue
                anchor_frame = int(frame)
            row.update(anchor_frame=anchor_frame, anchor_source=anchor_source)
            res = disk_points(g, anchor_frame, L_for(subset))
            row.update(res)
            # diagnostic: would the full recorded support reach the boundary?
            full = raw[raw.trackId == int(case.actor)].sort_values("frame").rename(columns={"xCenter": "x", "yCenter": "y"})
            full_res = disk_points(full.reset_index(drop=True), anchor_frame, L_for(subset))
            row.update(full_support_status=full_res.get("status"),
                       full_support_t_minus_frame=full_res.get("t_minus_frame"), full_support_t_plus_frame=full_res.get("t_plus_frame"),
                       rescued_by_full_support=bool(res["status"] != "ok" and full_res.get("status") == "ok"))
            if ctx is not None:
                arc = np.asarray(ctx["cps6"], float).reshape(3, 2)
                row.update(arc_theta1_deg=ctx["theta1_deg"], arc_theta2_deg=ctx["theta2_deg"],
                           arc_chord1=ctx["geometry"]["L1"], arc_chord2=ctx["geometry"]["L2"],
                           arc_qc_match_m=float(np.hypot(*(arc[1] - [res.get("qc_x", np.nan), res.get("qc_y", np.nan)]))))
                if "qm_x" in res:
                    row["arc_vs_disk_qminus_m"] = float(np.hypot(arc[0, 0] - res["qm_x"], arc[0, 1] - res["qm_y"]))
                if "qp_x" in res:
                    row["arc_vs_disk_qplus_m"] = float(np.hypot(arc[2, 0] - res["qp_x"], arc[2, 1] - res["qp_y"]))
            if res["status"] != "ok":
                rows.append(row)
                continue
            xy = np.array([[res["qm_x"], res["qm_y"]], [res["qc_x"], res["qc_y"]], [res["qp_x"], res["qp_y"]]])
            geo = ns["theta_geometry"](None, sid, xy.ravel())
            if geo is None:
                row.update(status="degenerate_geometry")
                rows.append(row)
                continue
            np.testing.assert_allclose(ns["theta_to_cps"](geo, geo["theta1"], geo["theta2"]), xy.ravel(), rtol=0, atol=1e-7)
            assert abs(geo["L1"] - L_for(subset)) < 1e-6 and abs(geo["L2"] - L_for(subset)) < 1e-6, (geo["L1"], geo["L2"])
            row.update(theta1_deg=geo["theta1"], theta2_deg=geo["theta2"], u1_source=geo["u1_source"])
            base = None
            if ctx is not None:
                base = Path(ctx["base"])
            else:
                pop = pop_by_uid.get(case.scenario_uid)
                if pop is not None and isinstance(pop.base, str) and pop.base and Path(pop.base).exists():
                    base = Path(pop.base)
                elif subset in CP_LABELS:
                    matches = sorted((ROOT / "hetero-param/results/esmini/xosc_base/srexp-petq3").glob(
                        f"*_{case.ego}_{case.actor}_f{case.metadata_min_frame + 1}.xosc"))
                    base = matches[0] if len(matches) == 1 else None
                else:
                    tmpl = pd.read_csv(ROOT / f"exp_cov_sampling/results/templates_{subset}.csv")
                    m = tmpl[(tmpl.scenario_id == sid) & (tmpl.min_frame == case.metadata_min_frame)]
                    base = Path(m.base.iloc[0]) if len(m) == 1 and Path(m.base.iloc[0]).exists() else None
            row["base"] = str(base) if base else ""
            row["base_available"] = base is not None
            if base is None:
                rows.append(row)
                continue
            source_paths.add(base)
            root = ET.parse(base).getroot()
            bp = {p.get("name"): p.get("value") for p in root.iter("ParameterDeclaration")}
            n_cp = len([c for c in root.iter("ControlPoint")])
            n_w = len([c for c in root.iter("ControlPoint") if c.get("weight") is not None])
            row.update(base_control_points=n_cp, base_weighted_control_points=n_w,
                       base_patch_compatible=bool(n_cp - n_w == 4))
            anchor_row = g[g.frame == anchor_frame].iloc[0]
            end_speed = float(anchor_row.speed) * 3.6
            frozen = {k: float(bp[k]) for k in ["Agent1_Offset", "Agent1_1_SA_DynamicDuration", "Agent1_Speed"]}
            row.update(end_speed_kmh=end_speed, base_sha256=digest(base),
                       fixed_offset_m=frozen["Agent1_Offset"], fixed_ramp_duration_s=frozen["Agent1_1_SA_DynamicDuration"],
                       fixed_start_speed_kmh=frozen["Agent1_Speed"])
            context = dict(row, cps6=xy.ravel(), geometry=geo, params=[geo["theta1"], geo["theta2"], end_speed],
                           fixed_declarations=frozen, geometry_variant=variant_for(subset))
            job = {"job_id": f"{subset}__{sid}__default", "scenario_id": sid, "class": subset, "dataset": "HetroD",
                   "recording": "00", "ego": int(case.ego), "target": int(case.actor),
                   "min_frame": int(case.metadata_min_frame), "max_frame": int(case.metadata_max_frame),
                   "source_xosc": str(base), "cps_xy": xy.tolist(), "anchor_frame": anchor_frame,
                   "geometry_variant_id": variant_for(subset),
                   "parameters": {"theta1_deg": None, "theta2_deg": None, "interaction_end_speed_kmh": None},
                   "source_context": {"subset": subset, "scenario_uid": case.scenario_uid, "source_tracks": str(track_path),
                                      "anchor_source": anchor_source, "window_L_m": L_for(subset),
                                      "t_minus_frame": res["t_minus_frame"], "t_plus_frame": res["t_plus_frame"]}}
            if ctx is not None and row["base_patch_compatible"]:
                disk_contexts.append(context)
                jobs_cohort.append(job)
            elif row["base_patch_compatible"]:
                jobs_rescued.append(job)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "disk_window_L10_points.csv", index=False)
    summary = table.groupby(["subset", "status"]).size().unstack(fill_value=0)
    summary["rescued_by_full_support"] = table.groupby("subset").rescued_by_full_support.sum()
    summary["in_arc_cohort_and_ok"] = table[table.in_arc_cohort & table.status.eq("ok")].groupby("subset").size()
    summary["render_jobs_cohort"] = pd.Series({j["class"]: 0 for j in jobs_cohort}).add(
        pd.Series([j["class"] for j in jobs_cohort]).value_counts(), fill_value=0)
    summary = summary.fillna(0).astype(int)
    summary.to_csv(OUT / "disk_window_L10_summary.csv")
    pop_cols = [c for c in table.columns if c not in ("cps6",)]
    table[pop_cols].to_csv(OUT / "ours3_disk_population.csv", index=False)
    (OUT / "ours3_disk_contexts.json").write_text(json.dumps(disk_contexts, indent=2, default=serial, allow_nan=False) + "\n")
    (OUT / "ours3_disk_default_jobs.json").write_text(json.dumps(
        {"batch_name": "ours3_disk_population_defaults", "stop_on_failure": False, "jobs": jobs_cohort}, indent=1) + "\n")
    (OUT / "ours3_disk_rescued_jobs.json").write_text(json.dumps(
        {"batch_name": "ours3_disk_rescued_defaults", "stop_on_failure": False, "jobs": jobs_rescued}, indent=1) + "\n")

    # ---- special 39_180: two frozen anchor frames (2641 special PET, 2609 population crossTraj) ----
    special = pd.read_parquet(SPECIAL_TRACKS)
    source_paths.add(SPECIAL_TRACKS)
    ga = special[special.role == "actor"].sort_values("frame").reset_index(drop=True)
    library.real_actor_track = lambda _t, sid: ga
    special_rows, special_jobs = [], []
    for anchor_frame, base in SPECIAL_BASES.items():
        source_paths.add(base)
        res = disk_points(ga, anchor_frame, L_DISK)
        xy = np.array([[res["qm_x"], res["qm_y"]], [res["qc_x"], res["qc_y"]], [res["qp_x"], res["qp_y"]]])
        geo = ns["theta_geometry"](None, "39_180", xy.ravel())
        special_rows.append(dict(scenario_id="39_180", anchor_frame=anchor_frame, base=str(base), **res,
                                 theta1_deg=geo["theta1"], theta2_deg=geo["theta2"], u1_source=geo["u1_source"]))
        common = dict(scenario_id="39_180", **{"class": "special_39_180"}, dataset="HetroD", recording="00", ego=39, target=180,
                      min_frame=2424, max_frame=3002, anchor_frame=anchor_frame, source_xosc=str(base),
                      cps_xy=xy.tolist(), geometry_variant_id=SPECIAL_VARIANT, expected_target_gt_end=2923,
                      source_context={"subset": "special_39_180", "source_tracks": str(SPECIAL_TRACKS),
                                      "scenario_uid": "HetroD/00/39_180/2424-3002", "window_L_m": L_DISK,
                                      "anchor_source": "frozen_special_2641" if anchor_frame == 2641 else "frozen_population_2609",
                                      "t_minus_frame": res["t_minus_frame"], "t_plus_frame": res["t_plus_frame"]})
        tag = f"a{anchor_frame}"
        special_jobs.append(dict(common, job_id=f"{tag}__default"))
        for axis in (1, 2):
            for label, delta in (("minus", -10), ("plus", 10)):
                special_jobs.append(dict(common, job_id=f"{tag}__theta{axis}_{label}10deg", **{f"theta{axis}_delta_deg": delta}))
        for label, factor in (("minus", .8), ("plus", 1.2)):
            special_jobs.append(dict(common, job_id=f"{tag}__end_speed_{label}20pct", end_speed_factor=factor))
    pd.DataFrame(special_rows).to_csv(OUT / "disk_window_L10_special_39_180.csv", index=False)
    (OUT / "ours3_disk_39_180_oat_jobs.json").write_text(json.dumps(
        {"batch_name": "ours3_disk_39_180_oat", "stop_on_failure": False, "jobs": special_jobs}, indent=1) + "\n")

    sources = [{"path": str(p), "sha256": digest(p)} for p in sorted(source_paths)]
    (OUT / "ours3_disk_manifest.json").write_text(json.dumps({
        "scope": "E10 stage 1: Euclidean L=10 m conflict-window extraction; no simulation or metrics",
        "window_rule": "q_c = frozen anchor sample; q-/q+ = first Euclidean-distance-L crossings walking backward/forward "
                       "in time inside the annotated episode window, linearly interpolated on the crossing segment; "
                       "unavailable if a boundary is not reached in the window (full-support rescue is diagnostic only)",
        "anchor_criteria": "unchanged from the arc-length arm (frozen contexts); fallback for non-cohort cases uses the same per-class criteria",
        "variable_parameters": ["theta1_deg", "theta2_deg", "end_speed_kmh"], "L_m": L_DISK,
        "n_source_cases": int(len(table)), "n_ok": int(table.status.eq("ok").sum()),
        "n_cohort_jobs": len(jobs_cohort), "n_rescued_jobs": len(jobs_rescued), "n_special_jobs": len(special_jobs),
        "sources": sources}, indent=2, allow_nan=False) + "\n")
    print(summary.to_string())
    print(pd.DataFrame(special_rows)[["anchor_frame", "status", "chord1", "chord2", "theta1_deg", "theta2_deg", "t_minus_frame", "t_plus_frame"]].to_string(index=False))


if __name__ == "__main__":
    main()
