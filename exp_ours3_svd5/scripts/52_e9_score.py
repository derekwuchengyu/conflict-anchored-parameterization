#!/usr/bin/env python3
"""E9 (39_180 singleton table T8): score every arm and build the table.

Phases (idempotent, --phase selects; default = all):
  nonpet  scripts/31_ours3_nonpet.py per arm (--method <arm>, prefix e9_<arm>)
  pet     scripts/32_bbox_pet.py on every e9_*_paired.csv with the separate
          cache results/bbox_pet_cache_e9.json
  score   DTW (traj_dtw AST-loaded from exp_cross_coverage/scripts/
          180_trajdtw_aggregate.py, as scripts/35 does), validity, background
          OBB solid hits, spread/bracket, composite D_m, T8 tables, report,
          manifest.

Validity per sample (target rows inside the metadata window):
  teleport   exp_cross_coverage/scripts/lib.py is_teleport (position-derived
             step speed vs state speed, first 10 steps ignored) -- executed
             arms only; analytic arms have derived state speed by construction
             and are labelled "analytic_kinematics".
  wrong-way  exp_cov_sampling/scripts/95_theta_figs.py exit_angle > 120 deg
             (generated path vs GT target path, 3 m chords at the GT end).
  a_lat/v    exp_coverage_velocity/scripts/cvlib.py validity_mask rule
             (v_max > 25 m/s or a_lat_max > 5 m/s^2, a_lat masked below 1 m/s),
             computed with the same gradient formula on each sample's own time
             grid (ours: 30 Hz esmini state; SVD: the 50-point decode).
  off-road   sr-tlkeep-experiment/scripts/33_e7_offroad.py drivable_union on
             tyms.xodr (buffer 0.5 m), flag if > 5 % of points outside. Reported
             twice: lane type "driving" only (the 33_e7 default, column
             offroad_driving) and driving + shoulder (column offroad, used in the
             gate) because the 39_180 motorcycle rides in a tyms.xodr shoulder
             lane and its recorded path is 0.12 m from the driving-only boundary.
  a_lat/v are computed on the 50-point uniform-time resample of every arm's
  window path (cvlib's convention); native-grid values are kept as diagnostics
  (30 Hz esmini state has 1-frame position hiccups that inflate a_lat).
  Every executed-arm teleport is also classified after/before the conflict
  arrival frame (teleport_after_conflict) with a sensitivity count.
Background per sample: analytic OBB (sr-tlkeep 41 Tracks.backgrounds grid/gate,
47_bg_diagnosis hit_detail SAT depth), solid = > 2 overlap frames and max
depth >= 0.1 m, target vs ALL rec00 vehicle tracks overlapping 2424-3002 except
ego 39 / target 180, and separately vs the 14 label-0 partners of ego 39
(exp_ego64_39180 lib_bg.bg_ids rule). Pedestrians (0.5 m boxes) are counted in a
separate column. Replay is pushed through the identical loop (zero reference).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
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
PY = sys.executable
SCENARIO = "39_180"
OUT = PROJECT / "results/e9_39_180"
RUNS = PROJECT / "runs"
PLANS = PROJECT / "plans/e9_39_180"
REAL_TRACKS = ROOT / "exp_cross_coverage/data/special_39_180/real_tracks.parquet"
RAW_TRACKS = ROOT / "HetroD-labeler/data/00_tracks.parquet"
LABELS = ROOT / "HetroD-labeler/data/00_labeled_scenarios.json"
DTW_SOURCE = ROOT / "exp_cross_coverage/scripts/180_trajdtw_aggregate.py"
TELEPORT_SOURCE = ROOT / "exp_cross_coverage/scripts/lib.py"
EXIT_SOURCE = ROOT / "exp_cov_sampling/scripts/95_theta_figs.py"
CVLIB_SOURCE = ROOT / "exp_coverage_velocity/scripts/cvlib.py"
E7_SOURCE = ROOT / "sr-tlkeep-experiment/scripts/33_e7_offroad.py"
V41_SOURCE = ROOT / "sr-tlkeep-experiment/scripts/41_validity_layers.py"
V47_SOURCE = ROOT / "sr-tlkeep-experiment/scripts/47_bg_diagnosis.py"
DIAG = PROJECT / "singleton_diagnostic/results/diagnostic.json"
OAT_REUSED = PLANS / "oat_reused_from_db3d7f5cbcdc61f3.csv"
PET_CACHE = PROJECT / "results/bbox_pet_cache_e9.json"
sys.path.insert(0, str(ROOT / "hetero-param"))
from hetero_param.similarity import core as SIMC  # noqa: E402
import importlib
RED = importlib.import_module("hetero_param.sweep.redundancy")  # the package exposes a same-named function

EGO, TARGET, LO, HI, GT_END, FPS = 39, 180, 2424, 3002, 2923, 30.0
SEEDS = [20260910, 20260911, 20260912]
MEASURES = ["pet", "dmin", "alpha", "cpoint", "uc", "dtw"]
BRACKET = [("pet", "pet"), ("dmin", "min_dist"), ("alpha", "conflict_angle"), ("cx", "conflict_x"),
           ("cy", "conflict_y"), ("uc", "agent_arr_speed")]
WRONGWAY_DEG, ALAT_LIM, V_LIM, OUT_FRAC_DEFAULT = 120.0, 5.0, 25.0, 0.05
SOLID_FRAMES, SOLID_DEPTH = 2, 0.1

def uturn_arms():
    """859_881 belongs to no fitted class: every SVD arm is an EXTERNAL (keeptl) basis,
    and there is no OAT / Sobol expert sweep for it."""
    return {
        "svd_extbasis_recon": ("analytic", "basis", "rank 5 keeptl basis (50 scenes; 859_881 absent by construction)",
                               dict(batch_dirs=[RUNS / "svd_uturn_859_881_e9_extbasis_recon"])),
        "svd_extbasis_gauss_h1": ("analytic", "basis+bandwidth", "keeptl basis (50) + h_ext = h_loo(V_train_d) = 0.0824",
                                  dict(batch_dirs=[RUNS / "svd_uturn_859_881_e9_extbasis_gauss_h1"])),
        "svd_extbasis_gauss_h2": ("analytic", "basis+bandwidth", "keeptl basis (50) + h_ext = 2 h_loo = 0.1649",
                                  dict(batch_dirs=[RUNS / "svd_uturn_859_881_e9_extbasis_gauss_h2"])),
        "ours3_recon": ("executed", "none", "the recording's own (theta1, theta2, EndSpeed) decode",
                        dict(batch_dirs=[RUNS / "ours3_uturn_859_881_e9_recon"])),
        "ours3_condkde": ("executed", "bandwidth", "class h = 0.49718 (standardized, 39 keeptl disk contexts) around the scenario vector",
                          dict(batch_dirs=[RUNS / f"ours3_uturn_859_881_e9_condkde_{s}" for s in SEEDS])),
        # SAKURA is scored on THIS substrate too (unlike the 39_180 table, whose SAKURA cells
        # come from the Table-5 gate), so every cell of the uturn group shares one gate.
        "sakura_plain": ("executed", "none", "SAKURA start-end NURBS chord, constant speed",
                         dict(batch_dirs=[RUNS / "sakura_plain_defaults_extra_uturn_859_881"])),
        "sakura_route": ("executed", "none", "SAKURA-route recipe; 0 % routed for this scenario, so the chord is rendered",
                         dict(batch_dirs=[RUNS / "sakura_route_defaults_uturn_859_881"])),
        "sakura_route_kde_cond": ("executed", "bandwidth", "SAKURA (offset, v_avg) KDE, keeptl h = 0.91018 around the scenario's own pair",
                                  dict(batch_dirs=[RUNS / "sakura_route_kde_859_881_cond"])),
    }


ARMS = {
    # name: (kind, class_data_used, definable, sources)
    "svd_pool513_fullfit": ("analytic", "basis", "rank 5 basis from 61 cutinl (513 pool, incl. 39_180)", dict(batch_dirs=[RUNS / "svd_39_180_e9_pool513_fullfit"])),
    "svd_matched_fullfit": ("analytic", "basis", "rank 5 basis from 48 matched cutinl (incl. 39_180)", dict(batch_dirs=[RUNS / "svd_39_180_e9_matched_fullfit"])),
    "svd_matched_logo": ("analytic", "basis", "rank 5 basis from 44 matched cutinl (group of 39_180 excluded)", dict(batch_dirs=[RUNS / "svd_39_180_e9_matched_logo"])),
    "svd_extbasis_gauss_h1": ("analytic", "basis+bandwidth", "LOGO basis (44) + h_ext = h_loo(V_train_d) = 0.0886", dict(batch_dirs=[RUNS / "svd_39_180_e9_extbasis_gauss_h1"])),
    "svd_extbasis_gauss_h2": ("analytic", "basis+bandwidth", "LOGO basis (44) + h_ext = 2 h_loo = 0.1772", dict(batch_dirs=[RUNS / "svd_39_180_e9_extbasis_gauss_h2"])),
    "svd_rawjitter_s0.5": ("analytic", "none", "no basis; sigma_xy = 0.5 m, sigma_dur = 0.5 s chosen externally", dict(batch_dirs=[RUNS / "svd_39_180_e9_rawjitter_s0.5"])),
    "svd_rawjitter_s1": ("analytic", "none", "no basis; sigma_xy = 1.0 m, sigma_dur = 0.5 s chosen externally", dict(batch_dirs=[RUNS / "svd_39_180_e9_rawjitter_s1"])),
    "svd_rawjitter_s2": ("analytic", "none", "no basis; sigma_xy = 2.0 m, sigma_dur = 0.5 s chosen externally", dict(batch_dirs=[RUNS / "svd_39_180_e9_rawjitter_s2"])),
    "ours3_oat_a2641": ("executed", "none", "3 expert knobs (theta1, theta2, v_end), 5 levels each, anchor 2641", dict(oat_anchor=2641)),
    "ours3_oat_a2609": ("executed", "none", "3 expert knobs (theta1, theta2, v_end), 5 levels each, anchor 2609", dict(oat_anchor=2609)),
    "ours3_sobol_a2641": ("executed", "none", "expert ranges theta +/- 15 deg, v_end [0, 30] km/h, scrambled Sobol", dict(batch_dirs=[RUNS / f"ours3_39_180_e9_sobol_{s}" for s in SEEDS])),
    "ours3_condkde_a2641": ("executed", "bandwidth", "class h = 0.5665 (standardized, 48 cutinl disk contexts) around the special vector", dict(batch_dirs=[RUNS / f"ours3_39_180_e9_condkde_{s}" for s in SEEDS])),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float) + "\n")


def ast_load(path, names, namespace):
    src = Path(path).read_text()
    tree = ast.parse(src)
    picked = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in picked} != set(names):
        raise RuntimeError(f"{path}: expected {names}, found {[n.name for n in picked]}")
    module = ast.fix_missing_locations(ast.Module(body=picked, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return {n.name: hashlib.sha256(ast.get_source_segment(src, n).encode()).hexdigest() for n in picked}


def sample_list(arm):
    kind, label, definable, cfg = ARMS[arm]
    if "oat_anchor" in cfg:
        anchor = cfg["oat_anchor"]
        reused = pd.read_csv(OAT_REUSED)
        paths = [Path(p) for p in reused[reused.anchor.eq(anchor)].sample_json]
        for p in sorted((RUNS / f"ours3_{SCENARIO}_e9_oat").rglob("sample.json")):
            if p.parent.name.startswith(f"a{anchor}__"):
                paths.append(p)
        return sorted(set(paths))
    paths = []
    for d in cfg["batch_dirs"]:
        paths.extend(sorted(d.rglob("sample.json")))
    return paths


# ── phase: non-PET via scripts/31 ───────────────────────────────────────────
def phase_nonpet(arms):
    for arm in arms:
        paths = sample_list(arm)
        if not paths:
            print(f"[e9 nonpet] {arm}: no samples yet; skipped", flush=True)
            continue
        lst = OUT / f"e9_{arm}_samples.json"
        write_json(lst, [str(p) for p in paths])
        cmd = [PY, "-B", str(PROJECT / "scripts/31_ours3_nonpet.py"), "--sample-list", str(lst),
               "--output-dir", str(OUT), "--output-prefix", f"e9_{arm}", "--method", arm]
        print(f"[e9 nonpet] {arm}: {len(paths)} samples", flush=True)
        with (OUT / f"e9_{arm}_31_stdout.txt").open("w") as so, (OUT / f"e9_{arm}_31_stderr.txt").open("w") as se:
            rc = subprocess.run(cmd, stdout=so, stderr=se, cwd=PROJECT).returncode
        if rc != 0:
            raise RuntimeError(f"31 failed for {arm}; see {OUT / f'e9_{arm}_31_stderr.txt'}")


def phase_pet(arms):
    paired = [OUT / f"e9_{arm}_paired.csv" for arm in arms if (OUT / f"e9_{arm}_paired.csv").exists()]
    cmd = [PY, "-B", str(PROJECT / "scripts/32_bbox_pet.py"), "--output-dir", str(OUT), "--output-prefix", "e9_bbox_pet",
           "--cache-path", str(PET_CACHE)]
    for p in paired:
        cmd += ["--paired", str(p)]
    print(f"[e9 pet] {len(paired)} paired files", flush=True)
    with (OUT / "e9_bbox_pet_32_stdout.txt").open("w") as so, (OUT / "e9_bbox_pet_32_stderr.txt").open("w") as se:
        rc = subprocess.run(cmd, stdout=so, stderr=se, cwd=PROJECT).returncode
    if rc != 0:
        raise RuntimeError("32 failed; see e9_bbox_pet_32_stderr.txt")


# ── scoring helpers ─────────────────────────────────────────────────────────
class Helpers:
    def __init__(self):
        self.hashes = {}
        ns = {"np": np, "pd": pd}
        self.hashes["dtw"] = ast_load(DTW_SOURCE, ["arc_resample", "dtw2", "traj_dtw"], ns)
        self.traj_dtw = ns["traj_dtw"]
        ns = {"np": np, "FPS": FPS}
        self.hashes["teleport"] = ast_load(TELEPORT_SOURCE, ["teleport_info", "is_teleport"], ns)
        self.is_teleport, self.teleport_info = ns["is_teleport"], ns["teleport_info"]
        ns = {"np": np}
        self.hashes["exit_angle"] = ast_load(EXIT_SOURCE, ["exit_angle"], ns)
        self.exit_angle = ns["exit_angle"]
        ns = {"np": np, "pd": pd, "json": json, "SIMC": SIMC, "FPS": FPS, "HDATA": ROOT / "HetroD-labeler/data",
              "PED_DIM": 0.5, "STATIONARY_M": 1.0, "VEH_CLASSES": ("car", "motorcycle", "truck", "bicycle")}
        self.hashes["v41"] = ast_load(V41_SOURCE, ["Tracks", "_pair_min_dist", "grid_traj"], ns)
        self.Tracks, self.pair_min_dist, self.grid_traj = ns["Tracks"], ns["_pair_min_dist"], ns["grid_traj"]
        ns = {"np": np, "RED": RED, "FPS": FPS}
        self.hashes["v47"] = ast_load(V47_SOURCE, ["_sat_depth", "hit_detail"], ns)
        self.hit_detail = ns["hit_detail"]
        # off-road union (33_e7 imports trajectory_plots/xodr_lanes and run_label_lib)
        sys.path.insert(0, str(ROOT / "sr-tlkeep-experiment/scripts"))
        sys.path.insert(0, str(ROOT / "trajectory_plots"))
        spec = importlib.util.spec_from_file_location("e7_offroad", E7_SOURCE)
        e7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(e7)
        self.area = e7.drivable_union()                       # 33_e7 as specified: lane type "driving" only
        self.area_shoulder = self.union_with_shoulder(e7)      # same recipe, types ("driving", "shoulder")
        self.out_frac = float(e7.OUT_FRAC)
        from shapely import vectorized
        self.contains = lambda x, y: vectorized.contains(self.area, np.asarray(x, float), np.asarray(y, float))
        self.contains_shoulder = lambda x, y: vectorized.contains(self.area_shoulder, np.asarray(x, float), np.asarray(y, float))
        self.hashes["e7"] = {"33_e7_offroad.py": sha256(E7_SOURCE)}
        tracks = pd.read_parquet(REAL_TRACKS)
        a = tracks[tracks.role.eq("actor") & tracks.scenario_id.eq(SCENARIO)].sort_values("frame")
        e = tracks[tracks.role.eq("ego") & tracks.scenario_id.eq(SCENARIO)].sort_values("frame")
        self.gt_actor = a
        self.gt_actor_xy = a[["x", "y"]].to_numpy(float)
        self.gt_ego = e
        t0 = time.monotonic()
        self.tracks = self.Tracks()
        self.bgs_all = self.tracks.backgrounds(EGO, TARGET, LO, HI)
        self.label0 = self.label0_ids()
        self.bgs_label0 = [b for b in self.bgs_all if b["tid"] in self.label0]
        print(f"[e9 score] backgrounds: {len(self.bgs_all)} tracks overlap {LO}-{HI} "
              f"({sum(1 for b in self.bgs_all if not b['ped'])} vehicles, {sum(1 for b in self.bgs_all if b['ped'])} pedestrians); "
              f"label-0 partners present {len(self.bgs_label0)}/{len(self.label0)} ({time.monotonic() - t0:.0f}s)", flush=True)

    @staticmethod
    def union_with_shoulder(e7):
        """33_e7.drivable_union recipe (ds 0.3, buffer BUF, holes filled) with lane types driving + shoulder.
        The 39_180 motorcycle rides in a tyms.xodr 'shoulder' lane; the driving-only union puts the recorded
        GT path 0.12 m from its boundary, so the specified rule penalises GT-faithful samples. Both are reported."""
        import xodr_lanes
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        roads, _ = xodr_lanes.parse(e7.XODR)
        polys = []
        for road in roads.values():
            for inner, outer in road.lane_edges(ds=0.3, types=("driving", "shoulder")).values():
                try:
                    poly = Polygon(np.vstack([inner, outer[::-1]])).buffer(0)
                    if poly.is_valid and poly.area > 0:
                        polys.append(poly)
                except Exception:  # noqa: BLE001
                    continue
        area = unary_union(polys).buffer(e7.BUF)
        return e7._fill_interior_rings(area)

    def label0_ids(self):
        """exp_ego64_39180/scripts/lib_bg.py bg_ids rule: label-0 partners of ego 39 with >= 2 % cover of 2424-2923."""
        lab = json.loads(LABELS.read_text())
        ids = sorted(int(v["actor_id"]) for v in lab.values() if int(v["ego_id"]) == EGO and v.get("label_idx") == 0)
        n_win = GT_END - LO + 1
        out = []
        for i in ids:
            t = self.tracks.by_id.get(i)
            if t is None:
                continue
            m = (t["frame"] >= LO) & (t["frame"] <= GT_END)
            if m.sum() >= 2 and m.sum() / n_win >= 0.02:
                out.append(i)
        return out

    def kinematics(self, frame, x, y, resample=True):
        """cvlib.kinematics formula. resample=True: the sample's window path interpolated onto 50 uniform time
        points over its own duration (cvlib's 50-point convention, identical for every arm); False: native grid."""
        tq = (np.asarray(frame, float) - LO) / FPS
        if resample:
            t50 = np.linspace(tq[0], tq[-1], 50)
            x, y, tq = np.interp(t50, tq, x), np.interp(t50, tq, y), t50
        vx, vy = np.gradient(x, tq), np.gradient(y, tq)
        sp = np.hypot(vx, vy)
        ax, ay = np.gradient(vx, tq), np.gradient(vy, tq)
        with np.errstate(invalid="ignore", divide="ignore"):
            alat = np.abs(vx * ay - vy * ax) / np.maximum(sp, 1e-6)
        alat = np.where(sp >= 1.0, alat, 0.0)
        return float(sp.max()), float(np.nanmax(alat))

    def validity(self, target, analytic):
        g = target.sort_values("frame")
        f, x, y = (g[k].to_numpy(float) for k in ("frame", "x", "y"))
        r = {}
        if analytic:
            r.update(teleport=False, teleport_status="analytic_kinematics", teleport_ps_max=np.nan, teleport_state_max=np.nan)
        else:
            gg = pd.DataFrame(dict(frame=f, x=x, y=y, speed=g.speed_mps.to_numpy(float)))
            ps, st = self.teleport_info(gg, FPS)
            r.update(teleport=bool(self.is_teleport(gg, FPS)), teleport_status="executed", teleport_ps_max=ps, teleport_state_max=st)
            tt = f / FPS
            dt = np.diff(tt)
            step = np.hypot(np.diff(x), np.diff(y))[dt > 0] / dt[dt > 0]
            idx = np.where(step[10:] > 30.0)[0] + 10
            r["teleport_first_frame"] = float(f[idx[0] + 1]) if r["teleport"] and len(idx) else np.nan
            r["teleport_n_steps"] = int(len(idx)) if r["teleport"] else 0
        ang = self.exit_angle(self.gt_actor_xy, np.column_stack([x, y]))
        r.update(exit_angle_deg=ang, wrongway=bool(np.isfinite(ang) and ang > WRONGWAY_DEG))
        vmax, alat = self.kinematics(f, x, y, resample=True)
        vmax_n, alat_n = self.kinematics(f, x, y, resample=False)
        r.update(v_max_mps=vmax, alat_max_mps2=alat, alat_flag=bool(alat > ALAT_LIM), vmax_flag=bool(vmax > V_LIM),
                 v_max_native_mps=vmax_n, alat_max_native_mps2=alat_n)
        if not analytic:
            r["state_v_max_mps"] = float(g.speed_mps.max())
        frac_drv = 1.0 - float(self.contains(x, y).mean())
        frac_sh = 1.0 - float(self.contains_shoulder(x, y).mean())
        r.update(offroad_driving_frac=frac_drv, offroad_driving=bool(frac_drv > self.out_frac),
                 offroad_frac=frac_sh, offroad=bool(frac_sh > self.out_frac), n_points=len(f))
        r["valid_kinematic"] = not (r["teleport"] or r["wrongway"] or r["alat_flag"] or r["vmax_flag"])
        r["valid"] = r["valid_kinematic"] and not r["offroad"]
        r["valid_strict_driving_only"] = r["valid"] and not r["offroad_driving"]
        return r

    def background(self, target, length, width, hi=None, prefix="bg_"):
        """Analytic OBB solid hits of the target rows with frame <= hi (hi=HI: metadata window incl. the
        post-goal tail; hi=GT_END: within the target's GT support 2424-2923). Columns get `prefix`."""
        g = target.sort_values("frame")
        hi = HI if hi is None else hi
        var = self.grid_traj(g.frame.to_numpy(float), g.x.to_numpy(float), g.y.to_numpy(float), LO, hi, length, width)
        r = dict(bg_all_solid_hits=0, bg_all_any_hits=0, bg_all_min_dist=np.nan, bg_all_solid_ids="", bg_all_max_depth=0.0,
                 bg_label0_solid_hits=0, bg_label0_any_hits=0, bg_label0_min_dist=np.nan, bg_label0_solid_ids="",
                 bg_ped_solid_hits=0, bg_ped_any_hits=0, bg_first_solid_frame=np.nan,
                 bg_all_solid_moving=0, bg_all_solid_stationary=0)
        if var is None:
            r["bg_status"] = "too_short"
            return {prefix + k[3:]: v for k, v in r.items()}
        r["bg_status"] = "ok"
        vdiag = float(np.hypot(var.length, var.width)) / 2.0
        mins = {"all": np.inf, "label0": np.inf}
        for bg in self.bgs_all:
            md = self.pair_min_dist(var, bg["traj"])
            if md is None:
                continue
            groups = ["all"] + (["label0"] if bg["tid"] in self.label0 else [])
            if bg["ped"]:
                if md <= vdiag + bg["diag"] + 0.2:
                    hd = self.hit_detail(var, bg["traj"])
                    if hd is not None:
                        r["bg_ped_any_hits"] += 1
                        if hd["n_frames"] > SOLID_FRAMES and hd["max_depth"] >= SOLID_DEPTH:
                            r["bg_ped_solid_hits"] += 1
                continue
            for gname in groups:
                mins[gname] = min(mins[gname], md)
            if md <= vdiag + bg["diag"] + 0.2:
                hd = self.hit_detail(var, bg["traj"])
                if hd is None:
                    continue
                solid = hd["n_frames"] > SOLID_FRAMES and hd["max_depth"] >= SOLID_DEPTH
                for gname in groups:
                    r[f"bg_{gname}_any_hits"] += 1
                    if solid:
                        r[f"bg_{gname}_solid_hits"] += 1
                        r[f"bg_{gname}_solid_ids"] += f"{bg['tid']}:{hd['n_frames']}f/{hd['max_depth']:.2f}m;"
                if solid:
                    r["bg_all_max_depth"] = max(r["bg_all_max_depth"], float(hd["max_depth"]))
                    r["bg_all_solid_moving" if bg["moving"] else "bg_all_solid_stationary"] += 1
                    if not np.isfinite(r["bg_first_solid_frame"]) or hd["first_frame"] < r["bg_first_solid_frame"]:
                        r["bg_first_solid_frame"] = float(hd["first_frame"])
        r["bg_all_min_dist"] = mins["all"] if np.isfinite(mins["all"]) else np.nan
        r["bg_label0_min_dist"] = mins["label0"] if np.isfinite(mins["label0"]) else np.nan
        return {prefix + k[3:]: v for k, v in r.items()}


def load_sample_meta(path):
    s = json.loads(Path(path).read_text())
    sc = s.get("context", {}).get("source_context", s.get("job", {}).get("source_context", {})) or {}
    meta = dict(seed=s.get("seed", sc.get("seed")), draw=s.get("draw", sc.get("draw")),
                sigma_or_h=s.get("sigma_or_h"), analytic=bool(s.get("analytic", False)),
                upstream_status=s.get("status"), run_id=s.get("run_id"),
                duration_raw_s=s.get("decode", {}).get("duration_raw_s"), duration_applied_s=s.get("decode", {}).get("duration_applied_s"),
                duration_clipped=s.get("decode", {}).get("duration_clipped"),
                end_speed_clipped=sc.get("end_speed_clipped"), oat_level=sc.get("level"),
                end_speed_requested_kmh=(sc.get("sampler_parameters_requested") or {}).get("interaction_end_speed_kmh"),
                class_data_used_in_sample=s.get("class_data_used", sc.get("class_data_used")))
    if "z" in s:
        meta["z"] = json.dumps([round(v, 6) for v in s["z"]])
    return meta


def phase_score(arms):
    H = Helpers()
    pet_all = pd.read_csv(OUT / "e9_bbox_pet_paired.csv")
    pet_replay = pd.read_csv(OUT / "e9_bbox_pet_replay.csv").iloc[0]
    rows, replay_row = [], None
    for arm in arms:
        kind, label, definable, cfg = ARMS[arm]
        paired_path = OUT / f"e9_{arm}_paired.csv"
        if not paired_path.exists():
            print(f"[e9 score] {arm}: no paired file; skipped", flush=True)
            continue
        paired = pd.read_csv(paired_path)
        pet = pet_all[pet_all.input_paired_path.map(lambda p: str(Path(p).resolve())).eq(str(paired_path.resolve()))].set_index("input_case_index")
        assert len(pet) == len(paired), (arm, len(pet), len(paired))
        if replay_row is None:
            rep = pd.read_csv(OUT / f"e9_{arm}_replay.csv").iloc[0]
            gt = H.gt_actor
            tg = gt[gt.frame.between(LO, GT_END)]
            tidy_gt = pd.DataFrame(dict(frame=tg.frame.astype(float), x=tg.x, y=tg.y, speed_mps=tg.speed))
            v = H.validity(tidy_gt, analytic=True)
            b = H.background(tidy_gt, float(tg.length.median()), float(tg.width.median()))
            b.update(H.background(tidy_gt, float(tg.length.median()), float(tg.width.median()), hi=GT_END, prefix="gts_"))
            dtw, cov = H.traj_dtw(tg[["x", "y"]], gt[gt.frame.between(LO, GT_END)][["x", "y"]])
            replay_row = dict(arm="replay", kind="recorded", class_data_used="none", seed=None, draw=None, sample_id=f"replay_{SCENARIO}",
                              score_status="ok", pet_score_status="ok", clock_aligned=True,
                              pet=float(pet_replay.pet), min_dist=float(rep.min_dist), conflict_angle=float(rep.conflict_angle),
                              conflict_x=float(rep.conflict_x), conflict_y=float(rep.conflict_y), agent_arr_speed=float(rep.agent_arr_speed),
                              err_pet=0.0, err_dmin=0.0, err_alpha=0.0, err_cpoint=0.0, err_uc=0.0, err_dtw=float(dtw), dtw_coverage=cov,
                              **v, **b, analytic=False, upstream_status="recorded")
            replay_row["teleport_status"] = "recorded_kinematics"
            replay_row.update(teleport_after_conflict=False, valid_preconflict_sens=bool(v["valid"]))
            rows.append(replay_row)
        t0 = time.monotonic()
        for idx, r in paired.iterrows():
            p = pet.loc[idx]
            meta = load_sample_meta(r.sample_json)
            row = dict(arm=arm, kind=kind, class_data_used=label, sample_id=r.sample_id, sample_json=r.sample_json,
                       score_status=r.score_status, score_detail=r.get("score_detail"), pet_score_status=p.pet_score_status,
                       clock_aligned=bool(r.clock_aligned), theta1_deg=r.get("theta1_deg"), theta2_deg=r.get("theta2_deg"),
                       interaction_end_speed_kmh=r.get("interaction_end_speed_kmh"), **meta)
            real_pet, gen_pet = float(p.real_pet), float(p.generated_pet)
            both = np.isfinite([real_pet, gen_pet]).all()
            row.update(pet=gen_pet, real_pet=real_pet, pet_type=p.generated_pet_type, pet_no_event=bool(np.isinf(gen_pet)),
                       pet_flip=bool(both and np.sign(real_pet) != np.sign(gen_pet)),
                       min_dist=r.generated_min_dist, conflict_angle=r.generated_conflict_angle, conflict_x=r.generated_conflict_x,
                       conflict_y=r.generated_conflict_y, agent_arr_speed=r.generated_agent_arr_speed, min_ttc=r.generated_min_ttc,
                       err_pet=abs(real_pet - gen_pet) if both else np.nan,
                       err_dmin=abs(r.generated_min_dist - r.real_min_dist),
                       err_alpha=abs(r.generated_conflict_angle - r.real_conflict_angle),
                       err_cpoint=float(np.hypot(r.generated_conflict_x - r.real_conflict_x, r.generated_conflict_y - r.real_conflict_y)),
                       err_uc=abs(r.generated_agent_arr_speed - r.real_agent_arr_speed),
                       ego_replay_abs_position_m_max=r.get("ego_replay_abs_position_m_max"))
            if r.score_status == "failed" or not isinstance(r.get("trajectory_path"), str):
                row.update(err_dtw=np.nan, valid=False, validity_status="score_failed")
                rows.append(row)
                continue
            tidy = pd.read_parquet(r.trajectory_path)
            target = tidy[tidy.role.eq("target") & tidy.frame.between(LO - .5, HI + .5)].sort_values("frame")
            dtw, cov = H.traj_dtw(target[["x", "y"]], H.gt_actor[["x", "y"]]) if len(target) >= 2 else (np.nan, np.nan)
            row.update(err_dtw=float(dtw), dtw_coverage=cov, generated_samples=len(target),
                       generated_end_frame=float(target.frame.max()), generated_start_frame=float(target.frame.min()))
            row.update(H.validity(target, analytic=meta["analytic"]))
            arr = float(r.get("generated_agent_arr_frame", np.nan))
            row["generated_agent_arr_frame"] = arr
            row["teleport_after_conflict"] = bool(row.get("teleport") and np.isfinite(row.get("teleport_first_frame", np.nan))
                                                  and np.isfinite(arr) and row["teleport_first_frame"] > arr)
            row["valid_preconflict_sens"] = bool((not row["wrongway"]) and (not row["alat_flag"]) and (not row["vmax_flag"])
                                                 and (not row["offroad"]) and ((not row["teleport"]) or row["teleport_after_conflict"]))
            row.update(H.background(target, float(target.length.iloc[0]), float(target.width.iloc[0])))
            row.update(H.background(target, float(target.length.iloc[0]), float(target.width.iloc[0]), hi=GT_END, prefix="gts_"))
            row["validity_status"] = "scored"
            rows.append(row)
        print(f"[e9 score] {arm}: {len(paired)} samples ({time.monotonic() - t0:.0f}s)", flush=True)
    S = pd.DataFrame(rows)
    S["valid"] = S["valid"].fillna(False).astype(bool)
    S["scored_ok"] = S.score_status.eq("ok") & S.pet_score_status.eq("ok")
    S["valid_scored"] = S.valid & S.scored_ok
    # composite: arm set = arms with >= 1 valid scored sample; b_k = max over the set of the per-arm medians
    arm_set = [a for a in S.arm.unique() if a != "replay" and S[S.arm.eq(a) & S.valid_scored].shape[0] > 0]
    med = {a: {k: float(np.nanmedian(S.loc[S.arm.eq(a) & S.valid_scored, f"err_{k}"])) if S.loc[S.arm.eq(a) & S.valid_scored, f"err_{k}"].notna().any() else np.nan
               for k in MEASURES} for a in arm_set}
    b = {k: float(np.nanmax([med[a][k] for a in arm_set])) for k in MEASURES}
    used = [k for k in MEASURES if np.isfinite(b[k]) and b[k] > 0]
    for k in MEASURES:
        S[f"norm_{k}"] = S[f"err_{k}"] / b[k] if k in used else np.nan
    S["composite"] = S[[f"norm_{k}" for k in used]].mean(axis=1, skipna=False)
    S["composite_nanmean"] = S[[f"norm_{k}" for k in used]].mean(axis=1, skipna=True)
    # nearest / median / worst among valid scored samples with a finite composite
    S["rank_role"] = ""
    for a in S.arm.unique():
        g = S[S.arm.eq(a) & S.valid_scored & S.composite.notna()].sort_values("composite")
        if a == "replay" or len(g) == 0:
            continue
        S.loc[g.index[0], "rank_role"] = "nearest"
        S.loc[g.index[-1], "rank_role"] = "worst" if len(g) > 1 else "nearest"
        if len(g) > 2:
            S.loc[g.index[len(g) // 2], "rank_role"] = "median"
    S.to_csv(OUT / "T8_samples.csv", index=False)
    build_table(S, med, b, used, arm_set, H)
    return S


def iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) else np.nan


def build_table(S, med, b, used, arm_set, H):
    diag = json.loads(DIAG.read_text())
    rep = S[S.arm.eq("replay")].iloc[0]
    T = []
    T.append(dict(arm="replay", class_data_used="none", kind="recorded", definable="recorded target; zero reference",
                  n_requested=1, n_executed=1, n_valid=int(rep.valid), n_scored_ok=1,
                  teleport_n=0, offroad_n=int(rep.offroad), wrongway_n=int(rep.wrongway), alat_n=int(rep.alat_flag), vmax_n=int(rep.vmax_flag),
                  offroad_driving_only_n=int(rep.offroad_driving), n_valid_strict_driving_only=int(rep.valid_strict_driving_only),
                  n_valid_kinematic=int(rep.valid_kinematic), offroad_frac_median=float(rep.offroad_frac),
                  teleport_after_conflict_n=0, n_valid_preconflict_sens=int(rep.valid),
                  bg_solid_rate_all=float(rep.bg_all_solid_hits > 0), bg_solid_rate_label0=float(rep.bg_label0_solid_hits > 0),
                  bg_any_rate_all=float(rep.bg_all_any_hits > 0), bg_ped_solid_rate=float(rep.bg_ped_solid_hits > 0),
                  bg_solid_rate_all_gtsupport=float(rep.gts_all_solid_hits > 0), bg_solid_rate_label0_gtsupport=float(rep.gts_label0_solid_hits > 0),
                  bg_solid_moving_rate_gtsupport=float(rep.gts_all_solid_moving > 0), bg_solid_stationary_rate_gtsupport=float(rep.gts_all_solid_stationary > 0),
                  bg_solid_rate_all_valid_gtsupport=float(rep.gts_all_solid_hits > 0),
                  bg_all_min_dist_median=rep.bg_all_min_dist, esmini_agent_bg_collision_frames="see e9_esmini_bg_collisions.csv",
                  pet_median=rep.pet, dmin_median=rep.min_dist, alpha_median=rep.conflict_angle, cx_median=rep.conflict_x,
                  cy_median=rep.conflict_y, uc_median=rep.agent_arr_speed,
                  **{f"err_{k}_median": 0.0 for k in MEASURES}, **{f"err_{k}_nearest": 0.0 for k in MEASURES},
                  D_m=0.0, pet_iqr=0.0, dmin_iqr=0.0, bracket_n_of_6=np.nan, bracket="reference",
                  duration_clipped_n=0, end_speed_clipped_n=0, pet_no_event_n=0, pet_flip_n=0))
    T.append(dict(arm="svd_d5_singleton_N1", class_data_used="none", kind="undefined",
                  definable=f"N=1: centered rank {diag['centered_matrix_rank']}, singular values {diag['returned_singular_values']}, "
                            f"LOO bandwidth {diag['kde_bandwidth']} ({diag['kde_bandwidth_error']['message']})",
                  n_requested=0, n_executed=0, n_valid=0, n_scored_ok=0, teleport_n=0, offroad_n=0, wrongway_n=0, alat_n=0, vmax_n=0,
                  bg_solid_rate_all=np.nan, bg_solid_rate_label0=np.nan, bg_any_rate_all=np.nan, bg_ped_solid_rate=np.nan,
                  bg_all_min_dist_median=np.nan, esmini_agent_bg_collision_frames="n/a",
                  pet_median=np.nan, dmin_median=np.nan, alpha_median=np.nan, cx_median=np.nan, cy_median=np.nan, uc_median=np.nan,
                  **{f"err_{k}_median": np.nan for k in MEASURES}, **{f"err_{k}_nearest": np.nan for k in MEASURES},
                  D_m=np.nan, pet_iqr=np.nan, dmin_iqr=np.nan, bracket_n_of_6=np.nan, bracket="undefined (no samples)",
                  duration_clipped_n=0, end_speed_clipped_n=0, pet_no_event_n=0, pet_flip_n=0))
    for arm in [a for a in ARMS if a in set(S.arm)]:
        kind, label, definable, cfg = ARMS[arm]
        g = S[S.arm.eq(arm)]
        v = g[g.valid_scored]
        nearest = g[g.rank_role.eq("nearest")]
        n_req = len(g)
        n_exec = int(g.upstream_status.eq("completed").sum())
        brackets = {}
        for key, col in BRACKET:
            vals = v[col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            brackets[key] = bool(len(vals) >= 2 and vals.min() <= rep[col] <= vals.max()) if len(vals) else False
        Dm = float(np.mean([med[arm][k] / b[k] for k in used])) if arm in med and all(np.isfinite(med[arm][k]) for k in used) else np.nan
        T.append(dict(arm=arm, class_data_used=label, kind=kind, definable=definable,
                      n_requested=n_req, n_executed=n_exec, n_valid=int(g.valid.sum()), n_scored_ok=int(g.scored_ok.sum()),
                      n_valid_scored=len(v),
                      teleport_n=int(g.teleport.fillna(False).astype(bool).sum()), offroad_n=int(g.offroad.fillna(False).astype(bool).sum()),
                      wrongway_n=int(g.wrongway.fillna(False).astype(bool).sum()), alat_n=int(g.alat_flag.fillna(False).astype(bool).sum()),
                      vmax_n=int(g.vmax_flag.fillna(False).astype(bool).sum()),
                      offroad_driving_only_n=int(g.offroad_driving.fillna(False).astype(bool).sum()),
                      n_valid_kinematic=int(g.valid_kinematic.fillna(False).astype(bool).sum()),
                      offroad_frac_median=float(np.nanmedian(g.offroad_frac)) if g.offroad_frac.notna().any() else np.nan,
                      n_valid_strict_driving_only=int(g.valid_strict_driving_only.fillna(False).astype(bool).sum()),
                      teleport_after_conflict_n=int(g.teleport_after_conflict.fillna(False).astype(bool).sum()),
                      n_valid_preconflict_sens=int(g.valid_preconflict_sens.fillna(False).astype(bool).sum()),
                      bg_solid_rate_all=float((g.bg_all_solid_hits.fillna(0) > 0).mean()) if len(g) else np.nan,
                      bg_solid_rate_label0=float((g.bg_label0_solid_hits.fillna(0) > 0).mean()) if len(g) else np.nan,
                      bg_any_rate_all=float((g.bg_all_any_hits.fillna(0) > 0).mean()) if len(g) else np.nan,
                      bg_ped_solid_rate=float((g.bg_ped_solid_hits.fillna(0) > 0).mean()) if len(g) else np.nan,
                      bg_solid_rate_all_gtsupport=float((g.gts_all_solid_hits.fillna(0) > 0).mean()) if len(g) else np.nan,
                      bg_solid_rate_label0_gtsupport=float((g.gts_label0_solid_hits.fillna(0) > 0).mean()) if len(g) else np.nan,
                      bg_solid_moving_rate_gtsupport=float((g.gts_all_solid_moving.fillna(0) > 0).mean()) if len(g) else np.nan,
                      bg_solid_stationary_rate_gtsupport=float((g.gts_all_solid_stationary.fillna(0) > 0).mean()) if len(g) else np.nan,
                      bg_solid_rate_all_valid_gtsupport=float((v.gts_all_solid_hits.fillna(0) > 0).mean()) if len(v) else np.nan,
                      bg_all_min_dist_median=float(np.nanmedian(g.bg_all_min_dist)) if g.bg_all_min_dist.notna().any() else np.nan,
                      esmini_agent_bg_collision_frames="see e9_esmini_bg_collisions.csv" if kind == "executed" else "n/a (analytic)",
                      pet_median=float(np.nanmedian(v.pet[np.isfinite(v.pet)])) if np.isfinite(v.pet).any() else np.nan,
                      dmin_median=float(np.nanmedian(v.min_dist)) if len(v) else np.nan,
                      alpha_median=float(np.nanmedian(v.conflict_angle)) if len(v) else np.nan,
                      cx_median=float(np.nanmedian(v.conflict_x)) if len(v) else np.nan,
                      cy_median=float(np.nanmedian(v.conflict_y)) if len(v) else np.nan,
                      uc_median=float(np.nanmedian(v.agent_arr_speed)) if len(v) else np.nan,
                      **{f"err_{k}_median": med.get(arm, {}).get(k, np.nan) for k in MEASURES},
                      **{f"err_{k}_nearest": float(nearest[f"err_{k}"].iloc[0]) if len(nearest) else np.nan for k in MEASURES},
                      nearest_sample_id=nearest.sample_id.iloc[0] if len(nearest) else "",
                      D_m=Dm, pet_iqr=iqr(v.pet), dmin_iqr=iqr(v.min_dist),
                      bracket_n_of_6=int(sum(brackets.values())), bracket=";".join(f"{k}={'Y' if ok else 'N'}" for k, ok in brackets.items()),
                      duration_clipped_n=int(g.duration_clipped.fillna(False).astype(bool).sum()) if "duration_clipped" in g else 0,
                      end_speed_clipped_n=int(g.end_speed_clipped.fillna(False).astype(bool).sum()) if "end_speed_clipped" in g else 0,
                      pet_no_event_n=int(g.pet_no_event.fillna(False).astype(bool).sum()), pet_flip_n=int(g.pet_flip.fillna(False).astype(bool).sum()),
                      pet_finite_n=int(np.isfinite(v.pet).sum())))
    T = pd.DataFrame(T)
    T.to_csv(OUT / "T8_singleton_table.csv", index=False)
    write_json(OUT / "T8_composite.json", dict(arm_set=arm_set, b_k=b, measures_used=used, per_arm_medians=med,
                                               composite_rule="D_m = mean over measures of median_k(m) / b_k, b_k = max over the arm set of the per-arm median; medians over valid scored samples (score ok, PET ok, no teleport/wrong-way/a_lat/v/off-road flag)"))
    write_json(OUT / "score_helpers_hashes.json", H.hashes)
    print(T[["arm", "class_data_used", "n_requested", "n_valid", "n_valid_scored" if "n_valid_scored" in T else "n_valid", "D_m"]].to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", nargs="+", default=["nonpet", "pet", "score"])
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--scenario", choices=("39_180", "uturn_859_881"), default="39_180")
    parser.add_argument("--radius3", action="store_true", help="score the separate U-turn L=3 original-anchor batch")
    parser.add_argument("--label-v2", action="store_true", help="score updated 859_881 label in isolated outputs")
    parser.add_argument("--actor-start-frame", type=int, default=None,
                        help="score isolated label-v2 runs whose target support starts at this frame")
    args = parser.parse_args()
    if args.label_v2 and (args.scenario != "uturn_859_881" or not args.radius3):
        parser.error("--label-v2 requires --scenario uturn_859_881 --radius3")
    if args.radius3 and args.scenario != "uturn_859_881":
        parser.error("--radius3 requires --scenario uturn_859_881")
    if args.actor_start_frame is not None and not (args.label_v2 and args.radius3):
        parser.error("--actor-start-frame requires --radius3 --label-v2")
    if args.scenario != "39_180":
        global SCENARIO, OUT, PLANS, REAL_TRACKS, DIAG, OAT_REUSED, PET_CACHE, ARMS
        global EGO, TARGET, LO, HI, GT_END
        SCENARIO = "859_881"
        OUT = PROJECT / "results/e9_uturn_859_881"
        PLANS = PROJECT / "plans/e9_uturn_859_881"
        REAL_TRACKS = ROOT / "exp_cross_coverage/data/uturn_859_881/real_tracks.parquet"
        DIAG = PROJECT / "singleton_diagnostic/results_uturn_859_881/diagnostic.json"
        OAT_REUSED = PLANS / "oat_reused.csv"
        PET_CACHE = PROJECT / "results/bbox_pet_cache_e9_uturn.json"
        EGO, TARGET, LO, HI, GT_END = 859, 881, 14435, 15711, 15630
        ARMS = uturn_arms()
        if args.radius3:
            OUT = PROJECT / "results/e9_uturn_859_881_l3"
            PET_CACHE = PROJECT / "results/bbox_pet_cache_e9_uturn_l3.json"
            ARMS["ours3_recon"][3]["batch_dirs"] = [RUNS / "ours3_uturn_859_881_e9_l3_recon"]
            ARMS["ours3_condkde"][3]["batch_dirs"] = [RUNS / f"ours3_uturn_859_881_e9_l3_condkde_{s}" for s in SEEDS]
        if args.label_v2:
            entry = json.loads(LABELS.read_text())["859_881"]
            LO, HI = int(entry["min_frame"]), int(entry["max_frame"])
            GT_END = int(pd.read_parquet(REAL_TRACKS).query("role == 'actor'").frame.max())
            OUT = PROJECT / "results/e9_uturn_859_881_l3_labelv2"
            PET_CACHE = PROJECT / "results/bbox_pet_cache_e9_uturn_l3_labelv2.json"
            ARMS["ours3_recon"][3]["batch_dirs"] = [RUNS / "ours3_uturn_859_881_e9_l3_labelv2_recon"]
            ARMS["ours3_condkde"][3]["batch_dirs"] = [RUNS / f"ours3_uturn_859_881_e9_l3_labelv2_condkde_{s}" for s in SEEDS]
            for arm in ("svd_extbasis_recon", "svd_extbasis_gauss_h1", "svd_extbasis_gauss_h2"):
                ARMS[arm][3]["batch_dirs"] = [RUNS / f"svd_uturn_859_881_e9_labelv2_{arm.removeprefix('svd_')}"]
            ARMS["sakura_plain"][3]["batch_dirs"] = [RUNS / "sakura_plain_defaults_extra_uturn_859_881_labelv2"]
            ARMS["sakura_route"][3]["batch_dirs"] = [RUNS / "sakura_route_defaults_uturn_859_881_labelv2"]
            ARMS["sakura_route_kde_cond"][3]["batch_dirs"] = [RUNS / "sakura_route_kde_859_881_cond_labelv2"]
        if args.actor_start_frame is not None:
            start = args.actor_start_frame
            def latest_batch(root):
                candidates = [p for p in root.iterdir() if p.is_dir() and (p / "batch_status.json").exists()]
                if not candidates:
                    raise FileNotFoundError(f"no completed batch under {root}")
                return max(candidates, key=lambda p: (p / "batch_status.json").stat().st_mtime)
            REAL_TRACKS = ROOT / f"exp_cross_coverage/data/uturn_859_881_start{start}/real_tracks.parquet"
            GT_END = int(pd.read_parquet(REAL_TRACKS).query("role == 'actor'").frame.max())
            OUT = PROJECT / f"results/e9_uturn_859_881_l3_labelv2_start{start}"
            PET_CACHE = PROJECT / f"results/bbox_pet_cache_e9_uturn_l3_labelv2_start{start}.json"
            ARMS["ours3_recon"][3]["batch_dirs"] = [latest_batch(RUNS / f"ours3_uturn_859_881_e9_l3_labelv2_start{start}_recon")]
            ARMS["ours3_condkde"][3]["batch_dirs"] = [latest_batch(RUNS / f"ours3_uturn_859_881_e9_l3_labelv2_start{start}_condkde_{s}") for s in SEEDS]
    args.arms = args.arms or list(ARMS)
    OUT.mkdir(parents=True, exist_ok=True)
    if "nonpet" in args.phase:
        phase_nonpet(args.arms)
    if "pet" in args.phase:
        phase_pet(list(ARMS))
    if "score" in args.phase:
        phase_score(list(ARMS))


if __name__ == "__main__":
    main()
