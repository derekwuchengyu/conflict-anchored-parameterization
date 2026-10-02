#!/usr/bin/env python3
"""E10 stage 2: regenerate the exp_cross_coverage d-sweep (L in {10,5,3,2}) and the 3x3
(pos, frame) anchor sweep under the Euclidean-DISK conflict window (paper 04_impl.tex):
q- / q+ are the last / first target samples at Euclidean distance L from the spatial
anchor q_c (chords == L exactly), instead of the arc-length +-step points of the cached
arc arms.

Per subset (one subprocess each — exp_cross_coverage/scripts/lib.py chdirs and reads
CC_SUBSET at import) and per arm:
  * base xosc = the cached ARC-LENGTH base of that arm (120_stepm_sweep / 80_cp3_method
    dirs for the d-sweep, 150_anchor_sweep._xbase_for dirs for the anchor arms);
  * q_c = the recorded target sample (real_tracks.parquet, role == actor) at the arm's anchor
    frame, resolved by the SAME rule the base generator used: d-sweep arms = the pipeline's
    legacy PET frame (parampath.pet_anchors -> labeler pet_frame1), anchor arms =
    generate._anchor_frame_for(<pos anchor>) (force_anchor=True in 150). When the base still
    holds its complete 3 weighted CPs the middle CP is verified to coincide (<=1e-3 m) with
    q_c (100 % on every arm in the 2026-09-10 probe); bases whose legacy spacing guards
    dropped interior CPs (1-2 weighted CPs, e.g. 35/61 cutinl bases) are patched to the full
    3-point window anyway (paper rule; same as 16/20 for the ours3 cohort) and flagged in the
    status table (base_n_weighted, qc_in_base_cps). An anchor frame outside the episode
    window (pipeline PET bug: event outside the label window) is "unavailable" - no render;
  * q-/q+ from 16_disk_window_extract.disk_points at radius L = the arm's step; a window
    boundary inside the disk -> status "unavailable:<why>", NOT rendered (paper rule);
  * the 3 weighted CPs are replaced (exp_coverage_velocity cvrender.patch_cps semantics:
    7 CPs, knots [0,0,0,0,0.5,1,1.5,2,2,2,2], weight 5.0; uturn conflict-W=8 keeps the
    middle-CP weight 8 as 120/150 _patch_conflict_weight does) in a patched copy of the
    base, then EX.to_esmini_replay + EX.run_and_extract exactly as 150.render_one;
  * rows via PL.trajs_to_rows, teleport screen lib.is_teleport, trim_lead_still, and the
    gate-v2 descriptor call chain of 170_pet_gate_recompute.compute_desc.

Outputs: runs/disk_sweep/<subset>/<arm>/<base-stem>/{patched_source.xosc,run.xosc,run.csv}
         results/disk_sweep/{cp3s{L}|anch{PF}}_trajectories_{subset}{wsuf}_disk.parquet
         results/disk_sweep/{cp3s{L}|anch{PF}}_descriptors_{subset}{wsuf}_disk_g2.parquet
         results/disk_sweep/_anchor_s3/... (special_1786_1797 anchor arms at astep 3)
         results/disk_sweep/_anchor_s5/... (fixed-pair main-subset radius check)
         results/disk_sweep/disk_sweep_status_{subset}.csv, 18_progress.log, manifest.json
Usage: python -B scripts/18_disk_anchor_sweep.py [--subsets a,b] [--arms x,y]
       (driver; spawns `--worker <subset>` subprocesses). Cached on file existence.
Read-only upstream: nothing under exp_cross_coverage / hetero-param / sr-tlkeep-experiment
is written.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ours3_svd5_mpl")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
CC = ROOT / "exp_cross_coverage"
SR = ROOT / "sr-tlkeep-experiment"
CC_RUNS = CC / "esmini_runs"
CC_SCRIPTS = CC / "scripts"
OUT = PROJECT / "results" / "disk_sweep"
RUNS = PROJECT / "runs" / "disk_sweep"
LOG = OUT / "18_progress.log"
PY = "/home/hcis-s19/micromamba/envs/nps/bin/python"
DISK_SRC = PROJECT / "scripts" / "16_disk_window_extract.py"
CV_SRC = ROOT / "exp_coverage_velocity" / "scripts" / "cvrender.py"
STEPS = [10, 5, 3, 2]
COMBOS = [P + F for P in "pmx" for F in "pmx"]
QC_TOL = 1e-3
# (subset, weight suffix, anchor astep of best_anchor_summary_v5.csv, extra asteps)
SUBSETS = [
    ("cutinl", "", 10, (5,)),
    ("keeptl", "", 10, (5,)),
    ("keeptl_sw", "", 10, (5,)),
    ("special_39_180", "", 10, ()),
    ("special_1786_1797", "", 5, (3,)),   # 180 RENDERED = (3, 5); s3 = _bak_anchor_s3
    ("uturn_859_881", "_w8", 5, ()),
]
CONFLICT_W = {"uturn_859_881": 8.0}
N_RENDER_THREADS = 12
N_DESC_PROCS = 8


# ── helpers ──────────────────────────────────────────────────────────────────
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(msg: str):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.environ.get('CC_SUBSET', 'driver')}] {msg}"
    with LOG.open("a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def existing_functions(path: Path, names, namespace: dict):
    """AST-load named top-level functions of `path` (16_disk_window_extract convention)."""
    parsed = ast.parse(path.read_text())
    body = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != set(names):
        raise ValueError(f"Missing requested existing function in {path}: {names}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def literal_constant(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def load_geometry_functions() -> dict:
    ns = {"np": np, "ET": ET, "Path": Path}
    existing_functions(DISK_SRC, ["crossing", "disk_points"], ns)
    ns["KNOTS_7CP"] = literal_constant(CV_SRC, "KNOTS_7CP")
    existing_functions(CV_SRC, ["patch_cps"], ns)
    assert ns["KNOTS_7CP"] == [0, 0, 0, 0, 0.5, 1.0, 1.5, 2, 2, 2, 2], ns["KNOTS_7CP"]
    return ns


def dsweep_base_dir(step: int) -> Path:
    """120_stepm_sweep._xbase_for: step 10 = 80_cp3_method XBASE (_xosc_base_cp3)."""
    return CC_RUNS / ("_xosc_base_cp3" if step == 10 else f"_xosc_base_cp3s{step}")


def anchor_base_dir(P: str, F: str, step: int) -> Path:
    """150_anchor_sweep._xbase_for (+_pfv): _pf2 namespace when pos or frame anchor is p."""
    pfv = "_pf2" if "p" in (P, F) else ""
    return CC_RUNS / f"_xosc_base_anch_{P}{F}_s{step}{pfv}"


def arm_specs(subset: str, wsuf: str, astep: int, extra: tuple):
    """Ordered arm list: {arm, prefix, L, base_dir, out_dir, kind}."""
    specs = []
    for s in STEPS:
        specs.append(dict(arm=f"cp3s{s}", prefix=f"cp3s{s}", L=float(s), kind="d",
                          base_dir=dsweep_base_dir(s), out_dir=OUT))
    for step in (astep, *extra):
        sub = OUT if step == astep else OUT / f"_anchor_s{step}"
        for c in COMBOS:
            arm = f"anch{c}" if step == astep else f"anch{c}_s{step}"
            # The main subsets have no cached L=5 anchor bases. Their L=10
            # bases carry the same anchor event; run_arm replaces all three
            # weighted control points at the requested disk radius.
            base_step = astep if step == 5 and subset in {"cutinl", "keeptl", "keeptl_sw"} else step
            specs.append(dict(arm=arm, prefix=f"anch{c}", L=float(step), kind="anchor",
                              base_dir=anchor_base_dir(c[0], c[1], base_step), out_dir=sub, astep=step))
    return specs


def find_base(base_dir: Path, ego: int, actor: int, mf: int):
    if not base_dir.exists():
        return None
    hits = sorted(base_dir.glob(f"*_{ego}_{actor}_f{mf + 1}.xosc"))
    return hits[0] if len(hits) == 1 else None


def nurbs_cps(xosc: Path):
    """(all CP xy in order, weighted CP xy, weights, n_knots) of the single Nurbs."""
    root = ET.parse(xosc).getroot()
    nurbs = list(root.iter("Nurbs"))
    if len(nurbs) != 1:
        raise RuntimeError(f"{xosc.name}: expected 1 Nurbs, found {len(nurbs)}")
    allcp, wcp, w = [], [], []
    for cp in nurbs[0].iter("ControlPoint"):
        wp = cp.find(".//WorldPosition")
        xy = (float(wp.get("x")), float(wp.get("y"))) if wp is not None else (np.nan, np.nan)
        allcp.append(xy)
        if cp.get("weight") is not None:
            wcp.append(xy)
            w.append(float(cp.get("weight")))
    return np.array(allcp), np.array(wcp), w, len(list(nurbs[0].iter("Knot")))


def set_middle_weight(xosc: Path, w: float):
    """120/150 _patch_conflict_weight on a complete 3-CP set (asserted upstream)."""
    tree = ET.parse(xosc)
    cps = [cp for cp in tree.getroot().iter("ControlPoint") if cp.get("weight") is not None]
    assert len(cps) == 3, len(cps)
    cps[1].set("weight", f"{w:g}")
    tree.write(xosc, encoding="unicode", xml_declaration=True)


# ── worker (one subset; CC_SUBSET / CC_CONFLICT_W already in the environment) ──
G_JOB = {}
_W = {}
ANCH = {"p": "pet", "m": "min_dist", "x": "traj_cross"}   # 150_anchor_sweep.ANCH


def anchor_frame_for(spec: dict, sid: str, ego: int, actor: int, mf: int, xf: int):
    """The frame the base generator clustered the CPs around (memoised per scenario).
    d-sweep (120/80: SampleConfig anchor='pet', no force_anchor): convert2yaml passes the
    labeler's pet_frame1 -> sampling._select_anchor('pet') = pet_frame1 (parampath.pet_anchors
    back-compat accessor). Anchor arms (150: force_anchor=True): generate._anchor_frame_for
    with the pos anchor of the combo (post-2026-08-15 agent-own PET frame for 'p')."""
    PL, PP, GEN = _W["PL"], _W["PP"], _W["GEN"]
    key = ("legacy", sid) if spec["kind"] == "d" else (ANCH[spec["prefix"][4]], sid)
    cache = _W["anchor_cache"]
    if key not in cache:
        if spec["kind"] == "d":
            cache[key] = PP.pet_anchors(PL.DATASET, ego, actor)[0]
        else:
            cache[key] = GEN._anchor_frame_for(PL.DATASET, ego, actor, mf, xf, key[0])
    return key[0], cache[key]


def _traj(g):
    """170_pet_gate_recompute._traj (derived=False: generated-side kinematics from esmini)."""
    Traj = _W["Traj"]
    fps = _W["FPS"]
    t = g.frame.values.astype(float) / fps
    sp = g.speed.values.astype(float)
    return Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
                y=g.y.values.astype(float), heading=g.heading_deg.values.astype(float), speed=sp,
                accel=np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp), fps=fps,
                length=float(g.length.iloc[0]), width=float(g.width.iloc[0]), meta={})


def _desc_gen(sid):
    try:
        a, e = G_JOB[sid]
        d = _W["SWC"].descriptors(_traj(a), _traj(e), estimator="bbox")
        d.update({"scenario_id": sid})
        return d
    except Exception as ex:  # noqa: BLE001
        return {"scenario_id": sid, "error": f"{type(ex).__name__}: {ex}"}


def compute_desc(tr: pd.DataFrame):
    """170.compute_desc (gen mode): teleport screen, trim_lead_still, SWC.descriptors bbox.
    Returns (desc df, teleport sids, error sids)."""
    global G_JOB
    G_JOB = {}
    L = _W["L"]
    teleports = []
    for (sid, role), g in tr.groupby(["scenario_id", "role"]):
        if role != "agent":
            continue
        g = g.sort_values("frame")
        e = tr[(tr.scenario_id == sid) & (tr.role == "ego")].sort_values("frame")
        if not len(e):
            continue
        if L.is_teleport(g):
            teleports.append(sid)
            continue
        G_JOB[sid] = (_W["trim_lead_still"](g), e)
    if not G_JOB:
        return pd.DataFrame(columns=["scenario_id"]), teleports, []
    with Pool(N_DESC_PROCS) as pool:
        out = [r for r in pool.imap_unordered(_desc_gen, list(G_JOB), chunksize=8) if r]
    dd = pd.DataFrame(out)
    errors = []
    if "error" in dd.columns:
        errors = dd[dd.error.notna()].scenario_id.tolist()
        dd = dd[dd.error.isna()]
    return dd, teleports, errors


def run_arm(spec: dict, subset: str, wsuf: str, meta: pd.DataFrame, by_actor: dict, geo: dict,
            cw: float | None, only_arms):
    EX, PL = _W["EX"], _W["PL"]
    arm, prefix, Lr = spec["arm"], spec["prefix"], spec["L"]
    out_dir = spec["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t_file = out_dir / f"{prefix}_trajectories_{subset}{wsuf}_disk.parquet"
    d_file = out_dir / f"{prefix}_descriptors_{subset}{wsuf}_disk_g2.parquet"
    arm_dir = RUNS / subset / arm
    status_file = arm_dir / "status.csv"
    if only_arms and arm not in only_arms:
        if status_file.exists():
            return pd.read_csv(status_file)
        return None
    if status_file.exists() and t_file.exists() and d_file.exists():
        st = pd.read_csv(status_file)
        log(f"{arm}: cached ({int((st.status == 'ok').sum())} ok / {len(st)})")
        return st
    if status_file.exists() and not spec["base_dir"].exists():
        return pd.read_csv(status_file)
    t0 = time.time()
    arm_dir.mkdir(parents=True, exist_ok=True)
    common = dict(subset=subset, arm=arm, prefix=prefix, kind=spec["kind"], L_m=Lr,
                  astep=spec.get("astep", np.nan), base_dir=str(spec["base_dir"]))
    rows = {}
    if not spec["base_dir"].exists():
        for r in meta.itertuples():
            rows[r.scenario_id] = dict(common, scenario=r.scenario_id, status="not_reproducible:base_dir_missing")
        st = pd.DataFrame(rows.values())
        st.to_csv(status_file, index=False)
        log(f"{arm}: NOT REPRODUCIBLE — base dir missing {spec['base_dir']}")
        return st
    jobs = []
    for r in meta.itertuples():
        sid = str(r.scenario_id)
        ego, actor, mf, xf = int(r.ego), int(r.actor), int(r.min_frame), int(r.max_frame)
        row = dict(common, scenario=sid, ego=ego, actor=actor, min_frame=mf, max_frame=xf)
        base = find_base(spec["base_dir"], ego, actor, mf)
        if base is None:
            row["status"] = "base_missing"
            rows[sid] = row
            continue
        row.update(base=str(base), base_sha256=sha256(base))
        try:
            allcp, wcp, w, nk = nurbs_cps(base)
        except Exception as ex:  # noqa: BLE001
            row["status"] = f"base_parse_error:{type(ex).__name__}"
            rows[sid] = row
            continue
        row.update(base_n_cp=len(allcp), base_n_weighted=len(wcp), base_n_knots=nk)
        if len(allcp) - len(wcp) != 4:
            row["status"] = f"base_incompatible:{len(allcp)}cp/{len(wcp)}w"
            rows[sid] = row
            continue
        g = by_actor.get(sid)
        if g is None:
            row["status"] = "no_track"
            rows[sid] = row
            continue
        rule, af = anchor_frame_for(spec, sid, ego, actor, mf, xf)
        row.update(anchor_rule=rule, anchor_frame=af)
        if af is None:
            row["status"] = "unavailable:no_anchor_frame"
            rows[sid] = row
            continue
        hit = np.flatnonzero(g.frame.to_numpy(int) == int(af))
        if len(hit) != 1:
            row["status"] = "unavailable:anchor_outside_window"
            rows[sid] = row
            continue
        k = int(hit[0])
        qc = g[["x", "y"]].to_numpy(float)[k]
        row.update(qc_x=float(qc[0]), qc_y=float(qc[1]))
        if len(wcp):
            dcp = np.hypot(wcp[:, 0] - qc[0], wcp[:, 1] - qc[1])
            row.update(qc_in_base_cps=bool(dcp.min() <= QC_TOL), qc_base_cp_index=int(dcp.argmin()),
                       qc_base_cp_dist_m=float(dcp.min()))
        if len(wcp) == 3:
            row.update(arc_chord1=float(np.hypot(*(wcp[0] - qc))), arc_chord2=float(np.hypot(*(wcp[2] - qc))))
            if np.hypot(*(wcp[1] - qc)) > QC_TOL:
                row["status"] = "qc_mismatch_base_middle_cp"
                rows[sid] = row
                continue
        anchor_frame = int(af)
        res = geo["disk_points"](g, anchor_frame, Lr)
        row.update(disk_status=res["status"],
                   d_window_start_to_qc=res.get("d_window_start_to_qc"),
                   d_window_end_to_qc=res.get("d_window_end_to_qc"),
                   chord1=res.get("chord1"), chord2=res.get("chord2"),
                   t_minus_frame=res.get("t_minus_frame"), t_plus_frame=res.get("t_plus_frame"))
        if len(wcp) == 3 and "qm_x" in res:
            row["arc_vs_disk_qminus_m"] = float(np.hypot(wcp[0, 0] - res["qm_x"], wcp[0, 1] - res["qm_y"]))
        if len(wcp) == 3 and "qp_x" in res:
            row["arc_vs_disk_qplus_m"] = float(np.hypot(wcp[2, 0] - res["qp_x"], wcp[2, 1] - res["qp_y"]))
        if res["status"] != "ok":
            row["status"] = f"unavailable:{res['status']}"
            rows[sid] = row
            continue
        xy = np.array([[res["qm_x"], res["qm_y"]], [res["qc_x"], res["qc_y"]], [res["qp_x"], res["qp_y"]]])
        assert abs(row["chord1"] - Lr) < 1e-6 and abs(row["chord2"] - Lr) < 1e-6, (row["chord1"], row["chord2"])
        run_dir = arm_dir / base.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        patched = run_dir / "patched_source.xosc"
        try:
            geo["patch_cps"](base, xy.ravel(), patched)
            if cw:
                set_middle_weight(patched, cw)
            pall, pw, pwts, pnk = nurbs_cps(patched)
            assert len(pall) == 7 and len(pw) == 3 and pnk == 11, (len(pall), len(pw), pnk)
            assert np.abs(pw - xy).max() < 1e-3, np.abs(pw - xy).max()
            assert pwts == [5.0, cw or 5.0, 5.0], pwts
        except Exception as ex:  # noqa: BLE001
            row["status"] = f"patch_fail:{type(ex).__name__}: {str(ex)[:80]}"
            rows[sid] = row
            continue
        row.update(patched=str(patched), qm_x=xy[0, 0], qm_y=xy[0, 1], qp_x=xy[2, 0], qp_y=xy[2, 1])
        rows[sid] = row
        jobs.append((sid, ego, actor, mf, xf, patched, run_dir, xy))

    def render_one(job):
        sid, ego, actor, mf, xf, patched, run_dir, xy = job
        try:
            xosc, csv = run_dir / "run.xosc", run_dir / "run.csv"
            if not csv.exists() or not xosc.exists():
                EX.to_esmini_replay(patched, PL.DATASET, ego, actor, mf, xf, xosc, None, agent_replay=False)
            _a, pw, _w, _k = nurbs_cps(xosc)
            if len(pw) != 3 or np.abs(pw - xy).max() > 1e-3:
                return sid, None, "replay_xosc_cp_mismatch"
            res = EX.run_and_extract(xosc, PL.DATASET, ego, actor, mf, xf, csv)
            if res is None or res.get("agent") is None:
                return sid, None, "no_agent_traj"
            return sid, res, None
        except Exception as ex:  # noqa: BLE001
            return sid, None, f"{type(ex).__name__}: {str(ex)[:80]}"

    traj_rows = []
    n_ok = 0
    with ThreadPoolExecutor(N_RENDER_THREADS) as tp:
        for sid, res, err in tp.map(render_one, jobs):
            if err:
                rows[sid]["status"] = f"render_fail:{err}"
                continue
            n_ok += 1
            traj_rows.extend(PL.trajs_to_rows(res, sid, prefix, "disk"))
    tr = pd.DataFrame(traj_rows, columns=PL.ROW_COLUMNS)
    tr.to_parquet(t_file)
    dd, teleports, errors = compute_desc(tr)
    dd.to_parquet(d_file)
    for sid in teleports:
        rows[sid]["status"] = "teleport"
    for sid in errors:
        rows[sid]["status"] = "desc_error"
    for sid in dd.scenario_id.tolist():
        rows[sid]["status"] = "ok"
    st = pd.DataFrame(rows.values())
    st.to_csv(status_file, index=False)
    log(f"{arm}: L={Lr:g} candidates={len(meta)} rendered={n_ok} desc_ok={len(dd)} "
        f"teleport={len(teleports)} unavailable={int(st.status.str.startswith('unavailable').sum())} "
        f"other={int((~st.status.isin(['ok', 'teleport']) & ~st.status.str.startswith('unavailable')).sum())} "
        f"({time.time() - t0:.0f}s)")
    return st


def worker(subset: str, only_arms):
    sub = [s for s in SUBSETS if s[0] == subset]
    assert len(sub) == 1, subset
    _name, wsuf, astep, extra = sub[0]
    assert os.environ.get("CC_SUBSET") == subset
    cw = CONFLICT_W.get(subset)
    if cw:
        assert float(os.environ.get("CC_CONFLICT_W", 0)) == cw
    sys.path.insert(0, str(CC_SCRIPTS))
    import lib as L  # noqa: E402  (chdirs to hetero-param)
    from lib import PL  # noqa: E402
    from run_label_lib import trim_lead_still  # noqa: E402
    from hetero_param.sweep import core as SWC  # noqa: E402
    from hetero_param.similarity.core import Traj  # noqa: E402
    from hetero_param.similarity import interaction_sim as IS  # noqa: E402
    from hetero_param import esmini_exec as EX  # noqa: E402
    from hetero_param import parampath as PP  # noqa: E402
    from hetero_param import generate as GEN  # noqa: E402
    assert L.NAME == subset
    gate_default = inspect.signature(IS.pet).parameters["gate"].default
    assert gate_default == "bbox", f"PET gate default is {gate_default!r}, expected v2 'bbox'"
    _W.update(L=L, PL=PL, trim_lead_still=trim_lead_still, SWC=SWC, Traj=Traj, EX=EX, FPS=L.FPS,
              PP=PP, GEN=GEN, anchor_cache={})
    geo = load_geometry_functions()
    meta = pd.read_csv(L.DDIR / "real_meta.csv")
    meta["scenario_id"] = meta.scenario_id.astype(str)
    tracks_path = L.DDIR / "real_tracks.parquet"
    tracks = pd.read_parquet(tracks_path)
    by_actor = {str(sid): g.sort_values("frame").reset_index(drop=True)
                for sid, g in tracks[tracks.role == "actor"].groupby("scenario_id")}
    log(f"start: {len(meta)} scenarios, data_dir={L.DDIR}, conflict_w={cw}, fps={L.FPS}, "
        f"gate={gate_default}, esmini={EX._esmini_bin()}")
    specs = arm_specs(subset, wsuf, astep, extra)
    all_status = []
    for spec in specs:
        try:
            st = run_arm(spec, subset, wsuf, meta, by_actor, geo, cw, only_arms)
        except Exception:  # noqa: BLE001
            log(f"{spec['arm']}: FAILED\n{traceback.format_exc()}")
            continue
        if st is not None:
            all_status.append(st)
    if all_status:
        st = pd.concat(all_status, ignore_index=True)
        status_path = OUT / f"disk_sweep_status_{subset}.csv"
        if only_arms and status_path.exists():
            old = pd.read_csv(status_path)
            st = pd.concat([old[~old.arm.isin(st.arm.unique())], st], ignore_index=True)
        cols = ["subset", "arm", "scenario", "status", "L_m", "chord1", "chord2",
                "arc_vs_disk_qminus_m", "arc_vs_disk_qplus_m", "arc_chord1", "arc_chord2",
                "base_n_weighted", "qc_in_base_cps", "anchor_rule", "anchor_frame"]
        st = st[cols + [c for c in st.columns if c not in cols]]
        st.to_csv(status_path, index=False)
        summ = st.groupby(["arm", "status"]).size().unstack(fill_value=0)
        log("status summary:\n" + summ.to_string())
    meta_info = dict(subset=subset, wsuf=wsuf, astep=astep, extra_asteps=list(extra), conflict_w=cw,
                     fps=L.FPS, pet_gate_default=gate_default, data_dir=str(L.DDIR),
                     real_tracks=str(tracks_path), real_tracks_sha256=sha256(tracks_path),
                     real_meta_sha256=sha256(L.DDIR / "real_meta.csv"),
                     esmini_bin=str(EX._esmini_bin().resolve()), esmini_sha256=sha256(EX._esmini_bin().resolve()),
                     arms=[{"arm": s["arm"], "prefix": s["prefix"], "L": s["L"], "base_dir": str(s["base_dir"]),
                            "base_dir_exists": s["base_dir"].exists(), "out_dir": str(s["out_dir"])} for s in specs])
    (OUT / f"_worker_meta_{subset}.json").write_text(json.dumps(meta_info, indent=1) + "\n")
    log("done")


# ── driver ────────────────────────────────────────────────────────────────────
def driver(subsets, only_arms):
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    log(f"driver start: subsets={subsets}")
    for name, wsuf, astep, extra in SUBSETS:
        if name not in subsets:
            continue
        env = dict(os.environ, CC_SUBSET=name, MPLCONFIGDIR="/tmp/ours3_svd5_mpl",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("CC_CONFLICT_W", None)
        if name in CONFLICT_W:
            env["CC_CONFLICT_W"] = f"{CONFLICT_W[name]:g}"
        cmd = [PY, "-B", str(Path(__file__).resolve()), "--worker", name]
        if only_arms:
            cmd += ["--arms", ",".join(only_arms)]
        t0 = time.time()
        with (OUT / "logs" / f"18_{name}.log").open("a") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(PROJECT)).returncode
        log(f"{name}: worker rc={rc} ({time.time() - t0:.0f}s)")
    write_manifest()


def write_manifest():
    scripts = [Path(__file__).resolve(), DISK_SRC, CV_SRC, CC_SCRIPTS / "lib.py",
               CC_SCRIPTS / "120_stepm_sweep.py", CC_SCRIPTS / "80_cp3_method.py",
               CC_SCRIPTS / "150_anchor_sweep.py", CC_SCRIPTS / "170_pet_gate_recompute.py",
               CC_SCRIPTS / "180_trajdtw_aggregate.py", SR / "scripts" / "run_label_lib.py",
               SR / "scripts" / "pipeline_lib.py", ROOT / "hetero-param/hetero_param/esmini_exec.py",
               ROOT / "hetero-param/hetero_param/sweep/core.py",
               ROOT / "hetero-param/hetero_param/similarity/interaction_sim.py"]
    manifest = dict(
        scope="E10 stage 2: d-sweep + 3x3 anchor sweep re-rendered under the Euclidean-disk window (chords == L)",
        window_rule="q_c = recorded target sample at the arm's anchor frame (d-sweep: legacy labeler pet_frame1 via "
                    "parampath.pet_anchors; anchor arms: generate._anchor_frame_for(pos anchor)); verified == middle "
                    "weighted CP of the base whenever the base kept its 3 CPs; "
                    "q-/q+ = first Euclidean-distance-L crossings walking backward/forward in time inside the "
                    "episode window (16_disk_window_extract.disk_points); unavailable arms are skipped, not clipped",
        patch_rule="cvrender.patch_cps: 3 weighted CPs replaced, 7 CPs, knots [0,0,0,0,0.5,1,1.5,2,2,2,2], "
                    "weight 5.0 (uturn_859_881: middle-CP weight 8)",
        descriptor_rule="170_pet_gate_recompute.compute_desc gen mode: lib.is_teleport screen, trim_lead_still, "
                        "SWC.descriptors(estimator='bbox') with the library default PET gate 'bbox' (v2)",
        scripts=[{"path": str(p), "sha256": sha256(p)} for p in scripts if p.exists()],
        subsets=[], base_files=[])
    seen = set()
    for name, *_ in SUBSETS:
        wm = OUT / f"_worker_meta_{name}.json"
        if wm.exists():
            manifest["subsets"].append(json.loads(wm.read_text()))
        stf = OUT / f"disk_sweep_status_{name}.csv"
        if stf.exists():
            st = pd.read_csv(stf)
            if "base" in st.columns:
                for b, h in st[["base", "base_sha256"]].dropna().drop_duplicates().itertuples(index=False):
                    if b not in seen:
                        seen.add(b)
                        manifest["base_files"].append({"path": b, "sha256": h})
    manifest["n_base_files"] = len(manifest["base_files"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"manifest written: {len(manifest['base_files'])} base files, {len(manifest['subsets'])} subsets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help="internal: run one subset in this process")
    ap.add_argument("--subsets", default=",".join(s[0] for s in SUBSETS))
    ap.add_argument("--arms", default=None, help="restrict to these arm names (comma list)")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a] if args.arms else None
    if args.worker:
        worker(args.worker, arms)
    else:
        driver([s for s in args.subsets.split(",") if s], arms)
