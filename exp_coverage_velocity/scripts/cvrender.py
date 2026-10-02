"""Executed arms for exp_coverage_velocity: OURS (NURBS control points) and
SAKURA-route (road-following when a route exists), each as a parameter table
+ de Gelder KDE, rendered headless in esmini, vectorised immediately, files
deleted.

OURS parameter vector (9-D, one row per real scenario):
    cp1x cp1y cp2x cp2y cp3x cp3y   3 weighted NURBS shape CPs in arc order
                                     (minPET critical point +-10 m arc length,
                                      flanks clipped to the path ends; from
                                      exp_cp3_points cp3pm10_points_keeplt.csv)
    Agent1_Speed Agent1_1_SA_EndSpeed Agent1_1_SA_DynamicDuration
                                     the pipeline speed event of the petq3 base
                                     (= "minPET frame" speed event)
  Base xosc = sr-tlkeep "ours" (srexp-petq3) with its weighted CPs replaced by
  the 3 CPs (weight 5, order-4 knots for 7 CPs) -- the exp_cp3d10 surgery.

SAKURA-route parameter vector (2-D):
    Agent1_Offset                    lateral lane offset applied to all waypoints
    Speed_avg [km/h]                 planned-route length / recorded duration
                                     (constant: Agent1_Speed = SA_EndSpeed)
  Base xosc = sr-tlkeep "sakura" (srexp-none); FollowTrajectory replaced by an
  AssignRouteAction over the GlobalRoutePlanner waypoints + stop-at-goal
  (exp_cross_coverage/scripts/sakura_route.py).  48/50 keeptl routed; the 2
  unroutable (230_179, 1733_1708) keep the plain sakura chord.

Render worker: patch -> to_esmini_replay(overrides) -> [apply_route] ->
esmini headless -> extract agent+ego -> 101-D vector, arc path, kinematics,
teleport flag, interaction descriptors -> unlink xosc+csv.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402

sys.path.insert(0, str(L.SR / "scripts"))
import pipeline_lib as PL  # noqa: E402
from hetero_param.similarity import core as SIMC  # noqa: E402
from hetero_param.sweep import core as SWC  # noqa: E402

EX = PL.EX
XC = L.BASE / "exp_cross_coverage"
sys.path.insert(0, str(XC / "scripts"))
import sakura_route as SKR  # noqa: E402

CP3_CSV = L.BASE / "exp_cp3_points" / "results" / f"cp3pm10_points_{L.CP3_LABEL[L.NAME]}.csv"
ROUTE_JSON = XC / "results" / f"route_plan_{L.NAME}.json"
RUNS = L.EXP / "esmini_runs" / L.NAME
RUNS.mkdir(parents=True, exist_ok=True)
XBASE = L.HP / "results" / "esmini" / "xosc_base"

# OURS = the user's canonical theta parameterization (memory theta-parameterization):
#   p = (Agent1_Offset, SA_EndSpeed, SA_DynamicDuration, theta1, theta2)
# The three CPs are a CONSTRUCTION: P- (crit-10 m) frozen, chord lengths L1/L2
# frozen, Pc = P- + L1 R(th1) u1, P+ = Pc + L2 R(th1+th2) u1, u1 = incidence
# direction (first GT sample -> P-, or the start heading when P- is the start).
# Angle jitter rotates the downstream chain rigidly, so control points never
# wander independently and a flank clipped to the start stays put.
# The earlier absolute-coordinate 9-D variant (OURS_CPABS_COLS) produced the
# "outward swing after departure" artifact: an independent ~3 m jitter on a
# weight-5 CP that coincides with the start point.
OURS_COLS = ["Agent1_Offset", "Agent1_1_SA_EndSpeed", "Agent1_1_SA_DynamicDuration",
             "theta1", "theta2"]
OURS_CPABS_COLS = ["cp1x", "cp1y", "cp2x", "cp2y", "cp3x", "cp3y",
                   "Agent1_Speed", "Agent1_1_SA_EndSpeed", "Agent1_1_SA_DynamicDuration"]
GEOM_COLS = ["pm_x", "pm_y", "u1x", "u1y", "L1", "L2", "theta1", "theta2", "u1_source"]
SAK_COLS = ["Agent1_Offset", "Speed_avg"]
KNOTS_7CP = [0, 0, 0, 0, 0.5, 1.0, 1.5, 2, 2, 2, 2]
DESC_KEYS = ["pet", "pet_abs", "min_dist", "conflict_angle", "agent_arr_speed",
             "closing_speed", "drac", "min_ttc", "conflict_x", "conflict_y"]


# ─── scenario bookkeeping ───────────────────────────────────────────────────
def scen_rows(T):
    return list(T.meta[["scenario_id", "ego", "actor", "min_frame", "max_frame"]]
                .itertuples(index=False, name=None))


def base_xosc(method, ego, actor, mf, xf=None):
    strat = "srexp-petq3" if method == "ours" else "srexp-none"
    hits = list((XBASE / strat).glob(f"*_{ego}_{actor}_f{mf + 1}.xosc"))
    if not hits:
        raise FileNotFoundError(f"{method} base for {ego}_{actor} f{mf + 1}")
    return hits[0]


def ensure_bases(T):
    """Generate any missing pipeline base xosc (sr-tlkeep gen_base; run
    SEQUENTIALLY -- generation is not thread-safe).  Returns missing list."""
    missing = []
    for sid, ego, actor, mf, xf in scen_rows(T):
        for m in ("ours", "sakura"):
            try:
                base_xosc(m, ego, actor, mf)
            except FileNotFoundError:
                try:
                    out = PL.gen_base(m, int(ego), int(actor), int(mf), int(xf), label=L.LABEL_OF[L.NAME])
                except Exception as e:  # noqa: BLE001  (Stage A demotes the agent etc.)
                    out = None
                    print(f"[bases] {sid} {m}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                if out is None:
                    missing.append((sid, m))
    return missing


def has_base(method, ego, actor, mf):
    try:
        base_xosc(method, ego, actor, mf)
        return True
    except FileNotFoundError:
        return False


# ─── parameter tables ───────────────────────────────────────────────────────
def load_cp3():
    df = pd.read_csv(CP3_CSV)
    out = {}
    for sid, g in df.groupby("scenario_id"):
        if sid not in L.load().idx:
            continue
        g = g.sort_values("s_from_crit")
        assert list(g.kind) == ["m10", "crit", "p10"], (sid, list(g.kind))
        out[sid] = g[["x", "y"]].values.reshape(-1)      # arc order
    return out


def ours_cpabs_table(T):
    """The retired absolute-coordinate 9-D table (kept for the record)."""
    cp3 = load_cp3()
    rows, idx = [], []
    for sid, ego, actor, mf, xf in scen_rows(T):
        if not has_base("ours", ego, actor, mf) or sid not in cp3:
            continue
        bp = EX._base_params(base_xosc("ours", ego, actor, mf))
        rows.append([*cp3[sid], bp["Agent1_Speed"], bp["Agent1_1_SA_EndSpeed"],
                     bp["Agent1_1_SA_DynamicDuration"]])
        idx.append(sid)
    return pd.DataFrame(rows, columns=OURS_CPABS_COLS, index=idx)


def _signed_angle(u, v):
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)))


def _rot(u, deg):
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]])


def theta_geometry(T, sid, cps6):
    """Frozen construction for one scenario from its clipped cp3 triple.
    Returns dict(pm, u1, L1, L2, theta1, theta2, u1_source) or None."""
    xy = np.asarray(cps6, float).reshape(3, 2)
    g = L.real_actor_track(T, sid)
    init = np.array([g.x.values[0], g.y.values[0]])
    v1 = xy[0] - init
    L0 = float(np.hypot(*v1))
    if L0 >= 1.0:
        u1, src = v1 / L0, "init_pt"
    else:                                   # P- is the start: use the start heading
        arc = L.arc_resample(g.x.values, g.y.values, 200)
        s_ = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(arc, axis=0).T))])
        k = int(np.searchsorted(s_, min(2.0, s_[-1] * 0.5)))
        d = arc[max(k, 1)] - arc[0]
        u1, src = d / (np.hypot(*d) + 1e-9), "start_heading"
    v2, v3 = xy[1] - xy[0], xy[2] - xy[1]
    L1, L2 = float(np.hypot(*v2)), float(np.hypot(*v3))
    if min(L1, L2) < 1.0:
        return None
    u2, u3 = v2 / L1, v3 / L2
    return dict(pm=xy[0], u1=u1, L1=L1, L2=L2, theta1=_signed_angle(u1, u2),
                theta2=_signed_angle(u2, u3), u1_source=src)


def theta_to_cps(geom, th1, th2):
    pm, u1 = np.asarray(geom["pm"], float), np.asarray(geom["u1"], float)
    pc = pm + geom["L1"] * _rot(u1, th1)
    pp = pc + geom["L2"] * _rot(u1, th1 + th2)
    return np.concatenate([pm, pc, pp])


def ours_table(T, write_geom=True):
    """5-D theta table + frozen geometry (results/theta_geom.csv)."""
    cp3 = load_cp3()
    rows, idx, geoms = [], [], []
    for sid, ego, actor, mf, xf in scen_rows(T):
        if not has_base("ours", ego, actor, mf) or sid not in cp3:
            print(f"[ours_table] skip {sid} (no base / no cp3)", flush=True)
            continue
        geo = theta_geometry(T, sid, cp3[sid])
        if geo is None:
            print(f"[ours_table] skip {sid} (degenerate chord < 1 m)", flush=True)
            continue
        bp = EX._base_params(base_xosc("ours", ego, actor, mf))
        rows.append([bp["Agent1_Offset"], bp["Agent1_1_SA_EndSpeed"],
                     bp["Agent1_1_SA_DynamicDuration"], geo["theta1"], geo["theta2"]])
        idx.append(sid)
        geoms.append([geo["pm"][0], geo["pm"][1], geo["u1"][0], geo["u1"][1], geo["L1"], geo["L2"],
                      geo["theta1"], geo["theta2"], geo["u1_source"]])
    if write_geom:
        pd.DataFrame(geoms, columns=GEOM_COLS, index=idx).to_csv(L.RESULTS / "theta_geom.csv")
    return pd.DataFrame(rows, columns=OURS_COLS, index=idx)


_GEOM = None


def _geom(sid):
    global _GEOM
    if _GEOM is None:
        df = pd.read_csv(L.RESULTS / "theta_geom.csv", index_col=0)
        _GEOM = {k: dict(pm=np.array([r.pm_x, r.pm_y]), u1=np.array([r.u1x, r.u1y]), L1=r.L1, L2=r.L2)
                 for k, r in df.iterrows()}
    return _GEOM[sid]


def load_routes():
    r = json.load(open(ROUTE_JSON))
    return {k: (v if v else None) for k, v in r.items()}


def sakura_table(T):
    routes = load_routes()
    rows, idx = [], []
    for sid, ego, actor, mf, xf in scen_rows(T):
        if not has_base("sakura", ego, actor, mf):
            print(f"[sakura_table] skip {sid} (no base)", flush=True)
            continue
        idx.append(sid)
        bp = EX._base_params(base_xosc("sakura", ego, actor, mf))
        g = L.real_actor_track(T, sid)
        dur = (g.frame.values[-1] - g.frame.values[0]) / L.FPS
        wp = routes.get(sid)
        if wp:
            length = SKR.route_length(wp)
        else:                                   # chord fallback: real path length
            length = float(np.hypot(np.diff(g.x.values), np.diff(g.y.values)).sum())
        rows.append([bp["Agent1_Offset"], 3.6 * length / dur])
    return pd.DataFrame(rows, columns=SAK_COLS, index=idx)


# ─── xosc surgery ───────────────────────────────────────────────────────────
def patch_cps(base: Path, cps6, out: Path):
    """Replace the weighted shape CPs of the Agent1 Nurbs by 3 fixed
    WorldPositions (weight 5) with order-4 knots for 7 CPs (exp_cp3d10)."""
    tree = ET.parse(base)
    root = tree.getroot()
    nurbs = [n for n in root.iter("Nurbs")]
    if len(nurbs) != 1:
        raise RuntimeError(f"expected 1 Nurbs, found {len(nurbs)}")
    nb = nurbs[0]
    children = list(nb)
    cp_elems = [c for c in children if c.tag == "ControlPoint"]
    weighted = [c for c in cp_elems if c.get("weight") is not None]
    if weighted:
        insert_at = children.index(weighted[0])
        for c in weighted:
            nb.remove(c)
    else:
        insert_at = children.index(cp_elems[-1])
    for c in [c for c in list(nb) if c.tag == "Knot"]:
        nb.remove(c)
    xy = np.asarray(cps6, float).reshape(3, 2)
    for i in range(3):
        el = ET.Element("ControlPoint", {"weight": "5.0"})
        pos = ET.SubElement(el, "Position")
        ET.SubElement(pos, "WorldPosition", {"x": f"{xy[i, 0]:.4f}", "y": f"{xy[i, 1]:.4f}"})
        nb.insert(insert_at + i, el)
    n_cp = len([c for c in list(nb) if c.tag == "ControlPoint"])
    if len(KNOTS_7CP) != n_cp + int(nb.get("order", "4")):
        raise RuntimeError(f"knot mismatch: {n_cp} CPs")
    for k in KNOTS_7CP:
        ET.SubElement(nb, "Knot", {"value": str(k)})
    tree.write(out, encoding="unicode", xml_declaration=True)


OFFSET_HALF_RANGE = 0.5      # the pipeline's own DistributionRange for Agent1_Offset
CLIP_OFFSET = os.environ.get("CV_CLIP_OFFSET", "1") != "0"
_BASE_OFF = {}


def clip_offset(base: Path, o: float) -> float:
    """Clip a sampled lane offset to base +- 0.5 m (retrieval-scenarios
    convert2yaml.build_param_xosc range).  Larger offsets make esmini snap
    the LanePosition to the neighbouring lane -- a teleport, i.e. an execution
    artifact, not a scenario.  Disable with CV_CLIP_OFFSET=0."""
    if not CLIP_OFFSET:
        return float(o)
    if base not in _BASE_OFF:
        _BASE_OFF[base] = float(EX._base_params(base)["Agent1_Offset"])
    b = _BASE_OFF[base]
    return float(min(max(o, b - OFFSET_HALF_RANGE), b + OFFSET_HALF_RANGE))


_REAL_END = {}


def real_endpoint(sid):
    """Last GT sample of the real agent (x, y) -- the honest stop target: the
    NURBS goal LanePosition sits 2-3 m short of it and a 3 m tolerance added
    another 3 m, leaving every executed path ~6 m short (arc floor 1.1 eps)."""
    if sid not in _REAL_END:
        T = L.load()
        g = L.real_actor_track(T, sid)
        _REAL_END[sid] = (float(g.x.values[-1]), float(g.y.values[-1]))
    return _REAL_END[sid]


def _stop_event_world(actor, x, y, tolerance):
    return (
        f'<Event name="{actor}_StopAtEndEvent" priority="parallel" maximumExecutionCount="1">'
        f'<Action name="{actor}_StopAtEndAction"><PrivateAction><LongitudinalAction><SpeedAction>'
        f'<SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time"/>'
        f'<SpeedActionTarget><AbsoluteTargetSpeed value="0"/></SpeedActionTarget>'
        f'</SpeedAction></LongitudinalAction></PrivateAction></Action>'
        f'<StartTrigger><ConditionGroup><Condition name="end_reached" delay="0" conditionEdge="none">'
        f'<ByEntityCondition><TriggeringEntities triggeringEntitiesRule="any">'
        f'<EntityRef entityRef="{actor}"/></TriggeringEntities>'
        f'<EntityCondition><ReachPositionCondition tolerance="{tolerance}"><Position>'
        f'<WorldPosition x="{x:.3f}" y="{y:.3f}"/>'
        f'</Position></ReachPositionCondition></EntityCondition></ByEntityCondition>'
        f'</Condition></ConditionGroup></StartTrigger></Event>'
    )


STOP_TOL = 1.5


def add_stop_at_end(xosc: Path, sid: str, tolerance: float = STOP_TOL):
    """Inject a stop event at the REAL agent's last GT position (both executed
    arms).  esmini otherwise keeps driving after the trajectory ends."""
    txt = xosc.read_text()
    i = txt.find('<Maneuver name="Agent1_Maneuver">')
    j = txt.find("</Maneuver>", i)
    if i < 0 or j < 0:
        return False
    x, y = real_endpoint(sid)
    txt = txt[:j] + _stop_event_world("Agent1", x, y, tolerance) + txt[j:]
    xosc.write_text(txt)
    return True


# ─── one render ─────────────────────────────────────────────────────────────
_ROUTES = None
_REAL_LW = {}


def _routes():
    global _ROUTES
    if _ROUTES is None:
        _ROUTES = load_routes()
    return _ROUTES


def _traj(frame, x, y, heading_deg, speed, length, width):
    t = np.asarray(frame, float) / L.FPS
    sp = np.asarray(speed, float)
    acc = np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp)
    return SIMC.Traj(frame=np.asarray(frame, float), x=np.asarray(x, float),
                     y=np.asarray(y, float), heading=np.asarray(heading_deg, float),
                     speed=sp, accel=acc, fps=L.FPS, length=float(length),
                     width=float(width), meta={})


def teleport(a):
    d = np.hypot(np.diff(a.x), np.diff(a.y)) * L.FPS
    return bool(len(d) and d.max() > 30.0 and d.max() > np.nanmax(a.speed) + 10.0)


def describe(agent, ego):
    try:
        d = SWC.descriptors(agent, ego, estimator="bbox")
        return {k: float(d.get(k, np.nan)) for k in DESC_KEYS} | {"pet_type": str(d.get("pet_type", ""))}
    except Exception as e:  # noqa: BLE001
        return {k: np.nan for k in DESC_KEYS} | {"pet_type": f"error:{type(e).__name__}"}


def render_one(job):
    """job = dict(method, sid, ego, actor, mf, xf, params(list), uid, keep=False)
    -> dict row (vector as list, arc as list, descriptors, flags)."""
    method, sid = job["method"], job["sid"]
    ego, actor, mf, xf = job["ego"], job["actor"], job["mf"], job["xf"]
    p = np.asarray(job["params"], float)
    wd = RUNS / f"{method}_{sid}_{job['uid']}"
    wd.mkdir(parents=True, exist_ok=True)
    row = dict(method=method, sid=sid, uid=job["uid"], ok=False, err="", **job.get("_tags", {}))
    try:
        base = base_xosc(method, ego, actor, mf)
        xosc, csv = wd / "r.xosc", wd / "r.csv"
        if method == "ours":
            tmp = wd / "patched.xosc"
            if len(p) == 9:                          # retired absolute-CP variant
                patch_cps(base, p[:6], tmp)
                ov = {"Agent1_Speed": max(0.0, p[6]), "Agent1_1_SA_EndSpeed": max(0.0, p[7]),
                      "Agent1_1_SA_DynamicDuration": max(0.1, p[8])}
            else:                                    # theta parameterization (5-D)
                patch_cps(base, theta_to_cps(_geom(sid), p[3], p[4]), tmp)
                ov = {"Agent1_Offset": clip_offset(base, p[0]), "Agent1_1_SA_EndSpeed": max(0.0, p[1]),
                      "Agent1_1_SA_DynamicDuration": max(0.1, p[2])}
            EX.to_esmini_replay(tmp, PL.DATASET, ego, actor, mf, xf, xosc, ov, agent_replay=False)
            add_stop_at_end(xosc, sid)           # stop at the real endpoint (both executed arms)
        else:
            v = max(0.0, p[1])
            ov = {"Agent1_Offset": clip_offset(base, p[0]), "Agent1_Speed": v, "Agent1_1_SA_EndSpeed": v}
            EX.to_esmini_replay(base, PL.DATASET, ego, actor, mf, xf, xosc, ov, agent_replay=False)
            wp = _routes().get(sid)
            if wp:
                SKR.apply_route(xosc, [tuple(w)[:3] for w in wp], stop_at_goal=False)
            add_stop_at_end(xosc, sid)           # stop at the real endpoint (both executed arms)
        res = EX.run_and_extract(xosc, PL.DATASET, ego, actor, mf, xf, csv, reuse=False)
        if res is None or res.get("agent") is None or res.get("ego") is None:
            row["err"] = "no traj"
            return row
        a, e = res["agent"], res["ego"]
        vp = L.vector_from_track(a.frame, a.x, a.y, a.speed)
        if vp is None:
            row["err"] = "short"
            return row
        vec, arc = vp
        row.update(vector=vec.tolist(), arc=arc.reshape(-1).tolist(),
                   teleport=teleport(a), n_frames=int(len(a.frame)),
                   v_max=float(np.nanmax(a.speed)), path_len=float(np.hypot(np.diff(a.x), np.diff(a.y)).sum()))
        row.update(describe(a, e))
        row["ok"] = True
    except Exception as ex:  # noqa: BLE001
        row["err"] = f"{type(ex).__name__}: {ex}"[:200]
    finally:
        if not job.get("keep"):
            shutil.rmtree(wd, ignore_errors=True)
    return row


def _mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return float("inf")


def _parts_dir(out: Path):
    d = out.with_name(out.stem + "_parts")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_renders(out: Path):
    """The final parquet if present, else the concatenation of the part files."""
    parts = sorted(_parts_dir(out).glob("part-*.parquet"))
    frames = [pd.read_parquet(f) for f in parts]
    if out.exists():
        frames.insert(0, pd.read_parquet(out))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def finalize_renders(out: Path):
    """Concatenate part files into the single parquet (once, at the end) and
    remove the parts.  Atomic via temp + os.replace."""
    df = load_renders(out)
    if df.empty:
        return df
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)
    for f in _parts_dir(out).glob("part-*.parquet"):
        f.unlink()
    return df


def run_plan(plan: pd.DataFrame, out: Path, workers=12, chunksize=4, log_every=500,
             mem_floor_gb=8.0):
    """Render a plan with a process pool.  The pool is forked BEFORE the job
    tables are built so each worker stays ~300 MB (imports only) instead of
    privatising a ~1 GB copy of the plan; rows are appended as part files,
    submission pauses while MemAvailable < mem_floor_gb, parts are concatenated
    once at the end.  Resumable by uid."""
    from multiprocessing import Pool
    import time
    with Pool(workers, maxtasksperchild=2000) as pool:      # fork while the parent is small
        done = set()
        existing = load_renders(out)
        if len(existing):
            done = set(existing.uid)
        del existing
        todo = plan[~plan.uid.isin(done)]
        print(f"[render] {len(plan)} planned, {len(done)} done, {len(todo)} to run", flush=True)
        tags = [c for c in plan.columns if c not in ("method", "sid", "ego", "actor", "mf", "xf", "uid", "params")]

        def jobs_iter():
            for n, r in enumerate(todo.itertuples()):
                if n % 50 == 0:
                    while _mem_available_gb() < mem_floor_gb:
                        print(f"[render] paused: MemAvailable {_mem_available_gb():.1f} GB < {mem_floor_gb} GB", flush=True)
                        time.sleep(20)
                yield dict(method=r.method, sid=r.sid, ego=int(r.ego), actor=int(r.actor),
                           mf=int(r.mf), xf=int(r.xf), params=list(r.params), uid=r.uid,
                           _tags={t: getattr(r, t) for t in tags})

        part_idx = len(list(_parts_dir(out).glob("part-*.parquet")))
        rows, t0, nok, n_jobs = [], time.time(), 0, len(todo)
        if n_jobs:
            for k, r in enumerate(pool.imap_unordered(render_one, jobs_iter(), chunksize=chunksize), 1):
                rows.append(r)
                nok += int(r["ok"])
                if k % log_every == 0 or k == n_jobs:
                    el = time.time() - t0
                    print(f"[render] {k}/{n_jobs}  {k / el * 60:.0f}/min  ok={nok}/{k}  mem_avail={_mem_available_gb():.1f}GB", flush=True)
                    _flush(rows, out, part_idx)
                    part_idx += 1
                    rows = []
        if rows:
            _flush(rows, out, part_idx)
    finalize_renders(out)


def _flush(rows, out, part_idx):
    """Write one part file atomically (temp + os.replace)."""
    f = _parts_dir(out) / f"part-{part_idx:05d}.parquet"
    tmp = f.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, f)
