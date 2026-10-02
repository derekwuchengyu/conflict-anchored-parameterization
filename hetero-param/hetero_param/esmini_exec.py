"""
esmini_exec — run the GENERATED .xosc through esmini (headless) and feed the executed
trajectories into the similarity / sweep suites, instead of the analytic NURBS.

Why surgery is needed: the pipeline's .xosc is CARLA/PISA-oriented — the ego hangs on
an ExternalControl AV planner and Agent1's whole event chain waits on the
FLAG-AV_CONNECTED parameter, so standalone esmini never terminates. This module
rewrites a generated .xosc into a self-contained, terminating scenario:

  ego    → REPLAY: FollowTrajectoryAction over the real recorded ego trajectory
           (polyline, absolute Timing — true dataset replay)
  Agent1 → UNCHANGED NURBS FollowTrajectory + speed profile from the generated
           artifact; only its start trigger is rewired from FLAG-AV_CONNECTED to
           SimulationTime > 0 (the CARLA semantics of "start when the AV is live")
  infra  → local .xodr path, inline Vehicle/Pedestrian entities with the DATASET
           bbox dims (no /opt/Catalogs), PISA outcome-detection story removed,
           StopTrigger = SimulationTime > window length + margin

then runs  esmini --headless --fixed_timestep 1/fps --csv_logger  and parses the log
into similarity.core.Traj objects (frame = min_frame + t·fps; kinematics by finite
differences — the same estimator the sweep variants use).

MAIN ENTRY POINTS
  esmini_pair(dataset, ego, actor, min_frame, max_frame, ...) →
      {"agent": Traj, "ego": Traj, "xosc": Path, "edf": DataFrame}
  esmini_swept_set(dataset, ego, actor, min_frame, max_frame, axes, ...) → SweptSet
      one esmini run per grid member (Agent1_Offset / Agent1_1_TA_Offset shifted by
      the lateral deltas; Agent1_Speed & Agent1_1_SA_EndSpeed scaled by the speed
      factor) — consumable directly by sweep.coverage / variance / redundancy.
  Known startup artifact: Agent1's event chain needs ~3 sim steps before the
  trajectory starts (inherited from the artifact's trigger chain), so the agent
  stands ~0.1 s at spawn; this is part of the executed artifact and is measured.
"""
from __future__ import annotations
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import paths, config as C, generate as GEN
from .similarity import core as SIMC
from .sweep.core import SweptSet, Variant, descriptors

ESMINI_DIR = paths.RESULTS / "esmini"

# The user's esmini install ships a config.yml with `pause: True` (interactive use)
# that CANNOT be overridden from the CLI and freezes headless runs at t=0. esmini
# resolves its default config relative to the BINARY location, so we run through a
# symlink in our own bin/ (no config.yml next to it).
_ESMINI_BIN = ESMINI_DIR / "bin" / "esmini"

# esmini runs in a sweep execute concurrently (each is a fast single-threaded
# subprocess). None = auto (cpu_count − 2); set to 1 to force serial.
ESMINI_WORKERS: int | None = None


def _esmini_bin() -> Path:
    if not _ESMINI_BIN.exists():
        _ESMINI_BIN.parent.mkdir(parents=True, exist_ok=True)
        try:
            _ESMINI_BIN.symlink_to(paths.ESMINI_BIN)
        except FileExistsError:
            pass   # another thread/process won the first-run race — same target
    return _ESMINI_BIN


def _run_esmini(xosc: Path, csv_out: Path, fps: float, timeout: int = 120) -> bool:
    import subprocess
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(_esmini_bin()), "--osc", str(xosc), "--headless",
           "--csv_logger", str(csv_out), "--fixed_timestep", str(1.0 / fps),
           "--disable_stdout"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


# ── base .xosc (the pipeline artifact) ───────────────────────────────────────

def base_xosc(dataset: str, ego: int, actor: int, min_frame: int, max_frame: int,
              label: int | None = None, sample_cfg: C.SampleConfig | None = None,
              strategy_name: str = "5pt-default", refresh: bool = False,
              force_anchor: bool = False) -> Path | None:
    """Generate (or reuse) the pipeline .xosc for one labeled scenario.
    force_anchor=True makes the cfg's anchor (pet / min_dist / traj_cross) take
    effect exactly in Stage A — required for non-pet anchors, which the pipeline
    otherwise zeroes/ignores."""
    out_dir = ESMINI_DIR / "xosc_base" / strategy_name
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = list(out_dir.glob(f"*_{ego}_{actor}_f{min_frame + 1}.xosc"))
    if hits and not refresh:
        return hits[0]
    if label is None:
        label = _label_of(dataset, ego, actor, min_frame)
        if label is None:
            return None
    cfg = sample_cfg or C.SAMPLE_STRATEGIES.get(strategy_name)
    return GEN.generate(dataset, ego, [actor], [], min_frame, max_frame, label,
                        out_dir, sample_cfg=cfg, force_anchor=force_anchor)


def _label_of(dataset, ego, actor, min_frame):
    for s in SIMC.list_scenarios(dataset):
        if s["ego"] == ego and s["actor"] == actor and s["min_frame"] == min_frame:
            return s["label"]
    return None


# ── XML surgery: CARLA package → self-contained esmini replay scenario ───────

# 2026-08-19(user 規範):所有 ours 相關專案的 xosc 必須保留 retrieval-scenarios
# 原本的 CatalogLocations(/opt/Catalogs/*)與 CatalogReference 參數化 entity
# ($Ego_Vehicle / $AgentN_Type)。CATALOG_ENTITIES=True 時 surgery 不內聯
# entity、且把 CatalogLocations 標準化為下列 block(/opt/Catalogs 已存在,
# esmini 可解析;kinematics 不受 bbox 來源影響 — 軌跡驅動)。
CATALOG_ENTITIES = True
_CATALOG_DIRS = [("VehicleCatalog", "/opt/Catalogs/Vehicles"),
                 ("PedestrianCatalog", "/opt/Catalogs/Pedestrians"),
                 ("ControllerCatalog", "/opt/Catalogs/Controllers"),
                 ("EnvironmentCatalog", "/opt/Catalogs/Environments")]


def _standard_catalogs(root: ET.Element) -> None:
    """把 CatalogLocations 換成標準 /opt/Catalogs block(缺則插在 FileHeader 後)。"""
    for cl in root.findall("CatalogLocations"):
        root.remove(cl)
    cl = ET.Element("CatalogLocations")
    for tag, path in _CATALOG_DIRS:
        c = ET.SubElement(cl, tag)
        ET.SubElement(c, "Directory", path=path)
    idx = 0
    for i, child in enumerate(list(root)):
        if child.tag in ("FileHeader", "ParameterDeclarations"):
            idx = i + 1
    root.insert(idx, cl)


def _inline_entity(obj: ET.Element, name: str, length: float, width: float) -> None:
    """Replace a CatalogReference ScenarioObject body with an inline Vehicle
    (dataset bbox dims; generic performance/axles — the entity is trajectory-driven)."""
    ref = obj.find("CatalogReference")
    entry = ref.get("entryName", "") if ref is not None else ""
    for child in list(obj):
        obj.remove(child)
    L = length if np.isfinite(length) and length > 0.1 else 4.5
    W = width if np.isfinite(width) and width > 0.1 else 1.8
    if entry.startswith("$"):
        entry = ""                      # parameterized name — decide by dims below
    is_ped = "walker" in entry or "pedestrian" in entry or (L < 1.2 and W < 1.2)
    if is_ped:
        ped = ET.SubElement(obj, "Pedestrian", name=f"{name}_ped", mass="80.0",
                            model="ped", pedestrianCategory="pedestrian")
        bb = ET.SubElement(ped, "BoundingBox")
        ET.SubElement(bb, "Center", x="0", y="0", z="0.9")
        ET.SubElement(bb, "Dimensions", width=f"{W}", length=f"{L}", height="1.8")
        ET.SubElement(ped, "Properties")
        return
    veh = ET.SubElement(obj, "Vehicle", name=f"{name}_vehicle", vehicleCategory="car")
    ET.SubElement(veh, "ParameterDeclarations")
    ET.SubElement(veh, "Performance", maxSpeed="70", maxAcceleration="15",
                  maxDeceleration="15")
    bb = ET.SubElement(veh, "BoundingBox")
    ET.SubElement(bb, "Center", x="0", y="0", z="0.75")
    ET.SubElement(bb, "Dimensions", width=f"{W}", length=f"{L}", height="1.5")
    ax = ET.SubElement(veh, "Axles")
    wb = max(L * 0.6, 0.8)
    ET.SubElement(ax, "FrontAxle", maxSteering="0.5", wheelDiameter="0.6",
                  trackWidth=f"{W}", positionX=f"{wb}", positionZ="0.3")
    ET.SubElement(ax, "RearAxle", maxSteering="0.0", wheelDiameter="0.6",
                  trackWidth=f"{W}", positionX="0", positionZ="0.3")
    ET.SubElement(veh, "Properties")


def _ego_replay_story(ego_traj: SIMC.Traj, min_frame: int) -> ET.Element:
    """Story: ego follows the real recorded trajectory with ABSOLUTE timing —
    vertex k at t = (frame_k − min_frame)/fps. True replay, terminates with the data."""
    story = ET.Element("Story", name="story_EgoReplay")
    act = ET.SubElement(story, "Act", name="act_EgoReplay")
    mg = ET.SubElement(act, "ManeuverGroup", name="mg_EgoReplay",
                       maximumExecutionCount="1")
    actors = ET.SubElement(mg, "Actors", selectTriggeringEntities="false")
    ET.SubElement(actors, "EntityRef", entityRef="Ego")
    man = ET.SubElement(mg, "Maneuver", name="EgoReplay_Maneuver")
    ev = ET.SubElement(man, "Event", name="EgoReplay_Event", priority="overwrite",
                       maximumExecutionCount="1")
    action = ET.SubElement(ev, "Action", name="EgoReplay_Follow")
    pa = ET.SubElement(action, "PrivateAction")
    ra = ET.SubElement(pa, "RoutingAction")
    fta = ET.SubElement(ra, "FollowTrajectoryAction")
    traj = ET.SubElement(fta, "Trajectory", name="EgoReplayTrajectory", closed="false")
    shape = ET.SubElement(traj, "Shape")
    poly = ET.SubElement(shape, "Polyline")
    t0 = ego_traj.frame[0]
    hdg = np.radians(ego_traj.heading)
    for f, x, y, h in zip(ego_traj.frame, ego_traj.x, ego_traj.y, hdg):
        v = ET.SubElement(poly, "Vertex", time=f"{(f - t0) / ego_traj.fps:.6f}")
        pos = ET.SubElement(v, "Position")
        ET.SubElement(pos, "WorldPosition", x=f"{x:.4f}", y=f"{y:.4f}", h=f"{h:.5f}")
    tref = ET.SubElement(fta, "TimeReference")
    ET.SubElement(tref, "Timing", domainAbsoluteRelative="absolute",
                  scale="1", offset="0")
    ET.SubElement(fta, "TrajectoryFollowingMode", followingMode="position")
    st = ET.SubElement(ev, "StartTrigger")
    cg = ET.SubElement(st, "ConditionGroup")
    cond = ET.SubElement(cg, "Condition", name="ego_replay_start", delay="0.0",
                         conditionEdge="none")
    bv = ET.SubElement(cond, "ByValueCondition")
    ET.SubElement(bv, "SimulationTimeCondition", value="0.0", rule="greaterThan")
    ast = ET.SubElement(act, "StartTrigger")
    acg = ET.SubElement(ast, "ConditionGroup")
    ac = ET.SubElement(acg, "Condition", name="act_start", delay="0.0",
                       conditionEdge="none")
    abv = ET.SubElement(ac, "ByValueCondition")
    ET.SubElement(abv, "SimulationTimeCondition", value="0.0", rule="greaterThan")
    ET.SubElement(act, "StopTrigger")
    return story


def _sim_time_stop(t_end: float) -> ET.Element:
    stop = ET.Element("StopTrigger")
    cg = ET.SubElement(stop, "ConditionGroup")
    cond = ET.SubElement(cg, "Condition", name="end_of_window", delay="0.0",
                         conditionEdge="none")
    bv = ET.SubElement(cond, "ByValueCondition")
    ET.SubElement(bv, "SimulationTimeCondition", value=f"{t_end:.3f}",
                  rule="greaterThan")
    return stop


def to_esmini_replay(xosc_in: Path, dataset: str, ego: int, actor: int,
                     min_frame: int, max_frame: int, xosc_out: Path,
                     param_overrides: dict | None = None,
                     margin_s: float = 1.0,
                     shape_weight: float | None = None,
                     agent_replay: bool = False) -> Path:
    """Rewrite a pipeline .xosc into a self-contained esmini scenario:
    replay ego, SimulationTime-triggered Agent1, local paths, inline entities,
    terminating StopTrigger. `param_overrides` sets ParameterDeclaration values
    (used for the esmini-executed sweep variants)."""
    fps = paths.dataset_of(dataset)["fps"]
    ego_traj = SIMC.real_traj(dataset, ego, min_frame, max_frame)
    agent_traj = SIMC.real_traj(dataset, actor, min_frame, max_frame)
    if ego_traj is None:
        raise ValueError(f"ego {ego} has no trajectory in [{min_frame},{max_frame}]")

    tree = ET.parse(xosc_in)
    root = tree.getroot()

    # parameters (variant overrides)
    for pd_ in root.iter("ParameterDeclaration"):
        name = pd_.get("name")
        if param_overrides and name in param_overrides:
            pd_.set("value", repr(param_overrides[name]))

    # optional NURBS shape-weight override (pipeline hardcodes 5.0 on shape points;
    # lower weight = looser fit to the sampled real points, stronger endpoint-offset
    # penetration into the mid-curve — the fidelity↔controllability tradeoff)
    if shape_weight is not None:
        for cp in root.iter("ControlPoint"):
            if cp.get("weight") is not None:
                cp.set("weight", f"{shape_weight}")

    # agent_replay: swap the challenge agent's NURBS for a POLYLINE of the real
    # trajectory, WITHOUT timing (TimeReference None, position mode) — geometry is
    # exact replay while progression stays driven by the SpeedActions, so the
    # critical-frame experiment can parameterize timing alone.
    if agent_replay and agent_traj is not None:
        for traj_el in root.iter("Trajectory"):
            shp = traj_el.find("Shape")
            if shp is None or shp.find("Nurbs") is None:
                continue                              # ego replay polyline: skip
            shp.remove(shp.find("Nurbs"))
            poly = ET.SubElement(shp, "Polyline")
            hdg = np.radians(agent_traj.heading)
            t0 = agent_traj.frame[0]
            for f, x, y, h in zip(agent_traj.frame, agent_traj.x, agent_traj.y, hdg):
                v = ET.SubElement(poly, "Vertex",
                                  time=f"{(f - t0) / agent_traj.fps:.6f}")
                pos = ET.SubElement(v, "Position")
                ET.SubElement(pos, "WorldPosition", x=f"{x:.4f}", y=f"{y:.4f}",
                              h=f"{h:.5f}")

    # local road network
    for lf in root.iter("LogicFile"):
        lf.set("filepath", str(paths.dataset_of(dataset)["xodr"]))
    if CATALOG_ENTITIES:
        # 保留 CatalogReference entity($Ego_Vehicle/$AgentN_Type),
        # CatalogLocations 標準化為 /opt/Catalogs block
        _standard_catalogs(root)
    else:
        for cl in root.findall("CatalogLocations"):
            root.remove(cl)
        # inline entities with dataset dims (challenge agent is "Agent1" for
        # vehicles, "Pedestrian1" for VRU — treat every non-Ego object as agent)
        for obj in root.iter("ScenarioObject"):
            if obj.get("name") == "Ego":
                _inline_entity(obj, "Ego", ego_traj.length, ego_traj.width)
            elif agent_traj is not None:
                _inline_entity(obj, obj.get("name"), agent_traj.length,
                               agent_traj.width)

    # drop every ControllerAction (ExternalControl / ACCController need catalogs;
    # the default esmini controller follows trajectories)
    for private in root.iter("Private"):
        for pa in list(private.findall("PrivateAction")):
            if pa.find("ControllerAction") is not None:
                private.remove(pa)

    storyboard = root.find("Storyboard")

    # remove the PISA outcome-detection story (server-side flags, inert here)
    for story in list(storyboard.findall("Story")):
        if "ParameterManeuver" in story.get("name", ""):
            storyboard.remove(story)

    # rewire Agent1's start: FLAG-AV_CONNECTED → SimulationTime > 0
    for cond in storyboard.iter("Condition"):
        bv = cond.find("ByValueCondition")
        pc = bv.find("ParameterCondition") if bv is not None else None
        if pc is not None and pc.get("parameterRef") == "FLAG-AV_CONNECTED":
            bv.remove(pc)
            ET.SubElement(bv, "SimulationTimeCondition", value="0.0",
                          rule="greaterThan")
            cond.set("conditionEdge", "none")

    # neutralize every FLAG-based StopTrigger (act + storyboard level)
    t_end = (max_frame - min_frame) / fps + margin_s
    for parent in [storyboard] + list(storyboard.iter("Act")):
        for stop in list(parent.findall("StopTrigger")):
            if stop.find("ConditionGroup") is not None:
                parent.remove(stop)
                parent.append(_sim_time_stop(t_end) if parent is storyboard
                              else ET.Element("StopTrigger"))
    if storyboard.find("StopTrigger") is None:
        storyboard.append(_sim_time_stop(t_end))

    # ego replay story (before the storyboard StopTrigger to keep schema order)
    stop = storyboard.find("StopTrigger")
    storyboard.remove(stop)
    storyboard.append(_ego_replay_story(ego_traj, min_frame))
    storyboard.append(stop)

    xosc_out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(xosc_out, encoding="utf-8", xml_declaration=True)
    return xosc_out


# ── run + extract ────────────────────────────────────────────────────────────

def _smooth(v: np.ndarray, win: int = 5) -> np.ndarray:
    """Centered moving average — esmini position logs carry ~cm-level jitter that
    finite differences amplify (2 cm @30 fps → 18 m/s² accel spikes)."""
    if len(v) < win:
        return v
    k = np.ones(win) / win
    pad = np.concatenate([np.full(win // 2, v[0]), v, np.full(win // 2, v[-1])])
    return np.convolve(pad, k, mode="valid")


def _derive_smoothed(frame, x, y, fps, win: int = 5):
    """derive_kinematics on jitter-smoothed positions (positions themselves are
    returned raw to the caller — only the derived heading/speed/accel use the
    smoothed signal)."""
    return SIMC.derive_kinematics(frame, _smooth(np.asarray(x, float), win),
                                  _smooth(np.asarray(y, float), win), fps)


def _parse_full_csv(csv_path: Path):
    """esmini csv_logger → {entity: DataFrame(t, x, y, h, speed, vx, vy, ax, ay)}.
    Uses esmini's OWN kinematics columns — position logs carry lane-snap jitter at
    junctions that finite differences would amplify into fake accel spikes, while
    Current_Speed / Acc are the simulator's true state."""
    import pandas as pd
    lines = Path(csv_path).read_text().splitlines()
    hi = next((i for i, l in enumerate(lines) if l.strip().startswith("Index")), None)
    if hi is None:
        return {}
    cols = [c.strip() for c in lines[hi].split(",")]
    df = pd.read_csv(csv_path, skiprows=hi + 1, header=None,
                     usecols=range(len(cols) - 1), names=cols[:-1],
                     skipinitialspace=True)
    out = {}
    for n in ("1", "2"):
        name_col = f"#{n} Entity_Name [-]"
        if name_col not in df.columns:
            continue
        ent = str(df[name_col].iloc[0]).strip()
        sub = pd.DataFrame({
            "t": df["TimeStamp [s]"].astype(float),
            "x": df[f"#{n} World_Position_X [m]"].astype(float),
            "y": df[f"#{n} World_Position_Y [m]"].astype(float),
            "z": df[f"#{n} World_Position_Z [m]"].astype(float),
            "h": df[f"#{n} World_Heading_Angle [rad]"].astype(float),
            "speed": df[f"#{n} Current_Speed [m/s]"].astype(float),
            "vx": df[f"#{n} Vel_X [m/s]"].astype(float),
            "vy": df[f"#{n} Vel_Y [m/s]"].astype(float),
            "ax": df[f"#{n} Acc_X [m/s2]"].astype(float),
            "ay": df[f"#{n} Acc_Y [m/s2]"].astype(float),
        })
        out[ent] = sub[sub["z"].abs() < 50].drop_duplicates("t").sort_values("t")
    return out


def run_and_extract(xosc: Path, dataset: str, ego: int, actor: int,
                    min_frame: int, max_frame: int, csv_out: Path | None = None,
                    reuse: bool = True):
    """esmini --headless → {"agent": Traj, "ego": Traj} (None on failure).
    reuse=True skips the esmini run when the csv already exists (with one forced
    re-run if the cached csv turns out truncated/unparseable). Frames are
    min_frame + t·fps (float); heading/speed/accel come from esmini's own state
    columns — symmetric with real_traj, which uses the dataset's recorded ones."""
    fps = paths.dataset_of(dataset)["fps"]
    if csv_out is None:
        csv_out = Path(str(xosc).replace(".xosc", ".csv"))
    csv_out = Path(csv_out)
    cached = reuse and csv_out.exists()
    if not cached and not _run_esmini(Path(xosc), csv_out, fps):
        return None
    res = extract_trajs(csv_out, dataset, ego, actor, min_frame, max_frame)
    if res is None and cached:                       # truncated cache → re-run once
        if not _run_esmini(Path(xosc), csv_out, fps):
            return None
        res = extract_trajs(csv_out, dataset, ego, actor, min_frame, max_frame)
    return res


def extract_trajs(csv_out: Path, dataset: str, ego: int, actor: int,
                  min_frame: int, max_frame: int):
    """Parse an existing esmini csv into {"agent": Traj, "ego": Traj} (None on failure)."""
    fps = paths.dataset_of(dataset)["fps"]
    try:
        ents = _parse_full_csv(csv_out)
    except Exception:                                # truncated/corrupt cache file
        return None
    agent_name = next((n for n in ents if n != "Ego"), None)
    if agent_name is None or "Ego" not in ents:
        return None
    dims = {"Ego": SIMC.real_traj(dataset, ego, min_frame, max_frame),
            agent_name: SIMC.real_traj(dataset, actor, min_frame, max_frame)}
    out = {}
    for entity, key in ((agent_name, "agent"), ("Ego", "ego")):
        d = ents.get(entity)
        if d is None:
            return None
        # score only the labeled window — the StopTrigger margin second after
        # max_frame exists just so esmini finishes the last in-window step
        d = d[min_frame + d["t"] * fps <= max_frame + 0.5]
        if len(d) < 3:
            return None
        frame = min_frame + d["t"].values * fps
        speed = d["speed"].values
        # longitudinal accel = d(Current_Speed)/dt — esmini's Acc_X/Y (and the
        # position log) spike at junction lane-snaps; Current_Speed stays smooth
        accel = np.gradient(speed, d["t"].values)
        ref = dims[entity]
        out[key] = SIMC.Traj(frame=frame, x=d["x"].values, y=d["y"].values,
                             heading=np.degrees(d["h"].values), speed=speed,
                             accel=accel, fps=fps,
                             length=ref.length if ref else float("nan"),
                             width=ref.width if ref else float("nan"),
                             meta={"dataset": dataset, "kind": "esmini",
                                   "entity": entity, "csv": str(csv_out)})
    return out


def esmini_pair(dataset: str, ego: int, actor: int, min_frame: int, max_frame: int,
                label: int | None = None, sample_cfg: C.SampleConfig | None = None,
                strategy_name: str = "5pt-default",
                param_overrides: dict | None = None,
                tag: str = "base", refresh: bool = False,
                shape_weight: float | None = None, force_anchor: bool = False,
                agent_replay: bool = False):
    """One-shot: pipeline .xosc → replay surgery → esmini → Trajs. Cached by tag."""
    base = base_xosc(dataset, ego, actor, min_frame, max_frame, label,
                     sample_cfg, strategy_name, force_anchor=force_anchor)
    if base is None:
        return None
    if shape_weight is not None:
        tag = f"{tag}_w{shape_weight:g}"
    run_dir = ESMINI_DIR / "runs" / strategy_name / base.stem
    xosc = run_dir / f"{tag}.xosc"
    csv = run_dir / f"{tag}.csv"
    if refresh or not csv.exists():
        to_esmini_replay(base, dataset, ego, actor, min_frame, max_frame, xosc,
                         param_overrides, shape_weight=shape_weight,
                         agent_replay=agent_replay)
    res = run_and_extract(xosc, dataset, ego, actor, min_frame, max_frame, csv)
    if res is not None:
        res["xosc"] = xosc
    return res


# ── esmini-executed similarity (same output shape as SIM.full_similarity) ────

def full_similarity_esmini(dataset: str, ego: int, actor: int, min_frame: int,
                           max_frame: int, label: int | None = None,
                           sample_cfg: C.SampleConfig | None = None,
                           strategy_name: str = "5pt-default",
                           estimator: str = "bbox",
                           force_anchor: bool = False) -> dict | None:
    """Path + interaction similarity where the parameterized agent is the ESMINI-
    EXECUTED artifact (replay ego in the same run). Key layout matches
    similarity.full_similarity, so analytic and executed rows share one CSV. Extra
    key: ego_replay_ade — the replay-fidelity sanity check (should be ≈ 0)."""
    from .similarity import path_sim, interaction_sim as IS
    res = esmini_pair(dataset, ego, actor, min_frame, max_frame, label,
                      sample_cfg, strategy_name, force_anchor=force_anchor)
    if res is None:
        return None
    a = SIMC.real_traj(dataset, actor, min_frame, max_frame)
    e = SIMC.real_traj(dataset, ego, min_frame, max_frame)
    if a is None or e is None:
        return None
    out = path_sim.similarity(a.xy, res["agent"].xy)
    rd = IS.descriptors(a, e, estimator)
    pd_ = IS.descriptors(res["agent"], res["ego"], estimator)
    out.update({f"real_{k}": v for k, v in rd.items()})
    out.update({f"param_{k}": v for k, v in pd_.items()})
    out.update(IS.similarity(rd, pd_, a.fps))
    out["ego_replay_ade"] = path_sim.ade(e.xy, res["ego"].xy)
    return out


# ── esmini-executed swept set (drop-in for sweep.coverage/variance/redundancy) ─

def _base_params(xosc: Path) -> dict:
    vals = {}
    for pd_ in ET.parse(xosc).getroot().iter("ParameterDeclaration"):
        if pd_.get("parameterType") == "double":
            try:
                vals[pd_.get("name")] = float(pd_.get("value"))
            except (TypeError, ValueError):
                pass
    return vals


def _three(v: float, d: float) -> list[float]:
    return [v - d, v, v + d]


def pipeline_axes(bp: dict, prefix: str) -> dict[str, list[float]]:
    """The REAL param.xosc sweep knobs (convert2yaml.build_param_xosc), each at
    3 values (min, default, max) → a 3×3×3 grid.
      non-ped: {Agent}_Offset ±0.5 | {Agent}_1_SA_EndSpeed ±speed_step (km/h,
               step = clamp(int(0.5·S), 10, 20), floor ≥0) |
               {Agent}_1_SA_DynamicDuration ±duration_step (step = min(int(0.5·d),2);
               d<2 uses the pipeline's [2,4] bounds around the default)
      ped:     Pedestrian1_Offset ±0.5 | Pedestrian1_1_Delay (pipeline ±2 s bounds,
               ≥0; midpoint when the default sits on a bound) |
               Pedestrian1_1_TA_Offset ±0.5"""
    if prefix == "Pedestrian1":
        d = bp.get("Pedestrian1_1_Delay", 0.0)
        lo, hi = max(0.0, d - 2.0), d + 2.0
        mid = d if lo < d < hi else (lo + hi) / 2.0
        return {"Pedestrian1_Offset": _three(bp["Pedestrian1_Offset"], 0.5),
                "Pedestrian1_1_Delay": [lo, mid, hi],
                "Pedestrian1_1_TA_Offset": _three(bp["Pedestrian1_1_TA_Offset"], 0.5)}
    S = bp[f"{prefix}_1_SA_EndSpeed"]                    # km/h
    sstep = max(min(int(S * 0.5), 20), 10)
    dur = bp[f"{prefix}_1_SA_DynamicDuration"]
    dstep = min(int(dur * 0.5), 2)
    dvals = [2.0, dur, 4.0] if dur < 2 else [max(0.0, dur - dstep), dur, dur + dstep]
    return {f"{prefix}_Offset": _three(bp[f"{prefix}_Offset"], 0.5),
            f"{prefix}_1_SA_EndSpeed": [max(0.0, S - sstep), S, S + sstep],
            f"{prefix}_1_SA_DynamicDuration": dvals}


def _grid_of(axes: dict[str, list[float]]) -> list[dict]:
    from itertools import product
    names = list(axes)
    return [dict(zip(names, combo)) for combo in product(*axes.values())]


def esmini_swept_set(dataset: str, ego: int, actor: int, min_frame: int,
                     max_frame: int, label: int | None = None,
                     sample_cfg: C.SampleConfig | None = None,
                     strategy_name: str = "5pt-default",
                     estimator: str = "bbox",
                     verbose: bool = True,
                     shape_weight: float | None = None,
                     force_anchor: bool = False,
                     axes_override: dict | None = None,
                     agent_replay: bool = False) -> SweptSet | None:
    """Render the PIPELINE parameter grid (pipeline_axes: the same knobs and ranges
    param.xosc declares for the CARLA sweep, 3 values each = 27 members) by
    EXECUTING one esmini run per member. Variant.params carries the actual xosc
    parameter names/values. `axes_override` supplies a FIXED axes dict (e.g. the
    scenario's reference-method axes) so every sampling method sweeps the exact
    same parameter values — removes the residual axis confound in method
    comparisons. Returns a sweep.core.SweptSet."""
    base = base_xosc(dataset, ego, actor, min_frame, max_frame, label,
                     sample_cfg, strategy_name, force_anchor=force_anchor)
    if base is None:
        return None
    bp = _base_params(base)
    prefix = "Pedestrian1" if "Pedestrian1_Offset" in bp else "Agent1"
    a = SIMC.real_traj(dataset, actor, min_frame, max_frame)
    e = SIMC.real_traj(dataset, ego, min_frame, max_frame)
    if a is None or e is None:
        return None
    from .sweep.core import _derived_kinematics_copy
    out = SweptSet(dataset, ego, actor, min_frame, max_frame, agent=a, ego_traj=e,
                   real=descriptors(_derived_kinematics_copy(a), e, estimator))
    grid = _grid_of(axes_override if axes_override is not None
                    else pipeline_axes(bp, prefix))
    fps = paths.dataset_of(dataset)["fps"]
    run_dir = ESMINI_DIR / "runs" / strategy_name / base.stem

    # phase 1 — surgery for every member whose csv is not cached (cheap, serial)
    members, jobs = [], []
    for ov in grid:
        tag = "_".join(f"{n.split('_')[-1][:2].lower()}{v:+.2f}" for n, v in ov.items())
        if shape_weight is not None:
            tag = f"{tag}_w{shape_weight:g}"
        xosc, csv = run_dir / f"{tag}.xosc", run_dir / f"{tag}.csv"
        members.append((ov, tag, csv))
        if not csv.exists():
            to_esmini_replay(base, dataset, ego, actor, min_frame, max_frame, xosc,
                             ov, shape_weight=shape_weight, agent_replay=agent_replay)
            jobs.append((xosc, csv))

    # phase 2 — esmini runs in PARALLEL (independent single-threaded subprocesses;
    # subprocess.run releases the GIL, so a thread pool is enough)
    if jobs:
        import os
        from concurrent.futures import ThreadPoolExecutor
        workers = max(1, min(len(jobs), ESMINI_WORKERS or (os.cpu_count() or 4) - 2))
        with ThreadPoolExecutor(workers) as pool:
            oks = list(pool.map(lambda j: _run_esmini(j[0], j[1], fps), jobs))
        if verbose and not all(oks):
            print(f"    {oks.count(False)}/{len(jobs)} esmini runs failed")

    # phase 3 — extract + score (serial; cached csvs get a retry via run_and_extract)
    for ov, tag, csv in members:
        res = run_and_extract(run_dir / f"{tag}.xosc", dataset, ego, actor,
                              min_frame, max_frame, csv)
        if res is None:
            if verbose:
                print(f"    variant {tag}: no trajectory, skipped")
            continue
        vt = res["agent"]
        out.variants.append(Variant(params=dict(ov), traj=vt,
                                    desc=descriptors(vt, res["ego"], estimator)))
    return out if out.variants else None
