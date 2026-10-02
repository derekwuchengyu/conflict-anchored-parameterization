"""Background traffic for the 39_180 scenarios.

The retrieval pipeline never puts label-0 traffic into a scenario:
`new_batch_convert.py` skips `label_idx == 0` outright and builds
`replayer_actors` only from the OTHER labelled interactions of the same ego.
So the "none" partners of ego 39 — the ordinary traffic sharing the junction —
are absent from every generated xosc.

This module adds them back by post-hoc surgery on the already-parameterized
xosc, so agent1's parameterization stays bit-identical: each background track
becomes an extra ScenarioObject replaying its recorded trajectory, spawned when
its track starts and deleted when it ends (esmini supports Add/DeleteEntityAction).
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib64 as L  # noqa: E402

from hetero_param.similarity import core as SIMC  # noqa: E402

LABELS = Path("/home/hcis-s19/Documents/ChengYu/HetroD-labeler/data/"
              "00_labeled_scenarios.json")
META = Path("/home/hcis-s19/Documents/ChengYu/New_HetroD/HetroD/data/"
            "00_tracksMeta.csv")
NONE_LABEL = 0
T_END_FRAME = 2923          # agent1's GT track ends here; window = 2424..2923

# background models — deliberately NOT the ego's car_white or agent1's mc.osgb,
# so the two scenario actors stay identifiable among the traffic
CAR_MODELS = ["car_blue.osgb", "car_red.osgb", "van_red.osgb", "car_yellow.osgb"]
MODEL = {"truck": ("truck", "truck_yellow.osgb"),
         "bus": ("bus", "bus_blue.osgb"),
         "motorcycle": ("motorbike", "scooter.osgb"),
         "bicycle": ("bicycle", "cyclist.osgb"),
         "pedestrian": ("pedestrian", "walkman.osgb")}


def bg_ids(min_cover: float = 0.02) -> list[int]:
    """Track ids the labeller paired with ego 39 under label 0 (= NONE)."""
    lab = json.loads(LABELS.read_text())
    ids = sorted(int(v["actor_id"]) for v in lab.values()
                 if int(v["ego_id"]) == L.EGO and v.get("label_idx") == NONE_LABEL)
    out = []
    n_win = T_END_FRAME - L.MIN_FRAME + 1
    for i in ids:
        t = SIMC.real_traj(L.DATASET, i, L.MIN_FRAME, T_END_FRAME)
        if t is not None and len(t.frame) / n_win >= min_cover:
            out.append(i)
    return out


def track_class() -> dict[int, str]:
    m = pd.read_csv(META)
    return dict(zip(m.trackId.astype(int), m["class"].astype(str)))


def set_scale_mode(el: ET.Element, mode: str = "ModelToBB") -> None:
    """Force esmini to scale the 3D MODEL onto the declared BoundingBox.

    Without it, attaching `model3d` makes esmini adopt the *model's* bounds as
    the entity bounding box: the ego became 4.5 x 2.1 m instead of the recorded
    4.47 x 1.90, the motorcycle 2.0 x 0.71 instead of 1.61 x 0.67, and both
    picked up a 1.5 m longitudinal box-centre offset (`bb_x`).  That is what the
    renderer draws AND what esmini's own `--collision` detector uses, so the
    picture and esmini's verdict drifted away from the dataset geometry the
    metrics are computed on.
    """
    props = el.find("Properties")
    if props is None:
        props = ET.SubElement(el, "Properties")
    for p in props.findall("Property"):
        if p.get("name") == "scaleMode":
            p.set("value", mode)
            return
    ET.SubElement(props, "Property", name="scaleMode", value=mode)


def _entity(name: str, cls: str, length: float, width: float,
            car_ix: int) -> ET.Element:
    obj = ET.Element("ScenarioObject", name=name)
    cat, model = MODEL.get(cls, ("car", CAR_MODELS[car_ix % len(CAR_MODELS)]))
    Lm = float(length) if np.isfinite(length) and length > 0.1 else (
        0.6 if cls == "pedestrian" else 4.4)
    Wm = float(width) if np.isfinite(width) and width > 0.1 else (
        0.6 if cls == "pedestrian" else 1.8)
    if cls == "pedestrian":
        ped = ET.SubElement(obj, "Pedestrian", name=f"{name}_ped", mass="80.0",
                            model="ped", pedestrianCategory="pedestrian",
                            model3d=f"../models/{model}")
        bb = ET.SubElement(ped, "BoundingBox")
        ET.SubElement(bb, "Center", x="0", y="0", z="0.9")
        ET.SubElement(bb, "Dimensions", width=f"{Wm}", length=f"{Lm}", height="1.8")
        ET.SubElement(ped, "Properties")
        set_scale_mode(ped)
        return obj
    veh = ET.SubElement(obj, "Vehicle", name=f"{name}_vehicle",
                        vehicleCategory=cat, model3d=f"../models/{model}")
    ET.SubElement(veh, "ParameterDeclarations")
    ET.SubElement(veh, "Performance", maxSpeed="70", maxAcceleration="15",
                  maxDeceleration="15")
    bb = ET.SubElement(veh, "BoundingBox")
    ET.SubElement(bb, "Center", x="0", y="0", z="0.75")
    ET.SubElement(bb, "Dimensions", width=f"{Wm}", length=f"{Lm}",
                  height="1.5" if cat != "truck" else "3.0")
    ax = ET.SubElement(veh, "Axles")
    wb = max(Lm * 0.6, 0.8)
    ET.SubElement(ax, "FrontAxle", maxSteering="0.5", wheelDiameter="0.6",
                  trackWidth=f"{Wm}", positionX=f"{wb}", positionZ="0.3")
    ET.SubElement(ax, "RearAxle", maxSteering="0.0", wheelDiameter="0.6",
                  trackWidth=f"{Wm}", positionX="0", positionZ="0.3")
    ET.SubElement(veh, "Properties")
    set_scale_mode(veh)
    return obj


def _world_pos(parent: ET.Element, x: float, y: float, h: float) -> None:
    pos = ET.SubElement(parent, "Position")
    ET.SubElement(pos, "WorldPosition", x=f"{x:.4f}", y=f"{y:.4f}", h=f"{h:.5f}")


def _time_cond(name: str, t: float) -> ET.Element:
    cond = ET.Element("Condition", name=name, delay="0.0", conditionEdge="none")
    bv = ET.SubElement(cond, "ByValueCondition")
    ET.SubElement(bv, "SimulationTimeCondition", value=f"{max(t, 0.0):.3f}",
                  rule="greaterThan")
    return cond


def _event(name: str, t: float) -> ET.Element:
    ev = ET.Element("Event", name=name, priority="parallel",
                    maximumExecutionCount="1")
    return ev


def _attach_trigger(ev: ET.Element, name: str, t: float) -> None:
    st = ET.SubElement(ev, "StartTrigger")
    cg = ET.SubElement(st, "ConditionGroup")
    cg.append(_time_cond(name, t))


def _bg_story(name: str, traj, t_start: float, t_end: float,
              window_end: float) -> ET.Element:
    """Replay story for one background actor, with spawn / despawn.

    esmini activates every declared entity at t=0, so a track that starts late
    would otherwise sit parked at its entry point, and one that ends early would
    freeze there for the rest of the run.  Delete-then-Add brackets the replay
    with the track's real lifetime.
    """
    story = ET.Element("Story", name=f"story_{name}")
    act = ET.SubElement(story, "Act", name=f"act_{name}")
    mg = ET.SubElement(act, "ManeuverGroup", name=f"mg_{name}",
                       maximumExecutionCount="1")
    actors = ET.SubElement(mg, "Actors", selectTriggeringEntities="false")
    ET.SubElement(actors, "EntityRef", entityRef=name)
    man = ET.SubElement(mg, "Maneuver", name=f"man_{name}")

    late = t_start > 0.05
    early = t_end < window_end - 0.05

    if late:
        ev = ET.SubElement(man, "Event", name=f"{name}_hide",
                           priority="parallel", maximumExecutionCount="1")
        a = ET.SubElement(ev, "Action", name=f"{name}_hide_a")
        ga = ET.SubElement(a, "GlobalAction")
        ea = ET.SubElement(ga, "EntityAction", entityRef=name)
        ET.SubElement(ea, "DeleteEntityAction")
        _attach_trigger(ev, f"{name}_hide_t", 0.0)

        ev = ET.SubElement(man, "Event", name=f"{name}_spawn",
                           priority="parallel", maximumExecutionCount="1")
        a = ET.SubElement(ev, "Action", name=f"{name}_spawn_a")
        ga = ET.SubElement(a, "GlobalAction")
        ea = ET.SubElement(ga, "EntityAction", entityRef=name)
        add = ET.SubElement(ea, "AddEntityAction")
        _world_pos(add, traj.x[0], traj.y[0], np.radians(traj.heading[0]))
        _attach_trigger(ev, f"{name}_spawn_t", max(t_start - 0.05, 0.0))

    ev = ET.SubElement(man, "Event", name=f"{name}_follow",
                       priority="parallel", maximumExecutionCount="1")
    a = ET.SubElement(ev, "Action", name=f"{name}_follow_a")
    pa = ET.SubElement(a, "PrivateAction")
    ra = ET.SubElement(pa, "RoutingAction")
    fta = ET.SubElement(ra, "FollowTrajectoryAction")
    tr = ET.SubElement(fta, "Trajectory", name=f"traj_{name}", closed="false")
    shape = ET.SubElement(tr, "Shape")
    poly = ET.SubElement(shape, "Polyline")
    f0 = traj.frame[0]
    for f, x, y, h in zip(traj.frame, traj.x, traj.y, np.radians(traj.heading)):
        v = ET.SubElement(poly, "Vertex", time=f"{(f - f0) / traj.fps:.6f}")
        _world_pos(v, x, y, h)
    tref = ET.SubElement(fta, "TimeReference")
    ET.SubElement(tref, "Timing", domainAbsoluteRelative="absolute",
                  scale="1", offset="0")
    ET.SubElement(fta, "TrajectoryFollowingMode", followingMode="position")
    _attach_trigger(ev, f"{name}_follow_t", max(t_start - 0.02, 0.0))

    if early:
        ev = ET.SubElement(man, "Event", name=f"{name}_despawn",
                           priority="parallel", maximumExecutionCount="1")
        a = ET.SubElement(ev, "Action", name=f"{name}_despawn_a")
        ga = ET.SubElement(a, "GlobalAction")
        ea = ET.SubElement(ga, "EntityAction", entityRef=name)
        ET.SubElement(ea, "DeleteEntityAction")
        _attach_trigger(ev, f"{name}_despawn_t", t_end)

    ast = ET.SubElement(act, "StartTrigger")
    acg = ET.SubElement(ast, "ConditionGroup")
    acg.append(_time_cond(f"{name}_act", 0.0))
    ET.SubElement(act, "StopTrigger")
    return story


def add_background(xosc_in: Path, xosc_out: Path,
                   ids: list[int] | None = None) -> tuple[Path, list[int]]:
    """Copy `xosc_in` with the label-0 traffic added as replayed entities."""
    ids = bg_ids() if ids is None else ids
    cls_of = track_class()
    tree = ET.parse(xosc_in)
    root = tree.getroot()
    entities = root.find("Entities")
    storyboard = root.find("Storyboard")
    init_actions = storyboard.find("Init/Actions")
    window_end = (T_END_FRAME - L.MIN_FRAME) / L.FPS

    stop = storyboard.find("StopTrigger")
    if stop is not None:
        storyboard.remove(stop)

    used = []
    car_ix = 0
    for tid in ids:
        traj = SIMC.real_traj(L.DATASET, tid, L.MIN_FRAME, T_END_FRAME)
        if traj is None or len(traj.frame) < 5:
            continue
        cls = cls_of.get(tid, "car")
        name = f"Bg{tid}"
        entities.append(_entity(name, cls, traj.length, traj.width, car_ix))
        if cls not in MODEL:
            car_ix += 1
        priv = ET.SubElement(init_actions, "Private", entityRef=name)
        pa = ET.SubElement(priv, "PrivateAction")
        ta = ET.SubElement(pa, "TeleportAction")
        _world_pos(ta, traj.x[0], traj.y[0], np.radians(traj.heading[0]))
        t_start = (traj.frame[0] - L.MIN_FRAME) / L.FPS
        t_end = (traj.frame[-1] - L.MIN_FRAME) / L.FPS
        storyboard.append(_bg_story(name, traj, t_start, t_end, window_end))
        used.append(tid)

    if stop is not None:
        storyboard.append(stop)
    xosc_out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(xosc_out, encoding="utf-8", xml_declaration=True)
    return xosc_out, used


if __name__ == "__main__":
    ids = bg_ids()
    cls_of = track_class()
    print(f"label-0 (NONE) partners of ego {L.EGO}: {ids}")
    n_win = T_END_FRAME - L.MIN_FRAME + 1
    for i in ids:
        t = SIMC.real_traj(L.DATASET, i, L.MIN_FRAME, T_END_FRAME)
        print(f"  {i:4d} {cls_of.get(i,'?'):11s} frames {t.frame[0]:.0f}-{t.frame[-1]:.0f}"
              f"  cover {100 * len(t.frame) / n_win:3.0f}%  "
              f"{t.length:.2f} x {t.width:.2f} m")
