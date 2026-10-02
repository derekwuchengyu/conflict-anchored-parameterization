"""Execute a scenario in esmini with the ego replaying its recorded trajectory.

to_esmini_replay rewrites a logical xosc into a self-contained esmini file:
the ego follows its recorded polyline with absolute timing, Agent1 starts at
simulation time 0, controller actions are removed, and the run stops at the
end of the annotated window plus a margin.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

CATALOGS = [("VehicleCatalog", "Vehicles"), ("PedestrianCatalog", "Pedestrians"),
            ("ControllerCatalog", "Controllers"), ("EnvironmentCatalog", "Environments")]


def _standard_catalogs(root, catalog_root):
    for cl in root.findall("CatalogLocations"):
        root.remove(cl)
    cl = ET.Element("CatalogLocations")
    for tag, sub in CATALOGS:
        ET.SubElement(ET.SubElement(cl, tag), "Directory", path=f"{catalog_root}/{sub}")
    idx = 0
    for i, child in enumerate(list(root)):
        if child.tag in ("FileHeader", "ParameterDeclarations"):
            idx = i + 1
    root.insert(idx, cl)


def _condition(parent, name):
    cond = ET.SubElement(parent, "Condition", name=name, delay="0.0", conditionEdge="none")
    ET.SubElement(ET.SubElement(cond, "ByValueCondition"), "SimulationTimeCondition",
                  value="0.0", rule="greaterThan")


def _ego_replay_story(ego, fps):
    """Story: ego follows its recorded trajectory; vertex k at t = (frame_k - frame_0) / fps."""
    story = ET.Element("Story", name="story_EgoReplay")
    act = ET.SubElement(story, "Act", name="act_EgoReplay")
    mg = ET.SubElement(act, "ManeuverGroup", name="mg_EgoReplay", maximumExecutionCount="1")
    ET.SubElement(ET.SubElement(mg, "Actors", selectTriggeringEntities="false"), "EntityRef", entityRef="Ego")
    man = ET.SubElement(mg, "Maneuver", name="EgoReplay_Maneuver")
    ev = ET.SubElement(man, "Event", name="EgoReplay_Event", priority="overwrite", maximumExecutionCount="1")
    action = ET.SubElement(ev, "Action", name="EgoReplay_Follow")
    fta = ET.SubElement(ET.SubElement(ET.SubElement(action, "PrivateAction"), "RoutingAction"),
                        "FollowTrajectoryAction")
    traj = ET.SubElement(fta, "Trajectory", name="EgoReplayTrajectory", closed="false")
    poly = ET.SubElement(ET.SubElement(traj, "Shape"), "Polyline")
    frames = ego.frame.to_numpy(float)
    for f, x, y, h in zip(frames, ego.x, ego.y, np.radians(ego.heading.to_numpy(float))):
        v = ET.SubElement(poly, "Vertex", time=f"{(f - frames[0]) / fps:.6f}")
        ET.SubElement(ET.SubElement(v, "Position"), "WorldPosition", x=f"{x:.4f}", y=f"{y:.4f}", h=f"{h:.5f}")
    ET.SubElement(ET.SubElement(fta, "TimeReference"), "Timing",
                  domainAbsoluteRelative="absolute", scale="1", offset="0")
    ET.SubElement(fta, "TrajectoryFollowingMode", followingMode="position")
    _condition(ET.SubElement(ET.SubElement(ev, "StartTrigger"), "ConditionGroup"), "ego_replay_start")
    _condition(ET.SubElement(ET.SubElement(act, "StartTrigger"), "ConditionGroup"), "act_start")
    ET.SubElement(act, "StopTrigger")
    return story


def _sim_time_stop(t_end):
    stop = ET.Element("StopTrigger")
    cond = ET.SubElement(ET.SubElement(stop, "ConditionGroup"), "Condition",
                         name="end_of_window", delay="0.0", conditionEdge="none")
    ET.SubElement(ET.SubElement(cond, "ByValueCondition"), "SimulationTimeCondition",
                  value=f"{t_end:.3f}", rule="greaterThan")
    return stop


def to_esmini_replay(xosc_in, xosc_out, ego, xodr, fps, min_frame, max_frame,
                     catalog_root="/opt/Catalogs", margin_s=1.0):
    """ego: recorded ego window track (columns frame, x, y, heading [deg])."""
    tree = ET.parse(xosc_in)
    root = tree.getroot()
    for lf in root.iter("LogicFile"):
        lf.set("filepath", str(xodr))
    _standard_catalogs(root, catalog_root)
    for private in root.iter("Private"):                 # default controller follows trajectories
        for pa in list(private.findall("PrivateAction")):
            if pa.find("ControllerAction") is not None:
                private.remove(pa)
    storyboard = root.find("Storyboard")
    for story in list(storyboard.findall("Story")):      # server-side outcome story
        if "ParameterManeuver" in story.get("name", ""):
            storyboard.remove(story)
    for cond in storyboard.iter("Condition"):            # FLAG-AV_CONNECTED -> t > 0
        bv = cond.find("ByValueCondition")
        pc = bv.find("ParameterCondition") if bv is not None else None
        if pc is not None and pc.get("parameterRef") == "FLAG-AV_CONNECTED":
            bv.remove(pc)
            ET.SubElement(bv, "SimulationTimeCondition", value="0.0", rule="greaterThan")
            cond.set("conditionEdge", "none")
    t_end = (max_frame - min_frame) / fps + margin_s
    for parent in [storyboard] + list(storyboard.iter("Act")):
        for stop in list(parent.findall("StopTrigger")):
            if stop.find("ConditionGroup") is not None:
                parent.remove(stop)
                parent.append(_sim_time_stop(t_end) if parent is storyboard else ET.Element("StopTrigger"))
    if storyboard.find("StopTrigger") is None:
        storyboard.append(_sim_time_stop(t_end))
    stop = storyboard.find("StopTrigger")
    storyboard.remove(stop)
    storyboard.append(_ego_replay_story(ego, fps))
    storyboard.append(stop)
    xosc_out = Path(xosc_out)
    xosc_out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(xosc_out, encoding="utf-8", xml_declaration=True)
    return xosc_out


def run(xosc, workdir, esmini_bin, fps, timeout=60):
    """Run esmini headless at the dataset frame rate; returns the csv log path."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "run.csv"
    cmd = [str(esmini_bin), "--osc", str(Path(xosc).resolve()), "--headless",
           "--csv_logger", str(csv_path.resolve()), "--fixed_timestep", str(1.0 / fps),
           "--logfile_path", str((workdir / "esmini.log").resolve()), "--disable_stdout"]
    subprocess.run(cmd, cwd=workdir, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return csv_path


def read_csv(csv_path, min_frame, fps) -> pd.DataFrame:
    """esmini csv log -> rows (entity, time_s, frame, x, y, heading_deg, speed_mps)."""
    lines = Path(csv_path).read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("Index"))
    columns = [c.strip() for c in lines[start].split(",")]
    while columns and not columns[-1]:
        columns.pop()
    df = pd.read_csv(csv_path, skiprows=start + 1, names=columns, header=None,
                     usecols=range(len(columns)), skipinitialspace=True)
    out = []
    for column in columns:
        m = re.fullmatch(r"#(\d+) Entity_Name \[-\]", column)
        if not m:
            continue
        n = m.group(1)
        out.append(pd.DataFrame(dict(
            entity=str(df[column].iloc[0]).strip(), time_s=df["TimeStamp [s]"].astype(float),
            x=df[f"#{n} World_Position_X [m]"].astype(float), y=df[f"#{n} World_Position_Y [m]"].astype(float),
            heading_deg=np.degrees(df[f"#{n} World_Heading_Angle [rad]"].astype(float)),
            speed_mps=df[f"#{n} Current_Speed [m/s]"].astype(float))))
    tidy = pd.concat(out, ignore_index=True)
    tidy.insert(2, "frame", min_frame + tidy.time_s * fps)
    return tidy
