#!/usr/bin/env python3
"""WP2 - "stop at end" re-execution of four already-rendered arms.

Motivation (HANDOFF_OPUS_20260911.md, user decision 3).  In the existing
executions the challenge agent keeps rolling after its trajectory ends: when the
FollowTrajectoryAction completes, esmini's default controller takes the entity
over and carries it along the road at its last speed until the storyboard's
SimulationTime StopTrigger (window + 1 s).  Table 5 then scores that free-running
tail as if the method had produced it.  The SVD arm already has a like-for-like
answer (scripts/28_truncate_to_polyline_support.py: drop the target rows after
the last decoded vertex); the ours arms have no such support flag, so the fair
fix is to make the *simulation* end the motion: brake the target to 0 the moment
its trajectory ends, exactly as the SAKURA-route bases do at their goal
(runs/sakura_route_defaults/.../run.xosc, Event `Agent1_StopAtGoalEvent`:
SpeedAction, step dynamics, AbsoluteTargetSpeed 0).

Mechanism.  The source of every stop sample is the *existing* `run.xosc` of the
corresponding window sample - the fully resolved executable that esmini already
ran.  A single Event is inserted textually just before `</Maneuver>` of
`Agent1_Maneuver`; nothing else in the file is touched, so `diff` shows added
lines only and the ElementTree signature of the rest is bit-identical.  The
trigger candidates, in the order the pilot tries them:

  action_complete  StoryboardElementStateCondition storyboardElementType="action"
                   storyboardElementRef="<the Agent1 FollowTrajectoryAction>"
                   state="completeState"                         (preferred)
  event_complete   ... storyboardElementType="event" ref=<its Event> completeState
  action_end       ... storyboardElementType="action" state="endTransition"
  reach_position   ReachPositionCondition (tolerance 2.0 m) at the LAST control
                   point (ours: the goal LanePosition verbatim) / the LAST
                   polyline vertex (SVD: WorldPosition)

Execution is byte-for-byte the same recipe as scripts/20_ours3_generate.py and
scripts/26_svd_execute.py: the same esmini binary through bin/esmini, the same
flags (--headless --csv_logger --fixed_timestep 1/fps --logfile_path
--disable_stdout, nice 10, 60 s timeout), one worker, and `parse_tidy` of 20 for
the tidy trajectory.parquet, so the 20-sample layout and the sample.json keys the
31 / 32 / 45 scorers read are preserved.  For the SVD arms the additive
`within_decoded_polyline_support` column of 26 is reproduced from the source
sample's decoded polyline so the schema matches its window counterpart.

Stages
  pilot   : --n samples per arm, every trigger candidate, PROOF from run.csv
            (speed 0 within 0.3 s of the trajectory end, then < 0.05 m drift,
            ego identical to the source run, xosc diff = insertions only)
  render  : mass render of one arm with the trigger the pilot proved
  status  : counts per arm

Nothing under runs/<source batch>/ or any existing results file is modified.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RAW = ROOT / "HetroD-labeler/data/00_tracks.parquet"
BINARY = ROOT / "esmini/bin/esmini"
RUNNER_20 = PROJECT / "scripts/20_ours3_generate.py"
RUNNER_26 = PROJECT / "scripts/26_svd_execute.py"
OUT = PROJECT / "results/x_stop"
PLANS = PROJECT / "plans/x_stop"
LOG = OUT / "progress.log"
MIN_FREE = 10 * 1024 ** 3
FPS = 30.0
SUBPROCESS_TIMEOUT_S = 60

STOP_EVENT = "Agent1_StopAtEndEvent"
STOP_ACTION = "Agent1_StopAtEndAction"
STOP_CONDITION = "trajectory_end"
REACH_TOLERANCE_M = 2.0
TRIGGERS = ["action_complete", "event_complete", "action_end", "reach_position"]

# proof gates (handoff WP2)
ZERO_SPEED_MPS = 1e-3           # esmini writes exact 0.0 for a step-to-zero SpeedAction
ZERO_WITHIN_S = 0.3
DRIFT_MAX_M = 0.05
EGO_IDENTITY_MAX = 1e-6

_spec = importlib.util.spec_from_file_location("ours3_runner_20", RUNNER_20)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)      # parse_tidy, sha256, atomic_json, params, signature

ARMS = {
    "ours3_disk_stop": dict(
        source_arm="ours3_disk", kind="ours",
        source_batch="runs/ours3_disk_population_defaults", suffix=None, expect=466,   # 469 pre-v3; 470 with cut-in right at L=5; 466 after reverting cut-in right to L=10
        label="ours3 disk defaults, stop-at-end (executed)",
        window_arm="ours3_disk"),
    "ours3_disk_kde_stop_s20260910": dict(
        source_arm="ours3_disk_kde", kind="ours",
        source_batch="runs/ours3_disk_kde_s20260910_draw000001_001000_pool1000", suffix=None, expect=5000,
        label="ours3 disk + KDE seed 20260910, stop-at-end (executed)",
        window_arm="ours3_disk_kde"),
    "svd_exec_stop_recon_fullfit": dict(
        source_arm="svd_exec_E3_recon_fullfit", kind="svd",
        source_batch="runs/svd_d5_executed_recon", suffix="__fullfit", expect=486,   # 489 before the 2026-09-14 v3 rerun
        label="SVD executed (timed Polyline, E3) - fullfit recon, stop-at-end",
        window_arm="svd_exec_E3_recon_fullfit"),
    "svd_exec_stop_kde": dict(
        source_arm="svd_exec_E3_kde_kde", kind="svd",
        source_batch="runs/svd_d5_executed_kde", suffix=None, expect=1500,
        label="SVD executed (timed Polyline, E3) - KDE, stop-at-end",
        window_arm="svd_exec_E3_kde_kde"),
}


# ── small helpers ────────────────────────────────────────────────────────────

def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [93] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def source_batch_dir(arm: str) -> Path:
    spec = ARMS[arm]
    latest = PROJECT / spec["source_batch"] / "latest_batch.json"
    return Path(json.loads(latest.read_text())["batch_dir"])


def source_samples(arm: str) -> list[Path]:
    """sample.json paths of the source batch, filtered by the arm's suffix."""
    spec = ARMS[arm]
    root = source_batch_dir(arm)
    out = []
    for sj in sorted(root.glob("*/sample.json")):
        if spec["suffix"] and not sj.parent.name.endswith(spec["suffix"]):
            continue
        out.append(sj)
    return out


# ── xosc surgery: insert ONE Event, textual, nothing else touched ────────────

def locate(xosc_path: Path) -> dict:
    """Names / geometry the stop trigger needs, validated against the parsed tree."""
    root = ET.parse(xosc_path).getroot()
    sb = root.find("Storyboard")
    hit = None
    for story in sb.findall("Story"):
        for act in story.findall("Act"):
            for mg in act.findall("ManeuverGroup"):
                actors = [e.get("entityRef") for e in mg.findall("Actors/EntityRef")]
                if actors != ["Agent1"]:
                    continue
                for man in mg.findall("Maneuver"):
                    evs = [e for e in man.findall("Event") if e.find(".//FollowTrajectoryAction") is not None]
                    if not evs:
                        continue
                    assert len(evs) == 1, f"{xosc_path}: {len(evs)} Agent1 FollowTrajectory events"
                    acts = [a for a in evs[0].findall("Action") if a.find(".//FollowTrajectoryAction") is not None]
                    assert len(acts) == 1
                    assert hit is None, f"{xosc_path}: more than one Agent1 trajectory maneuver"
                    fta = evs[0].find(".//FollowTrajectoryAction")
                    shape = fta.find(".//Trajectory/Shape")
                    nurbs, poly = shape.find("Nurbs"), shape.find("Polyline")
                    assert (nurbs is None) != (poly is None), f"{xosc_path}: shape must be Nurbs xor Polyline"
                    if nurbs is not None:
                        cps = nurbs.findall("ControlPoint")
                        last = cps[-1].find("Position")
                        shape_kind, n_geom = "Nurbs", len(cps)
                    else:
                        verts = poly.findall("Vertex")
                        last = verts[-1].find("Position")
                        shape_kind, n_geom = "Polyline", len(verts)
                    assert len(list(last)) == 1, "unexpected Position content"
                    hit = dict(maneuver=man.get("name"), event=evs[0].get("name"), action=acts[0].get("name"),
                               shape=shape_kind, n_geometry=n_geom,
                               last_position_xml=ET.tostring(list(last)[0], encoding="unicode").strip(),
                               last_position_tag=list(last)[0].tag)
    assert hit is not None, f"{xosc_path}: no Agent1 FollowTrajectory maneuver"
    existing = sorted(e.get("name") for e in root.iter("Event")
                      if "StopAtEnd" in (e.get("name") or "") or "StopAtGoal" in (e.get("name") or ""))
    assert not existing, f"{xosc_path}: a stop event already exists {existing}"
    return hit


def stop_event_xml(info: dict, trigger: str, indent: str) -> str:
    """The inserted Event, formatted with the maneuver's own indentation."""
    i, i2, i3, i4, i5, i6, i7 = (indent + "    " * k for k in range(7))
    if trigger == "reach_position":
        pos = info["last_position_xml"]
        # a WorldPosition may carry h; ReachPositionCondition only uses the point
        if info["last_position_tag"] == "WorldPosition":
            el = ET.fromstring(pos)
            keep = {k: el.get(k) for k in ("x", "y") if el.get(k) is not None}
            pos = "<WorldPosition " + " ".join(f'{k}="{v}"' for k, v in keep.items()) + " />"
        cond = (f'{i5}<ByEntityCondition>\n'
                f'{i6}<TriggeringEntities triggeringEntitiesRule="any">\n'
                f'{i7}<EntityRef entityRef="Agent1" />\n'
                f'{i6}</TriggeringEntities>\n'
                f'{i6}<EntityCondition>\n'
                f'{i7}<ReachPositionCondition tolerance="{REACH_TOLERANCE_M}">\n'
                f'{i7}    <Position>\n'
                f'{i7}        {pos}\n'
                f'{i7}    </Position>\n'
                f'{i7}</ReachPositionCondition>\n'
                f'{i6}</EntityCondition>\n'
                f'{i5}</ByEntityCondition>')
    else:
        kind, ref, state = dict(
            action_complete=("action", info["action"], "completeState"),
            event_complete=("event", info["event"], "completeState"),
            action_end=("action", info["action"], "endTransition"),
        )[trigger]
        cond = (f'{i5}<ByValueCondition>\n'
                f'{i6}<StoryboardElementStateCondition storyboardElementType="{kind}"'
                f' storyboardElementRef="{ref}" state="{state}" />\n'
                f'{i5}</ByValueCondition>')
    return (
        f'{i}<Event name="{STOP_EVENT}" priority="parallel" maximumExecutionCount="1">\n'
        f'{i2}<Action name="{STOP_ACTION}">\n'
        f'{i3}<PrivateAction>\n'
        f'{i4}<LongitudinalAction>\n'
        f'{i5}<SpeedAction>\n'
        f'{i5}    <SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time" />\n'
        f'{i5}    <SpeedActionTarget>\n'
        f'{i5}        <AbsoluteTargetSpeed value="0" />\n'
        f'{i5}    </SpeedActionTarget>\n'
        f'{i5}</SpeedAction>\n'
        f'{i4}</LongitudinalAction>\n'
        f'{i3}</PrivateAction>\n'
        f'{i2}</Action>\n'
        f'{i2}<StartTrigger>\n'
        f'{i3}<ConditionGroup>\n'
        f'{i4}<Condition name="{STOP_CONDITION}" delay="0" conditionEdge="none">\n'
        f'{cond}\n'
        f'{i4}</Condition>\n'
        f'{i3}</ConditionGroup>\n'
        f'{i2}</StartTrigger>\n'
        f'{i}</Event>\n')


def insert_stop_event(src_path: Path, dst_path: Path, info: dict, trigger: str) -> dict:
    """Textual insertion just before the Agent1 maneuver's </Maneuver>."""
    text = src_path.read_text()
    open_tag = f'<Maneuver name="{info["maneuver"]}">'
    starts = [m.start() for m in re.finditer(re.escape(open_tag), text)]
    assert len(starts) == 1, f"{src_path}: {len(starts)} maneuvers named {info['maneuver']}"
    close = text.find("</Maneuver>", starts[0])
    assert close > 0, f"{src_path}: no </Maneuver> after {info['maneuver']}"
    line_start = text.rfind("\n", 0, close) + 1
    close_indent = text[line_start:close]
    assert close_indent.strip() == "", "unexpected content before </Maneuver>"
    block = stop_event_xml(info, trigger, close_indent + "    ")
    out = text[:line_start] + block + text[line_start:]
    dst_path.write_text(out)
    diff = list(difflib.unified_diff(text.splitlines(True), out.splitlines(True),
                                     "source_run.xosc", "run.xosc", n=0))
    removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    added = [d for d in diff if d.startswith("+") and not d.startswith("+++")]
    return dict(trigger=trigger, inserted_lines=len(added), removed_lines=len(removed),
                diff_is_insertion_only=not removed, diff="".join(diff),
                source_xosc=str(src_path), source_xosc_sha256=G.sha256(src_path),
                bytes_before=len(text), bytes_after=len(out))


def verify_insertion(src_path: Path, dst_path: Path, info: dict, trigger: str) -> dict:
    """Structural proof: dst minus the added Event is signature-identical to src."""
    src = ET.parse(src_path).getroot()
    dst_tree = ET.parse(dst_path)
    dst = dst_tree.getroot()
    added = [e for e in dst.iter("Event") if e.get("name") == STOP_EVENT]
    assert len(added) == 1, f"{dst_path}: {len(added)} stop events"
    parents = [p for p in dst.iter("Maneuver") if added[0] in list(p)]
    assert len(parents) == 1 and parents[0].get("name") == info["maneuver"]
    ev = added[0]
    speed = ev.findall(".//SpeedAction")
    assert len(speed) == 1
    assert speed[0].find("SpeedActionDynamics").get("dynamicsShape") == "step"
    assert float(speed[0].find(".//AbsoluteTargetSpeed").get("value")) == 0.0
    conds = ev.findall(".//Condition")
    assert len(conds) == 1
    if trigger == "reach_position":
        assert ev.find(".//ReachPositionCondition") is not None
        trigger_desc = f"ReachPositionCondition tolerance={REACH_TOLERANCE_M} at the last {info['shape']} point"
    else:
        sec = ev.find(".//StoryboardElementStateCondition")
        assert sec is not None
        trigger_desc = (f"StoryboardElementStateCondition type={sec.get('storyboardElementType')} "
                        f"ref={sec.get('storyboardElementRef')} state={sec.get('state')}")
    parents[0].remove(ev)
    same = G.signature(src) == G.signature(dst)
    assert same, f"{dst_path}: the rest of the scenario changed"
    return dict(stop_event_added=True, rest_of_scenario_identical=True, trigger_description=trigger_desc,
                parameter_declarations_unchanged=G.params(src_path) == G.params(dst_path))


# ── esmini execution ─────────────────────────────────────────────────────────

def esmini_command(local_bin: Path, executable: Path, csv_path: Path, wd: Path, fps: float) -> list[str]:
    return ["nice", "-n", "10", str(local_bin), "--osc", str(executable), "--headless",
            "--csv_logger", str(csv_path), "--fixed_timestep", str(1.0 / fps),
            "--logfile_path", str(wd / "esmini.log"), "--disable_stdout"]


LOGLINE = re.compile(r"^\[(\d+\.\d+)\]\s+\[info\]\s+(.*)$")


def log_events(log_path: Path) -> dict:
    """Sim times of the storyboard transitions the proof needs."""
    out = dict(trajectory_action_complete_s=None, trajectory_event_complete_s=None,
               stop_action_start_s=None, stop_event_complete_s=None, ego_replay_end_s=None,
               storyboard_stop_s=None)
    if not log_path.exists():
        return out
    for line in log_path.read_text(errors="replace").splitlines():
        m = LOGLINE.match(line.strip())
        if not m:
            continue
        t, msg = float(m.group(1)), m.group(2)
        if "TrajectoryAction" in msg and "completeState" in msg and out["trajectory_action_complete_s"] is None:
            out["trajectory_action_complete_s"] = t
        elif "TrajectoryEvent" in msg and "completeState" in msg and "complete after" not in msg \
                and out["trajectory_event_complete_s"] is None:
            out["trajectory_event_complete_s"] = t
        elif STOP_ACTION in msg and "runningState" in msg and "startTransition" in msg \
                and out["stop_action_start_s"] is None:
            out["stop_action_start_s"] = t
        elif STOP_EVENT in msg and "completeState" in msg and "complete after" not in msg \
                and out["stop_event_complete_s"] is None:
            out["stop_event_complete_s"] = t
        elif "EgoReplay_Follow" in msg and "completeState" in msg and out["ego_replay_end_s"] is None:
            out["ego_replay_end_s"] = t
        elif "storyBoard" in msg and "stopTransition" in msg:
            out["storyboard_stop_s"] = t
    return out


def proof_from_traces(stop_tidy: pd.DataFrame, src_tidy: pd.DataFrame, events: dict) -> dict:
    """The WP2 proof, entirely from the logged traces (run.csv -> tidy)."""
    st = stop_tidy[stop_tidy.role.eq("target")].sort_values("time_s").reset_index(drop=True)
    ss = src_tidy[src_tidy.role.eq("target")].sort_values("time_s").reset_index(drop=True)
    et = stop_tidy[stop_tidy.role.eq("ego")].sort_values("time_s").reset_index(drop=True)
    es = src_tidy[src_tidy.role.eq("ego")].sort_values("time_s").reset_index(drop=True)
    r = dict(n_rows_stop=int(len(st)), n_rows_source=int(len(ss)),
             stop_end_s=float(st.time_s.max()), source_end_s=float(ss.time_s.max()))
    # ego identity (same time grid in both runs).  Scope note: after the ego replay
    # polyline runs out (log: EgoReplay_Follow -> completeState) esmini's default
    # controller drives the ego for the remaining ~1 s of storyboard margin, and that
    # free-running tail is not reproducible across ANY perturbation of the .xosc file
    # (a whitespace-only edit of the same scenario moves it by up to ~1 cm / 0.03 rad).
    # Those rows are outside the metadata window and outside every scorer, so the gate
    # is applied to the replay-driven part and to the scored window; the tail is reported.
    n = min(len(et), len(es))
    r["ego_rows_compared"] = int(n)
    r["ego_time_max_abs_diff_s"] = float(np.max(np.abs(et.time_s.values[:n] - es.time_s.values[:n]))) if n else np.nan
    dxy = np.hypot(et.x.values[:n] - es.x.values[:n], et.y.values[:n] - es.y.values[:n]) if n else np.array([])
    dsp = np.abs(et.speed_mps.values[:n] - es.speed_mps.values[:n]) if n else np.array([])
    dhd = np.abs(et.heading_deg.values[:n] - es.heading_deg.values[:n]) if n else np.array([])
    r["ego_max_abs_dxy_m"] = float(dxy.max()) if n else np.nan
    r["ego_max_abs_dspeed_mps"] = float(dsp.max()) if n else np.nan
    r["ego_max_abs_dheading_deg"] = float(dhd.max()) if n else np.nan
    t_ego = events.get("ego_replay_end_s")
    pre = (et.time_s.values[:n] <= (t_ego if t_ego is not None else np.inf) + 1e-9)
    win = et.within_metadata_window.values[:n].astype(bool)
    r["ego_replay_end_s"] = t_ego
    r["ego_rows_replay_driven"] = int(pre.sum())
    r["ego_max_abs_dxy_m_replay"] = float(dxy[pre].max()) if pre.any() else np.nan
    r["ego_max_abs_dheading_deg_replay"] = float(dhd[pre].max()) if pre.any() else np.nan
    r["ego_max_abs_dxy_m_after_replay"] = float(dxy[~pre].max()) if (~pre).any() else 0.0
    r["ego_max_abs_dxy_m_in_window"] = float(dxy[win].max()) if win.any() else np.nan
    r["ego_max_abs_dheading_deg_in_window"] = float(dhd[win].max()) if win.any() else np.nan
    r["ego_rows_differing"] = int((dxy > EGO_IDENTITY_MAX).sum())
    r["ego_identical"] = bool(n == len(es) == len(et)
                              and (r["ego_max_abs_dxy_m_replay"] or 0.0) <= EGO_IDENTITY_MAX
                              and (r["ego_max_abs_dheading_deg_replay"] or 0.0) <= EGO_IDENTITY_MAX
                              and (r["ego_max_abs_dxy_m_in_window"] or 0.0) <= EGO_IDENTITY_MAX
                              and float(dsp[pre].max() if pre.any() else 0.0) <= EGO_IDENTITY_MAX)
    r["ego_identical_all_rows"] = bool(n == len(es) == len(et) and r["ego_rows_differing"] == 0)
    t_end = events.get("trajectory_action_complete_s")
    r["trajectory_end_s"] = t_end
    if t_end is None:
        r.update(stop_triggered=False, proof_status="no_trajectory_end_in_log",
                 target_prefix_max_dxy_m=np.nan, speed_zero_s=np.nan, speed_zero_delay_s=np.nan,
                 post_stop_drift_m=np.nan, post_stop_rows=0, post_end_path_m=np.nan,
                 source_post_end_path_m=np.nan)
        return r
    # (a) nothing before the trajectory end changed
    pre_t = st[st.time_s <= t_end + 1e-9]
    pre_s = ss[ss.time_s <= t_end + 1e-9]
    m = min(len(pre_t), len(pre_s))
    r["target_prefix_rows"] = int(m)
    r["target_prefix_max_dxy_m"] = float(np.max(np.hypot(pre_t.x.values[:m] - pre_s.x.values[:m],
                                                         pre_t.y.values[:m] - pre_s.y.values[:m]))) if m else np.nan
    # (b) speed reaches 0 within ZERO_WITHIN_S of the trajectory end
    after = st[st.time_s >= t_end - 1e-9].reset_index(drop=True)
    zero = after[after.speed_mps.abs() <= ZERO_SPEED_MPS]
    if len(zero):
        t0 = float(zero.time_s.iloc[0])
        r["speed_zero_s"] = t0
        r["speed_zero_delay_s"] = t0 - t_end
        tail = st[st.time_s >= t0 - 1e-9]
        x0, y0 = float(tail.x.iloc[0]), float(tail.y.iloc[0])
        r["post_stop_rows"] = int(len(tail) - 1)
        r["post_stop_drift_m"] = float(np.max(np.hypot(tail.x.values - x0, tail.y.values - y0)))
        r["post_stop_max_speed_mps"] = float(np.max(np.abs(tail.speed_mps.values)))
    else:
        r.update(speed_zero_s=np.nan, speed_zero_delay_s=np.nan, post_stop_rows=0,
                 post_stop_drift_m=np.nan, post_stop_max_speed_mps=float(np.max(np.abs(after.speed_mps.values))))
    # (c) how much post-trajectory path the stop removed
    def path_after(df):
        a = df[df.time_s >= t_end - 1e-9]
        return float(np.hypot(np.diff(a.x.values), np.diff(a.y.values)).sum()) if len(a) > 1 else 0.0
    r["post_end_path_m"] = path_after(st)
    r["source_post_end_path_m"] = path_after(ss)
    r["post_end_rows"] = int((st.time_s >= t_end - 1e-9).sum())
    r["stop_triggered"] = bool(events.get("stop_action_start_s") is not None)
    if r["post_end_rows"] <= 1:
        r["proof_status"] = "trajectory_ends_at_or_after_sim_stop"
    elif not r["stop_triggered"]:
        r["proof_status"] = "stop_action_never_started"
    elif not np.isfinite(r.get("speed_zero_delay_s", np.nan)):
        r["proof_status"] = "speed_never_zero"
    elif r["speed_zero_delay_s"] > ZERO_WITHIN_S + 1e-9:
        r["proof_status"] = "speed_zero_too_late"
    elif r["post_stop_drift_m"] > DRIFT_MAX_M:
        r["proof_status"] = "drift_after_stop"
    elif not r["ego_identical"]:
        r["proof_status"] = "ego_changed"
    elif np.isfinite(r["target_prefix_max_dxy_m"]) and r["target_prefix_max_dxy_m"] > EGO_IDENTITY_MAX:
        r["proof_status"] = "target_prefix_changed"
    else:
        r["proof_status"] = "ok"
    return r


def run_one(sj: Path, arm: str, batch: Path, local_bin: Path, run_id: str, trigger: str,
            raw_all: pd.DataFrame, keep_diff: bool) -> dict:
    """Re-execute one source sample with the stop event; never deletes anything."""
    spec = ARMS[arm]
    src = json.loads(sj.read_text())
    sid = src["sample_id"]
    wd = batch / sid
    wd.mkdir(parents=True, exist_ok=True)
    report_path = wd / "sample.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old.get("status") == "completed" and all((wd / n).exists() for n in ("run.xosc", "run.csv", "trajectory.parquet")):
            return old
    start = time.perf_counter()
    job = src.get("job", {})
    context = dict(src.get("context", {}))
    report = dict(sample_id=sid, sample_dir=str(wd), job=job, run_id=run_id, method=arm,
                  arm=arm, arm_label=spec["label"], execution_label=src.get("execution_label"),
                  status="preparing", all_xosc_retained=True, scoring_performed=False,
                  parameters_requested=src.get("parameters_requested", {}),
                  parameters_applied=src.get("parameters_applied", {}),
                  is_nominal_default=src.get("is_nominal_default", False),
                  source_sample_json=str(sj), source_sample_json_sha256=G.sha256(sj),
                  source_run_xosc=str(sj.parent / "run.xosc"),
                  source_trajectory_path=str(sj.parent / "trajectory.parquet"),
                  source_arm=spec["source_arm"], window_arm=spec["window_arm"])
    executable, csv_path = wd / "run.xosc", wd / "run.csv"
    sim_start = time.perf_counter()
    try:
        if src.get("status") != "completed":
            raise RuntimeError(f"source status={src.get('status')}")
        src_xosc = sj.parent / "run.xosc"
        info = locate(src_xosc)
        surgery = insert_stop_event(src_xosc, executable, info, trigger)
        assert surgery["diff_is_insertion_only"], "the xosc diff is not insertion-only"
        checks = verify_insertion(src_xosc, executable, info, trigger)
        if not keep_diff:
            surgery.pop("diff")
        context.update(stop_at_end=dict(
            event=STOP_EVENT, action=STOP_ACTION, trigger=trigger,
            trigger_description=checks["trigger_description"],
            pattern="SAKURA-route Agent1_StopAtGoalEvent (SpeedAction step -> AbsoluteTargetSpeed 0)",
            source_run_xosc=str(src_xosc), source_run_xosc_sha256=surgery["source_xosc_sha256"],
            **{f"traj_{k}": info[k] for k in ("maneuver", "event", "action", "shape", "n_geometry")}),
            no_stop_at_goal_added=False,
            stop_at_end_added=True)
        report.update(context=context, xosc_surgery=surgery, checks=checks,
                      prepare_s=time.perf_counter() - start)
        lo, hi = context["metadata_window_frames"]
        fps = float(context.get("fps", FPS))
        raw = raw_all[raw_all.trackId.isin([context["ego"], context["target"]])
                      & raw_all.frame.between(lo, hi)].sort_values(["trackId", "frame"])
        cmd = esmini_command(local_bin, executable, csv_path, wd, fps)
        report.update(command=cmd, subprocess_cwd=str(wd), status="executing")
        G.atomic_json(report_path, json_safe(report))
        sim_start = time.perf_counter()
        with (wd / "stdout.txt").open("wb") as so, (wd / "stderr.txt").open("wb") as se:
            res = subprocess.run(cmd, cwd=wd, stdout=so, stderr=se, timeout=SUBPROCESS_TIMEOUT_S)
        report.update(simulate_s=time.perf_counter() - sim_start, returncode=res.returncode)
        if res.returncode != 0:
            raise RuntimeError(f"esmini returned {res.returncode}")
        parse_start = time.perf_counter()
        tidy = G.parse_tidy(csv_path, raw, sid, context)
        if spec["kind"] == "svd":
            t_last = float(src["decoded_polyline"]["t"][-1])
            tidy["within_decoded_polyline_support"] = tidy.time_s <= t_last + 0.5 / fps
            report["decoded_polyline_end_s"] = t_last
        tidy.to_parquet(wd / "trajectory.parquet", index=False)
        tidy.to_csv(wd / "trajectory.csv", index=False, float_format="%.12g")
        events = log_events(wd / "esmini.log")
        src_tidy = pd.read_parquet(sj.parent / "trajectory.parquet")
        report["storyboard_events"] = events
        report["stop_proof"] = proof_from_traces(tidy, src_tidy, events)
        report.update(status="completed", parse_export_s=time.perf_counter() - parse_start,
                      trajectory_rows=len(tidy),
                      observed_time_range_s=[float(tidy.time_s.min()), float(tidy.time_s.max())],
                      trajectory_path=str(wd / "trajectory.parquet"))
    except subprocess.TimeoutExpired:
        report.update(status="timeout", timeout_s=SUBPROCESS_TIMEOUT_S, simulate_s=time.perf_counter() - sim_start)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded, never dropped
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["total_s"] = time.perf_counter() - start
    report["artifacts"] = [{"path": str(p), "sha256": G.sha256(p), "size_bytes": p.stat().st_size}
                           for p in (executable, csv_path, wd / "trajectory.parquet", wd / "trajectory.csv") if p.exists()]
    G.atomic_json(report_path, json_safe(report))
    return report


def prepare_batch(arm: str, batch_name: str, jobs: list[str], trigger: str, extra: dict) -> tuple[Path, str, Path]:
    out = PROJECT / "runs" / batch_name
    out.mkdir(parents=True, exist_ok=True)
    sources = {RUNNER_20, RUNNER_26, BINARY, Path(__file__).resolve(), RAW}
    provenance = [{"path": str(p), "sha256": G.sha256(p), "size_bytes": p.stat().st_size} for p in sorted(sources)]
    protocol = dict(method="stop_at_end", arm=arm, arm_label=ARMS[arm]["label"], batch_name=batch_name,
                    source_batch=str(source_batch_dir(arm)), source_arm=ARMS[arm]["source_arm"],
                    window_arm=ARMS[arm]["window_arm"], trigger=trigger, n_jobs=len(jobs), jobs=jobs,
                    stop_event=STOP_EVENT, stop_action=STOP_ACTION,
                    pattern="SAKURA-route Agent1_StopAtGoalEvent (SpeedAction step -> AbsoluteTargetSpeed 0)",
                    sources=provenance, subprocess_timeout_s=SUBPROCESS_TIMEOUT_S, worker_count=1, nice=10,
                    min_free_disk_bytes=MIN_FREE, esmini_flags=["--headless", "--csv_logger",
                    "--fixed_timestep 1/fps", "--logfile_path", "--disable_stdout"],
                    all_xosc_retained=True, scoring_performed=False, **extra)
    run_id = hashlib.sha256(json.dumps(json_safe(protocol), sort_keys=True).encode()).hexdigest()[:16]
    batch = out / run_id
    batch.mkdir(exist_ok=True)
    local_bin = PROJECT / "bin/esmini"
    local_bin.parent.mkdir(exist_ok=True)
    if not local_bin.exists():
        local_bin.symlink_to(BINARY)
    assert local_bin.resolve() == BINARY.resolve()
    if (local_bin.parent / "config.yml").exists():
        raise RuntimeError("Unexpected local esmini config; will not silently override behavior")
    snap = batch / "runner_snapshot.py"
    if not snap.exists():
        shutil.copyfile(__file__, snap)
    G.atomic_json(batch / "protocol.json", json_safe(protocol))
    return batch, run_id, local_bin


# ── stages ───────────────────────────────────────────────────────────────────

def source_tail_m(sj: Path) -> float:
    """Path (m) the source target drove AFTER its trajectory ended - what the stop removes."""
    ev = log_events(sj.parent / "esmini.log")
    t_end = ev.get("trajectory_action_complete_s")
    if t_end is None:
        return 0.0
    d = pd.read_parquet(sj.parent / "trajectory.parquet", columns=["role", "time_s", "x", "y"])
    a = d[d.role.eq("target") & (d.time_s >= t_end - 1e-9)].sort_values("time_s")
    return float(np.hypot(np.diff(a.x.values), np.diff(a.y.values)).sum()) if len(a) > 1 else 0.0


def pilot_selection(arm: str, sjs: list[Path], n: int, scan: int) -> list[Path]:
    """The n samples with the largest free-running tail among an evenly spread scan.

    The stop-at-end fix only bites where the source keeps driving after the
    trajectory ends, so the pilot must prove itself on such samples (samples whose
    trajectory outlives the storyboard StopTrigger are a documented no-op case and
    are reported separately by the render stage)."""
    done = [sj for sj in sjs if json.loads(sj.read_text()).get("status") == "completed"]
    step = max(1, len(done) // max(1, scan))
    cand = done[::step][:scan]
    tails = [(sj, source_tail_m(sj)) for sj in cand]
    tails.sort(key=lambda t: (-t[1], t[0].parent.name))
    return [t[0] for t in tails[:n]]


def stage_pilot(args):
    rows, chosen = [], {}
    raw_all = pd.read_parquet(RAW, columns=["trackId", "frame", "xCenter", "yCenter", "heading",
                                            "xVelocity", "yVelocity", "length", "width"])
    for arm in args.arm:
        sjs = source_samples(arm)
        log(f"pilot {arm}: {len(sjs)} source samples available")
        take = pilot_selection(arm, sjs, args.n, args.scan)
        log(f"pilot {arm}: samples {[p.parent.name for p in take]}")
        for trigger in TRIGGERS:
            batch, run_id, local_bin = prepare_batch(arm, f"{arm}_pilot", [p.parent.name for p in take],
                                                     trigger, dict(stage="pilot", trigger_candidates=TRIGGERS))
            ok = 0
            for sj in take:
                rep = run_one(sj, arm, batch, local_bin, run_id, trigger, raw_all, keep_diff=True)
                pf = rep.get("stop_proof", {})
                row = dict(arm=arm, trigger=trigger, sample_id=rep["sample_id"], status=rep["status"],
                           batch_dir=str(batch), error=rep.get("error"),
                           inserted_lines=(rep.get("xosc_surgery") or {}).get("inserted_lines"),
                           removed_lines=(rep.get("xosc_surgery") or {}).get("removed_lines"),
                           rest_identical=(rep.get("checks") or {}).get("rest_of_scenario_identical"),
                           trigger_description=(rep.get("checks") or {}).get("trigger_description"),
                           **{f"log_{k}": v for k, v in (rep.get("storyboard_events") or {}).items()},
                           **{k: pf.get(k) for k in ("proof_status", "trajectory_end_s", "stop_triggered",
                                                     "speed_zero_s", "speed_zero_delay_s", "post_stop_drift_m",
                                                     "post_stop_rows", "post_end_path_m", "source_post_end_path_m",
                                                     "target_prefix_max_dxy_m", "ego_identical", "ego_identical_all_rows",
                                                     "ego_max_abs_dxy_m", "ego_max_abs_dxy_m_replay",
                                                     "ego_max_abs_dxy_m_in_window", "ego_max_abs_dxy_m_after_replay",
                                                     "ego_rows_differing", "ego_replay_end_s",
                                                     "ego_max_abs_dspeed_mps", "ego_max_abs_dheading_deg",
                                                     "n_rows_stop", "n_rows_source")})
                rows.append(row)
                ok += int(rep["status"] == "completed" and pf.get("proof_status") == "ok")
                if rep["status"] == "completed" and rep.get("xosc_surgery", {}).get("diff"):
                    (PLANS / f"pilot_diff_{arm}_{trigger}_{rep['sample_id']}.txt").write_text(
                        rep["xosc_surgery"]["diff"])
            log(f"pilot {arm} / {trigger}: {ok}/{len(take)} proved")
            if ok == len(take):
                chosen.setdefault(arm, trigger)
                if not args.all_triggers:
                    break
        if arm not in chosen:
            log(f"pilot {arm}: NO trigger proved on all {len(take)} samples")
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    PLANS.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "stop_pilot_samples.csv", index=False)
    (PLANS / "stop_trigger_choice.json").write_text(json.dumps(dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), triggers_tried=TRIGGERS, chosen=chosen,
        gates=dict(zero_speed_mps=ZERO_SPEED_MPS, zero_within_s=ZERO_WITHIN_S, drift_max_m=DRIFT_MAX_M,
                   ego_identity_max=EGO_IDENTITY_MAX),
        n_per_arm=args.n), indent=2) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table[["arm", "trigger", "sample_id", "status", "proof_status", "trajectory_end_s",
                     "speed_zero_delay_s", "post_stop_drift_m", "source_post_end_path_m",
                     "post_end_path_m", "ego_identical"]].to_string(index=False))
    return 0 if len(chosen) == len(args.arm) else 1


def stage_render(args):
    choice = json.loads((PLANS / "stop_trigger_choice.json").read_text())["chosen"]
    raw_all = pd.read_parquet(RAW, columns=["trackId", "frame", "xCenter", "yCenter", "heading",
                                            "xVelocity", "yVelocity", "length", "width"])
    for arm in args.arm:
        trigger = args.trigger or choice.get(arm)
        assert trigger, f"{arm}: no proved trigger; run the pilot first"
        sjs = source_samples(arm)
        if args.limit:
            sjs = sjs[:args.limit]
        expect = ARMS[arm]["expect"]
        if not args.limit and len(sjs) != expect:
            log(f"WARN {arm}: {len(sjs)} source samples, expected {expect}")
        batch, run_id, local_bin = prepare_batch(arm, arm, [p.parent.name for p in sjs], trigger,
                                                 dict(stage="render", expected_samples=expect))
        log(f"render {arm}: {len(sjs)} jobs, trigger={trigger}, batch={batch}")
        status = dict(run_id=run_id, batch_dir=str(batch), arm=arm, arm_label=ARMS[arm]["label"],
                      trigger=trigger, status="running", samples=[])
        t0 = time.perf_counter()
        rows = []
        for n, sj in enumerate(sjs):
            if shutil.disk_usage(batch).free < MIN_FREE:
                status.update(status="paused_low_disk", remaining_sample=sj.parent.name)
                G.atomic_json(batch / "batch_status.json", json_safe(status))
                log("PAUSED: less than 10 GiB free; nothing deleted")
                return 75
            rep = run_one(sj, arm, batch, local_bin, run_id, trigger, raw_all, keep_diff=False)
            pf = rep.get("stop_proof", {})
            status["samples"].append({k: rep.get(k) for k in ("sample_id", "status", "error")})
            rows.append(dict(sample_id=rep["sample_id"], status=rep["status"], sample_dir=rep["sample_dir"],
                             trajectory_path=rep.get("trajectory_path"),
                             scenario_id=rep.get("context", {}).get("scenario_id"),
                             subset=rep.get("context", {}).get("subset"),
                             scenario_uid=(rep.get("context", {}).get("source_context") or {}).get("scenario_uid"),
                             trigger=trigger, source_sample_json=rep.get("source_sample_json"),
                             inserted_lines=(rep.get("xosc_surgery") or {}).get("inserted_lines"),
                             removed_lines=(rep.get("xosc_surgery") or {}).get("removed_lines"),
                             **{k: pf.get(k) for k in ("proof_status", "trajectory_end_s", "stop_triggered",
                                                       "speed_zero_delay_s", "post_stop_drift_m",
                                                       "post_end_path_m", "source_post_end_path_m",
                                                       "target_prefix_max_dxy_m", "ego_identical",
                                                       "ego_identical_all_rows", "ego_max_abs_dxy_m",
                                                       "ego_max_abs_dxy_m_replay", "ego_max_abs_dxy_m_in_window",
                                                       "ego_max_abs_dxy_m_after_replay", "ego_rows_differing",
                                                       "n_rows_stop", "n_rows_source")},
                             simulate_s=rep.get("simulate_s"), total_s=rep.get("total_s"), error=rep.get("error")))
            if (n + 1) % 250 == 0 or n + 1 == len(sjs):
                G.atomic_json(batch / "batch_status.json", json_safe(status))
                log(f"render {arm}: {n + 1}/{len(sjs)} ({time.perf_counter() - t0:.0f}s)")
        manifest = pd.DataFrame(rows)
        manifest.to_csv(batch / "manifest.csv", index=False)
        n_ok = int(manifest.status.eq("completed").sum())
        status.update(status="completed" if n_ok == len(manifest) else "completed_with_failures",
                      elapsed_s=time.perf_counter() - t0, n_completed=n_ok, n_jobs=len(manifest),
                      proof_status_counts=manifest.proof_status.value_counts(dropna=False).astype(int).to_dict())
        G.atomic_json(batch / "batch_status.json", json_safe(status))
        G.atomic_json(PROJECT / "runs" / arm / "latest_batch.json",
                      dict(batch_dir=str(batch), status=status["status"], run_id=run_id, arm=arm,
                           trigger=trigger, manifest=str(batch / "manifest.csv")))
        log(f"render {arm}: {n_ok}/{len(manifest)} completed in {status['elapsed_s']:.0f}s; "
            f"proof {status['proof_status_counts']}")
    return 0


def stage_status(args):
    rows = []
    for arm in args.arm:
        lb = PROJECT / "runs" / arm / "latest_batch.json"
        if not lb.exists():
            rows.append(dict(arm=arm, batch=None, n=0, note="not rendered"))
            continue
        b = Path(json.loads(lb.read_text())["batch_dir"])
        m = pd.read_csv(b / "manifest.csv")
        rows.append(dict(arm=arm, batch=str(b), n=len(m), completed=int(m.status.eq("completed").sum()),
                         expect=ARMS[arm]["expect"],
                         proof=json.dumps(m.proof_status.value_counts(dropna=False).astype(int).to_dict())))
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("pilot", "render", "status"))
    ap.add_argument("--arm", action="append", choices=list(ARMS), default=[])
    ap.add_argument("--n", type=int, default=3, help="pilot samples per arm")
    ap.add_argument("--scan", type=int, default=150, help="pilot: source samples scanned for the tail ranking")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trigger", choices=TRIGGERS, default=None, help="override the pilot's choice")
    ap.add_argument("--all-triggers", action="store_true",
                    help="pilot: run every candidate on every arm (full evidence table) instead of stopping "
                         "at the first one that proves")
    args = ap.parse_args()
    if not args.arm:
        args.arm = list(ARMS)
    OUT.mkdir(parents=True, exist_ok=True)
    PLANS.mkdir(parents=True, exist_ok=True)
    return dict(pilot=stage_pilot, render=stage_render, status=stage_status)[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
