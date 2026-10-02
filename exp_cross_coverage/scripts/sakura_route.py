"""SAKURA-route: road-following when the GlobalRoutePlanner finds a route,
NURBS chord only as the fallback.

Plain SAKURA (`SampleConfig("none", n_points=0)`) always emits a start->end
NURBS, which chord-cuts every turn.  This variant asks the CARLA
GlobalRoutePlanner — the same planner Stage A already uses inside
convert2yaml.get_route_start_end_info, with the same candidate-waypoint search
and the same best-offset selection rule — whether a route exists between the
agent's first and last recorded position:

  route found  -> the rendered path IS that route: the NURBS FollowTrajectory
                  action is replaced by an AssignRouteAction over the planned
                  waypoints, plus a stop-at-goal event so the agent does not
                  drive on past the recorded endpoint.  No control points.
  no route     -> the generated xosc is left exactly as SAKURA produced it.

Why the route waypoints and not a bare AcquirePositionAction on the endpoints:
esmini's own router cannot connect these junction roads on tyms.xodr
("Route::AddWaypoint Skip waypoint for scenario routes since path not found"),
so the action completes at t=0 and the agent drives straight on.  Handing it
the planner's waypoints (2 m resolution) makes every hop trivially connectable.

Everything happens as surgery on the generated .xosc — no existing repo is
modified, and the plain SAKURA artifacts stay byte-identical.
"""
from __future__ import annotations

import re
from pathlib import Path

_GRP_CACHE: dict = {}


def _planner(carla_map):
    key = id(carla_map)
    if key not in _GRP_CACHE:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        _GRP_CACHE[key] = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    return _GRP_CACHE[key]


def plan_route(carla_map, sx, sy, ex, ey):
    """Planned route between two dataset-frame positions, or None.

    Mirrors convert2yaml.get_route_start_end_info: candidate waypoints within a
    2 m radius at both ends, every (start, end) pair traced, the pair with the
    smallest sum of squared lane offsets wins.  Returns the deduplicated
    [(road_id, lane_id, s), ...] of that route.
    """
    import carla
    import convert2yaml as CV

    sl = carla.Location(float(sx), -float(sy), 0)
    el = carla.Location(float(ex), -float(ey), 0)
    try:
        sw = CV.find_all_overlapping_waypoints(carla_map, sl, radius=2, sample_num_points=1000)
        ew = CV.find_all_overlapping_waypoints(carla_map, el, radius=2, sample_num_points=1000)
    except Exception:  # noqa: BLE001
        return None
    grp = _planner(carla_map)
    best, best_d = None, float("inf")
    for a in sw:
        for b in ew:
            try:
                r = grp.trace_route(a, b)
            except Exception:  # noqa: BLE001
                continue
            if not r:
                continue
            d = CV.compute_lane_offset(sl, a) ** 2 + CV.compute_lane_offset(el, b) ** 2
            if d < best_d:
                best_d, best = d, r
    if not best:
        return None
    out = []
    for wp, _opt in best:
        loc = wp.transform.location
        w = (int(wp.road_id), int(wp.lane_id), round(float(wp.s), 2),
             round(float(loc.x), 3), round(-float(loc.y), 3))
        if not out or w[:3] != out[-1][:3]:
            out.append(w)
    return out if len(out) >= 2 else None


def route_length(waypoints) -> float:
    """Planned-route arc length [m] from the waypoints' dataset-frame xy."""
    import math
    return sum(math.dist(waypoints[i][3:5], waypoints[i + 1][3:5])
               for i in range(len(waypoints) - 1))


# ── .xosc surgery ────────────────────────────────────────────────────────────

_ACTION_RE = r'<Action name="{actor}_Event\d+_TrajectoryAction">(.*?)</Action>'


def _route_xml(waypoints, offset_expr: str, name: str) -> str:
    parts = [f'<RoutingAction><AssignRouteAction><Route name="{name}" closed="false">']
    for i, (r, l, s) in enumerate(waypoints):
        off = "0" if i == 0 else offset_expr        # spawn keeps its own offset param
        parts.append(f'<Waypoint routeStrategy="shortest"><Position>'
                     f'<LanePosition roadId="{r}" laneId="{l}" s="{s}" offset="{off}"/>'
                     f'</Position></Waypoint>')
    parts.append("</Route></AssignRouteAction></RoutingAction>")
    return "".join(parts)


def _stop_event_xml(actor: str, goal, tolerance: float = 3.0) -> str:
    r, l, s = goal
    return (
        f'<Event name="{actor}_StopAtGoalEvent" priority="parallel" maximumExecutionCount="1">'
        f'<Action name="{actor}_StopAtGoalAction"><PrivateAction><LongitudinalAction><SpeedAction>'
        f'<SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time"/>'
        f'<SpeedActionTarget><AbsoluteTargetSpeed value="0"/></SpeedActionTarget>'
        f'</SpeedAction></LongitudinalAction></PrivateAction></Action>'
        f'<StartTrigger><ConditionGroup><Condition name="goal_reached" delay="0" conditionEdge="none">'
        f'<ByEntityCondition><TriggeringEntities triggeringEntitiesRule="any">'
        f'<EntityRef entityRef="{actor}"/></TriggeringEntities>'
        f'<EntityCondition><ReachPositionCondition tolerance="{tolerance}"><Position>'
        f'<LanePosition roadId="{r}" laneId="{l}" s="{s}" offset="0"/>'
        f'</Position></ReachPositionCondition></EntityCondition></ByEntityCondition>'
        f'</Condition></ConditionGroup></StartTrigger></Event>'
    )


def apply_route(xosc_path: Path, waypoints, actor: str = "Agent1",
                stop_at_goal: bool = True) -> bool:
    """Rewrite the actor's trajectory action into an AssignRouteAction over
    `waypoints` (+ a stop-at-goal event).  Returns False if nothing matched."""
    p = Path(xosc_path)
    txt = p.read_text()
    m = re.search(_ACTION_RE.format(actor=re.escape(actor)), txt, flags=re.S)
    if not m:
        return False
    inner = m.group(1)
    r_m = re.search(r"<RoutingAction>.*?</RoutingAction>", inner, flags=re.S)
    if not r_m:
        return False
    off_m = re.search(r'offset="(\$\{?[^"]*\}?)"', inner)
    offset_expr = off_m.group(1) if off_m else "0"
    new_inner = inner[:r_m.start()] + _route_xml(waypoints, offset_expr,
                                                 f"{actor}_planned_route") + inner[r_m.end():]
    txt = txt[:m.start(1)] + new_inner + txt[m.end(1):]
    if stop_at_goal:
        anchor = f'<Maneuver name="{actor}_Maneuver">'
        i = txt.find(anchor)
        if i >= 0:
            j = txt.find("</Maneuver>", i)
            if j >= 0:
                txt = txt[:j] + _stop_event_xml(actor, waypoints[-1]) + txt[j:]
    p.write_text(txt)
    return True
