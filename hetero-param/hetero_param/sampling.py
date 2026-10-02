"""
Configurable critical-point / critical-frame selector — the one *new* algorithm.

`sample()` is a drop-in replacement for retrieval-scenarios/convert2yaml.py::
read_agent_trajectory (orig L759). It reproduces every field of the returned `info`
dict verbatim, and changes ONLY how `info["pet_info"]` (the NURBS intermediate
control points, a.k.a. Use_route) is built — driven by the active SampleConfig.

install() monkeypatches convert2yaml.read_agent_trajectory = sample so the existing
Stage-A pipeline is steered without editing the existing repo. See generate.py.
"""
from __future__ import annotations
import contextlib
import numpy as np

from . import config as C


def _err(msg: str) -> None:
    print(f"[sampling] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# anchor / candidate-frame selection (the configurable part)
# ─────────────────────────────────────────────────────────────────────────────
def _select_anchor(cfg: C.SampleConfig, pet_frame, min_speed_frame, min_dis_frame,
                   traj_cross_frame=None):
    """Choose the critical frame the points cluster around, per cfg.anchor.
    Falls back gracefully to whatever anchor is available (legacy = pet or min_speed)."""
    if cfg.anchor == "traj_cross":   # 軌跡交錯點 (spatial path crossing)
        return traj_cross_frame if traj_cross_frame is not None else (min_dis_frame or pet_frame)
    if cfg.anchor == "min_dist":     # temporal closest approach
        return min_dis_frame if min_dis_frame is not None else pet_frame
    if cfg.anchor == "min_speed":
        return min_speed_frame if min_speed_frame is not None else pet_frame
    # "pet" (default / legacy): pet_frame else min_speed_frame
    return pet_frame if pet_frame is not None else min_speed_frame


def _sym_offsets(n):
    """Symmetric offsets around 0 for n points: n=4 -> [-1.5,-0.5,.5,1.5]; n=3 -> [-1,0,1]."""
    return [i - (n - 1) / 2.0 for i in range(int(n))]


def _arc_to_frame(target_arc, frames, cum_dist):
    idx = int(np.searchsorted(cum_dist, target_arc))
    idx = min(max(idx, 0), len(frames) - 1)
    return int(frames[idx])


def _uniform_frames(n, frames, cum_dist):
    if cum_dist is None or len(cum_dist) < 2 or cum_dist[-1] <= 0:
        return list(frames[1:-1])
    total = cum_dist[-1]
    locs = [(i + 1) / (n + 1) * total for i in range(int(n))]   # interior, evenly spaced
    return [_arc_to_frame(t, frames, cum_dist) for t in locs]


def _candidate_frames(cfg, critical_frame, quartile_frames, min_dis_frame, fps,
                      frames=None, cum_dist=None):
    """Build the ordered list of candidate frames per cfg.placement, BEFORE the
    (shared, legacy) bounds/spacing filtering is applied."""
    w = int(round(cfg.window_s * fps))
    q = list(quartile_frames)
    p = cfg.placement
    n = cfg.n_points if isinstance(cfg.n_points, int) else 3

    if p in ("none", "full"):
        return []
    if p == "midpoint":
        return [q[1]] if len(q) >= 2 else (q[:1] if q else [])
    if p == "conflict":
        return [critical_frame] if critical_frame is not None else []
    if p == "interaction":
        if min_dis_frame is not None:
            return [min_dis_frame]
        return [critical_frame] if critical_frame is not None else []
    if p == "uniform":
        # N evenly-spaced points along the WHOLE trajectory (arc-length)
        if frames is not None and cum_dist is not None:
            return _uniform_frames(n, frames, cum_dist)
        return q
    if p == "critdist":
        # N points concentrated around the crossing, spaced by step_m metres (arc-length)
        if critical_frame is None or frames is None or cum_dist is None:
            return _uniform_frames(n, frames, cum_dist) if frames is not None else q
        crit_idx = min(max(int(np.searchsorted(frames, critical_frame)), 0), len(frames) - 1)
        crit_arc = cum_dist[crit_idx]
        return [_arc_to_frame(crit_arc + off * cfg.step_m, frames, cum_dist)
                for off in _sym_offsets(n)]
    if p == "crittime":
        # N points concentrated around the crossing, spaced by step_s seconds
        if critical_frame is None:
            return _uniform_frames(n, frames, cum_dist) if frames is not None else q
        step = max(1, int(round(cfg.step_s * fps)))
        return [int(critical_frame + off * step) for off in _sym_offsets(n)]
    if p == "base_densify":
        # uniform base (covers the whole path -> no chord-cut) + n extra points at the anchor
        base = _uniform_frames(3, frames, cum_dist) if frames is not None else q
        if critical_frame is None:
            return base
        step = max(1, int(round(cfg.step_s * fps)))
        return base + [int(critical_frame + off * step) for off in _sym_offsets(n)]
    if p == "anchor+uniform":
        # ONE point at the anchored critical frame + the arc-length quartiles
        # (25/50/75%) — 4 nominal points; the shared legacy spacing guard below may
        # merge the anchor with a nearby quartile (reported via n_shape).
        base = _uniform_frames(3, frames, cum_dist) if frames is not None else q
        if critical_frame is None:
            return base
        return [int(critical_frame)] + base
    if p == "critwindow":
        if critical_frame is None:
            return q
        return [critical_frame - w, critical_frame, critical_frame + w]
    if p == "default":  # legacy: crit +/- window + quartiles
        if critical_frame is None:
            return q
        return [critical_frame - w, critical_frame, critical_frame + w] + q
    _err(f"unknown placement {p!r}; using legacy 'default'")
    if critical_frame is None:
        return q
    return [critical_frame - w, critical_frame, critical_frame + w] + q


# ─────────────────────────────────────────────────────────────────────────────
# drop-in for read_agent_trajectory
# ─────────────────────────────────────────────────────────────────────────────
def sample(traj_df, agent_id, start_frame, end_frame,
           pet_frame=None, min_dis_frame=None, traj_cross_frame=None, frame_rate=30):
    """Config-driven read_agent_trajectory. Same signature & return contract.

    Everything except the construction of `pet_info` is identical to the original
    (convert2yaml.read_agent_trajectory) so all downstream consumers are unaffected.
    """
    cfg = C.get_active_sample_config()

    # --- identical bookkeeping to the original -------------------------------
    try:
        agent_traj = traj_df.xs(int(agent_id), level="track_id")
    except KeyError as e:
        raise ValueError(f"Agent ID {agent_id} not found in trajectory data: {e}")

    mask = (agent_traj.index > int(start_frame)) & (agent_traj.index < int(end_frame))
    agent_traj = agent_traj.loc[mask]
    if agent_traj.empty:
        raise ValueError(f"Agent ID {agent_id} has no data in the specified frame range.")
    agent_traj = agent_traj.sort_index()

    start_x, start_y = agent_traj.iloc[0][["x", "y"]]
    end_x, end_y = agent_traj.iloc[-1][["x", "y"]]
    mid_x, mid_y = agent_traj.iloc[len(agent_traj) // 2][["x", "y"]]

    start_speed = float(agent_traj.iloc[0]["velocity"]) * 3.6
    max_velocity = float(max(agent_traj["velocity"]))
    end_speed = max_velocity * 3.6

    first_frame = int(agent_traj.index[0])
    last_frame = int(agent_traj.index[-1])
    travel_time = float((last_frame - first_frame) / frame_rate)

    max_speed_frame = int(agent_traj[agent_traj["velocity"] == max_velocity].index[0])

    min_speed_frame = None
    if min_dis_frame is not None and min_dis_frame > int(start_frame):
        m = (agent_traj.index > int(start_frame)) & (agent_traj.index < int(min_dis_frame))
        before = agent_traj.loc[m]
        if not before.empty:
            min_velocity = float(min(before["velocity"]))
            try:
                min_speed_frame = int(before[before["velocity"] == min_velocity].index[0])
            except Exception as e:
                _err(f"min_speed_frame failed for agent {agent_id}: {e}")
                min_speed_frame = None

    # cumulative-distance quartiles (identical to original)
    coords = agent_traj[["x", "y"]].values
    dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]
    quartile_frames = []
    for qloc in (0.25 * total_dist, 0.5 * total_dist, 0.75 * total_dist):
        idx = min(int(np.searchsorted(cum_dist, qloc)), len(agent_traj) - 1)
        quartile_frames.append(agent_traj.index[idx])

    start_x, start_y = float(start_x), float(start_y)
    end_x, end_y = float(end_x), float(end_y)
    mid_x, mid_y = float(mid_x), float(mid_y)

    # --- the CONFIGURABLE part: choose the NURBS intermediate control points ---
    pet_info = {}
    fps = int(frame_rate)
    critical_frame = _select_anchor(cfg, pet_frame, min_speed_frame, min_dis_frame, traj_cross_frame)
    forced = C.get_forced_anchor_frame()
    if forced is not None:   # generate.py injected the anchor frame (overrides pipeline zeroing)
        critical_frame = int(forced)

    candidates = _candidate_frames(cfg, critical_frame, quartile_frames, min_dis_frame, fps,
                                   frames=agent_traj.index.values.astype(int), cum_dist=cum_dist)
    if candidates:
        candidates = sorted(set(int(f) for f in candidates))
        prev_frame = -np.inf
        prev_pos = None
        kept = []
        for frame in candidates:
            # identical legacy bounds/spacing guard (operator precedence preserved):
            # reject if too near the start, OR (too near the end AND not the crit frame),
            # OR too near the previously kept frame.
            if (frame - 0.5 * fps < first_frame
                    or frame + 0.5 * fps > last_frame and frame != critical_frame
                    or frame - 0.5 * fps < prev_frame):
                continue
            try:
                x, y = float(agent_traj.loc[frame, "x"]), float(agent_traj.loc[frame, "y"])
            except KeyError:
                continue
            if prev_pos is not None and np.hypot(x - prev_pos[0], y - prev_pos[1]) < cfg.min_dist_m:
                continue
            kept.append(frame)
            prev_frame = frame
            prev_pos = (x, y)
        for frame in kept:
            try:
                pet_info[frame] = agent_traj.loc[frame, ["x", "y", "orientation"]].to_dict()
            except KeyError as e:
                _err(f"pet_info frame {frame} missing for agent {agent_id}: {e}")

    return {
        "start_x": start_x, "start_y": start_y,
        "end_x": end_x, "end_y": end_y,
        "start_speed": start_speed, "end_speed": end_speed,
        "travel_time": travel_time,
        "mid_x": mid_x, "mid_y": mid_y,
        "start_frame": first_frame, "end_frame": last_frame,
        "max_speed_frame": max_speed_frame,
        "min_speed_frame": min_speed_frame,
        "pet_info": pet_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# monkeypatch install / uninstall
# ─────────────────────────────────────────────────────────────────────────────
_original = None


def install():
    """Replace convert2yaml.read_agent_trajectory with our config-driven sample()."""
    global _original
    import convert2yaml
    if _original is None:
        _original = convert2yaml.read_agent_trajectory
    convert2yaml.read_agent_trajectory = sample
    return _original


def uninstall():
    global _original
    if _original is not None:
        import convert2yaml
        convert2yaml.read_agent_trajectory = _original
        _original = None


@contextlib.contextmanager
def overridden(cfg: C.SampleConfig | None = None):
    """Context manager: temporarily install the override (and optionally set cfg)."""
    prev_cfg = C.get_active_sample_config()
    if cfg is not None:
        C.set_active_sample_config(cfg)
    install()
    try:
        yield
    finally:
        uninstall()
        C.set_active_sample_config(prev_cfg)
