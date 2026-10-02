"""Approved new-experiment bbox-PET angle adapter.

The function body matches the reviewed proposal. Only this bbox-PET call receives
radian copies and heading_in_degrees=False; other metrics keep degree Traj data.
The original proposal remains under review_only for provenance. No FPS, gate,
threshold, frame cast, upstream source, or trajectory change is introduced.
"""
from __future__ import annotations

import numpy as np


def pet_bbox_degrees(agent, ego, *, calculate_pet_fn,
                     conflict_threshold: float = 1.4, gate: str = "bbox"):
    """Return the same PET fields as interaction_sim.pet(estimator='bbox').

    Input: Traj.heading in degrees, Traj.fps from its dataset. Caller example:
        pet_bbox_degrees(a, e, calculate_pet_fn=ISIM.pet_utils.calculate_pet)

    The existing pet_utils API already supports heading_in_degrees=False.
    Converting copies here AND setting that flag supplies radians to the legacy
    zone loop while preventing the swept-body gate from converting again.
    """
    fps = float(agent.fps)
    if not np.isfinite(fps) or fps <= 0 or fps != float(ego.fps):
        raise ValueError("Both trajectories must use the same positive dataset FPS")

    agent_heading_rad = np.radians(np.asarray(agent.heading, dtype=float))
    ego_heading_rad = np.radians(np.asarray(ego.heading, dtype=float))
    value, _frame1, _frame2, meta = calculate_pet_fn(
        agent.xy, ego.xy, agent.frame.astype(int), ego.frame.astype(int),
        length1=agent.length, width1=agent.width, heading1=agent_heading_rad,
        length2=ego.length, width2=ego.width, heading2=ego_heading_rad,
        conflict_threshold=conflict_threshold,
        fps=fps,                         # unchanged; 39_180 remains 30 Hz
        gate=gate,                       # unchanged gate and threshold
        heading_in_degrees=False,        # avoid a second conversion in the gate
    )
    value = float(value)
    return {
        "pet": value,
        "pet_abs": abs(value),
        "pet_type": meta.get("type") or meta.get("reason"),
        "min_path_dist": float(meta.get("min_distance", float("nan"))),
    }
