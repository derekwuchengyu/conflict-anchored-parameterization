"""REVIEW ONLY: proposed new-experiment bbox-PET unit adapter.

This file is not connected to any experiment runner or scorer. It does not
modify upstream pet_utils, Traj objects, recorded data, FPS, or XOSC behavior.
If approved, the caller supplies the already-selected legacy calculate_pet
function and uses this adapter ONLY for bbox-PET. Collision SAT and all other
descriptors continue receiving degree-valued Traj.heading.
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
