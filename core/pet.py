"""Post-encroachment time (PET) on window-clipped trajectories.

Conflict zones follow Hetro_PET/pet_utils.calculate_pet:
  - each vehicle's zone is a circle on its own path at the any-time closest
    centerline point pair; the radius is the other vehicle's width;
  - gate: if the closest path-to-path distance exceeds conflict_threshold,
    there is no conflict zone (PET = inf);
  - enter/exit = first/last frame the vehicle's rotated bounding box overlaps
    its zone;
  - PET = (second vehicle's entry - first vehicle's exit) / fps, positive when
    vehicle 1 passes first.
Positions at boundary frames are sampled from each vehicle's own trajectory.
"""
from __future__ import annotations

import numpy as np

CONFLICT_THRESHOLD = 2.0   # gate on the any-time path-to-path min distance [m]
FPS = 30.0


def _zone_overlap(pos, heading_rad, length, width, center, radius):
    """Per-frame rotated-rectangle (vehicle bbox) vs circle overlap."""
    d = np.asarray(center)[None, :] - pos
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    lx = c * d[:, 0] + s * d[:, 1]          # circle center in body frame
    ly = -s * d[:, 0] + c * d[:, 1]
    dx = np.maximum(np.abs(lx) - length / 2.0, 0.0)
    dy = np.maximum(np.abs(ly) - width / 2.0, 0.0)
    return np.hypot(dx, dy) <= radius


def _pos_at(frames, pos, f):
    """Position sampled from a vehicle's own trajectory at (nearest) frame f."""
    i = int(np.abs(frames - f).argmin())
    return pos[i].copy()


def pet_window(f1, p1, h1_deg, l1, w1, f2, p2, h2_deg, l2, w2,
               conflict_threshold: float = CONFLICT_THRESHOLD, fps: float = FPS) -> dict:
    """PET of a vehicle pair over (already window-clipped) trajectories.

    f*: frame numbers (sorted); p*: (N, 2) xy; h*_deg: heading in degrees;
    l*/w*: vehicle length/width. Positive PET = vehicle 1 passes first.
    """
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    h1, h2 = np.radians(np.asarray(h1_deg, float)), np.radians(np.asarray(h2_deg, float))

    if len(np.intersect1d(f1, f2)) == 0:
        return dict(pet=np.inf, reason="no_frame_overlap")

    out: dict = {}
    D2 = ((p1[:, None, :] - p2[None, :, :]) ** 2).sum(-1)
    i1, i2 = np.unravel_index(int(np.argmin(D2)), D2.shape)
    path_min = float(np.sqrt(D2[i1, i2]))
    out.update(path_min_distance=round(path_min, 4),
               zone1_center=p1[i1].copy(), zone2_center=p2[i2].copy())

    # same-frame minimum distance (fallback anchor)
    common, a1, a2 = np.intersect1d(f1, f2, return_indices=True)
    if len(common):
        dsame = np.hypot(*(p1[a1] - p2[a2]).T)
        k = int(np.argmin(dsame))
        out.update(min_distance=round(float(dsame[k]), 4),
                   min_distance_frame=int(common[k]),
                   v1_pos_mindist=_pos_at(f1, p1, common[k]),
                   v2_pos_mindist=_pos_at(f2, p2, common[k]))

    if path_min > conflict_threshold:
        out.update(pet=np.inf, reason="no_conflict_zone_entry")
        return out

    z1 = _zone_overlap(p1, h1, l1, w1, p1[i1], w2)
    z2 = _zone_overlap(p2, h2, l2, w2, p2[i2], w1)
    if not z1.any() or not z2.any():
        out.update(pet=np.inf, reason="no_conflict_zone_entry")
        return out

    enter1, exit1 = int(f1[z1].min()), int(f1[z1].max())
    enter2, exit2 = int(f2[z2].min()), int(f2[z2].max())
    out.update(enter1=enter1, exit1=exit1, enter2=enter2, exit2=exit2)

    if exit1 < enter2:                     # vehicle 1 clears first
        pet, pf1, pf2, first = (enter2 - exit1) / fps, exit1, enter2, 1
    elif exit2 < enter1:                   # vehicle 2 clears first
        pet, pf1, pf2, first = -(enter1 - exit2) / fps, exit2, enter1, 2
    else:                                  # occupancy overlaps in time: PET = 0
        pet, pf1, pf2, first = 0.0, max(enter1, enter2), min(exit1, exit2), 0
    out.update(pet=round(pet, 3), pet_frame1=pf1, pet_frame2=pf2, first_passer=first,
               v1_pos_pet1=_pos_at(f1, p1, pf1), v2_pos_pet1=_pos_at(f2, p2, pf1),
               v1_pos_pet2=_pos_at(f1, p1, pf2), v2_pos_pet2=_pos_at(f2, p2, pf2))
    return out


def actor_anchor(res: dict, actor_is: int = 2) -> dict:
    """Anchor of the parameterized actor: its position at pet_frame1.

    Falls back to the same-frame minimum-distance point when PET is undefined.
    """
    v = f"v{actor_is}"
    if np.isfinite(res.get("pet", np.inf)):
        return dict(anchor="pet", crit_frame=res["pet_frame1"], crit_pos=res[f"{v}_pos_pet1"])
    if "min_distance_frame" in res:
        return dict(anchor="min_dist", crit_frame=res["min_distance_frame"],
                    crit_pos=res[f"{v}_pos_mindist"])
    return dict(anchor="none")
