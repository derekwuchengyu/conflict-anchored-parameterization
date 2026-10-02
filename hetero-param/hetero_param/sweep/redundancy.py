"""
sweep.redundancy — how much of the swept set is EFFECTIVELY duplicated: parameter
combinations that render nearly the same test case add cost without adding coverage.
Redundancy is the complement of scenario diversity (SUNRISE §3.4: Euclidean
parameter-space dissimilarity #53, DTW trajectory dissimilarity #55, MOSAT
average-dissimilarity diversity #64).

PAIRWISE DISSIMILARITY D_ij between swept members (choose the space)
  mode="descriptor"  Euclidean distance between z-scored descriptor vectors
                     (outcome space: two variants that produce the same interaction
                     are redundant test cases, whatever their parameters).
                     Members with any non-finite descriptor are excluded (reported).
                     Degenerate descriptors (zero spread) contribute 0.
  mode="path"        length-normalized 2-D DTW between the variant agent PATHS
                     (geometry space; metres) — similarity.path_sim.dtw.

METRICS (threshold eps: z-units for descriptor mode, metres for path mode)
  diversity        mean_{i<j} D_ij                 (MOSAT-style average dissimilarity)
  nn_i             min_{j≠i} D_ij                  nearest-neighbour distance
  redundant_frac   |{i : nn_i < eps}| / n          members with a near-duplicate
  n_eff            greedy ε-cover size: walk members in order, keep one as a
                   representative iff its distance to every kept representative
                   ≥ eps; n_eff = #representatives (deterministic in member order)
  redundancy       1 − n_eff / n  ∈ [0, 1)         0 = every member distinct at eps
  curve            redundancy at each eps in eps_list (sensitivity to the threshold)

INPUT   SweptSet (from sweep.core.sweep)
OUTPUT  report() → {mode, eps, n, n_used, diversity, min_nn, median_nn,
                    redundant_frac, n_eff, redundancy, curve}
"""
from __future__ import annotations
import numpy as np

from ..similarity import path_sim
from . import core

DEFAULT_EPS = {"descriptor": 0.5, "path": 0.25}
DEFAULT_EPS_LIST = {"descriptor": (0.1, 0.25, 0.5, 1.0, 2.0),
                    "path": (0.05, 0.1, 0.25, 0.5, 1.0)}


def descriptor_matrix(swept: core.SweptSet, keys=None, min_col_finite: float = 0.8):
    """(X z-scored, used-member indices, used keys). Headings linearized. Columns
    that are non-finite for more than (1 − min_col_finite) of the members are dropped
    first (e.g. PET/TTC = inf on a no-conflict scenario), THEN members with any
    remaining non-finite value. Zero-spread columns become 0 so they never separate
    members."""
    keys = list(keys or swept.keys)
    cols, kept_keys = [], []
    for k in keys:
        v = swept.values(k)
        if core.is_heading_key(k):
            v, _ = core.linearize_headings(v)
        if len(v) and np.isfinite(v).mean() >= min_col_finite:
            cols.append(v)
            kept_keys.append(k)
    if not cols:
        return np.zeros((0, 0)), [], []
    X = np.column_stack(cols)
    used = np.where(np.isfinite(X).all(axis=1))[0]
    X = X[used]
    if len(X):
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        X = (X - mu) / sd
    return X, used.tolist(), kept_keys


def dissimilarity_matrix(swept: core.SweptSet, mode: str = "descriptor",
                         keys=None, dtw_subsample: int = 5):
    """Symmetric pairwise dissimilarity matrix + (used member indices, used keys)."""
    if mode == "descriptor":
        X, used, kept = descriptor_matrix(swept, keys)
        if len(X) == 0:
            return np.zeros((0, 0)), (used, kept)
        diff = X[:, None, :] - X[None, :, :]
        return np.linalg.norm(diff, axis=2), (used, kept)
    if mode != "path":
        raise ValueError(f"unknown mode {mode!r} (use 'descriptor' or 'path')")
    used = list(range(len(swept.variants)))
    paths = [v.traj.xy for v in swept.variants]
    n = len(paths)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = path_sim.dtw(paths[i][::dtw_subsample], paths[j],
                                             subsample=dtw_subsample)
    return D, (used, ["xy_path"])


def nn_distances(D: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance per member."""
    if len(D) < 2:
        return np.array([])
    M = D + np.diag(np.full(len(D), np.inf))
    return M.min(axis=1)


def effective_count(D: np.ndarray, eps: float) -> int:
    """Greedy ε-cover: keep member i as a representative iff it is ≥ eps away from
    every representative kept so far. Deterministic in member (grid) order."""
    reps: list[int] = []
    for i in range(len(D)):
        if all(D[i, r] >= eps for r in reps):
            reps.append(i)
    return len(reps)


def metrics(D: np.ndarray, eps: float) -> dict:
    """Redundancy metrics of one dissimilarity matrix at one threshold."""
    n = len(D)
    if n < 2:
        return {"n_used": n, "diversity": float("nan"), "min_nn": float("nan"),
                "median_nn": float("nan"), "redundant_frac": float("nan"),
                "n_eff": n, "redundancy": float("nan")}
    iu = np.triu_indices(n, k=1)
    nn = nn_distances(D)
    n_eff = effective_count(D, eps)
    return {"n_used": n, "diversity": float(D[iu].mean()),
            "min_nn": float(nn.min()), "median_nn": float(np.median(nn)),
            "redundant_frac": float((nn < eps).mean()),
            "n_eff": n_eff, "redundancy": float(1.0 - n_eff / n)}


# ── collision detection (oriented bounding boxes, separating-axis test) ──────

def _obb_corners(x, y, h_deg, length, width):
    """(4,2) rotated bbox corners around center (correct radians, unlike the GT
    parquet estimator's raw-degree quirk — collision here is geometric truth)."""
    h = np.radians(h_deg)
    c, s = np.cos(h), np.sin(h)
    dx, dy = length / 2.0, width / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def _obb_overlap(c1, c2) -> bool:
    """Separating Axis Theorem for two convex quads (2 edge normals each):
    True iff the boxes overlap (no separating axis exists)."""
    for rect in (c1, c2):
        for i in (0, 1):
            edge = rect[i + 1] - rect[i]
            axis = np.array([-edge[1], edge[0]])
            p1, p2 = c1 @ axis, c2 @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False                       # separating axis → disjoint
    return True


def collision_flag(agent, ego) -> dict:
    """Did the two rotated bboxes ever overlap on the shared timeline?
    Positions/headings interpolated onto the common integer-frame grid; dims from
    the Traj (dataset bbox). Returns {collision, first_frame, n_overlap_frames}."""
    lo = max(agent.frame.min(), ego.frame.min())
    hi = min(agent.frame.max(), ego.frame.max())
    if hi - lo < 2 or not (np.isfinite(agent.length) and np.isfinite(ego.length)):
        return {"collision": False, "first_frame": float("nan"), "n_overlap_frames": 0}
    grid = np.arange(lo, hi + 1)
    ax = np.interp(grid, agent.frame, agent.x)
    ay = np.interp(grid, agent.frame, agent.y)
    ah = np.interp(grid, agent.frame, np.unwrap(agent.heading, period=360.0))
    ex = np.interp(grid, ego.frame, ego.x)
    ey = np.interp(grid, ego.frame, ego.y)
    eh = np.interp(grid, ego.frame, np.unwrap(ego.heading, period=360.0))
    # cheap gate: centers further apart than the two half-diagonals cannot overlap
    gate = (np.hypot(agent.length, agent.width) + np.hypot(ego.length, ego.width)) / 2.0
    close = np.hypot(ax - ex, ay - ey) <= gate
    hits = []
    for i in np.where(close)[0]:
        c1 = _obb_corners(ax[i], ay[i], ah[i], agent.length, agent.width)
        c2 = _obb_corners(ex[i], ey[i], eh[i], ego.length, ego.width)
        if _obb_overlap(c1, c2):
            hits.append(grid[i])
    return {"collision": bool(hits),
            "first_frame": float(hits[0]) if hits else float("nan"),
            "n_overlap_frames": len(hits)}


def collision_redundancy(swept: core.SweptSet) -> dict:
    """Outcome-equivalence redundancy with COLLISION (bbox contact) as the label:
    combos with the same collision/no-collision outcome are redundant for a
    does-it-crash experiment — keep one representative per label."""
    labels, first_frames = [], []
    for v in swept.variants:
        cf = collision_flag(v.traj, swept.ego_traj)
        labels.append("collision" if cf["collision"] else "no_collision")
        first_frames.append(cf["first_frame"])
    real_cf = collision_flag(swept.agent, swept.ego_traj)
    n = len(labels)
    reps: dict[str, dict] = {}
    for lab, v in zip(labels, swept.variants):
        reps.setdefault(lab, dict(v.params))
    return {"n": n, "n_collision": labels.count("collision"),
            "n_no_collision": labels.count("no_collision"),
            "real_label": "collision" if real_cf["collision"] else "no_collision",
            "n_classes": len(reps),
            "redundancy": float(1.0 - len(reps) / n) if n else float("nan"),
            "representatives": reps, "labels": labels,
            "first_frames": first_frames,
            "collision_params": [dict(v.params) for lab, v in zip(labels, swept.variants)
                                 if lab == "collision"]}


# ── criticality-equivalence redundancy (outcome = dangerous / safe) ──────────
# Thresholds (surrogate-safety literature):
#   TTC_K = 1.5 s — the standard urban conflict threshold (van der Horst 1990
#           braking-urgency; SSAM's default max-TTC is 1.5 s); 2.0 s = conservative.
#   PET_G = 1.0 s — severe-conflict PET at intersections (Hydén 1987 severity
#           grading; common urban studies); 1.5–2.0 s = screening band. (SSAM's
#           PET 5.0 s default is a RECORDING envelope, not a danger threshold.)
TTC_K = 1.5
PET_G = 1.0


def criticality_labels(swept: core.SweptSet, ttc_k: float = TTC_K,
                       pet_g: float = PET_G) -> list[str]:
    """'dangerous' iff min_TTC < ttc_k OR |PET| < pet_g (inf = no event = safe)."""
    out = []
    for v in swept.variants:
        ttc, pet = v.desc.get("min_ttc"), v.desc.get("pet_abs")
        dang = (ttc is not None and np.isfinite(ttc) and ttc < ttc_k) or \
               (pet is not None and np.isfinite(pet) and pet < pet_g)
        out.append("dangerous" if dang else "safe")
    return out


def criticality_redundancy(swept: core.SweptSet, ttc_k: float = TTC_K,
                           pet_g: float = PET_G) -> dict:
    """Outcome-equivalence redundancy: parameter combos with the SAME danger label
    are redundant for safety testing — keep one representative per label, skip the
    rest. redundancy = 1 − n_classes/n (with binary labels this is ≥ 1 − 2/n; the
    informative parts are the split and WHICH combos flip the label)."""
    labels = criticality_labels(swept, ttc_k, pet_g)
    n = len(labels)
    reps: dict[str, dict] = {}
    for lab, v in zip(labels, swept.variants):
        reps.setdefault(lab, dict(v.params))
    r = swept.real
    real_dang = (np.isfinite(r.get("min_ttc", np.inf)) and r["min_ttc"] < ttc_k) or \
                (np.isfinite(r.get("pet_abs", np.inf)) and r["pet_abs"] < pet_g)
    return {"ttc_k": ttc_k, "pet_g": pet_g, "n": n,
            "n_dangerous": labels.count("dangerous"),
            "n_safe": labels.count("safe"),
            "real_label": "dangerous" if real_dang else "safe",
            "n_classes": len(reps),
            "redundancy": float(1.0 - len(reps) / n) if n else float("nan"),
            "representatives": reps, "labels": labels,
            "dangerous_params": [dict(v.params) for lab, v in zip(labels, swept.variants)
                                 if lab == "dangerous"]}


def report(swept: core.SweptSet, mode: str = "descriptor", eps: float | None = None,
           keys=None, eps_list=None) -> dict:
    """Full redundancy report for one swept set."""
    D, (used, kept_keys) = dissimilarity_matrix(swept, mode, keys)
    eps = DEFAULT_EPS[mode] if eps is None else eps
    eps_list = DEFAULT_EPS_LIST[mode] if eps_list is None else eps_list
    out = {"mode": mode, "eps": float(eps), "n": len(swept.variants),
           "used_keys": list(kept_keys), **metrics(D, eps)}
    out["curve"] = {float(e): metrics(D, float(e))["redundancy"] for e in eps_list}
    return out
