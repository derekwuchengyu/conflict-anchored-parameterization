"""Shared helpers for exp_ego64_39180.

Scenario: HetroD rec00 `39_180` — ego = track 39 (car), agent1 = track 180
(motorcycle), label 8 (KEEP + CUTIN_L), window frames 2424..3002 (16.6 s).

Method: "ours" best config for special_39_180 (memory our-method-best-config):
cp3 = 3 interior NURBS control points at (minPET crit - 10 m, crit, crit + 10 m)
arc length, speed event ramping to the agent's REAL speed at the traj_cross
frame  =>  SampleConfig(anchor="pet", placement="critdist", n_points=3,
step_m=10) + generate(force_anchor=True, speed_model="real",
speed_anchor="traj_cross")   ("px" arm, d = 10).

Everything upstream (hetero-param, exp_cross_coverage, sr-tlkeep-experiment) is
imported READ-ONLY; every artifact this project makes stays under
exp_ego64_39180/.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
CC = Path("/home/hcis-s19/Documents/ChengYu/exp_cross_coverage")
SR = Path("/home/hcis-s19/Documents/ChengYu/sr-tlkeep-experiment")
HP = Path("/home/hcis-s19/Documents/ChengYu/hetero-param")

sys.path.insert(0, str(SR / "scripts"))
sys.path.insert(0, str(HP))

# ── scenario constants ───────────────────────────────────────────────────────
DATASET = "HetroD"
EGO, ACTOR = 39, 180
MIN_FRAME, MAX_FRAME = 2424, 3002
LABEL = 8
FPS = 30.0
# scored / displayed window: the agent GT track ends at frame 2923
MAX_T = (2923 - 2424) / 30.0
SCN = "39_180"
STEM = "HetroD-01KEEP_02CUTIN_L_39_180_f2425"

# ours best config for this scenario ("px" arm at d = 10 m, pet_frame-fixed)
POS_ANCHOR, FRM_ANCHOR, STEP_M = "pet", "traj_cross", 10.0

# read-only inputs
BASE_XOSC = CC / "esmini_runs" / "_xosc_base_anch_px_s10_pf2" / f"{STEM}.xosc"
REAL_TRACKS = CC / "data" / "special_39_180" / "real_tracks.parquet"

# project-local outputs
RUNS = EXP / "runs"            # headless search renders (xosc + csv)
SHOTS = EXP / "shots"          # interaction-moment stills
GIFS = EXP / "gifs"
RESULTS = EXP / "results"
FIGS = EXP / "figs"
CAP = EXP / "capture"          # scratch tga frames (deleted after encoding)
for _d in (RUNS, SHOTS, GIFS, RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# esmini: use the symlinked binary so the install's config.yml is NOT loaded
# (esmini resolves the default config relative to the binary path).
ESMINI = HP / "results" / "esmini" / "bin" / "esmini"

# Inline the entities with the RECORDED dimensions instead of keeping the
# pipeline's CatalogReference.  /opt/Catalogs exists on this machine, so the
# surgery would otherwise hand esmini the CARLA blueprint boxes — ego
# 4.5 x 2.1 and motorcycle 2.0 x 0.71, both with `Center x="1.5"` — while every
# metric in this project (esmini_exec.run_and_extract -> similarity/sweep) is
# computed on the DATASET boxes (4.4721 x 1.9 / 1.6086 x 0.6748) centred on the
# recorded position.  Keeping the catalog boxes made esmini render and
# collision-check a different geometry from the one being scored.
from hetero_param import esmini_exec as _EX  # noqa: E402
_EX.CATALOG_ENTITIES = False

# new parameter names added to the base xosc by `parameterize()`
P_MAJOR, P_MINOR = "Crit_Off_Major", "Crit_Off_Minor"
# the two live speed-event knobs of the pipeline parameterization
P_ES, P_DU, P_V0 = ("Agent1_1_SA_EndSpeed",
                    "Agent1_1_SA_DynamicDuration", "Agent1_Speed")
P_THETA = "Crit_Theta"
# taper of the crit-CP displacement onto its two flank CPs.  A single displaced
# CP makes a local bulge that weight tuning cannot smooth out (exp_critpos_offset);
# spreading the same displacement over the neighbours keeps the curve natural.
TAPER = (0.35, 1.0, 0.35)


# ── xosc parameterization ────────────────────────────────────────────────────
def crit_cps(xosc: Path) -> list[tuple[float, float]]:
    """The 3 weighted interior NURBS control points (arc order)."""
    root = ET.parse(xosc).getroot()
    out = []
    for cp in root.iter("ControlPoint"):
        if cp.get("weight") is None:
            continue
        wp = cp.find("Position/WorldPosition")
        out.append((float(wp.get("x")), float(wp.get("y"))))
    return out


def parameterize(base: Path, out: Path,
                 taper: tuple[float, float, float] = TAPER) -> Path:
    """Expose the crit control point as two xosc parameters.

    Three axes, all linear in the control-point coordinates so they fit esmini's
    arithmetic-only expressions:

    - `Crit_Off_Major` slides the crit CP along the local path tangent and
      `Crit_Off_Minor` along the local normal (left-positive); both are spread
      over the three interior CPs with `taper` weights so the NURBS stays smooth
      (exp_critpos_offset "wide" taper trick, adapted to a 3-CP shape).
    - `Crit_Theta` [m] is the project's established THETA axis
      (exp_cutin_straight 08_theta_d_sweep): an **antisymmetric flank offset** —
      the crit-d CP moves +theta along the normal, the crit+d CP moves -theta,
      the crit CP itself does not move.  It bends the path THROUGH the conflict
      point instead of shifting it, which is what makes the sampled paths look
      different rather than merely displaced.  Realized S peak-to-peak is
      ~1.567*|theta| on this 7-CP order-4 NURBS.

    Coordinates become esmini expressions
    x = ${x0 + w*ux*$Major + w*vx*$Minor +- vx*$Theta}.
    """
    tree = ET.parse(base)
    root = tree.getroot()
    cps = [cp for cp in root.iter("ControlPoint") if cp.get("weight") is not None]
    if len(cps) != 3:
        raise RuntimeError(f"expected 3 weighted CPs, got {len(cps)}")
    pts = [(float(c.find("Position/WorldPosition").get("x")),
            float(c.find("Position/WorldPosition").get("y"))) for c in cps]

    # local frame at the crit CP: tangent = flank-to-flank chord, normal = left
    dx, dy = pts[2][0] - pts[0][0], pts[2][1] - pts[0][1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    vx, vy = -uy, ux

    decls = root.find("ParameterDeclarations")
    have = {p.get("name") for p in decls.iter("ParameterDeclaration")}
    for name in (P_MAJOR, P_MINOR, P_THETA):
        if name not in have:
            ET.SubElement(decls, "ParameterDeclaration", name=name,
                          parameterType="double", value="0.0")

    # antisymmetric: +1 on the upstream flank, 0 at crit, -1 on the downstream
    theta_sign = (1.0, 0.0, -1.0)
    for cp, (x0, y0), w, ts in zip(cps, pts, taper, theta_sign):
        wp = cp.find("Position/WorldPosition")
        tx = f" + {ts * vx:.6f}*${P_THETA}" if ts else ""
        ty = f" + {ts * vy:.6f}*${P_THETA}" if ts else ""
        wp.set("x", f"${{{x0:.4f} + {w * ux:.6f}*${P_MAJOR} "
                    f"+ {w * vx:.6f}*${P_MINOR}{tx}}}")
        wp.set("y", f"${{{y0:.4f} + {w * uy:.6f}*${P_MAJOR} "
                    f"+ {w * vy:.6f}*${P_MINOR}{ty}}}")

    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def base_params(xosc: Path) -> dict[str, float | str]:
    root = ET.parse(xosc).getroot()
    out = {}
    for p in root.iter("ParameterDeclaration"):
        v = p.get("value")
        try:
            out[p.get("name")] = float(v)
        except (TypeError, ValueError):
            out[p.get("name")] = v
    return out


# ── esmini execution ─────────────────────────────────────────────────────────
def run_esmini(xosc: Path, csv_out: Path, timeout: int = 180,
               extra: list[str] | None = None, cwd: Path | None = None) -> bool:
    cmd = ["nice", "-n", "10", str(ESMINI), "--osc", str(xosc), "--headless",
           "--fixed_timestep", f"{1.0 / FPS:.6f}",
           "--collision", "--csv_logger", str(csv_out),
           "--disable_stdout", "--disable_log"]
    if extra:
        cmd += extra
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           cwd=str(cwd) if cwd else None)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and csv_out.exists()


_NUM = re.compile(r"^\s*#(\d) (.+?)\s*\[?")


_DIMS: dict[str, tuple[float, float]] = {}


def dataset_dims() -> dict[str, tuple[float, float]]:
    """(length, width) of ego and agent1 as RECORDED — the boxes every metric
    in this project uses (esmini_exec.run_and_extract takes them from
    real_traj, not from the csv's bb_* columns)."""
    if not _DIMS:
        from hetero_param.similarity import core as SIMC
        for role, tid in (("ego", EGO), ("agent", ACTOR)):
            t = SIMC.real_traj(DATASET, tid, MIN_FRAME, MAX_FRAME)
            _DIMS[role] = (float(t.length), float(t.width))
    return _DIMS


def read_csv(path: Path) -> pd.DataFrame | None:
    """esmini --csv_logger output → tidy frame (t, ego/agent pose + speed)."""
    try:
        df = pd.read_csv(path, skiprows=6, skipinitialspace=True,
                         low_memory=False)
    except Exception:  # noqa: BLE001
        return None
    df.columns = [c.strip() for c in df.columns]
    need = {f"#{i} {k}" for i in (1, 2) for k in
            ("World_Position_X [m]", "World_Position_Y [m]",
             "World_Heading_Angle [rad]", "Current_Speed [m/s]",
             "bb_length [m]", "bb_width [m]")}
    if not need.issubset(df.columns):
        return None
    out = pd.DataFrame({"t": df["TimeStamp [s]"].astype(float)})
    for i, role in ((1, "ego"), (2, "agent")):
        out[f"{role}_x"] = df[f"#{i} World_Position_X [m]"].astype(float)
        out[f"{role}_y"] = df[f"#{i} World_Position_Y [m]"].astype(float)
        out[f"{role}_h"] = df[f"#{i} World_Heading_Angle [rad]"].astype(float)
        out[f"{role}_v"] = df[f"#{i} Current_Speed [m/s]"].astype(float)
        # NOT the csv's bb_length/bb_width: those are whatever entity
        # description esmini resolved (catalog blueprint or inlined box).  The
        # scored geometry is the recorded one, always.
        Lm, Wm = dataset_dims()[role]
        out[f"{role}_L"] = Lm
        out[f"{role}_W"] = Wm
    col = df["#1 collision_ids"].astype(str).str.strip()
    out["collision"] = (col != "") & (col.str.lower() != "nan")
    out["frame"] = MIN_FRAME + np.round(out.t.values * FPS).astype(int)
    return out


# ── scoring ──────────────────────────────────────────────────────────────────
def _rect(x, y, h, L, W):
    from shapely.geometry import Polygon
    c, s = math.cos(h), math.sin(h)
    hl, hw = L / 2.0, W / 2.0
    pts = [(x + c * a - s * b, y + s * a + c * b)
           for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
    return Polygon(pts)


def bbox_gap(df: pd.DataFrame) -> np.ndarray:
    """Per-frame bounding-box separation [m] (0 while the boxes overlap)."""
    gaps = np.empty(len(df))
    for i, r in enumerate(df.itertuples()):
        a = _rect(r.ego_x, r.ego_y, r.ego_h, r.ego_L, r.ego_W)
        b = _rect(r.agent_x, r.agent_y, r.agent_h, r.agent_L, r.agent_W)
        gaps[i] = a.distance(b)
    return gaps


def teleport_index(df: pd.DataFrame, role: str = "agent") -> int | None:
    """Index of the FIRST esmini route-overrun snap, or None.

    A one-frame jump far beyond the entity's own state speed means the agent
    reached the end of its NURBS route and esmini snapped it metres in a single
    step.  The first 10 steps are ignored: applying a lane Offset teleports the
    start laterally by design.

    Measured on this scenario: the snap ALWAYS happens after the closest
    approach (120/120 sampled runs), so the interaction itself is intact and the
    right response is to crop the run here rather than throw the sample away —
    which matters a lot, because the snap rate is strongly theta-dependent
    (70 % at theta in [-6,-4] vs 2 % at [+4,+6]) and rejecting them would have
    silently deleted half the theta axis.
    """
    x, y = df[f"{role}_x"].values, df[f"{role}_y"].values
    if len(x) < 12:
        return None
    ps = np.hypot(np.diff(x), np.diff(y)) * FPS
    vmax = float(df[f"{role}_v"].max())
    bad = np.where((np.arange(len(ps)) > 10) & (ps > 30.0) & (ps > vmax + 10.0))[0]
    return int(bad[0]) if len(bad) else None


def crop_clean(df: pd.DataFrame, role: str = "agent") -> pd.DataFrame:
    """Rows up to (not including) the first route-overrun snap."""
    k = teleport_index(df, role)
    return df if k is None else df.iloc[:k].reset_index(drop=True)


def is_teleport(df: pd.DataFrame, role: str = "agent") -> bool:
    return teleport_index(df, role) is not None


def score(df: pd.DataFrame) -> dict:
    """Criticality summary of one render."""
    gap = bbox_gap(df)
    k = int(np.argmin(gap))
    ctr = np.hypot(df.agent_x - df.ego_x, df.agent_y - df.ego_y).values
    hit = bool(df.collision.any()) or float(gap.min()) <= 0.0
    n_hit = int((gap <= 0.0).sum())
    return {
        "min_gap": float(gap.min()),
        "min_gap_t": float(df.t.values[k]),
        "min_gap_frame": int(df.frame.values[k]),
        "min_ctr": float(ctr.min()),
        "collision": hit,
        "n_overlap_frames": n_hit,
        "ego_v_at_min": float(df.ego_v.values[k]),
        "agent_v_at_min": float(df.agent_v.values[k]),
        "agent_v_max": float(df.agent_v.max()),
        "t_end": float(df.t.values[-1]),
        "teleport": is_teleport(df),
    }


def gt_tracks() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_parquet(REAL_TRACKS)
    return (t[t.role == "ego"].sort_values("frame").reset_index(drop=True),
            t[t.role != "ego"].sort_values("frame").reset_index(drop=True))
