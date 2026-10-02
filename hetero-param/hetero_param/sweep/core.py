"""
sweep.core — build the SET of concrete scenarios a parameterization renders, plus the
interaction descriptors of every member. Coverage / variance / redundancy are all
properties of this one swept set, so they share this builder.

SWEPT PARAMETERS (SweepAxes — defaults = the three real CARLA-swept param.xosc knobs
at {min, default, max}, matching the legacy flat coverage.py grid):
  start_offsets   Agent1_Offset        lateral start offset (m), taper weight 1→0
  end_offsets     Agent1_1_TA_Offset   lateral end offset  (m), taper weight 0→1
  speed_factors   Agent1_1_SA_EndSpeed x real speed profile (scales arc-length-vs-t;
                  the curve is extended tangentially when the scaled arc outruns it,
                  so no variant ever parks at the endpoint)
  shifts_s        interaction timing   continuous shift of the agent timeline (s),
                  default off (bbox PET floors frames to ints internally)

RENDERING one variant
  base NURBS (parampath.parameterized_path, the exact path esmini would follow)
  → lateral taper shift: p'(s) = p(s) + [so·(1−s̄) + eo·s̄] · n̂(s)   (n̂ = unit normal)
  → timed by the REAL actor's arc-length profile scaled by the speed factor:
      arc(t) = min(sf · arc_real(t), arc_max)   (real, non-uniform speed shape kept)
  → frames shifted by round(shift_s · fps)
  → kinematics by finite differences (similarity.core.derive_kinematics)

DESCRIPTORS per member (variant agent vs REAL ego) — the similarity-suite set
(PET [same estimator as the GT parquet], min TTC, min distance, arrival state at the
conflict point) merged with the interaction.py criticality set (conflict angle,
closing speed, DRAC).

OUTPUT  sweep() → SweptSet {variants: [Variant(params, traj, desc)], real: dict
        (the replay point = descriptors of the real pair), ego/actor Trajs, keys}
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .. import interaction as IX
from ..config import SampleConfig
from .. import parampath as PP
from ..similarity import core as SIMC, interaction_sim as IS

# descriptor keys the property modules operate on (heading keys get circular handling)
DESCRIPTOR_KEYS = ("pet_abs", "min_ttc", "min_dist", "conflict_angle",
                   "closing_speed", "drac",
                   "agent_arr_speed", "agent_arr_accel", "agent_arr_heading")


@dataclass(frozen=True)
class SweepAxes:
    start_offsets: tuple = (-0.5, 0.0, 0.5)
    end_offsets: tuple = (-0.5, 0.0, 0.5)
    speed_factors: tuple = (0.85, 1.0, 1.15)
    shifts_s: tuple = (0.0,)

    def grid(self):
        """Every (start_off, end_off, speed_factor, shift_s) combination."""
        return [dict(start_off=so, end_off=eo, speed_factor=sf, shift_s=sh)
                for so in self.start_offsets for eo in self.end_offsets
                for sf in self.speed_factors for sh in self.shifts_s]


DEFAULT_AXES = SweepAxes()


@dataclass
class Variant:
    params: dict           # the swept parameter values of this member
    traj: SIMC.Traj        # rendered agent trajectory
    desc: dict             # interaction descriptors vs the real ego


@dataclass
class SweptSet:
    dataset: str
    ego: int
    actor: int
    min_frame: int
    max_frame: int
    variants: list = field(default_factory=list)
    real: dict = field(default_factory=dict)       # replay point descriptors
    agent: SIMC.Traj | None = None                 # real agent (for spatial envelope)
    ego_traj: SIMC.Traj | None = None
    keys: tuple = DESCRIPTOR_KEYS

    def values(self, key: str) -> np.ndarray:
        return np.array([v.desc.get(key, float("nan")) for v in self.variants], float)

    def param_values(self, name: str) -> np.ndarray:
        return np.array([v.params.get(name, float("nan")) for v in self.variants], float)

    def param_names(self) -> list[str]:
        """Union of parameter names across members (esmini pipeline grids carry the
        actual xosc parameter names; analytic grids carry start_off/end_off/...)."""
        names: list[str] = []
        for v in self.variants:
            for k in v.params:
                if k not in names:
                    names.append(k)
        return names


def taper_shift(curve: np.ndarray, start_off: float, end_off: float) -> np.ndarray:
    """Shift each point along the local normal, tapering start_off@s=0 → end_off@s=1
    (same construction as the legacy flat coverage.py — the analytic analog of
    Agent1_Offset at the start and Agent1_1_TA_Offset at the end)."""
    if start_off == 0.0 and end_off == 0.0:
        return curve
    tang = np.gradient(curve, axis=0)
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    perp = np.stack([-tang[:, 1], tang[:, 0]], axis=1) / nrm
    t = np.linspace(0, 1, len(curve))[:, None]
    return curve + (start_off * (1 - t) + end_off * t) * perp


def _extend_tangentially(curve: np.ndarray, cs: np.ndarray, needed: float):
    """Append a straight segment along the end tangent so the curve's arc length
    strictly exceeds `needed`. Without this, variants whose scaled arc outruns the
    curve would PARK at the endpoint, and finite differences would fabricate
    heading = atan2(0,0) = 0° and a huge one-frame decel spike that corrupts the
    criticality / arrival descriptors."""
    i = len(curve) - 1
    while i > 0 and np.linalg.norm(curve[i] - curve[i - 1]) < 1e-6:
        i -= 1
    tang = curve[i] - curve[i - 1] if i > 0 else np.array([1.0, 0.0])
    tang = tang / max(np.linalg.norm(tang), 1e-9)
    ext = needed - cs[-1] + 1.0
    return (np.vstack([curve, curve[-1] + tang * ext]),
            np.append(cs, cs[-1] + ext))


def variant_traj(base_curve: np.ndarray, real: SIMC.Traj, params: dict) -> SIMC.Traj:
    """Render one variant Traj from the base NURBS curve + one parameter combination.
    Keeps the REAL (non-uniform) speed shape: arc(t) = sf · arc_real(t); the curve is
    extended tangentially whenever the scaled arc outruns it (no parked tail).
    shift_s is applied CONTINUOUSLY (float frames) — only the bbox PET estimator
    floors frames to ints internally (see interaction_sim.pet)."""
    curve = taper_shift(base_curve, params["start_off"], params["end_off"])
    rs = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(real.x), np.diff(real.y)))])
    cs = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(curve[:, 0]),
                                                   np.diff(curve[:, 1])))])
    needed = rs[-1] * params["speed_factor"]
    if needed >= cs[-1]:
        curve, cs = _extend_tangentially(curve, cs, needed)
    arc = rs * params["speed_factor"]
    px = np.interp(arc, cs, curve[:, 0])
    py = np.interp(arc, cs, curve[:, 1])
    frame = real.frame + params["shift_s"] * real.fps
    heading, speed, accel = SIMC.derive_kinematics(frame, px, py, real.fps)
    return SIMC.Traj(frame=frame, x=px, y=py, heading=heading, speed=speed,
                     accel=accel, fps=real.fps, length=real.length, width=real.width,
                     meta={**real.meta, "kind": "variant", "params": dict(params)})


def _criticality(agent: SIMC.Traj, ego: SIMC.Traj) -> dict:
    """conflict_angle / closing_speed / drac at closest approach (interaction.py set)."""
    r = IX._pair_interaction(agent.frame, agent.x, agent.y,
                             ego.frame, ego.x, ego.y, agent.fps)
    if r is None:
        return {"conflict_angle": float("nan"), "closing_speed": float("nan"),
                "drac": float("nan")}
    return {"conflict_angle": r["conflict_angle"], "closing_speed": r["closing_speed"],
            "drac": r["drac"]}


def descriptors(agent: SIMC.Traj, ego: SIMC.Traj, estimator: str = "bbox") -> dict:
    """Full descriptor dict for one (agent, ego) pair: similarity-suite descriptors
    (PET / TTC / min-dist / arrival state) + criticality (angle / closing / DRAC)."""
    return {**IS.descriptors(agent, ego, estimator), **_criticality(agent, ego)}


def _derived_kinematics_copy(t: SIMC.Traj) -> SIMC.Traj:
    """Same positions, kinematics recomputed by finite differences — so the replay
    point is measured with the SAME estimator as the variants (no recorded-vs-derived
    heading/accel bias in the covers_real tests)."""
    heading, speed, accel = SIMC.derive_kinematics(t.frame, t.x, t.y, t.fps)
    return SIMC.Traj(frame=t.frame, x=t.x, y=t.y, heading=heading, speed=speed,
                     accel=accel, fps=t.fps, length=t.length, width=t.width,
                     meta={**t.meta, "kinematics": "derived"})


def sweep(dataset: str, ego: int, actor: int, min_frame: int, max_frame: int,
          sample_cfg: SampleConfig | None = None, axes: SweepAxes = DEFAULT_AXES,
          estimator: str = "bbox", n_samples: int = 300,
          real_kinematics: str = "derived") -> SweptSet | None:
    """Render the full parameter grid for one labeled scenario and score every member
    against the real ego. Returns None if a real trajectory is unavailable.
    real_kinematics: 'derived' (default) scores the replay point with the same
    finite-difference kinematics as the variants (estimator-consistent — recorded
    parquet headings differ from derived ones by ~1–2° and would bias covers_real);
    'recorded' keeps the raw parquet kinematics."""
    a = SIMC.real_traj(dataset, actor, min_frame, max_frame)
    e = SIMC.real_traj(dataset, ego, min_frame, max_frame)
    if a is None or e is None:
        return None
    base = PP.parameterized_path(dataset, ego, actor, min_frame, max_frame,
                                 sample_cfg or _default_cfg(), n_samples=n_samples)
    a_scored = _derived_kinematics_copy(a) if real_kinematics == "derived" else a
    out = SweptSet(dataset, ego, actor, min_frame, max_frame,
                   agent=a, ego_traj=e, real=descriptors(a_scored, e, estimator))
    for params in axes.grid():
        vt = variant_traj(base, a, params)
        out.variants.append(Variant(params=params, traj=vt,
                                    desc=descriptors(vt, e, estimator)))
    return out


def _default_cfg() -> SampleConfig:
    from .. import config as C
    return C.SAMPLE_STRATEGIES["5pt-default"]


# ── shared numeric helpers (used by coverage / variance / redundancy) ────────

def is_heading_key(key: str) -> bool:
    return key.endswith("heading")


def wrap180(a):
    """Wrap angle(s) to (−180, 180]."""
    return np.asarray(a, float) - 360.0 * np.round(np.asarray(a, float) / 360.0)


def circular_mean(deg: np.ndarray) -> float:
    r = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))


def linearize_headings(values: np.ndarray, real: float | None = None):
    """Map headings (any convention: 0–360 or ±180) onto a LINEAR axis: rotate all by
    the set's circular mean and wrap to ±180, so range/std/inside-tests behave.
    Valid while the set spans < 360°, which a parameter sweep always satisfies.
    Returns (linearized values, linearized real)."""
    v = np.asarray(values, float)
    m = circular_mean(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0
    lin = wrap180(v - m)
    lin_real = float(wrap180(real - m)) if real is not None and np.isfinite(real) else float("nan")
    return lin, lin_real
