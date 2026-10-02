"""
Configuration objects that parameterize the *choice* of parameterization — the very
thing this paper studies. Two orthogonal knobs:

  SampleConfig   — WHICH sample points go into the NURBS challenge path
                   (Experiment 1: point count/placement; Experiment 2: temporal anchor)
  StrategyConfig — A/B/C parameterization families (Experiment 3):
                     A = initial-condition only (no shape points, constant speed)
                     B = + activity/critical-frame (no shape points, real speed profile)
                     C = + trajectory shape (NURBS shape points, real speed profile)

`sampling.py` reads the *active* SampleConfig via get_active_sample_config(); `generate.py`
sets it (and the active StrategyConfig) before driving Stage A.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Union

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset fps — MUST be used explicitly everywhere (inD=25, HetroD=30).
# The bug we are avoiding: paper_analysis/gt_fidelity.py hardcodes FPS=25 for both.
# ─────────────────────────────────────────────────────────────────────────────
DATASET_FPS = {"HetroD": 30, "inD": 25}
DATASET_ID = {"HetroD": "00", "inD": "27"}


# ─────────────────────────────────────────────────────────────────────────────
# SampleConfig — Experiment 1 (point) & Experiment 2 (frame)
# ─────────────────────────────────────────────────────────────────────────────
# anchor    : which real frame is the "critical frame" the points cluster around.
#             "pet"       -> PET conflict frame  (pet_frame1)
#             "min_dist"  -> min-distance frame  (min_distance_frame)  [interaction]
#             "min_speed" -> min-speed frame before the critical frame
# placement : how the intermediate NURBS control points are chosen.
#             "none"      -> no intermediate points (start+end only)
#             "midpoint"  -> single 50%-arclength point
#             "conflict"  -> single point at the anchored critical frame
#             "interaction" -> single point at the min-distance frame
#             "uniform"   -> arclength quartiles (25/50/75%)
#             "critwindow"-> [crit-window, crit, crit+window]
#             "default"   -> [crit-window, crit, crit+window] + quartiles  (the legacy behavior)
#             "full"      -> sentinel: challenge agent is full-replayed (handled in generate.py)
# n_points  : nominal number of intermediate points (int) or "full".
# window_s  : half-window in seconds around the critical frame (legacy = 3.0s).
# min_dist_m: drop points closer than this to the previous kept point (legacy = 0.5m).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SampleConfig:
    name: str = "5pt-default"
    anchor: str = "pet"
    placement: str = "default"
    n_points: Union[int, str] = 5
    window_s: float = 3.0       # legacy half-window for "critwindow"
    step_m: float = 3.0         # arc-length spacing between points for "critdist"
    step_s: float = 1.0         # time spacing between points for "crittime"
    min_dist_m: float = 0.5

    @property
    def is_full_replay(self) -> bool:
        return self.placement == "full" or self.n_points == "full"


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1b — sampling METHOD / position (slide 25): given a point budget, does it
# matter HOW you place the points? uniform along the whole trajectory vs concentrated
# near the crossing, by arc-length distance (critdist) or by frame time (crittime); and
# whether the crossing is the trajectory/PET point (anchor=pet) or the interaction/
# min-distance point (anchor=min_dist).
# ─────────────────────────────────────────────────────────────────────────────
def anchor_methods(n_points: int = 4, step_s: float = 0.7) -> dict[str, "SampleConfig"]:
    """The sampling-METHOD comparison (the main axis of Exp 1b): at a fixed point budget,
    WHERE do the points go? Four clean methods matching the domain vocabulary —
      uniform     : evenly along the whole trajectory
      traj_cross  : concentrated at the 軌跡交錯點 (spatial trajectory-crossing point)
      pet         : concentrated at the PET frame (temporal post-encroachment)
      min_dist    : concentrated at the min-distance frame (temporal closest approach)
    """
    # Each anchored method = a UNIFORM base (arc-length quartiles) PLUS extra density around
    # its anchor (placement "default" = [crit-w, crit, crit+w] + quartiles). This is
    # "upsampling near the crossing" on top of a base — not cramming all points at one point
    # (which degenerates the NURBS). 'uniform' is the base alone.
    return {
        "uniform":    SampleConfig("uniform",    anchor="pet",        placement="uniform",  n_points=n_points),
        "traj_cross": SampleConfig("traj_cross", anchor="traj_cross", placement="default", n_points=n_points, window_s=1.0),
        "pet":        SampleConfig("pet",        anchor="pet",        placement="default", n_points=n_points, window_s=1.0),
        "min_dist":   SampleConfig("min_dist",   anchor="min_dist",   placement="default", n_points=n_points, window_s=1.0),
    }


def sampling_methods(n_points: int = 4) -> dict[str, "SampleConfig"]:
    return {
        "uniform":          SampleConfig(f"uniform-{n_points}",     anchor="pet",
                                         placement="uniform",  n_points=n_points),
        "critdist@pet":     SampleConfig(f"critdist-{n_points}@pet", anchor="pet",
                                         placement="critdist", n_points=n_points, step_m=3.0),
        "crittime@pet":     SampleConfig(f"crittime-{n_points}@pet", anchor="pet",
                                         placement="crittime", n_points=n_points, step_s=1.0),
        "critdist@mindist": SampleConfig(f"critdist-{n_points}@mindist", anchor="min_dist",
                                         placement="critdist", n_points=n_points, step_m=3.0),
        "crittime@mindist": SampleConfig(f"crittime-{n_points}@mindist", anchor="min_dist",
                                         placement="crittime", n_points=n_points, step_s=1.0),
    }


# Experiment 1 sweep — spatial point count / placement (all with anchor="pet"
# unless the placement itself implies a different anchor).
SAMPLE_STRATEGIES: dict[str, SampleConfig] = {
    "0pt":            SampleConfig("0pt",            anchor="pet",      placement="none",        n_points=0),
    "1pt-midpoint":   SampleConfig("1pt-midpoint",   anchor="pet",      placement="midpoint",    n_points=1),
    "1pt-conflict":   SampleConfig("1pt-conflict",   anchor="pet",      placement="conflict",    n_points=1),
    "1pt-interaction":SampleConfig("1pt-interaction",anchor="min_dist", placement="interaction", n_points=1),
    "3pt-uniform":    SampleConfig("3pt-uniform",    anchor="pet",      placement="uniform",     n_points=3),
    "3pt-critwindow": SampleConfig("3pt-critwindow", anchor="pet",      placement="critwindow",  n_points=3),
    "5pt-default":    SampleConfig("5pt-default",    anchor="pet",      placement="default",     n_points=5),
    "full-replay":    SampleConfig("full-replay",    anchor="pet",      placement="full",        n_points="full"),
}

# Experiment 2 — the three temporal anchors, placement fixed at critwindow so only
# the anchor varies (isolates the frame choice).
ANCHORS = ("pet", "min_dist", "min_speed")

def anchor_config(anchor: str) -> SampleConfig:
    """A SampleConfig that varies only the temporal anchor (Experiment 2a)."""
    return SampleConfig(name=f"anchor-{anchor}", anchor=anchor,
                        placement="critwindow", n_points=3)


# ─────────────────────────────────────────────────────────────────────────────
# StrategyConfig — Experiment 3 (A / B / C)
# ─────────────────────────────────────────────────────────────────────────────
# use_shape      : keep the NURBS intermediate shape points (C) or drop them (A, B).
# constant_speed : flatten the speed event to the start speed (A) or keep the real
#                  profile toward the critical-frame speed (B, C).
# sample         : the SampleConfig used when use_shape is True.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StrategyConfig:
    name: str                       # "A" | "B" | "C"
    use_shape: bool
    constant_speed: bool
    sample: SampleConfig = field(default_factory=lambda: SAMPLE_STRATEGIES["5pt-default"])

    @property
    def effective_sample(self) -> SampleConfig:
        """The SampleConfig actually fed to the sampler for this strategy."""
        if self.use_shape:
            return self.sample
        # A and B have no shape points.
        return replace(self.sample, name=f"{self.name}-noshape",
                       placement="none", n_points=0)


# Mapping to de Gelder & Op den Camp (arXiv 2409.01117), whose scenario = initial state
# + activities (maneuvers over time), parameterized on straight highways:
#   A = initial state + CONSTANT VELOCITY, no activity, endpoint-determined path.
#       == de Gelder P1 (the UN R157 baseline they show is biased). "初始條件 only".
#   B = A + the longitudinal ACTIVITY (speed/acceleration evolution toward the critical
#       frame); path still endpoint-determined (highway lane-following assumption).
#       == de Gelder's "add acceleration / activity" family (P2/P4) — the richer, better
#       highway parameterization. "有速度活動、無路徑形狀".
#   C = B + free TRAJECTORY-SHAPE parameters (NURBS shape control points).
#       == NO de Gelder analog: on highways the lateral path is a 1-parameter lane-change
#       activity, so they never parameterize free path shape. C is the extension that
#       heterogeneous, unstructured intersections REQUIRE (path is underdetermined).
# So A,B ⊆ de Gelder's parameter space; C extends beyond it — and the paper shows hetero
# traffic needs C. (de Gelder's SVD dimensionality reduction P5–7 is an orthogonal knob.)
STRATEGY_A = StrategyConfig("A", use_shape=False, constant_speed=True)   # de Gelder P1 / UN R157
STRATEGY_B = StrategyConfig("B", use_shape=False, constant_speed=False)  # de Gelder P2/P4 (activity)
STRATEGY_C = StrategyConfig("C", use_shape=True,  constant_speed=False)  # beyond de Gelder (path shape)
STRATEGIES = {"A": STRATEGY_A, "B": STRATEGY_B, "C": STRATEGY_C}


# ─────────────────────────────────────────────────────────────────────────────
# Active config — the singleton the monkeypatched sampler reads.
# generate.py sets this before invoking Stage A; sampling.py reads it.
# ─────────────────────────────────────────────────────────────────────────────
_ACTIVE_SAMPLE: SampleConfig = SAMPLE_STRATEGIES["5pt-default"]
_ACTIVE_STRATEGY: StrategyConfig | None = None


def set_active_sample_config(cfg: SampleConfig) -> None:
    global _ACTIVE_SAMPLE
    _ACTIVE_SAMPLE = cfg


def get_active_sample_config() -> SampleConfig:
    return _ACTIVE_SAMPLE


def set_active_strategy(cfg: StrategyConfig | None) -> None:
    """Set the active strategy; also aligns the active SampleConfig to it."""
    global _ACTIVE_STRATEGY
    _ACTIVE_STRATEGY = cfg
    if cfg is not None:
        set_active_sample_config(cfg.effective_sample)


def get_active_strategy() -> StrategyConfig | None:
    return _ACTIVE_STRATEGY


# Forced anchor frame — generate.py computes the anchor frame (pet / min_dist / traj_cross)
# for the (ego, actor) pair and sets it here, so the monkeypatched sampler uses the config's
# anchor EVEN in the real generation pipeline (which otherwise zeroes min_dis for non-CREEP and
# knows nothing of traj_cross). None = use the frames convert_to_yaml passes (default behavior).
_FORCED_ANCHOR_FRAME: int | None = None


def set_forced_anchor_frame(frame) -> None:
    global _FORCED_ANCHOR_FRAME
    _FORCED_ANCHOR_FRAME = None if frame is None else int(frame)


def get_forced_anchor_frame():
    return _FORCED_ANCHOR_FRAME


# Forced speed model for generation (so the exported xosc's speed event reflects the method's
# critical-frame timing): None = pipeline default; "const" = constant (start) speed over the
# whole duration (none/A); "real" = ramp timed to reach the forced anchor frame (uniform@X etc).
_FORCED_SPEED: str | None = None


def set_forced_speed(mode) -> None:
    global _FORCED_SPEED
    _FORCED_SPEED = mode


def get_forced_speed():
    return _FORCED_SPEED


# Forced target speed (km/h) for the speed event = the param agent's REAL speed at the anchor
# (critical) frame. Makes uniform@pet vs uniform@min_dis differ by END SPEED (not just duration),
# so anchored methods stay distinct even when both durations hit the 0.1s floor. None = keep the
# pipeline default End (window max speed).
_FORCED_SPEED_VALUE: float | None = None


def set_forced_speed_value(v) -> None:
    global _FORCED_SPEED_VALUE
    _FORCED_SPEED_VALUE = None if v is None else float(v)


def get_forced_speed_value():
    return _FORCED_SPEED_VALUE


# Forced speed-event frame — DECOUPLES the speed event's critical frame from the sampling
# (CP position) anchor: generate.py sets this from its `speed_anchor` argument so e.g. CP
# positions can anchor at traj_cross while the speed ramp still targets the minPET frame.
# None = the speed event uses the forced ANCHOR frame (legacy coupled behavior).
_FORCED_SPEED_FRAME: int | None = None


def set_forced_speed_frame(frame) -> None:
    global _FORCED_SPEED_FRAME
    _FORCED_SPEED_FRAME = None if frame is None else int(frame)


def get_forced_speed_frame():
    return _FORCED_SPEED_FRAME
