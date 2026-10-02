"""
similarity.core — shared data model + loaders for the similarity suite.

INPUT REPRESENTATION
  `Traj` = one road user's motion over a frame window, in the same world frame that
  parampath / geometry already use:
    frame   (N,)  dataset frame numbers (float)
    x, y    (N,)  position (m)
    heading (N,)  raw dataset heading (deg, as stored in the tracks parquet)
    speed   (N,)  m/s
    accel   (N,)  m/s^2, signed LONGITUDINAL (deceleration < 0)
    fps           dataset frame rate (HetroD 30, inD 25 — from paths.DATASETS)
    length,width  bbox (m) — required by the bbox PET estimator

LOADERS
  real_traj(dataset, track_id, min_frame, max_frame)
      the recorded track from the raw tracks parquet (heading/velocity/accel columns
      used directly; accel projected onto the velocity direction => signed longitudinal).
  param_traj(dataset, ego, actor, min_frame, max_frame, sample_cfg)
      the analytic NURBS challenge path (parampath.parameterized_path — the exact path
      esmini would follow) TIME-parameterized by the REAL actor's arc-length-vs-frame
      (interaction.time_parameterize) => same frames as the real actor, real speed
      profile, parameterized geometry. Kinematics derived by finite differences.
  list_scenarios(dataset, ...)
      labeled (ego, actor, frame-window) scenarios tagged with superclass /
      scenario_type / agent_class — the iteration unit for distribution_sim.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import paths, config as C, parampath as PP, superclass as SC
from ..interaction import time_parameterize


@dataclass
class Traj:
    frame: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    accel: np.ndarray
    fps: float
    length: float = float("nan")
    width: float = float("nan")
    meta: dict = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    def __len__(self) -> int:
        return len(self.frame)


_raw_cache: dict[str, pd.DataFrame] = {}


def _raw_tracks(dataset: str) -> pd.DataFrame:
    """Raw levelX-style tracks parquet, indexed by trackId (has heading/width/length)."""
    if dataset not in _raw_cache:
        df = pd.read_parquet(paths.dataset_of(dataset)["tracks_parquet"])
        _raw_cache[dataset] = df.set_index("trackId", drop=False)
    return _raw_cache[dataset]


def real_traj(dataset: str, track_id: int, min_frame: int, max_frame: int) -> Traj | None:
    """The real recorded track over [min_frame, max_frame] (None if <2 rows)."""
    fps = paths.dataset_of(dataset)["fps"]
    df = _raw_tracks(dataset)
    try:
        d = df.loc[[int(track_id)]]
    except KeyError:
        return None
    d = d[(d["frame"] >= min_frame) & (d["frame"] <= max_frame)]
    d = d.sort_values("frame").drop_duplicates("frame")
    if len(d) < 2:
        return None
    vx, vy = d["xVelocity"].values, d["yVelocity"].values
    speed = np.hypot(vx, vy)
    if "xAcceleration" in d.columns and "yAcceleration" in d.columns:
        # longitudinal (signed) accel = a . v_hat ; deceleration < 0
        denom = np.where(speed < 1e-6, 1e-6, speed)
        accel = (d["xAcceleration"].values * vx + d["yAcceleration"].values * vy) / denom
    else:
        accel = np.gradient(speed, d["frame"].values / fps)
    return Traj(frame=d["frame"].values.astype(float),
                x=d["xCenter"].values.astype(float), y=d["yCenter"].values.astype(float),
                heading=d["heading"].values.astype(float), speed=speed.astype(float),
                accel=np.asarray(accel, float), fps=fps,
                length=float(np.nanmedian(d["length"].values)),
                width=float(np.nanmedian(d["width"].values)),
                meta={"dataset": dataset, "track_id": int(track_id), "kind": "real"})


def derive_kinematics(frame: np.ndarray, x: np.ndarray, y: np.ndarray, fps: float):
    """(heading_deg, speed, accel) by finite differences — for synthetic paths."""
    t = np.asarray(frame, float) / fps
    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    dx, dy = np.gradient(x), np.gradient(y)
    speed = np.hypot(dx, dy) / dt
    heading = np.degrees(np.arctan2(dy, dx))
    accel = np.gradient(speed) / dt
    return heading, speed, accel


def param_traj(dataset: str, ego: int, actor: int, min_frame: int, max_frame: int,
               sample_cfg: C.SampleConfig | None = None, end_offset: float = 0.0,
               n_samples: int = 250) -> Traj | None:
    """The parameterized challenge agent: NURBS geometry under `sample_cfg`
    (default 5pt-default = the pipeline's default-value render), timed by the real
    actor's speed profile so temporal metrics (PET/TTC/arrival state) are defined."""
    if sample_cfg is None:
        sample_cfg = C.SAMPLE_STRATEGIES["5pt-default"]
    real = real_traj(dataset, actor, min_frame, max_frame)
    if real is None:
        return None
    curve = PP.parameterized_path(dataset, ego, actor, min_frame, max_frame,
                                  sample_cfg, n_samples=n_samples, end_offset=end_offset)
    px, py = time_parameterize(curve, real.x, real.y)
    heading, speed, accel = derive_kinematics(real.frame, px, py, real.fps)
    return Traj(frame=real.frame.copy(), x=np.asarray(px, float), y=np.asarray(py, float),
                heading=heading, speed=speed, accel=accel, fps=real.fps,
                length=real.length, width=real.width,
                meta={"dataset": dataset, "track_id": int(actor), "kind": "param",
                      "ego": int(ego), "sample_cfg": sample_cfg})


def list_scenarios(dataset: str, labels=None, per_class: int | None = None,
                   group_key: str = "superclass", limit: int | None = None) -> list[dict]:
    """Labeled scenarios as dicts with ego/actor/min_frame/max_frame/label plus the
    three category views (superclass, scenario_type, agent_class). `per_class` caps
    the count per `group_key` so every category is represented."""
    paths.add_import_paths()
    import parameterization_fidelity as PF
    lab = PF.load_labeled_scenarios(paths.dataset_of(dataset)["data_id"])
    rows = []
    for (ego, actor, minf), meta in lab.items():
        if labels and meta["label"] not in labels:
            continue
        rows.append(dict(dataset=dataset, ego=ego, actor=actor, min_frame=minf,
                         max_frame=meta["max_frame"], label=meta["label"],
                         superclass=SC.superclass_of(meta["label"]),
                         scenario_type=SC.scenario_type(meta["label"]),
                         agent_class=SC.agent_behavior(meta["label"])))
    rows.sort(key=lambda r: (r[group_key], r["ego"], r["actor"], r["min_frame"]))
    if per_class:
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            c = r[group_key]
            if seen.get(c, 0) >= per_class:
                continue
            seen[c] = seen.get(c, 0) + 1
            kept.append(r)
        rows = kept
    return rows[:limit] if limit else rows
