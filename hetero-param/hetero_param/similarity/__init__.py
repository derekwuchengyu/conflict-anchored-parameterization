"""
hetero_param.similarity — modular similarity suite (real vs parameterized scenario).

Three orthogonal modules, one shared data model:

  core.py              Traj (frame/x/y/heading/speed/accel + fps + bbox),
                       real_traj / param_traj loaders, list_scenarios
  path_sim.py          路徑相似:  DTW, ADE (+ FDE, path_dev)
  interaction_sim.py   互動相似:  PET, TTC, min-distance-in-motion, arrival state at
                       the interaction point {speed, accel(decel), heading}
  distribution_sim.py  分布相似度: per-category |PET| distribution of the default-value
                       render vs real-world (KS, Wasserstein, JSD)

Quick use (from hetero-param/, nps env):
  from hetero_param import similarity as SIM
  SIM.path_similarity("HetroD", ego=301, actor=303, min_frame=4509, max_frame=4836)
  SIM.interaction_similarity("HetroD", 301, 303, 4509, 4836)
  SIM.full_similarity("HetroD", 301, 303, 4509, 4836)     # both, one NURBS build
  SIM.distribution_similarity("HetroD", per_class=30)     # per-category DataFrame
"""
from __future__ import annotations

from .. import config as C
from . import core, path_sim, interaction_sim, distribution_sim
from .core import Traj, real_traj, param_traj, list_scenarios          # noqa: F401

# canonical entry points, one per similarity family
path_similarity = path_sim.compare
interaction_similarity = interaction_sim.compare
distribution_similarity = distribution_sim.compare


def full_similarity(dataset: str, ego: int, actor: int, min_frame: int, max_frame: int,
                    sample_cfg: C.SampleConfig | None = None, estimator: str = "bbox",
                    end_offset: float = 0.0) -> dict | None:
    """Path + interaction similarity for one scenario, building each trajectory once.
    Returns one flat dict: path metrics (ade/fde/dtw/path_dev) + {real_/param_/d_}
    interaction descriptors. None if a trajectory is unavailable."""
    a = core.real_traj(dataset, actor, min_frame, max_frame)
    e = core.real_traj(dataset, ego, min_frame, max_frame)
    p = core.param_traj(dataset, ego, actor, min_frame, max_frame,
                        sample_cfg, end_offset=end_offset)
    if a is None or e is None or p is None:
        return None
    out = path_sim.similarity(a.xy, p.xy)
    rd = interaction_sim.descriptors(a, e, estimator)
    pd_ = interaction_sim.descriptors(p, e, estimator)
    out.update({f"real_{k}": v for k, v in rd.items()})
    out.update({f"param_{k}": v for k, v in pd_.items()})
    out.update(interaction_sim.similarity(rd, pd_, a.fps))
    return out
