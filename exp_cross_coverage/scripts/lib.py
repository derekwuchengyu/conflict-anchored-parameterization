"""Shared paths + helpers for exp_cross_coverage.

Read-only reuse of sr-tlkeep-experiment (pipeline_lib, data, generated,
descriptors) and hetero-param internals. All NEW artifacts stay inside
this project (esmini_runs/, results/, figs/).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SR = Path("/home/hcis-s19/Documents/ChengYu/sr-tlkeep-experiment")
HP = Path("/home/hcis-s19/Documents/ChengYu/hetero-param")
sys.path.insert(0, str(SR / "scripts"))
os.chdir(HP)   # sr-tlkeep convention: render/descriptor stages run from hetero-param

import pipeline_lib as PL  # noqa: E402  (pulls in hetero-param sys.path)
from hetero_param import esmini_exec as EX  # noqa: E402

# subset switch (CC_SUBSET env). sr-tlkeep subsets: tlkeep (default) | keeptl
# | keeptl_sw | cutinr. Project-local subsets (extracted by 110_extract.py):
# agenttl (label 1, all dirs) | cutinl (8) | hookturn (10) | uturn (geometric)
# | special_39_180
NAME = os.environ.get("CC_SUBSET", "tlkeep")
LABEL = {"tlkeep": 2, "keeptl": 1, "keeptl_sw": 1, "cutinr": 7,
         "agenttl": 1, "cutinl": 8, "hookturn": 10, "uturn": 0,
         "special_39_180": 8}.get(NAME, 1)   # single-scenario subsets: meta
                                             # label wins; 1 = benign fallback
SUF = "" if NAME == "tlkeep" else f"_{NAME}"   # suffix for per-subset outputs
FPS = 30.0
_LOCAL_DDIR = EXP / "data" / NAME
DDIR = _LOCAL_DDIR if _LOCAL_DDIR.exists() else SR / "data" / NAME
GDIR = SR / "generated" / NAME       # ours / svd_kde parquets (sr subsets only)
_SR_RDESC = SR / "results" / f"{NAME}_descriptors.parquet"
RDESC = _SR_RDESC if _SR_RDESC.exists() or NAME == "tlkeep" \
    else EXP / "results" / f"real_desc_{NAME}.parquet"
RUNS = EXP / "esmini_runs"
RESULTS = EXP / "results"
FIGS = EXP / "figs"

METHOD = "ours"                      # petq3 = minPET anchor + quartiles
LV = ("-", "0", "+")                 # axis position min/mid/max (grid order)

# tags of the 5 combos already rendered by sr-tlkeep (reused, not re-rendered)
REUSE = {"o0e0d0": "base", "o-e0d0": "of-", "o+e0d0": "of+",
         "o0e-d0": "en-", "o0e+d0": "en+"}


def cross_plan(base: Path):
    """All 27 {min,mid,max}^3 override dicts for one base xosc.
    Returns (prefix, {tag: overrides}) with tags o{-0+}e{-0+}d{-0+}."""
    bp = EX._base_params(base)
    prefix = "Pedestrian1" if "Pedestrian1_Offset" in bp else "Agent1"
    axes = EX.pipeline_axes(bp, prefix)
    names = list(axes)               # [Offset, EndSpeed|Delay, Duration|TA_Offset]
    plan = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                tag = f"o{LV[i]}e{LV[j]}d{LV[k]}"
                plan[tag] = {names[0]: axes[names[0]][i],
                             names[1]: axes[names[1]][j],
                             names[2]: axes[names[2]][k]}
    return prefix, plan


def teleport_info(g, fps: float = FPS):
    """Max single-step position-derived speed vs esmini state speed, IGNORING
    the first 10 steps: applying a lane Offset teleports the agent laterally in
    one frame right at trajectory start (|Offset|·fps m/s — benign, before any
    interaction), whereas the corrupting post-route-end junction snaps happen
    mid/late trajectory (446_407: steps at 41–98%), inside the scored window."""
    import numpy as np
    t = g.frame.values.astype(float) / fps
    dt = np.diff(t)
    m = dt > 0
    ps = np.hypot(np.diff(g.x.values), np.diff(g.y.values))[m] / dt[m]
    ps = ps[10:]
    if len(ps) == 0:
        return 0.0, 0.0
    return float(ps.max()), float(g.speed.max())


def is_teleport(g, fps: float = FPS):
    ps_max, st_max = teleport_info(g, fps)
    return ps_max > 30.0 and ps_max > st_max + 10.0


def render_tag_local(base: Path, ego: int, actor: int, mf: int, xf: int,
                     tag: str, overrides: dict):
    """Like PL.render_tag but xosc/csv cached under THIS project's esmini_runs."""
    run_dir = RUNS / base.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    xosc = run_dir / f"{tag}.xosc"
    csv = run_dir / f"{tag}.csv"
    if not csv.exists() or not xosc.exists():
        EX.to_esmini_replay(base, PL.DATASET, ego, actor, mf, xf, xosc,
                            overrides, agent_replay=PL.AGENT_REPLAY[METHOD])
    return EX.run_and_extract(xosc, PL.DATASET, ego, actor, mf, xf, csv)
