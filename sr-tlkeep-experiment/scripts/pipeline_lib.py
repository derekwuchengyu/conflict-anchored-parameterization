"""Shared pipeline helpers: generate base xosc + esmini-render base/variants
for the three rendered methods, reusing hetero-param internals.

Methods
- sakura     : matrix method "none" (SAKURA / UN R157 baseline) — start/end-only
               NURBS, constant average speed (speed_model="const").
- ours_pos   : SAMPLE_STRATEGIES["5pt-default"] — critical position = minPET
               anchor + quartiles, legacy real-speed event.
- ours_frame : exp_esmini_frame "c-pet" choice — replay geometry
               (agent_replay=True), EndSpeed/Duration set by the minPET frame.

Variant scheme (range extremes, one-at-a-time, 4 per scenario):
- sakura / ours_pos : {prefix}_Offset at {min,max}, {prefix}_1_SA_EndSpeed at
  {min,max} of the pipeline param.xosc ranges (pipeline_axes).
- ours_frame        : EndSpeed at {min,max}, DynamicDuration at {min,max} of
  speed_axes around the minPET-frame centre (Offset is inert under replay
  geometry).
"""
from __future__ import annotations

import sys
from pathlib import Path

HP = Path("/home/hcis-s19/Documents/ChengYu/hetero-param")
sys.path.insert(0, str(HP))

import numpy as np  # noqa: E402

from hetero_param import esmini_exec as EX  # noqa: E402
from hetero_param import generate as GEN  # noqa: E402
from hetero_param import config as C  # noqa: E402
from hetero_param import parampath as PP  # noqa: E402
from hetero_param.similarity import core as SIMC  # noqa: E402
from hetero_param.config import SampleConfig  # noqa: E402

DATASET = "HetroD"
FPS = 30.0
LABEL = 2

CFGS = {
    "sakura": SampleConfig("none", placement="none", n_points=0),
    "ours_pos": C.SAMPLE_STRATEGIES["5pt-default"],
    "ours_frame": SampleConfig("framex", anchor="pet", placement="none", n_points=0),
    # corrected "ours": start/end + ONE minPET point + 3 arc-length quartiles
    # (25/50/75%); speed event = legacy real timing anchored at the minPET frame
    "ours": SampleConfig("petq3", anchor="pet", placement="anchor+uniform",
                         n_points=1),
}
STRAT = {"sakura": "srexp-none", "ours_pos": "srexp-5pt",
         "ours_frame": "srexp-framex", "ours": "srexp-petq3"}
SPEED_MODEL = {"sakura": "const", "ours_pos": None, "ours_frame": None,
               "ours": None}
AGENT_REPLAY = {"sakura": False, "ours_pos": False, "ours_frame": True,
                "ours": False}


def gen_base(method: str, ego: int, actor: int, mf: int, xf: int,
             label: int = LABEL) -> Path | None:
    """Generate (or reuse) the pipeline base .xosc for one scenario+method."""
    out_dir = EX.ESMINI_DIR / "xosc_base" / STRAT[method]
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = list(out_dir.glob(f"*_{ego}_{actor}_f{mf + 1}.xosc"))
    if hits:
        return hits[0]
    return GEN.generate(DATASET, ego, [actor], [], mf, xf, label, out_dir,
                        sample_cfg=CFGS[method], speed_model=SPEED_MODEL[method])


def frame_center_overrides(ego: int, actor: int, mf: int, xf: int, prefix: str):
    """ours_frame centre: EndSpeed/Duration implied by the minPET frame.
    None if the pet frame is unavailable/outside the window."""
    pet_f, _mdf, _tcf = PP.anchor_frames(DATASET, ego, actor, mf, xf)
    if pet_f is None or not (mf < pet_f < xf):
        return None
    real_a = SIMC.real_traj(DATASET, actor, mf, xf)
    if real_a is None:
        return None
    k = int(np.argmin(np.abs(real_a.frame - pet_f)))
    end_speed = float(real_a.speed[k]) * 3.6
    duration = max((pet_f - real_a.frame[0]) / real_a.fps, 0.1)
    return {f"{prefix}_1_SA_EndSpeed": end_speed,
            f"{prefix}_1_SA_DynamicDuration": float(duration)}


def _minmax_variants(axes: dict, names: list[str], center: dict | None):
    """4 one-at-a-time variants: each named param at its axis min and max."""
    out = {}
    abbrev = {n: n.split("_")[-1][:2].lower() for n in names}
    for n in names:
        lo, hi = min(axes[n]), max(axes[n])
        for v, sgn in ((lo, "-"), (hi, "+")):
            ov = dict(center) if center else {}
            ov[n] = v
            out[f"{abbrev[n]}{sgn}"] = ov
    return out


def variant_plan(method: str, base: Path, ego: int, actor: int, mf: int, xf: int):
    """Return (center_overrides | None, {tag: overrides}) or None (skip)."""
    bp = EX._base_params(base)
    prefix = "Pedestrian1" if "Pedestrian1_Offset" in bp else "Agent1"
    if method in ("sakura", "ours_pos", "ours"):
        axes = EX.pipeline_axes(bp, prefix)
        if prefix == "Agent1":
            names = [f"{prefix}_Offset", f"{prefix}_1_SA_EndSpeed"]
        else:
            names = ["Pedestrian1_Offset", "Pedestrian1_1_Delay"]
        return None, _minmax_variants(axes, names, None)
    # ours_frame
    ov = frame_center_overrides(ego, actor, mf, xf, prefix)
    if ov is None:
        return None
    S = ov[f"{prefix}_1_SA_EndSpeed"]
    sstep = max(min(int(S * 0.5), 20), 10)
    d = ov[f"{prefix}_1_SA_DynamicDuration"]
    dstep = min(int(d * 0.5), 2)
    dvals = [2.0, d, 4.0] if d < 2 else [max(0.0, d - dstep), d, d + dstep]
    axes = {f"{prefix}_1_SA_EndSpeed": [max(0.0, S - sstep), S, S + sstep],
            f"{prefix}_1_SA_DynamicDuration": dvals}
    names = [f"{prefix}_1_SA_EndSpeed", f"{prefix}_1_SA_DynamicDuration"]
    return ov, _minmax_variants(axes, names, ov)


def render_tag(method: str, base: Path, ego: int, actor: int, mf: int, xf: int,
               tag: str, overrides: dict | None):
    """Surgery + esmini + Traj extraction for one variant. Cached by tag."""
    run_dir = EX.ESMINI_DIR / "runs" / STRAT[method] / base.stem
    xosc = run_dir / f"{tag}.xosc"
    csv = run_dir / f"{tag}.csv"
    if not csv.exists() or not xosc.exists():
        EX.to_esmini_replay(base, DATASET, ego, actor, mf, xf, xosc, overrides,
                            agent_replay=AGENT_REPLAY[method])
    return EX.run_and_extract(xosc, DATASET, ego, actor, mf, xf, csv)


def trajs_to_rows(res: dict, scenario_id: str, method: str, tag: str):
    """Flatten run_and_extract output into long-format rows."""
    rows = []
    for role in ("ego", "agent"):
        tr = res.get(role)
        if tr is None:
            continue
        for i in range(len(tr.frame)):
            rows.append((scenario_id, method, tag, role, float(tr.frame[i]),
                         float(tr.x[i]), float(tr.y[i]),
                         float(tr.heading[i]), float(tr.speed[i]),
                         float(tr.length), float(tr.width)))
    return rows


ROW_COLUMNS = ["scenario_id", "method", "tag", "role", "frame", "x", "y",
               "heading_deg", "speed", "length", "width"]
