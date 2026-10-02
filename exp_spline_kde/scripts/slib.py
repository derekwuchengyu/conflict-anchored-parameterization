"""Shared paths + helpers for exp_spline_kde.

新方法:B-Spline / NURBS 控制點參數化 + KDE(spline control-point KDE)。
每個 tlkeep 場景 base xosc 的 NURBS 有 4 個 weight=5 的 WorldPosition
sample 點(petq3:minPET anchor + 弧長四分位)— 控制點即「操控轉折座標」。
把它們的 (x,y) 當參數(spline_geo,8 維),或再加上速度曲線旋鈕
Agent1_Speed / Agent1_1_SA_EndSpeed / Agent1_1_SA_DynamicDuration
(spline_full,11 維),照 de Gelder baseline 的統計規格(LOO-CV bandwidth、
dependent sampling)擬 KDE、抽新控制點,寫回 xosc esmini render。

Read-only reuse of exp_cross_coverage (lib/teleport/baseline results),
sr-tlkeep-experiment (data/generated/core) and hetero-param; all NEW
artifacts stay inside this project (esmini_runs/, results/, figs/).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
XC = Path("/home/hcis-s19/Documents/ChengYu/exp_cross_coverage")
sys.path.insert(0, str(XC / "scripts"))
import lib as CL  # noqa: E402  (chdirs to hetero-param, imports PL/EX)

SR, HP = CL.SR, CL.HP
sys.path.insert(0, str(SR))          # for core.{kde_sampling,svd_param,wasserstein}
PL, EX = CL.PL, CL.EX

NAME, LABEL, FPS = CL.NAME, CL.LABEL, CL.FPS
METHOD = CL.METHOD                   # 'ours' = petq3 (same base xosc as exp_cross_coverage)
DDIR, GDIR, RDESC = CL.DDIR, CL.GDIR, CL.RDESC
XRES = XC / "results"                # baseline cross_descriptors / teleport_excluded
RUNS = EXP / "esmini_runs"
RESULTS = EXP / "results"
FIGS = EXP / "figs"

teleport_info, is_teleport = CL.teleport_info, CL.is_teleport

# parameter columns: 4 NURBS shape CPs (x,y) + speed-curve knobs
CP_COLS = [f"cp{i}{a}" for i in range(1, 5) for a in ("x", "y")]
KNOB_COLS = ["Agent1_Speed", "Agent1_1_SA_EndSpeed",
             "Agent1_1_SA_DynamicDuration"]
COLS = CP_COLS + KNOB_COLS
VARIANTS = {"spline_geo": 8, "spline_full": 11}      # KDE dimensionality
SEEDS = {"spline_geo": 20260807, "spline_full": 20260806}
NW = 10000
SID_PREFIX = {"spline_geo": "cpg", "spline_full": "cpf"}


def weighted_cps(xosc: Path):
    """The weighted (shape) NURBS ControlPoints of the base xosc, document
    order. Start/heading-fix/end CPs carry no weight attribute and are
    excluded — this is the same filter patch_weights/shifted_base rely on."""
    tree = ET.parse(xosc)
    pts = [cp for cp in tree.getroot().iter("ControlPoint")
           if cp.get("weight") is not None]
    return tree, pts


def cp_xy(xosc: Path) -> np.ndarray:
    """(4, 2) world coords of the 4 weighted sample points."""
    _, pts = weighted_cps(xosc)
    if len(pts) != 4:
        raise RuntimeError(f"{xosc.name}: {len(pts)} weighted CPs (expected 4)")
    out = []
    for cp in pts:
        wp = cp.find(".//WorldPosition")
        out.append([float(wp.get("x")), float(wp.get("y"))])
    return np.asarray(out)


def patched_base(base: Path, cps: np.ndarray, out: Path) -> Path:
    """Copy of the base xosc with the 4 weighted WorldPosition CPs replaced
    by `cps` (4, 2). Knots/weights untouched (same CP count → same knots)."""
    tree, pts = weighted_cps(base)
    if len(pts) != 4:
        raise RuntimeError(f"{base.name}: {len(pts)} weighted CPs (expected 4)")
    for cp, (x, y) in zip(pts, cps):
        wp = cp.find(".//WorldPosition")
        wp.set("x", f"{float(x):.4f}")
        wp.set("y", f"{float(y):.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out)
    return out
