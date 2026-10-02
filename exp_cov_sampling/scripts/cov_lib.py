"""exp_cov_sampling — 4-param cp3 取樣涵蓋實驗的共用 lib。

問題:現行參數化 = 3 個速度/位移 xosc 參數 (Agent1_Offset / SA_EndSpeed /
SA_DynamicDuration) + critical point (crit CP 的 taper 位移,法向 Crit_Lat、
切向 Crit_Long,expression param)。固定每場景 27 render 預算下,哪種取樣
方式最能涵蓋真實軌跡與真實互動?

六 subset 沿用 exp_cross_coverage v5 pinned config(anchor combo / d / W8),
base xosc 直接重用 exp_cross_coverage 的 _xosc_base_* gen cache。
所有 render 進 exp_cov_sampling/esmini_runs/{subset}/{base_stem}/{tag}.xosc。
"""
from __future__ import annotations

import hashlib
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/hcis-s19/Documents/ChengYu")
EXP = ROOT / "exp_cov_sampling"
CC = ROOT / "exp_cross_coverage"
SR = ROOT / "sr-tlkeep-experiment"
RUNS = EXP / "esmini_runs"
TEMPLATES = RUNS / "_templates"
R = EXP / os.environ.get("COV_RESULTS_DIR", "results")
FIGS = EXP / "figs"
FPS = 30.0
DATASET = "HetroD"

os.environ.setdefault("CC_SUBSET", "tlkeep")  # CCL 的 subset 常數不使用,只借函式
sys.path.insert(0, str(CC / "scripts"))
import lib as CCL  # noqa: E402  (side effects: chdir(hetero-param), sys.path+=SR/scripts)
import pipeline_lib as PL  # noqa: E402
from hetero_param import esmini_exec as EX  # noqa: E402

import xml.etree.ElementTree as ET  # noqa: E402

# ── subset 表(v5 pinned;README exp_cross_coverage 2026-08-15)────────────
ANCH = {"p": "pet", "m": "min_dist", "x": "traj_cross"}
SUBSETS = {
    #                    mode      d   P     F     conflict-W
    "cutinl":            dict(mode="anchor", d=10, P="x", F="x", w=None),
    "keeptl":            dict(mode="anchor", d=10, P="x", F="m", w=None),
    "keeptl_sw":         dict(mode="legacy", d=10, P=None, F=None, w=None),
    "special_39_180":    dict(mode="anchor", d=10, P="p", F="x", w=None),
    "special_1786_1797": dict(mode="legacy", d=5,  P=None, F=None, w=None),
    "uturn_859_881":     dict(mode="anchor", d=5,  P="x", F="x", w=8.0),
    # cutinr(2026-08-19 ours-coverage 新增):d=±5 依 120 step_m sweep,
    # anchor xx 借 cutinl(cut-in 家族;cutinr 無 anchor sweep)
    "cutinr":            dict(mode="anchor", d=5,  P="x", F="x", w=None),
}
MULTI = ["cutinl", "keeptl", "keeptl_sw"]
SPECIALS = ["special_39_180", "special_1786_1797", "uturn_859_881"]
# gen fallback label(照 exp_cross_coverage lib.py 慣例;77/88 不能餵 gen)
FALLBACK_LABEL = {"cutinl": 8, "cutinr": 7, "keeptl": 1, "keeptl_sw": 1, "special_39_180": 8}

STRATS = ["grid_legacy", "grid_crit", "frac4", "lhs4", "sobol4"]
SCOL = {  # 圖用色(real=ink #33322e 依 house style;CVD-validated)
    "grid_legacy": "#a1541c", "grid_crit": "#d99000", "frac4": "#2a78d6",
    "lhs4": "#0f8a60", "sobol4": "#7a4fbf", "oracle": "#c23b3b",
}
SLAB = {
    "grid_legacy": "grid 27 (Off×EndSpd×Dur)", "grid_crit": "grid 27 (EndSpd×Dur×CritLat)",
    "frac4": "frac 3^(4-1) (4 axes)", "lhs4": "LHS 27 (4 axes)",
    "sobol4": "Sobol 27 (4 axes)", "oracle": "greedy oracle 27",
}

CRIT_TAPER = (0.7, 1.0, 0.7)   # crit-d / crit / crit+d(wide taper 截到 3 CP)
CRIT_RANGE = 2.0               # Crit_Lat / Crit_Long ±2 m(probe/cp3d10 慣例)
AXES4 = ("e", "d", "clat", "clong")  # 連續策略的 4 軸(Offset 死旋鈕,固定 base)
PARAM_NAMES = {"o": "Agent1_Offset", "e": "Agent1_1_SA_EndSpeed",
               "d": "Agent1_1_SA_DynamicDuration",
               "clat": "Crit_Lat", "clong": "Crit_Long",
               "t1": "Theta1", "t2": "Theta2",
               "ta": "Agent1_1_TA_Offset"}


def base_dir(name: str) -> Path:
    c = SUBSETS[name]
    if c["mode"] == "anchor":
        return CC / "esmini_runs" / f"_xosc_base_anch_{c['P']}{c['F']}_s{c['d']}"
    return CC / "esmini_runs" / ("_xosc_base_cp3" if c["d"] == 10
                                 else f"_xosc_base_cp3s{c['d']}")


# user 2026-08-19:1669_1657(停等 63 s,duration 離群)自 cutin 移除
EXCLUDE = {"cutinl": {"1669_1657"}}


def data_dir(name: str) -> Path:
    p = CC / "data" / name
    return p if p.exists() else SR / "data" / name


def meta(name: str) -> pd.DataFrame:
    m = pd.read_csv(data_dir(name) / "real_meta.csv")
    ex = EXCLUDE.get(name, set())
    return m[~m.scenario_id.isin(ex)].reset_index(drop=True)


def real_tracks(name: str) -> pd.DataFrame:
    t = pd.read_parquet(data_dir(name) / "real_tracks.parquet")
    ex = EXCLUDE.get(name, set())
    return t[~t.scenario_id.isin(ex)].reset_index(drop=True)


def real_desc(name: str) -> pd.DataFrame:
    p = CC / "results" / f"real_desc_{name}_g2.parquet"
    if not p.exists():
        p = R / f"real_desc_{name}_g2.parquet"
    d = pd.read_parquet(p)
    if "error" in d.columns:
        d = d[d.error.isna()]
    d = d[d.set == "real"] if "set" in d.columns else d
    d = d[~d.scenario_id.isin(EXCLUDE.get(name, set()))]
    return d.set_index("scenario_id")


def find_base(name: str, ego: int, actor: int, mf: int) -> Path | None:
    hits = sorted(base_dir(name).glob(f"*_{ego}_{actor}_f{mf + 1}.xosc"))
    return hits[0] if hits else None


# ── template:surgery 一次 + crit expression param + (uturn) W8 ──────────

def _expr(base: float, cl: float, ct: float) -> str:
    s = f"${{{base:.4f}"
    for coef, pname in ((cl, "Crit_Long"), (ct, "Crit_Lat")):
        s += f" {'+' if coef >= 0 else '-'} {abs(coef):.6f}*${pname}"
    return s + "}"


def template_path(name: str, base_stem: str) -> Path:
    return TEMPLATES / f"{name}__{base_stem}.xosc"


def cp_geometry(base: Path):
    """base xosc(凍結字面值)的 3 weighted CP 座標 + 每點切向/法向。"""
    root = ET.parse(base).getroot()
    cps = [cp for cp in root.iter("ControlPoint") if cp.get("weight") is not None]
    if len(cps) != 3:
        return None
    wps = [cp.find(".//WorldPosition") for cp in cps]
    xy = np.array([[float(w.get("x")), float(w.get("y"))] for w in wps])
    T, N = [], []
    for i in range(3):
        t = xy[min(i + 1, 2)] - xy[max(i - 1, 0)]
        t = t / (np.hypot(*t) + 1e-9)
        T.append(t)
        N.append([-t[1], t[0]])
    return xy, np.array(T), np.array(N)


def build_template(name: str, ego: int, actor: int, mf: int, xf: int,
                   base: Path, out: Path, taper=CRIT_TAPER) -> bool:
    """回傳 crit_ok(恰 3 個 weighted CP → crit 軸有效)。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    EX.to_esmini_replay(base, DATASET, ego, actor, mf, xf, tmp, None,
                        agent_replay=False)
    tree = ET.parse(tmp)
    root = tree.getroot()
    cps = [cp for cp in root.iter("ControlPoint") if cp.get("weight") is not None]
    crit_ok = len(cps) == 3
    if crit_ok:
        wps = [cp.find(".//WorldPosition") for cp in cps]
        xy = np.array([[float(w.get("x")), float(w.get("y"))] for w in wps])
        for i, w in enumerate(wps):
            t = xy[min(i + 1, 2)] - xy[max(i - 1, 0)]
            nrm = np.hypot(*t) + 1e-9
            t = t / nrm
            n_ = np.array([-t[1], t[0]])
            f = taper[i]
            w.set("x", _expr(xy[i, 0], f * t[0], f * n_[0]))
            w.set("y", _expr(xy[i, 1], f * t[1], f * n_[1]))
        cw = SUBSETS[name]["w"]
        if cw:
            cps[1].set("weight", f"{cw:g}")
    pdecls = root.find(".//ParameterDeclarations")
    for pname in ("Crit_Long", "Crit_Lat"):
        e = ET.SubElement(pdecls, "ParameterDeclaration")
        e.set("name", pname)
        e.set("parameterType", "double")
        e.set("value", "0.0")
    tree.write(out)
    tmp.unlink()
    return crit_ok


# ── 軸與取樣 ───────────────────────────────────────────────────────────────

def axes_for(template: Path) -> dict[str, list[float]]:
    bp = EX._base_params(template)
    ax = EX.pipeline_axes(bp, "Agent1")
    return {"o": ax["Agent1_Offset"], "e": ax["Agent1_1_SA_EndSpeed"],
            "d": ax["Agent1_1_SA_DynamicDuration"],
            "clat": [-CRIT_RANGE, 0.0, CRIT_RANGE],
            "clong": [-CRIT_RANGE, 0.0, CRIT_RANGE]}


def mid_point(axes: dict) -> dict:
    return {"o": axes["o"][1], "e": axes["e"][1], "d": axes["d"][1],
            "clat": 0.0, "clong": 0.0}


def tag_of(pt: dict) -> str:
    key = ",".join(f"{round(float(pt[k]), 4):.4f}"
                   for k in ("o", "e", "d", "clat", "clong"))
    return "p" + hashlib.md5(key.encode()).hexdigest()[:8]


def strategy_points(axes: dict, sid: str) -> dict[str, list[dict]]:
    from scipy.stats import qmc
    o3, e3, d3 = axes["o"], axes["e"], axes["d"]
    cl3, cg3 = axes["clat"], axes["clong"]
    mid = mid_point(axes)
    out = {}
    out["grid_legacy"] = [{**mid, "o": o, "e": e, "d": d}
                          for o in o3 for e in e3 for d in d3]
    out["grid_crit"] = [{**mid, "e": e, "d": d, "clat": c}
                        for e in e3 for d in d3 for c in cl3]
    out["frac4"] = [{**mid, "e": e3[a], "d": d3[b], "clat": cl3[c],
                     "clong": cg3[(a + b + c) % 3]}
                    for a in range(3) for b in range(3) for c in range(3)]
    lo = np.array([min(e3), min(d3), -CRIT_RANGE, -CRIT_RANGE])
    hi = np.array([max(e3), max(d3), CRIT_RANGE, CRIT_RANGE])
    seed = zlib.crc32(sid.encode()) & 0xFFFFFFFF
    for strat, sampler in (("lhs4", qmc.LatinHypercube(d=4, seed=seed)),
                           ("sobol4", qmc.Sobol(d=4, scramble=True, seed=seed))):
        n = 27 if strat == "lhs4" else 32
        u = sampler.random(n)[:27]
        pts = qmc.scale(u, lo, hi)
        out[strat] = [{**mid, "e": p[0], "d": p[1], "clat": p[2], "clong": p[3]}
                      for p in pts]
    return out


OAT_AXES = ("o", "e", "d", "clat", "clong")
OAT_FRACS = (-1.0, -0.5, 0.5, 1.0)


def oat_points(axes: dict) -> list[tuple[str, float, dict]]:
    """(axis, frac, point);另加 ("base", 0, mid)。dur<2 的 mid 夾回範圍。"""
    mid = mid_point(axes)
    out = [("base", 0.0, dict(mid))]
    for ax in OAT_AXES:
        lo, hi = min(axes[ax]), max(axes[ax])
        m = float(np.clip(mid[ax], lo, hi))
        for frac in OAT_FRACS:
            v = m + frac * ((hi - m) if frac > 0 else (m - lo))
            out.append((ax, frac, {**mid, ax: float(v)}))
    return out


# ── render 一筆(thread-safe;esmini 是 subprocess)──────────────────────

def render_tag(name: str, template: Path, ego: int, actor: int, mf: int,
               xf: int, tag: str, pt: dict):
    run_dir = RUNS / name / template.stem.split("__", 1)[1]
    run_dir.mkdir(parents=True, exist_ok=True)
    xosc = run_dir / f"{tag}.xosc"
    csv = run_dir / f"{tag}.csv"
    if not xosc.exists() or not csv.exists():
        tree = ET.parse(template)
        want = {PARAM_NAMES[k]: float(v) for k, v in pt.items()}
        for pd_ in tree.getroot().iter("ParameterDeclaration"):
            nm = pd_.get("name")
            if nm in want:
                pd_.set("value", repr(want[nm]))
        tree.write(xosc)
    return EX.run_and_extract(xosc, DATASET, ego, actor, mf, xf, csv)


# ── 軌跡/描述子共用 ────────────────────────────────────────────────────────

from run_label_lib import trim_lead_still, arc_resample, wrap180  # noqa: E402,F401
from hetero_param.sweep import core as SWC  # noqa: E402,F401
from hetero_param.similarity.core import Traj  # noqa: E402


def traj_of(g: pd.DataFrame) -> Traj:
    t = g.frame.values.astype(float) / FPS
    sp = g.speed.values.astype(float)
    return Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
                y=g.y.values.astype(float),
                heading=g.heading_deg.values.astype(float), speed=sp,
                accel=np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp),
                fps=FPS, length=float(g.length.iloc[0]),
                width=float(g.width.iloc[0]), meta={})


is_teleport = CCL.is_teleport
teleport_info = CCL.teleport_info


def dtw2(a: np.ndarray, b: np.ndarray) -> float:
    """length-normalized 2D DTW(100_similarity_comparison 慣例)。"""
    na, nb = len(a), len(b)
    C = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        Ci = C[i - 1]
        row = D[i]
        prev = D[i - 1]
        for j in range(1, nb + 1):
            row[j] = Ci[j - 1] + min(prev[j], row[j - 1], prev[j - 1])
    return float(D[na, nb] / (na + nb))


def resample_path(g: pd.DataFrame, npt: int = 50) -> np.ndarray | None:
    x, y = g.x.values.astype(float), g.y.values.astype(float)
    if len(x) < 2:
        return None
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    if s[-1] <= 1e-6:
        return None
    u = np.linspace(0.0, s[-1], npt)
    return np.column_stack([np.interp(u, s, x), np.interp(u, s, y)])


def window_path(g: pd.DataFrame, cxy: tuple[float, float],
                half_m: float = 20.0) -> pd.DataFrame:
    """±20 m 弧長窗(fig11 慣例):以離 real conflict point 最近點為中心。"""
    x, y = g.x.values.astype(float), g.y.values.astype(float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    i = int(np.argmin(np.hypot(x - cxy[0], y - cxy[1])))
    m = (s >= s[i] - half_m) & (s <= s[i] + half_m)
    return g.iloc[m]
