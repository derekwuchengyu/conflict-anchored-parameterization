"""HD1 + EXP-F: three-layer validity (traffic / map / physics) + ego outcome,
ANALYTIC, on tlkeep-298.

Arms (count-matched ~5/scenario where possible; actual counts reported):
  ours        cached esmini runs  hetero-param/results/esmini/runs/srexp-petq3
              interior tags (base, en+, en-, of+, of-) = primary count-matched set;
              corner tags (cc++, cc+-, cc-+, cc--) reported as arm "ours_corner".
  sakura      srexp-none cached runs (base + en+/- + of+/-).
  svd_kde     generated/svd_kde/trajectories_fidelity.parquet, first 5 sample ids
              per kernel-centre scenario (deterministic sort). Single stored draw.
  svd_d3      generated/svd_d3/trajectories.parquet (1 recon per scenario).
  real        the REAL actor pushed through the IDENTICAL loop = zero reference
              (Delta-above-real correction; ~7% of scenarios have bbox-noise
              overlap with background motorcycles).

Per-variant layers
  (a) TRAFFIC  variant vs every OTHER track overlapping the labeled window
      (ego + actor excluded).  All scored trajectories (variants AND the real
      actor) are resampled to the common integer-frame grid (dt = 1/30 s) with
      the SAME finite-difference heading estimation (similarity.core.
      derive_kinematics); BACKGROUND tracks keep their RECORDED parquet heading
      (identical for every arm, so no arm bias; FD heading degenerates for the
      23% stationary background).  Cheap min-centre-distance gate, then
      hetero_param.sweep.redundancy.collision_flag (OBB SAT).  Vehicles-only
      PRIMARY; pedestrians (parquet dims 0x0) get nominal 0.5x0.5 m and a
      SEPARATE column set.  Moving (< 1 m net displacement = stationary) vs
      stationary strata.  Intrusion = min centre dist < background width.
  (b) MAP      off-road flag, drivable union + scoring thresholds imported from
      scripts/33_e7_offroad.py (buffer 0.5 m, flag if > 5% of points outside).
  (c) PHYSICS  per actor class (car / motorcycle / truck) dataset P99.9 limits
      of a_lat = v^2*kappa, |a_lon|, |jerk| from ALL real tracks of that class.
      Estimator (identical for limits and flags, documented):
        1. linear-interp x,y onto a uniform dt = 0.1 s grid;
        2. centred moving-average smoothing, window 7 samples (0.7 s);
        3. v, heading by finite differences (np.gradient); a_lon = dv/dt;
           jerk = d(a_lon)/dt;
        4. kappa = |x'y'' - y'x''| / (x'^2+y'^2)^1.5, masked where v < 0.5 m/s;
           a_lat = v^2 * kappa.
      PRE-REGISTERED physics primary = a_lat violation rate.
      NOTE the |a_lon|/jerk limits are inflated by drone-tracking position noise
      in the real tracks (estimator-identical on both sides, so the gate is
      LENIENT, not biased): they stay secondary endpoints.
  (d) EGO      collision_flag vs the REAL ego replay (recorded track, all arms
      score against the same reference) + criticality band
      min_TTC < 1.5 s OR |PET| < 1 s (bbox PET via hetero_param
      interaction_sim, estimator-identical to the GT parquet; --no-pet falls
      back to a TTC-only band).

Extra pass: +-25% window-shift anchoring sensitivity for the MOVING vehicle
stratum of the KDE arms (svd_kde, svd_d3).

Outputs
  results/validity_layers_tlkeep.csv    one row per variant (+ real rows)
  results/validity_layers_summary.csv   method x layer rates, Delta-above-real,
                                        strata, per-second, per-class, shifts
  results/validity_layers_stats.csv     scenario-paired sign-flip permutation vs
                                        ours + cluster-bootstrap 95% CIs (Holm
                                        within each layer family)

Usage: micromamba run -n nps python 41_validity_layers.py [--limit N] [--no-pet]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
EXP = SCRIPTS.parent
HP = Path("/home/hcis-s19/Documents/ChengYu/hetero-param")
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(HP))

from hetero_param.similarity import core as SIMC          # noqa: E402
from hetero_param.similarity import interaction_sim as ISIM  # noqa: E402
from hetero_param.sweep import redundancy_mod as RED      # noqa: E402
from hetero_param import stats as HSTATS                  # noqa: E402

HDATA = Path("/home/hcis-s19/Documents/ChengYu/HetroD-labeler/data")
RUNS = HP / "results" / "esmini" / "runs"
FPS = 30.0
DT_PHYS = 0.1          # physics-estimator grid
SMOOTH_WIN = 7         # 0.7 s centred moving average
V_MIN_KAPPA = 0.5      # m/s — curvature masked below (jitter blow-up guard)
P_LIMIT = 99.9         # class percentile for physics limits
PED_DIM = 0.5          # nominal pedestrian bbox (m)
STATIONARY_M = 1.0     # net displacement below => stationary background
TTC_K, PET_G = 1.5, 1.0
KDE_PER_CENTER = 5
OURS_INTERIOR = ("base", "en+", "en-", "of+", "of-")
OURS_CORNER = ("cc++", "cc+-", "cc-+", "cc--")
SAKURA_TAGS = OURS_INTERIOR
SEED = 20260718
N_BOOT = 10000

VEH_CLASSES = ("car", "motorcycle", "truck", "bicycle")


# ── shared estimators ────────────────────────────────────────────────────────

def smooth(v: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    if len(v) < win:
        return v
    k = np.ones(win) / win
    pad = np.concatenate([np.full(win // 2, v[0]), v, np.full(win // 2, v[-1])])
    return np.convolve(pad, k, mode="valid")


def physics_profile(frame: np.ndarray, x: np.ndarray, y: np.ndarray):
    """(a_lat, |a_lon|, |jerk|) sample arrays on the documented dt=0.1 s grid."""
    t = np.asarray(frame, float) / FPS
    if t[-1] - t[0] < 1.0:
        return None
    grid = np.arange(t[0], t[-1] + 1e-9, DT_PHYS)
    if len(grid) < SMOOTH_WIN + 2:
        return None
    xs = smooth(np.interp(grid, t, np.asarray(x, float)))
    ys = smooth(np.interp(grid, t, np.asarray(y, float)))
    dx, dy = np.gradient(xs, DT_PHYS), np.gradient(ys, DT_PHYS)
    v = np.hypot(dx, dy)
    ddx, ddy = np.gradient(dx, DT_PHYS), np.gradient(dy, DT_PHYS)
    a_lon = np.gradient(v, DT_PHYS)
    jerk = np.gradient(a_lon, DT_PHYS)
    denom = (dx * dx + dy * dy) ** 1.5
    kappa = np.where(v >= V_MIN_KAPPA,
                     np.abs(dx * ddy - dy * ddx) / np.where(denom < 1e-9, 1e-9, denom),
                     0.0)
    return v * v * kappa, np.abs(a_lon), np.abs(jerk)


def grid_traj(frame, x, y, lo, hi, length, width):
    """Resample onto the common integer-frame grid clipped to [lo, hi] with
    FD heading/speed (the arm-uniform estimation basis). None if < 3 frames."""
    frame = np.asarray(frame, float)
    order = np.argsort(frame)
    frame, x, y = frame[order], np.asarray(x, float)[order], np.asarray(y, float)[order]
    g0 = int(np.ceil(max(frame.min(), lo)))
    g1 = int(np.floor(min(frame.max(), hi)))
    if g1 - g0 < 2:
        return None
    grid = np.arange(g0, g1 + 1, dtype=float)
    xi, yi = np.interp(grid, frame, x), np.interp(grid, frame, y)
    h, v, a = SIMC.derive_kinematics(grid, xi, yi, FPS)
    return SIMC.Traj(frame=grid, x=xi, y=yi, heading=h % 360.0, speed=v, accel=a,
                     fps=FPS, length=float(length), width=float(width))


# ── background cache ─────────────────────────────────────────────────────────

class Tracks:
    def __init__(self):
        df = pd.read_parquet(HDATA / "00_tracks.parquet")
        df = df.sort_values(["trackId", "frame"])
        self.cls = {int(k): v for k, v in
                    json.load(open(HDATA / "00_trackid_class.json")).items()}
        self.by_id = {}
        for tid, g in df.groupby("trackId"):
            self.by_id[int(tid)] = dict(
                frame=g["frame"].values.astype(float),
                x=g["xCenter"].values.astype(float),
                y=g["yCenter"].values.astype(float),
                heading=g["heading"].values.astype(float),
                length=float(np.nanmedian(g["length"].values)),
                width=float(np.nanmedian(g["width"].values)))
        self.default_dims = {}
        for c in VEH_CLASSES:
            ls = [t["length"] for i, t in self.by_id.items() if self.cls.get(i) == c]
            ws = [t["width"] for i, t in self.by_id.items() if self.cls.get(i) == c]
            self.default_dims[c] = (float(np.median(ls)), float(np.median(ws)))
        self.default_dims["pedestrian"] = (PED_DIM, PED_DIM)

    def clip(self, tid: int, lo: int, hi: int):
        t = self.by_id[tid]
        m = (t["frame"] >= lo) & (t["frame"] <= hi)
        if m.sum() < 3:
            return None
        return t["frame"][m], t["x"][m], t["y"][m], t["heading"][m]

    def backgrounds(self, ego: int, actor: int, lo: int, hi: int):
        """List of background dicts clipped to the window."""
        out = []
        for tid, t in self.by_id.items():
            if tid in (ego, actor):
                continue
            if t["frame"][-1] < lo or t["frame"][0] > hi:
                continue
            c = self.clip(tid, lo, hi)
            if c is None:
                continue
            f, x, y, h = c
            cls = self.cls.get(tid, "car")
            ped = cls == "pedestrian"
            L, W = (PED_DIM, PED_DIM) if ped else (t["length"], t["width"])
            if not ped and (not np.isfinite(L) or L <= 0):
                L, W = self.default_dims.get(cls, self.default_dims["car"])
            moving = float(np.hypot(x[-1] - x[0], y[-1] - y[0])) >= STATIONARY_M
            out.append(dict(
                tid=tid, cls=cls, ped=ped, moving=moving,
                traj=SIMC.Traj(frame=f, x=x, y=y, heading=h,
                               speed=np.zeros_like(f), accel=np.zeros_like(f),
                               fps=FPS, length=float(L), width=float(W)),
                diag=float(np.hypot(L, W)) / 2.0))
        return out


# ── traffic layer ────────────────────────────────────────────────────────────

def _pair_min_dist(v: SIMC.Traj, b: SIMC.Traj):
    lo = max(v.frame[0], b.frame[0])
    hi = min(v.frame[-1], b.frame[-1])
    if hi - lo < 2:
        return None
    grid = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1, dtype=float)
    vx, vy = np.interp(grid, v.frame, v.x), np.interp(grid, v.frame, v.y)
    bx, by = np.interp(grid, b.frame, b.x), np.interp(grid, b.frame, b.y)
    return float(np.hypot(vx - bx, vy - by).min())


def score_traffic(var: SIMC.Traj, bgs: list, subset=None) -> dict:
    """Variant vs every background: gate on min centre distance, SAT on the rest."""
    vdiag = float(np.hypot(var.length, var.width)) / 2.0
    r = dict(bg_collision=False, bg_n_hits=0, bg_min_dist=np.inf,
             bg_intrusion=False, bg_first_collision_frame=np.nan,
             bg_collision_moving=False, bg_collision_stationary=False,
             bg_intrusion_moving=False, bg_intrusion_stationary=False,
             ped_collision=False, ped_min_dist=np.inf, ped_intrusion=False,
             n_bg_vehicles=0, n_bg_peds=0)
    for bg in bgs:
        if subset is not None and not subset(bg):
            continue
        md = _pair_min_dist(var, bg["traj"])
        if md is None:
            continue
        if bg["ped"]:
            r["n_bg_peds"] += 1
            r["ped_min_dist"] = min(r["ped_min_dist"], md)
            if md < PED_DIM:
                r["ped_intrusion"] = True
            if md <= vdiag + bg["diag"] + 0.2:
                if RED.collision_flag(var, bg["traj"])["collision"]:
                    r["ped_collision"] = True
            continue
        r["n_bg_vehicles"] += 1
        r["bg_min_dist"] = min(r["bg_min_dist"], md)
        stratum = "moving" if bg["moving"] else "stationary"
        if md < bg["traj"].width:
            r["bg_intrusion"] = True
            r[f"bg_intrusion_{stratum}"] = True
        if md <= vdiag + bg["diag"] + 0.2:
            cf = RED.collision_flag(var, bg["traj"])
            if cf["collision"]:
                r["bg_collision"] = True
                r[f"bg_collision_{stratum}"] = True
                r["bg_n_hits"] += 1
                ff = cf["first_frame"]
                if not np.isfinite(r["bg_first_collision_frame"]) or \
                        ff < r["bg_first_collision_frame"]:
                    r["bg_first_collision_frame"] = ff
    for k in ("bg_min_dist", "ped_min_dist"):
        if not np.isfinite(r[k]):
            r[k] = np.nan
    return r


# ── physics limits from the real dataset ─────────────────────────────────────

def physics_limits(tracks: Tracks, classes) -> dict:
    lims = {}
    for c in classes:
        alat, alon, jerk = [], [], []
        n = 0
        for tid, t in tracks.by_id.items():
            if tracks.cls.get(tid) != c:
                continue
            prof = physics_profile(t["frame"], t["x"], t["y"])
            if prof is None:
                continue
            alat.append(prof[0]); alon.append(prof[1]); jerk.append(prof[2])
            n += 1
        lims[c] = dict(
            a_lat=float(np.percentile(np.concatenate(alat), P_LIMIT)),
            a_lon=float(np.percentile(np.concatenate(alon), P_LIMIT)),
            jerk=float(np.percentile(np.concatenate(jerk), P_LIMIT)),
            n_tracks=n)
    return lims


def score_physics(var: SIMC.Traj, lim: dict) -> dict:
    prof = physics_profile(var.frame, var.x, var.y)
    if prof is None:
        return dict(alat_max=np.nan, alon_max=np.nan, jerk_max=np.nan,
                    alat_viol=False, alon_viol=False, jerk_viol=False,
                    physics_scored=False)
    alat, alon, jerk = (float(p.max()) for p in prof)
    return dict(alat_max=alat, alon_max=alon, jerk_max=jerk,
                alat_viol=alat > lim["a_lat"], alon_viol=alon > lim["a_lon"],
                jerk_viol=jerk > lim["jerk"], physics_scored=True)


# ── map layer ────────────────────────────────────────────────────────────────

def load_offroad():
    spec = importlib.util.spec_from_file_location("e7", SCRIPTS / "33_e7_offroad.py")
    e7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e7)
    from shapely import vectorized
    area = e7.drivable_union()

    def score(var: SIMC.Traj) -> dict:
        inside = vectorized.contains(area, var.x, var.y)
        frac = 1.0 - float(inside.mean())
        return dict(offroad_frac=frac, offroad_flag=frac > e7.OUT_FRAC)
    return score


# ── ego layer ────────────────────────────────────────────────────────────────

def score_ego(var: SIMC.Traj, ego: SIMC.Traj, use_pet: bool) -> dict:
    cf = RED.collision_flag(var, ego)
    tt = ISIM.ttc(var, ego)
    p = np.nan
    if use_pet:
        try:
            p = ISIM.pet(var, ego, estimator="bbox")["pet"]
        except Exception:
            p = np.nan
    crit = bool(cf["collision"]) or \
        (np.isfinite(tt["min_ttc"]) and tt["min_ttc"] < TTC_K) or \
        (np.isfinite(p) and abs(p) < PET_G)
    return dict(ego_collision=bool(cf["collision"]),
                ego_first_collision_frame=cf["first_frame"],
                ego_min_ttc=float(tt["min_ttc"]), ego_pet=float(p),
                ego_critical=crit)


# ── variant collection ───────────────────────────────────────────────────────

def find_run_dir(cache: str, scenario_id: str):
    root = RUNS / cache
    pat = re.compile(rf"_{scenario_id}_f(\d+)$")
    for d in root.iterdir():
        m = pat.search(d.name)
        if m:
            return d, int(m.group(1))
    return None, None


def esmini_agent(csv_path: Path, f0: int):
    """(frame, x, y) of the non-Ego entity from a cached esmini run CSV.
    Leading standstill frames (esmini spawn wait, ~0.1 s, FD speed < 0.05 m/s)
    are trimmed (up to 1 s) — the 0 -> cruise jump otherwise forges a_lon/jerk
    spikes that belong to the runner, not the generated variant."""
    from hetero_param import esmini_exec as EX
    try:
        ents = EX._parse_full_csv(csv_path)
    except Exception:
        return None
    name = next((n for n in ents if n != "Ego"), None)
    if name is None:
        return None
    d = ents[name]
    if len(d) < 3:
        return None
    t, x, y = d["t"].values, d["x"].values, d["y"].values
    v = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-9)
    j = 0
    while j < len(v) and v[j] < 0.05 and t[j] - t[0] < 1.0:
        j += 1
    return f0 + t[j:] * FPS, x[j:], y[j:]


# ── summary / stats helpers ──────────────────────────────────────────────────

def pooled(df: pd.DataFrame, col: str) -> float:
    v = df[col].astype(float)
    return float(v.mean()) if len(v) else np.nan


def cluster_boot_ci(df: pd.DataFrame, col: str, rng) -> tuple[float, float]:
    g = df.groupby("scenario_id")[col].agg(["sum", "count"])
    k, n = g["sum"].values.astype(float), g["count"].values.astype(float)
    if len(k) < 2 or n.sum() == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(k), size=(N_BOOT, len(k)))
    rates = k[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1.0)
    return (float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [np.nan] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# ── main ─────────────────────────────────────────────────────────────────────

def main(limit: int | None, use_pet: bool):
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    meta = pd.read_csv(EXP / "data" / "tlkeep" / "real_meta.csv")
    if limit:
        meta = meta.head(limit)
    print(f"[setup] tlkeep scenarios: {len(meta)}")

    tracks = Tracks()
    print(f"[setup] tracks loaded ({len(tracks.by_id)}), "
          f"class defaults: {tracks.default_dims}")

    actor_cls = {r.scenario_id: tracks.cls.get(int(r.actor), "car")
                 for r in meta.itertuples()}
    need_cls = sorted(set(actor_cls.values()))
    print(f"[setup] actor classes present: {need_cls}; computing P{P_LIMIT} limits ...")
    lims = physics_limits(tracks, need_cls)
    for c, l in lims.items():
        print(f"  {c:11s} a_lat<{l['a_lat']:.2f}  |a_lon|<{l['a_lon']:.2f} "
              f"jerk<{l['jerk']:.2f}  ({l['n_tracks']} tracks)")

    offroad = load_offroad()
    print("[setup] drivable union built")

    kde = pd.read_parquet(EXP / "generated" / "svd_kde" / "trajectories_fidelity.parquet")
    d3 = pd.read_parquet(EXP / "generated" / "svd_d3" / "trajectories.parquet")
    kde_groups = {k: g for k, g in kde.groupby("center_key")}
    d3_groups = {k: g for k, g in d3.groupby("center_key")}

    rows = []
    shift_rows = []          # anchoring-sensitivity accumulation
    missing_runs = {"ours": [], "sakura": []}
    skipped = []

    for si, r in enumerate(meta.itertuples()):
        sid = r.scenario_id
        ego_id, act_id = int(r.ego), int(r.actor)
        lo, hi = int(r.min_frame), int(r.max_frame)
        cls = actor_cls[sid]
        lim = lims[cls]
        bgs = tracks.backgrounds(ego_id, act_id, lo, hi)
        ego_traj = SIMC.real_traj("HetroD", ego_id, lo, hi)
        real_actor = tracks.by_id.get(act_id)
        if ego_traj is None or real_actor is None:
            skipped.append((sid, "no ego/actor track"))
            continue
        a_dims = (real_actor["length"], real_actor["width"])
        if not np.isfinite(a_dims[0]) or a_dims[0] <= 0:
            a_dims = tracks.default_dims.get(cls, tracks.default_dims["car"])

        variants = []   # (arm, tag, frame, x, y, length, width)

        # real zero reference — identical loop
        c = tracks.clip(act_id, lo, hi)
        if c is not None:
            f, x, y, _h = c
            variants.append(("real", "real", f, x, y, *a_dims))

        for cache, arm_tags in (("srexp-petq3", ("ours", OURS_INTERIOR + OURS_CORNER)),
                                ("srexp-none", ("sakura", SAKURA_TAGS))):
            arm, tags = arm_tags
            d, f0 = find_run_dir(cache, sid)
            if d is None:
                missing_runs[arm].append(sid)
                continue
            for tag in tags:
                p = d / f"{tag}.csv"
                if not p.exists():
                    continue
                res = esmini_agent(p, f0)
                if res is None:
                    skipped.append((sid, f"{arm}:{tag} unparseable"))
                    continue
                a = arm if (arm != "ours" or tag in OURS_INTERIOR) else "ours_corner"
                variants.append((a, tag, *res, *a_dims))

        kdims = tracks.default_dims.get(cls, tracks.default_dims["car"])
        g = kde_groups.get(sid)
        if g is not None:
            for sc in sorted(g.scenario_id.unique())[:KDE_PER_CENTER]:
                s = g[g.scenario_id == sc]
                variants.append(("svd_kde", sc, s.frame.values, s.x.values,
                                 s.y.values, *kdims))
        g = d3_groups.get(sid)
        if g is not None:
            for sc in sorted(g.scenario_id.unique())[:1]:
                s = g[g.scenario_id == sc]
                variants.append(("svd_d3", sc, s.frame.values, s.x.values,
                                 s.y.values, *a_dims))

        for arm, tag, f, x, y, L, W in variants:
            var = grid_traj(f, x, y, lo, hi, L, W)
            if var is None:
                skipped.append((sid, f"{arm}:{tag} <3 frames in window"))
                continue
            row = dict(arm=arm, scenario_id=sid, tag=str(tag), actor_class=cls,
                       duration_s=float((var.frame[-1] - var.frame[0]) / FPS))
            row.update(score_traffic(var, bgs))
            row.update(offroad(var))
            row.update(score_physics(var, lim))
            row.update(score_ego(var, ego_traj, use_pet))
            rows.append(row)

            # anchoring sensitivity: KDE arms, moving-vehicle stratum only
            if arm in ("svd_kde", "svd_d3"):
                span = hi - lo
                for sh_lab, sh in (("-25%", -0.25 * span), ("+25%", 0.25 * span)):
                    vs = grid_traj(np.asarray(f, float) + sh, x, y, lo, hi, L, W)
                    if vs is None:
                        continue
                    tr = score_traffic(vs, bgs,
                                       subset=lambda b: (not b["ped"]) and b["moving"])
                    shift_rows.append(dict(
                        arm=arm, shift=sh_lab, scenario_id=sid, tag=str(tag),
                        bg_collision_moving=tr["bg_collision_moving"],
                        bg_intrusion_moving=tr["bg_intrusion_moving"]))

        if (si + 1) % 20 == 0 or si == len(meta) - 1:
            print(f"[loop] {si + 1}/{len(meta)} scenarios "
                  f"({time.time() - t_start:.0f}s, {len(rows)} variant rows)")

    out = pd.DataFrame(rows)
    out_path = EXP / "results" / "validity_layers_tlkeep.csv"
    out.to_csv(out_path, index=False)
    print(f"[write] {out_path}  ({len(out)} rows)")

    # ── summary ──────────────────────────────────────────────────────────────
    arms = [a for a in ("ours", "ours_corner", "sakura", "svd_kde", "svd_d3", "real")
            if a in set(out.arm)]
    real_df = out[out.arm == "real"]
    flag_cols = ["bg_collision", "bg_intrusion", "bg_collision_moving",
                 "bg_collision_stationary", "bg_intrusion_moving",
                 "bg_intrusion_stationary", "ped_collision", "ped_intrusion",
                 "offroad_flag", "alat_viol", "alon_viol", "jerk_viol",
                 "ego_collision", "ego_critical"]
    summary = []
    for arm in arms:
        d = out[out.arm == arm]
        for cls in ["all"] + sorted(d.actor_class.unique()):
            dd = d if cls == "all" else d[d.actor_class == cls]
            rr = real_df if cls == "all" else real_df[real_df.actor_class == cls]
            if not len(dd):
                continue
            row = dict(arm=arm, actor_class=cls, n_variants=len(dd),
                       n_scenarios=dd.scenario_id.nunique(),
                       mean_duration_s=pooled(dd, "duration_s"))
            for c in flag_cols:
                row[f"{c}_rate"] = pooled(dd, c)
            for c in ("bg_collision", "bg_intrusion", "offroad_flag",
                      "ego_collision"):
                row[f"{c}_delta_real"] = (pooled(dd, c) - pooled(rr, c)
                                          if len(rr) else np.nan)
            dur = dd.duration_s.sum()
            row["bg_collision_per_s"] = dd.bg_collision.sum() / dur if dur else np.nan
            row["bg_hits_per_s"] = dd.bg_n_hits.sum() / dur if dur else np.nan
            if cls == "all":
                lo_ci, hi_ci = cluster_boot_ci(dd, "bg_collision", rng)
                row["bg_collision_ci_lo"], row["bg_collision_ci_hi"] = lo_ci, hi_ci
                lo_ci, hi_ci = cluster_boot_ci(dd, "alat_viol", rng)
                row["alat_viol_ci_lo"], row["alat_viol_ci_hi"] = lo_ci, hi_ci
                lo_ci, hi_ci = cluster_boot_ci(dd, "offroad_flag", rng)
                row["offroad_ci_lo"], row["offroad_ci_hi"] = lo_ci, hi_ci
            summary.append(row)

    sh = pd.DataFrame(shift_rows)
    if len(sh):
        base_mov = {a: pooled(out[out.arm == a], "bg_collision_moving")
                    for a in ("svd_kde", "svd_d3") if a in set(out.arm)}
        for (arm, s), g in sh.groupby(["arm", "shift"]):
            summary.append(dict(
                arm=f"{arm}_shift{s}", actor_class="all(moving-veh)",
                n_variants=len(g), n_scenarios=g.scenario_id.nunique(),
                bg_collision_moving_rate=pooled(g, "bg_collision_moving"),
                bg_intrusion_moving_rate=pooled(g, "bg_intrusion_moving"),
                bg_collision_moving_delta_vs_base=(
                    pooled(g, "bg_collision_moving") - base_mov.get(arm, np.nan))))

    sm = pd.DataFrame(summary)
    sm_path = EXP / "results" / "validity_layers_summary.csv"
    sm.to_csv(sm_path, index=False)
    print(f"[write] {sm_path}")

    # ── paired stats vs ours ─────────────────────────────────────────────────
    families = {"traffic": ["bg_collision", "bg_intrusion"],
                "map": ["offroad_flag"],
                "physics": ["alat_viol", "alon_viol", "jerk_viol"],
                "ego": ["ego_collision", "ego_critical"]}
    stat_rows = []
    ours_rates = {c: out[out.arm == "ours"].groupby("scenario_id")[c].mean()
                  for cs in families.values() for c in cs}
    for fam, cols in families.items():
        fam_p = []
        for col in cols:
            for arm in arms:
                if arm in ("ours", "real"):
                    continue
                a_r = out[out.arm == arm].groupby("scenario_id")[col].mean()
                common = a_r.index.intersection(ours_rates[col].index)
                diffs = (a_r.loc[common] - ours_rates[col].loc[common]).values
                obs, p = HSTATS.paired_permutation_test(diffs)
                stat_rows.append(dict(family=fam, metric=col, arm=arm,
                                      n_scenarios=len(common),
                                      mean_paired_diff_vs_ours=obs, p_raw=p))
                fam_p.append(p)
        adj = holm(fam_p)
        k = 0
        for srow in stat_rows:
            if srow["family"] == fam and "p_holm" not in srow:
                srow["p_holm"] = adj[k]
                k += 1
    st = pd.DataFrame(stat_rows)
    st_path = EXP / "results" / "validity_layers_stats.csv"
    st.to_csv(st_path, index=False)
    print(f"[write] {st_path}")

    # ── console report ───────────────────────────────────────────────────────
    show = ["arm", "actor_class", "n_variants", "n_scenarios",
            "bg_collision_rate", "bg_collision_delta_real",
            "bg_collision_moving_rate", "bg_collision_stationary_rate",
            "bg_collision_per_s", "bg_intrusion_rate", "ped_collision_rate",
            "offroad_flag_rate", "alat_viol_rate", "alon_viol_rate",
            "jerk_viol_rate", "ego_collision_rate", "ego_critical_rate"]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\n=== validity layers summary (tlkeep-298) ===")
        cols = [c for c in show if c in sm.columns]
        print(sm[sm.actor_class.isin(["all", "all(moving-veh)"])][
            [c for c in sm.columns if c in cols or c.startswith("bg_collision_moving")]
        ].to_string(index=False))
        print("\n=== paired sign-flip permutation vs ours ===")
        print(st.to_string(index=False))

    print(f"\nvariant counts per arm:\n{out.groupby('arm').size().to_string()}")
    print(f"missing run dirs: ours={sorted(set(missing_runs['ours']))} "
          f"sakura={sorted(set(missing_runs['sakura']))}")
    if skipped:
        print(f"skipped items ({len(skipped)}): {skipped[:20]}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="first N scenarios (smoke test only — full run reports all)")
    ap.add_argument("--no-pet", action="store_true",
                    help="skip bbox PET (TTC-only criticality band)")
    main(ap.parse_args().limit, use_pet=not ap.parse_args().no_pet)
