"""End-to-end final-spec analysis for one scenario label.

Stages: extract(+direction filter) -> render sakura/ours -> SVD/KDE ->
descriptors (spawn-trimmed) -> fidelity (count-matched KDE + Nw sensitivity)
-> variety & coverage (signed PET) -> figures.

Usage (from anywhere, nps env):
  python 20_run_label.py --label 1 --name keeptl --direction n2e
  python 20_run_label.py --label 7 --name cutinr --direction all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
EXP = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EXP))
import pipeline_lib as PL  # noqa: E402
from core.svd_param import SvdParameterization, split_vector  # noqa: E402
from core.kde_sampling import loo_bandwidth, sample_dependent  # noqa: E402
from core.wasserstein import empirical_wasserstein  # noqa: E402
from hetero_param.sweep import core as SWC  # noqa: E402
from hetero_param.similarity.core import Traj  # noqa: E402

HDATA = Path("/home/hcis-s19/Documents/ChengYu/HetroD-labeler/data")
FPS = 30.0
NT, NY, NTH = 50, 2, 1
D = 3
NW_FID = 10000
SEED = 20260803
TRIM_SPEED, TRIM_MAX_S = 0.05, 0.6
METHODS = ("sakura", "ours")
VTAGS = ("of-", "of+", "en-", "en+")
RNG = np.random.default_rng(SEED)

COLS = ["scenario_id", "method", "tag", "role", "frame", "x", "y",
        "heading_deg", "speed", "center_key", "duration_s"]


def circ_mean_deg(a):
    r = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))) % 360.0


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def trim_lead_still(g: pd.DataFrame) -> pd.DataFrame:
    sp = g.speed.values
    mv = np.flatnonzero(sp > TRIM_SPEED)
    if len(mv) == 0 or mv[0] == 0:
        return g
    t = (g.frame.values - g.frame.values[0]) / FPS
    return g if t[mv[0]] > TRIM_MAX_S else g.iloc[mv[0]:]


# turn filters: (entry-heading arc, exit-heading arc), degrees CCW from +x
TURN_FILTERS = {"n2e": ((225.0, 315.0), (315.0, 45.0)),
                "s2w": ((45.0, 135.0), (135.0, 225.0))}


def _in_arc(h, lo, hi):
    return (lo <= h <= hi) if lo <= hi else (h >= lo or h <= hi)


def _direction_ok(a: pd.DataFrame, direction: str) -> bool:
    """a = actor rows (raw tracks schema) sorted by frame."""
    if direction in ("r2l", "s2n"):
        dx = a.xCenter.iloc[-1] - a.xCenter.iloc[0]
        dy = a.yCenter.iloc[-1] - a.yCenter.iloc[0]
        if direction == "r2l":      # westbound: right -> left
            return dx < 0 and abs(dx) > abs(dy)
        return dy > 0 and abs(dy) > abs(dx)   # s2n: south -> north
    (lo0, hi0), (lo1, hi1) = TURN_FILTERS[direction]
    spd = np.hypot(a.xVelocity.values, a.yVelocity.values)
    hd = a.heading.values
    k = max(5, int(0.15 * len(a)))
    m0, m1 = spd[:k] > 0.5, spd[-k:] > 0.5
    if m0.sum() < 3 or m1.sum() < 3:
        return False
    h0 = circ_mean_deg(hd[:k][m0])
    h1 = circ_mean_deg(hd[-k:][m1])
    return _in_arc(h0, lo0, hi0) and _in_arc(h1, lo1, hi1)


def arc_resample(x, y, k=50):
    """Resample a path to k points uniform in ARC LENGTH (speed-free)."""
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-9:
        return np.stack([np.full(k, x[0]), np.full(k, y[0])], 1)
    sq = np.linspace(0.0, s[-1], k)
    return np.stack([np.interp(sq, s, x), np.interp(sq, s, y)], 1)


def dtw_row(Ri: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Length-normalised 2-D DTW of one path (K,2) vs a batch (Ng,K,2)."""
    Ng, K, _ = G.shape
    C = np.linalg.norm(Ri[None, :, None, :] - G[:, None, :, :], axis=3)
    D = np.full((Ng, K + 1, K + 1), np.inf)
    D[:, 0, 0] = 0.0
    for s in range(2, 2 * K + 1):
        a0, a1 = max(1, s - K), min(K, s - 1)
        A = np.arange(a0, a1 + 1)
        B = s - A
        prev = np.minimum(np.minimum(D[:, A - 1, B - 1], D[:, A - 1, B]),
                          D[:, A, B - 1])
        D[:, A, B] = C[:, A - 1, B - 1] + prev
    return D[:, K, K] / (2.0 * K)


# ─── stage 1: extract ────────────────────────────────────────────────────────

def stage_extract(label: int, direction: str, ddir: Path):
    labels = json.loads((HDATA / "00_labeled_scenarios.json").read_text())
    tracks = pd.read_parquet(HDATA / "00_tracks.parquet")
    tracks = tracks.sort_values(["trackId", "frame"]).set_index("trackId",
                                                                drop=False)
    tracks.index.name = "tid"
    rows_out, trajs, durations, keys, meta_rows = [], [], [], [], []
    n_label = n_dir = 0
    for key, ent in labels.items():
        if ent.get("label_idx") != label:
            continue
        n_label += 1
        ego, actor = int(ent["ego_id"]), int(ent["actor_id"])
        mf, xf = int(ent["min_frame"]), int(ent["max_frame"])
        try:
            a = tracks.loc[[actor]]
            e = tracks.loc[[ego]]
        except KeyError:
            continue
        a = a[(a.frame >= mf) & (a.frame <= xf)]
        e = e[(e.frame >= mf) & (e.frame <= xf)]
        if len(a) < 10 or len(e) < 10:
            continue
        dur = float((a.frame.max() - a.frame.min()) / FPS)
        if dur <= 0.5:
            continue
        if direction != "all" and not _direction_ok(a, direction):
            continue
        n_dir += 1
        t = (a.frame.values - a.frame.values[0]) / FPS
        tq = np.linspace(t[0], t[-1], NT)
        xy = np.stack([np.interp(tq, t, a.xCenter.values),
                       np.interp(tq, t, a.yCenter.values)], axis=1)
        trajs.append(xy)
        durations.append(dur)
        keys.append(key)
        meta_rows.append({"scenario_id": key, "ego": ego, "actor": actor,
                          "min_frame": mf, "max_frame": xf, "label": label,
                          "duration_s": dur})
        for role, df in (("ego", e), ("actor", a)):
            sp = np.hypot(df.xVelocity.values, df.yVelocity.values)
            rows_out.append(pd.DataFrame({
                "scenario_id": key, "role": role, "track_id": df.trackId.values,
                "frame": df.frame.values, "x": df.xCenter.values,
                "y": df.yCenter.values, "heading_deg": df.heading.values,
                "speed": sp, "length": df.length.values,
                "width": df.width.values}))
    X_raw = np.asarray([np.concatenate([t.reshape(-1), [d]])
                        for t, d in zip(trajs, durations)])
    ddir.mkdir(parents=True, exist_ok=True)
    np.savez(ddir / "real_vectors.npz", X_raw=X_raw, keys=np.array(keys))
    pd.concat(rows_out, ignore_index=True).to_parquet(ddir / "real_tracks.parquet")
    pd.DataFrame(meta_rows).to_csv(ddir / "real_meta.csv", index=False)
    print(f"[extract] label={label}: {n_label} entries, kept after "
          f"direction '{direction}': {len(keys)}", flush=True)
    return len(keys)


# ─── stage 2: render ─────────────────────────────────────────────────────────

def _gen_one(args):
    method, ego, actor, mf, xf, label = args
    try:
        base = PL.gen_base(method, ego, actor, mf, xf, label)
        return (ego, actor, str(base) if base else None, None)
    except Exception as e:  # noqa: BLE001
        return (ego, actor, None, f"{type(e).__name__}: {e}")


def _render_one(job):
    method, sid, ego, actor, mf, xf, base, tag, ov = job
    try:
        res = PL.render_tag(method, Path(base), ego, actor, mf, xf, tag, ov)
        if res is None or res.get("agent") is None:
            return (sid, tag, None, "no agent traj")
        return (sid, tag, res, None)
    except Exception as e:  # noqa: BLE001
        return (sid, tag, None, f"{type(e).__name__}: {e}")


def stage_render(ddir: Path, gdir: Path, label: int):
    meta = pd.read_csv(ddir / "real_meta.csv")
    for method in METHODS:
        t0 = time.time()
        gen_args = [(method, int(r.ego), int(r.actor), int(r.min_frame),
                     int(r.max_frame), label) for r in meta.itertuples()]
        with Pool(6) as pool:
            gen_out = pool.map(_gen_one, gen_args, chunksize=4)
        bases = {f"{e}_{a}": b for (e, a, b, err) in gen_out if b}
        n_fail = len(gen_args) - len(bases)
        jobs = []
        for r in meta.itertuples():
            sid = f"{int(r.ego)}_{int(r.actor)}"
            if sid not in bases:
                continue
            try:
                plan = PL.variant_plan(method, Path(bases[sid]), int(r.ego),
                                       int(r.actor), int(r.min_frame),
                                       int(r.max_frame))
            except Exception:  # noqa: BLE001
                continue
            if plan is None:
                continue
            center, variants = plan
            jobs.append((method, sid, int(r.ego), int(r.actor),
                         int(r.min_frame), int(r.max_frame), bases[sid],
                         "base", center))
            for tag, ov in variants.items():
                jobs.append((method, sid, int(r.ego), int(r.actor),
                             int(r.min_frame), int(r.max_frame), bases[sid],
                             tag, ov))
        rows, n_err = [], 0
        with ThreadPoolExecutor(12) as tp:
            for (sid, tag, res, err) in tp.map(_render_one, jobs):
                if err:
                    n_err += 1
                    continue
                if tag != "base" and "ego" in res:
                    res = {"agent": res.get("agent")}
                rows.extend(PL.trajs_to_rows(res, sid, method, tag))
        out = gdir / method
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=PL.ROW_COLUMNS).to_parquet(
            out / "trajectories.parquet")
        print(f"[render:{method}] gen ok={len(bases)} fail={n_fail}, "
              f"renders={len(jobs)} err={n_err} ({time.time() - t0:.0f}s)",
              flush=True)


# ─── stage 3: SVD / KDE ─────────────────────────────────────────────────────

def _gen_traj_rows(X, ids, method, tag, center_keys, meta_i):
    rows = []
    for x, sid, ck in zip(X, ids, center_keys):
        y, theta = split_vector(x, NT, NY, NTH)
        dur = max(float(theta[0]), 0.5)
        mf = int(meta_i.loc[ck, "min_frame"])
        tq = np.linspace(0.0, dur, NT)
        vx = np.gradient(y[:, 0], tq)
        vy = np.gradient(y[:, 1], tq)
        speed = np.hypot(vx, vy)
        heading = np.degrees(np.arctan2(vy, vx)) % 360.0
        for i in range(NT):
            rows.append((sid, method, tag, "agent", mf + tq[i] * FPS,
                         y[i, 0], y[i, 1], heading[i], speed[i], ck, dur))
    return rows


def stage_svd(ddir: Path, gdir: Path, rdir: Path, name: str):
    dat = np.load(ddir / "real_vectors.npz")
    X_raw, keys = dat["X_raw"], dat["keys"]
    N = len(keys)
    meta_i = pd.read_csv(ddir / "real_meta.csv").set_index("scenario_id")
    P = SvdParameterization(NT, NY, NTH).fit(X_raw)
    ev = {d: P.explained_variance(d) for d in range(1, 8)}
    V3 = P.reduced(D)
    h = loo_bandwidth(V3)
    print(f"[svd] N={N} d3_var={ev[3]:.4f} h={h:.5f}", flush=True)
    out = gdir / "svd_d3"
    out.mkdir(parents=True, exist_ok=True)
    X_rec = P.reconstruct(V3, D)
    np.savez(out / "vectors.npz", X=X_rec, keys=keys, V=V3)
    pd.DataFrame(_gen_traj_rows(X_rec, [f"svdrec_{k}" for k in keys], "svd_d3",
                                "recon", keys, meta_i),
                 columns=COLS).to_parquet(out / "trajectories.parquet")
    boot = RNG.integers(0, N, size=N * 4)
    np.savez(out / "vectors_vc.npz", X=X_rec[boot], keys=keys[boot],
             boot_idx=boot)
    out = gdir / "svd_kde"
    out.mkdir(parents=True, exist_ok=True)
    for tag, nw in (("fidelity", NW_FID), ("vc", N * 4)):
        S, cidx = sample_dependent(V3, h, nw, RNG)
        Xg = P.reconstruct(S, D)
        np.savez(out / f"vectors_{tag}.npz", X=Xg, center_keys=keys[cidx], h=h)
        pd.DataFrame(_gen_traj_rows(Xg, [f"kde{tag[0]}_{i:05d}" for i in range(nw)],
                                    "svd_kde", tag, keys[cidx], meta_i),
                     columns=COLS).to_parquet(out / f"trajectories_{tag}.parquet")
    # count-matched fidelity subset (first N iid samples — deterministic)
    z = np.load(out / "vectors_fidelity.npz")
    Xcm, ckcm = z["X"][:N], z["center_keys"][:N]
    np.savez(out / "vectors_fidelity_cm.npz", X=Xcm, center_keys=ckcm)
    pd.DataFrame(_gen_traj_rows(Xcm, [f"kdecm_{i:05d}" for i in range(N)],
                                "svd_kde", "fid_cm", ckcm, meta_i),
                 columns=COLS).to_parquet(out / "trajectories_fid_cm.parquet")
    json.dump({"explained_variance": ev, "h_loo": h, "n_real": int(N),
               "svd_d3_mean_abs_recon_err": float(np.abs(X_rec - X_raw).mean())},
              open(rdir / f"{name}_svd_fit.json", "w"), indent=2)


# ─── stage 4: descriptors ────────────────────────────────────────────────────

G_REAL, G_GEN, G_EGO = {}, {}, {}


def _traj_df(g, L, W):
    t = g.frame.values.astype(float) / FPS
    sp = g.speed.values.astype(float)
    acc = np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp)
    return Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
                y=g.y.values.astype(float),
                heading=g.heading_deg.values.astype(float), speed=sp, accel=acc,
                fps=FPS, length=float(L), width=float(W), meta={})


def _desc_one(job):
    kind, sid, tag = job
    try:
        ck = sid
        if kind == "real":
            a = G_REAL[(sid, "actor")]
            e = G_REAL[(sid, "ego")]
            agent = SWC._derived_kinematics_copy(
                _traj_df(a, a.length.iloc[0], a.width.iloc[0]))
            ego = _traj_df(e, e.length.iloc[0], e.width.iloc[0])
        elif kind in METHODS:
            g = G_GEN[(kind, sid, tag)]
            agent = _traj_df(g, g.length.iloc[0], g.width.iloc[0])
            eg = G_EGO.get((kind, sid))
            if eg is None:
                return None
            ego = _traj_df(eg, eg.length.iloc[0], eg.width.iloc[0])
        else:
            g, ck = G_GEN[(kind, sid, tag)]
            src = G_REAL[(ck, "actor")]
            agent = _traj_df(g, src.length.iloc[0], src.width.iloc[0])
            e = G_REAL[(ck, "ego")]
            ego = _traj_df(e, e.length.iloc[0], e.width.iloc[0])
        d = SWC.descriptors(agent, ego, estimator="bbox")
        d.update({"set": kind, "scenario_id": sid, "tag": tag,
                  "center_key": ck})
        return d
    except Exception as e:  # noqa: BLE001
        return {"set": kind, "scenario_id": sid, "tag": tag,
                "error": f"{type(e).__name__}: {e}"}


def stage_descriptors(ddir: Path, gdir: Path, rdir: Path, name: str):
    global G_REAL, G_GEN, G_EGO
    G_REAL, G_GEN, G_EGO = {}, {}, {}
    real_tracks = pd.read_parquet(ddir / "real_tracks.parquet")
    for (sid, role), g in real_tracks.groupby(["scenario_id", "role"]):
        G_REAL[(sid, role)] = g.sort_values("frame")
    jobs = [("real", sid, "real") for (sid, role) in G_REAL if role == "actor"]
    for m in METHODS:
        df = pd.read_parquet(gdir / m / "trajectories.parquet")
        for (sid, tag, role), g in df.groupby(["scenario_id", "tag", "role"]):
            g = g.sort_values("frame")
            if role == "ego":
                if tag == "base":
                    G_EGO[(m, sid)] = g
            else:
                G_GEN[(m, sid, tag)] = trim_lead_still(g)
                jobs.append((m, sid, tag))
    for kind, folder, fname in (
            ("svd_d3", "svd_d3", "trajectories.parquet"),
            ("svd_kde", "svd_kde", "trajectories_vc.parquet"),
            ("svd_kde_fid", "svd_kde", "trajectories_fid_cm.parquet")):
        df = pd.read_parquet(gdir / folder / fname)
        for sid, g in df.groupby("scenario_id"):
            g = g.sort_values("frame")
            G_GEN[(kind, sid, "gen")] = (g, g.center_key.iloc[0])
            jobs.append((kind, sid, "gen"))
    print(f"[desc] {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with Pool(8) as pool:
        out = [r for r in pool.imap_unordered(_desc_one, jobs, chunksize=16) if r]
    df = pd.DataFrame(out)
    df.to_parquet(rdir / f"{name}_descriptors.parquet")
    nerr = df.error.notna().sum() if "error" in df else 0
    print(f"[desc] {len(df)} rows ({nerr} errors) in {time.time() - t0:.0f}s",
          flush=True)


# ─── stage 5: metrics ────────────────────────────────────────────────────────

def _vectors_from_renders(gdir: Path, method: str, tags, with_ids=False):
    df = pd.read_parquet(gdir / method / "trajectories.parquet")
    df = df[(df.role == "agent") & (df.tag.isin(tags))]
    vecs, ids, paths = [], [], []
    for (sid, tag), g in df.groupby(["scenario_id", "tag"]):
        g = trim_lead_still(g.sort_values("frame"))
        t = (g.frame.values - g.frame.values[0]) / FPS
        dur = float(t[-1])
        if dur < 0.5 or len(g) < 5:
            continue
        tq = np.linspace(0.0, dur, NT)
        xy = np.stack([np.interp(tq, t, g.x.values),
                       np.interp(tq, t, g.y.values)], axis=1)
        vecs.append(np.concatenate([xy.reshape(-1), [dur]]))
        ids.append(sid)
        paths.append(arc_resample(g.x.values, g.y.values, NT))
    if with_ids:
        return np.asarray(vecs), ids, np.asarray(paths)
    return np.asarray(vecs)


def stage_metrics(ddir: Path, gdir: Path, rdir: Path, name: str):
    dat = np.load(ddir / "real_vectors.npz")
    X_raw, real_keys = dat["X_raw"], list(dat["keys"])
    N = len(X_raw)
    P = SvdParameterization(NT, NY, NTH).fit(X_raw)
    Zw = P.weighted(X_raw)
    kde_all = np.load(gdir / "svd_kde" / "vectors_fidelity.npz")["X"]
    kde_cm = np.load(gdir / "svd_kde" / "vectors_fidelity_cm.npz")
    sak_v, sak_ids, sak_paths = _vectors_from_renders(gdir, "sakura", ("base",), True)
    ours_v, ours_ids, ours_paths = _vectors_from_renders(gdir, "ours", ("base",), True)
    W_sets = {
        "svd_d3": (np.load(gdir / "svd_d3" / "vectors.npz")["X"], None),
        "svd_kde": (kde_cm["X"], kde_all),
        "sakura": (sak_v, None),
        "ours": (ours_v, None),
    }
    rows = []
    for mname, (W, Wbig) in W_sets.items():
        w1 = empirical_wasserstein(Zw, P.weighted(W))
        row = {"method": mname, "n_W": len(W), "W1": w1}
        if Wbig is not None:
            row["W1_nw10000"] = empirical_wasserstein(Zw, P.weighted(Wbig))
        rows.append(row)
        print(f"[fid] {mname:8s} n={len(W):5d} W1={w1:.4f}"
              + (f"  (Nw=10000: {row.get('W1_nw10000'):.4f})" if Wbig is not None else ""),
              flush=True)
    pd.DataFrame(rows).to_csv(rdir / f"{name}_fidelity.csv", index=False)

    # ---- DTW path fidelity (arc-length resampled: geometry only, no speed) --
    real_tracks = pd.read_parquet(ddir / "real_tracks.parquet")
    rpaths, ridx = [], {}
    for i, k in enumerate(real_keys):
        g = real_tracks[(real_tracks.scenario_id == k)
                        & (real_tracks.role == "actor")].sort_values("frame")
        rpaths.append(arc_resample(g.x.values, g.y.values, NT))
        ridx[k] = i
    rpaths = np.asarray(rpaths)

    d3 = pd.read_parquet(gdir / "svd_d3" / "trajectories.parquet")
    d3_paths, d3_pairs = [], []
    for sid, g in d3.groupby("scenario_id"):
        g = g.sort_values("frame")
        d3_paths.append(arc_resample(g.x.values, g.y.values, NT))
        d3_pairs.append(sid.replace("svdrec_", ""))
    kdecm = pd.read_parquet(gdir / "svd_kde" / "trajectories_fid_cm.parquet")
    kde_paths, kde_pairs = [], []
    for sid, g in kdecm.groupby("scenario_id"):
        g = g.sort_values("frame")
        kde_paths.append(arc_resample(g.x.values, g.y.values, NT))
        kde_pairs.append(g.center_key.iloc[0])

    dtw_rows = []
    path_sets = {"svd_d3": (np.asarray(d3_paths), d3_pairs),
                 "svd_kde": (np.asarray(kde_paths), kde_pairs),
                 "sakura": (sak_paths, sak_ids),
                 "ours": (ours_paths, ours_ids)}
    for mname, (G, pairs) in path_sets.items():
        t0 = time.time()
        M = np.stack([dtw_row(rpaths[i], G) for i in range(len(rpaths))])
        import ot as _ot
        a = np.full(M.shape[0], 1.0 / M.shape[0])
        b = np.full(M.shape[1], 1.0 / M.shape[1])
        w1_dtw = float(_ot.emd2(a, b, M, numItermax=1_000_000))
        paired = np.array([M[ridx[p], j] for j, p in enumerate(pairs)
                           if p in ridx])
        dtw_rows.append({"method": mname, "n": M.shape[1],
                         "W1_dtw_m": w1_dtw,
                         "paired_dtw_median_m": float(np.median(paired)),
                         "paired_dtw_p90_m": float(np.percentile(paired, 90))})
        print(f"[dtw] {mname:8s} W1_dtw={w1_dtw:.3f} m  paired_med="
              f"{np.median(paired):.3f} m ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(dtw_rows).to_csv(rdir / f"{name}_fidelity_dtw.csv", index=False)

    # ---- interaction-point fidelity (unpaired dists + paired MAE + joint OT) -
    from scipy import stats as _st
    from scipy.spatial.distance import jensenshannon as _jsd
    desc_all = pd.read_parquet(rdir / f"{name}_descriptors.parquet")
    if "error" in desc_all.columns:
        desc_all = desc_all[desc_all.error.isna()]
    rd = desc_all[desc_all.set == "real"].set_index("scenario_id")
    base_sets = {
        "sakura": desc_all[(desc_all.set == "sakura") & (desc_all.tag == "base")],
        "svd_d3": desc_all[desc_all.set == "svd_d3"],
        "svd_kde": desc_all[desc_all.set == "svd_kde_fid"],
        "ours": desc_all[(desc_all.set == "ours") & (desc_all.tag == "base")],
    }
    IKEYS = ["pet", "pet_abs", "min_ttc", "min_dist", "conflict_angle",
             "closing_speed", "drac", "agent_arr_speed", "agent_arr_accel",
             "agent_arr_heading"]
    JKEYS = ["pet", "min_ttc", "min_dist", "conflict_angle", "closing_speed",
             "agent_arr_speed", "agent_arr_heading"]
    mu_head = circ_mean_deg(rd["agent_arr_heading"].values)
    irows = []
    for mname, dd in base_sets.items():
        pair_key = dd["center_key"] if "center_key" in dd else dd["scenario_id"]
        pk = [p.replace("svdrec_", "") for p in pair_key]
        # per-key metrics
        for key in IKEYS:
            g = dd[key].values.astype(float)
            r = rd[key].values.astype(float)
            if key == "agent_arr_heading":
                g = wrap180(g - mu_head)
                r = wrap180(r - mu_head)
            gf, rf = g[np.isfinite(g)], r[np.isfinite(r)]
            row = {"method": mname, "key": key,
                   "frac_finite": float(np.isfinite(g).mean()),
                   "real_frac_finite": float(np.isfinite(r).mean())}
            if len(gf) > 2 and len(rf) > 2:
                row["wasserstein_1d"] = float(_st.wasserstein_distance(rf, gf))
                ks = _st.ks_2samp(rf, gf)
                row["ks_stat"] = float(ks.statistic)
                row["ks_p"] = float(ks.pvalue)
                lo = min(rf.min(), gf.min())
                hi = max(rf.max(), gf.max())
                if hi > lo:
                    hr, _ = np.histogram(rf, bins=20, range=(lo, hi), density=True)
                    hg, _ = np.histogram(gf, bins=20, range=(lo, hi), density=True)
                    row["jsd"] = float(_jsd(hr + 1e-12, hg + 1e-12, base=2) ** 2)
            # paired MAE (pair with the real/kernel-centre scenario)
            diffs = []
            for j, p in enumerate(pk):
                if p not in rd.index:
                    continue
                gv, rv = g[j], float(rd.loc[p, key]) if key != "agent_arr_heading" \
                    else float(wrap180(rd.loc[p, key] - mu_head))
                if np.isfinite(gv) and np.isfinite(rv):
                    d_ = gv - rv
                    if key == "agent_arr_heading":
                        d_ = wrap180(d_)
                    diffs.append(abs(d_))
            if diffs:
                row["paired_mae_median"] = float(np.median(diffs))
                row["paired_mae_p90"] = float(np.percentile(diffs, 90))
                row["n_paired"] = len(diffs)
            irows.append(row)
        # paired conflict-point distance + arrival-time error
        cds, ats = [], []
        for j, p in enumerate(pk):
            if p not in rd.index:
                continue
            row_g = dd.iloc[j]
            row_r = rd.loc[p]
            if np.isfinite(row_g.conflict_x) and np.isfinite(row_r.conflict_x):
                cds.append(float(np.hypot(row_g.conflict_x - row_r.conflict_x,
                                          row_g.conflict_y - row_r.conflict_y)))
            tg = (row_g.agent_arr_frame - row_g.ego_arr_frame) / FPS
            tr = (row_r.agent_arr_frame - row_r.ego_arr_frame) / FPS
            if np.isfinite(tg) and np.isfinite(tr):
                ats.append(abs(float(tg - tr)))
        irows.append({"method": mname, "key": "conflict_point_dist_m",
                      "paired_mae_median": float(np.median(cds)) if cds else np.nan,
                      "paired_mae_p90": float(np.percentile(cds, 90)) if cds else np.nan,
                      "n_paired": len(cds)})
        irows.append({"method": mname, "key": "arr_time_gap_s",
                      "paired_mae_median": float(np.median(ats)) if ats else np.nan,
                      "paired_mae_p90": float(np.percentile(ats, 90)) if ats else np.nan,
                      "n_paired": len(ats)})
        # joint OT over z-scored interaction vectors (finite rows only)
        def _joint(df_):
            Vs = []
            for key in JKEYS:
                v = df_[key].values.astype(float)
                if key == "agent_arr_heading":
                    v = wrap180(v - mu_head)
                Vs.append(v)
            return np.stack(Vs, 1)
        Rj = _joint(rd.reset_index())
        Gj = _joint(dd)
        mu_ = np.nanmean(np.where(np.isfinite(Rj), Rj, np.nan), 0)
        sd_ = np.nanstd(np.where(np.isfinite(Rj), Rj, np.nan), 0)
        sd_ = np.where(sd_ < 1e-9, 1.0, sd_)
        Rz = (Rj - mu_) / sd_
        Gz = (Gj - mu_) / sd_
        rmask = np.isfinite(Rz).all(1)
        gmask = np.isfinite(Gz).all(1)
        w1j = empirical_wasserstein(Rz[rmask], Gz[gmask]) \
            if rmask.sum() > 2 and gmask.sum() > 2 else np.nan
        irows.append({"method": mname, "key": "JOINT_zscored_W1",
                      "wasserstein_1d": float(w1j),
                      "frac_finite": float(gmask.mean()),
                      "real_frac_finite": float(rmask.mean())})
        print(f"[ix ] {mname:8s} joint W1={w1j:.3f} "
              f"(finite gen {gmask.mean():.2f} / real {rmask.mean():.2f})",
              flush=True)
    pd.DataFrame(irows).to_csv(rdir / f"{name}_fidelity_interaction.csv",
                               index=False)

    # variety & coverage
    desc = pd.read_parquet(rdir / f"{name}_descriptors.parquet")
    if "error" in desc.columns:
        desc = desc[desc.error.isna()]
    real_desc = desc[desc.set == "real"]
    d3v = np.load(gdir / "svd_d3" / "vectors_vc.npz")
    d3_desc = desc[desc.set == "svd_d3"].set_index("scenario_id")
    boot_ids = [f"svdrec_{k}" for k in d3v["keys"]]
    sets_vec = {
        "sakura": _vectors_from_renders(gdir, "sakura", VTAGS),
        "svd_d3": d3v["X"],
        "svd_kde": np.load(gdir / "svd_kde" / "vectors_vc.npz")["X"],
        "ours": _vectors_from_renders(gdir, "ours", VTAGS),
    }
    sets_desc = {
        "sakura": desc[(desc.set == "sakura") & (desc.tag != "base")],
        "svd_d3": d3_desc.loc[[i for i in boot_ids if i in d3_desc.index]].reset_index(),
        "svd_kde": desc[desc.set == "svd_kde"],
        "ours": desc[(desc.set == "ours") & (desc.tag != "base")],
    }
    DESCRIPTORS = ["pet", "min_ttc", "min_dist", "conflict_angle",
                   "closing_speed", "drac", "agent_arr_speed",
                   "agent_arr_heading"]

    def wseries(X):
        return (np.asarray(X) * P.alpha)[:, : NT * NY]

    def nn_dists(A, B):
        out = np.empty(len(A))
        for i in range(0, len(A), 256):
            d = np.linalg.norm(A[i:i + 256, None, :] - B[None, :, :], axis=2)
            out[i:i + 256] = d.min(axis=1)
        return out

    def path_metrics(method, G, R):
        row = {"method": method, "aspect": "path", "n": len(G)}
        D_ = np.linalg.norm(G[:, None, :] - G[None, :, :], axis=2)
        iu = np.triu_indices(len(G), 1)
        row["diversity"] = float(D_[iu].mean())
        np.fill_diagonal(D_, np.inf)
        row["gen_nn_med"] = float(np.median(D_.min(1)))
        r2g = nn_dists(R, G)
        row["nn_med"] = float(np.median(r2g))
        DR = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=2)
        np.fill_diagonal(DR, np.inf)
        eps = float(np.median(DR.min(1)))
        row["eps_real_nn"] = eps
        row["frac_covered"] = float((r2g <= eps).mean())
        row["real_diversity"] = float(DR[np.isfinite(DR)].mean())
        return row

    def scalar_metrics(method, key, g, r):
        g = np.asarray(g, float)
        r = np.asarray(r, float)
        row = {"method": method, "aspect": key, "n": len(g)}
        if key == "agent_arr_heading":
            mu = circ_mean_deg(r[np.isfinite(r)])
            g = wrap180(g - mu)
            r = wrap180(r - mu)
        gf, rf = g[np.isfinite(g)], r[np.isfinite(r)]
        row["frac_finite"] = float(len(gf) / max(len(g), 1))
        if key == "pet":
            row["frac_pos"] = float((gf > 0).mean()) if len(gf) else np.nan
            row["real_frac_pos"] = float((rf > 0).mean()) if len(rf) else np.nan
        if len(gf) < 2 or len(rf) < 2:
            return row
        row.update(std=float(gf.std(ddof=1)),
                   iqr=float(np.subtract(*np.percentile(gf, [75, 25]))),
                   range=float(gf.max() - gf.min()),
                   real_std=float(rf.std(ddof=1)),
                   real_range=float(rf.max() - rf.min()),
                   covers_real_frac=float(((rf >= gf.min()) & (rf <= gf.max())).mean()),
                   nn_med=float(np.median(np.abs(rf[:, None] - gf[None, :]).min(axis=1))))
        lo, hi = max(gf.min(), rf.min()), min(gf.max(), rf.max())
        rr = rf.max() - rf.min()
        row["range_overlap"] = float(max(0.0, hi - lo) / rr) if rr > 1e-12 else np.nan
        return row

    rows = []
    Rw = wseries(X_raw)
    for m in ("sakura", "svd_d3", "svd_kde", "ours"):
        rows.append(path_metrics(m, wseries(sets_vec[m]), Rw))
        for key in DESCRIPTORS:
            rows.append(scalar_metrics(m, key, sets_desc[m][key].values,
                                       real_desc[key].values))
    for key in DESCRIPTORS:
        rows.append(scalar_metrics("real (ref)", key, real_desc[key].values,
                                   real_desc[key].values))
    out = pd.DataFrame(rows)
    out.to_csv(rdir / f"{name}_variety_coverage.csv", index=False)
    print(out[out.aspect.isin(["path", "pet", "agent_arr_heading"])]
          .to_string(index=False), flush=True)


# ─── stage 6: figures ────────────────────────────────────────────────────────

def stage_figures(ddir: Path, gdir: Path, rdir: Path, name: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COLORS = {"sakura": "#2a78d6", "svd_d3": "#eb6834", "svd_kde": "#1baf7a",
              "ours": "#eda100"}
    LABELS = {"sakura": "sakura (none)", "svd_d3": "SVD (d=3)",
              "svd_kde": "SVD+KDE (dep.)", "ours": "ours (minPET pos+frame)"}
    GRAY, INK = "#9a9a94", "#33322e"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": "#6f6e68", "ytick.color": "#6f6e68", "axes.grid": True,
        "grid.color": "#eceae5", "grid.linewidth": 0.8, "font.size": 10,
        "axes.titlesize": 11})
    real = pd.read_parquet(ddir / "real_tracks.parquet")
    real_agent = real[real.role == "actor"]
    sids = real_agent.scenario_id.unique()
    SUB = set(RNG.choice(sids, min(120, len(sids)), replace=False))

    def agent_paths(method):
        out = []
        if method == "real":
            for sid, g in real_agent.groupby("scenario_id"):
                if sid in SUB:
                    g = g.sort_values("frame")
                    out.append((g.x.values, g.y.values))
        elif method in METHODS:
            df = pd.read_parquet(gdir / method / "trajectories.parquet")
            df = df[(df.role == "agent") & (df.tag == "base")
                    & df.scenario_id.isin(SUB)]
            for sid, g in df.groupby("scenario_id"):
                g = trim_lead_still(g.sort_values("frame"))
                out.append((g.x.values, g.y.values))
        elif method == "svd_d3":
            df = pd.read_parquet(gdir / "svd_d3" / "trajectories.parquet")
            for sid, g in df.groupby("scenario_id"):
                if sid.replace("svdrec_", "") in SUB:
                    g = g.sort_values("frame")
                    out.append((g.x.values, g.y.values))
        else:
            # count-matched fidelity subset, subsampled to the SAME number of
            # paths as the real panel shows (fair visual density)
            df = pd.read_parquet(gdir / "svd_kde" / "trajectories_fid_cm.parquet")
            n_show = min(120, len(sids))
            pick = set(RNG.choice(df.scenario_id.unique(),
                                  min(n_show, df.scenario_id.nunique()),
                                  replace=False))
            for sid, g in df.groupby("scenario_id"):
                if sid in pick:
                    g = g.sort_values("frame")
                    out.append((g.x.values, g.y.values))
        return out

    real_paths = agent_paths("real")
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.6), sharex=True, sharey=True)
    for ax, m in zip(axes, ["real", "sakura", "svd_d3", "svd_kde", "ours"]):
        for x, y in real_paths:
            ax.plot(x, y, color=GRAY, lw=0.7, alpha=0.35, zorder=1)
        if m != "real":
            for x, y in agent_paths(m):
                ax.plot(x, y, color=COLORS[m], lw=0.8, alpha=0.45, zorder=2)
            ax.set_title(LABELS[m])
        else:
            ax.set_title(f"real (n={len(sids)}, {len(real_paths)} shown)")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.suptitle(f"{title} — agent paths: real (gray) vs generated (base sets)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(EXP / "figs" / f"{name}_fig1_trajectories.png", dpi=160)
    plt.close(fig)

    fid = pd.read_csv(rdir / f"{name}_fidelity.csv").set_index("method")
    order = [m for m in ("svd_d3", "svd_kde", "ours", "sakura")
             if m in fid.index]
    order = sorted(order, key=lambda m: fid.loc[m].W1)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, fid.loc[order].W1, height=0.55,
            color=[COLORS[m] for m in order])
    for y, m in zip(ypos, order):
        ax.text(fid.loc[m].W1 * 1.01 + 0.03, y, f"{fid.loc[m].W1:.2f}",
                va="center", fontsize=10, color=INK)
    ax.set_yticks(ypos, [LABELS[m] for m in order])
    ax.set_xlabel("empirical Wasserstein W̃₁ (α-weighted, count-matched; lower = better)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Fidelity — {title}")
    fig.tight_layout()
    fig.savefig(EXP / "figs" / f"{name}_fig2_fidelity.png", dpi=160)
    plt.close(fig)

    vc = pd.read_csv(rdir / f"{name}_variety_coverage.csv")
    methods = ["sakura", "svd_d3", "svd_kde", "ours"]
    short = ["sakura", "SVD", "SVD+KDE", "ours"]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.9, 1.1]})
    p = vc[vc.aspect == "path"].set_index("method").loc[methods]
    axes[0].bar(range(4), p.frac_covered, color=[COLORS[m] for m in methods],
                width=0.6)
    for i, v in enumerate(p.frac_covered):
        axes[0].text(i, v + 0.004, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_xticks(range(4), short, rotation=15)
    axes[0].set_ylabel("fraction of real paths covered (NN ≤ ε_real)")
    axes[0].set_title("(a) path coverage")
    axes[0].grid(axis="x", visible=False)
    axes[1].bar(range(4), p.diversity, color=[COLORS[m] for m in methods],
                width=0.6)
    axes[1].axhline(p.real_diversity.iloc[0], color=INK, lw=1.2, ls="--")
    axes[1].text(3.4, p.real_diversity.iloc[0] * 1.02, "real", fontsize=9,
                 ha="right")
    axes[1].set_xticks(range(4), short, rotation=15)
    axes[1].set_ylabel("mean pairwise distance (weighted path space)")
    axes[1].set_title("(b) path variety")
    axes[1].grid(axis="x", visible=False)
    aspects = ["pet", "min_ttc", "min_dist", "conflict_angle", "closing_speed",
               "agent_arr_speed"]
    alab = ["PET\n(signed)", "TTC", "min dist", "conflict\nangle",
            "closing\nspeed", "arrival\nspeed"]
    xa = np.arange(len(aspects))
    for j, m in enumerate(methods):
        dd = vc[vc.method == m].set_index("aspect")
        ratio = [dd.loc[a, "std"] / dd.loc[a, "real_std"] for a in aspects]
        axes[2].plot(xa + (j - 1.5) * 0.11, ratio, "o", ms=6.5,
                     color=COLORS[m], label=LABELS[m])
    axes[2].axhline(1.0, color=INK, lw=1.2, ls="--")
    axes[2].set_xticks(xa, alab, fontsize=8.5)
    axes[2].set_ylabel("std ratio (generated / real)")
    axes[2].set_title("(c) descriptor variety (1 = same spread)")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="x", visible=False)
    hh = vc[vc.aspect == "agent_arr_heading"].set_index("method")
    vals = [hh.loc[m, "std"] for m in methods]
    axes[3].bar(range(4), vals, color=[COLORS[m] for m in methods], width=0.6)
    rstd = hh.loc["sakura", "real_std"]
    axes[3].axhline(rstd, color=INK, lw=1.2, ls="--")
    axes[3].text(3.4, rstd * 1.02, f"real {rstd:.1f}°", fontsize=9, ha="right")
    for i, v in enumerate(vals):
        axes[3].text(i, v * 1.01, f"{v:.1f}°", ha="center", fontsize=9)
    axes[3].set_xticks(range(4), short, rotation=15)
    axes[3].set_ylabel("arrival heading std [deg]")
    axes[3].set_title("(d) arrival heading spread")
    axes[3].grid(axis="x", visible=False)
    fig.suptitle(f"Variety & coverage — {title}", y=1.0)
    fig.tight_layout()
    fig.savefig(EXP / "figs" / f"{name}_fig3_variety_coverage.png", dpi=160)
    plt.close(fig)

    # ---- fig4: coverage map (edge cases) + conflict span + NN profile ------
    dat = np.load(ddir / "real_vectors.npz")
    X_raw, real_keys = dat["X_raw"], list(dat["keys"])
    P = SvdParameterization(NT, NY, NTH).fit(X_raw)

    def wseries(X):
        return (np.asarray(X) * P.alpha)[:, : NT * NY]

    Rw = wseries(X_raw)
    vc_vecs = {
        "sakura": _vectors_from_renders(gdir, "sakura", VTAGS),
        "svd_d3": np.load(gdir / "svd_d3" / "vectors_vc.npz")["X"],
        "svd_kde": np.load(gdir / "svd_kde" / "vectors_vc.npz")["X"],
        "ours": _vectors_from_renders(gdir, "ours", VTAGS),
    }
    DR = np.linalg.norm(Rw[:, None, :] - Rw[None, :, :], axis=2)
    np.fill_diagonal(DR, np.inf)
    eps = float(np.median(DR.min(1)))
    nn = {}
    for m, G in vc_vecs.items():
        Gw = wseries(G)
        d_ = np.empty(len(Rw))
        for i in range(0, len(Rw), 256):
            d_[i:i + 256] = np.linalg.norm(
                Rw[i:i + 256, None, :] - Gw[None, :, :], axis=2).min(1)
        nn[m] = d_

    desc_all = pd.read_parquet(rdir / f"{name}_descriptors.parquet")
    if "error" in desc_all.columns:
        desc_all = desc_all[desc_all.error.isna()]
    rcp = desc_all[desc_all.set == "real"][["conflict_x", "conflict_y"]].values
    cp = {"sakura": desc_all[(desc_all.set == "sakura") & (desc_all.tag != "base")],
          "svd_d3": desc_all[desc_all.set == "svd_d3"],
          "svd_kde": desc_all[desc_all.set == "svd_kde"],
          "ours": desc_all[(desc_all.set == "ours") & (desc_all.tag != "base")]}

    real_paths_by_key = {}
    for sid, g in real_agent.groupby("scenario_id"):
        g = g.sort_values("frame")
        real_paths_by_key[sid] = (g.x.values, g.y.values)

    fig = plt.figure(figsize=(19, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.0], hspace=0.3)
    n_worst = max(3, int(round(0.10 * len(real_keys))))
    all_rx = np.concatenate([v[0] for v in real_paths_by_key.values()])
    all_ry = np.concatenate([v[1] for v in real_paths_by_key.values()])
    xlim = (all_rx.min() - 8, all_rx.max() + 8)
    ylim = (all_ry.min() - 8, all_ry.max() + 8)
    for j, m in enumerate(methods):
        ax = fig.add_subplot(gs[0, j])
        unc = nn[m] > eps
        worst = set(np.argsort(-nn[m])[:n_worst])
        for i, k in enumerate(real_keys):
            if i in worst:
                continue
            x, y = real_paths_by_key[k]
            ax.plot(x, y, color="#b9b8b2", lw=0.8, alpha=0.5, zorder=1)
        for i in worst:
            x, y = real_paths_by_key[real_keys[i]]
            ax.plot(x, y, color="#e34948", lw=1.3, alpha=0.9, zorder=3)
        c = cp[m][["conflict_x", "conflict_y"]].values
        c = c[np.isfinite(c).all(1)]
        ax.scatter(c[:, 0], c[:, 1], s=10, color=COLORS[m], alpha=0.4,
                   zorder=2, label="generated conflict pts")
        rc = rcp[np.isfinite(rcp).all(1)]
        ax.scatter(rc[:, 0], rc[:, 1], s=24, marker="x", color=INK, lw=1.1,
                   zorder=4, label="real conflict pts")
        ax.set_title(f"{LABELS[m]} — uncovered@ε {int(unc.sum())}/{len(unc)}")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x [m]")
        if j == 0:
            ax.set_ylabel("y [m]")
            ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axp = fig.add_subplot(gs[1, :])
    for m in methods:
        axp.plot(np.sort(nn[m]), color=COLORS[m], lw=1.8, label=LABELS[m])
    axp.axhline(eps, color=INK, lw=1.2, ls="--")
    axp.text(2, eps * 1.05, f"ε_real = {eps:.3f}", fontsize=9)
    axp.set_xlabel("real scenarios, sorted by distance to nearest generated variant")
    axp.set_ylabel("NN distance (weighted path space)")
    axp.set_title("coverage profile — the right-hand tail = edge cases no variant reaches")
    axp.legend(frameon=False, fontsize=8, ncol=4)
    fig.suptitle(f"Coverage map — {title}: red = the {n_worst} real paths this "
                 f"method covers WORST (edge cases); dots = conflict-point span "
                 f"of the variants", y=0.99)
    fig.savefig(EXP / "figs" / f"{name}_fig4_coverage_map.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[figs] saved {name}_fig1-4", flush=True)


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--direction", choices=("n2e", "s2w", "s2n", "r2l", "all"),
                    default="all")
    ap.add_argument("--title", default=None)
    ap.add_argument("--stages", default="extract,render,svd,desc,metrics,figs")
    args = ap.parse_args()
    ddir = EXP / "data" / args.name
    gdir = EXP / "generated" / args.name
    rdir = EXP / "results"
    title = args.title or args.name
    stages = args.stages.split(",")
    if "extract" in stages:
        stage_extract(args.label, args.direction, ddir)
    if "render" in stages:
        stage_render(ddir, gdir, args.label)
    if "svd" in stages:
        stage_svd(ddir, gdir, rdir, args.name)
    if "desc" in stages:
        stage_descriptors(ddir, gdir, rdir, args.name)
    if "metrics" in stages:
        stage_metrics(ddir, gdir, rdir, args.name)
    if "figs" in stages:
        stage_figures(ddir, gdir, rdir, args.name, title)
    print("LABEL RUN DONE", flush=True)
