#!/usr/bin/env python3
"""Question (e): epsilon-ball sweep.

How does the coverage fraction of every generator arm respond to the size of the
epsilon ball, and can one FIXED epsilon replace the current per-class
median-nearest-neighbour rule?  Should path (trajectory) and interaction keep
two separate epsilons?

This script is a pure OFFLINE recomputation.  Nothing is rendered, no esmini is
touched, no existing file is modified.  It reads the frozen descriptor pools
(results/e6_descriptors.parquet, results/e6_descriptors_sakura.parquet) and the
frozen real reference (results/e6_real_reference.csv) and re-derives the
scripts/46 coverage definition for a whole grid of absolute epsilon values.

COVERAGE DEFINITION (copied from scripts/46_coverage_similarity.py, which is
imported as a module so the ordering RNG, the valid-prefix rule, the z-scaling
and the class eps are literally the same objects):
  interaction : real i is covered by draw j iff ||z_j - z_i||_2 <= eps_int,
                z = 6-D (pet, d_min, alpha, conflict_x, conflict_y, u_c)
                standardized with the per-class REAL mean/std (ddof=1).
  path        : real i is covered by draw j iff kernel_center_uid[j] == uid[i]
                AND dtw[j] <= eps_path (metres).
  joint       : both, on the SAME draw j.
  coverage@b  : fraction of the real set covered by at least one of the first b
                draws of an ordering; 20 orderings (order 0 = natural draw
                order, orders 1..19 = numpy default_rng(60000+o) permutations);
                invalid draws cost budget but cannot cover (pool "valid" only,
                which is the contiguous Table-5-flagged prefix of the pool).

EXACT SWEEP ALGORITHM (one pass, every eps at once):
  interaction : per ordering, the running minimum over the draw order of the
                distance of each real to the draws seen so far, snapshotted at
                the reported budgets.  coverage(eps,b) = mean_i[R_int[i,b]<=eps]
                is then exact for every eps.
  path        : per ordering, the minimum ORDER RANK at which each real becomes
                path-covered, as a function of eps_path (a prefix-minimum of the
                rank over the eps_path grid).  covered@b  <=>  minrank < b.
  joint       : the same idea in 2-D.  Only draws centred on i can ever be a
                joint hit, so each such draw j is scattered into the grid cell
                (first eps_int >= d(i,j), first eps_path >= dtw_j) and a 2-D
                prefix-minimum of the rank gives, for EVERY (eps_int, eps_path)
                cell, the first budget at which the real becomes joint-covered.

REGRESSION GATE: at eps = the per-class median-NN values x {1, 1.5, 2} every
overlapping cell of results/table3_coverage.csv and
results/table3_coverage_sakura.csv must be reproduced exactly (< 1e-12).

Outputs (results/eps_sweep/): eps_sweep_coverage.csv, eps_sweep_corner.csv,
eps_sweep_joint_grid.csv, eps_real_nn_distributions.csv, eps_real_nn_values.csv,
regression_gate.json, eps_sweep_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
OUT = RES / "eps_sweep"
FIGDIR = PROJECT / "figures" / "eps_sweep"

# ── scripts/46 imported verbatim (same pattern as scripts/46b) ──────────────
_spec = importlib.util.spec_from_file_location("cov46", PROJECT / "scripts/46_coverage_similarity.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["cov46"] = M
_spec.loader.exec_module(M)

DESC = M.DESC
ZC = [f"z_{k}" for k in DESC]
SPACES = M.SPACES                      # interaction, path, joint
BUDGETS = M.BUDGETS                    # 10, 100, 1000 (+ the pool size)
EMULT = M.EMULT                        # 1.0, 1.5, 2.0
REAL_SETS = M.REAL_SETS                # matched489, disk469
CLASSES = M.CLASSES                    # tlkeep, keeptl, keeptl_sw, cutinl, cutinr
RARE_FRAC = 0.2

SAK_ARMS = {   # scripts/46b NEW_ARMS (the cond 39_180 arm has no real class and is excluded)
    "sakura_route_kde": dict(label="SAKURA-route + KDE (offset, v_avg) (executed)", kind="dependent", seeded=True),
    "sakura_route_defaults": dict(label="SAKURA-route defaults nominal (1 per scene)", kind="nominal", seeded=False),
    "sakura_plain": dict(label="SAKURA plain chord nominal (192 sakura_bc + extra cutinr)", kind="nominal", seeded=False),
}
ARM_SRC = {**{a: "e6_descriptors.parquet" for a in M.ARMS},
           **{a: "e6_descriptors_sakura.parquet" for a in SAK_ARMS}}
ARM_SPEC = {**M.ARMS, **SAK_ARMS}


# ── absolute eps grids ─────────────────────────────────────────────────────
def build_grid(lo, hi, n_wide, dlo, dhi, n_dense, rounds):
    g = np.round(np.concatenate([np.geomspace(lo, hi, n_wide), np.geomspace(dlo, dhi, n_dense)]), 6)
    return np.unique(np.concatenate([g, np.asarray(rounds, float)]))


GRID_INT = build_grid(0.05, 6.0, 16, 0.30, 2.0, 8, [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
GRID_PATH = build_grid(0.02, 10.0, 16, 0.10, 2.0, 8, [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# real reference, z-scaling, class eps  (asserted identical to table34_manifest)
# ═══════════════════════════════════════════════════════════════════════════
def load_real_and_eps():
    real = pd.read_csv(RES / "e6_real_reference.csv", low_memory=False)
    assert real.scenario_uid.is_unique      # 489 before the 2026-09-14 cutinl exclusion, 486 after
    for c in ("int_finite", "in_disk469", "in_matched489", "pet_finite"):
        real[c] = real[c].astype(bool)
    scaling = M.real_scaling(real)
    man = json.loads((RES / "table34_manifest.json").read_text())
    eps = {}
    for cls, g in real.groupby("subset"):
        gi = g[g.int_finite]
        eps[cls] = dict(interaction=M.cv_eps(gi[ZC].to_numpy(float)), path=float(np.nanmedian(g.nn_dtw_m)),
                        n_real_int=int(len(gi)), n_real=int(len(g)))
    d_eps = max(max(abs(eps[c]["interaction"] - man["eps"][c]["interaction"]),
                    abs(eps[c]["path"] - man["eps"][c]["path"])) for c in CLASSES)
    Zr = M.zscore(real, scaling)
    d_z = float(np.nanmax(np.abs(Zr - real[ZC].to_numpy(float))))
    assert d_eps < 1e-12 and d_z < 1e-9, (d_eps, d_z)
    print(f"[real] 489 scenarios; eps == table34_manifest.json (max |d| {d_eps:.2e}); z max |d| {d_z:.2e}", flush=True)
    return real, scaling, eps


def real_nn_tables(real, eps):
    """Real-to-real NN distances: interaction = the quantity whose per-class median IS the
    current eps_int (cvlib._eps on the class z matrix); path = nn_dtw_m from
    scripts/46 real_path_eps (defined once over the 489 matched489 reals of the class)."""
    rows = []
    for real_set in REAL_SETS:
        inset = real.in_disk469 if real_set == "disk469" else real.in_matched489
        for cls in CLASSES:
            Rall = real[real.subset.eq(cls) & inset]
            R = Rall[Rall.int_finite].reset_index(drop=True)
            Z = R[ZC].to_numpy(float)
            D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
            np.fill_diagonal(D, np.inf)
            nn = D.min(axis=1)
            if real_set == "matched489":
                assert abs(float(np.median(nn)) - eps[cls]["interaction"]) < 1e-12, cls
            for u, v in zip(R.scenario_uid, nn):
                rows.append(dict(real_set=real_set, subset=cls, space="interaction", scenario_uid=u, nn_dist=float(v)))
            if real_set == "matched489":
                for r in Rall.itertuples():
                    rows.append(dict(real_set=real_set, subset=cls, space="path", scenario_uid=r.scenario_uid,
                                     nn_dist=float(r.nn_dtw_m)))
    return pd.DataFrame(rows)


def summarise_nn(vals, **keys):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    q = lambda p: float(np.quantile(v, p)) if len(v) else np.nan
    return dict(**keys, n=int(len(v)), min=float(v.min()) if len(v) else np.nan, p05=q(.05), p10=q(.10), p25=q(.25),
                median=q(.50), p75=q(.75), p90=q(.90), p95=q(.95), max=float(v.max()) if len(v) else np.nan,
                mean=float(v.mean()) if len(v) else np.nan, std=float(v.std(ddof=1)) if len(v) > 1 else np.nan)


# ═══════════════════════════════════════════════════════════════════════════
# pools (scripts/46 / 46b semantics)
# ═══════════════════════════════════════════════════════════════════════════
def load_pools():
    d6 = pd.read_parquet(RES / "e6_descriptors.parquet")
    ds = pd.read_parquet(RES / "e6_descriptors_sakura.parquet")
    pools = {}
    for arm, src in ARM_SRC.items():
        df = d6 if src == "e6_descriptors.parquet" else ds
        A = df[df.arm.eq(arm)]
        if not len(A):
            print(f"[pool] {arm}: absent, skipped", flush=True)
            continue
        pools[arm] = A
        print(f"[pool] {arm}: {len(A)} rows, seeds {sorted(A.seed.unique())}, classes {sorted(A.subset.unique())}", flush=True)
    return pools


def pool_slice(Acls, seed, pool):
    """scripts/46 'for pool in (all, valid)': G is sorted by draw; the valid pool is the
    contiguous Table-5-flagged prefix and flagged-invalid draws inside it cannot cover."""
    Gfull = Acls[Acls.seed.eq(seed)].sort_values("draw").reset_index(drop=True)
    if pool == "all":
        G = Gfull
    else:
        flagged = Gfull.table5_available.to_numpy()
        n_prefix = int(np.argmin(flagged)) if not flagged.all() else len(flagged)
        if n_prefix == 0:
            return None, None
        G = Gfull.iloc[:n_prefix].reset_index(drop=True)
    valid_mask = np.array([bool(v) if pd.notna(v) else False for v in G.valid_gate.to_numpy()])
    return G, valid_mask


# ═══════════════════════════════════════════════════════════════════════════
# the sweep kernel: one (class, real_set, arm, seed, pool) cell, every eps at once
# ═══════════════════════════════════════════════════════════════════════════
def combo_sweep(G, Rall, valid_mask, pool, EI, EP, EPJ, idx_diag_j, gate_i, gate_pj,
                budgets, n_ord, masks, want_2d):
    """masks: {name: {space: boolean row mask over Rall}}.
    Returns {name: {space: {budget: (per-ordering coverage arrays)}}} plus the 2-D joint grid."""
    nR, Mn = len(Rall), len(G)
    nEI, nEP, nEPJ, nB = len(EI), len(EP), len(EPJ), len(budgets)

    ZG = G[ZC].to_numpy(float)
    ZR = Rall[ZC].to_numpy(float)
    okG = np.isfinite(ZG).all(axis=1)
    D = np.full((nR, Mn), np.inf)
    if okG.any():
        Gz = ZG[okG]
        for i in range(nR):
            D[i, okG] = np.linalg.norm(Gz - ZR[i], axis=1)
    D[~np.isfinite(D)] = np.inf              # a NaN real row can never be covered (46: NaN <= eps is False)
    if pool == "valid":
        D[:, ~valid_mask] = np.inf

    uid = Rall.scenario_uid.to_numpy()
    pos = {u: i for i, u in enumerate(uid)}
    ci = np.array([pos.get(c, -1) for c in G.kernel_center_uid.to_numpy()], int)
    dtw = G.dtw.to_numpy(float)
    if pool == "valid":
        dtw = np.where(valid_mask, dtw, np.inf)
    own = ci >= 0
    a_self = np.full(Mn, np.inf)             # interaction distance of a draw to ITS OWN kernel centre
    if own.any():
        w = np.flatnonzero(own)
        a_self[w] = D[ci[w], w]

    ii = np.searchsorted(EI, a_self, side="left")     # first grid value >= the distance
    jj = np.searchsorted(EP, dtw, side="left")
    jjJ = np.searchsorted(EPJ, dtw, side="left")
    path_ok = np.flatnonzero(own & (jj < nEP))
    joint_ok = np.flatnonzero(own & (ii < nEI) & (jjJ < nEPJ))

    ords = M.orderings(Mn, n_ord)
    res = {name: {sp: {b: [] for b in budgets} for sp in SPACES} for name in masks}
    cov2d = np.empty((len(ords), nEI, nEPJ)) if want_2d else None
    b2d = budgets[-1]
    for o, perm in enumerate(ords):
        rank = np.empty(Mn, float)
        rank[perm] = np.arange(Mn, dtype=float)
        # interaction: block-wise running minimum over the draw order
        cur = np.full(nR, np.inf)
        rint = np.empty((nB, nR))
        lo = 0
        for k, b in enumerate(budgets):
            if b > lo:
                cur = np.minimum(cur, D[:, perm[lo:b]].min(axis=1))
                lo = b
            rint[k] = cur
        # path: prefix-min of the rank over the eps_path grid
        P = np.full((nR, nEP), np.inf)
        if len(path_ok):
            np.minimum.at(P, (ci[path_ok], jj[path_ok]), rank[path_ok])
        np.minimum.accumulate(P, axis=1, out=P)
        # joint: 2-D prefix-min of the rank over (eps_int, eps_path)
        J = np.full((nR, nEI, nEPJ), np.inf)
        if len(joint_ok):
            np.minimum.at(J, (ci[joint_ok], ii[joint_ok], jjJ[joint_ok]), rank[joint_ok])
        np.minimum.accumulate(J, axis=1, out=J)
        np.minimum.accumulate(J, axis=2, out=J)
        Jdiag = J[:, np.arange(nEI), idx_diag_j]                        # (nR, nEI)
        Jgate = J[:, gate_i, gate_pj]                                   # (nR, n_mult)
        if want_2d:
            sel2d = masks["full"]["joint"]
            cov2d[o] = (J[sel2d] < b2d).mean(axis=0) if sel2d.any() else np.nan
        for name, msp in masks.items():
            for k, b in enumerate(budgets):
                s = msp["interaction"]
                if s.any():
                    res[name]["interaction"][b].append((rint[k][s][:, None] <= EI[None, :]).mean(axis=0))
                s = msp["path"]
                if s.any():
                    res[name]["path"][b].append((P[s] < b).mean(axis=0))
                s = msp["joint"]
                if s.any():
                    res[name]["joint"][b].append(np.concatenate([(Jdiag[s] < b).mean(axis=0),
                                                                 (Jgate[s] < b).mean(axis=0)]))
    return res, cov2d


def stats(c):
    """c = per-ordering coverage values -> (mean, min, max, natural order)."""
    c = np.asarray(c, float)
    return float(c.mean()), float(c.min()), float(c.max()), float(c[0])


# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="epsilon-ball sweep (question e)")
    ap.add_argument("--quick", action="store_true", help="smoke test: one class, two arms, coarse grids (orderings unchanged)")
    ap.add_argument("--n-orderings", type=int, default=M.NORD)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.monotonic()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    n_ord = args.n_orderings           # kept at 20 even in --quick so the regression gate stays exact
    grid_int, grid_path = (GRID_INT[::4], GRID_PATH[::4]) if args.quick else (GRID_INT, GRID_PATH)
    classes = ["cutinr"] if args.quick else CLASSES
    arms_keep = ["ours3_disk_kde", "svd_d5_fullfit_recon"] if args.quick else None

    real, scaling, eps = load_real_and_eps()
    nn = real_nn_tables(real, eps)
    nn.to_csv(outdir / "eps_real_nn_values.csv", index=False)
    srows = []
    for (rs, cls, sp), g in nn.groupby(["real_set", "subset", "space"]):
        srows.append(summarise_nn(g.nn_dist, real_set=rs, scope=cls, space=sp,
                                  current_eps=eps[cls]["interaction"] if sp == "interaction" else eps[cls]["path"]))
    for (rs, sp), g in nn.groupby(["real_set", "space"]):
        srows.append(summarise_nn(g.nn_dist, real_set=rs, scope="pooled", space=sp, current_eps=np.nan))
    nn_sum = pd.DataFrame(srows).sort_values(["real_set", "space", "scope"]).reset_index(drop=True)
    nn_sum.to_csv(outdir / "eps_real_nn_distributions.csv", index=False)
    for _, r in nn_sum[nn_sum.scope.eq("pooled") & nn_sum.real_set.eq("matched489")].iterrows():
        print(f"[nn] pooled matched489 {r.space}: n={r.n} min {r['min']:.4f} p25 {r.p25:.4f} median {r['median']:.4f} "
              f"p75 {r.p75:.4f} p90 {r.p90:.4f} max {r['max']:.3f}", flush=True)

    # quantile-matched diagonal: eps_path paired with eps_int at the same ECDF level of the
    # pooled (matched489) real NN distributions
    pool_int = np.sort(nn[nn.real_set.eq("matched489") & nn.space.eq("interaction")].nn_dist.to_numpy(float))
    pool_path = np.sort(nn[nn.real_set.eq("matched489") & nn.space.eq("path")].nn_dist.to_numpy(float))
    pool_int, pool_path = pool_int[np.isfinite(pool_int)], pool_path[np.isfinite(pool_path)]

    def diag_partner(e):
        q = float(np.mean(pool_int <= e))
        return float(np.quantile(pool_path, min(max(q, 0.0), 1.0))), q

    rare = pd.read_csv(RES / "final/tables3/rare_scenes.csv")
    rare_uids = {(c, rs): set(g[g.is_rare].scenario_uid) for (c, rs), g in rare.groupby(["class", "real_set"])}
    for (c, rs), u in sorted(rare_uids.items()):
        n_rank = int((rare["class"].eq(c) & rare.real_set.eq(rs)).sum())
        assert len(u) == int(math.ceil(RARE_FRAC * n_rank)), (c, rs, len(u), n_rank)

    pools = load_pools()
    if arms_keep:
        pools = {a: v for a, v in pools.items() if a in arms_keep}

    cov_rows, corner_rows, grid_rows, gate_rows = [], [], [], []
    for arm, A in pools.items():
        spec = ARM_SPEC[arm]
        for cls in classes:
            Acls = A[A.subset.eq(cls)]
            if not len(Acls):
                continue
            e_int, e_path = eps[cls]["interaction"], eps[cls]["path"]
            gate_int_vals = np.array([e_int * m for m in EMULT])
            gate_path_vals = np.array([e_path * m for m in EMULT])
            EI = np.unique(np.concatenate([grid_int, gate_int_vals]))
            EP = np.unique(np.concatenate([grid_path, gate_path_vals]))
            diag = np.array([diag_partner(e) for e in EI])              # (nEI, 2) = (eps_path, quantile)
            EPJ = np.unique(np.concatenate([EP, diag[:, 0]]))
            gate_i = np.searchsorted(EI, gate_int_vals)
            gate_p = np.searchsorted(EP, gate_path_vals)
            gate_pj = np.searchsorted(EPJ, gate_path_vals)
            assert np.allclose(EI[gate_i], gate_int_vals) and np.allclose(EP[gate_p], gate_path_vals)
            idx_diag_j = np.searchsorted(EPJ, diag[:, 0])
            assert np.allclose(EPJ[idx_diag_j], diag[:, 0])
            mult_i = {int(gate_i[u]): m for u, m in enumerate(EMULT)}
            mult_p = {int(gate_p[u]): m for u, m in enumerate(EMULT)}
            for real_set in REAL_SETS:
                inset = real.in_disk469 if real_set == "disk469" else real.in_matched489
                Rall = real[real.subset.eq(cls) & inset].reset_index(drop=True)
                if not len(Rall):
                    continue
                int_fin = Rall.int_finite.to_numpy()
                rmask = Rall.scenario_uid.isin(rare_uids[(cls, real_set)]).to_numpy()
                sel_space = dict(interaction=int_fin, path=np.ones(len(Rall), bool), joint=int_fin)
                masks = {"full": {sp: sel_space[sp] for sp in SPACES},
                         "rare": {sp: sel_space[sp] & rmask for sp in SPACES}}
                seeds = sorted(Acls.seed.unique()) if spec["seeded"] else [0]
                for pool in ("all", "valid"):
                    for seed in seeds:
                        G, vmask = pool_slice(Acls, seed, pool)
                        if G is None:
                            continue
                        Mn = len(G)
                        dependent = spec["kind"] == "dependent"
                        budgets = [b for b in sorted(set(BUDGETS + [Mn])) if b <= Mn] if dependent else [Mn]
                        nord_here = n_ord if dependent else 1
                        want_2d = real_set == "matched489" and pool == "all"
                        res, cov2d = combo_sweep(G, Rall, vmask, pool, EI, EP, EPJ, idx_diag_j, gate_i, gate_pj,
                                                 budgets, nord_here, masks, want_2d)
                        base = dict(subset=cls, real_set=real_set, arm=arm, arm_label=spec["label"], seed=seed,
                                    pool=pool, n_samples=Mn, n_samples_valid=int(vmask.sum()),
                                    n_samples_flagged=int(G.table5_available.sum()),
                                    n_orderings=nord_here if dependent else 0)
                        for name, rows in (("full", cov_rows), ("rare", corner_rows)):
                            for space in SPACES:
                                nR = int(masks[name][space].sum())
                                if nR == 0:
                                    continue
                                for b in budgets:
                                    C = np.asarray(res[name][space][b])          # (n_ord, n_eps)
                                    com = dict(base, space=space, budget=b, budget_is_pool_size=b == Mn, n_real=nR)
                                    if space == "interaction":
                                        for t, e in enumerate(EI):
                                            m_, mn_, mx_, nat = stats(C[:, t])
                                            rows.append(dict(com, eps_int=float(e), eps_path=np.nan, eps_quantile=np.nan,
                                                             eps_mult=mult_i.get(t, np.nan), coverage_mean=m_, coverage_min=mn_,
                                                             coverage_max=mx_, coverage_natural_order=nat))
                                        if name == "full":
                                            for u, mm in enumerate(EMULT):
                                                m_, mn_, mx_, nat = stats(C[:, gate_i[u]])
                                                gate_rows.append(dict(com, eps_mult=mm, coverage_mean=m_, coverage_min=mn_,
                                                                      coverage_max=mx_, coverage_natural_order=nat))
                                    elif space == "path":
                                        for t, e in enumerate(EP):
                                            m_, mn_, mx_, nat = stats(C[:, t])
                                            rows.append(dict(com, eps_int=np.nan, eps_path=float(e), eps_quantile=np.nan,
                                                             eps_mult=mult_p.get(t, np.nan), coverage_mean=m_, coverage_min=mn_,
                                                             coverage_max=mx_, coverage_natural_order=nat))
                                        if name == "full":
                                            for u, mm in enumerate(EMULT):
                                                m_, mn_, mx_, nat = stats(C[:, gate_p[u]])
                                                gate_rows.append(dict(com, eps_mult=mm, coverage_mean=m_, coverage_min=mn_,
                                                                      coverage_max=mx_, coverage_natural_order=nat))
                                    else:
                                        for t, e in enumerate(EI):                # quantile-matched diagonal
                                            m_, mn_, mx_, nat = stats(C[:, t])
                                            rows.append(dict(com, eps_int=float(e), eps_path=float(diag[t, 0]),
                                                             eps_quantile=float(diag[t, 1]), eps_mult=np.nan,
                                                             coverage_mean=m_, coverage_min=mn_, coverage_max=mx_,
                                                             coverage_natural_order=nat))
                                        for u, mm in enumerate(EMULT):             # the current per-class rule
                                            m_, mn_, mx_, nat = stats(C[:, len(EI) + u])
                                            rows.append(dict(com, eps_int=float(gate_int_vals[u]), eps_path=float(gate_path_vals[u]),
                                                             eps_quantile=np.nan, eps_mult=mm, coverage_mean=m_,
                                                             coverage_min=mn_, coverage_max=mx_, coverage_natural_order=nat))
                                            if name == "full":
                                                gate_rows.append(dict(com, eps_mult=mm, coverage_mean=m_, coverage_min=mn_,
                                                                      coverage_max=mx_, coverage_natural_order=nat))
                        if want_2d and masks["full"]["joint"].any():
                            in_ep = np.flatnonzero(np.isin(EPJ, EP))
                            mean_, min_, max_ = cov2d.mean(axis=0), cov2d.min(axis=0), cov2d.max(axis=0)
                            nRj = int(masks["full"]["joint"].sum())
                            for p, e in enumerate(EI):
                                for q in in_ep:
                                    grid_rows.append(dict(subset=cls, real_set=real_set, arm=arm, arm_label=spec["label"],
                                                          seed=seed, pool=pool, space="joint", budget=budgets[-1],
                                                          n_real=nRj, n_samples=Mn, eps_int=float(e), eps_path=float(EPJ[q]),
                                                          coverage_mean=float(mean_[p, q]), coverage_min=float(min_[p, q]),
                                                          coverage_max=float(max_[p, q])))
                        print(f"[sweep] {arm} {cls} {real_set} seed={seed} pool={pool} M={Mn} nR={len(Rall)} "
                              f"budgets={budgets} ({time.monotonic() - t0:.0f}s)", flush=True)

    cov = pd.DataFrame(cov_rows)
    cov.to_csv(outdir / "eps_sweep_coverage.csv", index=False)
    corner = pd.DataFrame(corner_rows)
    corner.to_csv(outdir / "eps_sweep_corner.csv", index=False)
    gridf = pd.DataFrame(grid_rows)
    gridf.to_csv(outdir / "eps_sweep_joint_grid.csv", index=False)
    print(f"[write] coverage {len(cov)}, corner {len(corner)}, joint grid {len(gridf)} rows", flush=True)

    # ── regression gate ────────────────────────────────────────────────────
    mine = pd.DataFrame(gate_rows)
    key = ["subset", "real_set", "arm", "seed", "pool", "eps_mult", "space", "budget"]
    assert mine.duplicated(key).sum() == 0, "gate rows are not unique"
    mine = mine.set_index(key).sort_index()
    gate = dict(per_table={})
    worst, total, missing = 0.0, 0, []
    for tbl in ("table3_coverage.csv", "table3_coverage_sakura.csv"):
        ref = pd.read_csv(RES / tbl)
        if arms_keep:
            ref = ref[ref.arm.isin(arms_keep)]
        if args.quick:
            ref = ref[ref.subset.isin(classes)]
        w, n, miss = 0.0, 0, 0
        for r in ref.itertuples():
            k = (r.subset, r.real_set, r.arm, r.seed, r.pool, r.eps_mult, r.space, r.budget)
            if k not in mine.index:
                miss += 1
                missing.append([str(x) for x in k])
                continue
            m_ = mine.loc[k]
            d = max(abs(m_.coverage_mean - r.coverage_mean), abs(m_.coverage_min - r.coverage_min),
                    abs(m_.coverage_max - r.coverage_max), abs(m_.coverage_natural_order - r.coverage_natural_order))
            w, n = max(w, float(d)), n + 1
        gate["per_table"][tbl] = dict(n_ref_rows=int(len(ref)), n_cells_compared=n, n_missing=miss, max_abs_diff=w)
        worst, total = max(worst, w), total + n
        print(f"[gate] {tbl}: {n}/{len(ref)} cells compared, max |diff| = {w:.3e}, missing {miss}", flush=True)
    gate.update(n_cells_compared=total, max_abs_diff=worst, n_missing=len(missing),
                missing_keys_sample=missing[:20], tolerance=1e-12,
                compared_columns=["coverage_mean", "coverage_min", "coverage_max", "coverage_natural_order"],
                status="PASS" if (worst < 1e-12 and not missing) else "FAIL")
    write_json(outdir / "regression_gate.json", gate)
    print(f"[gate] {gate['status']}: {total} cells, max |diff| = {worst:.3e}", flush=True)

    man = dict(script=str(Path(__file__).resolve()), generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               elapsed_s=round(time.monotonic() - t0, 1), quick=args.quick, n_orderings=n_ord,
               python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
               ordering_rng=f"order 0 = natural draw order; orders 1..{n_ord - 1} = numpy default_rng({M.SEED_ORDER}+o).permutation(M)",
               inputs={p: dict(sha256=sha256(RES / p), bytes=(RES / p).stat().st_size) for p in
                       ("e6_descriptors.parquet", "e6_descriptors_sakura.parquet", "e6_real_reference.csv",
                        "table3_coverage.csv", "table3_coverage_sakura.csv", "table34_manifest.json",
                        "final/tables3/rare_scenes.csv")},
               source_scripts={p: sha256(PROJECT / "scripts" / p) for p in
                               ("46_coverage_similarity.py", "46b_coverage_similarity_extra.py", "89_gen_metrics_tables3.py")},
               grids=dict(eps_int_global=[float(x) for x in grid_int], eps_path_global=[float(x) for x in grid_path],
                          n_eps_int_global=int(len(grid_int)), n_eps_path_global=int(len(grid_path)),
                          per_class_extra="each class grid additionally carries its own median-NN eps x {1,1.5,2}",
                          budgets="10, 100, 1000 and the pool size (dependent arms); pool size only (nominal/ceiling arms)",
                          joint="diagonal = eps_path paired with eps_int at the same ECDF level of the pooled matched489 real NN distributions (all budgets); full 2-D grid at real_set=matched489, pool=all, budget = the largest reported budget"),
               class_eps=eps, arms=sorted(pools), classes=classes, real_sets=REAL_SETS, rare_frac=RARE_FRAC,
               excluded=dict(e6_descriptors_sakura_39_180_cond_parquet="arm sakura_route_kde_cond_39_180 lives in class special_39_180, which has no real-set reference, so it has no coverage denominator"),
               outputs={f: dict(rows=int(n), bytes=(outdir / f).stat().st_size) for f, n in
                        (("eps_sweep_coverage.csv", len(cov)), ("eps_sweep_corner.csv", len(corner)),
                         ("eps_sweep_joint_grid.csv", len(gridf)), ("eps_real_nn_distributions.csv", len(nn_sum)),
                         ("eps_real_nn_values.csv", len(nn)))},
               regression_gate=gate)
    write_json(outdir / "eps_sweep_manifest.json", man)
    print(f"[done] {time.monotonic() - t0:.0f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
