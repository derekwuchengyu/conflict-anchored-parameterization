"""Shared spine for exp_coverage_velocity (coverage velocity / search efficiency).

Question. Three scenario-generation methods, each turned into *parameters + de
Gelder-style Gaussian KDE (LOO-CV bandwidth, dependent sampling)*:
  svd_kde : rank-3 SVD reduced params (algebraic, no execution)
  ours    : NURBS control-point parameterization (esmini execution)
  sakura  : start/end-only NURBS + constant speed (esmini execution)
At matched budgets 10^{0,1,1.5,2,2.5,3,3.5,4} concretes: (1) how many samples
until a single real replay (path & interaction) is covered, (2) does coverage
ever reach 100 %, (3) the growth curve.

Everything reads sr-tlkeep-experiment READ-ONLY.  The de Gelder math is
imported, never copied (core.svd_param / core.kde_sampling / core.wasserstein).

SPACE DISCIPLINE.  The upstream project uses two different metrics:
  w101 = alpha o x            (all 101 coords, incl. duration)  -- Eq.21 / W1
  w100 = (alpha o x)[:100]    (positions only, duration dropped) -- frac_covered
keeptl: eps101 = 0.6645, eps100 = 0.3980.  The rank-d orthogonality identity
holds ONLY in w101.  Every function takes an explicit ``space``.

FROZEN RULER.  alpha, mu, U, s, V, h, eps come from the fit on the FULL real
set and are never re-derived per arm.  Leave-one-out arms change the *model*,
never the ruler (refit models reconstruct to RAW, then re-weight with the
frozen alpha).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
BASE = Path("/home/hcis-s19/Documents/ChengYu")
SR = BASE / "sr-tlkeep-experiment"
HP = BASE / "hetero-param"
NAME = os.environ.get("CV_SUBSET", "keeptl")
SUF = "" if NAME == "keeptl" else f"_{NAME}"
# keeptl (the original testbed) lives at results/ and figs/; other subsets
# get their own subdirectory so every stage script stays unchanged.
RESULTS = EXP / "results" if NAME == "keeptl" else EXP / "results" / NAME
FIGS = EXP / "figs" if NAME == "keeptl" else EXP / "figs" / NAME
for _d in (RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SR))
sys.path.insert(0, str(SR / "scripts"))

from core.svd_param import SvdParameterization, split_vector  # noqa: E402,F401
from core.kde_sampling import loo_bandwidth                   # noqa: E402
from core.wasserstein import empirical_wasserstein            # noqa: E402,F401

# ─── constants (mirror sr-tlkeep-experiment/scripts/20_run_label.py) ─────────
NT, NY, NTH = 50, 2, 1
NX = NT * NY + NTH
D_DEFAULT = 3
FPS = 30.0
TRIM_SPEED, TRIM_MAX_S = 0.05, 0.6
VTAGS = ("of-", "of+", "en-", "en+")
SEED = 20260905
BUDGETS = [1, 3, 10, 32, 100, 316, 1000, 3162, 10000]      # 10^{0..4} step .5

# corner case per subset: keeptl is the user's circled scenario; any other
# subset takes its largest rank-3 residual (w100 LS floor) at load().
CORNER = os.environ.get("CV_CORNER") or {"keeptl": "230_179"}.get(NAME)
LABEL_OF = {"keeptl": 1, "keeptl_sw": 1, "tlkeep": 2, "cutinr": 7, "cutinl": 8,
            "cutin88": 7, "cutin_dir": 7}
CP3_LABEL = {"keeptl": "keeplt", "cutinr": "cutin", "tlkeep": "leftturn",
             "cutin88": "cutin88"}
METHODS = ("svd_kde", "ours", "sakura")

# ─── plotting (house style) ─────────────────────────────────────────────────
INK, GRAY, MUT, GRID_C = "#33322e", "#9a9a94", "#6f6e68", "#eceae5"
COL = {"real": INK, "svd_d3": "#eb6834", "svd_kde": "#0f8a60",
       "ours": "#d99000", "sakura": "#2a78d6", "cond": "#7856c2",
       "identity": "#c22e2e", "uniform": "#7aa6d9", "indep": "#3557b0",
       "boot": "#9a9a94", "gauss": "#1baf7a"}
LAB = {"real": "real", "svd_d3": "SVD (d=3)", "svd_kde": "SVD+KDE",
       "ours": "ours+KDE (θ1/θ2 + speed)", "sakura": "sakura-route+KDE (offset + avg speed)",
       "cond": "SVD+KDE, centre forced"}
RC = {"figure.facecolor": "white", "axes.facecolor": "white",
      "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK, "text.color": INK,
      "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
      "grid.color": GRID_C, "grid.linewidth": 0.8, "font.size": 10,
      "axes.titlesize": 11}


def figstyle(font=10.0, grid=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = dict(RC)
    rc["font.size"] = font
    rc["axes.grid"] = grid
    plt.rcParams.update(rc)
    return plt


def savefig(fig, plt, slug, dpi=160):
    fig.savefig(FIGS / f"{slug}.png", dpi=dpi)
    plt.close(fig)
    print(f"[fig] {slug}.png", flush=True)


# ─── testbed ────────────────────────────────────────────────────────────────
class Testbed:
    """Frozen full-set fit + frozen ruler.  Never refit `self.P`."""

    def __init__(self, name: str, d: int = D_DEFAULT):
        self.name, self.d = name, d
        ddir = SR / "data" / name
        z = np.load(ddir / "real_vectors.npz", allow_pickle=True)
        self.X = np.asarray(z["X_raw"], float)
        self.keys = [str(k) for k in z["keys"]]
        self.N = len(self.keys)
        self.meta = pd.read_csv(ddir / "real_meta.csv")
        assert list(self.meta.scenario_id) == self.keys, \
            "real_meta.csv row order must match real_vectors.npz keys"
        self.ddir, self.gdir = ddir, SR / "generated" / name
        self.idx = {k: i for i, k in enumerate(self.keys)}

        self.P = SvdParameterization(NT, NY, NTH).fit(self.X)
        self.alpha, self.mu = self.P.alpha, self.P.mu
        self.U, self.s, self.V = self.P.U, self.P.s, self.P.V
        self.Vd = self.P.reduced(d)
        self.h = loo_bandwidth(self.Vd)
        self.Xw = self.X * self.alpha
        self.Xc = self.Xw - self.mu
        self.eps = {sp: _eps(self.w(self.X, sp)) for sp in ("w100", "w101")}

    # spaces
    def w(self, X_raw, space):
        X_raw = np.atleast_2d(np.asarray(X_raw, float))
        if space == "raw":
            return X_raw
        Xw = X_raw * self.alpha
        return Xw[:, :NT * NY] if space == "w100" else Xw

    def wfromw(self, Xw, space):
        Xw = np.atleast_2d(np.asarray(Xw, float))
        return Xw[:, :NT * NY] if space == "w100" else Xw

    # reconstruction
    def recon_w(self, v, d=None):
        d = self.d if d is None else d
        v = np.atleast_2d(np.asarray(v, float))
        return self.mu + (v * self.s[:d]) @ self.U[:, :d].T

    def recon_raw(self, v, d=None):
        return self.recon_w(v, d) / self.alpha

    # floors
    def floor(self, i, d=None, space="w100"):
        """Smallest distance to real i reachable by ANY rank-d reconstruction.
        w101: orthogonal residual ||r_i|| (pathwise floor).  w100: least-squares
        floor on the projected coords (bounds E[d^2] only, NOT pathwise)."""
        d = self.d if d is None else d
        A = self.wfromw(self.U[:, :d].T * self.s[:d, None], space).T
        b = self.wfromw(self.Xc[i], space).ravel()
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(np.linalg.norm(A @ v - b))

    def floors(self, d=None, space="w100"):
        return np.array([self.floor(i, d, space) for i in range(self.N)])

    def centre_resid(self, i, d=None, space="w100"):
        """||P (x_i - recon(v_i))|| -- the residual AT the scenario's own
        centre.  Equals the floor in w101; in w100 it is >= the LS floor."""
        d = self.d if d is None else d
        e = self.Xc[i] - (self.Vd[i, :d] * self.s[:d]) @ self.U[:, :d].T
        return float(np.linalg.norm(self.wfromw(e, space)))

    def noise_gain(self, d=None, space="w100"):
        """sum_j s_j^2 ||P u_j||^2 (w101: sum s_j^2)."""
        d = self.d if d is None else d
        Pu = self.wfromw(self.U[:, :d].T, space)
        return float(np.sum(self.s[:d] ** 2 * np.sum(Pu ** 2, axis=1)))

    def expected_dist(self, i, c, d=None, space="w100"):
        """sqrt(E[dist^2]) for a forced-centre draw: cross term vanishes in
        expectation in every space, so the residual at the centre is used."""
        return float(np.sqrt(self.centre_resid(i, d, space) ** 2
                             + (c * self.h) ** 2 * self.noise_gain(d, space)))

    def min_dist_law(self, i, c, M, d=None, space="w101"):
        """E[min over M draws] via the small-ball law (exact only in w101)."""
        from math import gamma, pi
        d = self.d if d is None else d
        vol = pi ** (d / 2) / gamma(d / 2 + 1)
        K = vol / ((2 * pi) ** (d / 2) * np.prod(self.s[:d]))
        e_min_t = gamma(1 + 2 / d) * (M * K) ** (-2 / d)
        return float(np.sqrt(self.floor(i, d, space) ** 2
                             + (c * self.h) ** 2 * e_min_t))

    def holdout_floor(self, hold, d=None, space="w100"):
        """Refit on all-but-`hold`, reconstruct held-out to RAW, re-weight with
        the FROZEN alpha, measure.  Returns {i: dist}."""
        d = self.d if d is None else d
        hold = np.atleast_1d(hold)
        keep = np.setdiff1d(np.arange(self.N), hold)
        Q = SvdParameterization(NT, NY, NTH).fit(self.X[keep])
        out = {}
        for i in hold:
            xw_q = self.X[i] * Q.alpha
            A = (Q.U[:, :d] * Q.s[:d]).T
            v, *_ = np.linalg.lstsq(A.T, xw_q - Q.mu, rcond=None)
            rec_raw = (Q.mu + v @ A) / Q.alpha
            e = self.wfromw((rec_raw - self.X[i]) * self.alpha, space)
            out[int(i)] = float(np.linalg.norm(e))
        return out

    # geometry
    def paths(self, X_raw):
        X_raw = np.atleast_2d(np.asarray(X_raw, float))
        return X_raw[:, :NT * NY].reshape(-1, NT, NY)

    def durations(self, X_raw):
        return np.atleast_2d(np.asarray(X_raw, float))[:, NT * NY]


def _eps(R):
    DR = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=2)
    np.fill_diagonal(DR, np.inf)
    return float(np.median(DR.min(1)))


_CACHE = {}


def load(name=None, d=D_DEFAULT):
    global CORNER
    name = NAME if name is None else name
    if (name, d) not in _CACHE:
        _CACHE[(name, d)] = Testbed(name, d)
    T = _CACHE[(name, d)]
    if CORNER is None and name == NAME:
        CORNER = T.keys[int(np.argmax(T.floors(space="w100")))]
        print(f"[cvlib] {name}: corner case = largest rank-3 residual = {CORNER}", flush=True)
    return T


# ─── generic KDE over an arbitrary parameter table (ours / sakura) ──────────
class ParamKDE:
    """de Gelder-spec KDE on a (N, p) parameter matrix: per-column
    standardisation, scalar LOO-CV bandwidth, dependent sampling
    (uniform centre + N(0, h^2 I)), conditional sampling (forced centre)."""

    def __init__(self, Ptab: np.ndarray, cols):
        self.raw = np.asarray(Ptab, float)
        self.cols = list(cols)
        self.m = self.raw.mean(0)
        self.sd = np.where(self.raw.std(0) < 1e-12, 1.0, self.raw.std(0))
        self.Z = (self.raw - self.m) / self.sd
        self.N, self.p = self.Z.shape
        self.h = loo_bandwidth(self.Z)

    def to_raw(self, Z):
        return np.atleast_2d(Z) * self.sd + self.m

    def dependent(self, M, rng, c=1.0):
        idx = rng.integers(0, self.N, size=M)
        return self.to_raw(self.Z[idx] + c * self.h * rng.standard_normal((M, self.p))), idx

    def conditional(self, i, M, rng, c=1.0):
        Z = self.Z[i] + c * self.h * rng.standard_normal((M, self.p))
        return self.to_raw(Z), np.full(M, i)


# ─── SVD samplers (return reduced params (M,d), centre idx) ────────────────
def s_dependent(T, M, rng, c=1.0):
    idx = rng.integers(0, T.N, size=M)
    return T.Vd[idx] + c * T.h * rng.standard_normal((M, T.d)), idx


def s_conditional(T, M, rng, centre, c=1.0):
    i = T.idx[centre] if isinstance(centre, str) else int(centre)
    return T.Vd[i] + c * T.h * rng.standard_normal((M, T.d)), np.full(M, i)


def s_independent(T, M, rng, c=1.0):
    idx = rng.integers(0, T.N, size=(M, T.d))
    return np.take_along_axis(T.Vd, idx, axis=0) + c * T.h * rng.standard_normal((M, T.d)), idx[:, 0]


def s_bootstrap(T, M, rng, c=0.0):
    idx = rng.integers(0, T.N, size=M)
    return T.Vd[idx].copy(), idx


def s_gauss_mle(T, M, rng, c=1.0):
    return rng.multivariate_normal(T.Vd.mean(0), np.cov(T.Vd.T), size=M), None


def s_uniform(T, M, rng, k=1.0):
    lo, hi = T.Vd.min(0), T.Vd.max(0)
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * k
    return rng.uniform(mid - half, mid + half, size=(M, T.d)), None


# ─── coverage ───────────────────────────────────────────────────────────────
def nn_dists(A, B, chunk=256):
    A, B = np.asarray(A, float), np.asarray(B, float)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        dd = np.linalg.norm(A[i:i + chunk, None, :] - B[None, :, :], axis=2)
        out[i:i + chunk] = dd.min(axis=1)
    return out


def coverage(r2g, eps):
    return float((np.asarray(r2g) <= eps).mean())


def running_min_dist(R, G):
    """(N_real, M) running minimum distance of each real to the first m
    generated samples, m = 1..M.  Coverage-vs-budget in one pass."""
    R, G = np.asarray(R, float), np.asarray(G, float)
    out = np.empty((len(R), len(G)))
    for i in range(len(R)):
        d = np.linalg.norm(G - R[i], axis=1)
        out[i] = np.minimum.accumulate(d)
    return out


def first_hit(run_min, eps):
    """(N_real,) 1-based index of the first sample within eps, or inf."""
    hit = run_min <= eps
    out = np.full(len(run_min), np.inf)
    for i in range(len(run_min)):
        j = np.flatnonzero(hit[i])
        if len(j):
            out[i] = j[0] + 1
    return out


# ─── geometry / dip ─────────────────────────────────────────────────────────
def arc_resample(x, y, k=NT):
    dd = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(dd)])
    if s[-1] <= 1e-9:
        return np.stack([np.full(k, x[0]), np.full(k, y[0])], 1)
    sq = np.linspace(0.0, s[-1], k)
    return np.stack([np.interp(sq, s, x), np.interp(sq, s, y)], 1)


def dtw(A, B):
    B = np.asarray(B, float)
    if B.ndim == 2:
        B = B[None]
    M, K, _ = B.shape
    C = np.linalg.norm(A[None, :, None, :] - B[:, None, :, :], axis=3)
    Dm = np.full((M, K + 1, K + 1), np.inf)
    Dm[:, 0, 0] = 0.0
    for t in range(2, 2 * K + 1):
        a0, a1 = max(1, t - K), min(K, t - 1)
        A_ = np.arange(a0, a1 + 1)
        B_ = t - A_
        prev = np.minimum(np.minimum(Dm[:, A_ - 1, B_ - 1], Dm[:, A_ - 1, B_]),
                          Dm[:, A_, B_ - 1])
        Dm[:, A_, B_] = C[:, A_ - 1, B_ - 1] + prev
    return Dm[:, K, K] / (2.0 * K)


def dip(paths):
    return np.asarray(paths)[..., 1].min(axis=-1)


# ─── validity ───────────────────────────────────────────────────────────────
def kinematics(T, X_raw):
    X_raw = np.atleast_2d(np.asarray(X_raw, float))
    P_ = T.paths(X_raw)
    dur = np.maximum(T.durations(X_raw), 0.5)
    out = np.empty((len(X_raw), 3))
    for m, (p, du) in enumerate(zip(P_, dur)):
        tq = np.linspace(0.0, du, NT)
        vx, vy = np.gradient(p[:, 0], tq), np.gradient(p[:, 1], tq)
        sp = np.hypot(vx, vy)
        ax_, ay = np.gradient(vx, tq), np.gradient(vy, tq)
        with np.errstate(invalid="ignore", divide="ignore"):
            alat = np.abs(vx * ay - vy * ax_) / np.maximum(sp, 1e-6)
        alat = np.where(sp >= 1.0, alat, 0.0)      # a_lat is meaningless near standstill
        out[m] = [sp.max(), np.nanmax(alat), du]
    return pd.DataFrame(out, columns=["v_max", "alat_max", "dur"])


def validity_mask(T, X_raw, v_lim=25.0, alat_lim=5.0):
    k = kinematics(T, X_raw)
    return ((k.v_max <= v_lim) & (k.alat_max <= alat_lim)
            & (T.durations(X_raw) > 0.5)).values


# ─── upstream artifacts (read-only) ─────────────────────────────────────────
def load_kde_artifact(T, tag="fidelity"):
    z = np.load(T.gdir / "svd_kde" / f"vectors_{tag}.npz", allow_pickle=True)
    return np.asarray(z["X"], float), np.array([str(c) for c in z["center_keys"]])


def trim_lead_still(g):
    sp = g.speed.values
    mv = np.flatnonzero(sp > TRIM_SPEED)
    if len(mv) == 0 or mv[0] == 0:
        return g
    t = (g.frame.values - g.frame.values[0]) / FPS
    return g if t[mv[0]] > TRIM_MAX_S else g.iloc[mv[0]:]


def vector_from_track(frame, x, y, speed):
    """One rendered/real agent track -> (RAW 101-D vector, arc path) or None.
    Exactly 20_run_label.py:_vectors_from_renders incl. trim_lead_still."""
    g = pd.DataFrame({"frame": frame, "x": x, "y": y, "speed": speed}).sort_values("frame")
    g = trim_lead_still(g)
    t = (g.frame.values - g.frame.values[0]) / FPS
    dur = float(t[-1])
    if dur < 0.5 or len(g) < 5:
        return None
    tq = np.linspace(0.0, dur, NT)
    xy = np.stack([np.interp(tq, t, g.x.values), np.interp(tq, t, g.y.values)], 1)
    return np.concatenate([xy.reshape(-1), [dur]]), arc_resample(g.x.values, g.y.values, NT)


def vectors_from_renders(T, method, tags=VTAGS):
    df = pd.read_parquet(T.gdir / method / "trajectories.parquet")
    df = df[(df.role == "agent") & (df.tag.isin(tags))]
    vecs, ids, tg, paths = [], [], [], []
    for (sid, tag), g in df.groupby(["scenario_id", "tag"]):
        r = vector_from_track(g.frame.values, g.x.values, g.y.values, g.speed.values)
        if r is None:
            continue
        vecs.append(r[0]); ids.append(sid); tg.append(tag); paths.append(r[1])
    return np.asarray(vecs), ids, tg, np.asarray(paths)


def real_actor_track(T, sid):
    rt = pd.read_parquet(T.ddir / "real_tracks.parquet")
    return rt[(rt.scenario_id == sid) & (rt.role == "actor")].sort_values("frame")


def actor_groups(T):
    a = T.meta.actor.values
    return {int(v): np.flatnonzero(a == v) for v in np.unique(a)}


def rng(offset=0):
    return np.random.default_rng(SEED + offset)
