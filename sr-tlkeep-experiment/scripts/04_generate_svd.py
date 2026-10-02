"""Generate SVD(d=3) reconstructions and SVD+KDE(dependent) samples from the
906 real scenario vectors (fit on ALL data, per the experiment spec).

Outputs under generated/:
- svd_d3/vectors.npz            : X_rec (906, 101) + source keys
- svd_d3/trajectories.parquet   : reconstructed agent trajectories
- svd_kde/vectors_fidelity.npz  : 10000 samples (paper-style Nw)
- svd_kde/vectors_vc.npz        : 3624 samples (= 906*4, variety/coverage set)
- svd_kde/trajectories_*.parquet
- svd_d3/vectors_vc.npz         : bootstrap selection (Eq. 24 style) to 3624
- results/svd_fit.json          : explained variance, LOO bandwidth
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from core.svd_param import SvdParameterization, split_vector  # noqa: E402
from core.kde_sampling import loo_bandwidth, sample_dependent  # noqa: E402

NT, NY, NTH = 50, 2, 1
FPS = 30.0
D = 3
NW_FID = 10000
SEED = 20260801

dat = np.load(EXP / "data" / "real_vectors.npz")
X_raw, keys = dat["X_raw"], dat["keys"]
meta = pd.read_csv(EXP / "data" / "real_meta.csv").set_index("scenario_id")
N = X_raw.shape[0]
NW_VC = N * 4

P = SvdParameterization(NT, NY, NTH).fit(X_raw)
ev = {d: P.explained_variance(d) for d in range(1, 11)}
print("explained variance:", {k: round(v, 4) for k, v in ev.items()})

V3 = P.reduced(D)
h = loo_bandwidth(V3)
print(f"LOO-CV bandwidth h = {h:.5f}")

rng = np.random.default_rng(SEED)


def to_traj_rows(X, ids, method, tag, center_keys):
    rows = []
    for x, sid, ck in zip(X, ids, center_keys):
        y, theta = split_vector(x, NT, NY, NTH)
        dur = max(float(theta[0]), 0.5)
        mf = int(meta.loc[ck, "min_frame"])
        tq = np.linspace(0.0, dur, NT)
        vx = np.gradient(y[:, 0], tq)
        vy = np.gradient(y[:, 1], tq)
        speed = np.hypot(vx, vy)
        heading = np.degrees(np.arctan2(vy, vx)) % 360.0
        for i in range(NT):
            rows.append((sid, method, tag, "agent", mf + tq[i] * FPS,
                         y[i, 0], y[i, 1], heading[i], speed[i], ck, dur))
    return rows


COLS = ["scenario_id", "method", "tag", "role", "frame", "x", "y",
        "heading_deg", "speed", "center_key", "duration_s"]

# --- SVD(d=3): deterministic rank-3 reconstruction of each real scenario ----
X_rec = P.reconstruct(V3, D)
out = EXP / "generated" / "svd_d3"
np.savez(out / "vectors.npz", X=X_rec, keys=keys, V=V3)
ids = [f"svdrec_{k}" for k in keys]
pd.DataFrame(to_traj_rows(X_rec, ids, "svd_d3", "recon", keys),
             columns=COLS).to_parquet(out / "trajectories.parquet")
rec_err = np.abs(X_rec - X_raw).mean()
print(f"svd_d3: {len(X_rec)} reconstructions, mean abs err {rec_err:.3f}")

# V&C set: selection with replacement (Eq. 24 analogue — adds no new variety)
boot_idx = rng.integers(0, N, size=NW_VC)
np.savez(out / "vectors_vc.npz", X=X_rec[boot_idx], keys=keys[boot_idx],
         boot_idx=boot_idx)

# --- SVD+KDE(dependent) -----------------------------------------------------
out = EXP / "generated" / "svd_kde"
for tag, nw in (("fidelity", NW_FID), ("vc", NW_VC)):
    S, cidx = sample_dependent(V3, h, nw, rng)
    Xg = P.reconstruct(S, D)
    np.savez(out / f"vectors_{tag}.npz", X=Xg, V=S, center_idx=cidx,
             center_keys=keys[cidx], h=h)
    ids = [f"kde{tag[0]}_{i:05d}" for i in range(nw)]
    pd.DataFrame(to_traj_rows(Xg, ids, "svd_kde", tag, keys[cidx]),
                 columns=COLS).to_parquet(out / f"trajectories_{tag}.parquet")
    dneg = int((Xg[:, -1] <= 0.5).sum())
    print(f"svd_kde[{tag}]: {nw} samples, durations<=0.5s clipped: {dneg}")

json.dump({"explained_variance": ev, "d": D, "h_loo": h,
           "n_real": int(N), "nw_fidelity": NW_FID, "nw_vc": int(NW_VC),
           "svd_d3_mean_abs_recon_err": float(rec_err), "seed": SEED},
          open(EXP / "results" / "svd_fit.json", "w"), indent=2)
print("saved svd_fit.json")
