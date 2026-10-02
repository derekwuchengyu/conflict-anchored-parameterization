#!/usr/bin/env python3
"""RareCase cells for the uturn 859_881 group of thesis Table III.

Same definitions as scripts/89_gen_metrics_tables3.py's "RareCase 39_180" block, with the
donor class changed from cutinl to keeptl (859_881 belongs to no fitted class) and with every
arm read from ONE substrate: results/e9_uturn_859_881 (scripts/52 --scenario uturn_859_881).
--radius3 reads the independent L=3 substrate in results/e9_uturn_859_881_l3.
The 39_180 group takes its SAKURA validity from the Table-5 gate instead; here SAKURA is
executed and scored through the same T8 gate as the rest, which is stated in the table note.

  D_int              mean over the five interaction measures of (arm median |delta| / b_k),
                     b_k = the worst median of the group over ALL its arms (reconstruction and
                     KDE alike, the arm set of results/e9_uturn_859_881/T8_composite.json).
                     Printed per row from that row's RECONSTRUCTION arm, as in the 39_180 group.
  DTW [m]            raw median err_dtw (traj_dtw vs the recorded target path).
  variance ratio     mean over the six descriptors of var(z_draws) (ddof=1), z standardized with
                     the keeptl class real mean/sd, so the recorded class variance is 1.
  recovered frac.    share of draws within the keeptl eps of the single recording, joint
                     (interaction AND path); mean over the three seeds.
  valid / bg / offroad   straight from results/e9_uturn_859_881/T8_singleton_table.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
E9 = PROJECT / "results/e9_uturn_859_881"
RADIUS3 = False
MANIFEST = PROJECT / "results/final/tables3/gen_metrics_manifest.json"
DONOR = "keeptl"
DESC = ["pet", "d_min", "alpha", "conflict_x", "conflict_y", "u_c"]
DESC_SRC = dict(pet="pet", d_min="min_dist", alpha="conflict_angle", conflict_x="conflict_x",
                conflict_y="conflict_y", u_c="agent_arr_speed")
MEASURES = ["pet", "dmin", "alpha", "cpoint", "uc"]          # the five of D_int
# thesis row -> (reconstruction arm, KDE arm)
ROWS = [("SAKURA$^{0\\%\\,\\mathrm{routed}}$", "sakura_route", "sakura_route_kde_cond"),
        ("SVD(d=5)", None, None),
        ("SVD(d=5), external basis", "svd_extbasis_recon", "svd_extbasis_gauss_h1"),
        ("Ours", "ours3_recon", "ours3_condkde")]


def med_measures(S):
    """The five |delta| medians of one arm, in the T8 composite's own convention."""
    out = {}
    out["pet"] = np.nanmedian(np.abs(S.err_pet.to_numpy(float))) if "err_pet" in S else np.nan
    out["dmin"] = np.nanmedian(np.abs(S.err_dmin.to_numpy(float))) if "err_dmin" in S else np.nan
    out["alpha"] = np.nanmedian(np.abs(S.err_alpha.to_numpy(float))) if "err_alpha" in S else np.nan
    out["cpoint"] = np.nanmedian(S.err_cpoint.to_numpy(float)) if "err_cpoint" in S else np.nan
    out["uc"] = np.nanmedian(np.abs(S.err_uc.to_numpy(float))) if "err_uc" in S else np.nan
    return out


def main():
    global E9, RADIUS3
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius3", action="store_true", help="read the separate U-turn L=3 scored results")
    parser.add_argument("--label-v2", action="store_true", help="updated 859_881 label in isolated results")
    args = parser.parse_args()
    if args.label_v2 and not args.radius3:
        parser.error("--label-v2 requires --radius3")
    if args.radius3:
        RADIUS3 = True
        E9 = PROJECT / "results/e9_uturn_859_881_l3"
    if args.label_v2:
        E9 = PROJECT / "results/e9_uturn_859_881_l3_labelv2"
    man = json.loads(MANIFEST.read_text())
    sc, eps = man["real_scaling"][DONOR], man["eps"][DONOR]
    mean = np.array([sc["mean"][k] for k in DESC])
    sd = np.array([sc["sd"][k] for k in DESC])
    eps_i, eps_p = float(eps["interaction"]), float(eps["path"])
    S = pd.read_csv(E9 / "T8_samples.csv", low_memory=False)
    T = pd.read_csv(E9 / "T8_singleton_table.csv").set_index("arm")
    comp = json.loads((E9 / "T8_composite.json").read_text())
    # b_k = the worst median of THIS group, over every arm the group prints (reconstruction and
    # KDE alike). T8_composite.json's own b_k is computed over a smaller arm set (it drops arms
    # whose measures are all missing), which would put some ratios above 1.
    group_arms = [a for _l, r, k in ROWS for a in (r, k) if a]
    b = {m: float(np.nanmax([med_measures(S[S.arm.eq(a)])[m] for a in group_arms])) for m in MEASURES}
    rep = S[S.arm.eq("replay")].iloc[0]
    z0 = (np.array([float(rep[DESC_SRC[k]]) for k in DESC]) - mean) / sd

    rows = []
    for label, recon, kde in ROWS:
        r = dict(row=label, recon_arm=recon or "", kde_arm=kde or "")
        if recon:
            A = S[S.arm.eq(recon)]
            m = med_measures(A)
            ratios = {k: (m[k] / b[k]) if np.isfinite(m[k]) and np.isfinite(b[k]) and b[k] > 0 else np.nan
                      for k in MEASURES}
            fin = [v for v in ratios.values() if np.isfinite(v)]
            r["d_int"] = float(np.mean(fin)) if len(fin) == len(MEASURES) else np.nan
            r["d_int_n_measures"] = len(fin)
            r["dtw_m"] = float(np.nanmedian(A.err_dtw.to_numpy(float)))
            r.update({f"med_{k}": m[k] for k in MEASURES})
        else:
            r["d_int"], r["dtw_m"], r["d_int_n_measures"] = np.nan, np.nan, 0
        if kde:
            K = S[S.arm.eq(kde)].copy()
            X = np.column_stack([K[DESC_SRC[k]].to_numpy(float) for k in DESC])
            Z = (X - mean) / sd
            fin6 = np.isfinite(Z).all(axis=1)
            dtw = K.err_dtw.to_numpy(float)
            rec_i = np.full(len(K), False)
            rec_i[fin6] = np.linalg.norm(Z[fin6] - z0, axis=1) <= eps_i
            rec_p = np.isfinite(dtw) & (dtw <= eps_p)
            rec_j = rec_i & rec_p
            seeds = sorted(K.seed.dropna().unique())
            r["recovered_joint"] = float(np.mean([rec_j[K.seed.eq(s).to_numpy()].mean() for s in seeds]))
            r["recovered_interaction"] = float(np.mean([rec_i[K.seed.eq(s).to_numpy()].mean() for s in seeds]))
            r["recovered_path"] = float(np.mean([rec_p[K.seed.eq(s).to_numpy()].mean() for s in seeds]))
            tr = []
            for s in seeds:
                m_ = fin6 & K.seed.eq(s).to_numpy()
                tr.append(float(Z[m_].var(axis=0, ddof=1).mean()) if m_.sum() >= 2 else np.nan)
            r["var_ratio"] = float(np.nanmean(tr)) if np.isfinite(tr).any() else np.nan
            r["n_finite6"] = int(fin6.sum())
            t = T.loc[kde]
            r["n_executed"] = int(t.n_executed)
            r["valid_rate"] = float(t.n_valid) / float(t.n_executed)
            r["bg_collision"] = float(t.bg_solid_rate_all_gtsupport)
            r["offroad"] = float(t.offroad_n) / float(t.n_executed)
        rows.append(r)
    t = T.loc["replay"]
    rows.append(dict(row="real (reference)", recon_arm="replay", kde_arm="", n_executed=1,
                     valid_rate=float(t.n_valid), bg_collision=float(t.bg_solid_rate_all_gtsupport),
                     offroad=float(t.offroad_n)))
    out = pd.DataFrame(rows)
    out.to_csv(E9 / "tables3_rarecase_uturn_cells.csv", index=False)
    meta = dict(donor_class=DONOR, radius_m=3 if RADIUS3 else 5,
                anchor="original conflict point" if RADIUS3 else "crossTraj",
                eps_interaction=eps_i, eps_path=eps_p,
                real_scaling_n=sc["n"], b_k=b, composite_arm_set=comp["arm_set"],
                z0_recorded=z0.tolist(), substrate=str(E9.relative_to(PROJECT)),
                sources=["T8_samples.csv", "T8_singleton_table.csv", "T8_composite.json",
                         "results/final/tables3/gen_metrics_manifest.json"])
    (E9 / "tables3_rarecase_uturn_cells.json").write_text(json.dumps(meta, indent=1) + "\n")
    cols = ["row", "d_int", "d_int_n_measures", "dtw_m", "var_ratio", "recovered_joint",
            "valid_rate", "bg_collision", "offroad"]
    print(out[[c for c in cols if c in out]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
