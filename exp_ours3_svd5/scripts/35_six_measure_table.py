#!/usr/bin/env python3
"""Table 2: the six paper interaction discrepancies per arm (05_sc.tex, subsec:metrics).

Measures (per scenario, generated pair G vs replay pair R, ego replayed in both):
  pet     |PET(G) - PET(R)|      signed bbox PET, absolute difference (E0 decision 1);
                                  no-event (inf) or NaN pairs stay NaN, never 0
  dmin    |d_min(G) - d_min(R)|  min simultaneous centre-to-centre distance
  alpha   |alpha(G) - alpha(R)|  unsigned heading difference at closest approach
  cpoint  ||c(G) - c(R)||_2      width-weighted path-nearest conflict point
  uc      |u_c(G) - u_c(R)|      target speed at its sample nearest c (= agent_arr_speed)
  dtw     d_DTW(Y_target, X_target)  exp_cross_coverage/scripts/180_trajdtw_aggregate.py
                                  traj_dtw / arc_resample / dtw2 loaded verbatim by AST

The five non-PET quantities are read from the existing 30/31 scorer outputs and
the PET from the existing 32 outputs; nothing is rescored here. Only the DTW is
computed here, from the saved traces (ours/SAKURA: target rows inside the
metadata window, the same rows the scorer used; SVD: the decoded 50-point
vector, the same decode as scripts/30_interaction_metrics.py) against the GT
target path in the per-subset real_tracks.parquet.

Aggregation (eq. metric-aggregation / composite-discrepancy): for each
comparison set M and measure k, S_k = scenarios with a finite error for every
arm in M (after restricting to clock-aligned rows with score_status ok and PET
status ok); e_bar_{m,k} = median over S_k; b_k = max_m e_bar_{m,k};
D_m = mean_k e_bar_{m,k} / b_k over measures with b_k > 0.

Contrasts: paired median difference (ours minus SVD) on S_k with the 34 cluster
bootstrap (2000 reps, seed 20260910; cluster = global shared ego/target
connected component, 85 in the 513 population) plus a within-class-component
CI as sensitivity. p-values: two-sided cluster sign-flip permutation test of the
median paired difference (all differences inside one cluster flip together;
exact enumeration when 2^G <= 2^16, otherwise 20000 Monte Carlo flips, seed
20260910). Holm (sr-tlkeep-experiment/scripts/41_validity_layers.py holm, AST
loaded) over the 5 classes x 6 measures = 30-cell family per (set, contrast);
the pooled scope is its own 6-cell family. Primary contrast: ours3_disk vs
svd_d5_logo on PRIMARY.

Outputs: results/table2_six_measures_{primary,extended,sensitivity_arc,
sensitivity_gttrunc}.csv, table2_six_measures_cases.csv,
TABLE2_SIX_MEASURES_REPORT.md, table2_manifest.json.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RESULTS = PROJECT / "results"
DTW_SOURCE = ROOT / "exp_cross_coverage/scripts/180_trajdtw_aggregate.py"
HOLM_SOURCE = ROOT / "sr-tlkeep-experiment/scripts/41_validity_layers.py"
BOOT_SOURCE = PROJECT / "scripts/34_compare_fidelity.py"
CASES_SOURCE = RESULTS / "svd_d5_cases.csv"
DISK_POP = RESULTS / "ours3_disk_population.csv"
ARC_POP = RESULTS / "ours3_population.csv"

MEASURES = ["pet", "dmin", "alpha", "cpoint", "uc", "dtw"]
UNITS = dict(pet="s", dmin="m", alpha="deg", cpoint="m", uc="m/s", dtw="m")
PAPER = dict(pet="|dPET|", dmin="|d d_min|", alpha="|d alpha|", cpoint="||d c||", uc="|d u_c|", dtw="DTW(target)")
CLASSES = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr"]
SEED, PERM_MC, EXACT_MAX_GROUPS = 20260910, 20000, 16

ARMS = {
    "ours3_disk": dict(kind="trace", paired=RESULTS / "ours3_disk_population_nonpet_paired.csv",
                       pet=RESULTS / "bbox_pet_disk_paired.csv"),
    "ours3_arc": dict(kind="trace", paired=RESULTS / "ours3_population_nonpet_paired.csv",
                      pet=RESULTS / "bbox_pet_all_methods_paired.csv"),
    "ours3_disk_gttrunc": dict(kind="trace", paired=RESULTS / "ours3_disk_population_gttrunc_nonpet_paired.csv",
                               pet=RESULTS / "bbox_pet_disk_gttrunc_paired.csv"),
    "svd_d5_fullfit": dict(kind="svd", mode="fullfit", paired=RESULTS / "matched_cohort/svd_d5_nonpet_paired.csv",
                           pet=RESULTS / "bbox_pet_matched_paired.csv"),
    "svd_d5_logo": dict(kind="svd", mode="logo", paired=RESULTS / "matched_cohort/svd_d5_nonpet_paired.csv",
                        pet=RESULTS / "bbox_pet_matched_paired.csv"),
    "sakura_bc": dict(kind="trace", paired=RESULTS / "sakura_population_nonpet_paired.csv",
                      pet=RESULTS / "bbox_pet_all_methods_paired.csv"),
}
SETS = {
    "primary": dict(arms=["ours3_disk", "svd_d5_fullfit", "svd_d5_logo"], cohort=["ours3_disk"],
                    contrasts=[("ours3_disk", "svd_d5_logo"), ("ours3_disk", "svd_d5_fullfit")],
                    primary_contrast=("ours3_disk", "svd_d5_logo"),
                    label="PRIMARY: ours3 Euclidean-disk defaults vs matched SVD d5 (469-scene disk cohort)"),
    "extended": dict(arms=["ours3_disk", "svd_d5_fullfit", "svd_d5_logo", "sakura_bc"], cohort=["ours3_disk", "sakura_bc"],
                     contrasts=[("ours3_disk", "svd_d5_logo"), ("ours3_disk", "svd_d5_fullfit"), ("ours3_disk", "sakura_bc")],
                     primary_contrast=("ours3_disk", "svd_d5_logo"),
                     label="EXTENDED: + SAKURA bc on the SAKURA-available scenes (keeptl, keeptl_sw, cutinl)"),
    "sensitivity_arc": dict(arms=["ours3_arc", "svd_d5_fullfit", "svd_d5_logo"], cohort=["ours3_arc"],
                            contrasts=[("ours3_arc", "svd_d5_logo"), ("ours3_arc", "svd_d5_fullfit")],
                            primary_contrast=("ours3_arc", "svd_d5_logo"),
                            label="SENSITIVITY (arc-length window, historical geometry): ours3_arc vs SVD d5 (489-scene arc cohort)"),
    "sensitivity_gttrunc": dict(arms=["ours3_disk_gttrunc", "ours3_disk", "svd_d5_fullfit", "svd_d5_logo"], cohort=["ours3_disk"],
                                contrasts=[("ours3_disk_gttrunc", "svd_d5_logo"), ("ours3_disk_gttrunc", "svd_d5_fullfit"),
                                           ("ours3_disk_gttrunc", "ours3_disk")],
                                primary_contrast=("ours3_disk_gttrunc", "svd_d5_logo"),
                                label="SENSITIVITY (E4-S, GT-support truncation of the ours3_disk target trace) vs ours3_disk and SVD d5"),
}


# ── verbatim helpers loaded from the referenced scripts ─────────────────────
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ast_load(path, names, namespace):
    """Exec only the named top-level function definitions of a script, verbatim."""
    tree = ast.parse(Path(path).read_text())
    picked = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {n.name for n in picked} != set(names):
        raise RuntimeError(f"{path}: expected functions {names}")
    module = ast.Module(body=picked, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return {n.name: hashlib.sha256(ast.get_source_segment(Path(path).read_text(), n).encode()).hexdigest() for n in picked}


DTW_NS = {"np": np, "pd": pd}
DTW_HASHES = ast_load(DTW_SOURCE, ["arc_resample", "dtw2", "traj_dtw"], DTW_NS)
traj_dtw = DTW_NS["traj_dtw"]
HOLM_NS = {"np": np}
HOLM_HASHES = ast_load(HOLM_SOURCE, ["holm"], HOLM_NS)
holm = HOLM_NS["holm"]
_spec = importlib.util.spec_from_file_location("compare_fidelity_34", BOOT_SOURCE)
B34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B34)
bootstrap = B34.bootstrap


# ── per-scenario error table ────────────────────────────────────────────────
class GroundTruth:
    def __init__(self):
        self.cache = {}

    def actor_path(self, source_tracks, scenario_id):
        source = str(Path(source_tracks).resolve())
        if source not in self.cache:
            tracks = pd.read_parquet(source)
            self.cache[source] = {(str(sid), role): g.sort_values("frame")
                                  for (sid, role), g in tracks.groupby(["scenario_id", "role"])}
        return self.cache[source][(str(scenario_id), "actor")]


def load_pet(arm, paired_path, n_rows, cfg):
    pet = pd.read_csv(cfg["pet"])
    pet = pet[pet.input_paired_path.map(lambda p: str(Path(p).resolve())) == str(paired_path.resolve())]
    if cfg["kind"] == "svd":
        pet = pet[pet["mode"].eq(cfg["mode"])]
    pet = pet.set_index("input_case_index")
    return pet


def anchor_family(source):
    s = str(source)
    if "xx" in s or "traj_cross" in s:
        return "crossTraj"
    if "pet" in s or "legacy" in s:
        return "PET"
    return "unknown"


def build_cases():
    cases = pd.read_csv(CASES_SOURCE, dtype={"recording": str})
    fullfit = cases[cases["mode"].eq("fullfit")].drop_duplicates("scenario_uid").set_index("scenario_uid")
    # 85 global components before 2026-09-14; 84 after removing cutinl 39_180 / 39_189 / 1669_1657 at the cohort root
    n_groups = len(json.loads((RESULTS / "svd_d5_manifest.json").read_text())["group_members"])
    assert fullfit.group_id.nunique() == n_groups, (fullfit.group_id.nunique(), n_groups)
    # within-class connected components of shared ego/target identities (sensitivity cluster unit)
    within = {}
    for subset, g in fullfit.groupby("subset"):
        parent = {}
        def find(a):
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for r in g.itertuples():
            # ego and target track ids share one id space; merge on identity, not role
            ra, rb = find(("t", int(r.ego))), find(("t", int(r.actor)))
            if ra != rb:
                parent[ra] = rb
        for uid, r in g.iterrows():
            within[uid] = f"{subset}:{find(('t', int(r.ego)))[1]}"
    fullfit["group_id_within_class"] = pd.Series(within)
    disk_pop = pd.read_csv(DISK_POP).drop_duplicates("scenario_uid").set_index("scenario_uid")
    arc_pop = pd.read_csv(ARC_POP).drop_duplicates("scenario_uid").set_index("scenario_uid")
    anchor = {}
    for uid, r in disk_pop.iterrows():
        anchor[uid] = dict(anchor_source=r.anchor_source, anchor_family=anchor_family(r.anchor_source), anchor_from="ours3_disk_population.csv")
    for uid, r in arc_pop.iterrows():
        if uid not in anchor:
            anchor[uid] = dict(anchor_source=r.geometry_variant, anchor_family=anchor_family(r.geometry_variant), anchor_from="ours3_population.csv(geometry_variant)")
    return fullfit, anchor


def score_arm(arm, cfg, gt, fullfit, anchor):
    paired = pd.read_csv(cfg["paired"])
    paired["input_case_index"] = np.arange(len(paired))
    if cfg["kind"] == "svd":
        paired = paired[paired["mode"].eq(cfg["mode"])]
    pet = load_pet(arm, cfg["paired"], len(paired), cfg)
    if len(pet) != len(paired):
        raise RuntimeError(f"{arm}: PET rows {len(pet)} != paired rows {len(paired)}")
    npz_cache = {}
    rows = []
    t0 = time.monotonic()
    for r in paired.itertuples():
        p = pet.loc[r.input_case_index]
        for key in ("scenario_uid", "subset"):
            if str(getattr(p, key)) != str(getattr(r, key)):
                raise RuntimeError(f"{arm}: PET join mismatch on {key} at row {r.input_case_index}")
        if cfg["kind"] == "svd" and str(p["mode"]) != cfg["mode"]:
            raise RuntimeError("SVD mode mismatch")
        real_pet, gen_pet = float(p.real_pet), float(p.generated_pet)
        both = np.isfinite([real_pet, gen_pet]).all()
        err_pet = abs(real_pet - gen_pet) if both else np.nan
        if both and np.isfinite(p.absdiff_pet) and abs(float(p.absdiff_pet) - err_pet) > 1e-9:
            raise RuntimeError("absdiff_pet disagreement")
        d = {}
        d["err_pet"] = err_pet
        d["err_dmin"] = abs(r.generated_min_dist - r.real_min_dist)
        d["err_alpha"] = abs(r.generated_conflict_angle - r.real_conflict_angle)
        d["err_cpoint"] = float(np.hypot(r.generated_conflict_x - r.real_conflict_x, r.generated_conflict_y - r.real_conflict_y))
        d["err_uc"] = abs(r.generated_agent_arr_speed - r.real_agent_arr_speed)
        for key, col in (("err_dmin", "absdiff_min_dist"), ("err_alpha", "absdiff_conflict_angle"), ("err_uc", "absdiff_agent_arr_speed")):
            ref = float(getattr(r, col))
            if np.isfinite(ref) and abs(ref - d[key]) > 1e-9:
                raise RuntimeError(f"{arm}: {col} disagreement")
        # DTW: generated target path vs GT target path
        gg = gt.actor_path(r.source_tracks, r.scenario_id)
        if cfg["kind"] == "svd":
            npz_path = (PROJECT / r.reconstruction_npz).resolve()
            if npz_path not in npz_cache:
                with np.load(npz_path, allow_pickle=False) as z:
                    npz_cache[npz_path] = {k: z[k] for k in ("X_rec_fullfit", "X_rec_logo", "scenario_uid")}
            z = npz_cache[npz_path]
            idx = int(r.case_index)
            if str(z["scenario_uid"][idx]) != r.scenario_uid:
                raise RuntimeError("SVD npz row identity mismatch")
            vector = z[r.reconstruction_array_key][idx]
            xy = vector[:100].reshape(50, 2)          # time-major (x, y) pairs, same as scripts/30
            rg = pd.DataFrame(dict(x=xy[:, 0], y=xy[:, 1]))
            n_gen = 50
        else:
            tidy = pd.read_parquet(r.trajectory_path)
            lo, hi = float(r.metadata_min_frame), float(r.metadata_max_frame)
            tg = tidy[tidy.role.eq("target") & tidy.frame.between(lo - .5, hi + .5)].sort_values("frame")
            rg = tg[["x", "y"]]
            n_gen = len(tg)
        dtw, cov = traj_dtw(rg, gg) if n_gen >= 2 else (np.nan, np.nan)
        uid = r.scenario_uid
        a = anchor.get(uid, dict(anchor_source="missing", anchor_family="unknown", anchor_from="none"))
        rows.append(dict(arm=arm, subset=r.subset, scenario_uid=uid, scenario_id=r.scenario_id,
                         group_id=fullfit.group_id.get(uid, np.nan),
                         group_id_within_class=fullfit.group_id_within_class.get(uid, np.nan),
                         clock_aligned=bool(r.clock_aligned), score_status=r.score_status,
                         pet_score_status=p.pet_score_status,
                         eligible=bool(r.clock_aligned) and r.score_status == "ok" and p.pet_score_status == "ok",
                         real_pet=real_pet, generated_pet=gen_pet,
                         generated_no_event=bool(np.isinf(gen_pet)), real_no_event=bool(np.isinf(real_pet)),
                         pet_flip=bool(both and np.sign(real_pet) != np.sign(gen_pet)),
                         pet_reason=p.absdiff_pet_reason,
                         real_min_dist=r.real_min_dist, generated_min_dist=r.generated_min_dist,
                         real_conflict_angle=r.real_conflict_angle, generated_conflict_angle=r.generated_conflict_angle,
                         real_conflict_x=r.real_conflict_x, real_conflict_y=r.real_conflict_y,
                         generated_conflict_x=r.generated_conflict_x, generated_conflict_y=r.generated_conflict_y,
                         real_agent_arr_speed=r.real_agent_arr_speed, generated_agent_arr_speed=r.generated_agent_arr_speed,
                         **d, err_dtw=dtw, dtw_coverage=cov, generated_path_samples=n_gen, gt_path_samples=len(gg),
                         anchor_source=a["anchor_source"], anchor_family=a["anchor_family"], anchor_from=a["anchor_from"],
                         geometry_variant_id=getattr(r, "geometry_variant_id", "svd_decoded" if cfg["kind"] == "svd" else "unknown"),
                         input_paired_path=str(cfg["paired"]), input_case_index=int(r.input_case_index)))
    print(f"[{arm}] {len(rows)} scenarios scored, DTW in {time.monotonic() - t0:.1f}s", flush=True)
    out = pd.DataFrame(rows)
    assert out.scenario_uid.is_unique, arm
    return out


# ── statistics ──────────────────────────────────────────────────────────────
def signflip_p(delta, groups):
    """Two-sided cluster sign-flip test of median(delta); flips whole clusters."""
    delta = np.asarray(delta, float)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    if G < 2 or len(delta) == 0:
        return np.nan, G, "n/a"
    obs = abs(np.median(delta))
    if G <= EXACT_MAX_GROUPS:
        n_pat = 1 << G
        count, done = 0, 0
        chunk = 4096
        while done < n_pat:
            ids = np.arange(done, min(done + chunk, n_pat))
            signs = 1 - 2 * ((ids[:, None] >> np.arange(G)[None, :]) & 1)   # (chunk, G) in {+1,-1}
            flipped = delta[None, :] * signs[:, inv]
            count += int(np.sum(np.abs(np.median(flipped, axis=1)) >= obs - 1e-12))
            done = ids[-1] + 1
        return count / n_pat, G, f"exact_{n_pat}"
    rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(PERM_MC):
        signs = rng.choice([-1.0, 1.0], size=G)
        if abs(np.median(delta * signs[inv])) >= obs - 1e-12:
            count += 1
    return (count + 1) / (PERM_MC + 1), G, f"montecarlo_{PERM_MC}"


def tautology_flag(arm, subset_rows, measure):
    """Footnote flag from the ours anchor family of the scenarios in the cell."""
    if not arm.startswith("ours3"):
        return ""
    fam = subset_rows.anchor_family.value_counts(normalize=True)
    flags = []
    if measure in ("cpoint", "uc") and fam.get("crossTraj", 0) > 0:
        flags.append(f"tautology:crossTraj_anchor=conflict_point&u_c_instant({fam['crossTraj']:.0%})")
    if measure == "uc" and fam.get("PET", 0) > 0:
        flags.append(f"tautology:PET_anchor=u_c_instant({fam['PET']:.0%})")
    return ";".join(flags)


def evaluate_set(name, spec, cases):
    arms = spec["arms"]
    wide = {}
    for arm in arms:
        a = cases[cases.arm.eq(arm)].set_index("scenario_uid")
        wide[arm] = a
    cohort = set.intersection(*[set(wide[a].index) for a in spec["cohort"]])
    cohort = sorted(cohort & set.intersection(*[set(wide[a].index) for a in arms]))
    base = wide[arms[0]].loc[cohort]
    eligible = pd.Series(True, index=cohort)
    for arm in arms:
        eligible &= wide[arm].loc[cohort, "eligible"].astype(bool)
    rows, contrasts = [], []
    scopes = [(c, [u for u in cohort if base.subset[u] == c]) for c in CLASSES] + [("pooled", cohort)]
    for scope, uids in scopes:
        uids = [u for u in uids if eligible[u]]
        if not uids:
            continue
        medians = {}
        Sk = {}
        for m in MEASURES:
            col = f"err_{m}"
            finite = pd.Series(True, index=uids)
            for arm in arms:
                finite &= np.isfinite(wide[arm].loc[uids, col].astype(float))
            S = [u for u in uids if finite[u]]
            Sk[m] = S
            medians[m] = {arm: float(np.median(wide[arm].loc[S, col].astype(float))) if S else np.nan for arm in arms}
        bk = {m: (np.nanmax(list(medians[m].values())) if Sk[m] else np.nan) for m in MEASURES}
        Dm = {}
        for arm in arms:
            ratios = [medians[m][arm] / bk[m] for m in MEASURES if Sk[m] and np.isfinite(bk[m]) and bk[m] > 0]
            Dm[arm] = float(np.mean(ratios)) if ratios else np.nan
        for arm in arms:
            A = wide[arm].loc[uids]
            for m in MEASURES:
                S = Sk[m]
                col = f"err_{m}"
                rows.append(dict(comparison_set=name, scope=scope, arm=arm, measure=m, paper_measure=PAPER[m], unit=UNITS[m],
                                 n_cohort=len(base[base.subset.eq(scope)] if scope != "pooled" else base),
                                 n_eligible=len(uids), n_arm_finite=int(np.isfinite(A[col].astype(float)).sum()),
                                 n_common_Sk=len(S),
                                 n_generated_no_event=int(A.generated_no_event.sum()), n_real_no_event=int(A.real_no_event.sum()),
                                 n_both_no_event=int((A.generated_no_event & A.real_no_event).sum()),
                                 n_pet_sign_flip=int(A.pet_flip.sum()),
                                 median_err=medians[m][arm], b_k=bk[m],
                                 normalized=medians[m][arm] / bk[m] if np.isfinite(bk[m]) and bk[m] > 0 else np.nan,
                                 D_m=Dm[arm], n_measures_in_Dm=sum(1 for mm in MEASURES if Sk[mm] and bk[mm] > 0),
                                 dtw_coverage_median=float(A.loc[S, "dtw_coverage"].median()) if m == "dtw" and S else np.nan,
                                 tautology_flag=tautology_flag(arm, A, m)))
        for ours, ref in spec["contrasts"]:
            for m in MEASURES:
                S = Sk[m]
                col = f"err_{m}"
                x = wide[ours].loc[S, col].astype(float).to_numpy()
                y = wide[ref].loc[S, col].astype(float).to_numpy()
                gg = wide[ours].loc[S, "group_id"].astype(str).to_numpy()
                gw = wide[ours].loc[S, "group_id_within_class"].astype(str).to_numpy()
                lo, hi = bootstrap(x, y, gg) if len(S) else (np.nan, np.nan)
                wlo, whi = bootstrap(x, y, gw) if len(S) else (np.nan, np.nan)
                p, G, ptype = signflip_p(x - y, gg) if len(S) else (np.nan, 0, "n/a")
                contrasts.append(dict(comparison_set=name, scope=scope, contrast=f"{ours}_minus_{ref}", ours_arm=ours, ref_arm=ref,
                                      measure=m, paper_measure=PAPER[m], unit=UNITS[m], n_paired=len(S), n_groups_global=G,
                                      n_groups_within_class=len(np.unique(gw)) if len(S) else 0,
                                      ours_median=medians[m][ours], ref_median=medians[m][ref],
                                      median_paired_diff=float(np.median(x - y)) if len(S) else np.nan,
                                      ci95_low_global_group=lo, ci95_high_global_group=hi,
                                      ci95_low_within_class_group=wlo, ci95_high_within_class_group=whi,
                                      ours_lower_fraction=float(np.mean(x < y)) if len(S) else np.nan,
                                      p_signflip_raw=p, signflip_type=ptype,
                                      tautology_flag=tautology_flag(ours, wide[ours].loc[uids], m)))
    contrasts = pd.DataFrame(contrasts)
    # Holm within each (set, contrast) family: 5 classes x 6 measures; pooled separately (6)
    contrasts["p_holm"] = np.nan
    contrasts["holm_family"] = ""
    for (ctr, pooled), g in contrasts.groupby(["contrast", contrasts.scope.eq("pooled")]):
        idx = g.index[np.isfinite(g.p_signflip_raw)]
        if len(idx):
            contrasts.loc[idx, "p_holm"] = holm(contrasts.loc[idx, "p_signflip_raw"].tolist())
        contrasts.loc[g.index, "holm_family"] = f"{name}:{ctr}:{'pooled(6)' if pooled else f'classes({len(idx)})'}"
    contrasts["holm_significant_005"] = contrasts.p_holm < 0.05
    contrasts["favours"] = np.where(contrasts.median_paired_diff < 0, contrasts.ours_arm,
                                    np.where(contrasts.median_paired_diff > 0, contrasts.ref_arm, "tie"))
    contrasts["ci_global_excludes_0"] = (contrasts.ci95_high_global_group < 0) | (contrasts.ci95_low_global_group > 0)
    is_primary = (contrasts.ours_arm.eq(spec["primary_contrast"][0]) & contrasts.ref_arm.eq(spec["primary_contrast"][1]))
    contrasts["primary_contrast"] = is_primary
    return pd.DataFrame(rows), contrasts, cohort


# ── report ──────────────────────────────────────────────────────────────────
def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def report_set(name, spec, table, contrasts, cohort, lines):
    arms = spec["arms"]
    lines.append(f"\n## {name.upper()} — {spec['label']}\n")
    lines.append(f"Cohort: {len(cohort)} scenes (intersection of {spec['cohort']} and all arms). Rows: clock-aligned, score ok, PET ok. "
                 f"S_k = scenes finite for every arm in {arms}.\n")
    for scope in CLASSES + ["pooled"]:
        t = table[table.scope.eq(scope)]
        if t.empty:
            continue
        n_el = int(t.n_eligible.iloc[0])
        lines.append(f"\n### {scope} (n eligible = {n_el})\n")
        head = "| measure | n S_k | " + " | ".join(arms) + " | b_k | flags |"
        lines.append(head)
        lines.append("|" + "---|" * (len(arms) + 4))
        for m in MEASURES:
            tm = t[t.measure.eq(m)].set_index("arm")
            best = tm.median_err.idxmin() if tm.median_err.notna().any() else None
            cells = []
            for arm in arms:
                v = tm.median_err.get(arm, np.nan)
                cells.append(f"**{fmt(v)}**" if arm == best else fmt(v))
            flags = ";".join(sorted({f for f in tm.tautology_flag if f}))
            lines.append(f"| {PAPER[m]} ({UNITS[m]}) | {int(tm.n_common_Sk.iloc[0])} | " + " | ".join(cells) + f" | {fmt(tm.b_k.iloc[0])} | {flags} |")
        dm = t.drop_duplicates("arm").set_index("arm").D_m
        lines.append("| **D_m** | | " + " | ".join(f"**{fmt(dm.get(a, np.nan))}**" for a in arms) + " | | |")
        ne = t[t.measure.eq("pet")].set_index("arm")
        lines.append("| no-event (gen / real / both) | | " + " | ".join(
            f"{int(ne.n_generated_no_event.get(a, 0))} / {int(ne.n_real_no_event.get(a, 0))} / {int(ne.n_both_no_event.get(a, 0))}" for a in arms) + " | | |")
        lines.append("| PET sign flips | | " + " | ".join(f"{int(ne.n_pet_sign_flip.get(a, 0))}" for a in arms) + " | | |")
        c = contrasts[contrasts.scope.eq(scope)]
        if not c.empty:
            lines.append("")
            lines.append("| contrast | measure | n | groups (global/within) | median diff | 95% CI global-group | 95% CI within-class | p raw | p Holm | favours |")
            lines.append("|---|---|---:|---|---:|---|---|---:|---:|---|")
            for r in c.itertuples():
                star = "*" if r.holm_significant_005 else ""
                lines.append(f"| {r.contrast} | {PAPER[r.measure]} | {r.n_paired} | {r.n_groups_global}/{r.n_groups_within_class} | {fmt(r.median_paired_diff)} | "
                             f"[{fmt(r.ci95_low_global_group)}, {fmt(r.ci95_high_global_group)}] | [{fmt(r.ci95_low_within_class_group)}, {fmt(r.ci95_high_within_class_group)}] | "
                             f"{fmt(r.p_signflip_raw, 4)} | {fmt(r.p_holm, 4)}{star} | {r.favours if star else ('(' + r.favours + ')')} |")


def main():
    t0 = time.monotonic()
    RESULTS.mkdir(exist_ok=True)
    fullfit, anchor = build_cases()
    gt = GroundTruth()
    cases = pd.concat([score_arm(arm, cfg, gt, fullfit, anchor) for arm, cfg in ARMS.items()], ignore_index=True)
    cases.to_csv(RESULTS / "table2_six_measures_cases.csv", index=False)
    manifest = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), script=str(Path(__file__).resolve()), script_sha256=sha256(__file__),
                    measures={m: dict(paper=PAPER[m], unit=UNITS[m]) for m in MEASURES},
                    definitions=dict(
                        pet="|PET(G)-PET(R)| with signed bbox PET from scripts/32 outputs (absdiff_pet); non-finite pairs NaN",
                        dmin="|generated_min_dist - real_min_dist| from 30/31 outputs",
                        alpha="|generated_conflict_angle - real_conflict_angle| (unsigned, 0-180 deg) from 30/31 outputs",
                        cpoint="hypot(generated_conflict_xy - real_conflict_xy) from 30/31 outputs",
                        uc="|generated_agent_arr_speed - real_agent_arr_speed| from 30/31 outputs",
                        dtw="traj_dtw(generated target path, GT target path) from 180_trajdtw_aggregate.py: 100 arc-length samples, cost/(200), replay-prefix trim when generated arc length < 0.9 x GT (search within 1.5 x generated length, >=5 samples)",
                        generated_path_trace="target rows with frame in [metadata_min_frame-0.5, metadata_max_frame+0.5] (same rows as scripts/31); gttrunc arm: additionally within_target_gt_support==True via scripts/40",
                        generated_path_svd="X_rec_<mode>[case_index][:100].reshape(50,2) time-major xy, same decode as scripts/30 lines 231-251",
                        gt_path="real_tracks.parquet rows role=='actor' for the scenario_id, sorted by frame",
                        eligibility="clock_aligned & score_status=='ok' & pet_score_status=='ok' for every arm of the comparison set",
                        S_k="scenes finite for every arm in the comparison set, per measure and scope",
                        b_k="max over arms of the per-arm median on S_k",
                        D_m="mean over measures with b_k>0 of median/b_k",
                        contrast="median of (err_ours - err_ref) on S_k; negative favours ours",
                        ci="scripts/34 bootstrap(): 2000 cluster resamples of groups, seed 20260910; cluster = global group_id (85 components in 513) or within-class components (sensitivity)",
                        p="two-sided cluster sign-flip permutation test of median paired difference; exact enumeration of 2^G flips when G<=16 else 20000 Monte-Carlo flips seed 20260910",
                        holm="41_validity_layers.holm over the (set, contrast) family of 5 classes x 6 measures; pooled scope is its own 6-cell family",
                        tautology="flag from ours3_disk_population.csv anchor_source (arc-only scenes: ours3_population.csv geometry_variant): crossTraj anchor (cutinl xx/traj_cross) coincides with the conflict-point descriptor and the u_c instant; PET anchor (velocity_pet/legacy/pet_window/parampath:pet) coincides with the u_c instant"),
                    function_hashes=dict(dtw=DTW_HASHES, holm=HOLM_HASHES, bootstrap_module=sha256(BOOT_SOURCE)),
                    sources={}, n_per_cell={}, sets={})
    for p in [DTW_SOURCE, HOLM_SOURCE, BOOT_SOURCE, CASES_SOURCE, DISK_POP, ARC_POP] + sorted({c["paired"] for c in ARMS.values()} | {c["pet"] for c in ARMS.values()}):
        manifest["sources"][str(p)] = sha256(p)
    for src in cases.input_paired_path.unique():
        manifest["sources"].setdefault(str(src), sha256(src))
    for npz in sorted((PROJECT / "models/svd_d5_matched").glob("*/reconstructions.npz")):
        manifest["sources"][str(npz)] = sha256(npz)
    gttrunc_status = PROJECT / "runs/ours3_disk_population_defaults_gttrunc/1c11c4b192aa321f/batch_status.json"
    if gttrunc_status.exists():
        manifest["sources"][str(gttrunc_status)] = sha256(gttrunc_status)
        manifest["gttrunc_batch_status"] = json.loads(gttrunc_status.read_text())
    for src in sorted({str(Path(s).resolve()) for s in pd.concat([pd.read_csv(c["paired"], usecols=["source_tracks"]) for c in ARMS.values()]).source_tracks.unique()}):
        manifest["sources"][src] = sha256(src)
    lines = ["# Table 2 — six interaction discrepancy measures (05_sc.tex subsec:metrics)", "",
             "Per-scenario errors: PET from the signed bbox-PET scorer outputs (32), d_min / alpha / conflict point / u_c from the non-PET scorer outputs (30/31), DTW of the generated target path against the GT target path computed here with the verbatim 180_trajdtw_aggregate.py routine. "
             "Medians on the per-measure common finite set S_k; b_k = worst arm median; D_m = mean over measures of median/b_k (lower is better). Bold = lowest median in the row. "
             "Contrast = ours minus reference (negative favours ours); 95% CI = 34 cluster bootstrap on the 85 global shared ego/target components (within-class components as sensitivity); "
             "p = two-sided cluster sign-flip permutation test of the median paired difference (exact enumeration when <= 16 clusters, else 20000 Monte-Carlo flips); Holm over 5 classes x 6 measures per (set, contrast); pooled scope Holm-adjusted separately over its 6 cells. "
             "* = Holm-significant at 0.05; favours shown in parentheses when not significant. min_ttc is not a paper measure and is not shown. PET no-event rows (inf) stay NaN and are excluded from S_k; their counts are listed.", ""]
    summary_lines = []
    sets_out = {}
    for name, spec in SETS.items():
        table, contrasts, cohort = evaluate_set(name, spec, cases)
        out = pd.concat([table.assign(row_type="arm_measure"), contrasts.assign(row_type="contrast")], ignore_index=True, sort=False)
        out.to_csv(RESULTS / f"table2_six_measures_{name}.csv", index=False)
        sets_out[name] = (table, contrasts, cohort)
        manifest["sets"][name] = dict(label=spec["label"], arms=spec["arms"], cohort_arms=spec["cohort"], n_cohort=len(cohort),
                                      contrasts=[f"{a}_minus_{b}" for a, b in spec["contrasts"]], primary_contrast="_minus_".join(spec["primary_contrast"]))
        manifest["n_per_cell"][name] = {f"{r.scope}/{r.arm}/{r.measure}": dict(n_eligible=int(r.n_eligible), n_arm_finite=int(r.n_arm_finite), n_common_Sk=int(r.n_common_Sk),
                                                                             n_generated_no_event=int(r.n_generated_no_event), n_real_no_event=int(r.n_real_no_event))
                                        for r in table.itertuples()}
        report_set(name, spec, table, contrasts, cohort, lines)
        # plain statement
        prim = contrasts[contrasts.primary_contrast]
        sig = prim[prim["holm_significant_005"]]
        dm = table.drop_duplicates(["scope", "arm"])[["scope", "arm", "D_m"]]
        summary_lines.append(f"\n### {name}: primary contrast {spec['primary_contrast'][0]} vs {spec['primary_contrast'][1]}\n")
        summary_lines.append("D_m: " + "; ".join(f"{s}: " + ", ".join(f"{r.arm}={fmt(r.D_m)}" for r in dm[dm.scope.eq(s)].itertuples()) for s in CLASSES + ["pooled"] if (dm.scope.eq(s)).any()))
        if sig.empty:
            summary_lines.append("\nNo cell is Holm-significant at 0.05.")
        else:
            summary_lines.append("\nHolm-significant cells (favoured arm, median paired diff, p_Holm):")
            for r in sig.itertuples():
                summary_lines.append(f"- {r.scope} / {PAPER[r.measure]}: favours {r.favours} (diff {fmt(r.median_paired_diff)} {UNITS[r.measure]}, n={r.n_paired}, p_Holm={fmt(r.p_holm, 4)}){' [' + r.tautology_flag + ']' if r.tautology_flag else ''}")
        unadj = prim[~prim["holm_significant_005"] & prim.ci_global_excludes_0]
        if not unadj.empty:
            summary_lines.append("\nCI-excludes-zero but not Holm-significant (exploratory only): " + "; ".join(f"{r.scope}/{PAPER[r.measure]} ({r.favours})" for r in unadj.itertuples()))
    lines.append("\n## Plain statement after Holm\n")
    lines.extend(summary_lines)
    lines.append("\n## Deviations / notes\n")
    lines.append("- Eligibility requires clock alignment and ok status in every arm of the set, so n eligible per class is the disk/arc cohort minus clock-misaligned scenes; S_k further drops scenes where any arm has a non-finite error (PET no-event, estimator NaN).")
    lines.append("- The GT-support truncation arm (E4-S) is a derived batch (scripts/40) scored with the unchanged 31/32; it is a sensitivity row, not a replacement for the primary row.")
    lines.append("- The cluster sign-flip test is exact when a class has <= 16 global components (all 2^G patterns enumerated), so the smallest attainable p in such a class is 2^-G; the achievable minimum is reported through n_groups_global.")
    lines.append("- Tautology flags mark ours cells where the anchor instant used to build the parameterization coincides with the descriptor instant; they are footnotes, the numbers are unchanged.")
    (RESULTS / "TABLE2_SIX_MEASURES_REPORT.md").write_text("\n".join(lines) + "\n")
    manifest["elapsed_s"] = time.monotonic() - t0
    manifest["outputs"] = {str(p): sha256(p) for p in [RESULTS / "table2_six_measures_cases.csv", RESULTS / "TABLE2_SIX_MEASURES_REPORT.md"] + [RESULTS / f"table2_six_measures_{n}.csv" for n in SETS]}
    (RESULTS / "table2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("\n".join(summary_lines))
    print(f"\nelapsed {manifest['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
