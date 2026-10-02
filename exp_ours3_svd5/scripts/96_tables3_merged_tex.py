#!/usr/bin/env python3
"""96_tables3_merged_tex.py — merged-row .tex variant of the tables3 main tables.

User request (2026-09-11), two edits on top of the tables rendered by
`scripts/90_tables3_assemble.py`:

  1. drop the RECONSTRUCTION arms' 背景碰撞 / 出地圖 cells (and, in the RareCase table,
     the reconstruction arms' 有效率 cell), so every method collapses from a
     "<method>" + "<method>+KDE" pair into ONE row: the two similarity columns are the
     reconstruction arm, everything to the right of them is that method's KDE arm;
  2. give every column except the two similarity columns a shared header label
     "KDE Sample" (a `\\multicolumn` + `\\cmidrule` band over the right-hand block).

Nothing is recomputed from upstream data: every printed number is copied verbatim from
the rendered `tables3_<table>.csv`, which is what 90_tables3_assemble.py printed.
The BOLD flags are recomputed, because dropping the reconstruction validity cells changes
which row holds the best 背景碰撞 / 出地圖 / 有效率 value; the script asserts that the
similarity / variance / coverage / corner flags are unchanged and reports every validity
flag that moved.

Outputs, next to the inputs in results/final/tables3/:
    tables3_<table>_merged.tex  .md  .csv      for table in leftturn, cutin, rarecase
                                               (+ appendix_tlkeep, same treatment)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final" / "tables3"
NA = "–"
TIE = 1e-12

# the three main tables + the appendix, same treatment
TABLES = ["leftturn", "cutin", "rarecase", "appendix_tlkeep"]
FILE = {t: f"tables3_{t}" for t in TABLES}
TEX_TITLE = {"leftturn": "Left turn", "cutin": "Cut-in", "rarecase": "Rare case",
             "appendix_tlkeep": "Appendix: tlkeep"}
# where the DROPPED reconstruction-arm validity cells survive. D7 (the execution-end triplets)
# holds no special_39_180 rows at all, so the RareCase table has to point at D6 / D9 instead.
DETAIL_BLOCKS = {"rarecase": "blocks D6 / D9"}
DETAIL_BLOCKS_DEFAULT = "blocks D5 / D7"

# reconstruction arm -> its KDE partner. `sakura` (plain) has no KDE arm.
KDE_OF = {"sakura_route": "sakura_route_kde", "svd_d5": "svd_d5_kde",
          "svd_extbasis_kde": "svd_extbasis_kde", "ours": "ours_kde"}
# rows that are a KDE arm with no reconstruction partner: they print "–" in the two similarity
# columns and are NOT counted as having had reconstruction cells dropped by the merge.
KDE_ONLY_ROWS = {"svd_extbasis_kde"}
KDE_KEYS = set(v for v in KDE_OF.values() if v)
# 2026-09-11 (user): SAKURA and SAKURA-route are ONE method, not two baselines. The sakura_route arm
# already runs every scene — AssignRoute where results/route_plan_<cls>.json has a plan, the plain
# NURBS chord where it does not (scripts/85_sakura_arms.py, "210 fallback (no route)") — so it IS
# "route it if you can, don't if you can't", and its superscript is the routed share. The separate
# sakura_plain arm is a DIFFERENT recipe (the 25 rule overrides Agent1_1_SA_EndSpeed to the recorded
# endpoint speed, which the 210 fallback never does), so it is a variant, not SAKURA's unrouted half;
# it keeps its own row in the unmerged tables3_<t>.tex.
ROW_ORDER = ["sakura_route", "svd_d5", "svd_extbasis_kde", "ours", "real"]
GEN_METHODS = [m for m in ROW_ORDER if m != "real"]

SIM_COLS = ["sim_int", "sim_dtw"]

HDR = {
    "sim_int": dict(tex=r"$D_{\mathrm{int}}$ $\downarrow$", cjk=r"互動相似 $D_{\mathrm{int}}$ $\downarrow$",
                    md="互動相似 D_int ↓", rule="lower"),
    "sim_dtw": dict(tex=r"DTW~[m] $\downarrow$", cjk=r"軌跡相似 DTW~[m] $\downarrow$",
                    md="軌跡相似 DTW [m] ↓", rule="lower"),
    "var": dict(tex=r"Var.\ ratio $\to$ 1", cjk=r"變異比 $\to$ 1",
                md="變異比 Variance →1", rule="near1"),
    "cov": dict(tex=r"Cov.@1000 $\uparrow$", cjk=r"涵蓋@1000 $\uparrow$",
                md="涵蓋 Coverage@1000 ↑", rule="higher"),
    "corner": dict(tex=r"Corner cov.@1000 $\uparrow$", cjk=r"角落涵蓋@1000 $\uparrow$",
                   md="角落涵蓋 Corner cov.@1000 ↑", rule="higher"),
    "valid": dict(tex=r"Valid rate $\uparrow$", cjk=r"有效率 $\uparrow$",
                  md="有效率 Valid rate ↑", rule="higher"),
    "coll": dict(tex=r"Bg.\ collision $\downarrow$", cjk=r"背景碰撞 $\downarrow$",
                 md="背景碰撞 Bg. collision ↓", rule="lower"),
    "offroad": dict(tex=r"Off-road $\downarrow$", cjk=r"出地圖 $\downarrow$",
                    md="出地圖 Off-road ↓", rule="lower"),
}
# RareCase renames the coverage column (see 90_tables3_assemble.py COLS_RARE)
HDR_RARE_COV = dict(tex=r"Recovered frac.\ $\uparrow$", cjk=r"還原比例 $\uparrow$",
                    md="還原比例 Recovered frac. ↑", rule="higher")


def hdr(table: str, key: str) -> dict:
    if table == "rarecase" and key == "cov":
        return HDR_RARE_COV
    return HDR[key]


def col_keys(df: pd.DataFrame) -> list[str]:
    """the table's own column order, read off the rendered CSV header."""
    keys = []
    for c in df.columns:
        if c.endswith("_display"):
            k = c[: -len("_display")]
            if k in HDR:
                keys.append(k)
    return keys


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def scenario_tex(label: str) -> str:
    return (label.replace("→", r"$\to$").replace("_", r"\_"))


def method_tex(label: str) -> str:
    if label.startswith("SAKURA-route ("):
        pct = label.split("(")[1].split("%")[0]
        return rf"SAKURA$^{{{pct}\%\,\mathrm{{routed}}}}$"
    return label.replace("_", r"\_")


def method_md(label: str) -> str:
    if label.startswith("SAKURA-route ("):
        return "SAKURA (" + label.split("(", 1)[1]
    return label


# ─────────────────────────────────────────────────────────────────────────────
# merge
# ─────────────────────────────────────────────────────────────────────────────
def merge_table(table: str):
    # the *_display columns are the literal printed strings ("0.000", "–", "undefined") —
    # force dtype=str so pandas does not re-parse "0.867" back into a float and lose the format
    df = pd.read_csv(OUT / f"{FILE[table]}.csv",
                     dtype={f"{k}_display": str for k in HDR} | {f"{k}_bold": str for k in HDR})
    for k in HDR:
        if f"{k}_bold" in df.columns:
            df[f"{k}_bold"] = df[f"{k}_bold"].map({"True": True, "False": False})
    cols = col_keys(df)
    kde_block = [c for c in cols if c not in SIM_COLS]
    assert cols[:2] == SIM_COLS, (table, cols)

    changes, out_rows = [], []
    for gk, gdf in df.groupby("group", sort=False):
        by = {r.method_key: r for r in gdf.itertuples()}
        present = [m for m in ROW_ORDER if m in by]
        rows = []
        for mk in present:
            src = by[mk]
            row = dict(group=gk, scenario=src.scenario_label, scenario_tex=scenario_tex(src.scenario_label),
                       n_class=src.n_class, method=mk, method_label=method_md(src.method_label),
                       method_label_tex=method_tex(src.method_label),
                       kind=("real" if mk == "real" else "gen"), cells={})
            # similarity: from the reconstruction arm itself (the `real` row has none)
            for k in SIM_COLS:
                row["cells"][k] = dict(v=getattr(src, k), t=getattr(src, f"{k}_display"),
                                       n=getattr(src, f"{k}_n"), bold=False,
                                       arm=("(none)" if mk == "real" or mk in KDE_ONLY_ROWS
                                            else f"{mk} (reconstruction)"))
            # keep the superseded detail columns of the reconstruction arm, so the merged .csv is
            # self-contained and the D_m / normalised-DTW footnote stays true for THIS file
            for x in ("sim_dm", "sim_dm_n", "sim_dtw_norm", "sim_dtw_norm_n"):
                row.setdefault("extra", {})[x] = getattr(src, x, None)
            # KDE block: from the method's KDE arm; the `real` reference keeps its own replay cells
            if mk == "real":
                donor, donor_name = src, "real (recorded replay)"
            else:
                kk = KDE_OF[mk]
                donor = by.get(kk) if kk else None
                donor_name = kk if donor is not None else "(no KDE arm)"
            for k in kde_block:
                if donor is None:
                    row["cells"][k] = dict(v=float("nan"), t=NA, n=float("nan"), bold=False,
                                           arm="(no KDE arm)")
                else:
                    row["cells"][k] = dict(v=getattr(donor, k), t=getattr(donor, f"{k}_display"),
                                           n=getattr(donor, f"{k}_n"), bold=False, arm=donor_name)
            rows.append(row)

        # recompute bold: generated rows only, same rules and same exact-tie tolerance as 90_*
        for k in cols:
            rule = hdr(table, k)["rule"]
            cand = [(r, r["cells"][k]["v"]) for r in rows
                    if r["kind"] != "real" and pd.notna(r["cells"][k]["v"])]
            if not cand:
                continue
            if rule == "lower":
                best = min(v for _, v in cand)
                dist = lambda v: abs(v - best)
            elif rule == "higher":
                best = max(v for _, v in cand)
                dist = lambda v: abs(v - best)
            else:
                best = min(abs(v - 1.0) for _, v in cand)
                dist = lambda v: abs(abs(v - 1.0) - best)
            for r, v in cand:
                if dist(v) <= TIE:
                    r["cells"][k]["bold"] = True

        # audit: which flags moved relative to what 90_* printed for the SAME donor cell
        for r in rows:
            for k in cols:
                cell = r["cells"][k]
                if cell["t"] in (NA, "undefined"):
                    continue
                donor_key = r["method"] if (k in SIM_COLS or r["method"] == "real") else KDE_OF[r["method"]]
                was = bool(getattr(by[donor_key], f"{k}_bold"))
                if was != cell["bold"]:
                    changes.append(dict(table=table, group=gk, row=r["method_label"], column=k,
                                        donor=donor_key, value=cell["t"],
                                        was="bold" if was else "plain",
                                        now="bold" if cell["bold"] else "plain"))
        out_rows += rows
    present = [m for m in ROW_ORDER if m in set(df.method_key)]
    return out_rows, cols, kde_block, changes, present


# ─────────────────────────────────────────────────────────────────────────────
# captions / footnotes
# ─────────────────────────────────────────────────────────────────────────────
MERGE_TEX = (r"\textbf{Merged-row variant.} Each method is ONE row: the two similarity columns are its "
             r"reconstruction arm and every column under \textbf{KDE Sample} is that method's KDE arm "
             r"({mapping}). The "
             r"reconstruction arms' own background-collision and off-road rates, which the unmerged table "
             r"printed on the reconstruction rows, are DROPPED here; they are unchanged in "
             r"{tables3_unmerged} and in tables3\_detail {blocks}. ")
MAP_TEX = {"sakura_route": r"SAKURA$\to$SAKURA-route+KDE", "svd_d5": r"SVD\_d5$\to$SVD\_d5+KDE",
           "svd_extbasis_kde": r"SVD\_d5 (ext.\ basis)+KDE is itself a KDE arm",
           "ours": r"Ours$\to$Ours+KDE"}

SIM_DEF = (r"$D_{\mathrm{int}}$ = interaction similarity, the mean over the five interaction measures "
           r"($|\Delta\mathrm{PET}|$, $|\Delta d_{\min}|$, $|\Delta\alpha|$, $\|\Delta c\|$, $|\Delta u_c|$) "
           r"of (arm median / worst-arm median) on the arms' common scene set, unitless and only comparable "
           r"inside its own group; DTW~[m] = trajectory similarity, the raw median DTW between the generated "
           r"and the recorded target path in metres, not normalised; both lower is better, and the six-measure "
           r"$D_m$ they replace is kept in tables3\_detail block D10 and in tables3\_sixmeasures. ")
VAR_DEF = (r"Variance ratio = mean over the six interaction descriptors of "
           r"$\mathrm{var}(z_{\mathrm{gen}})/\mathrm{var}(z_{\mathrm{real}})$ (1.0 = the recorded spread). ")
COV_DEF_CLASS = (r"Coverage and corner coverage = joint (interaction $\wedge$ path) coverage at a budget of "
                 r"1000 draws, real set matched489, mean over 3 seeds $\times$ 20 orderings, corner = the "
                 r"top-20\,\% most nearest-neighbour-isolated reals. ")
COV_DEF_RARE = (r"Recovered fraction = the share of the draws that land inside $\varepsilon$ (joint, "
                r"interaction $\wedge$ path, the cutinl class $\varepsilon$) of the single recorded 39\_180 "
                r"descriptor vector, replacing coverage, which is undefined for one recorded scenario; valid "
                r"rate $=n_{\mathrm{valid}}/n_{\mathrm{executed}}$, replacing corner coverage, with a gate "
                r"that differs per row (see the report). ")
VALID_DEF_CLASS = (r"Background collision = chmed-horizon solid OBB hit rate; off-road = VL rate on the full "
                   r"horizon; both are now KDE-sample rates on every generated row that prints one. ")
VALID_DEF_RARE = (r"Background collision and off-road come from two substrates in this table: T8 "
                  r"(GT-support horizon 2424--2923, driving+shoulder union) for the SVD\_d5 and Ours rows' "
                  r"KDE cells and for the real reference, and Table-5 (chmed-horizon solid OBB hit, "
                  r"full-horizon VL off-road) for the SAKURA-route+KDE cells. ")
BOLD_REAL = (r"Best per column among the generated rows in bold (exact ties bold both); the real replay is a "
             r"reference and is never bolded. -- = {dash_gloss}. The real (reference) row is "
             r"itself a recorded replay, not a KDE draw: it sits under the \textbf{{KDE Sample}} band only as "
             r"the recorded floor for {real_cols}. ")
DASH_WITH_SAKURA = ("the method has no KDE arm (plain SAKURA, which was never KDE-sampled) or the column does "
                    "not apply to the recorded replay")
DASH_NO_SAKURA = "the column does not apply to the recorded replay"

NOTE_MERGE_MD = ("每個方法一列:相似度兩欄 = reconstruction arm,「KDE Sample」帶底下的欄位 = 該方法的 KDE arm。"
                 "{sakura_clause}"
                 "real (reference) 是實錄重放,不是 KDE 取樣,只是借該區塊印出實錄的 {real_cols_zh} 底線。")
NOTE_SAKURA_CLAUSE = "SAKURA(未 route)沒有 KDE arm,所以它的 KDE Sample 欄位全是「–」。"
NOTE_SAKURA_ONE = (
    "SAKURA is ONE method here, not two. The arm behind this row (sakura_route) runs every scene of "
    "the class: where results/route_plan_<class>.json has a plan the NURBS FollowTrajectory is replaced "
    "by an AssignRouteAction over the planned waypoints with constant speed = route length / recorded "
    "travel time, and where there is no plan the same base is rendered as the plain NURBS chord "
    "(scripts/85_sakura_arms.py, \"210 fallback (no route)\"). The superscript is that routed share "
    "({routed}), so the row already means \"route it where a route exists, do not where none does\". "
    "The separate SAKURA row of the unmerged {unmerged} is NOT this row's unrouted half but a different "
    "recipe — the 25 rule additionally overrides Agent1_1_SA_EndSpeed to the recorded target endpoint "
    "speed, which the 210 fallback never does — which is why the two differ even on the scenes neither "
    "routes (e.g. 39_180 at 0 % routed: D_int 0.648 plain vs 0.706 here). That variant scores better "
    "on Cut-in (right) and on 39_180 and is kept, with its own row, in {unmerged}.")
NOTE_DROP = ("Dropped by the merge: the reconstruction arms' 背景碰撞 / 出地圖 cells "
             "({dropped}). They are unchanged in {tables3_unmerged} and in tables3_detail {blocks}; "
             "nothing else in this table moved, except the bold flags of the validity columns, which are "
             "re-competed among the surviving (KDE) values only.")
NOTE_BOLD = ("Bold rule for this table's columns: {lower} → lowest; {higher} → highest; {near1} → closest "
             "to 1.0. The best value is chosen among the generated rows of the same scenario group only; "
             "exact ties bold both; the real reference is never bolded. Because this variant prints only the "
             "KDE-sample validity rates, the 背景碰撞 / 出地圖 bold is now a comparison between KDE arms, not "
             "between a reconstruction arm and a KDE arm as in the unmerged table.")
NOTE_SIM = ("相似度 is split into two columns. 互動相似 D_int = the mean over the FIVE interaction measures "
            "(|ΔPET| s, |Δd_min| m, |Δα| deg, ‖Δc‖ m, |Δu_c| m/s) of (arm median / worst-arm median b_k) on "
            "the arms' common scene set S_k — the same normalisation and the same S_k as the six-measure D_m "
            "it replaces, with DTW simply left out of the average, so it is a unitless 0–1 score and only "
            "comparable inside its own group. 軌跡相似 = the RAW median DTW between the generated and the "
            "recorded target path in metres, not normalised, so it IS comparable across groups and tables. "
            "Both are lower-is-better. The superseded six-measure D_m, the normalised DTW (median/b_k) and "
            "b_k itself are in tables3_detail block D10 and in the `sim_dm` / `sim_dtw_norm` columns of this "
            "table's .csv; D_m also still appears in tables3_sixmeasures.*")
NOTE_COV_CLASS = ("Coverage@1000 and corner coverage@1000: joint space (interaction ∧ path), pool = all draws, "
                  "real_set = matched489, mean over 3 seeds × 20 orderings. Corner = the same rule restricted "
                  "to the top-20 % most nearest-neighbour-isolated real scenarios of the class "
                  "(k = ceil(0.2 · n_rankable)).")
NOTE_VAR = ("變異比 = mean over the six interaction descriptors of var(z_gen)/var(z_real,class) on the draws "
            "with all six descriptors finite; 1.0 = the same spread as the recorded class.")
NOTE_VALID_CLASS = ("背景碰撞 = chmed-horizon solid background hit rate (OBB SAT, >2 overlap frames, "
                    "depth ≥ 0.1 m); 出地圖 = off-road VL rate on the full horizon.")
NOTE_SVD_CLASS = ("In this table the SVD_d5 row's KDE-Sample validity cells are the EXECUTED polytrunc KDE "
                  "variant (svd_exec_E3_kde_polytrunc_kde, 1 500 draws pooled / 300 per class); the "
                  "reconstruction arm's polytrunc rates (svd_exec_E3_recon_polytrunc_fullfit) are the cells "
                  "this variant drops, and the window and analytic counterparts of both are in tables3_detail. "
                  "This convention applies to the class tables only — the RareCase table's SVD_d5 cells come "
                  "from T8, where the KDE arm is undefined at N = 1.")
NOTE_CONV_CLASS = ("執行結束慣例 execution-end convention (背景碰撞 / 出地圖 only): the Ours row's cells are the "
                   "STOP-AT-END KDE execution (ours3_disk_kde_stop_s20260910, 5 000 draws = seed 20260910 only, "
                   "so the Ours validity n is 5 000 and not the 15 000 of the 3-seed window arm), the SVD_d5 "
                   "row's cells are POLYTRUNC, {sakura_conv}the real reference is a replay. "
                   "The mixed convention is conservative against ours: for "
                   "the printed KDE arms, SVD is shown under the convention kindest to it (pooled off-road "
                   "0.251 under polytrunc versus 0.295 under stop-at-end, solid 0.821 versus 0.827), while the "
                   "ours KDE rate is convention-invariant on the same seed (off-road 0.030 under both, solid "
                   "0.691 stop-at-end versus 0.694 window).")
NOTE_N_CLASS = ("Ours validity n = 5 000, not 15 000: only seed 20260910 was re-executed with stop-at-end, so "
                "the 背景碰撞 / 出地圖 cells of the Ours row change denominator AND seed coverage. All three "
                "conventions (3-seed window 15 000, 1-seed window 5 000, 1-seed stop-at-end 5 000) are printed "
                "side by side in tables3_detail block D7. The 變異比 / 涵蓋 / 角落涵蓋 cells of the same row "
                "still use all 3 seeds (15 000 draws) and are unaffected.")
NOTE_STOP_BODY = ("A stopped target is still a body on the road. At the FULL horizon that manufactures "
                  "collisions (pooled any-solid ours 0.563 → 0.738, SVD reconstruction 0.458 → 0.675), but at "
                  "the chmed horizon these tables use the effect is nearly nil: of the printed KDE arms, the "
                  "samples whose only solid hits start after the stop number 17/5 000 (Ours) and 9/1 500 (the "
                  "SVD consistency arm). The 293/5 000 · 52/1 500 figures quoted in RESULT_WP2 are "
                  "FULL-horizon counts.")
NOTE_REAL_CLASS = ("The real (reference) row is Table 5's replay cohort (results/final/T5.csv, arm real; "
                   "50 / 81 / 61 / 22 / 297 replays per class, 511 pooled), NOT the 489-scene cohort WP2 "
                   "re-scored (pooled n = 487 at chmed, 0.037 solid). Both are in tables3_detail block D7.")


APPENDIX_EXTRA = [
    "SAKURA has no tlkeep arm, so this appendix table carries the SVD and ours arms only; the caption's "
    "merge rule is stated for the arms this table actually prints.",
    "tlkeep 相似度 comes from results/final/T2.csv (ours-vs-SVD cohort: ours3_disk, svd_d5_fullfit, "
    "svd_d5_logo) because the 5-arm SAKURA denominator does not exist for tlkeep. 互動相似 D_int is therefore "
    "normalised inside that 3-arm set and is NOT comparable cell-by-cell with the other tables' D_int; "
    "軌跡相似 is a raw metre value and is comparable everywhere.",
]


def real_cols_phrase(cols: list[str]) -> tuple[str, str]:
    """the columns the real (reference) row actually prints, in English and Chinese."""
    has_valid = "valid" in cols
    if has_valid:
        return ("the valid-rate, background-collision and off-road columns", "有效率 / 背景碰撞 / 出地圖")
    return ("the two validity columns", "背景碰撞 / 出地圖")


def build_notes(table: str, cols: list[str], dropped: str, present: list[str],
                routed_note: str = "") -> list[str]:
    unm = f"tables3_{table}.tex"
    blocks = DETAIL_BLOCKS.get(table, DETAIL_BLOCKS_DEFAULT)
    _, real_zh = real_cols_phrase(cols)
    lower = " / ".join(hdr(table, k)["md"].replace(" ↓", "").replace(" ↑", "").replace(" →1", "")
                       for k in cols if hdr(table, k)["rule"] == "lower")
    higher = " / ".join(hdr(table, k)["md"].replace(" ↓", "").replace(" ↑", "").replace(" →1", "")
                        for k in cols if hdr(table, k)["rule"] == "higher")
    near1 = " / ".join(hdr(table, k)["md"].replace(" ↓", "").replace(" ↑", "").replace(" →1", "")
                       for k in cols if hdr(table, k)["rule"] == "near1")
    N = [NOTE_MERGE_MD.format(sakura_clause=(NOTE_SAKURA_CLAUSE if "sakura" in present else ""),
                              real_cols_zh=real_zh),
         NOTE_DROP.format(dropped=dropped, tables3_unmerged=unm, blocks=blocks)]
    if "sakura_route" in present:
        N += [NOTE_SAKURA_ONE.format(routed=routed_note, unmerged=unm)]
    N += [
         NOTE_BOLD.format(lower=lower, higher=higher, near1=near1),
         NOTE_SIM, NOTE_VAR]
    if table == "rarecase":
        N += [
            "This table's 3rd KDE-Sample column is the 還原比例 Recovered frac. = the share of draws inside ε "
            "(joint interaction ∧ path, the cutinl class ε) of the single recorded 39_180 scenario; it replaces "
            "coverage@1000, which is undefined for one recorded scenario, and higher is better. Variance at "
            "n = 1 is the trace ratio of the draws' z standardised by the cutinl class real std.",
            "有效率 Valid rate = n_valid / n_executed (it replaces corner coverage, which is undefined for a "
            "single recorded scenario), and higher is better. In this merged variant it too is a KDE-sample "
            "rate: the reconstruction arms' valid rate (0.000 for SAKURA, SAKURA-route, SVD_d5 and Ours alike, "
            "i.e. the single decoded scene fails the gate in every method) is dropped together with their "
            "背景碰撞 / 出地圖 and is kept in tables3_rarecase.tex and tables3_detail block D6. The gate is not "
            "identical across rows: T8's gate (no teleport / wrong-way / a_lat > 5 / v > 25 / >5 % outside the "
            "driving+shoulder union) for the Ours KDE cells and the real reference; a T8-like approximation "
            "from the Table-5 flags for SAKURA-route+KDE (its full Table-5 gate, 0.013, is in tables3_detail).",
            "背景碰撞 / 出地圖 substrates in this table: T8 (GT-support horizon 2424-2923, "
            "bg_solid_rate_all_gtsupport and offroad_n / n_executed on the 33_e7 driving+shoulder union) for "
            "the Ours row's KDE cells and the real reference; Table-5 (chmed horizon, VL off-road rule) for "
            "SAKURA-route+KDE. Each cell's substrate is in tables3_lineage.csv.",
            "SVD_d5 is `undefined` right of the similarity columns because SVD_d5+KDE is undefined at N = 1: "
            "the centred design matrix has rank 0, the LOO bandwidth selector fails and zero samples exist "
            "(T8 arm svd_d5_singleton_N1). The merge therefore leaves that row with its two similarity cells "
            "only; the executed reconstruction numbers it used to print (valid 0.000, collision 1.000, "
            "off-road 1.000) are in tables3_rarecase.tex and in tables3_detail block D6. The nearest definable "
            "external-basis arm (svd_extbasis_gauss_h1) is in tables3_detail and needs an externally supplied "
            "basis + bandwidth.",
            "執行結束慣例 execution-end convention: this table is NOT stop-at-end. WP2 executed exactly four "
            "class-level stop arms (ours3_disk_stop 469, ours3_disk_kde_stop_s20260910 5 000, "
            "svd_exec_stop_recon_fullfit 489, svd_exec_stop_kde 1 500) and 39_180 is in none of them, so the "
            "Ours KDE cells are the T8 WINDOW executions of ours3_condkde_a2641 (300 draws). The class tables' "
            "convention does not apply here.",
        ]
    else:
        sakura_conv = ("the SAKURA-route row's cells are its original window execution and "
                       if "sakura_route" in present else "")
        N += [NOTE_COV_CLASS, NOTE_VALID_CLASS, NOTE_SVD_CLASS,
              NOTE_CONV_CLASS.format(sakura_conv=sakura_conv), NOTE_N_CLASS,
              NOTE_STOP_BODY, NOTE_REAL_CLASS]
        if table == "appendix_tlkeep":
            N += APPENDIX_EXTRA
    return N


def build_caption(table: str, cols: list[str], present: list[str]) -> str:
    unm = "tables3\\_" + table.replace("_", r"\_") + ".tex"
    blocks = DETAIL_BLOCKS.get(table, DETAIL_BLOCKS_DEFAULT)
    mapping = ", ".join(MAP_TEX[m] for m in present if m in MAP_TEX)
    real_en, _ = real_cols_phrase(cols)
    cap = (f"{TEX_TITLE[table]} scenario group: method comparison. "
           + MERGE_TEX.replace("{tables3_unmerged}", unm).replace("{mapping}", mapping)
                      .replace("{blocks}", blocks)
           + SIM_DEF + VAR_DEF)
    cap += COV_DEF_RARE if table == "rarecase" else COV_DEF_CLASS
    cap += VALID_DEF_RARE if table == "rarecase" else VALID_DEF_CLASS
    cap += BOLD_REAL.format(dash_gloss=(DASH_WITH_SAKURA if "sakura" in present else DASH_NO_SAKURA),
                            real_cols=real_en)
    if table == "rarecase":
        cap += (r"SVD\_d5+KDE is undefined at $N=1$ (centred rank 0, no LOO bandwidth, zero samples), so the "
                r"SVD\_d5 row reads \emph{undefined} right of the similarity columns. Execution-end "
                r"convention: this table is \emph{not} stop-at-end; 39\_180 is in none of WP2's four "
                r"class-level stop arms, so the Ours cells are the T8 window executions of "
                r"ours3\_condkde\_a2641. ")
    else:
        sk = (r"the SAKURA-route row's are its original window execution and "
              if "sakura_route" in present else "")
        cap += (r"The SVD\_d5 row's validity cells are the executed polytrunc KDE variant; the Ours row's are "
                r"the stop-at-end KDE execution (ours3\_disk\_kde\_stop\_s20260910, 5000 draws pooled over the "
                r"five classes = seed 20260910 only, so the Ours validity cells are a 1-seed stop-at-end rate "
                r"and not the 3-seed window arm used for its variance and coverage cells); "
                + sk + r"the real reference is Table 5's 511-replay cohort. The mixed convention is "
                r"conservative against ours: SVD is shown under the convention kindest to it (pooled KDE "
                r"off-road 0.251 polytrunc versus 0.295 stop-at-end) while the ours KDE off-road rate is the "
                r"same 0.030 under both. ")
        if table == "appendix_tlkeep":
            cap += (r"SAKURA has no tlkeep arm, so this table carries the SVD and ours arms only, and its "
                    r"$D_{\mathrm{int}}$ is normalised inside the 3-arm T2 cohort and is not comparable "
                    r"cell-by-cell with the other tables' $D_{\mathrm{int}}$. ")
    return cap


# ─────────────────────────────────────────────────────────────────────────────
# renderers
# ─────────────────────────────────────────────────────────────────────────────
def tex_table(table: str, rows: list[dict], cols: list[str], kde_block: list[str],
              caption: str, notes: list[str]) -> str:
    n_lead = 3 + len(SIM_COLS)                       # Scenario, n, Method, D_int, DTW
    first, last = n_lead + 1, n_lead + len(kde_block)
    spec = "lll" + "r" * len(cols)
    L = ["% auto-generated by scripts/96_tables3_merged_tex.py — do not edit by hand",
         f"% merged-row variant of tables3_{table}.tex: reconstruction-arm validity cells dropped,",
         "% one row per method, shared 'KDE Sample' header band over the non-similarity columns.",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         rf"\caption{{{caption}}}", rf"\label{{tab:tables3-{table}-merged}}",
         rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         rf"\multicolumn{{{n_lead}}}{{c}}{{}} & \multicolumn{{{len(kde_block)}}}{{c}}{{KDE Sample}} \\",
         rf"\cmidrule(lr){{{first}-{last}}}",
         " & ".join(["Scenario", "$n$", "Method"] + [hdr(table, k)["tex"] for k in cols]) + r" \\",
         r"\midrule"]
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r["group"]:
            groups.append((r["group"], []))
        groups[-1][1].append(r)
    for gi, (_gk, grows) in enumerate(groups):
        if gi:
            L.append(r"\midrule")
        for ri, r in enumerate(grows):
            a = rf"\multirow{{{len(grows)}}}{{*}}{{{r['scenario_tex']}}}" if ri == 0 else ""
            b = rf"\multirow{{{len(grows)}}}{{*}}{{{r['n_class']}}}" if ri == 0 else ""
            vals = []
            for k in cols:
                c = r["cells"][k]
                t = c["t"].replace(NA, "--")
                vals.append(rf"\textbf{{{t}}}" if c["bold"] else t)
            L.append(" & ".join([a, b, r["method_label_tex"]] + vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    for f in notes:
        L.append("% note: " + tex_escape(f).replace("\n", " "))
    L += [r"\end{table*}", "",
          "% ---- header variants ----",
          "% (a) with a matching band over the two similarity columns:",
          rf"% \multicolumn{{3}}{{c}}{{}} & \multicolumn{{2}}{{c}}{{Similarity}} & "
          rf"\multicolumn{{{len(kde_block)}}}{{c}}{{KDE Sample}} \\",
          rf"% \cmidrule(lr){{4-{n_lead}}} \cmidrule(lr){{{first}-{last}}}",
          "% (b) Chinese header (needs xeCJK / ctex; compile with xelatex):",
          rf"% \multicolumn{{{n_lead}}}{{c}}{{}} & \multicolumn{{{len(kde_block)}}}{{c}}{{KDE 取樣 KDE Sample}} \\",
          rf"% \cmidrule(lr){{{first}-{last}}}",
          "% " + " & ".join(["場景 Scenario", "$n$", "方法 Method"]
                            + [hdr(table, k)["cjk"] for k in cols]) + r" \\"]
    return "\n".join(L) + "\n"


def md_table(table: str, rows: list[dict], cols: list[str], kde_block: list[str], notes: list[str]) -> str:
    head = ["Scenario 場景", "n", "Method 方法"] + [
        ("KDE·" if k in kde_block else "") + hdr(table, k)["md"] for k in cols]
    out = [f"## Table {TEX_TITLE[table]} — merged rows (KDE Sample band)", "",
           "`KDE·` marks the columns that the .tex prints under the shared **KDE Sample** header band.", "",
           "| " + " | ".join(head) + " |",
           "| " + " | ".join(["---", "---:", "---"] + ["---:"] * len(cols)) + " |"]
    prev = None
    for r in rows:
        scen = r["scenario"] if r["group"] != prev else ""
        nn = str(r["n_class"]) if r["group"] != prev else ""
        prev = r["group"]
        vals = [f"**{r['cells'][k]['t']}**" if r["cells"][k]["bold"] else r["cells"][k]["t"] for k in cols]
        out.append("| " + " | ".join([scen, nn, r["method_label"]] + vals) + " |")
    out += [""] + [f"- {f}" for f in notes]
    return "\n".join(out) + "\n"


def csv_table(table: str, rows: list[dict], cols: list[str]) -> pd.DataFrame:
    recs = []
    for r in rows:
        rec = dict(table=table, group=r["group"], scenario_label=r["scenario"], n_class=r["n_class"],
                   method_key=r["method"], method_label=r["method_label"], row_kind=r["kind"])
        for k in cols:
            c = r["cells"][k]
            rec[k] = c["v"]
            rec[f"{k}_display"] = c["t"]
            rec[f"{k}_bold"] = c["bold"]
            rec[f"{k}_n"] = c["n"]
            rec[f"{k}_arm"] = c["arm"]
        for x in ("sim_dm", "sim_dm_n", "sim_dtw_norm", "sim_dtw_norm_n"):
            rec[x] = (r.get("extra") or {}).get(x)
        recs.append(rec)
    return pd.DataFrame(recs)


def check_pdflatex(files: list[Path]) -> list[dict]:
    exe = shutil.which("pdflatex")
    out = []
    for p in files:
        if not exe:
            out.append(dict(file=p.name, ok=False, note="pdflatex not found"))
            continue
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "t.tex").write_text(p.read_text())
            (td / "main.tex").write_text(
                "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage{multirow}\n"
                "\\usepackage{graphicx}\n\\usepackage[T1]{fontenc}\n\\begin{document}\n"
                "\\input{t.tex}\n\\end{document}\n")
            r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                               cwd=td, capture_output=True, text=True)
            ok = r.returncode == 0
            note = "" if ok else "\n".join(l for l in r.stdout.splitlines() if l.startswith("!"))[:400]
            out.append(dict(file=p.name, ok=ok, note=note))
    return out


def main() -> None:
    all_changes, tex_files = [], []
    for table in TABLES:
        rows, cols, kde_block, changes, present = merge_table(table)
        all_changes += changes
        src = pd.read_csv(OUT / f"{FILE[table]}.csv")
        dropped_bits = []
        for mk in GEN_METHODS:
            if mk in KDE_ONLY_ROWS:      # nothing of theirs was dropped: they ARE the KDE arm
                continue
            s = src[src.method_key == mk]
            if s.empty:
                continue
            keys = ["coll", "offroad"] + (["valid"] if "valid" in cols else [])
            if s[keys].notna().any().any():
                dropped_bits.append(s.method_label.iloc[0].split(" (")[0])
        dropped = ", ".join(dict.fromkeys(dropped_bits))
        routed_note = "; ".join(
            f"{r['scenario']} {r['method_label'].split('(')[1].rstrip(')')}"
            for r in rows if r['method'] == 'sakura_route')
        notes = build_notes(table, cols, dropped, present, routed_note)
        caption = build_caption(table, cols, present)

        tex_p = OUT / f"{FILE[table]}_merged.tex"
        tex_p.write_text(tex_table(table, rows, cols, kde_block, caption, notes))
        (OUT / f"{FILE[table]}_merged.md").write_text(md_table(table, rows, cols, kde_block, notes))
        csv_table(table, rows, cols).to_csv(OUT / f"{FILE[table]}_merged.csv", index=False)
        tex_files.append(tex_p)
        print(f"[merge] {table}: {len(rows)} rows, {len(cols)} data columns "
              f"({len(kde_block)} under KDE Sample) -> {tex_p.name} / .md / .csv")

    ch = pd.DataFrame(all_changes)
    print("\n[bold] flags that moved vs the unmerged tables "
          "(expected: validity columns only, where a reconstruction arm used to win):")
    if ch.empty:
        print("  (none)")
    else:
        print(ch.to_string(index=False))
        bad = ch[~ch.column.isin(["coll", "offroad", "valid"])]
        if not bad.empty:
            print("\n[FAIL] a non-validity bold flag moved:")
            print(bad.to_string(index=False))
            sys.exit(1)

    print("\n[tex] pdflatex compile check:")
    fail = False
    for t in check_pdflatex(tex_files):
        print(f"  {t['file']}: {'OK' if t['ok'] else 'FAILED'} {t.get('note', '')[:200]}")
        fail |= not t["ok"]
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
