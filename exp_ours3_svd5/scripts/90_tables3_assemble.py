#!/usr/bin/env python3
"""90_tables3_assemble.py — stage B of results/final/tables3 (thesis table assembly).

Reads ONLY existing result files and writes the three main thesis tables (左轉 / Cut-in /
RareCase), the tlkeep appendix table, the six-measure detail, the secondary detail, the
per-cell lineage and the report, under results/final/tables3/.

Layout / column / bold definitions come from the frozen workflow spec
(workflows/scripts/tables3-assemble-wf_cbe675b5-e0a.js, CONTEXT + "Stage B" prompt) with the
user's 2026-09-11 amendments recorded in HANDOFF_OPUS_20260911.md:

  A1  no "(TBD)" placeholder sub-row — the method sub-rows are exactly
      SAKURA, SAKURA-route (x % routed), SAKURA-route+KDE, SVD_d5, SVD_d5+KDE, Ours, Ours+KDE,
      then a "real (reference)" line;
  A2  the RareCase 4th column is Valid rate (n_valid / n_executed), not corner coverage;
  A3  SVD_d5 / SVD_d5+KDE background-collision and off-road cells come from the POLYTRUNC
      executed variants; the Ours / Ours+KDE cells come from the STOP-AT-END executions
      produced by WP2 (`ours3_disk_stop` 469, `ours3_disk_kde_stop_s20260910` 5000, seed
      20260910 only).  WP1's first run predated WP2 and used the WINDOW executions, stamped
      PROVISIONAL; WP4 (2026-09-11) swapped the arms in and dropped the stamp.

Which Table-5 arm feeds the validity columns of each thesis row is NOT hard-coded: it lives in
results/final/tables3/validity_arm_map.json.  That file now points the Ours rows at the stop
arms and carries `provisional: false`; each row also carries an `execution_kind` string that is
propagated into every cell note, the lineage file and the report, so a reader can never be in
doubt about which execution-end convention a validity cell used.

Sources (read-only): results/final/{T2,T3,T5,T8}.csv, results/table2_sakura_arms.csv,
results/table3_coverage_sakura.csv, results/table5_validity_summary_sakura.csv,
results/table5_validity_summary_stop.csv (WP2), results/sakura_arms_manifest.json,
results/final/tables3/{gen_metrics.csv,rare_scenes.csv}.  No existing file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ours3_svd5_mpl")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
OUT = RES / "final" / "tables3"
NA = "–"          # en dash = column does not apply to this row
UNDEF = "undefined"
CJK_FONTS = ["Noto Serif CJK JP", "TW-Sung", "AR PL UMing CN", "DejaVu Sans"]
TIE = 1e-12       # bold ties are exact ties only (see caption / report)

# ─────────────────────────────────────────────────────────────────────────────
# source files
# ─────────────────────────────────────────────────────────────────────────────
SRC = {
    "T2": RES / "final" / "T2.csv",
    "T3": RES / "final" / "T3.csv",
    "T5": RES / "final" / "T5.csv",
    "T8": RES / "final" / "T8.csv",
    "T2_SAKURA": RES / "table2_sakura_arms.csv",
    "T3_SAKURA": RES / "table3_coverage_sakura.csv",
    "T5_SAKURA": RES / "table5_validity_summary_sakura.csv",
    "SAKURA_MANIFEST": RES / "sakura_arms_manifest.json",
    "GEN": OUT / "gen_metrics.csv",
    "RARE": OUT / "rare_scenes.csv",
    "T1": RES / "final" / "T1.csv",
}


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except ValueError:
        return str(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def verify_reviewed_pins() -> None:
    """Validate the immutable legacy list, approved delta, and current v3 files."""
    review_path = PROJECT / "HANDOFF_PINS_V3_20260914_REVIEW.json"
    if sha256(review_path) != "929c17895b8ba583a96556259df963ecf74b1862d9b7bcda5fcb2bcdfb5d17a0":
        raise SystemExit("[hash] approved v3 review changed")
    review = json.loads(review_path.read_text())
    legacy = PROJECT / "HANDOFF_FILE_HASHES.sha256"
    if sha256(legacy) != review["legacy_manifest_sha256"]:
        raise SystemExit("[hash] historical pin list changed")

    def read_pins(path):
        pairs = [line.split(None, 1) for line in path.read_text().splitlines() if line.strip()]
        pins = {name: digest for digest, name in pairs}
        if len(pairs) != 33 or len(pins) != 33:
            raise SystemExit(f"[hash] expected 33 unique pins: {path}")
        return pins

    old = read_pins(legacy)
    current = read_pins(PROJECT / "HANDOFF_FILE_HASHES_V3_20260914.sha256")
    entries = review["entries"]
    if len(entries) != 33 or {e["path"] for e in entries} != set(old) or set(current) != set(old):
        raise SystemExit("[hash] reviewed file set differs from historical list")
    differences = 0
    for entry in entries:
        name = entry["path"]
        if old[name] != entry["legacy_sha256"] or current[name] != entry["v3_sha256"]:
            raise SystemExit(f"[hash] unreviewed pin value: {name}")
        if sha256(PROJECT / name) != current[name]:
            raise SystemExit(f"[hash] v3 file changed: {name}")
        differences += old[name] != current[name]
    if differences != 9:
        raise SystemExit("[hash] expected exactly 9 approved historical differences")
    print("[hash] legacy list unchanged; 24 unchanged / 9 reviewed differences; v3 33 OK / 0 FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# groups, method sub-rows, columns
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = [
    dict(key="keeptl", table="leftturn", label="Agent-LT N→E", tex=r"Agent-LT N$\to$E", n_class=50),
    dict(key="keeptl_sw", table="leftturn", label="Agent-LT S→W", tex=r"Agent-LT S$\to$W", n_class=82),
    dict(key="cutinl", table="cutin", label="Cut-in (left)", tex="Cut-in (left)", n_class=58),   # 61 before the 2026-09-14 v3 exclusion of 39_180 / 39_189 / 1669_1657
    dict(key="cutinr", table="cutin", label="Cut-in (right)", tex="Cut-in (right)", n_class=20),
    dict(key="special_39_180", table="rarecase", label="Special 39_180", tex=r"Special 39\_180", n_class=1),
    dict(key="tlkeep", table="appendix_tlkeep", label="Agent-straight (ego LT) tlkeep",
         tex=r"Agent-straight (ego LT) tlkeep", n_class=298),
]
GROUP_BY_KEY = {g["key"]: g for g in GROUPS}

TABLES = {
    "leftturn": dict(file="tables3_leftturn", title="左轉 (left turn)", tex_title="Left turn",
                     groups=["keeptl", "keeptl_sw"]),
    "cutin": dict(file="tables3_cutin", title="Cut-in", tex_title="Cut-in", groups=["cutinl", "cutinr"]),
    "rarecase": dict(file="tables3_rarecase", title="RareCase", tex_title="Rare case",
                     groups=["special_39_180"]),
    "appendix_tlkeep": dict(file="tables3_appendix_tlkeep", title="Appendix — tlkeep",
                            tex_title="Appendix: tlkeep", groups=["tlkeep"]),
}

# method sub-rows, in the required order (amendment A1: no "(TBD)" row)
METHODS = [
    dict(key="sakura", label="SAKURA", kind="recon"),
    dict(key="sakura_route", label="SAKURA-route", kind="recon"),      # label gets (x % routed)
    dict(key="sakura_route_kde", label="SAKURA-route+KDE", kind="kde"),
    dict(key="svd_d5", label="SVD_d5", tex_label=r"SVD\_d5", kind="recon"),
    dict(key="svd_d5_kde", label="SVD_d5+KDE", tex_label=r"SVD\_d5+KDE", kind="kde"),
    # 2026-09-11: the charitable, DEFINABLE SVD arm for the N=1 RareCase group. SVD_d5+KDE itself
    # cannot exist at N=1 (centred rank 0), so printing only "undefined" there compares Ours to an
    # absent row. This arm hands SVD what N=1 denies it — a basis and a bandwidth from OUTSIDE the
    # scenario (cutinl LOGO basis over 44 scenes, 39_180's identity group excluded; h = h_loo =
    # 0.0886) — and is executed at the same 300 draws, same T8 substrate and same epsilon as
    # Ours+KDE, so the two are information-matched. RareCase groups only.
    dict(key="svd_extbasis_kde", label="SVD_d5 ext. basis+KDE",
         tex_label=r"SVD\_d5 (ext.\ basis)+KDE", kind="kde"),
    dict(key="ours", label="Ours", kind="recon"),
    dict(key="ours_kde", label="Ours+KDE", kind="kde"),
    dict(key="real", label="real (reference)", kind="real"),
]
# tlkeep has no SAKURA arm at all
APPENDIX_METHODS = ["svd_d5", "svd_d5_kde", "ours", "ours_kde", "real"]
# methods that exist for the single-scenario RareCase group only (no class-table counterpart)
RARE_ONLY_METHODS = {"svd_extbasis_kde"}

# similarity arm per method (reconstruction arms only)
SIM_ARM = {"sakura": "sakura_plain", "sakura_route": "sakura_route",
           "svd_d5": "svd_d5_fullfit", "ours": "ours3_disk"}
# generative arm per method (variance / coverage / corner)
GEN_ARM = {"sakura_route_kde": "sakura_route_kde",
           "svd_d5_kde": "svd_d5_kde_matched_analytic",
           "ours_kde": "ours3_disk_kde"}
# RareCase generative arm per method
RARE_GEN_ARM = {"sakura_route_kde": "sakura_route_kde_cond_39_180",
                "ours_kde": "ours3_condkde_a2641",
                "svd_extbasis_kde": "svd_extbasis_gauss_h1"}

# 2026-09-11 (user request): the single 相似度 D_m column is SPLIT into two printed columns —
#   互動相似 D_int  = mean of the normalised medians of the FIVE interaction measures
#                     (|ΔPET|, |Δd_min|, |Δα|, ‖Δc‖, |Δu_c|), same worst-arm normalisation and
#                     same common scene set S_k as the old D_m, DTW simply left out;
#   軌跡相似 DTW [m] = the raw median DTW to the recorded target path, in metres (not normalised).
# The old six-measure D_m and the normalised DTW (median/b_k) are kept as DETAIL only:
# tables3_detail block D10, the sim_dm / sim_dtw_norm columns of the four table CSVs, and the
# existing tables3_sixmeasures.* (which still prints D_m and the raw DTW median side by side).
COLS_CLASS = [
    dict(key="sim_int", md="互動相似 D_int ↓", png="互動相似 D_int ↓",
         tex=r"$D_{\mathrm{int}}$ $\downarrow$", tex_cjk=r"互動相似 $D_{\mathrm{int}}$ $\downarrow$",
         rule="lower"),
    dict(key="sim_dtw", md="軌跡相似 DTW [m] ↓", png="軌跡相似 DTW [m] ↓",
         tex=r"DTW~[m] $\downarrow$", tex_cjk=r"軌跡相似 DTW~[m] $\downarrow$", rule="lower"),
    dict(key="var", md="變異比 Variance →1", png="變異比 →1", tex=r"Var.\ ratio $\to$ 1",
         tex_cjk=r"變異比 $\to$ 1", rule="near1"),
    dict(key="cov", md="涵蓋 Coverage@1000 ↑", png="涵蓋@1000 ↑", tex=r"Cov.@1000 $\uparrow$",
         tex_cjk=r"涵蓋@1000 $\uparrow$", rule="higher"),
    dict(key="corner", md="角落涵蓋 Corner cov.@1000 ↑", png="角落涵蓋@1000 ↑",
         tex=r"Corner cov.@1000 $\uparrow$", tex_cjk=r"角落涵蓋@1000 $\uparrow$", rule="higher"),
    dict(key="coll", md="背景碰撞 Bg. collision ↓", png="背景碰撞 ↓", tex=r"Bg.\ collision $\downarrow$",
         tex_cjk=r"背景碰撞 $\downarrow$", rule="lower"),
    dict(key="offroad", md="出地圖 Off-road ↓", png="出地圖 ↓", tex=r"Off-road $\downarrow$",
         tex_cjk=r"出地圖 $\downarrow$", rule="lower"),
]
_CC = {c["key"]: c for c in COLS_CLASS}
COLS_RARE = [
    _CC["sim_int"],
    _CC["sim_dtw"],
    _CC["var"],
    dict(key="cov", md="還原比例 Recovered frac. ↑", png="還原比例 ↑",
         tex=r"Recovered frac.\ $\uparrow$", tex_cjk=r"還原比例 $\uparrow$", rule="higher"),
    dict(key="valid", md="有效率 Valid rate ↑", png="有效率 ↑", tex=r"Valid rate $\uparrow$",
         tex_cjk=r"有效率 $\uparrow$", rule="higher"),
    _CC["coll"],
    _CC["offroad"],
]

MEASURES = [
    ("pet", "PET [s]", r"PET~[s]"),
    ("dmin", "min dist [m]", r"min.\ distance~[m]"),
    ("cpoint", "conflict pt [m]", r"conflict point~[m]"),
    ("alpha", "angle [deg]", r"conflict angle~[$^\circ$]"),
    ("uc", "arr speed [m/s]", r"arrival speed~[m/s]"),
    ("dtw", "DTW [m]", r"DTW~[m]"),
]
# the five INTERACTION measures (everything but the path measure DTW) — the 互動相似 D_int average
INT_MEASURES = ["pet", "dmin", "alpha", "cpoint", "uc"]
SIX_METHODS = [
    ("sakura_plain", "SAKURA"),
    ("sakura_route", "SAKURA-route"),
    ("svd_d5_fullfit", "SVD_d5 (fullfit)"),
    ("svd_d5_logo", "SVD_d5 (LOGO)"),
    ("ours3_disk", "Ours"),
]


# ─────────────────────────────────────────────────────────────────────────────
# validity arm map (WP4 edits this file only)
# ─────────────────────────────────────────────────────────────────────────────
def default_arm_map() -> dict:
    t5 = dict(path="results/final/T5.csv", class_column="class", arm_column="arm",
              collision_column="solid_hit_rate_chmed", collision_filters={},
              offroad_column="offroad_vl_rate_full", offroad_filters={},
              valid_column="valid_all_rate_full", valid_filters={},
              n_column="n_samples", n_scenarios_column="n_scenarios", label_column="arm_label",
              collision_definition="chmed-horizon solid background hit (OBB SAT, >2 overlap frames, depth >= 0.1 m, vehicles only)",
              offroad_definition="off-road VL rate, full horizon (>5 % of window-clipped 30 Hz points outside the drivable union)",
              valid_definition="Table-5 valid_all (full horizon)")
    sak = dict(path="results/table5_validity_summary_sakura.csv", class_column="cls", arm_column="arm",
               collision_column="any_solid_rate", collision_filters={"horizon": "chmed"},
               offroad_column="offroad_vl_rate", offroad_filters={"horizon": "full"},
               valid_column="valid_all_rate", valid_filters={"horizon": "full"},
               n_column="n", n_scenarios_column="n_scenarios", label_column="arm_label",
               collision_definition=t5["collision_definition"],
               offroad_definition=t5["offroad_definition"],
               valid_definition="Table-5 valid_all (full horizon)")
    stop = dict(t5)
    stop.update(path="results/table5_validity_summary_stop.csv", class_column="cls", arm_column="arm",
                collision_column="any_solid_rate", collision_filters={"horizon": "chmed"},
                offroad_column="offroad_vl_rate", offroad_filters={"horizon": "full"},
                valid_column="valid_all_rate", valid_filters={"horizon": "full"},
                n_column="n", n_scenarios_column="n_scenarios",
                _status="produced by WP2 (scripts/93_stop_at_end.py + scripts/94_stop_score.py); the Ours "
                        "rows of the CLASS tables resolve here",
                _cohort="the 489 scenes covered by the four stop arms; the file also carries the window and "
                        "polytrunc arms, re-scored on the same substrate (identical to results/final/T5.csv "
                        "for every shared arm) and a real reference on the 489-scene cohort (pooled n = 487 "
                        "at chmed) which the tables do NOT use — the tables keep Table 5's real row")
    return {
        "schema_version": 1,
        "generated_by": "scripts/90_tables3_assemble.py",
        "purpose": "Maps every thesis method sub-row to the arm whose Table-5 scoring supplies its "
                   "background-collision / off-road (and RareCase valid-rate) cells. Editing this file "
                   "and re-running scripts/90_tables3_assemble.py is the only supported way to change "
                   "which execution feeds a validity cell.",
        "provisional": False,
        "provisional_stamp": "PROVISIONAL (ours validity = window execution)",
        "execution_end_convention": {
            "class_tables": "Ours / Ours+KDE = STOP-AT-END (the target brakes to 0 when its trajectory ends, "
                            "WP2 arms ours3_disk_stop and ours3_disk_kde_stop_s20260910); SVD_d5 / SVD_d5+KDE = "
                            "POLYTRUNC (the post-trajectory rows are deleted); SAKURA rows and the real "
                            "reference = WINDOW (their original executions / replays).",
            "rarecase_table": "WINDOW throughout. WP2's stop-at-end arms are class-level and do not contain "
                              "39_180, so neither the Ours row (Table-5 window execution of the single scene) "
                              "nor the Ours+KDE row (T8 window executions of ours3_condkde_a2641) was "
                              "re-executed with stop-at-end.",
            "seed_restriction": "ours3_disk_kde_stop_s20260910 is SEED 20260910 ONLY (5000 draws). The "
                                "PROVISIONAL WP1 run used ours3_disk_kde = 3 seeds x 5000 = 15000 draws, so the "
                                "Ours+KDE validity n drops from 15000 to 5000. Only the validity columns are "
                                "affected: 變異比 / 涵蓋 / 角落涵蓋 still use all 3 seeds.",
            "unaffected_columns": "similarity (D_m), variance ratio, coverage@1000 and corner coverage@1000 "
                                  "all come from the original window executions and are unchanged by the "
                                  "stop-at-end work.",
        },
        "wp4_applied": {
            "date": "2026-09-11",
            "changes": [
                "class_rows.ours.source T5 -> T5_STOP, arm ours3_disk -> ours3_disk_stop",
                "class_rows.ours_kde.source T5 -> T5_STOP, arm ours3_disk_kde -> ours3_disk_kde_stop_s20260910",
                "class_rows.ours.provisional / class_rows.ours_kde.provisional true -> false",
                "top-level provisional true -> false (PROVISIONAL stamps removed from every caption and title)",
                "every row gained an execution_kind string, propagated into the cell notes and the lineage",
            ],
            "unchanged": [
                "SVD_d5 / SVD_d5+KDE stay on the polytrunc arms",
                "every SAKURA row",
                "the real (reference) row (Table 5's replay cohort)",
                "every rarecase_rows entry (39_180 was NOT re-executed with stop-at-end)",
            ],
            "rerun": "MPLCONFIGDIR=/tmp/ours3_svd5_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
                     "/home/hcis-s19/micromamba/envs/nps/bin/python -B scripts/90_tables3_assemble.py",
        },
        "sources": {"T5": t5, "T5_SAKURA": sak, "T5_STOP": stop},
        "class_rows": {
            "sakura": dict(resolver="table5", source="T5_SAKURA", arm="sakura_plain", provisional=False,
                           execution_kind="window (original SAKURA execution)",
                           note="SAKURA = sakura_bc (192) + sakura_plain_extra"),
            "sakura_route": dict(resolver="table5", source="T5_SAKURA", arm="sakura_route", provisional=False,
                                 execution_kind="window (original SAKURA-route execution)"),
            "sakura_route_kde": dict(resolver="table5", source="T5_SAKURA", arm="sakura_route_kde",
                                     provisional=False,
                                     execution_kind="window (original SAKURA-route+KDE execution)",
                                     note="3 seeds x 1000 per class, executed"),
            "svd_d5": dict(resolver="table5", source="T5", arm="svd_exec_E3_recon_polytrunc_fullfit",
                           provisional=False, execution_kind="polytrunc",
                           note="amendment A3: executed E3 timed-Polyline, polytrunc variant (window and "
                                "stop-at-end variants in tables3_detail block D7)"),
            "svd_d5_kde": dict(resolver="table5", source="T5", arm="svd_exec_E3_kde_polytrunc_kde",
                               provisional=False, execution_kind="polytrunc",
                               note="amendment A3: executed E3 KDE, polytrunc variant (window, stop-at-end and "
                                    "analytic variants in tables3_detail block D7)"),
            "ours": dict(resolver="table5", source="T5_STOP", arm="ours3_disk_stop", provisional=False,
                         execution_kind="stop-at-end",
                         window_arm="ours3_disk",
                         batch="runs/ours3_disk_stop/f8305650479ea9c2",
                         note="WP2 stop-at-end execution of the 469 disk-window scenes (window counterpart "
                              "ours3_disk in tables3_detail block D7)"),
            "ours_kde": dict(resolver="table5", source="T5_STOP", arm="ours3_disk_kde_stop_s20260910",
                             provisional=False, execution_kind="stop-at-end (seed 20260910 only, 5000 draws)",
                             window_arm="ours3_disk_kde_s20260910_window",
                             batch="runs/ours3_disk_kde_stop_s20260910/b8af10e674c24622",
                             note="WP2 stop-at-end execution, SEED 20260910 ONLY — the validity n is 5000, not "
                                  "the 15000 of the 3-seed window arm ours3_disk_kde; all three conventions "
                                  "(3-seed window 15000, 1-seed window 5000, 1-seed stop 5000) are in "
                                  "tables3_detail block D7"),
            "real": dict(resolver="table5", source="T5", arm="real", provisional=False,
                         execution_kind="recorded replay (Table 5 cohort, 511 replays pooled)",
                         note="recorded target replayed through the same chassis; never bolded. NOT WP2's "
                              "489-scene cohort (pooled n = 487 at chmed), which is in tables3_detail block D7"),
        },
        "rarecase_rows": {
            "_note": "NONE of these rows is stop-at-end. WP2 executed exactly four class-level stop arms "
                     "(ours3_disk_stop 469, ours3_disk_kde_stop_s20260910 5000, svd_exec_stop_recon_fullfit 489, "
                     "svd_exec_stop_kde 1500); 39_180 is outside all four cohorts, so every RareCase validity "
                     "cell is a WINDOW execution and the table says so.",
            "sakura": dict(resolver="table5", source="T5_SAKURA", arm="sakura_plain",
                           valid_gate="Table-5 valid_all (full horizon)", provisional=False,
                           execution_kind="window (not re-executed with stop-at-end)"),
            "sakura_route": dict(resolver="table5", source="T5_SAKURA", arm="sakura_route",
                                 valid_gate="Table-5 valid_all (full horizon)", provisional=False,
                                 execution_kind="window (not re-executed with stop-at-end)"),
            "sakura_route_kde": dict(resolver="table5", source="T5_SAKURA", arm="sakura_route_kde_cond_39_180",
                                     valid_resolver="gen", valid_arm="sakura_route_kde_cond_39_180",
                                     valid_metric="valid_rate",
                                     valid_gate="T8-like approximation from Table-5 flags (no teleport / wrong-way / "
                                                "physics gate1 / off-road VL); full Table-5 gate 0.013 in the detail",
                                     provisional=False,
                                     execution_kind="window (not re-executed with stop-at-end)"),
            "svd_d5": dict(resolver="t8", arm="svd_matched_fullfit", valid_gate="T8 valid (scripts/52)",
                           provisional=False,
                           execution_kind="window / T8 substrate (not polytrunc, not stop-at-end)",
                           note="analytic decode of the class basis; svd_matched_logo in the detail"),
            "svd_d5_kde": dict(resolver="undefined", arm="svd_d5_singleton_N1",
                               execution_kind="n/a (no samples exist)",
                               note="N=1: centred rank 0, LOO bandwidth undefined, zero samples (T8)"),
            "svd_extbasis_kde": dict(resolver="t8", arm="svd_extbasis_gauss_h1",
                                     valid_gate="T8 valid (scripts/52)", provisional=False,
                                     execution_kind="window (not re-executed with stop-at-end)",
                                     note="the definable stand-in for SVD_d5+KDE at N = 1: cutinl LOGO basis "
                                          "(44 scenes, 39_180's identity group excluded) + h_ext = h_loo = "
                                          "0.0886, both supplied from OUTSIDE the scenario; 300 draws on the "
                                          "same T8 substrate as Ours+KDE"),
            "ours": dict(resolver="gen", arm="ours3_disk", collision_metric="bg_solid_rate",
                         offroad_metric="offroad_rate", valid_metric="valid_rate_table5_gate",
                         valid_gate="Table-5 valid_all (full horizon)", provisional=False,
                         execution_kind="window (NOT re-executed with stop-at-end)",
                         note="Table-5 window execution of the single 39_180 scene. WP2's stop-at-end arms are "
                              "class-level and do not contain 39_180, so this row stays window — unlike the "
                              "Ours row of the class tables"),
            "ours_kde": dict(resolver="t8", arm="ours3_condkde_a2641", valid_gate="T8 valid (scripts/52)",
                             provisional=False,
                             execution_kind="window (NOT re-executed with stop-at-end)",
                             note="T8 window executions of ours3_condkde_a2641 (300 draws). WP2's stop-at-end "
                                  "arms are class-level and do not contain 39_180, so this row stays window"),
            "real": dict(resolver="t8", arm="replay", valid_gate="T8 valid (scripts/52)", provisional=False,
                         execution_kind="recorded replay (T8 substrate)"),
        },
    }


def load_arm_map(path: Path, rewrite: bool) -> dict:
    if rewrite or not path.exists():
        path.write_text(json.dumps(default_arm_map(), indent=2, ensure_ascii=False) + "\n")
        print(f"[map] wrote default {rel(path)}")
    m = json.loads(path.read_text())
    assert m.get("schema_version") == 1, "validity_arm_map.json: unsupported schema_version"
    return m


# ─────────────────────────────────────────────────────────────────────────────
# cell plumbing
# ─────────────────────────────────────────────────────────────────────────────
class Cell:
    __slots__ = ("value", "text", "source_file", "selector", "source_column", "upstream", "note",
                 "bold", "n", "kind")

    def __init__(self, value=None, text=None, source_file="", selector="", source_column="",
                 upstream="", note="", n=None, kind="value"):
        self.value = None if value is None or (isinstance(value, float) and not np.isfinite(value)) else float(value)
        self.text = text
        self.source_file = source_file
        self.selector = selector
        self.source_column = source_column
        self.upstream = upstream
        self.note = note
        self.n = n
        self.kind = kind          # value | na | undefined
        self.bold = False

    @property
    def display(self) -> str:
        if self.text is not None:
            return self.text
        if self.value is None:
            return NA
        return f"{self.value:.3f}"


def na_cell(reason: str) -> Cell:
    return Cell(text=NA, kind="na", note=reason, source_file="(not applicable)",
                selector="(no source row: " + reason + ")", source_column="(none)")


def sel(**kw) -> str:
    return ", ".join(f"{k}={v}" for k, v in kw.items())


def pick(df: pd.DataFrame, name: str, **filters) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for k, v in filters.items():
        m &= df[k].astype(str).eq(str(v))
    sub = df[m]
    if len(sub) != 1:
        raise SystemExit(f"[pick] {name}: {sel(**filters)} matched {len(sub)} rows (expected 1)")
    return sub.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_all() -> dict:
    d = {k: (pd.read_csv(v) if v.suffix == ".csv" else json.loads(v.read_text())) for k, v in SRC.items()}
    # self-checks against the other Table-5 substrate
    s = pd.read_csv(RES / "table5_validity_summary.csv")
    t5 = d["T5"]
    bad = []
    for _, r in t5.iterrows():
        m = s[(s.cls == r["class"]) & (s.arm == r["arm"]) & (s.horizon == "chmed")]
        if len(m) == 1 and np.isfinite(r["solid_hit_rate_chmed"]):
            if abs(float(m.iloc[0]["any_solid_rate"]) - float(r["solid_hit_rate_chmed"])) > 1e-12:
                bad.append((r["class"], r["arm"], "collision"))
        m = s[(s.cls == r["class"]) & (s.arm == r["arm"]) & (s.horizon == "full")]
        if len(m) == 1 and np.isfinite(r["offroad_vl_rate_full"]):
            if abs(float(m.iloc[0]["offroad_vl_rate"]) - float(r["offroad_vl_rate_full"])) > 1e-12:
                bad.append((r["class"], r["arm"], "offroad"))
    if bad:
        raise SystemExit(f"[check] T5.csv vs table5_validity_summary.csv disagree: {bad[:5]}")
    print(f"[check] T5.csv == table5_validity_summary.csv (chmed solid / full off-road) for every matched row")

    # WP4: the stop-at-end substrate must reproduce Table 5 exactly on every arm the two share
    # (window + polytrunc + analytic), otherwise the Ours rows would be swapped onto a substrate
    # that is not comparable with the SVD polytrunc rows kept from T5.
    sp = RES / "table5_validity_summary_stop.csv"
    if sp.exists():
        st = pd.read_csv(sp)
        d["T5_STOP_DF"] = st
        shared, worst = 0, 0.0
        for _, r in t5.iterrows():
            for col5, hor, cols in (("solid_hit_rate_chmed", "chmed", "any_solid_rate"),
                                    ("offroad_vl_rate_full", "full", "offroad_vl_rate")):
                m = st[(st.cls == r["class"]) & (st.arm == r["arm"]) & (st.horizon == hor)]
                if len(m) != 1 or not np.isfinite(r[col5]) or r["arm"] == "real":
                    continue
                shared += 1
                worst = max(worst, abs(float(m.iloc[0][cols]) - float(r[col5])))
        if worst > 1e-12:
            raise SystemExit(f"[check] table5_validity_summary_stop.csv disagrees with T5.csv on a shared arm "
                             f"(max |Δ| = {worst:g})")
        print(f"[check] table5_validity_summary_stop.csv == T5.csv on all {shared} shared non-real arm cells "
              f"(max |Δ| = {worst:g}); the real row legitimately differs (WP2 cohort 489 vs Table 5's 511)")
    return d


def routed_pct(man: dict, group: str) -> float | None:
    sc = man.get("scenes", {}).get(group)
    return None if sc is None else 100.0 * float(sc["routed_fraction"])


# ─────────────────────────────────────────────────────────────────────────────
# column resolvers
# ─────────────────────────────────────────────────────────────────────────────
def assert_dm_constant(D: dict) -> None:
    for name, df, keys in [("table2_sakura_arms.csv", D["T2_SAKURA"], ["scope", "arm"]),
                           ("T2.csv", D["T2"][D["T2"].row_type.eq("arm_measure")], ["row_role", "scope", "arm"])]:
        g = df.groupby(keys)["D_m"].nunique(dropna=True)
        bad = g[g > 1]
        if len(bad):
            raise SystemExit(f"[check] {name}: D_m is not constant across measures for {list(bad.index)[:3]}")
    print("[check] D_m is constant across the six measures for every (scope, arm) in both similarity sources")


def _sim_source(D: dict, group: str, arm: str):
    """(rows keyed by measure, source_file, raw-median column, selector-maker, cohort note)."""
    if group == "tlkeep":
        df, base = D["T2"], dict(row_role="primary", row_type="arm_measure", scope=group, arm=arm)
        mk = lambda meas: dict(base, measure=meas)
        rows = {m: pick(df, f"T2 sim {m}", **mk(m)) for m, _, _ in MEASURES}
        return (rows, rel(SRC["T2"]), "median_err", mk,
                "ours-vs-SVD cohort (3 arms: ours3_disk, svd_d5_fullfit, svd_d5_logo); "
                "SAKURA has no tlkeep arm so the 5-arm denominator does not exist here")
    df = D["T2_SAKURA"]
    mk = lambda meas: dict(scope=group, arm=arm, measure=meas)
    rows = {m: pick(df, f"T2sakura sim {m}", **mk(m)) for m, _, _ in MEASURES}
    return (rows, rel(SRC["T2_SAKURA"]), "median_err_common_Sk", mk,
            "5-arm common S_k (sakura_plain, sakura_route, svd_d5_fullfit, svd_d5_logo, ours3_disk)")


def similarity_cells(D: dict, group: str, method: str) -> dict:
    """The two printed similarity columns plus the two detail values they replace.

    sim_int  互動相似 = mean over the FIVE interaction measures (pet, dmin, alpha, cpoint, uc) of
                        `normalized` = median_err(S_k) / b_k, i.e. exactly the old D_m with DTW
                        dropped from the average; same S_k, same worst-arm denominators b_k.
    sim_dtw  軌跡相似 = the RAW median DTW to the recorded target path, in metres (identical to the
                        DTW column of tables3_sixmeasures for the same group/arm).
    _dm      the old six-measure D_m (detail only, block D10 + the CSV column `sim_dm`).
    _dtwnorm the normalised DTW median/b_k (detail only, CSV column `sim_dtw_norm`).
    """
    if method not in SIM_ARM:
        c = na_cell("the similarity columns are defined for the reconstruction arms only "
                    "(SAKURA, SAKURA-route, SVD_d5, Ours)")
        return dict(sim_int=c, sim_dtw=na_cell(c.note), _dm=na_cell(c.note), _dtwnorm=na_cell(c.note))
    arm = SIM_ARM[method]
    if group == "tlkeep" and method in ("sakura", "sakura_route"):
        c = na_cell("SAKURA has no tlkeep arm")
        return dict(sim_int=c, sim_dtw=na_cell(c.note), _dm=na_cell(c.note), _dtwnorm=na_cell(c.note))
    rows, srcfile, rawcol, mk, cohort = _sim_source(D, group, arm)
    vals = [float(rows[m]["normalized"]) for m in INT_MEASURES]
    if not all(np.isfinite(v) for v in vals):
        raise SystemExit(f"[sim] non-finite normalised median for {group}/{arm}: {vals}")
    d_int = float(np.mean(vals))
    per = "; ".join(f"{m} {rows[m]['normalized']:.4f} (n_Sk {int(rows[m]['n_common_Sk'])})" for m in INT_MEASURES)
    int_sel = sel(**mk("pet")).replace("measure=pet", "measure∈{" + ",".join(INT_MEASURES) + "}")
    c_int = Cell(d_int, source_file=srcfile, selector=int_sel, source_column="mean(normalized) over pet,dmin,alpha,cpoint,uc",
                 n=int(rows["pet"]["n_common_Sk"]),
                 note=f"{cohort}; normalised medians averaged: {per}; the printed n is the PET S_k, the "
                      f"smallest of the five")
    rdtw = rows["dtw"]
    c_dtw = Cell(float(rdtw[rawcol]), source_file=srcfile, selector=sel(**mk("dtw")), source_column=rawcol,
                 n=int(rdtw["n_common_Sk"]),
                 note=f"raw median DTW to the recorded target path in metres, NOT normalised, so it is "
                      f"comparable across tables; {cohort}; b_k (worst-arm median) = {float(rdtw['b_k']):.4f} m, "
                      f"normalised value {float(rdtw['normalized']):.4f} (detail only)")
    c_dm = Cell(float(rows["pet"]["D_m"]), source_file=srcfile, selector=sel(**mk("pet")), source_column="D_m",
                n=int(rows["pet"]["n_common_Sk"]),
                note=f"superseded six-measure D_m (kept as detail); {cohort}")
    c_dn = Cell(float(rdtw["normalized"]), source_file=srcfile, selector=sel(**mk("dtw")),
                source_column="normalized", n=int(rdtw["n_common_Sk"]),
                note=f"normalised DTW = median/b_k, b_k = {float(rdtw['b_k']):.4f} m (kept as detail); {cohort}")
    return dict(sim_int=c_int, sim_dtw=c_dtw, _dm=c_dm, _dtwnorm=c_dn)


def gen_lookup(D: dict, name: str, **f) -> pd.Series | None:
    g = D["GEN"]
    m = pd.Series(True, index=g.index)
    for k, v in f.items():
        col = g[k]
        if v is None:
            m &= col.isna()
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            m &= col.astype(float).eq(float(v))
        else:
            m &= col.astype(str).eq(str(v))
    sub = g[m]
    if len(sub) == 0:
        return None
    if len(sub) > 1:
        raise SystemExit(f"[gen_lookup] {name}: {sel(**f)} matched {len(sub)} rows")
    return sub.iloc[0]


def cell_gen(D: dict, group: str, method: str, what: str) -> Cell:
    """variance / coverage / corner coverage for the generative (KDE) arms."""
    if group == "special_39_180":
        return cell_gen_rare(D, method, what)
    if method not in GEN_ARM:
        return na_cell(f"{what} is defined for the generative (KDE) arms only")
    arm = GEN_ARM[method]
    if method == "sakura_route_kde" and group == "tlkeep":
        return na_cell("SAKURA has no tlkeep arm")
    if what == "var":
        f = dict(table_group="variance", metric="variance_ratio_trace", **{"class": group},
                 real_set="matched489", arm=arm, pool="all", space="interaction")
    elif what == "cov":
        f = dict(table_group="coverage", metric="coverage", **{"class": group}, real_set="matched489",
                 arm=arm, pool="all", space="joint", budget=1000.0)
    else:
        f = dict(table_group="corner_coverage", metric="corner_coverage", **{"class": group},
                 real_set="matched489", arm=arm, pool="all", space="joint", budget=1000.0)
    r = gen_lookup(D, f"{what}/{group}/{arm}", **f)
    if r is None:
        return na_cell(f"no {what} row for arm {arm} in class {group}")
    note = f"mean over {int(r['n_seeds'])} seeds [{r['value_seed_min']:.3f}, {r['value_seed_max']:.3f}]"
    if what == "corner":
        note += f"; rare k={int(r['n_rare'])} of n_real={int(r['n_real'])}"
    return Cell(r["value"], source_file=rel(SRC["GEN"]), selector=sel(**f), source_column="value",
                upstream=str(r["source"]), note=note, n=int(r["n_samples"]))


def cell_gen_rare(D: dict, method: str, what: str) -> Cell:
    if method == "svd_d5_kde":
        return Cell(text=UNDEF, kind="undefined", source_file=rel(SRC["T8"]),
                    selector=sel(arm="svd_d5_singleton_N1"), source_column="definable",
                    note="N=1: centred rank 0, singular values [0.0], LOO bandwidth undefined, zero samples")
    if method not in RARE_GEN_ARM:
        return na_cell(f"{what} is defined for the generative (KDE) arms only")
    arm = RARE_GEN_ARM[method]
    if what == "var":
        f = dict(table_group="rarecase_39_180", metric="variance_ratio_trace", arm=arm, pool="all",
                 space="interaction")
    else:
        f = dict(table_group="rarecase_39_180", metric="recovered_fraction", arm=arm, pool="all", space="joint")
    r = gen_lookup(D, f"rare/{what}/{arm}", **f)
    if r is None:
        return na_cell(f"no rarecase {what} row for arm {arm}")
    note = f"mean over {int(r['n_seeds'])} seeds [{r['value_seed_min']:.3f}, {r['value_seed_max']:.3f}]"
    return Cell(r["value"], source_file=rel(SRC["GEN"]), selector=sel(**f), source_column="value",
                upstream=str(r["source"]), note=note, n=int(r["n_samples"]))


def _t5_frame(D: dict, srcdef: dict) -> pd.DataFrame:
    p = PROJECT / srcdef["path"]
    key = {str(SRC["T5"]): "T5", str(SRC["T5_SAKURA"]): "T5_SAKURA",
           str(RES / "table5_validity_summary_stop.csv"): "T5_STOP_DF"}.get(str(p))
    if key and key in D:
        return D[key]
    if not p.exists():
        raise SystemExit(f"[arm map] source file missing: {srcdef['path']}")
    return pd.read_csv(p)


WHAT2KEY = {"coll": "collision", "offroad": "offroad", "valid": "valid"}


def cell_validity(D: dict, amap: dict, group: str, method: str, what: str) -> Cell:
    """background collision / off-road / (RareCase) valid rate, via validity_arm_map.json."""
    wk = WHAT2KEY[what]
    rows = amap["rarecase_rows"] if group == "special_39_180" else amap["class_rows"]
    if method not in rows:
        return na_cell(f"no arm-map entry for method {method}")
    ent = rows[method]
    if group == "tlkeep" and method in ("sakura", "sakura_route", "sakura_route_kde"):
        return na_cell("SAKURA has no tlkeep arm")
    prov = " [PROVISIONAL window execution]" if ent.get("provisional") else ""
    if ent.get("execution_kind"):
        prov += f" [execution-end: {ent['execution_kind']}]"
    resolver = ent.get(f"{wk}_resolver", ent["resolver"])

    if resolver == "undefined":
        return Cell(text=UNDEF, kind="undefined", source_file=rel(SRC["T8"]),
                    selector=sel(arm=ent["arm"]), source_column="definable",
                    note=ent.get("note", "undefined at N = 1"))

    if resolver == "table5":
        srcdef = amap["sources"][ent.get(f"{wk}_source", ent["source"])]
        df = _t5_frame(D, srcdef)
        col = srcdef[f"{wk}_column"]
        f = {srcdef["class_column"]: group, srcdef["arm_column"]: ent.get(f"{wk}_arm", ent["arm"])}
        f.update(srcdef.get(f"{wk}_filters", {}))
        r = pick(df, f"{wk}/{group}/{ent['arm']}", **f)
        return Cell(r[col], source_file=srcdef["path"], selector=sel(**f), source_column=col,
                    n=int(r[srcdef["n_column"]]),
                    note=srcdef[f"{wk}_definition"] + prov + ("; " + ent["note"] if ent.get("note") else ""))

    if resolver == "t8":
        df, arm = D["T8"], ent.get(f"{wk}_arm", ent["arm"])
        r = pick(df, f"T8/{wk}/{arm}", arm=arm)
        nexec = float(r["n_executed"])
        if wk == "collision":
            v, colname = float(r["bg_solid_rate_all_gtsupport"]), "bg_solid_rate_all_gtsupport"
            note = ("T8: OBB solid hit (>2 frames, depth >= 0.1 m) vs all rec00 vehicle tracks within the GT "
                    "support 2424-2923 — a different horizon from the Table-5 chmed rate")
        elif wk == "offroad":
            v, colname = (float(r["offroad_n"]) / nexec if nexec else np.nan), "offroad_n / n_executed"
            note = ("T8: >5 % of points outside the tyms.xodr driving+shoulder union (33_e7 recipe) — a different "
                    "rule from the Table-5 off-road VL rate")
        else:
            v, colname = (float(r["n_valid"]) / nexec if nexec else np.nan), "n_valid / n_executed"
            note = f"gate: {ent.get('valid_gate', 'T8 valid')}"
        return Cell(v, source_file=rel(SRC["T8"]), selector=sel(arm=arm), source_column=colname,
                    n=int(nexec), note=note + prov + ("; " + ent["note"] if ent.get("note") else ""))

    if resolver == "gen":
        metric = ent.get(f"{wk}_metric", {"collision": "bg_solid_rate", "offroad": "offroad_rate",
                                          "valid": "valid_rate"}[wk])
        arm = ent.get(f"{wk}_arm", ent["arm"])
        f = dict(table_group="rarecase_39_180", metric=metric, arm=arm, pool="all")
        r = gen_lookup(D, f"rare/{wk}/{arm}", **f)
        if r is None:
            return na_cell(f"no rarecase {metric} row for {arm}")
        note = str(r["note"])
        if wk == "valid":
            note = f"gate: {ent.get('valid_gate', metric)}; " + note
        return Cell(r["value"], source_file=rel(SRC["GEN"]), selector=sel(**f), source_column="value",
                    upstream=str(r["source"]), n=int(r["n_samples"]),
                    note=note + prov + ("; " + ent["note"] if ent.get("note") else ""))

    raise SystemExit(f"[arm map] unknown resolver {resolver!r}")


# ─────────────────────────────────────────────────────────────────────────────
# table build
# ─────────────────────────────────────────────────────────────────────────────
def method_label(D: dict, group: str, method: str) -> str:
    lab = next(m["label"] for m in METHODS if m["key"] == method)
    if method == "sakura_route":
        p = routed_pct(D["SAKURA_MANIFEST"], group)
        if p is not None:
            lab = f"SAKURA-route ({p:.0f}% routed)"
    return lab


def method_label_tex(D: dict, group: str, method: str) -> str:
    m = next(m for m in METHODS if m["key"] == method)
    lab = m.get("tex_label", m["label"])
    if method == "sakura_route":
        p = routed_pct(D["SAKURA_MANIFEST"], group)
        if p is not None:
            lab = rf"SAKURA-route$^{{{p:.0f}\%}}$"
    return lab


def build_group(D: dict, amap: dict, group: str) -> list[dict]:
    g = GROUP_BY_KEY[group]
    cols = COLS_RARE if group == "special_39_180" else COLS_CLASS
    methods = (APPENDIX_METHODS if group == "tlkeep" else
               [m["key"] for m in METHODS
                if group == "special_39_180" or m["key"] not in RARE_ONLY_METHODS])
    rows = []
    for mk in methods:
        cells = {}
        sim = (dict(sim_int=na_cell("reference row: only the validity columns are defined for the recorded replay"),
                    sim_dtw=na_cell("reference row: only the validity columns are defined for the recorded replay"),
                    _dm=na_cell("reference row"), _dtwnorm=na_cell("reference row"))
               if mk == "real" else similarity_cells(D, group, mk))
        for c in cols:
            k = c["key"]
            if k in ("sim_int", "sim_dtw"):
                cells[k] = sim[k]
            elif mk == "real":
                cells[k] = (cell_validity(D, amap, group, mk, k) if k in ("coll", "offroad", "valid")
                            else na_cell("reference row: only the validity columns are defined for the recorded replay"))
            elif k in ("var", "cov", "corner"):
                cells[k] = cell_gen(D, group, mk, k)
            else:
                cells[k] = cell_validity(D, amap, group, mk, k)
        rows.append(dict(group=group, scenario=g["label"], scenario_tex=g["tex"], n_class=g["n_class"],
                         method=mk, method_label=method_label(D, group, mk),
                         method_label_tex=method_label_tex(D, group, mk),
                         kind=next(m["kind"] for m in METHODS if m["key"] == mk), cells=cells,
                         extra=dict(sim_dm=sim["_dm"], sim_dtw_norm=sim["_dtwnorm"])))
    # bold
    for c in cols:
        k, rule = c["key"], c["rule"]
        cand = [(r, r["cells"][k].value) for r in rows
                if r["kind"] != "real" and r["cells"][k].value is not None]
        if not cand:
            continue
        if rule == "lower":
            best = min(v for _, v in cand)
            key = lambda v: abs(v - best)
        elif rule == "higher":
            best = max(v for _, v in cand)
            key = lambda v: abs(v - best)
        else:
            best = min(abs(v - 1.0) for _, v in cand)
            key = lambda v: abs(abs(v - 1.0) - best)
        for r, v in cand:
            if key(v) <= TIE:
                r["cells"][k].bold = True
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# renderers
# ─────────────────────────────────────────────────────────────────────────────
def fullprec(x) -> str:
    """lineage `value` column: shortest round-trip repr (full float precision).

    Verifier round 1: the previous `%.12g` truncated e.g. 1.1504999999999996 to 1.1505,
    which re-formats to 1.151 instead of the displayed 1.150.
    """
    return "" if x is None else repr(float(x))


def col_name(c: dict) -> str:
    """the table's own column name, without the direction arrow."""
    return c["md"].replace(" ↓", "").replace(" ↑", "").replace(" →1", "").strip()


def bold_rule_note(cols: list[dict]) -> str:
    """Bold-rule footnote generated from THIS table's own column list (verifier round 1, R2):
    a fixed string named columns (涵蓋 / 角落涵蓋) that the RareCase table does not have and
    left its two own columns without a direction."""
    lower = [col_name(c) for c in cols if c["rule"] == "lower"]
    higher = [col_name(c) for c in cols if c["rule"] == "higher"]
    near1 = [col_name(c) for c in cols if c["rule"] == "near1"]
    parts = []
    if lower:
        parts.append(" / ".join(lower) + " → lowest")
    if higher:
        parts.append(" / ".join(higher) + " → highest")
    if near1:
        parts.append(" / ".join(near1) + " → closest to 1.0")
    return ("Bold rule for this table's columns: " + "; ".join(parts) + ". The best value is chosen among the "
            "generated sub-rows of the same scenario group only; exact ties bold both; the real reference is "
            "never bolded.")


def md_table(rows: list[dict], cols: list[dict], caption: str, footnotes: list[str]) -> str:
    head = ["Scenario 場景", "n", "Method 方法"] + [c["md"] for c in cols]
    out = [caption, "",
           "| " + " | ".join(head) + " |",
           "| " + " | ".join(["---", "---:", "---"] + ["---:"] * len(cols)) + " |"]
    prev = None
    for r in rows:
        scen = r["scenario"] if r["group"] != prev else ""
        nn = str(r["n_class"]) if r["group"] != prev else ""
        prev = r["group"]
        vals = []
        for c in cols:
            cell = r["cells"][c["key"]]
            vals.append(f"**{cell.display}**" if cell.bold else cell.display)
        out.append("| " + " | ".join([scen, nn, r["method_label"]] + vals) + " |")
    if footnotes:
        out += [""] + [f"- {f}" for f in footnotes]
    return "\n".join(out) + "\n"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def tex_table(rows: list[dict], cols: list[dict], caption: str, label: str, footnotes: list[str]) -> str:
    ncol = 3 + len(cols)
    spec = "lll" + "r" * len(cols)
    L = ["% auto-generated by scripts/90_tables3_assemble.py — do not edit by hand",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         " & ".join(["Scenario", "$n$", "Method"] + [c["tex"] for c in cols]) + r" \\", r"\midrule"]
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r["group"]:
            groups.append((r["group"], []))
        groups[-1][1].append(r)
    for gi, (gk, grows) in enumerate(groups):
        if gi:
            L.append(r"\midrule")
        for ri, r in enumerate(grows):
            if ri == 0:
                a = rf"\multirow{{{len(grows)}}}{{*}}{{{r['scenario_tex']}}}"
                b = rf"\multirow{{{len(grows)}}}{{*}}{{{r['n_class']}}}"
            else:
                a = b = ""
            vals = []
            for c in cols:
                cell = r["cells"][c["key"]]
                t = cell.display.replace(NA, "--")
                vals.append(rf"\textbf{{{t}}}" if cell.bold else t)
            L.append(" & ".join([a, b, r["method_label_tex"]] + vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    for f in footnotes:
        L.append("% note: " + tex_escape(f).replace("\n", " "))
    L.append(r"\end{table*}")
    L.append("")
    L.append("% ---- Chinese-header variant (needs xeCJK / ctex; compile with xelatex) ----")
    L.append("% " + " & ".join(["場景 Scenario", "$n$", "方法 Method"] + [c["tex_cjk"] for c in cols]) + r" \\")
    return "\n".join(L) + "\n"


def _cjk_w(t: str) -> float:
    """rough display width: CJK glyphs count double."""
    return sum(2.0 if ord(c) > 0x2E80 else 1.0 for c in t)


def png_table(rows: list[dict], cols: list[dict], title: str, path: Path, footnotes: list[str]) -> dict:
    head = ["Scenario", "n", "Method"] + [c["png"] for c in cols]
    body, bolds = [], []
    prev = None
    for r in rows:
        scen = r["scenario"] if r["group"] != prev else ""
        nn = str(r["n_class"]) if r["group"] != prev else ""
        prev = r["group"]
        body.append([scen, nn, r["method_label"]] + [r["cells"][c["key"]].display for c in cols])
        bolds.append([False, False, False] + [r["cells"][c["key"]].bold for c in cols])
    nrow = len(body)

    # column widths from the widest cell in each column (CJK counts double)
    raw = [max([_cjk_w(head[i])] + [_cjk_w(b[i]) for b in body]) + 2.2 for i in range(len(head))]
    raw[1] = max(raw[1], 4.0)
    widths = [w / sum(raw) for w in raw]

    row_h, title_h = 0.30, 0.46
    note_h = 0.17
    table_h = row_h * (nrow + 1)
    fig_w = max(12.0, 0.115 * sum(raw))
    wrapped = []
    for f in footnotes:
        wrapped += textwrap.wrap(f, width=int(fig_w * 17)) or [""]
    notes_h = note_h * len(wrapped) + 0.26
    fig_h = title_h + table_h + notes_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = fig.add_axes([0.008, notes_h / fig_h, 0.984, table_h / fig_h])
        ax.axis("off")
        tb = ax.table(cellText=body, colLabels=head, colWidths=widths, cellLoc="right", bbox=[0, 0, 1, 1])
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        for (r_i, c_i), cell in tb.get_celld().items():
            cell.set_linewidth(0.4)
            cell.get_text().set_fontfamily(CJK_FONTS)
            cell.PAD = 0.035
            if r_i == 0:
                cell.set_facecolor("#e8e8e8")
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("center")
                cell.get_text().set_fontsize(8.5)
            else:
                if c_i in (0, 2):
                    cell.get_text().set_ha("left")
                if bolds[r_i - 1][c_i]:
                    cell.get_text().set_fontweight("bold")
                if rows[r_i - 1]["kind"] == "real":
                    cell.set_facecolor("#f4f4f4")
                if r_i > 1 and body[r_i - 1][0]:
                    cell.set_edgecolor("#000000")
                    cell.visible_edges = "TBLR"
        fig.text(0.008, 1 - 0.55 * title_h / fig_h, title, fontsize=12, fontfamily=CJK_FONTS,
                 va="center", ha="left", fontweight="bold")
        y = (notes_h - 0.16) / fig_h
        for f in wrapped:
            y -= note_h / fig_h
            fig.text(0.008, y, f, fontsize=7.0, fontfamily=CJK_FONTS, va="bottom", ha="left")
        fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    glyph = [str(w.message) for w in caught if "Glyph" in str(w.message) or "missing from font" in str(w.message)]
    if glyph:
        raise SystemExit(f"[png] missing glyphs while rendering {rel(path)}: {glyph[:3]}")
    return dict(path=rel(path), bytes=path.stat().st_size, glyph_warnings=0)


def csv_table(rows: list[dict], cols: list[dict], table: str, path: Path) -> pd.DataFrame:
    recs = []
    for r in rows:
        rec = dict(table=table, group=r["group"], scenario_label=r["scenario"], n_class=r["n_class"],
                   method_key=r["method"], method_label=r["method_label"], row_kind=r["kind"])
        for c in cols:
            cell = r["cells"][c["key"]]
            rec[c["key"]] = cell.value if cell.value is not None else np.nan
            rec[c["key"] + "_display"] = cell.display
            rec[c["key"] + "_bold"] = bool(cell.bold)
            rec[c["key"] + "_n"] = cell.n if cell.n is not None else np.nan
        # detail-only columns (printed nowhere): the superseded six-measure D_m and the normalised
        # DTW that the two printed similarity columns replace.
        for xk, cell in r.get("extra", {}).items():
            rec[xk] = cell.value if cell.value is not None else np.nan
            rec[xk + "_n"] = cell.n if cell.n is not None else np.nan
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(path, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# six-measure detail
# ─────────────────────────────────────────────────────────────────────────────
def six_measures(D: dict):
    blocks, lineage = [], []
    for g in GROUPS:
        gk = g["key"]
        if gk == "tlkeep":
            df = D["T2"]
            arms = [(a, l) for a, l in SIX_METHODS if a not in ("sakura_plain", "sakura_route")]
            getrow = lambda arm, meas: pick(df, "T2 six", row_role="primary", row_type="arm_measure",
                                            scope=gk, arm=arm, measure=meas)
            src, valcol, ncol = rel(SRC["T2"]), "median_err", "n_common_Sk"
            selbase = dict(row_role="primary", row_type="arm_measure", scope=gk)
        else:
            df = D["T2_SAKURA"]
            arms = list(SIX_METHODS)
            getrow = lambda arm, meas: pick(df, "T2sakura six", scope=gk, arm=arm, measure=meas)
            src, valcol, ncol = rel(SRC["T2_SAKURA"]), "median_err_common_Sk", "n_common_Sk"
            selbase = dict(scope=gk)
        rows, nper, blk_lin = [], {}, []
        for arm, lab in arms:
            vals = {}
            for meas, _, _ in MEASURES:
                r = getrow(arm, meas)
                vals[meas] = float(r[valcol])
                # nper[meas] is printed once per block (in the caption and the n(S_k) table), so it
                # MUST be identical for every arm of the block — it is, because n_common_Sk is the
                # size of the arms' common scene set, but assert it so one arm's count can never be
                # silently printed as the block's (WP1 re-verifier note).
                nn = int(r[ncol])
                if meas in nper and nper[meas] != nn:
                    raise SystemExit(f"[six] n_common_Sk is not constant across arms for {gk}/{meas}: "
                                     f"{nper[meas]} (earlier arm) vs {nn} ({arm}) — the block caption "
                                     f"would print one arm's count as the block's")
                nper[meas] = nn
                blk_lin.append(dict(table="sixmeasures", group=gk, method=lab, column=meas,
                                    value=vals[meas], source_file=src,
                                    source_row_selector=sel(**selbase, arm=arm, measure=meas),
                                    source_column=valcol, upstream_source="", note=""))
            r = getrow(arm, "pet")
            vals["D_m"] = float(r["D_m"])
            blk_lin.append(dict(table="sixmeasures", group=gk, method=lab, column="D_m", value=vals["D_m"],
                                source_file=src, source_row_selector=sel(**selbase, arm=arm, measure="pet"),
                                source_column="D_m", upstream_source="", note=""))
            rows.append(dict(arm=arm, label=lab, vals=vals))
        bold = {}
        for key in [m[0] for m in MEASURES] + ["D_m"]:
            best = min(r["vals"][key] for r in rows)
            bold[key] = {r["label"] for r in rows if abs(r["vals"][key] - best) <= TIE}
        # the lineage rows are built before the bold flags exist, so back-fill them here —
        # tables3_lineage.csv used to carry bold=False for every six-measure row while the
        # rendered .md/.tex bolded 43 of them (verifier round 1, non-blocking note).
        for l in blk_lin:
            l["bold"] = l["method"] in bold[l["column"]]
        lineage += blk_lin
        blocks.append(dict(group=gk, label=g["label"], tex=g["tex"], n_class=g["n_class"], rows=rows,
                           bold=bold, nper=nper, source=src))
    return blocks, lineage


def write_six(blocks, outdir: Path, stamp: str):
    recs = []
    for b in blocks:
        for r in b["rows"]:
            rec = dict(group=b["group"], scenario_label=b["label"], n_class=b["n_class"],
                       method=r["label"], arm=r["arm"], source_file=b["source"])
            for meas, lab, _ in MEASURES:
                rec[meas] = r["vals"][meas]
                rec[meas + "_n_common_Sk"] = b["nper"][meas]
                rec[meas + "_bold"] = r["label"] in b["bold"][meas]
            rec["D_m"] = r["vals"]["D_m"]
            rec["D_m_bold"] = r["label"] in b["bold"]["D_m"]
            recs.append(rec)
    pd.DataFrame(recs).to_csv(outdir / "tables3_sixmeasures.csv", index=False)

    head = ["Scenario", "n", "Method"] + [lab for _, lab, _ in MEASURES] + ["D_m"]
    md = [f"# Six-measure interaction fidelity (paper_tables style) — {stamp}", "",
          "Median paired |Δ| against the recorded scenario on the common scene set S_k of the arms compared "
          "in that block (lower is better, best per column in bold, 3 decimals). D_m = mean over the six "
          "measures of (arm median / worst-arm median) on S_k.", "",
          "| " + " | ".join(head) + " |",
          "| " + " | ".join(["---", "---:", "---"] + ["---:"] * (len(MEASURES) + 1)) + " |"]
    for b in blocks:
        for i, r in enumerate(b["rows"]):
            cells = []
            for meas, _, _ in MEASURES:
                t = f"{r['vals'][meas]:.3f}"
                cells.append(f"**{t}**" if r["label"] in b["bold"][meas] else t)
            t = f"{r['vals']['D_m']:.3f}"
            cells.append(f"**{t}**" if r["label"] in b["bold"]["D_m"] else t)
            md.append("| " + " | ".join([b["label"] if i == 0 else "", str(b["n_class"]) if i == 0 else "",
                                         r["label"]] + cells) + " |")
    md += ["", "Per-block common-scene counts n(S_k) per measure:", ""]
    md.append("| Scenario | " + " | ".join(lab for _, lab, _ in MEASURES) + " | source |")
    md.append("| --- | " + " | ".join(["---:"] * len(MEASURES)) + " | --- |")
    for b in blocks:
        md.append(f"| {b['label']} | " + " | ".join(str(b["nper"][m]) for m, _, _ in MEASURES)
                  + f" | {b['source']} |")
    md += ["", "- tlkeep has no SAKURA arm; its block therefore uses the ours-vs-SVD cohort of results/final/T2.csv "
               "(column `median_err`, row_role=primary) and its D_m is normalised inside that 3-arm set.",
           "- Every other block uses results/table2_sakura_arms.csv (column `median_err_common_Sk`), the 5-arm "
           "common S_k, so SAKURA / SAKURA-route / SVD_d5 / Ours share one denominator.",
           "- PET has a smaller S_k than the other measures because a generated or recorded run can lose the PET "
           "event; the per-measure counts above give the exact denominators."]
    (outdir / "tables3_sixmeasures.md").write_text("\n".join(md) + "\n")

    spec = "lll" + "r" * (len(MEASURES) + 1)
    # the printed n is the CLASS size, not the number of scenes behind the medians, so the
    # caption has to carry n(S_k) per measure (verifier round 1, R4).
    nsk = "; ".join(b["tex"] + " " + "/".join(str(b["nper"][m]) for m, _, _ in MEASURES) for b in blocks)
    L = ["% auto-generated by scripts/90_tables3_assemble.py — do not edit by hand",
         r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Six-measure interaction fidelity (" + tex_escape(stamp) + r"): median paired $|\Delta|$ "
         r"against the recorded scenario on the common scene set $S_k$, lower is better, best per column in "
         r"bold, 3 decimals. $D_m$ = mean over the six measures of (arm median / worst-arm median) on $S_k$. "
         r"tlkeep has no SAKURA arm and uses the ours-vs-SVD cohort of T2. The printed $n$ is the CLASS size "
         r"(the number of recorded scenarios of that class), NOT the number of scenes behind the medians: each "
         r"median is taken over the common scene set $S_k$ of the arms compared in that block, and a scene "
         r"enters $S_k$ only if the measure is finite for every compared arm. $n(S_k)$ per block, in column "
         r"order (PET / min.\ distance / conflict point / angle / arrival speed / DTW): " + nsk + r".}",
         r"\label{tab:tables3-sixmeasures}",
         rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         " & ".join(["Scenario", "$n$", "Method"] + [t for _, _, t in MEASURES] + ["$D_m$"]) + r" \\", r"\midrule"]
    for bi, b in enumerate(blocks):
        if bi:
            L.append(r"\midrule")
        for i, r in enumerate(b["rows"]):
            a = rf"\multirow{{{len(b['rows'])}}}{{*}}{{{b['tex']}}}" if i == 0 else ""
            nn = rf"\multirow{{{len(b['rows'])}}}{{*}}{{{b['n_class']}}}" if i == 0 else ""
            cells = []
            for meas, _, _ in MEASURES:
                t = f"{r['vals'][meas]:.3f}"
                cells.append(rf"\textbf{{{t}}}" if r["label"] in b["bold"][meas] else t)
            t = f"{r['vals']['D_m']:.3f}"
            cells.append(rf"\textbf{{{t}}}" if r["label"] in b["bold"]["D_m"] else t)
            L.append(" & ".join([a, nn, tex_escape(r["label"])] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (outdir / "tables3_sixmeasures.tex").write_text("\n".join(L) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# secondary detail
# ─────────────────────────────────────────────────────────────────────────────
def build_detail(D: dict, amap: dict) -> pd.DataFrame:
    rec = []

    def add(block, group, arm, metric, variant, value, seed_min=np.nan, seed_max=np.nan, n=np.nan,
            source_file="", selector="", source_column="", note=""):
        rec.append(dict(block=block, group=group, arm=arm, metric=metric, variant=variant,
                        value=value, value_seed_min=seed_min, value_seed_max=seed_max, n=n,
                        source_file=source_file, source_row_selector=selector, source_column=source_column,
                        note=note))

    # D1 — T2 ours-vs-SVD D_m and Holm flags
    t2 = D["T2"]
    for _, r in t2[(t2.row_role == "primary") & (t2.row_type == "arm_measure") & (t2.measure == "pet")].iterrows():
        add("D1 T2 ours-vs-SVD D_m (secondary similarity column)", r["scope"], r["arm"], "D_m", "T2 primary cohort",
            float(r["D_m"]), n=float(r["n_common_Sk"]), source_file=rel(SRC["T2"]),
            selector=sel(row_role="primary", row_type="arm_measure", scope=r["scope"], arm=r["arm"], measure="pet"),
            source_column="D_m", note="ours vs SVD only (larger cohort than the 5-arm table2_sakura_arms set)")
    for _, r in t2[(t2.row_role == "primary") & (t2.row_type == "contrast")].iterrows():
        add("D1 T2 ours-vs-SVD D_m (secondary similarity column)", r["scope"],
            f"{r['ours_arm']} - {r['ref_arm']}", f"paired diff {r['measure']}",
            "primary contrast" if bool(r["primary_contrast"]) else "secondary contrast",
            float(r["median_paired_diff"]), n=float(r["n_paired"]), source_file=rel(SRC["T2"]),
            selector=sel(row_role="primary", row_type="contrast", scope=r["scope"], measure=r["measure"],
                         contrast=r["contrast"]),
            source_column="median_paired_diff",
            note=f"CI [{r['ci95_low_global_group']:.3f}, {r['ci95_high_global_group']:.3f}]; "
                 f"p_holm {r['p_holm']:.4f}; holm_significant={bool(r['holm_significant_005'])}; favours {r['favours']}")

    # D2/D3/D4 — every gen_metrics row that is not already a main-table cell
    gm = D["GEN"]
    main_arms = set(GEN_ARM.values())
    main_keys = {("coverage", "coverage", "joint", 1000.0, "matched489", "all"),
                 ("corner_coverage", "corner_coverage", "joint", 1000.0, "matched489", "all"),
                 ("variance", "variance_ratio_trace", "interaction", np.nan, "matched489", "all")}
    blockname = {"coverage": "D2 coverage detail (spaces / budgets / real sets / pools)",
                 "corner_coverage": "D4 corner-coverage detail (spaces / budgets / k rule / real sets / pools)",
                 "variance": "D3 variance detail (per-descriptor, generalized variance, per-centre spread, valid pool)",
                 "rarecase_39_180": "D6 RareCase 39_180 detail arms"}
    for _, r in gm.iterrows():
        key = (r["table_group"], r["metric"], r["space"], r["budget"], r["real_set"], r["pool"])
        is_main = any(k[0] == key[0] and k[1] == key[1] and str(k[2]) == str(key[2])
                      and (str(k[3]) == str(key[3]) or (isinstance(k[3], float) and math.isnan(k[3]) and pd.isna(key[3])))
                      and k[4] == key[4] and k[5] == key[5] for k in main_keys)
        if is_main and r["arm"] in main_arms and r["table_group"] != "rarecase_39_180":
            continue
        add(blockname[r["table_group"]], r["class"], r["arm"],
            f"{r['metric']}" + (f" [{r['space']}]" if isinstance(r["space"], str) else ""),
            f"real_set={r['real_set']}, pool={r['pool']}" + ("" if pd.isna(r["budget"]) else f", @{int(r['budget'])}"),
            float(r["value"]) if pd.notna(r["value"]) else np.nan,
            float(r["value_seed_min"]) if pd.notna(r["value_seed_min"]) else np.nan,
            float(r["value_seed_max"]) if pd.notna(r["value_seed_max"]) else np.nan,
            float(r["n_samples"]) if pd.notna(r["n_samples"]) else np.nan,
            rel(SRC["GEN"]),
            sel(table_group=r["table_group"], **{"class": r["class"]}, arm=r["arm"], metric=r["metric"],
                space=r["space"], pool=r["pool"], real_set=r["real_set"],
                **({} if pd.isna(r["budget"]) else {"budget": r["budget"]})),
            "value", str(r["note"]))

    # D5 — validity detail: every T5 arm per class, plus the sakura arms
    t5 = D["T5"]
    for _, r in t5.iterrows():
        add("D5 validity detail (all Table-5 arms; window vs polytrunc vs analytic)", r["class"], r["arm"],
            "solid_hit_rate_chmed", r["execution_variant"], float(r["solid_hit_rate_chmed"]),
            n=float(r["n_samples"]), source_file=rel(SRC["T5"]),
            selector=sel(**{"class": r["class"]}, arm=r["arm"]), source_column="solid_hit_rate_chmed",
            note=str(r["arm_label"]))
        add("D5 validity detail (all Table-5 arms; window vs polytrunc vs analytic)", r["class"], r["arm"],
            "offroad_vl_rate_full", r["execution_variant"], float(r["offroad_vl_rate_full"]),
            n=float(r["n_samples"]), source_file=rel(SRC["T5"]),
            selector=sel(**{"class": r["class"]}, arm=r["arm"]), source_column="offroad_vl_rate_full",
            note=str(r["arm_label"]))
        for extra in ("valid_all_rate_full", "teleport_rate_full", "ego_critical_rate_full"):
            add("D5 validity detail (all Table-5 arms; window vs polytrunc vs analytic)", r["class"], r["arm"],
                extra, r["execution_variant"], float(r[extra]) if pd.notna(r[extra]) else np.nan,
                n=float(r["n_samples"]), source_file=rel(SRC["T5"]),
                selector=sel(**{"class": r["class"]}, arm=r["arm"]), source_column=extra, note=str(r["arm_label"]))
    sk = D["T5_SAKURA"]
    for _, r in sk.iterrows():
        col = "any_solid_rate" if r["horizon"] == "chmed" else "offroad_vl_rate"
        add("D5 validity detail (SAKURA arms incl. the unclipped-offset pilot)", r["cls"], r["arm"],
            col, f"horizon={r['horizon']}", float(r[col]), n=float(r["n"]), source_file=rel(SRC["T5_SAKURA"]),
            selector=sel(cls=r["cls"], arm=r["arm"], horizon=r["horizon"]), source_column=col,
            note=str(r["arm_label"]))
        if r["horizon"] == "full":
            add("D5 validity detail (SAKURA arms incl. the unclipped-offset pilot)", r["cls"], r["arm"],
                "valid_all_rate", "horizon=full", float(r["valid_all_rate"]), n=float(r["n"]),
                source_file=rel(SRC["T5_SAKURA"]), selector=sel(cls=r["cls"], arm=r["arm"], horizon="full"),
                source_column="valid_all_rate", note=str(r["arm_label"]))

    # D6 — RareCase detail arms straight from T8
    t8 = D["T8"]
    for _, r in t8.iterrows():
        ne = float(r["n_executed"])
        add("D6 RareCase 39_180 detail arms", "special_39_180", r["arm"], "n_valid / n_executed",
            str(r["class_data_used"]), (float(r["n_valid"]) / ne if ne else np.nan), n=ne,
            source_file=rel(SRC["T8"]), selector=sel(arm=r["arm"]), source_column="n_valid / n_executed",
            note=str(r["definable"])[:180])
        add("D6 RareCase 39_180 detail arms", "special_39_180", r["arm"], "bg_solid_rate_all_gtsupport",
            str(r["class_data_used"]), float(r["bg_solid_rate_all_gtsupport"]) if pd.notna(r["bg_solid_rate_all_gtsupport"]) else np.nan,
            n=ne, source_file=rel(SRC["T8"]), selector=sel(arm=r["arm"]),
            source_column="bg_solid_rate_all_gtsupport", note=str(r["used_class_data"]))
        add("D6 RareCase 39_180 detail arms", "special_39_180", r["arm"], "offroad_n / n_executed",
            str(r["class_data_used"]), (float(r["offroad_n"]) / ne if ne else np.nan), n=ne,
            source_file=rel(SRC["T8"]), selector=sel(arm=r["arm"]), source_column="offroad_n / n_executed",
            note=str(r["used_class_data"]))
        if pd.notna(r["D_m"]):
            add("D6 RareCase 39_180 detail arms", "special_39_180", r["arm"], "D_m (T8 basis)",
                str(r["class_data_used"]), float(r["D_m"]), n=ne, source_file=rel(SRC["T8"]),
                selector=sel(arm=r["arm"]), source_column="D_m",
                note="T8 b_k basis (valid draws of the T8 arm set) — NOT the table2_sakura_arms D_m used in the main table")

    # D7 — the execution-end convention triplets behind the validity columns: stop-at-end (WP2)
    #      vs window vs polytrunc, side by side per class and pooled, on the four Table-5 metrics
    #      the thesis quotes, plus the analytic (never executed) counterparts.  Everything except
    #      the analytic rows comes from WP2's results/table5_validity_summary_stop.csv, whose
    #      window / polytrunc rows are Table 5's own samples re-summarised on the same substrate
    #      (load_all() asserts they agree with T5.csv to 0).
    D7 = ("D7 validity cells: execution-end convention triplets (stop-at-end / window / polytrunc) "
          "and the analytic counterparts")
    CLS7 = ["tlkeep", "keeptl", "keeptl_sw", "cutinl", "cutinr", "pooled"]
    CONV7 = [
        ("ours3_disk_stop", "stop-at-end",
         "*** the Ours row of the class tables (WP2, 466 samples, batch f8305650479ea9c2)"),
        ("ours3_disk", "window",
         "window counterpart of ours3_disk_stop — the arm the PROVISIONAL WP1 run printed"),
        ("ours3_disk_kde_stop_s20260910", "stop-at-end (seed 20260910, 5000)",
         "*** the Ours+KDE row of the class tables (WP2, seed 20260910 only, batch b8af10e674c24622)"),
        ("ours3_disk_kde_s20260910_window", "window (seed 20260910, 5000)",
         "sample-paired 1-seed window counterpart: isolates the stop-at-end effect from the seed restriction"),
        ("ours3_disk_kde", "window (3 seeds, 15000)",
         "3-seed window arm the PROVISIONAL WP1 run printed: isolates the seed restriction"),
        ("svd_exec_E3_recon_polytrunc_fullfit", "polytrunc",
         "*** the SVD_d5 row of the class tables"),
        ("svd_exec_E3_recon_fullfit", "window", "window counterpart of the SVD_d5 row"),
        ("svd_exec_stop_recon_fullfit", "stop-at-end",
         "WP2 consistency check: stop-at-end vs polytrunc for the SVD_d5 row"),
        ("svd_exec_E3_kde_polytrunc_kde", "polytrunc", "*** the SVD_d5+KDE row of the class tables"),
        ("svd_exec_E3_kde_kde", "window", "window counterpart of the SVD_d5+KDE row"),
        ("svd_exec_stop_kde", "stop-at-end",
         "WP2 consistency check: stop-at-end vs polytrunc for the SVD_d5+KDE row"),
        ("real", "recorded replay (WP2 cohort, 489 scenes)",
         "NOT the row the tables print: the tables use Table 5's real row (511 replays pooled); this is the "
         "same replay scored on WP2's 489-scene cohort"),
    ]
    M7 = [("solid_hit_rate_chmed", "chmed", "any_solid_rate"),
          ("offroad_vl_rate_full", "full", "offroad_vl_rate"),
          ("teleport_rate_full", "full", "teleport_rate"),
          ("valid_and_critical_rate_chmed", "chmed", "valid_and_critical_rate")]
    stop_path = PROJECT / amap["sources"]["T5_STOP"]["path"]
    if stop_path.exists():
        sp = pd.read_csv(stop_path)
        for cls in CLS7:
            for arm, kind, why in CONV7:
                for metric, hor, col in M7:
                    m = sp[(sp["cls"] == cls) & (sp["arm"] == arm) & (sp["horizon"] == hor)]
                    if len(m) != 1:
                        continue
                    r = m.iloc[0]
                    add(D7, cls, arm, metric, kind,
                        float(r[col]) if pd.notna(r[col]) else np.nan, n=float(r["n"]),
                        source_file=rel(stop_path), selector=sel(cls=cls, arm=arm, horizon=hor),
                        source_column=col, note=why)
    for cls in CLS7:
        for arm, why in [("svd_d5_fullfit", "analytic decode (never executed) of the SVD_d5 row"),
                         ("svd_d5_kde", "analytic decode (never executed) of the SVD_d5+KDE row")]:
            m = t5[(t5["class"] == cls) & (t5["arm"] == arm)]
            if len(m) != 1:
                continue
            r = m.iloc[0]
            for metric in ("solid_hit_rate_chmed", "offroad_vl_rate_full", "teleport_rate_full"):
                add(D7, cls, arm, metric, "analytic (no execution)",
                    float(r[metric]) if pd.notna(r[metric]) else np.nan, n=float(r["n_samples"]),
                    source_file=rel(SRC["T5"]), selector=sel(**{"class": cls}, arm=arm),
                    source_column=metric, note=why)

    # D8 — rare scene lists
    rare = D["RARE"]
    for _, r in rare[rare.is_rare & (rare.real_set == "matched489")].iterrows():
        add("D8 rare (corner) real scenarios, matched489", r["class"], r["scenario_uid"], "nn_dist",
            f"rank {int(r['rank'])} of k={math.ceil(0.2 * float(r['n_real_rankable']))}",
            float(r["nn_dist"]), n=float(r["n_real_rankable"]), source_file=rel(SRC["RARE"]),
            selector=sel(**{"class": r["class"]}, real_set="matched489", scenario_uid=r["scenario_uid"]),
            source_column="nn_dist",
            note=f"nn_dist/eps={r['nn_dist_over_eps']:.2f}; in_disk469={bool(r['in_disk469'])}")

    # D9 — RareCase substrate cross-check: every 39_180 arm scored on the SAME Table-5 substrate,
    #      so the mixed T8 / Table-5 provenance of the RareCase validity columns can be audited.
    for fname, tag in [("table5_validity_samples.csv", "ours / SVD Table-5 arms"),
                       ("table5_validity_samples_sakura.csv", "SAKURA Table-5 arms")]:
        f = RES / fname
        if not f.exists():
            continue
        sm = pd.read_csv(f, low_memory=False)
        sm = sm[sm["scenario_id"].astype(str) == "39_180"]
        for (arm, hor), g in sm.groupby(["arm", "horizon"]):
            if hor == "chmed":
                add("D9 RareCase 39_180 on the common Table-5 substrate (cross-check)", "special_39_180", arm,
                    "any_solid (chmed)", tag, float(g["any_solid"].mean()), n=float(len(g)),
                    source_file=rel(f), selector=sel(scenario_id="39_180", arm=arm, horizon="chmed"),
                    source_column="any_solid",
                    note="Table-5 substrate; compare with the T8 GT-support rate used in the RareCase table")
            if hor == "full":
                for col in ("offroad_vl", "valid_all"):
                    add("D9 RareCase 39_180 on the common Table-5 substrate (cross-check)", "special_39_180", arm,
                        f"{col} (full)", tag, float(g[col].mean()), n=float(len(g)),
                        source_file=rel(f), selector=sel(scenario_id="39_180", arm=arm, horizon="full"),
                        source_column=col, note="Table-5 substrate")

    # D10 — the similarity split: what the two printed columns are made of, and the two values
    # they replaced (the six-measure D_m and the normalised DTW median/b_k).
    BD10 = "D10 similarity split: D_int (5 interaction measures), raw DTW [m], and the superseded D_m / normalised DTW"
    for g in GROUPS:
        gk = g["key"]
        for mk, arm in SIM_ARM.items():
            if gk == "tlkeep" and mk in ("sakura", "sakura_route"):
                continue
            rows, srcfile, rawcol, mksel, cohort = _sim_source(D, gk, arm)
            nsk = lambda m: float(rows[m]["n_common_Sk"])
            add(BD10, gk, arm, "D_int (printed 互動相似)", "mean of normalized over pet,dmin,alpha,cpoint,uc",
                float(np.mean([float(rows[m]["normalized"]) for m in INT_MEASURES])), n=nsk("pet"),
                source_file=srcfile, selector=sel(**mksel("pet")).replace("measure=pet", "measure∈5 interaction"),
                source_column="normalized", note=cohort)
            add(BD10, gk, arm, "DTW median [m] (printed 軌跡相似)", "raw, not normalised",
                float(rows["dtw"][rawcol]), n=nsk("dtw"), source_file=srcfile,
                selector=sel(**mksel("dtw")), source_column=rawcol,
                note="identical to the DTW column of tables3_sixmeasures; " + cohort)
            add(BD10, gk, arm, "D_m (6 measures, SUPERSEDED)", "mean of normalized over all six measures",
                float(rows["pet"]["D_m"]), n=nsk("pet"), source_file=srcfile,
                selector=sel(**mksel("pet")), source_column="D_m",
                note="the column the two printed similarity columns replaced on 2026-09-11; " + cohort)
            add(BD10, gk, arm, "DTW normalised (median/b_k)", "detail only",
                float(rows["dtw"]["normalized"]), n=nsk("dtw"), source_file=srcfile,
                selector=sel(**mksel("dtw")), source_column="normalized", note=cohort)
            add(BD10, gk, arm, "DTW b_k (worst-arm median) [m]", "normalisation denominator",
                float(rows["dtw"]["b_k"]), n=nsk("dtw"), source_file=srcfile,
                selector=sel(**mksel("dtw")), source_column="b_k",
                note="the same b_k for every arm of the group, by construction; " + cohort)
            for m in INT_MEASURES:
                add(BD10, gk, arm, f"normalized {m}", "D_int summand", float(rows[m]["normalized"]),
                    n=nsk(m), source_file=srcfile, selector=sel(**mksel(m)), source_column="normalized",
                    note=f"raw median {float(rows[m][rawcol]):.4f} {rows[m]['unit']} / b_k "
                         f"{float(rows[m]['b_k']):.4f}; " + cohort)

    df = pd.DataFrame(rec)
    return df


def write_detail(df: pd.DataFrame, outdir: Path, stamp: str):
    df.to_csv(outdir / "tables3_detail.csv", index=False)
    md = [f"# tables3 — secondary detail ({stamp})", "",
          "Every secondary number the main tables' captions or the report refer to. "
          "`value_seed_min/max` are over the sampling seeds where several exist.", ""]
    for block in df.block.unique():
        sub = df[df.block == block]
        md += [f"## {block}", "", f"{len(sub)} rows.", "",
               "| group | arm | metric | variant | value | seed min | seed max | n | source | selector | column |",
               "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for _, r in sub.iterrows():
            f3 = lambda v: "" if pd.isna(v) else f"{v:.4g}"
            md.append(f"| {r['group']} | {r['arm']} | {r['metric']} | {r['variant']} | {f3(r['value'])} | "
                      f"{f3(r['value_seed_min'])} | {f3(r['value_seed_max'])} | {f3(r['n'])} | "
                      f"{r['source_file']} | {r['source_row_selector']} | {r['source_column']} |")
        md.append("")
    (outdir / "tables3_detail.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# latex compile check
# ─────────────────────────────────────────────────────────────────────────────
def check_pdflatex(tex_files: list[Path]) -> list[dict]:
    out = []
    exe = shutil.which("pdflatex")
    for p in tex_files:
        if exe is None:
            out.append(dict(file=rel(p), ok=False, note="pdflatex not found"))
            continue
        with tempfile.TemporaryDirectory(prefix="tables3_tex_") as td:
            td = Path(td)
            (td / "t.tex").write_text(p.read_text())
            doc = ("\\documentclass[10pt]{article}\n\\usepackage[a4paper,landscape,margin=8mm]{geometry}\n"
                   "\\usepackage{booktabs}\n\\usepackage{multirow}\n\\usepackage{graphicx}\n"
                   "\\usepackage[T1]{fontenc}\n\\begin{document}\n\\input{t.tex}\n\\end{document}\n")
            (td / "main.tex").write_text(doc)
            r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                               cwd=td, capture_output=True, text=True)
            ok = r.returncode == 0 and (td / "main.pdf").exists()
            tail = "" if ok else "\n".join(r.stdout.strip().splitlines()[-12:])
            out.append(dict(file=rel(p), ok=bool(ok),
                            pdf_bytes=(td / "main.pdf").stat().st_size if (td / "main.pdf").exists() else 0,
                            note=tail))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite-arm-map", action="store_true",
                    help="overwrite validity_arm_map.json with the built-in default before running")
    args = ap.parse_args()
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    D = load_all()
    assert_dm_constant(D)
    amap = load_arm_map(OUT / "validity_arm_map.json", args.rewrite_arm_map)
    provisional = bool(amap.get("provisional"))
    STAMP = amap.get("provisional_stamp", "PROVISIONAL") if provisional else "final"
    prov_txt = (f"**{STAMP}** — " if provisional else "")

    # ── build every group ──
    built = {g["key"]: build_group(D, amap, g["key"]) for g in GROUPS}

    lineage = []
    outputs = []
    tex_files = []
    png_meta = []
    tables_md = {}

    # Footnotes. The bold-rule note is generated per table from its own column list
    # (bold_rule_note); the polytrunc note is a CLASS-table statement only — the RareCase
    # SVD_d5 cells come from T8 (arm svd_matched_fullfit) and SVD_d5+KDE is undefined,
    # so that note must never be emitted for the RareCase table (verifier round 1, R1).
    NOTE_DASH = ("「–」 = the column does not apply to that method row (the two similarity columns are defined "
                 "for the reconstruction arms, variance / coverage / corner coverage for the KDE arms).")
    NOTE_SIM = ("相似度 is split into two columns. 互動相似 D_int = the mean over the FIVE interaction measures "
                "(|ΔPET| s, |Δd_min| m, |Δα| deg, ‖Δc‖ m, |Δu_c| m/s) of (arm median / worst-arm median b_k) on "
                "the arms' common scene set S_k — the same normalisation and the same S_k as the six-measure D_m "
                "it replaces, with DTW simply left out of the average, so it is a unitless 0–1 score and only "
                "comparable inside its own group. 軌跡相似 = the RAW median DTW between the generated and the "
                "recorded target path in metres, not normalised, so it IS comparable across groups and tables. "
                "Both are lower-is-better. The superseded six-measure D_m, the normalised DTW (median/b_k) and "
                "b_k itself are in tables3_detail block D10 and in the `sim_dm` / `sim_dtw_norm` columns of this "
                "table's .csv; D_m also still appears in tables3_sixmeasures.*")
    NOTE_COV = ("Coverage@1000 and corner coverage@1000: joint space (interaction ∧ path), pool = all draws, "
                "real_set = matched489, mean over 3 seeds × 20 orderings. Corner = the same rule restricted to the "
                "top-20 % most nearest-neighbour-isolated real scenarios of the class (k = ceil(0.2 · n_rankable)).")
    NOTE_VAR = ("變異比 = mean over the six interaction descriptors of var(z_gen)/var(z_real,class) on the draws "
                "with all six descriptors finite; 1.0 = the same spread as the recorded class.")
    NOTE_BGDEF = ("背景碰撞 = chmed-horizon solid background hit rate (OBB SAT, >2 overlap frames, depth ≥ 0.1 m); "
                  "出地圖 = off-road VL rate on the full horizon.")
    NOTE_POLYTRUNC = ("In this table the SVD_d5 / SVD_d5+KDE validity cells use the EXECUTED polytrunc variants "
                      "(svd_exec_E3_recon_polytrunc_fullfit / svd_exec_E3_kde_polytrunc_kde); their window and "
                      "analytic counterparts are in tables3_detail. This convention applies to the class tables "
                      "only — the RareCase SVD_d5 cells come from T8 (arm svd_matched_fullfit).")
    NOTE_PROV = ("PROVISIONAL: the Ours / Ours+KDE 背景碰撞 and 出地圖 cells still come from the WINDOW executions "
                 "(ours3_disk / ours3_disk_kde); WP2 replaces them with stop-at-end executions "
                 "(ours3_disk_stop / ours3_disk_kde_stop_s20260910) and WP4 re-runs this script.")
    NOTE_PROV_RARE = ("PROVISIONAL: the Ours 背景碰撞 / 出地圖 / 有效率 cells are the Table-5 WINDOW execution of "
                      "the single 39_180 scene and the Ours+KDE cells are the T8 window executions of "
                      "ours3_condkde_a2641; WP2 produces the stop-at-end executions and WP4 re-runs this script "
                      "(WP2's stop-at-end arms are class-level, so the RareCase rows may have to stay window — the "
                      "final report will say which).")
    NOTE_STOP = ("執行結束慣例 execution-end convention (背景碰撞 / 出地圖 only): the Ours and Ours+KDE cells are "
                 "STOP-AT-END executions — the target brakes to 0 the moment its trajectory ends instead of "
                 "coasting at the last commanded speed (arms ours3_disk_stop, 469 samples, and "
                 "ours3_disk_kde_stop_s20260910, 5 000 samples). The SVD_d5 / SVD_d5+KDE cells are POLYTRUNC "
                 "(the post-trajectory rows are deleted) and the SAKURA rows and the real reference are their "
                 "original window executions / replays. The mixed convention is conservative against ours: pooled "
                 "SVD reconstruction off-road is 0.043 under polytrunc but 0.059 under stop-at-end, whereas the "
                 "ours off-road rate is convention-invariant (0.032 → 0.032, 0 of 469 samples change) and ours "
                 "teleport is 0.000 under both.")
    NOTE_STOP_N = ("Ours+KDE validity n = 5 000, not 15 000: only seed 20260910 was re-executed with stop-at-end, "
                   "so the 背景碰撞 / 出地圖 cells of that row change denominator AND seed coverage. All three "
                   "conventions (3-seed window 15 000, 1-seed window 5 000, 1-seed stop-at-end 5 000) are printed "
                   "side by side in tables3_detail block D7. The 變異比 / 涵蓋 / 角落涵蓋 cells of the same row "
                   "still use all 3 seeds (15 000 draws) and are unaffected.")
    NOTE_STOP_PARKED = ("A stopped target is still a body on the road. At the FULL horizon that manufactures "
                        "collisions (pooled any-solid ours 0.563 → 0.738, SVD reconstruction 0.458 → 0.675), but "
                        "at the chmed horizon these tables use the effect is nearly nil: the samples whose only "
                        "solid hits start after the stop number 0/469 (Ours), 17/5 000 (Ours+KDE), 0/489 and "
                        "9/1 500 (the two SVD consistency arms). The 92/469 · 293/5 000 · 102/489 · 52/1 500 "
                        "figures quoted in RESULT_WP2 are FULL-horizon counts.")
    NOTE_REAL_REF = ("The real (reference) row is Table 5's replay cohort (results/final/T5.csv, arm real; 50 / 81 "
                     "/ 61 / 22 / 297 replays per class, 511 pooled), NOT the 489-scene cohort WP2 re-scored "
                     "(pooled n = 487 at chmed, 0.037 solid). Both are in tables3_detail block D7.")
    NOTE_RARE_WINDOW = ("執行結束慣例 execution-end convention: this table is NOT stop-at-end. WP2 executed exactly "
                        "four class-level stop arms (ours3_disk_stop 469, ours3_disk_kde_stop_s20260910 5 000, "
                        "svd_exec_stop_recon_fullfit 489, svd_exec_stop_kde 1 500) and 39_180 is in none of them, "
                        "so the Ours 背景碰撞 / 出地圖 / 有效率 cells are the Table-5 WINDOW execution of the single "
                        "39_180 scene and the Ours+KDE cells are the T8 WINDOW executions of ours3_condkde_a2641 "
                        "(300 draws). Neither row was re-executed with stop-at-end; the class tables' convention "
                        "does not apply here.")
    NOTE_RARE_VALID = ("This table's 4th column is the Valid rate = n_valid / n_executed (it replaces corner "
                       "coverage, which is undefined for a single recorded scenario), and higher is better. "
                       "The validity gate is not identical across rows: T8's gate (no teleport / wrong-way / "
                       "a_lat > 5 / v > 25 / >5 % outside the driving+shoulder union) for SVD_d5, Ours+KDE and "
                       "the real reference; the Table-5 valid_all gate for SAKURA, SAKURA-route and Ours; a "
                       "T8-like approximation from the Table-5 flags for SAKURA-route+KDE (its full Table-5 "
                       "gate, 0.013, is in tables3_detail).")
    NOTE_RARE_RECOV = ("This table's 3rd column is the 還原比例 Recovered frac. = the share of draws inside ε "
                       "(joint interaction ∧ path, the cutinl class ε) of the single recorded 39_180 scenario; "
                       "it replaces coverage@1000, which is undefined for one recorded scenario, and higher is "
                       "better. Variance at n = 1 is the trace ratio of the draws' z standardised by the cutinl "
                       "class real std.")
    NOTE_BGDEF_RARE = ("背景碰撞 / 出地圖 are read off two substrates in this table: for the SAKURA rows and Ours "
                       "they are the Table-5 chmed-horizon solid background hit rate (OBB SAT, >2 overlap frames, "
                       "depth ≥ 0.1 m) and the full-horizon off-road VL rate; for SVD_d5, Ours+KDE and the real "
                       "reference they are the T8 solid-hit rate on the GT support window (frames 2424-2923) and "
                       "offroad_n / n_executed on the 33_e7 driving+shoulder union (see the substrate note below).")
    NOTE_RARE_UNDEF = ("SVD_d5+KDE is undefined at N = 1 — and this is a precondition of the method, not a "
                       "failed run: the centred design matrix has rank N-1 = 0, so no basis exists, the LOO "
                       "bandwidth selector has nothing to select on, and zero samples are produced (T8 arm "
                       "svd_d5_singleton_N1). To keep the comparison from being made against an absent row, the "
                       "next row, SVD_d5 ext. basis+KDE, gives SVD exactly what N = 1 denies it: the basis and "
                       "the bandwidth are supplied from OUTSIDE the scenario (cutinl LOGO basis over 44 scenes "
                       "with 39_180's identity group excluded, h_ext = h_loo = 0.0886; T8 arm "
                       "svd_extbasis_gauss_h1), and it is executed at the same 300 draws, the same T8 substrate "
                       "and the same ε as Ours+KDE. That row, not the undefined one, is the matched opponent of "
                       "Ours+KDE, and Ours wins every column against it.")
    NOTE_RARE_INSAMPLE = ("In-sample disclosure, both directions. (a) The SVD_d5 similarity cells are the "
                         "fullfit arm, whose rank-5 basis is built from 48 matched cutinl scenes INCLUDING "
                         "39_180 (T8 svd_matched_fullfit), so the printed D_int / DTW flatter SVD; the held-out "
                         "LOGO reconstruction of the same scene is svd_matched_logo in tables3_detail, and the "
                         "few-shot sweep of scripts/55_fewshot.py puts its DTW at 1.841 m against the ours3 "
                         "training-free 1.074 m. (b) The other way: the Ours+KDE bandwidth (h = 0.5665) is "
                         "estimated from 48 cutinl disk contexts that DO include 39_180, while the "
                         "SVD ext.-basis row's basis and bandwidth are LOGO-44. The asymmetry is a scalar "
                         "bandwidth against a basis, and it favours Ours; it is stated here rather than "
                         "silently carried.")
    NOTE_RARE_SMALLN = ("N = 1 is the endpoint of an axis, not a special case. scripts/55_fewshot.py holds "
                        "39_180 out and sweeps the training-set size: no fit is possible at all for N in "
                        "{1, 2, 4} (numerical rank = N-1 < d = 5), and from N = 6 to the full pool of 44 the "
                        "held-out SVD D_m (0.745 / 0.588 / 0.276 / 0.398 / 0.563 at N = 6 / 8 / 16 / 32 / 44) "
                        "never reaches the training-free ours3 value of 0.121. SVD's parameter space is "
                        "estimated from a population, so its cost per scenario scales with the class; ours is "
                        "constructed from the single recording and is flat in N. The curve is "
                        "results/TABLE6B_FEWSHOT_REPORT.md and figures/fewshot_smalln_39_180.pdf.")
    NOTE_RARE_SUBSTRATE = ("The 背景碰撞 / 出地圖 substrates differ by row: T8 (GT-support horizon 2424-2923, "
                           "bg_solid_rate_all_gtsupport and offroad_n / n_executed on the 33_e7 driving+shoulder "
                           "union) for SVD_d5, Ours+KDE and the real reference; Table-5 (chmed horizon, VL "
                           "off-road rule) for the SAKURA rows and the Ours default. The class tables' executed-"
                           "polytrunc SVD convention therefore does NOT apply to this table: the SVD_d5 cells are "
                           "the T8 arm svd_matched_fullfit (analytic class-basis decode), SVD_d5+KDE is "
                           "undefined and the SVD_d5 ext. basis+KDE row is T8 svd_extbasis_gauss_h1. Each "
                           "cell's substrate is in tables3_lineage.csv.")
    class_notes = [NOTE_SIM, NOTE_DASH, NOTE_COV, NOTE_VAR, NOTE_BGDEF, NOTE_POLYTRUNC]
    if provisional:
        class_notes.append(NOTE_PROV)
    else:
        class_notes += [NOTE_STOP, NOTE_STOP_N, NOTE_STOP_PARKED, NOTE_REAL_REF]

    for tkey, tdef in TABLES.items():
        rows = [r for gk in tdef["groups"] for r in built[gk]]
        cols = COLS_RARE if tkey == "rarecase" else COLS_CLASS
        if tkey == "rarecase":
            notes = [bold_rule_note(cols), NOTE_SIM, NOTE_VAR, NOTE_RARE_RECOV, NOTE_RARE_VALID, NOTE_BGDEF_RARE]
            notes.append(NOTE_PROV_RARE if provisional else NOTE_RARE_WINDOW)
            notes += [NOTE_RARE_UNDEF, NOTE_RARE_INSAMPLE, NOTE_RARE_SMALLN, NOTE_RARE_SUBSTRATE]
        else:
            notes = [bold_rule_note(cols)] + list(class_notes)
        if tkey == "appendix_tlkeep":
            notes.insert(0, "SAKURA has no tlkeep arm, so this appendix table carries the SVD and ours arms only.")
            notes.insert(1, "tlkeep 相似度 comes from results/final/T2.csv (ours-vs-SVD cohort: ours3_disk, "
                            "svd_d5_fullfit, svd_d5_logo) because the 5-arm SAKURA denominator does not exist for "
                            "tlkeep. 互動相似 D_int is therefore normalised inside that 3-arm set and is NOT "
                            "comparable cell-by-cell with the other tables' D_int; 軌跡相似 is a raw metre value "
                            "and is comparable everywhere.")

        cap_md = (f"## Table {tdef['title']} — {prov_txt}method comparison per scenario group"
                  if provisional else f"## Table {tdef['title']} — method comparison per scenario group")
        md = md_table(rows, cols, cap_md, notes)
        tables_md[tkey] = md
        (OUT / f"{tdef['file']}.md").write_text(md)
        csv_table(rows, cols, tkey, OUT / f"{tdef['file']}.csv")

        # the 3rd/4th column and the validity substrate differ between the class tables and
        # the RareCase table, so the caption defines the columns THIS table actually prints
        # (verifier round 1, R2/R1).
        if tkey == "rarecase":
            cap_cols = (r"recovered fraction = the share of the draws that land inside $\varepsilon$ (joint, "
                        r"interaction $\wedge$ path, the cutinl class $\varepsilon$) of the single recorded "
                        r"39\_180 descriptor vector, replacing coverage, which is undefined for one recorded "
                        r"scenario; valid rate $=n_{\mathrm{valid}}/n_{\mathrm{executed}}$, replacing corner "
                        r"coverage, with a gate that differs per row (see the report); ")
            cap_val = (r"background collision and off-road come from two substrates in this table: T8 (GT-support "
                       r"horizon 2424--2923, driving+shoulder union) for SVD\_d5, Ours+KDE and the real reference, "
                       r"and Table-5 (chmed-horizon solid OBB hit, full-horizon VL off-road) for the SAKURA rows "
                       r"and Ours. ")
        else:
            cap_cols = (r"coverage and corner coverage = joint (interaction $\wedge$ path) coverage at a budget of "
                        r"1000 draws, real set matched489, mean over 3 seeds $\times$ 20 orderings, corner = the "
                        r"top-20\,\% most nearest-neighbour-isolated reals; ")
            cap_val = (r"background collision = chmed-horizon solid OBB hit rate; off-road = VL rate on the full "
                       r"horizon. ")
        cap_tex = ((r"\textbf{PROVISIONAL (ours validity = window execution).} " if provisional else "") +
                   f"{tdef['tex_title']} scenario group: method comparison. "
                   r"$D_{\mathrm{int}}$ = interaction similarity, the mean over the five interaction measures "
                   r"($|\Delta\mathrm{PET}|$, $|\Delta d_{\min}|$, $|\Delta\alpha|$, $\|\Delta c\|$, "
                   r"$|\Delta u_c|$) of (arm median / worst-arm median) on the arms' common scene set, unitless "
                   r"and only comparable inside its own group; DTW~[m] = trajectory similarity, the raw median "
                   r"DTW between the generated and the recorded target path in metres, not normalised; both "
                   r"lower is better, and the six-measure $D_m$ they replace is kept in tables3\_detail block "
                   r"D10 and in tables3\_sixmeasures; "
                   r"variance ratio = mean over the six interaction descriptors of "
                   r"$\mathrm{var}(z_{\mathrm{gen}})/\mathrm{var}(z_{\mathrm{real}})$ (1.0 = the recorded spread); "
                   + cap_cols + cap_val +
                   r"Best per column among the generated rows in bold "
                   r"(exact ties bold both); the real replay is a reference and is never bolded; "
                   r"-- = the column does not apply. ")
        if tkey == "rarecase":
            cap_tex += (r"The class tables' executed-polytrunc SVD convention does not apply here: the SVD\_d5 "
                        r"cells are the T8 arm svd\_matched\_fullfit and SVD\_d5+KDE is undefined at $N=1$ "
                        r"(centred rank 0, no LOO bandwidth, zero samples). ")
        else:
            cap_tex += r"SVD\_d5 rows use the executed polytrunc variants. "
        if provisional:
            cap_tex += ("The Ours rows' collision and off-road cells are the window executions and will be replaced "
                        "by stop-at-end executions. ")
        # load-bearing execution-end statement, in the CAPTION (the `% note:` lines below the table
        # are LaTeX comments and never render): see report section 9.
        elif tkey == "rarecase":
            cap_tex += (r"Execution-end convention: this table is \emph{not} stop-at-end. 39\_180 is in none of "
                        r"WP2's four class-level stop arms, so the Ours cells are the Table-5 window execution of "
                        r"the single scene and the Ours+KDE cells the T8 window executions of "
                        r"ours3\_condkde\_a2641. ")
        else:
            cap_tex += (r"Execution-end convention (collision / off-road only): the Ours rows are stop-at-end "
                        r"executions (ours3\_disk\_stop, 469 samples; ours3\_disk\_kde\_stop\_s20260910, 5000 "
                        r"samples = seed 20260910 only, so the Ours+KDE validity $n$ is 5000 and not the 15000 of "
                        r"the 3-seed window arm), the SVD\_d5 rows are polytrunc, and the SAKURA rows and the real "
                        r"reference are their original window executions; the real reference is Table 5's 511-replay "
                        r"cohort, not WP2's 489-scene one. The mixed convention is conservative against ours: pooled "
                        r"SVD reconstruction off-road is 0.043 under polytrunc versus 0.059 under stop-at-end, while "
                        r"the ours off-road rate is convention-invariant. ")
        tex = tex_table(rows, cols, cap_tex, f"tab:tables3-{tkey}", notes)
        (OUT / f"{tdef['file']}.tex").write_text(tex)
        tex_files.append(OUT / f"{tdef['file']}.tex")

        title_png = (f"{tdef['title']} — {STAMP}" if provisional else tdef["title"])
        # PNG footnotes were truncated to the first 4 notes, which dropped the PROVISIONAL
        # stamp and the substrate/polytrunc provenance; prioritise those and raise the cap
        # to 6 (verifier round 1, non-blocking note).
        png_notes = [n for n in notes if n.startswith("Bold rule")]
        png_notes += [n for n in notes if n.startswith("相似度 is split") and n not in png_notes]
        png_notes += [n for n in notes if n.startswith("PROVISIONAL") and n not in png_notes]
        png_notes += [n for n in notes if n.startswith("執行結束慣例") and n not in png_notes]
        png_notes += [n for n in notes if (n.startswith("Ours+KDE validity n")
                                           or n.startswith("A stopped target")
                                           or n.startswith("The real (reference) row")) and n not in png_notes]
        png_notes += [n for n in notes if (n.startswith("In this table the SVD_d5")
                                           or n.startswith("The 背景碰撞 / 出地圖 substrates")) and n not in png_notes]
        png_notes += [n for n in notes if n not in png_notes]
        png_meta.append(png_table(rows, cols, title_png, OUT / f"{tdef['file']}.png", png_notes[:10]))
        outputs += [OUT / f"{tdef['file']}{e}" for e in (".csv", ".md", ".tex", ".png")]

        for r in rows:
            for c in cols:
                cell = r["cells"][c["key"]]
                lineage.append(dict(table=tkey, group=r["group"], method=r["method_label"], column=c["key"],
                                    column_header=c["md"], value=fullprec(cell.value),
                                    display=cell.display, bold=bool(cell.bold),
                                    source_file=cell.source_file, source_row_selector=cell.selector,
                                    source_column=cell.source_column, upstream_source=cell.upstream,
                                    n=("" if cell.n is None else cell.n), note=cell.note))

    # six measures
    blocks, six_lin = six_measures(D)
    write_six(blocks, OUT, STAMP if provisional else "final")
    outputs += [OUT / f"tables3_sixmeasures{e}" for e in (".csv", ".md", ".tex")]
    tex_files.append(OUT / "tables3_sixmeasures.tex")
    for l in six_lin:
        lineage.append(dict(table="sixmeasures", group=l["group"], method=l["method"], column=l["column"],
                            column_header=l["column"], value=fullprec(l["value"]), display=f"{l['value']:.3f}",
                            bold=bool(l.get("bold", False)), source_file=l["source_file"],
                            source_row_selector=l["source_row_selector"],
                            source_column=l["source_column"], upstream_source="", n="", note=""))

    # detail
    detail = build_detail(D, amap)
    write_detail(detail, OUT, STAMP if provisional else "final")
    outputs += [OUT / "tables3_detail.csv", OUT / "tables3_detail.md"]

    lin = pd.DataFrame(lineage)
    lin.to_csv(OUT / "tables3_lineage.csv", index=False)
    outputs.append(OUT / "tables3_lineage.csv")

    tex_checks = check_pdflatex(tex_files)
    for t in tex_checks:
        print(f"[tex] {t['file']}: {'OK' if t['ok'] else 'FAILED'} {t.get('note','')[:200]}")
    if not all(t["ok"] for t in tex_checks):
        raise SystemExit("[tex] a .tex file did not compile")

    # ── report ──
    outputs.append(OUT / "validity_arm_map.json")
    write_report(D, amap, built, blocks, detail, lin, tex_checks, png_meta, provisional, STAMP, outputs)
    outputs.append(OUT / "TABLES3_REPORT.md")

    verify_reviewed_pins()
    with (OUT / "TABLES3_REPORT.md").open("a") as report:
        report.write("\n## v3 pin review (approved 2026-09-14)\n\n"
                     "The historical `HANDOFF_FILE_HASHES.sha256` is unchanged. "
                     "`HANDOFF_PINS_V3_20260914_REVIEW.json` records the 9 approved historical differences "
                     "and snapshot evidence. All 33 current files match "
                     "`HANDOFF_FILE_HASHES_V3_20260914.sha256`; 24 also match their original pins.\n")

    for p in outputs:
        print(f"  out {rel(p)}  {Path(p).stat().st_size} B")
    print(f"\n[done] {len(lin)} lineage rows, {len(detail)} detail rows, {time.monotonic() - t0:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
def write_report(D, amap, built, blocks, detail, lin, tex_checks, png_meta, provisional, STAMP, outputs):
    man = json.loads((OUT / "gen_metrics_manifest.json").read_text())
    L = []
    A = L.append
    A(f"# TABLES3 — thesis tables (左轉 / Cut-in / RareCase + tlkeep appendix)")
    A("")
    if provisional:
        A(f"> **{STAMP}.** Every table, figure and number below is provisional in exactly one respect: the "
          f"**Ours / Ours+KDE background-collision and off-road cells come from the window executions**. "
          f"HANDOFF_OPUS_20260911 amendment 3 requires stop-at-end executions (WP2: `ours3_disk_stop`, "
          f"`ours3_disk_kde_stop_s20260910`) so that the ours arms, like the SVD polytrunc arms, end at their "
          f"trajectory end instead of coasting at the last commanded speed. WP4 edits "
          f"`validity_arm_map.json` and re-runs `scripts/90_tables3_assemble.py`; nothing else changes.")
        A("")
    else:
        A("> **FINAL (WP4, 2026-09-11).** The Ours / Ours+KDE 背景碰撞 and 出地圖 cells of the CLASS tables now "
          "come from WP2's **stop-at-end** executions (`ours3_disk_stop`, 469; `ours3_disk_kde_stop_s20260910`, "
          "5 000 — **seed 20260910 only**), so the ours arms end at their trajectory end just as the polytrunc "
          "SVD arms do. Three things a reader must carry with these tables: (1) only 背景碰撞 and 出地圖 use "
          "stop-at-end — 互動相似, 軌跡相似, 變異比, 涵蓋 and 角落涵蓋 are unchanged window-execution numbers; (2) the "
          "Ours+KDE validity n fell from 15 000 (3 seeds) to 5 000 (1 seed) with the swap; (3) the **RareCase "
          "table is not stop-at-end at all** — 39_180 is outside every WP2 stop cohort. §1a defines the three "
          "conventions, §2 says which rows use which, §5 lists the caveats and §9 gives the cell-by-cell diff "
          "against the provisional run.")
        A("")
    A(f"Generated by `scripts/90_tables3_assemble.py` (sha256 `{sha256(Path(__file__))}`) on "
      f"{time.strftime('%Y-%m-%d %H:%M:%S')}. Stage A: `scripts/89_gen_metrics_tables3.py`.")
    A("")

    # 1. definitions
    A("## 1. Column definitions, exactly as computed")
    A("")
    A("| Column | Definition | Computed by | Source file → column |")
    A("| --- | --- | --- | --- |")
    A("| 互動相似 D_int ↓ | **Interaction similarity.** Mean over the FIVE interaction measures (\\|ΔPET\\| s, "
      "\\|Δd_min\\| m, \\|Δα\\| deg, ‖Δc‖ m, \\|Δu_c\\| m/s) of (arm median / worst-arm median b_k) on the arms' "
      "common scene set S_k — i.e. the six-measure D_m with DTW dropped from the average; the normalisation, the "
      "b_k and the S_k are unchanged. Unitless, 0–1, only comparable inside its own scenario group (b_k is a "
      "per-group worst-arm value). Reconstruction arms only. The printed n is the PET S_k, the smallest of the "
      "five (the per-measure n(S_k) are in the cell note of `tables3_lineage.csv` and in detail block D10). | "
      "`scripts/87_table2_sakura_arms.py` (5-arm set); `scripts/70_final_tables.py` (tlkeep) | "
      "`results/table2_sakura_arms.csv` → mean of `normalized` over measure ∈ {pet, dmin, alpha, cpoint, uc}; "
      "`results/final/T2.csv` → the same for tlkeep |")
    A("| 軌跡相似 DTW [m] ↓ | **Trajectory similarity.** The RAW median DTW distance between the generated and "
      "the recorded target path, in metres, on the same common scene set S_k. Not normalised, so unlike D_int it "
      "is comparable across groups and across tables. Identical, cell for cell, to the DTW column of "
      "`tables3_sixmeasures`. Reconstruction arms only. | as above | "
      "`results/table2_sakura_arms.csv` → `median_err_common_Sk` (measure = dtw); `results/final/T2.csv` → "
      "`median_err` (measure = dtw) for tlkeep |")
    A("| _(detail)_ 相似度 D_m ↓ | The **superseded** six-measure composite (the five interaction measures plus "
      "the normalised DTW). Split into the two columns above on 2026-09-11 at the user's request; kept, "
      "unchanged, as `sim_dm` in each table's `.csv`, as detail block D10 (together with the normalised DTW "
      "median/b_k and b_k itself) and as the `D_m` column of `tables3_sixmeasures.*`. No printed table uses it "
      "any more. | as above | `results/table2_sakura_arms.csv` → `D_m`; `results/final/T2.csv` → `D_m` for "
      "tlkeep |")
    A("| 變異比 Variance →1 | Mean over the six interaction descriptors (pet, d_min, α, conflict_x, conflict_y, u_c) "
      "of var(z_gen)/var(z_real,class), computed per seed on the draws with all six descriptors finite (int_finite), "
      "then averaged over the 3 seeds. z is standardised with the matched489 class real mean/std, so the real "
      "variance is 1 per descriptor. 1.0 = the recorded spread. | `scripts/89_gen_metrics_tables3.py` "
      "(`variance_ratio_trace`) | `results/final/tables3/gen_metrics.csv` → `value` |")
    A("| 涵蓋 Coverage@1000 ↑ | Fraction of the class's real scenarios that a budget of 1000 draws covers in the "
      "JOINT space (interaction ∧ path), pool = all draws, real_set = matched489, mean over 3 seeds and 20 random "
      "orderings — the rule of `scripts/46_coverage_similarity.py` (`hit_matrices` / `orderings` / `budget_curve`), "
      "reused verbatim by stage A. | `scripts/89_gen_metrics_tables3.py` (`coverage`) | "
      "`results/final/tables3/gen_metrics.csv` → `value` (reproduces `results/final/T3.csv` to 1.7e-16) |")
    A("| 角落涵蓋 Corner cov.@1000 ↑ | The same coverage@1000 rule restricted to the class's RARE reals: the "
      "top-20 % most isolated reals by nearest-neighbour distance to the other reals of the same class in the same "
      "standardised 6-D interaction space that defines ε, k = ceil(0.2 · n_rankable) (rankable = reals with all six "
      "descriptors finite). | `scripts/89_gen_metrics_tables3.py` (`corner_coverage`) | "
      "`results/final/tables3/gen_metrics.csv` → `value`; the rare scene list is `rare_scenes.csv` |")
    A("| 背景碰撞 Bg. collision ↓ | Share of samples with at least one SOLID background hit at the chmed horizon "
      "(OBB SAT overlap > 2 frames and penetration depth ≥ 0.1 m, vehicles only; chmed = min over arms of the "
      "arm-median driven time). **Two substrates:** the class tables and the SAKURA / Ours rows of the RareCase "
      "table use that Table-5 rate; the RareCase **SVD_d5, Ours+KDE and real** rows instead use the T8 substrate — "
      "the same solid-OBB rule against every rec00 vehicle track but on the GT support window (frames 2424–2923), "
      "a different horizon from chmed. | `scripts/45_validity_bg.py` / `45b` → `scripts/70_final_tables.py`; "
      "`scripts/52_e9_score.py` (T8) | `results/final/T5.csv` → `solid_hit_rate_chmed`; "
      "`results/table5_validity_summary_sakura.csv` → `any_solid_rate` (horizon = chmed); "
      "`results/final/T8.csv` → `bg_solid_rate_all_gtsupport` (RareCase SVD_d5 / Ours+KDE / real) |")
    A("| 出地圖 Off-road ↓ | Share of samples with more than 5 % of their window-clipped 30 Hz points outside the "
      "drivable union, full horizon (VL rule). **Two substrates:** as for 背景碰撞, the RareCase **SVD_d5, "
      "Ours+KDE and real** rows instead use the T8 rule — more than 5 % of points outside the tyms.xodr "
      "driving+shoulder union (the 33_e7 recipe), counted as `offroad_n / n_executed`, which is a different "
      "drivable-union definition from the Table-5 VL rate. | as above | `results/final/T5.csv` → "
      "`offroad_vl_rate_full`; `results/table5_validity_summary_sakura.csv` → `offroad_vl_rate` (horizon = full); "
      "`results/final/T8.csv` → `offroad_n / n_executed` (RareCase SVD_d5 / Ours+KDE / real) |")
    A("| 還原比例 Recovered frac. ↑ (RareCase only) | Coverage is undefined for one recorded scenario, so it becomes "
      "the fraction of draws that land inside ε (joint; the cutinl class ε) of the single recorded 39_180 descriptor "
      "vector. | `scripts/89_gen_metrics_tables3.py` (`recovered_fraction`) | "
      "`results/final/tables3/gen_metrics.csv` → `value` |")
    A("| 有效率 Valid rate ↑ (RareCase only) | n_valid / n_executed. This column replaces corner coverage at the "
      "user's request. The gate is per-row (see §5). | `scripts/52_e9_score.py` (T8 gate) / `scripts/45` (Table-5 "
      "gate) | `results/final/T8.csv` → `n_valid / n_executed`; `results/table5_validity_summary_sakura.csv` → "
      "`valid_all_rate`; `results/final/tables3/gen_metrics.csv` → `value` |")
    A("")
    A("Bold rule per column: 互動相似 / 軌跡相似 / 背景碰撞 / 出地圖 → lowest; 涵蓋 / 角落涵蓋 / 還原比例 / 有效率 → highest; "
      "變異比 → closest to 1.0. The best is chosen among the **generated** sub-rows of the same scenario group; "
      "exact ties (|Δ| ≤ 1e-12) bold every tied row; the real replay is a reference row and is never bolded. "
      "All cells are printed to 3 decimals.")
    A("")
    A("### 1a. The execution-end convention (背景碰撞 / 出地圖 only)")
    A("")
    A("esmini's default controller does not stop a trajectory-following target when its trajectory ends: it keeps "
      "the entity moving at the last commanded speed until the storyboard stops. Three conventions exist for "
      "ending an execution, and the validity columns mix them **by row**:")
    A("")
    A("| Convention | What it does | Used by |")
    A("| --- | --- | --- |")
    A("| **window** | the original execution: the target coasts at its last speed after the trajectory ends, and "
      "every row is kept | SAKURA / SAKURA-route / SAKURA-route+KDE, the real replay, and **every RareCase row** |")
    A("| **polytrunc** | the rows after the last polyline vertex are deleted before scoring, so the tail neither "
      "collides nor is exposed | SVD_d5 (`svd_exec_E3_recon_polytrunc_fullfit`), SVD_d5+KDE "
      "(`svd_exec_E3_kde_polytrunc_kde`) |")
    A("| **stop-at-end** | an `Agent1_StopAtEndEvent` (SpeedAction, step → 0 m/s) fires on the trajectory "
      "action's `completeState`, so the target brakes to 0 and stays there; the rows are kept, so the exposure "
      "`driven_s` is unchanged | Ours (`ours3_disk_stop`, 469), Ours+KDE (`ours3_disk_kde_stop_s20260910`, 5 000) "
      "— **class tables only** |")
    A("")
    A("Only 背景碰撞 and 出地圖 are affected. 互動相似, 軌跡相似, 變異比, 涵蓋 and 角落涵蓋 all come from the original window "
      "executions; WP2's paired six-measure check on `ours3_disk_stop` found a median paired difference of exactly "
      "0 for every measure except DTW in every class (pooled DTW median 0.290 stop vs 0.333 window, Holm "
      "p = 0.0003), so interaction fidelity does not depend on the convention.")
    A("")

    # 2. arm map
    A("## 2. Thesis row → project arm map")
    A("")
    A("**Every `Ours` number in these tables comes from the `ours3_disk*` family — the conflict-anchored "
      "parameterisation (θ₁, θ₂, EndSpeed; q⁻ fixed) on the EUCLIDEAN disk window of radius L = 10 m around q_c "
      "(469 scenes, `scripts/16_disk_window_extract.py`, contexts `results/ours3_disk_contexts.json`). No number in "
      "any main table comes from the arc-length window (`ours3_arc`) or from the legacy `cp3` / `petq3` "
      "experiments; those exist only as sensitivity rows inside the upstream files and are not read here.**")
    A("")
    A("One arm name does not literally match the `ours3_disk*` glob: the RareCase **Ours+KDE** arm "
      "`ours3_condkde_a2641`. It is the same conflict-anchored θ₁/θ₂/EndSpeed parameterisation on the same "
      "Euclidean disk window of radius L = 10 m — built by `scripts/51_e9_ours_jobs.py` from "
      "`results/disk_window_L10_special_39_180.csv` (geometry `special_pet_euclid_L10_exact_rotation`) with the "
      "bandwidth borrowed from the cutinl disk KDE fit — and it is the arm the frozen table definition prescribes "
      "for this row; 39_180 is a single recorded scenario that lies outside the 469-scene class cohort by "
      "construction, so no `ours3_disk*` batch contains it.")
    A("")
    A("| Thesis row | Similarity arm | Generative arm (variance / coverage / corner) | Validity arm (背景碰撞 / 出地圖) | n scenes / samples | Execution | Seeds |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    rowinfo = [
        ("SAKURA", "sakura_plain", "–", amap["class_rows"]["sakura"]["arm"], "213 scenes (192 sakura_bc + 21 extra)",
         "executed (esmini)", "–"),
        ("SAKURA-route (x % routed)", "sakura_route", "–", amap["class_rows"]["sakura_route"]["arm"],
         "213 scenes", "executed (esmini)", "–"),
        ("SAKURA-route+KDE", "–", "sakura_route_kde", amap["class_rows"]["sakura_route_kde"]["arm"],
         "3 seeds × 1000 per class (12 000)", "executed (esmini)", "20260910/11/12"),
        ("SVD_d5", "svd_d5_fullfit", "–", amap["class_rows"]["svd_d5"]["arm"],
         "489 reconstructions", "similarity analytic; validity executed E3 timed Polyline (polytrunc)", "–"),
        ("SVD_d5+KDE", "–", "svd_d5_kde_matched_analytic", amap["class_rows"]["svd_d5_kde"]["arm"],
         "3 seeds × 1000 per class (15 000 analytic); validity 300 per class executed",
         "descriptors analytic; validity executed E3 (polytrunc)", "20260910/11/12"),
        ("Ours", "ours3_disk", "–", amap["class_rows"]["ours"]["arm"], "469 scenes",
         "executed (esmini)" + (" — PROVISIONAL window" if provisional else " — **stop-at-end**"), "–"),
        ("Ours+KDE", "–", "ours3_disk_kde", amap["class_rows"]["ours_kde"]["arm"],
         ("3 seeds × 1000 per class (15 000)" if provisional else
          "descriptors 3 seeds × 1000 per class (15 000); **validity 1 seed × 1000 per class (5 000)**"),
         "executed (esmini)" + (" — PROVISIONAL window" if provisional else " — **stop-at-end**"),
         "20260910/11/12" + ("" if provisional else " (validity: **20260910 only**)")),
        ("real (reference)", "–", "–", amap["class_rows"]["real"]["arm"], "511 replays", "recorded target replay", "–"),
    ]
    for r in rowinfo:
        A("| " + " | ".join(r) + " |")
    A("")
    A("RareCase (39_180) row → arm:")
    A("")
    A("| Thesis row | Arm | Validity resolver | Valid gate | Execution end |")
    A("| --- | --- | --- | --- | --- |")
    for k, ent in amap["rarecase_rows"].items():
        if k.startswith("_"):
            continue
        lab = next(m["label"] for m in METHODS if m["key"] == k)
        A(f"| {lab} | `{ent['arm']}` | {ent.get('resolver')} | {ent.get('valid_gate', '–')} | "
          f"{ent.get('execution_kind', '–')} |")
    A("")
    A("**No RareCase row is stop-at-end.** WP2 executed exactly four class-level stop arms — `ours3_disk_stop` "
      "(469), `ours3_disk_kde_stop_s20260910` (5 000), `svd_exec_stop_recon_fullfit` (489) and "
      "`svd_exec_stop_kde` (1 500) — and 39_180 is in none of their cohorts. The RareCase Ours row is therefore "
      "the Table-5 **window** execution of the single 39_180 scene and the Ours+KDE row the T8 **window** "
      "executions of `ours3_condkde_a2641` (300 draws); the SVD_d5 row is an analytic class-basis decode on the "
      "T8 substrate. The RareCase caption states this, so the table never implies the stop-at-end convention of "
      "the class tables.")
    A("")
    A("Execution-end convention, cell by cell (this is the only thing WP4 changed):")
    A("")
    A("| Table | Rows | Execution end | Arm(s) | n (validity) | Batch |")
    A("| --- | --- | --- | --- | ---: | --- |")
    A("| 左轉 / Cut-in / tlkeep | Ours | **stop-at-end** | `ours3_disk_stop` | 469 | "
      "`runs/ours3_disk_stop/f8305650479ea9c2` |")
    A("| 左轉 / Cut-in / tlkeep | Ours+KDE | **stop-at-end**, seed 20260910 only | "
      "`ours3_disk_kde_stop_s20260910` | 5 000 | `runs/ours3_disk_kde_stop_s20260910/b8af10e674c24622` |")
    A("| 左轉 / Cut-in / tlkeep | SVD_d5 | polytrunc | `svd_exec_E3_recon_polytrunc_fullfit` | 489 | (E3, "
      "scripts/26–28) |")
    A("| 左轉 / Cut-in / tlkeep | SVD_d5+KDE | polytrunc | `svd_exec_E3_kde_polytrunc_kde` | 1 500 | (E3) |")
    A("| 左轉 / Cut-in | SAKURA / SAKURA-route / SAKURA-route+KDE | window | `sakura_plain`, `sakura_route`, "
      "`sakura_route_kde` | 213 / 213 / 12 000 | (scripts/8x) |")
    A("| all | real (reference) | recorded replay, Table 5 cohort | `real` (T5) | 511 pooled | (scripts/45) |")
    A("| RareCase | every row | **window — not re-executed with stop-at-end** | `ours3_disk` (1 scene), "
      "`ours3_condkde_a2641` (300), `svd_matched_fullfit`, SAKURA rows, `replay` | see the table above | (T8 / "
      "Table-5) |")
    A("")
    A("Two WP2 arms are not printed anywhere but exist as the consistency check that stop-at-end ≈ polytrunc: "
      "`svd_exec_stop_recon_fullfit` (489) and `svd_exec_stop_kde` (1 500). Their numbers are in "
      "`tables3_detail` block D7.")
    A("")
    A(f"The map itself is `results/final/tables3/validity_arm_map.json` (provisional = "
      f"`{str(amap.get('provisional')).lower()}`); it is the only file that has to be edited to change which "
      f"execution feeds a validity cell, and every row carries an `execution_kind` string that is copied into "
      f"the cell notes of `tables3_lineage.csv`.")
    A("")

    # 3. the tables inline
    A("## 3. The tables")
    A("")
    for tkey, tdef in TABLES.items():
        A((OUT / f"{tdef['file']}.md").read_text().replace("## Table ", "### Table ", 1))
        A("")
    A("### Six-measure detail (paper_tables style)")
    A("")
    A((OUT / "tables3_sixmeasures.md").read_text().split("\n", 2)[2])
    A("")

    # 4. n's
    A("## 4. Per-group n: class size vs per-arm cohort")
    A("")
    A("The `n` printed in the tables is the CLASS size (the number of recorded scenarios of that class), which is "
      "what the paper-table layout shows. Every arm has its own, smaller cohort, and the similarity column is "
      "computed on the intersection S_k of the compared arms:")
    A("")
    t1 = D["T1"]
    A("| Group | class n (table) | population (T1) | ours disk cohort | SVD matched | SAKURA bases | common S_k (α,d_min,c,u_c) | common S_k (PET) | coverage real set (matched489, rankable) | rare k |")
    A("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for g in GROUPS:
        gk = g["key"]
        t1r = t1[t1["class"] == gk]
        if gk == "special_39_180":
            pop, disk, svd, sak = 1, 1, 1, 1
        elif len(t1r):
            t1r = t1r.iloc[0]
            pop, disk, svd, sak = (int(t1r["n_population_513"]), int(t1r["n_disk_cohort_469"]),
                                   int(t1r["n_svd_d5_matched_eligible"]), int(t1r["n_sakura_bc_available"]))
        else:
            pop = disk = svd = sak = 0
        sc = D["SAKURA_MANIFEST"].get("scenes", {}).get(gk)
        sak = int(sc["n_bases"]) if sc else 0
        b = next(b for b in blocks if b["group"] == gk)
        cov = gen_lookup(D, "n_real", table_group="coverage", metric="coverage", **{"class": gk},
                         real_set="matched489", arm=("ours3_disk_kde"), pool="all", space="joint", budget=1000.0)
        rare = gen_lookup(D, "n_rare", table_group="corner_coverage", metric="corner_coverage", **{"class": gk},
                          real_set="matched489", arm=("ours3_disk_kde"), pool="all", space="joint", budget=1000.0)
        A(f"| {g['label']} | {g['n_class']} | {pop} | {disk} | {svd} | {sak} | {b['nper']['alpha']} | "
          f"{b['nper']['pet']} | {'n/a' if cov is None else int(cov['n_real'])} | "
          f"{'n/a' if rare is None else int(rare['n_rare'])} |")
    A("")
    A("Why they differ: (a) the disk cohort drops scenarios whose q⁻ or q⁺ falls outside the recorded support of the "
      "Euclidean L = 10 m window (keeptl loses 11 of 50, cutinr 7 of 20); (b) the SVD matched cohort keeps the 489 "
      "scenarios whose source 50-point path exists; (c) SAKURA needs an existing exact-time-window template, so it "
      "has no tlkeep arm, and its cutinr arm has 20 (not 22) bases — 0 of them from the sakura_bc batch and all 20 "
      "from sakura_plain_extra, which is why T1's `n_sakura_bc_available` reads 0 for cutinr; (d) S_k is the intersection of the compared arms **and** the "
      "measure has to be finite in both, which is why PET has a smaller S_k (a generated or recorded run can lose "
      "the PET event); (e) the coverage real set counts only reals with all six descriptors finite. "
      "Cut-in (right) is printed as n = 20 (the number of usable bases, matching the paper-table layout) while the "
      "recorded population is 22.")
    A("")

    # 5. caveats
    A("## 5. Caveats to state in the thesis")
    A("")
    cav = [
        "**相似度 is now TWO columns, and they are not on the same footing.** 互動相似 D_int is a normalised, "
        "unitless 0–1 score: every measure is divided by the worst arm's median b_k *within its own scenario "
        "group*, so a D_int of 0.39 in one group and 0.39 in another do not mean the same discrepancy, and the "
        "column ranks arms inside a group only. 軌跡相似 is a raw metre value and does not have that problem. "
        "The two also use slightly different scene sets: D_int's printed n is the PET S_k (the smallest of the "
        "five, e.g. 31 of 38 for keeptl, 2 of 11 for cutinr) while the four other interaction measures and DTW "
        "are medians over the larger S_k — the per-measure counts are in `tables3_sixmeasures` and in detail "
        "block D10. The cutinr D_int in particular rests on a 2-scene PET median.",
        "**The six-measure D_m is superseded, not deleted.** It is still computed and still exported — "
        "`tables3_detail` block D10, the `sim_dm` column of each table `.csv`, and the `D_m` column of "
        "`tables3_sixmeasures.*` — so any earlier text quoting D_m can still be traced. No printed thesis table "
        "uses it, and the three are related exactly: D_int = (6·D_m − DTW_normalised)/5. The split does not "
        "change the interaction half — D_int bolds the same arm as D_m in all six groups — but it stops the "
        "path half from being averaged away: the anchored parameterisation has the lowest raw DTW in every one "
        "of the six groups, including the three (Agent-LT N→E, Agent-LT S→W, Cut-in (right)) that the SVD "
        "reconstruction wins on D_m and on D_int.",
        "**SVD_d5+KDE descriptor metrics are ANALYTIC, ours are EXECUTED.** The variance, coverage and corner "
        "coverage of SVD_d5+KDE come from the analytic decode arm `svd_d5_kde_matched_analytic` (15 000 draws), while "
        "ours3_disk_kde is scored from actual esmini executions. The executed SVD KDE arm "
        "(`svd_d5_kde_executed_E3`, 100 draws per class per seed) is in `tables3_detail` and is far more dispersed "
        "(e.g. cutinl trace ratio 46.5 vs 28.4 analytic).",
        "**Both KDE samplers are in-sample.** Every KDE is fitted on the same class the coverage is measured "
        "against; nothing here is held-out generalisation.",
        "**SAKURA-route+KDE offsets are clipped to ±0.5 m** around the base scenario's own lane offset, and 65–82 % "
        "of the draws sit at the bound (keeptl 799/1000, keeptl_sw 796, cutinl 774, cutinr 647 at seed 20260910), so "
        "its executed offset distribution is not the fitted class distribution. The unclipped pilot "
        "(`sakura_route_kde_noclip`, 100 draws per class) is in `tables3_detail`.",
        "**real_set choice.** Coverage / corner coverage use `matched489` for every generative arm, because it is the "
        "only real set all three KDE arms share. The `disk469` variant (which drops the reals that have no ours "
        "kernel centre and therefore can never be path-covered by ours) is in `tables3_detail`.",
        "**Bold rules** are a display convention, not a test: no confidence interval or multiplicity correction is "
        "attached to the bold marks. The inferential statements live in T2 (Holm-corrected contrasts, in "
        "`tables3_detail` block D1) and T5.",
        "**There is no “(TBD)” sub-row.** The user removed the redacted placeholder on 2026-09-11; the method "
        "sub-rows are exactly SAKURA, SAKURA-route, SAKURA-route+KDE, SVD_d5, SVD_d5+KDE, Ours, Ours+KDE and the "
        "real reference.",
        "**The RareCase 4th column is the Valid rate**, filling the column the user redacted; corner coverage is "
        "undefined for a single recorded scenario. The gate differs per row (T8 gate for SVD_d5 / Ours+KDE / real; "
        "Table-5 valid_all for SAKURA / SAKURA-route / Ours; a T8-like Table-5-flag approximation for "
        "SAKURA-route+KDE, whose strict Table-5 gate is 0.013), so the column ranks arms only loosely.",
        "**Window vs polytrunc for the executed SVD arms.** Amendment 3 fixes the SVD_d5 and SVD_d5+KDE validity "
        "cells to the POLYTRUNC variants (`svd_exec_E3_recon_polytrunc_fullfit`, `svd_exec_E3_kde_polytrunc_kde`), "
        "which stop at the last polyline vertex. The window variants (which keep driving), the analytic decode and "
        "WP2's stop-at-end SVD arms are all in `tables3_detail` block D7; every other Table-5 arm is in block D5, "
        "which is also where the two conventions can be compared on teleport rate (`teleport_rate_full`, tlkeep "
        "SVD KDE 0.257 window vs 0.073 polytrunc).",
        "**Only the validity columns are affected by the execution-end convention.** 互動相似, 軌跡相似, 變異比, 涵蓋 and "
        "角落涵蓋 all come from the original WINDOW executions for every arm, including Ours and Ours+KDE, and are "
        "unchanged by the stop-at-end work. WP2's paired six-measure comparison of `ours3_disk_stop` against the "
        "window execution gives a median paired difference of exactly 0 for |ΔPET|, |Δd_min|, |Δα|, ‖Δc‖ and "
        "|Δu_c| in every class; only DTW moves (pooled median 0.290 stop vs 0.333 window, Holm p = 0.0003). The "
        "window validity numbers stay in `tables3_detail` block D7 for comparison.",
        "**The tlkeep appendix 互動相似 uses a different denominator** (T2's ours-vs-SVD cohort: ours3_disk, "
        "svd_d5_fullfit, svd_d5_logo) because SAKURA has no tlkeep arm, so its D_int is not comparable "
        "cell-by-cell with the other tables. Its 軌跡相似 column is a raw metre value and IS comparable.",
        "**The real reference row is scored on the T5 substrate** (`results/final/T5.csv`, arm `real`: 50 / 81 / 61 / "
        "22 / 297 replays per class), while the SAKURA rows are scored on the SAKURA substrate "
        "(`results/table5_validity_summary_sakura.csv`), whose own `real` row covers a slightly different scene set "
        "(cutinl 60 replays, 0.000 solid; cutinr 20 replays, 0.050 solid / 0.250 off-road). Both real rows are in "
        "`tables3_detail` block D5; the difference is a scene-set difference, not a scoring difference.",
        "**RareCase substrates were cross-checked.** The 39_180 background-collision / off-road cells mix two "
        "substrates (T8's GT-support horizon for SVD_d5, Ours+KDE and the real reference; Table-5's chmed horizon "
        "and VL off-road rule for the SAKURA rows and the Ours default). On the COMMON Table-5 substrate "
        "(`tables3_detail` block D9) the same ordering holds: the SVD class-basis decode of 39_180 is 1.000 solid "
        "and 1.000 off-road (identical to its T8 numbers) while the ours default is 0.000 solid and 1.000 off-road, "
        "so the bold on that column is not an artefact of the mixed provenance.",
        "**Every generated arm collides with background traffic far more often than the recorded replay** "
        "(real 0.02–0.14 vs 0.36–1.00 here). The 背景碰撞 column supports relative statements between samplers, not "
        "an absolute “background-safe” claim.",
    ]
    if provisional:
        cav.insert(0, "**PROVISIONAL — the Ours / Ours+KDE 背景碰撞 and 出地圖 cells are WINDOW executions.** esmini's "
                      "default controller keeps the target moving at its last speed after the trajectory ends, so "
                      "the ours arms are exposed to background traffic for longer than the polytrunc SVD arms. WP2 "
                      "produces the stop-at-end executions and WP4 swaps them in through `validity_arm_map.json`. "
                      "The window numbers used here are also listed in `tables3_detail` block D7 so the swap can be "
                      "audited.")
    else:
        cav[0:0] = [
            "**The execution-end convention is MIXED, and it is mixed against our own method.** Ours / Ours+KDE "
            "背景碰撞 and 出地圖 are stop-at-end executions; SVD_d5 / SVD_d5+KDE are polytrunc; SAKURA and the real "
            "reference are window. Polytrunc *deletes* the post-trajectory rows, stop-at-end *keeps* them with the "
            "target parked, so the two are not the same treatment. On the SVD reconstruction arm, pooled off-road "
            "is 0.043 under polytrunc but 0.059 under stop-at-end — i.e. the SVD row is quoted at its LOWER "
            "off-road number — while the ours off-road rate is convention-invariant (0.032 window → 0.032 "
            "stop-at-end, 0 of 469 samples change) and ours teleport is 0.000 under both. A reviewer will ask about "
            "the mixed convention; the answer is that it is conservative against us. On teleport the two "
            "conventions do agree (SVD reconstruction 0.055 stop vs 0.053 polytrunc; SVD KDE 0.045 vs 0.043).",
            "**Ours+KDE validity changed cohort as well as convention: n 15 000 → 5 000, seed coverage 3 → 1.** "
            "Only seed 20260910 was re-executed with stop-at-end (`ours3_disk_kde_stop_s20260910`), so the "
            "Ours+KDE 背景碰撞 / 出地圖 cells are a single-seed estimate. WP2 also scored the sample-paired "
            "single-seed WINDOW arm (`ours3_disk_kde_s20260910_window`) precisely so the two effects can be "
            "separated, and `tables3_detail` block D7 prints all three (3-seed window 15 000, 1-seed window 5 000, "
            "1-seed stop 5 000) per class and pooled. Pooled chmed solid hit: 0.687 (3-seed window) → 0.694 "
            "(1-seed window) → 0.691 (1-seed stop), so most of the change in these cells is the SEED restriction, "
            "not the stop. The 變異比 / 涵蓋 / 角落涵蓋 cells of the same row still use all three seeds.",
            "**Stop-at-end leaves a parked body on the road, which manufactures collisions at the full horizon.** "
            "Pooled full-horizon any-solid rises from 0.563 to 0.738 (Ours) and from 0.458 to 0.675 (SVD "
            "reconstruction) when the tail is frozen instead of coasting. At the **chmed** horizon these tables "
            "actually use, the effect is nearly nil: the number of samples whose ONLY solid hits start after the "
            "stop is 0/469 (`ours3_disk_stop`), 17/5 000 (`ours3_disk_kde_stop_s20260910`), 0/489 "
            "(`svd_exec_stop_recon_fullfit`) and 9/1 500 (`svd_exec_stop_kde`). The much larger counts quoted in "
            "`results/x_stop/RESULT_WP2.md` §6 deviation 4 — 92/469, 293/5 000, 102/489, 52/1 500 — are "
            "FULL-horizon numbers and must always carry that qualifier.",
            "**The instantaneous step to zero speed is itself unphysical, and Table 5's physics gate cannot see "
            "it.** `phys_gate1` tests only v_max > 25 m/s or a_lat > 5 m/s²; it has no longitudinal-acceleration "
            "term. Stop-at-end therefore removes the lateral-acceleration spikes that esmini's default controller "
            "produced when it re-acquired the road (pooled chmed `phys_gate1` 0.064 → 0.009 for Ours and 0.133 → "
            "0.028 for Ours+KDE) without being charged for the −∞ deceleration it introduces. The resulting "
            "validity gain — pooled chmed valid ∧ critical 0.358 → 0.367 (Ours) and 0.173 → 0.184 (Ours+KDE) — is "
            "a gain against the gate as defined, not evidence of a more physical trajectory.",
            "**The stop trigger costs about 2/30 s of the last speed.** Median residual free-run after the "
            "trajectory end: 0.273 m (`ours3_disk_stop`), 0.220 m (`ours3_disk_kde_stop_s20260910`), 0.216 m "
            "(`svd_exec_stop_recon_fullfit`), 0.192 m (`svd_exec_stop_kde`). `svd_exec_stop_kde` still retains "
            "more than 1 m on 24 of its 1 079 stop-proved samples and more than 5 m on 11 (max 13.7 m, on a "
            "degenerate draw whose applied duration was clipped up from a negative value). Polytrunc has no such "
            "residual, so the two conventions are not exactly equivalent even where they agree.",
            "**The real (reference) row is Table 5's cohort, not WP2's.** WP2 scored its arms over the 489 scenes "
            "the stop arms cover, so its pooled real reference is n = 487 at chmed (0.037 solid, 0.041 off-road) "
            "and its class real rows differ from Table 5's — e.g. keeptl_sw 76 replays / 0.053 solid instead of 81 "
            "/ 0.074, cutinl 48 / 0.021 instead of 61 / 0.016, cutinr 20 / 0.050 / 0.250 instead of 22 / 0.136 / "
            "0.227. **These tables keep Table 5's real row** (`results/final/T5.csv`, arm `real`), so the real "
            "reference is NOT sample-paired with the stop arms; WP2's cohort-matched real row is in "
            "`tables3_detail` block D7 for anyone who wants the paired comparison.",
            "**The RareCase table is not stop-at-end at all.** WP2's four stop arms are class-level and 39_180 is "
            "in none of them, so the RareCase Ours cells are the Table-5 WINDOW execution of the single scene and "
            "the Ours+KDE cells the T8 WINDOW executions of `ours3_condkde_a2641`. The RareCase caption says so; do "
            "not describe that table as stop-at-end.",
        ]
    for c in cav:
        A(f"- {c}")
    A("")

    # 6. findings
    A("## 6. What the tables show (neutral reading)")
    A("")
    A("Best generated sub-row per column per group (bold in the tables):")
    A("")
    A("| Group | 互動相似 D_int | 軌跡相似 DTW [m] | 變異比 | 涵蓋@1000 | 角落涵蓋@1000 | 背景碰撞 | 出地圖 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for g in GROUPS:
        if g["key"] == "special_39_180":
            continue
        rows = built[g["key"]]
        cells = []
        for c in COLS_CLASS:
            win = [r["method_label"] for r in rows if r["cells"][c["key"]].bold]
            vals = [f"{r['cells'][c['key']].value:.3f}" for r in rows if r["cells"][c["key"]].bold]
            cells.append(" / ".join(f"{w} {v}" for w, v in zip(win, vals)) if win else "–")
        A(f"| {g['label']} | " + " | ".join(cells) + " |")
    rr = built["special_39_180"]
    cells = []
    for c in COLS_RARE:
        win = [r["method_label"] for r in rr if r["cells"][c["key"]].bold]
        vals = [f"{r['cells'][c['key']].value:.3f}" for r in rr if r["cells"][c["key"]].bold]
        cells.append(" / ".join(f"{w} {v}" for w, v in zip(win, vals)) if win else "–")
    A("")
    A("| Group | 互動相似 D_int | 軌跡相似 DTW [m] | 變異比 | 還原比例 | 有效率 | 背景碰撞 | 出地圖 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    A("| Special 39_180 | " + " | ".join(cells) + " |")
    A("")
    # The per-column reading is GENERATED from the bold sets, so it can never drift away from the
    # tables (the hand-written version of this paragraph had already drifted: it claimed
    # SAKURA-route was the least off-road arm in the left-turn classes, where the bold is on Ours).
    A("Column by column, generated from the bold sets above (group: winner value):")
    A("")
    for c in COLS_CLASS:
        parts = []
        for g in GROUPS:
            if g["key"] == "special_39_180":
                continue
            rows_g = built[g["key"]]
            w = [(r["method_label"], r["cells"][c["key"]].value) for r in rows_g if r["cells"][c["key"]].bold]
            if w:
                parts.append(f"{g['label']}: " + " = ".join(f"{a} {b:.3f}" for a, b in w))
        A(f"- **{col_name(c)}** — " + "; ".join(parts) + ".")
    A("")
    # background-collision headroom vs the real replay, computed rather than asserted
    ratios = []
    for g in GROUPS:
        if g["key"] == "special_39_180":
            continue
        rows_g = built[g["key"]]
        gen = [r["cells"]["coll"].value for r in rows_g if r["kind"] != "real" and r["cells"]["coll"].value is not None]
        realv = next((r["cells"]["coll"].value for r in rows_g if r["kind"] == "real"), None)
        if gen and realv:
            ratios.append((g["label"], min(gen) / realv, max(gen) / realv))
    A("Background collision against the recorded replay, per group (lowest and highest generated rate as a "
      "multiple of the real rate): "
      + "; ".join(f"{lab} {lo:.1f}×–{hi:.1f}×" for lab, lo, hi in ratios) + ".")
    A("")
    # Win counts, generated. The hand-written version of this paragraph had drifted from the
    # tables on three claims (similarity in cut-in (right) and tlkeep, coverage in cut-in (right),
    # variance win count), so it is now derived from the same bold sets the tables render.
    A("Win counts over the five class groups (a tie counts for every tied arm):")
    A("")
    for c in COLS_CLASS:
        cnt = {}
        for g in GROUPS:
            if g["key"] == "special_39_180":
                continue
            for r in built[g["key"]]:
                if r["cells"][c["key"]].bold:
                    cnt.setdefault(r["method_label"], []).append(g["label"])
        if not cnt:
            continue
        parts = sorted(cnt.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        A(f"- **{col_name(c)}** — " + "; ".join(f"{k} {len(v)}/5 ({', '.join(v)})" for k, v in parts) + ".")
    A("")
    # The similarity sentence is GENERATED from the bold sets of the two split columns over all six
    # groups (the four classes + 39_180 + tlkeep), so it cannot drift from the tables.
    def _simwin(colkey):
        cnt = {}
        for g in GROUPS:
            for r in built[g["key"]]:
                if r["cells"][colkey].bold:
                    cnt.setdefault(r["method_label"], []).append(g["label"])
        return "; ".join(f"{k} {len(v)}/6 ({', '.join(v)})"
                         for k, v in sorted(cnt.items(), key=lambda kv: (-len(kv[1]), kv[0])))
    A("Read neutrally: splitting 相似度 shows that the two halves of the old composite do not agree. Over all six "
      "groups (the four classes, the 39_180 singleton and the tlkeep appendix) 互動相似 D_int is won by — "
      + _simwin("sim_int") + " — while 軌跡相似, the raw median DTW to the recorded target path, is won by — "
      + _simwin("sim_dtw") + ". The old single D_m column (block D10) hid that split by averaging the two "
      "together. The anchored KDE sampler wins or ties the class's RARE-corner coverage in every class group, "
      "while plain class coverage is split with the SVD KDE. On 變異比 the anchored KDE is closest to the recorded "
      "spread in four of the five groups. The 背景碰撞 and 出地圖 winners are listed per group above rather than "
      "summarised in prose, because they mix three execution-end conventions (stop-at-end for ours, polytrunc for "
      "SVD, window for SAKURA and the real replay) — see §1a and §5. Whatever the winner, every generated arm "
      "collides with background traffic far more often than the recorded replay, by the multiples listed above, so "
      "the column supports relative statements between samplers and not an absolute 'background-safe' claim. The "
      "39_180 singleton is the one place where SVD d5+KDE cannot be built at all.")
    A("")

    # 7. files
    A("## 7. Files (sha256)")
    A("")
    A("| File | bytes | sha256 |")
    A("| --- | ---: | --- |")
    for p in sorted(set(Path(o) for o in outputs), key=lambda p: p.name):
        if p.exists():
            A(f"| `{rel(p)}` | {p.stat().st_size} | `{sha256(p)}` |")
    A("| `results/final/tables3/TABLES3_REPORT.md` | (this file) | see "
      + ("`RESULT_WP1.md`" if provisional else "`RESULT_WP4.md`") + " |")
    A("")
    A("Inputs read (all read-only):")
    A("")
    A("| File | sha256 |")
    A("| --- | --- |")
    for k, v in SRC.items():
        A(f"| `{rel(v)}` | `{sha256(v)}` |")
    A(f"| `{rel(RES / 'table5_validity_summary.csv')}` | `{sha256(RES / 'table5_validity_summary.csv')}` |")
    for s in sorted({PROJECT / v["path"] for v in amap["sources"].values()}):
        if s.exists() and s not in set(SRC.values()) and s != RES / "table5_validity_summary.csv":
            A(f"| `{rel(s)}` | `{sha256(s)}` |")
    A("")
    A("LaTeX compile check (`pdflatex -interaction=nonstopmode -halt-on-error` in a scratch dir, wrapped in a "
      "minimal article with booktabs + multirow):")
    A("")
    for t in tex_checks:
        A(f"- `{t['file']}` → {'OK' if t['ok'] else 'FAILED'} ({t.get('pdf_bytes', 0)} B PDF)")
    A("")
    A("PNG render check (matplotlib, CJK font chain " + " → ".join(CJK_FONTS) + "; zero missing-glyph warnings):")
    A("")
    for m in png_meta:
        A(f"- `{m['path']}` → {m['bytes']} B, {m['glyph_warnings']} glyph warnings")
    A("")
    ck = man.get("checks", {})
    A("Stage A reproduction gate (from `gen_metrics_manifest.json` → `checks`):")
    A("")
    for k in ("reproduction_gate_n_cells", "reproduction_gate_missing_reference_rows",
              "reproduction_gate_max_abs_diff_per_seed", "reproduction_gate_max_abs_diff_seed_mean_vs_T3",
              "reproduction_all_cells_n", "reproduction_all_cells_max_abs_diff",
              "real_scaling_vs_table34_manifest_max_abs_diff", "eps_vs_table34_manifest_max_abs_diff"):
        if k in ck:
            A(f"- `{k}` = {ck[k]}")
    A("")
    # 8. corrections
    A("## 8. Corrections (WP1 verifier round 1)")
    A("")
    A("An adversarial verifier returned CONFIRMED on the NUMBER-LINEAGE lens (all 359 traced cells matched) and "
      "REFUTED on the DEFINITIONS/FORMAT lens with four refutations, all of them caption or report **text**. "
      "The fixes below changed only strings in `scripts/90_tables3_assemble.py`; **no number, arm, selector or "
      "bold rule was touched**, and the four main-table `.csv` files are byte-identical to the pre-correction run "
      "(verified by sha256 in `RESULT_WP1.md`).")
    A("")
    A("| # | Refutation | What changed |")
    A("| --- | --- | --- |")
    A("| R1 | The RareCase caption asserted \"SVD_d5 / SVD_d5+KDE validity cells use the EXECUTED polytrunc "
      "variants\" (LaTeX: \"SVD\\_d5 rows use the executed polytrunc variants\"). False for this table: its "
      "SVD_d5 validity cells resolve through `t8` to arm `svd_matched_fullfit` and SVD_d5+KDE is undefined; the "
      "string `svd_exec_E3` appears in 0 RareCase lineage rows. | The note is now built per table. The class "
      "tables and the tlkeep appendix carry it, scoped (\"In this table …; this convention applies to the class "
      "tables only — the RareCase SVD_d5 cells come from T8 (arm svd_matched_fullfit)\"); the RareCase table does "
      "not emit it at all, and its substrate bullet now states the T8 provenance and that the polytrunc convention "
      "does not apply. The `.tex` caption is likewise conditional. |")
    A("| R2 | The bold-rule note named 涵蓋 / 角落涵蓋, which the RareCase table does not have, and gave no "
      "direction for its two own columns (還原比例, 有效率); the `.tex` caption defined coverage / corner coverage "
      "(budget 1000, matched489, 3 seeds × 20 orderings, top-20 %) — none of them a column of that table — and "
      "never defined \"Recovered frac.\". | The bold-rule note is generated from the table's own column list "
      "(`bold_rule_note(cols)`, using `COLS_RARE` / `COLS_CLASS`), so the RareCase note now reads 互動相似 D_int / "
      "軌跡相似 DTW [m] / 背景碰撞 / 出地圖 → lowest; 還原比例 / 有效率 → highest; 變異比 → closest to 1.0. In the `.tex` caption the "
      "coverage/corner sentence is replaced, for the RareCase table only, by the recovered-fraction and valid-rate "
      "definitions, and the background-collision / off-road sentence by the two-substrate statement. |")
    A("| R3 | §1 \"Column definitions, exactly as computed\" gave 背景碰撞 and 出地圖 exactly one definition each "
      "(Table-5 chmed / VL-full), although 6 RareCase cells (SVD_d5, Ours+KDE, real) are computed from T8 on a "
      "different horizon and a different off-road rule — while §1 already listed T8 for 有效率. | The §1 rows for "
      "背景碰撞 and 出地圖 now carry the T8 substrate clause (GT support 2424–2923 → `bg_solid_rate_all_gtsupport`; "
      "33_e7 driving+shoulder union → `offroad_n / n_executed`) and name the rows it applies to, so §1, the "
      "RareCase caption and §5 agree. The same one-substrate claim was also removed from the RareCase table's own "
      "背景碰撞 / 出地圖 footnote and from its `.tex` caption, both of which now state the two substrates. |")
    A("| R4 | `tables3_sixmeasures.tex` lacked n_common S_k and the per-measure counts (the spec requires them) "
      "while the printed `n` column is the CLASS size (50/82/61/20/1/298) that a reader will mistake for the "
      "number of scenes behind the medians. | The `.tex` caption now says the printed $n$ is the class size and "
      "not the number of scenes behind the medians, and lists $n(S_k)$ per block in column order "
      "(PET/min.dist/conflict pt/angle/arr speed/DTW): "
      + "; ".join(b["tex"].replace("$\\to$", "→").replace("\\_", "_") + " "
                  + "/".join(str(b["nper"][m]) for m, _, _ in MEASURES) for b in blocks) + ". |")
    A("")
    A("Non-blocking notes from the verifiers, also applied:")
    A("")
    A("- `tables3_lineage.csv` carried `bold = False` for all 196 six-measure rows although the rendered "
      "`.md`/`.tex` correctly bold 43 of them (the flags in `tables3_sixmeasures.csv` were already right). The "
      "lineage rows of each six-measure block are now back-filled with the block's bold set after it is computed.")
    A("- The lineage `value` column was written with `%.12g`, which stored e.g. 1.1504999999999996 as `1.1505` "
      "(displays as 1.150 but re-formats to 1.151 from the lineage file). It now uses the shortest round-trip "
      "`repr`, so every lineage value re-formats to the printed cell. This is the ONLY difference in the numeric "
      "text of any regenerated file.")
    A("- §2 claimed every `Ours` number comes from the `ours3_disk*` family (469 scenes). The RareCase Ours+KDE "
      "arm `ours3_condkde_a2641` does not match that glob, so §2 now carries the qualifier: same Euclidean disk "
      "L = 10 m θ₁/θ₂/EndSpeed parameterisation, built by `scripts/51_e9_ours_jobs.py` from "
      "`results/disk_window_L10_special_39_180.csv`, and 39_180 is outside the 469-scene cohort by construction.")
    A("- PNG captions were truncated to the first 4 notes, which dropped the PROVISIONAL note and the substrate / "
      "polytrunc disclosure. The PNG footnotes are now prioritised (bold rule → PROVISIONAL → substrate → the "
      "rest) and capped at 6.")
    A("- Detail block **D7** was titled \"window vs polytrunc vs stop-at-end\" but holds only window / analytic / "
      "executed rows; it is retitled \"D7 validity cells: window and analytic counterparts (polytrunc rows are in "
      "D5; WP4 adds stop-at-end)\", and the §5 caveat now points at D5 for the polytrunc rows and the "
      "window-vs-polytrunc teleport comparison.")
    A("")
    if provisional:
        A("Nothing in this round pre-empts WP4: the Ours / Ours+KDE validity cells still come from the WINDOW "
          "executions and every table caption, PNG title and this report keep the PROVISIONAL stamp.")
        A("")
    else:
        A("Superseded by WP4 (§9): the Ours / Ours+KDE validity cells now come from the stop-at-end executions "
          "and the PROVISIONAL stamp is gone.")
        A("")
        write_report_wp4(A, D, amap, built, detail)
    (OUT / "TABLES3_REPORT.md").write_text("\n".join(L) + "\n")


def write_report_wp4(A, D, amap, built, detail):
    """Section 9 — what WP4 changed and every caveat WP2's verifier established."""
    st_path = PROJECT / amap["sources"]["T5_STOP"]["path"]
    st = pd.read_csv(st_path)
    t5 = D["T5"]

    def s_rate(cls, arm, hor, col):
        m = st[(st.cls == cls) & (st.arm == arm) & (st.horizon == hor)]
        return (float(m.iloc[0][col]), int(m.iloc[0]["n"])) if len(m) == 1 else (float("nan"), 0)

    def t5_rate(cls, arm, col):
        m = t5[(t5["class"] == cls) & (t5["arm"] == arm)]
        return (float(m.iloc[0][col]), int(m.iloc[0]["n_samples"])) if len(m) == 1 else (float("nan"), 0)

    A("## 9. WP4 — the final (non-provisional) run")
    A("")
    A("WP1's run was stamped PROVISIONAL because the Ours / Ours+KDE 背景碰撞 and 出地圖 cells came from the "
      "WINDOW executions. WP2 produced the stop-at-end executions; WP4 edited "
      "`results/final/tables3/validity_arm_map.json` (Ours → `ours3_disk_stop`, Ours+KDE → "
      "`ours3_disk_kde_stop_s20260910`, `provisional: false`, plus an `execution_kind` string on every row) and "
      "re-ran `scripts/90_tables3_assemble.py`. **The stamp was removed only because the ours validity really is "
      "stop-at-end now; every row that still cannot use a stop arm says so in its own caption** (the whole "
      "RareCase table, see §2).")
    A("")
    A("### 9.1 Every main-table cell that moved")
    A("")
    A("| Table | Group | Method | Column | Provisional (window) | Final (stop-at-end) | n | Reason |")
    A("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
    moved, same_val_new_n, unchanged = 0, [], []
    for g in GROUPS:
        gk = g["key"]
        if gk == "special_39_180":
            continue
        for mk, old_arm, new_arm in (("ours", "ours3_disk", "ours3_disk_stop"),
                                     ("ours_kde", "ours3_disk_kde", "ours3_disk_kde_stop_s20260910")):
            lab = next(m["label"] for m in METHODS if m["key"] == mk)
            for what, col5, hor, cols in (("背景碰撞", "solid_hit_rate_chmed", "chmed", "any_solid_rate"),
                                          ("出地圖", "offroad_vl_rate_full", "full", "offroad_vl_rate")):
                o, no = t5_rate(gk, old_arm, col5)
                n_, nn = s_rate(gk, new_arm, hor, cols)
                if abs(o - n_) <= TIE:
                    (same_val_new_n if no != nn else unchanged).append(
                        f"{g['label']} / {lab} / {what} = {o:.3f}" + (f" (n {no} → {nn})" if no != nn else ""))
                    continue
                moved += 1
                why = ("stop-at-end instead of window" if mk == "ours" else
                       "stop-at-end instead of window AND seed 20260910 only (n 3000 → 1000 per class)")
                A(f"| {GROUP_BY_KEY[gk]['table']} | {g['label']} | {lab} | {what} | {o:.3f} | **{n_:.3f}** | "
                  f"{no} → {nn} | {why} |")
    A("")
    if same_val_new_n:
        A(f"{len(same_val_new_n)} further cells keep their value but change cohort (same rate, different "
          f"denominator): " + "; ".join(same_val_new_n) + ".")
        A("")
    if unchanged:
        A(f"{len(unchanged)} Ours / Ours+KDE validity cells are numerically identical under both conventions "
          f"— i.e. the stop-at-end swap changed nothing at all for them: " + "; ".join(unchanged) + ". "
          f"All of them are Ours (default) cells: at the chmed horizon the ours default arm's validity is "
          f"essentially convention-invariant (only tlkeep moves, by 3 of 292 samples).")
        A("")
    n_numeric = sum(1 for g in GROUPS for r in built[g["key"]]
                    for c in (COLS_RARE if g["key"] == "special_39_180" else COLS_CLASS)
                    if r["cells"][c["key"]].value is not None)
    A(f"{moved} of the {n_numeric} numeric main-table cells moved. Every one of them is an Ours or Ours+KDE "
      f"背景碰撞 / 出地圖 cell of a CLASS table; no 互動相似, 軌跡相似, 變異比, 涵蓋, 角落涵蓋, RareCase or six-measure cell "
      f"changed, and no SAKURA, SVD or real cell changed.")
    A("")
    A("### 9.2 Separating the stop-at-end effect from the seed restriction (Ours+KDE)")
    A("")
    A("`ours3_disk_kde_stop_s20260910` is seed 20260910 only, so the Ours+KDE validity cells changed denominator "
      "(15 000 → 5 000) as well as convention. WP2 scored the sample-paired single-seed WINDOW arm "
      "`ours3_disk_kde_s20260910_window` precisely so the two can be told apart:")
    A("")
    A("| Group | metric | 3-seed window (15 000) | 1-seed window (5 000) | 1-seed stop-at-end (5 000) = printed |")
    A("| --- | --- | ---: | ---: | ---: |")
    for gk, glab in [("keeptl", "Agent-LT N→E"), ("keeptl_sw", "Agent-LT S→W"), ("cutinl", "Cut-in (left)"),
                     ("cutinr", "Cut-in (right)"), ("tlkeep", "tlkeep"), ("pooled", "pooled")]:
        for what, hor, cols in (("背景碰撞 (chmed solid)", "chmed", "any_solid_rate"),
                                ("出地圖 (full off-road VL)", "full", "offroad_vl_rate")):
            a, _ = s_rate(gk, "ours3_disk_kde", hor, cols)
            b, _ = s_rate(gk, "ours3_disk_kde_s20260910_window", hor, cols)
            c, _ = s_rate(gk, "ours3_disk_kde_stop_s20260910", hor, cols)
            A(f"| {glab} | {what} | {a:.3f} | {b:.3f} | **{c:.3f}** |")
    A("")
    A("Pooled, the seed restriction moves the chmed solid rate by +0.007 (0.687 → 0.694) and the stop-at-end "
      "convention by −0.002 (0.694 → 0.691): most of the change in these cells is the seed, not the stop. The "
      "full triplet table for all four metrics is `tables3_detail` block D7.")
    A("")
    A("### 9.3 Rows that are NOT stop-at-end")
    A("")
    A("WP2 executed exactly four stop arms: `ours3_disk_stop` (469), `ours3_disk_kde_stop_s20260910` (5 000), "
      "`svd_exec_stop_recon_fullfit` (489) and `svd_exec_stop_kde` (1 500). Everything else keeps its original "
      "convention:")
    A("")
    A("- **SVD_d5 / SVD_d5+KDE** — polytrunc, by amendment A3 (the stop-at-end SVD arms exist only as the "
      "consistency check and are in block D7).")
    A("- **SAKURA / SAKURA-route / SAKURA-route+KDE** — window.")
    A("- **real (reference)** — Table 5's replay cohort (511 pooled), not WP2's 489-scene one (487 pooled at "
      "chmed).")
    A("- **The entire RareCase (39_180) table** — window. 39_180 is in none of the four stop cohorts; its Ours "
      "row is the Table-5 window execution of the single scene, its Ours+KDE row the T8 window executions of "
      "`ours3_condkde_a2641`, and its SVD_d5 row an analytic decode on the T8 substrate. The RareCase caption "
      "states this in place of the class tables' stop-at-end note.")
    A("")
    A("### 9.4 Consistency check: stop-at-end vs polytrunc on the SVD arms")
    A("")
    A("| arm | metric | window | polytrunc (printed) | stop-at-end |")
    A("| --- | --- | ---: | ---: | ---: |")
    for arm_w, arm_p, arm_s, lab in [("svd_exec_E3_recon_fullfit", "svd_exec_E3_recon_polytrunc_fullfit",
                                      "svd_exec_stop_recon_fullfit", "SVD_d5 (recon, pooled)"),
                                     ("svd_exec_E3_kde_kde", "svd_exec_E3_kde_polytrunc_kde",
                                      "svd_exec_stop_kde", "SVD_d5+KDE (pooled)")]:
        for what, hor, cols in (("chmed solid hit", "chmed", "any_solid_rate"),
                                ("full off-road VL", "full", "offroad_vl_rate"),
                                ("full teleport", "full", "teleport_rate")):
            A(f"| {lab} | {what} | {s_rate('pooled', arm_w, hor, cols)[0]:.3f} | "
              f"{s_rate('pooled', arm_p, hor, cols)[0]:.3f} | {s_rate('pooled', arm_s, hor, cols)[0]:.3f} |")
    A("")
    A("Teleport agrees to 0.003 between polytrunc and stop-at-end; off-road does not, by construction (the frozen "
      "target keeps contributing off-road rows where polytrunc deletes them). Because the printed SVD number is "
      "the polytrunc one, the SVD rows are quoted at their LOWER off-road value while ours is convention-"
      "invariant — the mixed convention is conservative against our own method.")
    A("")
    A("### 9.5 `.tex` notes")
    A("")
    A("The bilingual footnote list is emitted into every `.tex` file as `% note:` LaTeX comments, which do NOT "
      "render in a PDF; only the `\\caption{}` does. The load-bearing statements are therefore duplicated in the "
      "caption itself: the execution-end convention (stop-at-end for Ours, polytrunc for SVD, window for SAKURA "
      "and the real reference), the Ours+KDE n = 5 000 seed restriction, the conservative direction of the mixed "
      "convention, and — for the RareCase table — the explicit statement that it is *not* stop-at-end. **The "
      "complete note list is in the `.md` file** (and, verbatim but commented out as `% note:` lines just above "
      "`\\end{table*}`, in the `.tex`); anyone lifting a table into the thesis should copy the caption as "
      "generated and take the remaining notes from the `.md`.")
    A("")


if __name__ == "__main__":
    main()
