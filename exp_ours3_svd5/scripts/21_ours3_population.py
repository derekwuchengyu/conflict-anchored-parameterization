#!/usr/bin/env python3
"""Freeze the documented existing geometry sources; extract ours3 defaults.

Read-only upstream operation. No simulator, scoring, source mutation, or new
control-point construction. All 513 source cases remain in the eligibility table.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT / "results"
VELOCITY = ROOT / "exp_coverage_velocity/scripts"
CP_LABELS = {"tlkeep": "leftturn", "keeptl": "keeplt", "cutinr": "cutin"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def existing_functions(path, names, namespace):
    parsed = ast.parse(path.read_text())
    body = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != set(names):
        raise ValueError(f"Missing requested existing function in {path}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    cases_path = OUT / "svd_d5_cases.csv"
    cases = pd.read_csv(cases_path)
    cases = cases[cases["mode"] == "fullfit"]
    assert cases.scenario_uid.is_unique
    ns = {"np": np, "NT": 50}
    existing_functions(VELOCITY / "cvlib.py", ["arc_resample"], ns)
    library = SimpleNamespace(arc_resample=ns["arc_resample"])
    ns["L"] = library
    existing_functions(VELOCITY / "cvrender.py",
                       ["_signed_angle", "_rot", "theta_geometry", "theta_to_cps"], ns)
    source_paths = {cases_path, Path(__file__), VELOCITY / "cvlib.py", VELOCITY / "cvrender.py"}
    rows, contexts = [], []
    for subset, subset_cases in cases.groupby("subset", sort=False):
        track_path = Path(subset_cases.source_tracks.iloc[0])
        source_paths.add(track_path)
        tracks = pd.read_parquet(track_path)
        by_actor = {str(s): g.sort_values("frame") for s, g in
                    tracks[tracks.role == "actor"].groupby("scenario_id")}
        library.real_actor_track = lambda _t, sid: by_actor[str(sid)]
        cp_groups, template_table = {}, None
        if subset in CP_LABELS:
            geometry_source = ROOT / f"exp_cp3_points/results/cp3pm10_points_{CP_LABELS[subset]}.csv"
            source_paths.add(geometry_source)
            cp_table = pd.read_csv(geometry_source)
            cp_groups = {str(s): g.sort_values("s_from_crit") for s, g in cp_table.groupby("scenario_id")}
            source_variant = "velocity_pet_cp3pm10"
        else:
            geometry_source = ROOT / f"exp_cov_sampling/results/templates_{subset}.csv"
            source_paths.add(geometry_source)
            template_table = pd.read_csv(geometry_source)
            source_variant = "existing_regular_xx_cp3d10" if subset == "cutinl" else "existing_legacy_cp3d10"
        for case in subset_cases.itertuples(index=False):
            row = dict(subset=subset, scenario_id=case.scenario_id,
                       scenario_uid=case.scenario_uid, case_index=case.case_index,
                       ego=case.ego, actor=case.actor,
                       min_frame=case.metadata_min_frame, max_frame=case.metadata_max_frame,
                       source_tracks=str(track_path), geometry_source=str(geometry_source),
                       geometry_variant=source_variant, status="not_checked", detail="")
            try:
                if subset in CP_LABELS:
                    base_dir = ROOT / "hetero-param/results/esmini/xosc_base/srexp-petq3"
                    matches = sorted(base_dir.glob(f"*_{case.ego}_{case.actor}_f{case.metadata_min_frame + 1}.xosc"))
                    if len(matches) != 1:
                        raise ValueError(f"base_count={len(matches)}; no fallback source selected")
                    base = matches[0]
                    cp = cp_groups[str(case.scenario_id)]
                    if list(cp.kind) != ["m10", "crit", "p10"]:
                        raise ValueError("Expected existing m10/crit/p10 triple")
                    xy = cp[["x", "y"]].to_numpy(float)
                else:
                    match = template_table[(template_table.scenario_id == case.scenario_id)
                                           & (template_table.min_frame == case.metadata_min_frame)
                                           & (template_table.max_frame == case.metadata_max_frame)]
                    if len(match) != 1:
                        raise ValueError(f"template_rows={len(match)}; no fallback source selected")
                    base = Path(match.base.iloc[0])
                    tree = ET.parse(base)
                    weighted = [c for c in tree.getroot().iter("ControlPoint") if c.get("weight") is not None]
                    if len(weighted) != 3:
                        raise ValueError(f"Existing weighted CP count={len(weighted)}, expected 3")
                    xy = np.asarray([[float(c.find("Position/WorldPosition").get(k)) for k in ("x", "y")]
                                     for c in weighted])
                source_paths.add(base)
                root = ET.parse(base).getroot()
                bp = {p.get("name"): p.get("value") for p in root.iter("ParameterDeclaration")}
                geo = ns["theta_geometry"](None, case.scenario_id, xy.ravel())
                if geo is None:
                    raise ValueError("Existing exact geometry rejects L1 or L2 below 1 m")
                np.testing.assert_allclose(ns["theta_to_cps"](geo, geo["theta1"], geo["theta2"]),
                                           xy.ravel(), rtol=0, atol=1e-7)
                g = by_actor[str(case.scenario_id)]
                distance = np.linalg.norm(g[["x", "y"]].to_numpy() - xy[1], axis=1)
                minimum = float(distance.min())
                matches = np.flatnonzero(distance <= minimum + 1e-9)
                if minimum > 0.001:
                    raise ValueError(f"Central CP is {minimum:.6g} m from nearest GT sample; no substitute anchor")
                if len(matches) != 1:
                    raise ValueError(f"Central CP matches {len(matches)} times; anchor time ambiguous")
                anchor = g.iloc[int(matches[0])]
                end_speed = float(anchor.speed) * 3.6
                if not np.isfinite(end_speed) or end_speed < 0:
                    raise ValueError("Invalid recorded anchor speed")
                # These values are recorded, never sampled or overridden.
                frozen = {k: float(bp[k]) for k in ["Agent1_Offset", "Agent1_1_SA_DynamicDuration", "Agent1_Speed"]}
                row.update(status="eligible", base=str(base), base_sha256=digest(base),
                           theta1_deg=geo["theta1"], theta2_deg=geo["theta2"],
                           end_speed_kmh=end_speed, anchor_frame=int(anchor.frame),
                           anchor_match_distance_m=minimum,
                           fixed_offset_m=frozen["Agent1_Offset"],
                           fixed_ramp_duration_s=frozen["Agent1_1_SA_DynamicDuration"],
                           fixed_start_speed_kmh=frozen["Agent1_Speed"],
                           old_base_end_speed_kmh=float(bp["Agent1_1_SA_EndSpeed"]),
                           actual_actor_min_frame=int(g.frame.min()), actual_actor_max_frame=int(g.frame.max()))
                contexts.append(dict(row, cps6=xy.ravel(), geometry=geo,
                                     params=[geo["theta1"], geo["theta2"], end_speed],
                                     fixed_declarations=frozen))
            except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
                row.update(status="unavailable", detail=f"{type(error).__name__}: {error}")
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ours3_population.csv", index=False)
    (OUT / "ours3_contexts.json").write_text(json.dumps(contexts, indent=2, default=serial, allow_nan=False) + "\n")
    summary = table.groupby(["subset", "status"]).size().unstack(fill_value=0)
    summary.to_csv(OUT / "ours3_population_summary.csv")
    sources = [{"path": str(p), "sha256": digest(p)} for p in sorted(source_paths)]
    (OUT / "ours3_population_manifest.json").write_text(json.dumps({
        "scope": "source selection and parameter extraction; not executed fidelity results",
        "variable_parameters": ["theta1_deg", "theta2_deg", "end_speed_kmh"],
        "n_source_cases": len(table), "n_eligible": len(contexts),
        "anchor_rule": "unique recorded point matching the existing central CP within 0.001 m",
        "selection": "fixed existing source map, no per-result source fallback or regeneration",
        "special_39_180": "separate; regular class uses its existing xx geometry, never replaces it with special PET configuration",
        "no_simulation_or_metrics_run": True, "sources": sources
    }, indent=2, allow_nan=False) + "\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
