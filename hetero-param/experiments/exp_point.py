"""
Experiment 1 — critical POINT selection (spatial).

Sweep the sample-point strategy for each scenario and measure geometric fidelity of the
parameterized NURBS challenge path vs the real actor. The path is evaluated analytically
(parampath) — the path esmini would follow — so this needs neither CARLA nor esmini.
Headline metric = path_dev (alignment-free, speed-independent).

Usage (from hetero-param/):
  micromamba run -n nps python experiments/exp_point.py --dataset HetroD \
      --labels 3,4,5 --n-scenarios 30 --out results/exp_point.csv
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hetero_param import paths, parampath as PP, config as C, superclass as SC  # noqa: E402
paths.add_import_paths()
import parameterization_fidelity as PF                                          # noqa: E402

FIELDS = ["dataset", "ego", "actor", "min_frame", "max_frame", "label",
          "scenario_type", "agent_class", "superclass", "strategy", "n_shape",
          "path_dev", "ade", "dtw", "owd", "n_real",
          "param_min_dist", "gt_min_dist", "delta_min_dist", "reason"]

DEFAULT_STRATEGIES = ["0pt", "1pt-midpoint", "1pt-conflict", "1pt-interaction",
                      "3pt-uniform", "3pt-critwindow", "5pt-default"]


def load_scenarios(dataset: str, labels=None, limit=None, per_class: int | None = None,
                   group_key: str = "scenario_type"):
    """Load labeled scenarios, tagging each with scenario_type (EGO+AGENT, keeps CUTOUT/TL/
    CUTIN distinct) and agent_class (AGENT-only). per_class caps the number kept PER group_key
    so every category is represented in the balanced breakdown."""
    data_id = paths.dataset_of(dataset)["data_id"]
    lab = PF.load_labeled_scenarios(data_id)
    rows = []
    for (ego, actor, minf), meta in lab.items():
        if labels and meta["label"] not in labels:
            continue
        rows.append(dict(dataset=dataset, ego=ego, actor=actor, min_frame=minf,
                         max_frame=meta["max_frame"], label=meta["label"],
                         scenario_type=SC.scenario_type(meta["label"]),
                         agent_class=SC.agent_behavior(meta["label"])))
    rows.sort(key=lambda r: (r[group_key], r["ego"], r["actor"], r["min_frame"]))
    if per_class:
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            c = r[group_key]
            if seen.get(c, 0) >= per_class:
                continue
            seen[c] = seen.get(c, 0) + 1
            kept.append(r)
        rows = kept
    return rows[:limit] if limit else rows


def _breakdown(rows, strategies, group_key="scenario_type", metric="path_dev"):
    """Print median(metric) as a category x strategy table + the best strategy per category."""
    classes = sorted({r[group_key] for r in rows})
    print(f"\n  === median {metric} by {group_key} x strategy ===")
    print("    " + group_key.ljust(20) + "N   " + "".join(s[:11].ljust(12) for s in strategies) + "best")
    for c in classes:
        cells, best, bestv = [], None, float("inf")
        for s in strategies:
            vals = [r[metric] for r in rows if r[group_key] == c and r["strategy"] == s
                    and not math.isnan(r[metric])]
            if vals:
                m = median(vals)
                cells.append(f"{m:.3f}".ljust(12))
                if m < bestv:
                    bestv, best = m, s
            else:
                cells.append("n/a".ljust(12))
        n = len({(r["ego"], r["actor"], r["min_frame"]) for r in rows if r[group_key] == c})
        print("    " + c.ljust(20) + f"{n:<4d}" + "".join(cells) + (best or ""))


def run(dataset, labels, n_scen, strategies, out_csv: Path, per_class=None, group_key="scenario_type"):
    scen = load_scenarios(dataset, labels, n_scen, per_class=per_class, group_key=group_key)
    print(f"[exp_point] {len(scen)} scenarios x {len(strategies)} strategies "
          f"(dataset={dataset}, labels={labels}, per_class={per_class}, group_by={group_key})")
    rows = []
    for i, s in enumerate(scen):
        sc = SC.superclass_of(s["label"])
        for name in strategies:
            cfg = C.SAMPLE_STRATEGIES[name]
            try:
                m = PP.full_fidelity(dataset, s["ego"], s["actor"],
                                     s["min_frame"], s["max_frame"], cfg)
            except Exception as e:
                m = {"n_shape": 0, "path_dev": float("nan"), "ade": float("nan"),
                     "dtw": float("nan"), "owd": float("nan"), "n_real": 0,
                     "param_min_dist": float("nan"), "gt_min_dist": float("nan"),
                     "delta_min_dist": float("nan"), "reason": f"{type(e).__name__}: {str(e)[:120]}"}
            rows.append({**s, "superclass": sc, "strategy": name, **m})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})

    print(f"[exp_point] wrote {out_csv}  ({len(rows)} rows)")
    print("  median path_dev by strategy (all agent classes pooled):")
    for name in strategies:
        vals = [r["path_dev"] for r in rows
                if r["strategy"] == name and not math.isnan(r["path_dev"])]
        med = f"{median(vals):.3f}" if vals else "n/a"
        print(f"    {name:16s} n={len(vals):3d}  median path_dev={med}")
    _breakdown(rows, strategies, group_key=group_key, metric="path_dev")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HetroD")
    ap.add_argument("--labels", default="3,4,5",
                    help="label indices (3/4/5 = TW SIDEPASS/PARALLEL/UNDERPASS); empty = all")
    ap.add_argument("--n-scenarios", type=int, default=None)
    ap.add_argument("--per-class", type=int, default=None,
                    help="cap scenarios per category (balanced breakdown)")
    ap.add_argument("--group-by", default="agent_class", choices=["agent_class", "scenario_type"],
                    help="agent_class = Agent1 (_02) behavior only (default); scenario_type = ego+agent")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    ap.add_argument("--out", default="results/exp_point.csv")
    a = ap.parse_args()
    labels = [int(x) for x in a.labels.split(",") if x.strip()] if a.labels.strip() else None
    strategies = [s for s in a.strategies.split(",") if s.strip()]
    out = Path(a.out)
    if not out.is_absolute():
        out = paths.THIS_PROJECT / out
    run(a.dataset, labels, a.n_scenarios, strategies, out, per_class=a.per_class, group_key=a.group_by)


if __name__ == "__main__":
    main()
