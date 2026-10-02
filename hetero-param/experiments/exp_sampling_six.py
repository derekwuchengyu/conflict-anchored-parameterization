"""Ablation: point placement (uniform vs interaction anchors) scored with the
CURRENT six-measure set (exp_ours3_svd5/scripts/35_six_measure_table.py):
  dtw, pet, min_dist, conflict_angle(alpha), conflict_point(cpoint), u_c.
Fixed 4-point budget; real ego in both pairs.
"""
import sys, csv, math
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/hcis-s19/Documents/ChengYu/hetero-param")
from hetero_param import paths, config as C
from hetero_param.similarity import core, interaction_sim as IS, path_sim as PS
from experiments.exp_point import load_scenarios

METHODS = ["uniform", "traj_cross", "pet", "min_dist"]
FIELDS = ["dataset", "ego", "actor", "min_frame", "max_frame", "label", "agent_class",
          "method", "n_shape", "dtw", "d_pet", "d_min_dist", "d_alpha", "d_cpoint", "d_uc",
          "real_pet_finite", "param_pet_finite"]

def wrap180(d):
    d = abs(float(d)) % 360.0
    return d if d <= 180.0 else 360.0 - d

def alpha_at_closest(a, e):
    """unsigned heading difference at the closest-approach frame (deg)."""
    al = IS._align(a, e)
    if al is None:
        return float("nan")
    grid, ax, ay, ex, ey = al
    i = int(np.argmin(np.hypot(ax - ex, ay - ey)))
    fr = grid[i]
    ha = np.interp(fr, a.frame, np.unwrap(np.radians(a.heading)))
    he = np.interp(fr, e.frame, np.unwrap(np.radians(e.heading)))
    return wrap180(np.degrees(ha - he))

def run(dataset, per_class, out_csv):
    scen = load_scenarios(dataset, None, None, per_class=per_class, group_key="agent_class")
    methods = C.anchor_methods(4)
    print(f"[six] {dataset}: {len(scen)} scenarios x {len(METHODS)} methods", flush=True)
    rows = []
    for n, s in enumerate(scen):
        ds, eg, ac, f0, f1 = s["dataset"], s["ego"], s["actor"], s["min_frame"], s["max_frame"]
        try:
            a = core.real_traj(ds, ac, f0, f1)
            e = core.real_traj(ds, eg, f0, f1)
            rd = IS.descriptors(a, e, "bbox") if (a is not None and e is not None) else None
            ra = alpha_at_closest(a, e) if rd else float("nan")
        except Exception:
            a = e = rd = None
        for m in METHODS:
            row = {**s, "method": m}
            try:
                p = core.param_traj(ds, eg, ac, f0, f1, methods[m])
                if p is None or rd is None:
                    raise ValueError("no traj")
                pdd = IS.descriptors(p, e, "bbox")
                pa = alpha_at_closest(p, e)
                row["n_shape"] = int(p.meta.get("n_shape", 0)) if isinstance(p.meta, dict) else None
                row["dtw"] = PS.dtw(a.xy, p.xy, 5)
                row["d_pet"] = IS._finite_delta(rd["pet_abs"], pdd["pet_abs"])
                row["d_min_dist"] = IS._finite_delta(rd["min_dist"], pdd["min_dist"])
                row["d_alpha"] = abs(ra - pa) if np.isfinite(ra) and np.isfinite(pa) else float("nan")
                row["d_cpoint"] = float(np.hypot(rd["conflict_x"] - pdd["conflict_x"],
                                                 rd["conflict_y"] - pdd["conflict_y"]))
                row["d_uc"] = IS._finite_delta(rd["agent_arr_speed"], pdd["agent_arr_speed"])
                row["real_pet_finite"] = bool(np.isfinite(rd["pet"]))
                row["param_pet_finite"] = bool(np.isfinite(pdd["pet"]))
            except Exception as ex:
                for k in ("n_shape", "dtw", "d_pet", "d_min_dist", "d_alpha", "d_cpoint", "d_uc"):
                    row.setdefault(k, float("nan"))
                row["real_pet_finite"] = row["param_pet_finite"] = False
            rows.append(row)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(scen)}", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print(f"[six] wrote {out_csv} ({len(rows)} rows)", flush=True)

if __name__ == "__main__":
    ds = sys.argv[1]
    run(ds, int(sys.argv[2]), sys.argv[3])
