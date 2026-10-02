"""Extract a project-local subset (data/{name}/) + its real GT interaction
descriptors (results/real_desc_{name}.parquet) so 80/90/95 can run on classes
sr-tlkeep never covered.

Modes
  --label N --name X          all scenarios of label N (direction-free)
  --uturn --name uturn        geometric U-turn agents: |net heading| >= 150°
                              over moving samples, any label > 0 (77/88 excl.)
  --single EGO_ACTOR --name X one scenario key from the labels json

Conventions = sr-tlkeep 20_run_label stage_extract / stage_descriptors:
window trim >=10 rows, dur > 0.5 s, real agent scored with derived
kinematics, ego as recorded, SWC.descriptors(estimator="bbox").
Usage: micromamba run -n nps python 110_extract.py --label 8 --name cutinl
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib as L  # noqa: E402
from hetero_param.sweep import core as SWC  # noqa: E402
from hetero_param.similarity.core import Traj  # noqa: E402

FPS = 30.0
HDATA = Path("/home/hcis-s19/Documents/ChengYu/HetroD-labeler/data")

G_REAL = {}


def extract(name: str, label: int | None, uturn: bool, single: str | None):
    labels = json.loads((HDATA / "00_labeled_scenarios.json").read_text())
    tracks = pd.read_parquet(HDATA / "00_tracks.parquet")
    tracks = tracks.sort_values(["trackId", "frame"]).set_index("trackId",
                                                                drop=False)
    tracks.index.name = "tid"
    rows_out, meta_rows = [], []
    for key, ent in labels.items():
        li = ent.get("label_idx")
        if single is not None:
            if key != single:
                continue
        elif uturn:
            if not li or li in (77, 88):
                continue
        elif li != label:
            continue
        ego, actor = int(ent["ego_id"]), int(ent["actor_id"])
        mf, xf = int(ent["min_frame"]), int(ent["max_frame"])
        try:
            a = tracks.loc[[actor]]
            e = tracks.loc[[ego]]
        except KeyError:
            continue
        a = a[(a.frame >= mf) & (a.frame <= xf)]
        e = e[(e.frame >= mf) & (e.frame <= xf)]
        if len(a) < 10 or len(e) < 10:
            continue
        dur = float((a.frame.max() - a.frame.min()) / FPS)
        if dur <= 0.5:
            continue
        if uturn:
            sp = np.hypot(a.xVelocity.values, a.yVelocity.values)
            m = sp > 0.5
            if m.sum() < 10:
                continue
            hu = np.unwrap(np.radians(a.heading.values[m]))
            if abs(np.degrees(hu[-1] - hu[0])) < 150:
                continue
        meta_rows.append({"scenario_id": key, "ego": ego, "actor": actor,
                          "min_frame": mf, "max_frame": xf,
                          "label": li, "duration_s": dur})
        for role, df in (("ego", e), ("actor", a)):
            sp = np.hypot(df.xVelocity.values, df.yVelocity.values)
            rows_out.append(pd.DataFrame({
                "scenario_id": key, "role": role,
                "track_id": df.trackId.values, "frame": df.frame.values,
                "x": df.xCenter.values, "y": df.yCenter.values,
                "heading_deg": df.heading.values, "speed": sp,
                "length": df.length.values, "width": df.width.values}))
    ddir = L.EXP / "data" / name
    ddir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows_out, ignore_index=True).to_parquet(ddir / "real_tracks.parquet")
    pd.DataFrame(meta_rows).to_csv(ddir / "real_meta.csv", index=False)
    print(f"[extract] {name}: kept {len(meta_rows)} scenarios")
    return ddir


def _traj_df(g, Lw, W):
    t = g.frame.values.astype(float) / FPS
    sp = g.speed.values.astype(float)
    acc = np.gradient(sp, t) if len(t) > 2 else np.zeros_like(sp)
    return Traj(frame=g.frame.values.astype(float), x=g.x.values.astype(float),
                y=g.y.values.astype(float),
                heading=g.heading_deg.values.astype(float), speed=sp, accel=acc,
                fps=FPS, length=float(Lw), width=float(W), meta={})


def _desc_one(sid):
    try:
        a = G_REAL[(sid, "actor")]
        e = G_REAL[(sid, "ego")]
        agent = SWC._derived_kinematics_copy(
            _traj_df(a, a.length.iloc[0], a.width.iloc[0]))
        ego = _traj_df(e, e.length.iloc[0], e.width.iloc[0])
        d = SWC.descriptors(agent, ego, estimator="bbox")
        d.update({"set": "real", "scenario_id": sid, "tag": "real"})
        return d
    except Exception as e:  # noqa: BLE001
        return {"set": "real", "scenario_id": sid, "tag": "real",
                "error": f"{type(e).__name__}: {e}"}


def real_descriptors(name: str, ddir: Path):
    global G_REAL
    real_tracks = pd.read_parquet(ddir / "real_tracks.parquet")
    for (sid, role), g in real_tracks.groupby(["scenario_id", "role"]):
        G_REAL[(sid, role)] = g.sort_values("frame")
    sids = [sid for (sid, role) in G_REAL if role == "actor"]
    with Pool(8) as pool:
        out = [r for r in pool.imap_unordered(_desc_one, sids, chunksize=8)
               if r]
    df = pd.DataFrame(out)
    nerr = int(df.error.notna().sum()) if "error" in df else 0
    df.to_parquet(L.EXP / "results" / f"real_desc_{name}.parquet")
    print(f"[real-desc] {name}: {len(df)} rows ({nerr} errors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", type=int, default=None)
    ap.add_argument("--uturn", action="store_true")
    ap.add_argument("--single", default=None)
    a = ap.parse_args()
    d = extract(a.name, a.label, a.uturn, a.single)
    real_descriptors(a.name, d)
