#!/usr/bin/env python3
"""Compare L=5/10 with each subset's fixed L=10-selected spatial/temporal pair."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
EXP = HERE.parents[1]
ROOT = EXP.parent
DS = EXP / "results/disk_sweep"
spec = spec_from_file_location("disk_aggregate", EXP / "scripts/19_disk_trajdtw_aggregate.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
F = mod.load_180_functions()
PAIRS = {"cutinl": "pp", "keeptl": "pm", "keeptl_sw": "pp"}
rows, sources = [], []
for subset, pair in PAIRS.items():
    real_path = mod.data_dir(subset) / "real_tracks.parquet"
    tracks = pd.read_parquet(real_path)
    sources.append(real_path)
    status_path = DS / f"disk_sweep_status_{subset}.csv"
    status = pd.read_csv(status_path)
    sources.append(status_path)
    a = status[(status.arm == f"anch{pair}_s5") & (status.status == "ok")]
    b = status[(status.arm == f"anch{pair}") & (status.status == "ok")]
    common = a.merge(b, on="scenario", suffixes=("_5", "_10"))
    assert len(common) and (common.anchor_frame_5 == common.anchor_frame_10).all()
    assert np.allclose(common.chord1_5, 5) and np.allclose(common.chord2_5, 5)
    gt = {sid: g.sort_values("frame") for (sid, role), g in tracks.groupby(["scenario_id", "role"])
          if role == "actor"}
    arms = {}
    for L, folder in [(5, DS / "_anchor_s5"), (10, DS)]:
        d = folder / f"anch{pair}_descriptors_{subset}_disk_g2.parquet"
        t = folder / f"anch{pair}_trajectories_{subset}_disk.parquet"
        assert d.exists() and t.exists(), (d, t)
        sources.extend([d, t])
        arms[f"L{L}"] = F["build_arm"](d, t, gt)
    piv = F["aggregate"](subset, arms)
    assert np.isfinite(piv.loc[["L5", "L10"], "composite"]).all(), (subset, piv)
    for arm in ("L5", "L10"):
        r = piv.loc[arm]
        rows.append(dict(subset=subset, pair=pair, L_m=int(arm[1:]),
                         **{k: float(r[k]) for k in mod.KEYS},
                         composite=float(r.composite), cov_med=float(r.cov_med),
                         n_common_pet=piv.attrs["n"]["pet"],
                         n_common_dtw=piv.attrs["n"]["traj_dtw"]))
out = DS / "pair_radius_L5_L10_disk.csv"
pd.DataFrame(rows).to_csv(out, index=False)
manifest = dict(rule="fixed L=10-selected pair per subset; L=5 vs L=10 on per-key common scenarios; "
                     "six-key v5 aggregate normalized over these two arms only",
                scorer=str(mod.SRC_180),
                scorer_sha256=hashlib.sha256(mod.SRC_180.read_bytes()).hexdigest(),
                inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(DS / "pair_radius_L5_L10_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(pd.DataFrame(rows)[["subset", "pair", "L_m", "composite", "n_common_pet", "n_common_dtw"]].to_string(index=False))
