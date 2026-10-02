"""
Geometric fidelity of a generated scenario vs the real trajectory.

Wraps parameterization_fidelity.py (which already has correct per-dataset fps) to run
esmini on a .xosc, extract the challenge agent (Agent1) path, and compare it to the
real actor's recorded path. Headline metric = symmetric path_dev (alignment-free,
speed-independent) so Experiment 1/2 measure route/shape fidelity, not speed error.
"""
from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np

from . import paths
paths.add_import_paths()
import parameterization_fidelity as PF  # noqa: E402


def symmetric_path_dev(a: np.ndarray, b: np.ndarray) -> float:
    """max(OWD(a->b), OWD(b->a)) — symmetric mean nearest-point deviation (metres)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(max(PF.compute_owd(a, b), PF.compute_owd(b, a)))


def _nan_result(reason: str, esmini_ok: bool) -> dict:
    return {"esmini_ok": esmini_ok, "reason": reason,
            "path_dev": float("nan"), "ade": float("nan"), "fde": float("nan"),
            "dtw": float("nan"), "owd": float("nan"), "n_repl": 0, "n_orig": 0}


def real_actor_path(dataset: str, actor_id: int, min_frame: int, max_frame: int) -> np.ndarray:
    """The real actor's recorded (x, y) path over [min_frame, max_frame]."""
    data_id = paths.dataset_of(dataset)["data_id"]
    tracks = PF.load_dataset_tracks(data_id)
    try:
        odf = tracks.xs(int(actor_id), level="trackId").reset_index()
    except KeyError:
        return np.empty((0, 2))
    odf = odf[(odf["frame"] >= min_frame) & (odf["frame"] <= max_frame)]
    odf = odf.sort_values("frame").drop_duplicates("frame")
    return odf[["xCenter", "yCenter"]].values


def run_esmini_to_edf(xosc_path, fps: float, csv_out: Path | None = None):
    """Run esmini once; return the parsed csv_logger DataFrame (or None on failure)."""
    tmp = None
    if csv_out is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        csv_out = Path(tmp.name)
        tmp.close()
    ok = PF.run_esmini(Path(xosc_path), Path(csv_out), fixed_timestep=1.0 / fps)
    if not ok:
        return None
    return PF.parse_esmini_csv(str(csv_out))


def metrics_from_edf(edf, dataset: str, actor_id: int, min_frame: int, max_frame: int) -> dict:
    """Geometric fidelity of Agent1 (from a parsed esmini df) vs the real actor."""
    if edf is None:
        return _nan_result("esmini failed", esmini_ok=False)
    a1 = edf[edf["entity"] == "Agent1"].drop_duplicates("timestamp").sort_values("timestamp")
    if a1.empty:
        return _nan_result("Agent1 not in esmini CSV", esmini_ok=True)
    repl = a1[["x", "y"]].values

    orig = real_actor_path(dataset, actor_id, min_frame, max_frame)
    if len(orig) == 0:
        return _nan_result(f"actor {actor_id} not in dataset range", esmini_ok=True)

    ade, fde = PF.compute_ade_fde(orig, repl)
    return {
        "esmini_ok": True, "reason": "",
        "path_dev": symmetric_path_dev(orig, repl),
        "ade": ade, "fde": fde,
        "dtw": PF.compute_dtw(orig, repl),
        "owd": PF.compute_owd(orig, repl),
        "n_repl": int(len(repl)), "n_orig": int(len(orig)),
    }


def evaluate(xosc_path, dataset: str, actor_id: int, min_frame: int, max_frame: int,
             csv_out: Path | None = None) -> dict:
    """Run esmini on the .xosc and compute geometric fidelity of Agent1 vs the real actor."""
    fps = paths.dataset_of(dataset)["fps"]
    edf = run_esmini_to_edf(xosc_path, fps, csv_out)
    return metrics_from_edf(edf, dataset, actor_id, min_frame, max_frame)
