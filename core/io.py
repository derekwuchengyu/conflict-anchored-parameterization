"""Load recorded tracks (levelXdata format, e.g. HetroD / inD) and scenario lists."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["trackId", "frame", "xCenter", "yCenter", "heading",
                 "xVelocity", "yVelocity", "length", "width"]
SCENARIO_COLUMNS = ["scenario_id", "class", "ego", "target", "min_frame", "max_frame"]


def load_tracks(path, track_ids=None) -> pd.DataFrame:
    """Read a tracks file (.parquet or .csv). Heading is in degrees.

    Returns columns trackId, frame, x, y, heading, speed [m/s], length, width.
    """
    path = Path(path)
    ids = None if track_ids is None else sorted({int(t) for t in track_ids})
    if path.suffix == ".parquet":
        filters = None if ids is None else [("trackId", "in", ids)]
        df = pd.read_parquet(path, columns=TRACK_COLUMNS, filters=filters)
    else:
        df = pd.read_csv(path, usecols=TRACK_COLUMNS)
        if ids is not None:
            df = df[df.trackId.isin(ids)]
    df = df.rename(columns={"xCenter": "x", "yCenter": "y"})
    df["speed"] = np.hypot(df.xVelocity, df.yVelocity)
    return df.drop(columns=["xVelocity", "yVelocity"]).sort_values(["trackId", "frame"]).reset_index(drop=True)


def window(tracks: pd.DataFrame, track_id: int, min_frame: int, max_frame: int) -> pd.DataFrame:
    """Samples of one track inside [min_frame, max_frame], sorted by frame."""
    g = tracks[(tracks.trackId == int(track_id)) & tracks.frame.between(min_frame, max_frame)]
    return g.drop_duplicates("frame").sort_values("frame").reset_index(drop=True)


def load_scenarios(path) -> pd.DataFrame:
    """Scenario list: one ego-target interaction per row.

    Required columns: scenario_id, class, ego, target, min_frame, max_frame.
    Optional columns: base_xosc (logical scenario to patch) and anchor_frame
    (a target frame used as the conflict anchor instead of the configured method).
    """
    df = pd.read_csv(path, dtype={"scenario_id": str})
    missing = set(SCENARIO_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for c in ("ego", "target", "min_frame", "max_frame"):
        df[c] = df[c].astype(int)
    return df
