"""Minimal OpenDRIVE lane-geometry extractor for tyms.xodr.

Supports line / arc / spiral planView geometries (spiral via numeric
integration), piecewise laneOffset, and piecewise lane width polynomials.
Coordinates are the same world frame as the HetroD tracks parquet.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


def _poly_pieces(elems, s_attr):
    """[(s0, (a,b,c,d)), ...] sorted by s0."""
    out = []
    for e in elems:
        out.append((float(e.get(s_attr)), tuple(float(e.get(k)) for k in "abcd")))
    return sorted(out, key=lambda t: t[0])


def _eval_pieces(pieces, s):
    """Evaluate piecewise cubic poly at array s (piece-local ds)."""
    s = np.asarray(s)
    val = np.zeros_like(s, dtype=float)
    for i, (s0, (a, b, c, d)) in enumerate(pieces):
        s1 = pieces[i + 1][0] if i + 1 < len(pieces) else np.inf
        m = (s >= s0) & (s < s1) if np.isfinite(s1) else (s >= s0)
        ds = s[m] - s0
        val[m] = a + b * ds + c * ds ** 2 + d * ds ** 3
    return val


@dataclass
class Lane:
    lane_id: int
    lane_type: str
    width_pieces: list
    travel_dir: str  # forward / backward / undirected / ''


@dataclass
class Road:
    road_id: int
    name: str
    length: float
    junction: int
    lane_offset_pieces: list
    left_lanes: list = field(default_factory=list)   # id ascending 1,2,...
    right_lanes: list = field(default_factory=list)  # id descending -1,-2,...
    geoms: list = field(default_factory=list)
    predecessor: tuple = None  # (elementType, elementId)
    successor: tuple = None

    def sample_ref(self, ds=0.25):
        """Reference line -> (s, x, y, hdg) arrays."""
        S, X, Y, H = [], [], [], []
        for g in self.geoms:
            s0, x0, y0, h0, glen, gtype, prm = g
            n = max(int(np.ceil(glen / ds)), 2)
            sl = np.linspace(0, glen, n, endpoint=False)
            if gtype == "line":
                x = x0 + sl * np.cos(h0)
                y = y0 + sl * np.sin(h0)
                h = np.full_like(sl, h0)
            elif gtype == "arc":
                k = prm["curvature"]
                h = h0 + k * sl
                x = x0 + (np.sin(h) - np.sin(h0)) / k
                y = y0 - (np.cos(h) - np.cos(h0)) / k
            elif gtype == "spiral":
                k0, k1 = prm["curvStart"], prm["curvEnd"]
                fine = np.linspace(0, glen, max(int(glen / 0.05), 10))
                kf = k0 + (k1 - k0) * fine / glen
                hf = h0 + np.concatenate([[0], np.cumsum(
                    (kf[1:] + kf[:-1]) / 2 * np.diff(fine))])
                xf = x0 + np.concatenate([[0], np.cumsum(
                    (np.cos(hf[1:]) + np.cos(hf[:-1])) / 2 * np.diff(fine))])
                yf = y0 + np.concatenate([[0], np.cumsum(
                    (np.sin(hf[1:]) + np.sin(hf[:-1])) / 2 * np.diff(fine))])
                x = np.interp(sl, fine, xf)
                y = np.interp(sl, fine, yf)
                h = np.interp(sl, fine, hf)
            else:
                raise ValueError(f"unsupported geometry {gtype}")
            S.append(s0 + sl); X.append(x); Y.append(y); H.append(h)
        S, X, Y, H = (np.concatenate(a) for a in (S, X, Y, H))
        # append exact road end
        g = self.geoms[-1]
        S = np.append(S, self.length)
        X = np.append(X, X[-1] + (S[-1] - S[-2]) * np.cos(H[-1]))
        Y = np.append(Y, Y[-1] + (S[-1] - S[-2]) * np.sin(H[-1]))
        H = np.append(H, H[-1])
        return S, X, Y, H

    def lane_edges(self, ds=0.25, types=("driving",)):
        """Per-lane inner/outer edge polylines.

        Returns {lane_id: (inner_xy, outer_xy)} for lanes whose type is in
        `types`; cumulative widths measured from the laneOffset-shifted center.
        """
        S, X, Y, H = self.sample_ref(ds)
        nx, ny = -np.sin(H), np.cos(H)  # +t (left) normal
        off = _eval_pieces(self.lane_offset_pieces, S) if self.lane_offset_pieces else 0
        cx, cy = X + off * nx, Y + off * ny
        edges = {}
        for side, sign in ((self.left_lanes, +1), (self.right_lanes, -1)):
            cum = np.zeros_like(S)
            for lane in side:
                w = _eval_pieces(lane.width_pieces, S)
                inner, outer = cum.copy(), cum + w
                if lane.lane_type in types:
                    edges[lane.lane_id] = (
                        np.column_stack([cx + sign * inner * nx, cy + sign * inner * ny]),
                        np.column_stack([cx + sign * outer * nx, cy + sign * outer * ny]),
                    )
                cum = cum + w
        return edges

    def lane_polygon(self, lane_id, ds=0.25):
        """Closed polygon (Nx2) of one lane."""
        inner, outer = self.lane_edges(ds, types=("driving", "shoulder", "sidewalk",
                                                  "biking", "border", "none"))[lane_id]
        return np.vstack([inner, outer[::-1]])

    def travel_heading(self, lane_id, ds=0.25):
        """(start_heading_deg, end_heading_deg, net_turn_deg) along the lane's
        travel direction (right lanes: +s; left lanes: -s)."""
        _, _, _, H = self.sample_ref(ds)
        h = np.rad2deg(H)
        if lane_id > 0:
            h = (h[::-1] + 180)
        h = np.unwrap(h, period=360)
        return h[0] % 360, h[-1] % 360, h[-1] - h[0]


def parse(path):
    root = ET.parse(path).getroot()
    roads = {}
    for r in root.findall("road"):
        road = Road(
            road_id=int(r.get("id")), name=r.get("name", ""),
            length=float(r.get("length")), junction=int(r.get("junction")),
            lane_offset_pieces=_poly_pieces(r.findall("lanes/laneOffset"), "s"),
        )
        link = r.find("link")
        if link is not None:
            for tag in ("predecessor", "successor"):
                e = link.find(tag)
                if e is not None:
                    setattr(road, tag, (e.get("elementType"), int(e.get("elementId"))))
        for g in r.findall("planView/geometry"):
            base = (float(g.get("s")), float(g.get("x")), float(g.get("y")),
                    float(g.get("hdg")), float(g.get("length")))
            if g.find("line") is not None:
                road.geoms.append(base + ("line", {}))
            elif g.find("arc") is not None:
                road.geoms.append(base + ("arc", {"curvature": float(g.find("arc").get("curvature"))}))
            elif g.find("spiral") is not None:
                sp = g.find("spiral")
                road.geoms.append(base + ("spiral", {"curvStart": float(sp.get("curvStart")),
                                                     "curvEnd": float(sp.get("curvEnd"))}))
            else:
                raise ValueError(f"road {road.road_id}: unknown geometry")
        sec = r.find("lanes/laneSection")  # tyms.xodr: single section per road
        for side_tag, target in (("left", road.left_lanes), ("right", road.right_lanes)):
            side = sec.find(side_tag)
            if side is None:
                continue
            for ln in side.findall("lane"):
                vl = ln.find("userData/vectorLane")
                target.append(Lane(
                    lane_id=int(ln.get("id")), lane_type=ln.get("type"),
                    width_pieces=_poly_pieces(ln.findall("width"), "sOffset"),
                    travel_dir=vl.get("travelDir") if vl is not None else "",
                ))
        road.left_lanes.sort(key=lambda l: l.lane_id)          # 1, 2, ...
        road.right_lanes.sort(key=lambda l: -l.lane_id)        # -1, -2, ...
        roads[road.road_id] = road

    junctions = {}
    for j in root.findall("junction"):
        conns = []
        for c in j.findall("connection"):
            conns.append(dict(
                incoming=int(c.get("incomingRoad")),
                connecting=int(c.get("connectingRoad")),
                contact=c.get("contactPoint"),
                lane_links=[(int(l.get("from")), int(l.get("to")))
                            for l in c.findall("laneLink")],
            ))
        junctions[int(j.get("id"))] = conns
    return roads, junctions
