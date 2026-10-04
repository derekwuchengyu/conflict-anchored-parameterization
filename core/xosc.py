"""Write conflict-anchored parameters into a logical OpenSCENARIO file.

The base file is a logical scenario whose Agent1 follows one Nurbs trajectory
(order 4). The weighted shape control points are replaced by the three points
[q_in, q_c, q_out] (weight 5), and the end speed is written to the parameter
Agent1_1_SA_EndSpeed [km/h]. All other declarations stay unchanged.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

SPEED_PARAMETER = "Agent1_1_SA_EndSpeed"
KNOTS_7CP = [0, 0, 0, 0, 0.5, 1.0, 1.5, 2, 2, 2, 2]
CP_WEIGHT = 5.0


def read_parameters(path) -> dict:
    return {p.get("name"): p.get("value") for p in ET.parse(path).getroot().iter("ParameterDeclaration")}


def patch_cps(base, cps6, out):
    """Replace the weighted shape CPs of the Agent1 Nurbs by 3 WorldPositions
    (weight 5) with order-4 knots for 7 control points."""
    tree = ET.parse(base)
    nurbs = list(tree.getroot().iter("Nurbs"))
    if len(nurbs) != 1:
        raise RuntimeError(f"{base}: expected 1 Nurbs, found {len(nurbs)}")
    nb = nurbs[0]
    children = list(nb)
    cp_elems = [c for c in children if c.tag == "ControlPoint"]
    weighted = [c for c in cp_elems if c.get("weight") is not None]
    if weighted:
        insert_at = children.index(weighted[0])
        for c in weighted:
            nb.remove(c)
    else:
        insert_at = children.index(cp_elems[-1])
    for c in [c for c in list(nb) if c.tag == "Knot"]:
        nb.remove(c)
    xy = np.asarray(cps6, float).reshape(3, 2)
    for i in range(3):
        el = ET.Element("ControlPoint", {"weight": f"{CP_WEIGHT:.1f}"})
        pos = ET.SubElement(el, "Position")
        ET.SubElement(pos, "WorldPosition", {"x": f"{xy[i, 0]:.4f}", "y": f"{xy[i, 1]:.4f}"})
        nb.insert(insert_at + i, el)
    n_cp = len([c for c in list(nb) if c.tag == "ControlPoint"])
    if len(KNOTS_7CP) != n_cp + int(nb.get("order", "4")):
        raise RuntimeError(f"{base}: knot mismatch for {n_cp} control points")
    for k in KNOTS_7CP:
        ET.SubElement(nb, "Knot", {"value": str(k)})
    tree.write(out, encoding="unicode", xml_declaration=True)


def set_middle_cp_weight(path, weight):
    """Override the weight of the conflict control point q_c."""
    tree = ET.parse(path)
    cp = [c for c in tree.getroot().findall(".//Nurbs/ControlPoint") if c.get("weight") is not None]
    if len(cp) != 3:
        raise RuntimeError(f"{path}: expected 3 weighted control points")
    cp[1].set("weight", f"{weight:g}")
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def set_parameters(path, values: dict):
    """Set ParameterDeclaration values in place (floats written with repr)."""
    tree = ET.parse(path)
    found = set()
    for p in tree.getroot().iter("ParameterDeclaration"):
        if p.get("name") in values:
            p.set("value", repr(float(values[p.get("name")])))
            found.add(p.get("name"))
    if found != set(values):
        raise KeyError(f"{path}: parameters not declared: {sorted(set(values) - found)}")
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def write_scenario(base, out, cps6, end_speed_kmh, middle_cp_weight=None):
    """Logical xosc for one parameter vector."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    patch_cps(base, cps6, out)
    if middle_cp_weight is not None and float(middle_cp_weight) != CP_WEIGHT:
        set_middle_cp_weight(out, float(middle_cp_weight))
    set_parameters(out, {SPEED_PARAMETER: end_speed_kmh})
    return out
