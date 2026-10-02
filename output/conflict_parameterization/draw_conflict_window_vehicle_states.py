"""Generate conflict-window vehicle-state diagrams.

Usage examples:
  python draw_conflict_window_vehicle_states.py
  python draw_conflict_window_vehicle_states.py --cutin
  python draw_conflict_window_vehicle_states.py --uturn
  python draw_conflict_window_vehicle_states.py --target-only
"""
from math import cos, sin, radians
from pathlib import Path
import subprocess
import sys

ARGS = sys.argv[1:]
CUTIN = "--cutin" in ARGS
UTURN = "--uturn" in ARGS
TARGET_ONLY = "--target-only" in ARGS

HERE = Path(__file__).resolve().parent
if TARGET_ONLY:
    NAME = "conflict_window_vehicle_states_target_only"
elif UTURN:
    NAME = "conflict_window_vehicle_states_uturn"
elif CUTIN:
    NAME = "conflict_window_vehicle_states_cutin"
else:
    NAME = "conflict_window_vehicle_states"
SVG = HERE / f"{NAME}.svg"
PNG = HERE / f"{NAME}.png"

L = 3.0
psi0, theta1, theta2 = 0.0, 25.0, 20.0
q0, qm = (0.0, 0.0), (155 / 90, 0.0)

if CUTIN:
    psi0, theta1, theta2 = 0.0, 30.0, -30.0
elif UTURN:
    psi0, theta1, theta2 = 0.0, 30.0, -50.0
    q0 = (0.0, 0.0)
    qm = (1.8, 0.0)

if not UTURN:
    qc = (qm[0] + L * cos(radians(psi0 + theta1)),
          qm[1] + L * sin(radians(psi0 + theta1)))
    qp = (qc[0] + L * cos(radians(psi0 + theta1 + theta2)),
          qc[1] + L * sin(radians(psi0 + theta1 + theta2)))
    qe = (qp[0] + 1.3 * cos(radians(psi0 + theta1 + theta2)),
          qp[1] + 1.3 * sin(radians(psi0 + theta1 + theta2)))
else:
    qc = (qm[0] + L * cos(radians(psi0 + theta1)),
          qm[1] + L * sin(radians(psi0 + theta1)))
    qp = (qc[0] + L * cos(radians(psi0 + theta1 + theta2)),
          qc[1] + L * sin(radians(psi0 + theta1 + theta2)))
    qe = (-2.4, 4.8)


def xy(p):
    return f"{p[0]:.3f},{p[1]:.3f}"


def step(p, radius, deg):
    return p[0] + radius * cos(radians(deg)), p[1] - radius * sin(radians(deg))


def line(a, b, style):
    return f'<path d="M {xy(a)} L {xy(b)}" {style}/>'


def canvas(p):
    return 135 + 90 * p[0], 540 - 90 * p[1]


if UTURN:
    q0 = (0.0, 0.0)
    qm = (1.8, 0.0)
    qc = (3.0, 1.8)
    qp = (4.2, 4.4)
    qe = (-1.2, 6.8)
    psi0 = 0.0
    theta1 = 30.0
    theta2 = -50.0

p0, pm, pc, pp, pe = map(canvas, (q0, qm, qc, qp, qe))
end_heading = 0 if CUTIN else 60
if UTURN:
    end_heading = 180
if not CUTIN and not UTURN:
    pe = (825, 110)


def math_label(x, y, char, sub="", anchor="middle", color="#28313a", size=30):
    sub_element = (f'<tspan font-size="{size * 0.65:g}" baseline-shift="sub" '
                   f'font-style="normal" font-weight="normal">{sub}</tspan>') if sub else ""
    weight = "bold" if char in ("q", "h") else "normal"
    vector_arrow = (f'<path d="M {x-12:g},{y-size-2:g} H {x+8:g} m -5,-3 l 5,3 -5,3" '
                    f'fill="none" stroke="{color}" stroke-width="1.5"/>') if char == "h" else ""
    return (vector_arrow + f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
            f'font-size="{size:g}" fill="{color}" font-style="italic" '
            f'font-weight="{weight}" paint-order="stroke" stroke="white" stroke-width="4">{char}{sub_element}</text>')


def point_label(sub, x, y, anchor="start", color="#414852"):
    return math_label(x, y, "q", sub, anchor=anchor, color=color)


def turn_arc(center, radius, start, end):
    direction = 1 if end > start else -1
    a = step(center, radius, start + direction * 4.0)
    display_end = start + .55 * (end - start)
    b = step(center, radius, display_end)
    return (f'<path d="M {xy(a)} A {radius:g},{radius:g} 0 0 {int(end < start)} {xy(b)}" '
            f'fill="none" stroke="#e67e22" stroke-width="4.2" '
            f'marker-end="url(#angle-arrow)"/>')


def car(center, heading, opacity, symbol="target-car"):
    return (f'<use href="#{symbol}" xlink:href="#{symbol}" '
            f'transform="translate({xy(center)}) rotate({90-heading:g})" '
            f'opacity="{opacity:g}"/>')


parts = ['''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1100" height="810" viewBox="0 0 1100 810">
<title>One target vehicle at its initial, conflict-window entry, and end states</title>
<desc>A continuous target path passes through the labeled parameterization points. Dashed rays show the two signed angle references, and one radius marks the conflict window. Pale road lines show the scenario topology.</desc>
<defs>
  <marker id="segment-arrow" markerWidth="9" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse">
    <path d="M 0,0 L 9,4 L 0,8 z" fill="#313842"/>
  </marker>
  <marker id="route-arrow" markerWidth="15" markerHeight="13" refX="14" refY="6.5" orient="auto" markerUnits="userSpaceOnUse">
    <path d="M 0,0 L 15,6.5 L 0,13 z" fill="#35424a" stroke="white" stroke-width="1.2"/>
  </marker>
  <marker id="angle-arrow" markerWidth="12" markerHeight="11" refX="11" refY="5.5" orient="auto" markerUnits="userSpaceOnUse">
    <path d="M 0,0 L 12,5.5 L 0,11 z" fill="#e67e22"/>
  </marker>
  <marker id="ego-arrow" markerWidth="20" markerHeight="18" refX="19" refY="9" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0,0 L 20,9 L 0,18 z" fill="#c5ae62"/></marker>
  <g id="target-car" stroke-linejoin="round">
    <path d="M -12,-44 Q -20,-42 -21,-30 L -21,30 Q -20,43 -12,44 L 12,44 Q 20,43 21,30 L 21,-30 Q 20,-42 12,-44 Z" fill="#23689a" stroke="#164969" stroke-width="1.7"/>
    <path d="M -17,-22 Q 0,-29 17,-22 L 13,-8 Q 0,-12 -13,-8 Z" fill="white"/>
    <path d="M -13,17 Q 0,20 13,17 L 17,30 Q 0,36 -17,30 Z" fill="white"/>
    <path d="M -18,-17 L -15,-6 L -15,15 L -18,24 Z M 18,-17 L 15,-6 L 15,15 L 18,24 Z" fill="white"/>
    <path d="M -20,-15 L -25,-11 L -25,-7 L -20,-8 M 20,-15 L 25,-11 L 25,-7 L 20,-8" fill="#23689a"/>
    <path d="M -15,-36 L -11,-39 M 11,-39 L 15,-36" stroke="white" stroke-width="3" stroke-linecap="round"/>
    <path d="M -15,37 L -10,39 M 10,39 L 15,37" stroke="white" stroke-width="2"/>
  </g>
</defs>
<rect width="1100" height="810" fill="white"/>
<g font-family="TeX Gyre Pagella, Palatino Linotype, Book Antiqua, serif" stroke-linecap="round" stroke-linejoin="round">
''']

if not TARGET_ONLY:
    ego_glyph = parts[0].split('<g id="target-car"', 1)[1].split('</g>', 1)[0]
    ego_glyph = '<g id="ego-car"' + ego_glyph.replace('#23689a', '#edcc65').replace('#164969', '#bd9d3f') + '</g>'
    parts[0] = parts[0].replace('</defs>', ego_glyph + '\n</defs>')

if not TARGET_ONLY:
    if CUTIN:
        for y in (337.5, 607.5):
            parts.append(line((55, y), (1045, y), 'stroke="#d8dee2" stroke-width="2"'))
        parts.append(line((55, 472.5), (1045, 472.5), 'stroke="#e0e4e7" stroke-width="2" stroke-dasharray="23 17"'))
    elif UTURN:
        for x in (250, 610):
            parts.append(line((x, 120), (x, 740), 'stroke="#dce2e5" stroke-width="2"'))
        parts.append(line((200, 260), (1000, 260), 'stroke="#dce2e5" stroke-width="2" stroke-dasharray="16 14"'))
    else:
        for y in (340, 620):
            parts.append(line((55, y), (1045, y), 'stroke="#dce2e5" stroke-width="2"'))
        for a, b in (((520, 340), (785, 75)), ((720, 340), (985, 75))):
            parts.append(line(a, b, 'stroke="#dce2e5" stroke-width="2"'))

if not TARGET_ONLY:
    parts.append(f'<circle cx="{pc[0]:.3f}" cy="{pc[1]:.3f}" r="{90 * L:g}" fill="#eaf3f6" fill-opacity="0.65" stroke="#8cadba" stroke-width="1.8" stroke-dasharray="8 6"/>')
    parts.append(f'<text x="{pc[0]:.3f}" y="{pc[1]-287:.3f}" text-anchor="middle" font-size="24" fill="#567c8d">Conflict window</text>')
    parts.append(f'<text x="55" y="60" font-size="25" fill="#28313a">{"(b) Cut-in" if CUTIN else "(c) U-turn" if UTURN else "(a) Left turn"}</text>')
    ego_x = 155 if CUTIN else 920
    ego_style = 'fill="none" stroke="#d3b85c" stroke-width="5.5" opacity="0.55"'
    ego_start = (ego_x + (44 if CUTIN else -44), pc[1])
    ego_end = (1030 if CUTIN else 70, pc[1])
    parts.append(line(ego_start, ego_end, ego_style + ' marker-end="url(#ego-arrow)"'))
    parts.append(f'<text x="{ego_x}" y="{pc[1]-44:.3f}" text-anchor="middle" font-size="20" fill="#a58a37">Ego direction</text>')
    parts.append(car((ego_x, pc[1]), 0 if CUTIN else 180, .48, symbol="ego-car"))

if TARGET_ONLY:
    route = [f'M {xy(p0)}']
    route.extend(f'L {xy(p)}' for p in (pm, pc, pp, pe))
    route_style = 'fill="none" stroke="#35424a" stroke-width="5.5"'
    parts.append(f'<path d="{" ".join(route)}" {route_style}/>' )
else:
    points = (p0, pm, pc, pp, pe)
    tangents = ((pm[0]-p0[0], 0), (pm[0]-p0[0], 0),
                tuple(.55*(pp[k]-pm[k]) for k in (0, 1)),
                tuple((.5 if CUTIN else .4)*(pe[k]-pc[k]) for k in (0, 1)),
                tuple(pe[k]-pp[k] for k in (0, 1)))
    route = [f'M {xy(p0)}']
    for i in range(len(points)-1):
        a, b = points[i:i+2]
        c1 = tuple(a[k]+tangents[i][k]/3 for k in (0, 1))
        c2 = tuple(b[k]-tangents[i+1][k]/3 for k in (0, 1))
        route.append(f'C {xy(c1)} {xy(c2)} {xy(b)}')
    parts.append(f'<path d="{" ".join(route)}" fill="none" stroke="#35424a" stroke-width="5"/>')
    end_dx, end_dy = pe[0]-pp[0], pe[1]-pp[1]
    end_len = (end_dx**2 + end_dy**2)**0.5
    end_unit = (end_dx/end_len, end_dy/end_len)
    arrow_start = (pe[0]-95*end_unit[0], pe[1]-95*end_unit[1])
    arrow_end = (pe[0]-52*end_unit[0], pe[1]-52*end_unit[1])
    parts.append(line(arrow_start, arrow_end,
                      'fill="none" stroke="#35424a" stroke-width="5" marker-end="url(#route-arrow)"'))

for origin, angle in ((pm, psi0), (pc, psi0 + theta1)):
    parts.append(line(origin, step(origin, 190, angle),
                      'fill="none" stroke="#e67e22" stroke-width="4" stroke-dasharray="14 8" opacity="0.9"'))

angle_radius = 135
# Keep the turn-angle arrows visible even in the target-only variant.
for idx, (center, start, end) in enumerate(((pm, psi0, psi0 + theta1),
                                            (pc, psi0 + theta1, psi0 + theta1 + theta2)), 1):
    parts.append(turn_arc(center, angle_radius, start, end))
    mid = (start + end) / 2.0
    x, y = step(center, 198, mid)
    parts.append(math_label(x, y + 10, "θ", str(idx), color="#e67e22", size=40))

if TARGET_ONLY:
    # Place L beside the straight segments rather than on the radius.
    mid1 = ((pm[0] + pc[0]) / 2.0, (pm[1] + pc[1]) / 2.0)
    mid2 = ((pc[0] + pp[0]) / 2.0, (pc[1] + pp[1]) / 2.0)
    l_pos = ((mid1[0] + mid2[0]) / 2.0 + 18, (mid1[1] + mid2[1]) / 2.0 - 12)
    parts.append(math_label(l_pos[0], l_pos[1], "L", size=29, color="#527586"))
else:
    mid1 = ((pm[0] + pc[0]) / 2.0, (pm[1] + pc[1]) / 2.0)
    mid2 = ((pc[0] + pp[0]) / 2.0, (pc[1] + pp[1]) / 2.0)
    l_pos = ((mid1[0] + mid2[0]) / 2.0 + 18, (mid1[1] + mid2[1]) / 2.0 - 12)
    parts.append(math_label(l_pos[0], l_pos[1], "L", size=29, color="#527586"))

parts.append(car(p0, psi0, .43))
parts.append(car(pe, end_heading, .43))
parts.append(car(pm, psi0, 1.0))
for p, color, radius in ((pc, "#d99a16", 11), (pp, "#248664", 10)):
    parts.append(f'<circle cx="{p[0]:.3f}" cy="{p[1]:.3f}" r="{radius}" fill="{color}" stroke="white" stroke-width="2"/>')
for p, color in ((p0, "#536779"), (pm, "#248664"), (pe, "#536779")):
    radius = 9 if p == pm else 6
    parts.append(f'<circle cx="{p[0]:.3f}" cy="{p[1]:.3f}" r="{radius}" fill="{color}" stroke="white" stroke-width="2.5"/>')

labels = [("0", p0[0], p0[1] + 73, "middle"), ("in", pm[0], pm[1] + 73, "middle")]
if CUTIN:
    labels += [("c", pc[0], pc[1] - 52, "middle"), ("out", pp[0], pp[1] - 55, "middle"), ("end", pe[0] + 10, pe[1] + 80, "middle")]
elif UTURN:
    labels += [("c", pc[0] - 28, pc[1] - 54, "middle"), ("out", pp[0] + 28, pp[1] - 58, "middle"), ("end", pe[0] - 30, pe[1] + 72, "middle")]
else:
    labels += [("c", pc[0] + 40, pc[1] + 51, "start"), ("out", pp[0] + 46, pp[1] + 10, "start"), ("end", pe[0] + 55, pe[1] + 4, "start")]
for sub, x, y, anchor in labels:
    parts.append(point_label(sub, x, y, anchor, "#254f73" if sub in ("in", "out") else "#414852"))

parts.append('</g></svg>\n')

SVG.write_text("\n".join(parts), encoding="utf-8")
subprocess.run(["rsvg-convert", "--width", "3300", "--height", "2430", "--output", str(PNG), str(SVG)], check=True)
print(SVG)
print(PNG)
