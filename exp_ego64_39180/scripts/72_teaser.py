"""Stage 72 — the teaser: recorded interaction -> parameterized samples.

  left    the recording on the drone footage (stage 73, --clean)
  arrow   what happens in between, named above and below it
  right   a 2x2 grid (stage 71, --clean): the reconstruction -- the three
          control points at their fitted angles, executed in esmini -- in a
          green frame, then three samples in red frames because they collide

Captions carry the parameters, so nothing is burned into the panels.  The
samples are written as offsets from the reconstruction's own angles, which is
what the parameterization varies.

Run:  micromamba run -n nps python scripts/72_teaser.py --scale 1.45 --out teaser_print.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib64 as L  # noqa: E402

DJ = "/usr/share/fonts/truetype/dejavu"
OUT = L.EXP / "figs_teaser"

MARG, GAP = 40, 28
BG, INK, ACC = (255, 255, 255), (18, 20, 24), (206, 38, 26)
RULE, NAVY, GREEN = (206, 210, 216), (38, 54, 86), (24, 150, 62)
RECON_SLOT = 99                      # stage 71 --tag default --slots 99


def conflict_at_contact(tag: str) -> float:
    """Angle between the two headings at first contact (or closest approach)."""
    df = L.read_csv(L.RUNS / "search" / f"{tag}.csv")
    df = L.crop_clean(df[df.t <= L.MAX_T].reset_index(drop=True))
    gap = L.bbox_gap(df)
    k = int(np.argmax(gap <= 0.0)) if (gap <= 0.0).any() else int(np.argmin(gap))
    r = df.iloc[k]
    rel = np.degrees(r.agent_h - r.ego_h) % 360
    return 360 - rel if rel > 180 else rel


def rich(d, xy, parts, font) -> None:
    x, y = xy
    for txt, col in parts:
        d.text((x, y), txt, fill=col, font=font)
        x += d.textlength(txt, font=font)


def caption(slot: int, r, cfl: float, i: int) -> tuple[str, list]:
    """(title, lines); a line is a list of (text, colour) runs."""
    vend = [("v", INK), ("end", INK), (" = ", INK),
            (f"{r.vend:.1f} km/h", INK if slot == RECON_SLOT else ACC)]
    if slot == RECON_SLOT:
        return "Reconstruction", [
            [("θ₁ = ", INK), (f"{r.h1:.2f}°", INK)],
            [("θ₂ = ", INK),
             (f"{'−' if r.defl < 0 else ''}{abs(r.defl):.2f}°", INK)],
            vend]
    sgn = lambda v: ("− " if v < 0 else "+ ") + f"{abs(v):.2f}°"
    return f"Test scenario {i}", [
        [("θ₁ = θ₁ ", INK), (sgn(r.theta1), ACC)],
        [("θ₂ = θ₂ ", INK), (sgn(r.theta2), ACC)],
        vend]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="2,4,14", help="the sample panels")
    ap.add_argument("--real", default="real_video_contact_teaser.png")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="caption text scale; 1.45 makes it ~8 pt at \\textwidth")
    ap.add_argument("--out", default="teaser.png")
    a = ap.parse_args()
    sc = a.scale
    cells = [RECON_SLOT] + [int(s) for s in a.slots.split(",")]
    ang = pd.read_csv(L.RESULTS / "cp_angles.csv").set_index("slot")

    f_ttl = ImageFont.truetype(f"{DJ}/DejaVuSerif-Italic.ttf", int(38 * sc))
    f_val = ImageFont.truetype(f"{DJ}/DejaVuSans-Bold.ttf", int(34 * sc))
    f_sub = ImageFont.truetype(f"{DJ}/DejaVuSans-Bold.ttf", int(23 * sc))
    LN, TG = int(47 * sc), int(16 * sc)

    # The arrow labels are the largest type in the figure, but letting them set
    # the column width pushes the panels down to ~1.5 in.  Fix the column and
    # grow the type to fill it instead.
    ARROW_LINES = (("Conflict-anchored", -132), ("parameterization", -78),
                   ("Sample and scenario", 26), ("generation", 80))
    ARROW_W = int(1030 * sc / 1.6)
    _m = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    fs_arw = int(90 * sc / 1.6)
    while fs_arw > 12:
        f_arw = ImageFont.truetype(f"{DJ}/DejaVuSans-Bold.ttf", fs_arw)
        if max(_m.textlength(t, font=f_arw) for t, _ in ARROW_LINES) <= ARROW_W - 70:
            break
        fs_arw -= 1

    real = Image.open(OUT / a.real).convert("RGB")
    pans = [Image.open(OUT / f"sim_slot{s:02d}_clean.png").convert("RGB")
            for s in cells]
    PW, PH = pans[0].size
    cap_h = int(38 * sc) + TG + 3 * LN
    cell_h = PH + int(20 * sc) + cap_h
    grid_w = 2 * PW + GAP
    grid_h = 2 * cell_h + GAP

    Wc = MARG * 2 + real.width + ARROW_W + grid_w
    Hc = MARG * 2 + max(real.height, grid_h)
    cv = Image.new("RGB", (Wc, Hc), BG)
    d = ImageDraw.Draw(cv)

    ry = MARG + (Hc - 2 * MARG - real.height) // 2
    cv.paste(real, (MARG, ry))
    d.rectangle([MARG - 1, ry - 1, MARG + real.width, ry + real.height],
                outline=RULE, width=2)

    # the arrow, with the two things it stands for named above and below it
    ax0 = MARG + real.width + int(30 * sc)
    ax1 = MARG + real.width + ARROW_W - int(30 * sc)
    ay = Hc // 2
    hd = int(34 * sc)
    d.line([(ax0, ay), (ax1 - hd * 0.6, ay)], fill=NAVY, width=int(10 * sc))
    d.polygon([(ax1, ay), (ax1 - hd, ay - hd * 0.62),
               (ax1 - hd, ay + hd * 0.62)], fill=NAVY)
    for txt, dy in ARROW_LINES:
        w = d.textlength(txt, font=f_arw)
        d.text(((ax0 + ax1) / 2 - w / 2, ay + int(dy * sc)), txt, fill=NAVY,
               font=f_arw)

    gx = MARG + real.width + ARROW_W
    gy = MARG + (Hc - 2 * MARG - grid_h) // 2
    for i, slot in enumerate(cells):
        x = gx + (i % 2) * (PW + GAP)
        y = gy + (i // 2) * (cell_h + GAP)
        cv.paste(pans[i], (x, y))
        if slot == RECON_SLOT:                      # green frame: the recon
            d.rectangle([x + 2, y + 2, x + PW - 3, y + PH - 3], outline=GREEN,
                        width=6)
        d.rectangle([x - 1, y - 1, x + PW, y + PH], outline=RULE, width=2)

        r = ang.loc[slot]
        cfl = conflict_at_contact(r.tag)     # reported, not drawn any more
        print(f"  {r.tag:8s} slot {slot:2d}: conflict {cfl:.0f} deg")
        ttl, lines = caption(slot, r, cfl, i)
        cy = y + PH + int(20 * sc)
        d.text((x + 4, cy), ttl, fill=INK, font=f_ttl)
        w = d.textlength(ttl, font=f_ttl)
        d.line([(x + 4, cy + int(50 * sc)), (x + 4 + w, cy + int(50 * sc))],
               fill=(150, 220, 175) if slot == RECON_SLOT else (240, 160, 150),
               width=max(4, int(4 * sc)))
        for j, parts in enumerate(lines):
            ly = cy + int(38 * sc) + TG + j * LN
            if parts[0][0] == "v":                  # v_end with a subscript
                d.text((x + 4, ly), "v", fill=INK, font=f_val)
                vw = d.textlength("v", font=f_val)
                d.text((x + 4 + vw, ly + int(16 * sc)), "end", fill=INK,
                       font=f_sub)
                sw = d.textlength("end", font=f_sub)
                rich(d, (x + 4 + vw + sw, ly), parts[2:], f_val)
            else:
                rich(d, (x + 4, ly), parts, f_val)

    p = OUT / a.out
    cv.save(p)
    pt = 72 * 7.17 / Wc
    print(f"wrote {p}  {Wc}x{Hc}   at \\textwidth: caption "
          f"{f_val.size * pt:.1f} pt, title {f_ttl.size * pt:.1f} pt, arrow "
          f"{f_arw.size * pt:.1f} pt, panel {PW * pt / 72:.2f} in wide")


if __name__ == "__main__":
    main()
