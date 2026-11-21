"""Regenerate docs/assets/hero.svg from real pipeline output.

Source: samples/landmarks_baggy_15reps.json (a real 15-rep set in loose clothes), run through
analysis.analyze and imu.simulate. Only the skeleton is drawn, never the video frame.

    python docs/assets/make_hero.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import analysis  # noqa: E402
import imu  # noqa: E402

W, H = 1200, 640
BG, CARD, LINE, INK, MUTED = "#0E151C", "#151E27", "#27333F", "#E8EDF2", "#8C99A6"
GREEN, RED, AMBER, BLUE = "#3DBE74", "#F0525A", "#F2B33D", "#7FB2E5"
SANS = "Archivo, 'Helvetica Neue', Arial, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
BONES = [("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
         ("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "foot"), ("ankle", "foot")]
SHORT = {"SHALLOW": "depth", "NO_LOCKOUT": "lockout", "HIP_SAG": "sag", "HIP_PIKE": "pike",
         "HEAD_DROP": "head", "FAST_DESCENT": "fast"}


def main():
    lms = json.loads((ROOT / "samples" / "landmarks_baggy_15reps.json").read_text())
    a = analysis.analyze(lms)
    w = imu.simulate(lms, seed=1)
    s = a["side"][0]
    frames = lms["frames"]
    t = np.array([f["t"] for f in frames])
    reps = a["reps"]
    t0, t1 = reps[0]["start"] - 0.3, reps[-1]["end"] + 0.3

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Form Coach analysing a real 15-rep push-up set: skeleton at the bottom of a rep, a '
         f'scoreboard of all 15 reps with 3 flagged, and elbow, hip and simulated chest-strap traces">',
         '<defs><radialGradient id="g" cx="15%" cy="-10%" r="90%"><stop offset="0" stop-color="#16222E"/>'
         f'<stop offset="0.6" stop-color="{BG}"/></radialGradient></defs>',
         f'<rect width="{W}" height="{H}" rx="20" fill="url(#g)"/>']

    # header
    o.append(f'<circle cx="52" cy="62" r="8" fill="{GREEN}"/><circle cx="74" cy="62" r="8" fill="{RED}"/>')
    o.append(f'<text x="96" y="76" font-family="{SANS}" font-weight="800" font-size="40" fill="{INK}" '
             f'letter-spacing="-1">Form Coach</text>')
    o.append(f'<text x="44" y="112" font-family="{SANS}" font-size="17" fill="{MUTED}">Film a push-up set from '
             f'the side. Every rep gets measured, then you get the two fixes that matter most.</text>')
    o.append(f'<text x="{W - 44}" y="70" text-anchor="end" font-family="{MONO}" font-size="13" fill="{MUTED}">'
             f'null-hack · NullHacks{{Helsinki}}</text>')

    # left card: skeleton (bottom of a rep, ghost at the top)
    cx, cy, cw, ch = 44, 146, 430, 450
    o.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="16" fill="{CARD}" stroke="{LINE}"/>')
    o.append(f'<text x="{cx + 20}" y="{cy + 32}" font-family="{MONO}" font-size="12" fill="{MUTED}" '
             f'letter-spacing="1">REAL SET · LOOSE HOODIE + TROUSERS</text>')
    bot = reps[7]
    top_t = reps[8]["start"] - 0.05
    fb = min(frames, key=lambda f: abs(f["t"] - bot["bottom"]))
    ft = min(frames, key=lambda f: abs(f["t"] - top_t))
    pts = [v[:2] for f in (fb, ft) for k, v in f["lm"].items() if k.startswith(s + "_") or k == "nose"]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    sc = min((cw - 60) / (max(xs) - min(xs)), (ch - 170) / (max(ys) - min(ys)))
    ox = cx + (cw - (max(xs) - min(xs)) * sc) / 2 - min(xs) * sc
    area_top, area_bot = cy + 60, cy + ch - 110
    oy = (area_top + area_bot) / 2 - (max(ys) + min(ys)) / 2 * sc

    def P(v):
        return ox + v[0] * sc, oy + v[1] * sc

    def skel(lm, ghost):
        g = []
        for p, q in BONES:
            (x1, y1), (x2, y2) = P(lm[f"{s}_{p}"]), P(lm[f"{s}_{q}"])
            g.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{"#3A4856" if ghost else GREEN}" stroke-width="{3 if ghost else 6}" '
                     f'stroke-linecap="round"/>')
        if not ghost:
            for j in {j for b in BONES for j in b}:
                x, y = P(lm[f"{s}_{j}"])
                g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{INK}"/>')
        return g

    floor = max(P(fb["lm"][f"{s}_wrist"])[1], P(fb["lm"][f"{s}_foot"])[1]) + 10
    o.append(f'<line x1="{cx + 20}" y1="{floor:.1f}" x2="{cx + cw - 20}" y2="{floor:.1f}" stroke="{LINE}" '
             f'stroke-width="2" stroke-dasharray="2 6"/>')
    o += skel(ft["lm"], True) + skel(fb["lm"], False)
    hx, hy = P(fb["lm"][f"{s}_hip"])
    o.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="11" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    o.append(f'<text x="{hx:.1f}" y="{hy - 20:.1f}" text-anchor="middle" font-family="{MONO}" font-size="12" '
             f'fill="{BLUE}">hip found under loose fabric</text>')
    ex, ey = P(fb["lm"][f"{s}_elbow"])
    o.append(f'<text x="{cx + 20}" y="{cy + ch - 56}" font-family="{SANS}" font-weight="800" font-size="30" '
             f'fill="{INK}">{bot["min_elbow"]:.0f}°</text>')
    o.append(f'<text x="{cx + 20}" y="{cy + ch - 30}" font-family="{MONO}" font-size="12" fill="{MUTED}">'
             f'elbow at the bottom of rep {bot["n"]} · ghost = top of rep</text>')

    # right: scoreboard + traces
    rx, rw = 500, W - 44 - 500
    summ = a["summary"]
    o.append(f'<text x="{rx}" y="{cy + 18}" font-family="{MONO}" font-size="12" fill="{MUTED}" letter-spacing="1">'
             f'EVERY REP · {summ["rep_count"]} COUNTED · {summ["clean_reps"]} CLEAN</text>')
    per = 8
    tw, gap, th_ = (rw - (per - 1) * 8) / per, 8, 46
    for k, r in enumerate(reps):
        x = rx + (k % per) * (tw + gap)
        y = cy + 32 + (k // per) * (th_ + 10)
        bad = bool(r["issues"])
        o.append(f'<rect x="{x:.1f}" y="{y}" width="{tw:.1f}" height="{th_}" rx="9" '
                 f'fill="{"#2A1C22" if bad else CARD}" stroke="{LINE}"/>')
        o.append(f'<rect x="{x + 1:.1f}" y="{y}" width="{tw - 2:.1f}" height="4" rx="2" fill="{RED if bad else GREEN}"/>')
        label = SHORT.get(r["issues"][0]["code"], "") if bad else "✓"
        o.append(f'<text x="{x + tw / 2:.1f}" y="{y + 22}" text-anchor="middle" font-family="{MONO}" '
                 f'font-size="13" fill="{INK}">{r["n"]}</text>')
        o.append(f'<text x="{x + tw / 2:.1f}" y="{y + 38}" text-anchor="middle" font-family="{MONO}" '
                 f'font-size="11" fill="{RED if bad else MUTED}">{label}</text>')

    def X(tt):
        return rx + (tt - t0) / (t1 - t0) * rw

    elbow = np.array([np.nan if v is None else v for v in a["series"]["elbow"]])
    hip = np.array([np.nan if v is None else v for v in a["series"]["hip_dev"]])
    acc = np.linalg.norm(np.array(w["sensors"]["chest"]["acc_world"]), axis=1)
    lanes = [("ELBOW °", elbow, (30, 185), GREEN, [90, 160]),
             ("HIPS OFF YOUR LINE °", hip, (-30, 30), BLUE, [-15, 15]),
             ("CHEST STRAP · SIMULATED m/s²", acc, (7, 13), AMBER, [9.81])]
    y = cy + 162
    lane_h = 78
    m = (t >= t0) & (t <= t1)
    for r in reps:
        if r["issues"]:
            o.append(f'<rect x="{X(r["start"]):.1f}" y="{y - 14}" width="{X(r["end"]) - X(r["start"]):.1f}" '
                     f'height="{3 * lane_h + 2 * 24 + 8}" fill="{RED}" opacity="0.14" rx="4"/>')
    for label, series, (lo, hi), col, refs in lanes:
        o.append(f'<text x="{rx}" y="{y - 2}" font-family="{MONO}" font-size="11" fill="{MUTED}" '
                 f'letter-spacing="1">{label}</text>')

        def Y(v, lo=lo, hi=hi, y=y):
            return y + lane_h - (np.clip(v, lo, hi) - lo) / (hi - lo) * (lane_h - 12)

        for v in refs:
            o.append(f'<line x1="{rx}" y1="{Y(v):.1f}" x2="{rx + rw}" y2="{Y(v):.1f}" stroke="#3A4856" '
                     f'stroke-dasharray="4 5"/>')
        pts = " ".join(f"{X(tt):.1f},{Y(v):.1f}" for tt, v in zip(t[m][::2], series[m][::2]) if not np.isnan(v))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        y += lane_h + 24
    o.append("</svg>")
    out = Path(__file__).with_name("hero.svg")
    out.write_text("\n".join(o))
    print("wrote", out)


if __name__ == "__main__":
    main()
