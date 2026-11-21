"""Regenerate docs/assets/hero.svg from real pipeline output (synth + imu).

    python docs/assets/make_hero.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import imu  # noqa: E402
import synth  # noqa: E402

W, H = 1200, 560
SLATE, GRID, CHALK, MUTED = "#1F2933", "#2A3742", "#EEF1F4", "#8A9AA8"
GREEN, RED, AMBER, BLUE = "#5AD25A", "#EB2828", "#F2B33D", "#7FB2E5"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
SERIF = "Georgia, 'Times New Roman', serif"


def pick_sample():
    for seed in range(200):
        s = synth.generate(n=1, seed=seed, faults=["HIP_SAG"], camera="ideal")[0]
        reps = s["labels"]["reps"]
        if 5 <= len(reps) <= 6 and sum("HIP_SAG" in r["faults"] for r in reps) >= 2:
            return s
    raise RuntimeError("no suitable sample")


def main():
    s = pick_sample()
    lms, reps = s["landmarks"], s["labels"]["reps"]
    side = s["labels"]["params"]["near_side"][0]
    frames = lms["frames"]
    t = np.array([f["t"] for f in frames])

    def ang(lm, a, b, c):
        pa, pb, pc = (np.array(lm[f"{side}_{j}"][:2]) for j in (a, b, c))
        v1, v2 = pa - pb, pc - pb
        return float(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))

    elbow = np.array([ang(f["lm"], "shoulder", "elbow", "wrist") for f in frames])
    hip = np.array([180 - ang(f["lm"], "shoulder", "hip", "ankle") for f in frames])
    sensor = imu.simulate(lms, noise=True, seed=1)
    acc = np.array(sensor["sensors"]["chest"]["acc_world"])[:, 1]

    sag = next(r for r in reps if "HIP_SAG" in r["faults"])
    i_bot = int(np.argmin(np.abs(t - sag["bottom"])))
    i_top = int(np.argmin(np.abs(t - max(0.1, reps[0]["start"] - 0.2))))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'role="img" aria-label="Simulated athlete at the bottom of a push-up with sagging hips, '
           f'next to elbow angle, hip deviation and simulated chest-strap accelerometer traces">',
           f'<rect width="{W}" height="{H}" rx="18" fill="{SLATE}"/>']
    for x in range(30, W, 30):
        out.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="{GRID}" stroke-width="1"/>')
    for y in range(30, H, 30):
        out.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')

    out.append(f'<text x="44" y="78" font-family="{SERIF}" font-size="52" fill="{CHALK}">null-hack</text>')
    out.append(f'<text x="46" y="110" font-family="{MONO}" font-size="16" fill="{MUTED}">'
               f'one push-up set · three signals · zero wearables needed</text>')
    out.append(f'<text x="{W - 44}" y="62" text-anchor="end" font-family="{MONO}" font-size="14" '
               f'fill="{MUTED}">NullHacks{{Helsinki}} · physical AI</text>')

    # ---- figure panel (left)
    fx0, fy0, fw, fh = 40, 150, 470, 360
    xs = [v[0] for f in (frames[i_bot], frames[i_top]) for v in f["lm"].values()]
    ys = [v[1] for f in (frames[i_bot], frames[i_top]) for v in f["lm"].values()]
    sc = min((fw - 40) / (max(xs) - min(xs)), (fh - 90) / (max(ys) - min(ys)))
    ox = fx0 + (fw - (max(xs) - min(xs)) * sc) / 2 - min(xs) * sc
    oy = fy0 + fh - 70 - max(ys) * sc

    def P(v):
        return ox + v[0] * sc, oy + v[1] * sc

    bones = synth.BONES

    def skeleton(lm, ghost=False, red=()):
        g = []
        for a, b in bones:
            (x1, y1), (x2, y2) = P(lm[f"{side}_{a}"]), P(lm[f"{side}_{b}"])
            col = MUTED if ghost else (RED if a in red or b in red else GREEN)
            g.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
                     f'stroke-width="{3 if ghost else 7}" stroke-linecap="round" '
                     f'opacity="{0.45 if ghost else 1}"/>')
        if not ghost:
            for j in {j for bone in bones for j in bone}:
                x, y = P(lm[f"{side}_{j}"])
                r, col = (10, RED) if j in red else (6, CHALK)
                g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}"/>')
        return g

    floor_y = max(P(frames[i_bot]["lm"][f"{side}_foot"])[1], P(frames[i_bot]["lm"][f"{side}_wrist"])[1]) + 8
    out.append(f'<line x1="{fx0}" y1="{floor_y:.1f}" x2="{fx0 + fw}" y2="{floor_y:.1f}" stroke="{MUTED}" '
               f'stroke-width="2" stroke-dasharray="2 6"/>')
    out += skeleton(frames[i_top]["lm"], ghost=True)
    out += skeleton(frames[i_bot]["lm"], red=("hip",))
    hx, hy = P(frames[i_bot]["lm"][f"{side}_hip"])
    sag_deg = hip[i_bot]
    out.append(f'<line x1="{hx:.1f}" y1="{hy + 14:.1f}" x2="{hx:.1f}" y2="{floor_y + 34:.1f}" stroke="{RED}" '
               f'stroke-width="1.5"/>')
    out.append(f'<text x="{hx:.1f}" y="{floor_y + 54:.1f}" text-anchor="middle" font-family="{MONO}" '
               f'font-size="15" fill="{RED}">rep {sag["n"]} · HIP_SAG {sag_deg:.0f}°</text>')
    out.append(f'<text x="{fx0}" y="{fy0 - 8}" font-family="{MONO}" font-size="13" fill="{MUTED}">'
               f'SIMULATED ATHLETE · {s["labels"]["params"]["height_m"]:.2f} m · ghost = top of rep</text>')

    # ---- strip chart (right)
    cx0, cw = 560, 500
    t0, t1 = t[0], t[-1]

    def X(tt):
        return cx0 + (tt - t0) / (t1 - t0) * cw

    lanes = [("ELBOW ANGLE °", elbow, (50, 185), GREEN, [(90, "90° depth"), (160, "160° lock")]),
             ("HIPS OFF LINE °", hip, (-5, 35), BLUE, [(15, "15° limit")]),
             ("CHEST STRAP · ACC m/s² (simulated)", acc, (6.5, 13.0), AMBER, [(9.81, "1 g")])]
    lane_h, gap, y = 92, 26, 160
    for r in reps:
        bad = bool(r["faults"])
        out.append(f'<rect x="{X(r["start"]):.1f}" y="{y - 14}" width="{X(r["end"]) - X(r["start"]):.1f}" '
                   f'height="{3 * lane_h + 2 * gap + 14}" fill="{RED if bad else CHALK}" '
                   f'opacity="{0.13 if bad else 0.04}" rx="4"/>')
        out.append(f'<text x="{(X(r["start"]) + X(r["end"])) / 2:.1f}" y="{y + 3 * lane_h + 2 * gap + 22}" '
                   f'text-anchor="middle" font-family="{MONO}" font-size="12" '
                   f'fill="{RED if bad else MUTED}">rep {r["n"]}</text>')
    for label, series, (lo, hi), col, refs in lanes:
        out.append(f'<text x="{cx0}" y="{y - 2}" font-family="{MONO}" font-size="12" fill="{MUTED}">{label}</text>')

        def Y(v, lo=lo, hi=hi, y=y):
            return y + lane_h - (np.clip(v, lo, hi) - lo) / (hi - lo) * (lane_h - 14)

        for v, name in refs:
            out.append(f'<line x1="{cx0}" y1="{Y(v):.1f}" x2="{cx0 + cw}" y2="{Y(v):.1f}" stroke="{MUTED}" '
                       f'stroke-width="1" stroke-dasharray="4 5" opacity="0.7"/>')
            out.append(f'<text x="{cx0 + cw + 6}" y="{Y(v) + 4:.1f}" font-family="{MONO}" font-size="11" '
                       f'fill="{MUTED}">{name}</text>')
        step = max(1, len(t) // 400)
        pts = " ".join(f"{X(tt):.1f},{Y(v):.1f}" for tt, v in zip(t[::step], series[::step]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2" '
                   f'stroke-linejoin="round"/>')
        y += lane_h + gap
    out.append(f'<line x1="{X(t[i_bot]):.1f}" y1="150" x2="{X(t[i_bot]):.1f}" '
               f'y2="{160 + 3 * lane_h + 2 * gap}" stroke="{RED}" stroke-width="1.5"/>')
    out.append("</svg>")
    path = Path(__file__).with_name("hero.svg")
    path.write_text("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
