"""Theme layer — simulated athletes -> labelled push-up dataset (see CONTRACT.md §6).

A procedural kinematic body does push-ups. Anthropometrics, tempo, depth, faults,
fatigue and the camera are randomised; every frame is projected to 2D pixels in the
*same schema as pose.extract*, so analysis/benchmark code runs on simulated and real
clips alike. Each sample carries exact ground truth (rep times + fault codes).

Model (sagittal plane, metres, x = direction the athlete faces, y = up):
  ankle and wrist are fixed on the floor; the elbow angle drives the motion; the
  shoulder sits where an arm of that bend meets a rigid ankle->shoulder body line;
  hip sag/pike, head drop are offsets from that line.
"""
from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

NAMES = ["nose", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
         "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
         "l_heel", "r_heel", "l_foot", "r_foot"]
FAULT_CODES = ["SHALLOW", "NO_LOCKOUT", "HIP_SAG", "HIP_PIKE", "HEAD_DROP", "FAST_DESCENT"]
BONES = [("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
         ("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "foot"),
         ("ankle", "foot")]
_HALF_WIDTH = {"ear": 0.07, "shoulder": 0.19, "elbow": 0.22, "wrist": 0.2, "hip": 0.15,
               "knee": 0.12, "ankle": 0.1, "heel": 0.1, "foot": 0.1}   # for a 1.75 m body


# ---------------------------------------------------------------- geometry

def _circles(p0, r0, p1, r1):
    d = float(np.linalg.norm(p1 - p0))
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h = math.sqrt(max(r0 * r0 - a * a, 0.0))
    mid = p0 + a * (p1 - p0) / d
    perp = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / d
    return mid + h * perp, mid - h * perp


def _rot(v, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _ease(x):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0.0), 1.0))


class _Body:
    def __init__(self, rng, height_m):
        k = height_m
        self.height = height_m
        self.L = 0.78 * k * rng.uniform(0.97, 1.03)       # ankle -> shoulder
        self.u = 0.186 * k * rng.uniform(0.95, 1.05)      # upper arm
        self.f = 0.146 * k * rng.uniform(0.95, 1.05)      # forearm (to wrist)
        self.hip_frac = rng.uniform(0.60, 0.65)
        self.neck = 0.10 * k
        self.ankle = np.array([0.0, 0.08])
        r_top = self.reach(172.0)
        self.wrist = np.array([math.sqrt(self.L ** 2 - (r_top - 0.05) ** 2), 0.03])

    def reach(self, elbow_deg):
        e = math.radians(elbow_deg)
        return math.sqrt(self.u ** 2 + self.f ** 2 - 2 * self.u * self.f * math.cos(e))

    def pose(self, elbow_deg, hip_dev_deg=0.0, head_drop_deg=0.0):
        """Joint -> (x, y) metres. hip_dev > 0 sag, < 0 pike."""
        s1, s2 = _circles(self.wrist, self.reach(elbow_deg), self.ankle, self.L)
        sh = s1 if s1[1] > s2[1] else s2                          # the higher solution
        e1, e2 = _circles(self.wrist, self.f, sh, self.u)
        el = e1 if e1[0] < e2[0] else e2                          # elbow points to the feet
        axis = (sh - self.ankle) / self.L
        down = np.array([axis[1], -axis[0]])
        if down[1] > 0:
            down = -down
        a1, a2 = self.L * self.hip_frac, self.L * (1 - self.hip_frac)
        # offset o so that atan(o/a1) + atan(o/a2) == |dev|  (bisection)
        lo, hi, dev = 0.0, min(a1, a2), abs(hip_dev_deg)
        for _ in range(40):
            o = (lo + hi) / 2
            lo, hi = (o, hi) if math.degrees(math.atan(o / a1) + math.atan(o / a2)) < dev else (lo, o)
        hip = self.ankle + axis * a1 + down * lo * (1 if hip_dev_deg >= 0 else -1)
        knee = self.ankle + (hip - self.ankle) * 0.52
        neck_dir = _rot((sh - hip) / np.linalg.norm(sh - hip), -head_drop_deg)
        ear = sh + neck_dir * self.neck
        nose = ear + _rot(neck_dir, -70) * 0.06 * self.height / 1.75
        return {"shoulder": sh, "elbow": el, "wrist": self.wrist, "hip": hip, "knee": knee,
                "ankle": self.ankle, "heel": self.ankle + np.array([-0.04, 0.05]),
                "foot": np.array([self.ankle[0] + 0.07, 0.0]), "ear": ear, "nose": nose}


# ---------------------------------------------------------------- one athlete

def _plan_faults(rng, n_reps, faults):
    per_rep = [[] for _ in range(n_reps)]
    fatigue_from = None
    if faults == "none":
        return per_rep, None
    if faults == "mixed":
        for r in per_rep:
            if rng.random() < 0.35:
                r.append(str(rng.choice(FAULT_CODES)))
                if rng.random() < 0.15:
                    extra = str(rng.choice(FAULT_CODES))
                    if extra not in r:
                        r.append(extra)
        if n_reps >= 4 and rng.random() < 0.35:
            fatigue_from = int(rng.integers(n_reps // 2 + 1, n_reps + 1))
            for r in per_rep[fatigue_from - 1:]:
                if "HIP_SAG" not in r:
                    r.append("HIP_SAG")
                if "HIP_PIKE" in r:
                    r.remove("HIP_PIKE")
    else:
        codes = [faults] if isinstance(faults, str) else list(faults)
        for c in codes:
            if c not in FAULT_CODES:
                raise ValueError(f"unknown fault code {c!r}")
            k = max(1, min(n_reps - 1, n_reps // 2))
            for idx in rng.choice(n_reps, size=k, replace=False):
                per_rep[int(idx)].append(c)
        clean = [i for i, r in enumerate(per_rep) if not r]
        if not clean:                                      # keep at least one clean rep
            per_rep[int(rng.integers(n_reps))] = []
    for r in per_rep:                                      # sag and pike are exclusive
        if "HIP_SAG" in r and "HIP_PIKE" in r:
            r.remove("HIP_PIKE")
    return per_rep, fatigue_from


def _camera(rng, mode):
    if mode == "ideal":
        return {"w": 1280, "h": 720, "fps": 30.0, "yaw_deg": 0.0, "roll_deg": 0.0,
                "facing": 1, "fill": 0.7, "ground": 0.85, "shift": 0.0, "jitter_px": 0.0,
                "drop_p": 0.0, "burst": False}
    w, h = [(640, 480), (1280, 720), (1280, 720), (1920, 1080)][int(rng.integers(4))]
    return {"w": w, "h": h, "fps": float(rng.choice([24.0, 30.0, 30.0, 60.0])),
            "yaw_deg": float(rng.uniform(0, 25)), "roll_deg": float(rng.uniform(-3, 3)),
            "facing": int(rng.choice([-1, 1])), "fill": float(rng.uniform(0.5, 0.8)),
            "ground": float(rng.uniform(0.78, 0.9)), "shift": float(rng.uniform(-0.06, 0.06)),
            "jitter_px": float(rng.uniform(0.3, 2.5) * w / 1280),
            "drop_p": float(rng.uniform(0, 0.02)), "burst": bool(rng.random() < 0.3)}


CLOTH_WEIGHT = {"shoulder": 0.5, "hip": 1.0, "knee": 0.6}   # shirt / waistband / trousers


def _cloth_plan(cloth_rng, clothing):
    """Loose clothing: MediaPipe tracks the fabric outline, so torso points hang toward where
    the cloth falls (image-down in a plank), sway, jitter more and lose visibility."""
    baggy = clothing == "baggy" or (clothing == "random" and cloth_rng.random() < 0.5)
    if not baggy:
        return None
    return {"hang": float(cloth_rng.uniform(0.02, 0.05)), "sway": float(cloth_rng.uniform(0.008, 0.02)),
            "freq": float(cloth_rng.uniform(0.6, 1.4)), "phase": float(cloth_rng.uniform(0, 2 * math.pi)),
            "jitter": float(cloth_rng.uniform(1.5, 3.5)), "vis_drop": float(cloth_rng.uniform(0.15, 0.35)),
            "seed": int(cloth_rng.integers(2**31))}


def _athlete(rng, idx, faults, camera, cloth=None):
    body = _Body(rng, float(rng.uniform(1.55, 1.95)))
    cam = _camera(rng, camera)
    n_reps = int(rng.integers(4, 9) if camera == "ideal" else rng.integers(3, 13))
    rep_faults, fatigue_from = _plan_faults(rng, n_reps, faults)
    natural_sag = float(rng.uniform(-3, 3))
    head_base = float(rng.uniform(0, 8))

    # top angle held *after* rep k (k = 0 is the lead-in); NO_LOCKOUT lowers both ends
    tops = [float(rng.uniform(166, 177)) for _ in range(n_reps + 1)]
    for i, fs in enumerate(rep_faults):
        if "NO_LOCKOUT" in fs:
            tops[i] = tops[i + 1] = float(rng.uniform(146, 152))
    # a rep lacks lockout iff the arm never straightens at either end (neighbours share tops)
    for i, fs in enumerate(rep_faults):
        if "NO_LOCKOUT" in fs:
            fs.remove("NO_LOCKOUT")
        if max(tops[i], tops[i + 1]) < 155:
            fs.append("NO_LOCKOUT")

    # timeline of segments (t0, t1, a0, a1, rep_index or None)
    t = float(rng.uniform(0.5, 1.5))
    segs = [(0.0, t, tops[0], tops[0], None)]
    reps = []
    for i, fs in enumerate(rep_faults):
        fatigue = 0 if fatigue_from is None or i + 1 < fatigue_from else (i + 2 - fatigue_from)
        bottom = float(rng.uniform(105, 125) if "SHALLOW" in fs else rng.uniform(65, 85))
        d = float(rng.uniform(0.2, 0.33) if "FAST_DESCENT" in fs else rng.uniform(0.6, 1.2))
        hold = float(rng.uniform(0.0, 0.15))
        a = float(rng.uniform(0.5, 1.0)) * (1 + 0.1 * fatigue)
        start, bot_t = t, t + d
        end = bot_t + hold + a
        sag = 0.0
        if "HIP_SAG" in fs:
            sag = float(rng.uniform(16, 26)) + 2.0 * fatigue
        elif "HIP_PIKE" in fs:
            sag = -float(rng.uniform(16, 26))
        drop = float(rng.uniform(35, 50)) if "HEAD_DROP" in fs else 0.0
        segs += [(start, bot_t, tops[i], bottom, i), (bot_t, bot_t + hold, bottom, bottom, i),
                 (bot_t + hold, end, bottom, tops[i + 1], i)]
        reps.append({"n": i + 1, "start": round(start, 3), "bottom": round(bot_t, 3),
                     "end": round(end, 3), "faults": fs,
                     "_sag": sag, "_drop": drop, "_top0": tops[i], "_bottom": bottom})
        pause = float(rng.uniform(0.15, 1.5))
        segs.append((end, end + pause, tops[i + 1], tops[i + 1], None))
        t = end + pause
    total = t + float(rng.uniform(0.0, 0.5))

    fps, w, h = cam["fps"], cam["w"], cam["h"]
    n_frames = int(math.ceil(total * fps))
    yaw, facing = math.radians(cam["yaw_deg"]), cam["facing"]
    near = "r" if facing > 0 else "l"
    scale = body.height / 1.75

    # pixel scale + placement from the top pose so the athlete fits the frame
    ref = body.pose(tops[0])
    xs = [p[0] for p in ref.values()]
    extent_m = (max(xs) - min(xs)) * math.cos(yaw) + 0.2
    ppm = cam["fill"] * w / extent_m
    ppm = min(ppm, cam["ground"] * h * 0.8 / 0.9)          # head must stay inside
    ground_y = cam["ground"] * h
    cx = (max(xs) + min(xs)) / 2
    x_off = w * (0.5 + cam["shift"])
    roll = math.radians(cam["roll_deg"])

    vis_near = float(rng.uniform(0.88, 0.99))
    vis_far = {j: float(rng.uniform(0.05, 0.55)) for j in _HALF_WIDTH}
    burst = None
    if cam["burst"]:
        b0 = int(rng.integers(0, max(1, n_frames - 10)))
        burst = (b0, b0 + int(rng.integers(3, 9)))

    cloth_noise = np.random.default_rng(cloth["seed"]) if cloth else None
    frames = []
    for fi in range(n_frames):
        tt = fi / fps
        elbow, rep_i = tops[-1], None
        for (t0, t1, a0, a1, ri) in segs:
            if t0 <= tt <= t1:
                elbow = a0 + (a1 - a0) * _ease((tt - t0) / (t1 - t0) if t1 > t0 else 1.0)
                rep_i = ri
                break
        sag, drop = natural_sag, head_base
        if rep_i is not None:
            r = reps[rep_i]
            span = max(r["_top0"] - r["_bottom"], 1e-6)
            depth = min(max((r["_top0"] - elbow) / span, 0.0), 1.0)
            if tt > r["bottom"]:                         # on the way up use the end top
                span = max(tops[rep_i + 1] - r["_bottom"], 1e-6)
                depth = min(max((tops[rep_i + 1] - elbow) / span, 0.0), 1.0)
            sag += r["_sag"] * _ease(depth)
            drop += r["_drop"] * _ease(depth)
        pose = body.pose(elbow, sag, drop)

        dropped = (rng.random() < cam["drop_p"]) or (burst and burst[0] <= fi < burst[1])
        if dropped:
            frames.append({"i": fi, "t": round(tt, 4), "lm": None, "world": None})
            continue
        hipm = pose["hip"]
        lm, world = {}, {}
        for name in NAMES:
            side, joint = (name[0], name[2:]) if name[1] == "_" else (near, name)
            X, Y = pose[joint]
            z = _HALF_WIDTH.get(joint, 0.0) * scale * (-1 if side == near else 1)
            if name == "nose":
                z = 0.0
            xp = x_off + facing * ((X - cx) * math.cos(yaw) + z * math.sin(yaw)) * ppm
            yp = ground_y - Y * ppm
            dx, dy = xp - w / 2, yp - h / 2
            xp = w / 2 + dx * math.cos(roll) - dy * math.sin(roll)
            yp = h / 2 + dx * math.sin(roll) + dy * math.cos(roll)
            if cam["jitter_px"]:
                xp += rng.normal(0, cam["jitter_px"])
                yp += rng.normal(0, cam["jitter_px"])
            v = vis_near if (side == near or name == "nose") else vis_far[joint]
            v = float(np.clip(v + rng.normal(0, 0.02), 0, 1))
            if cloth and joint in CLOTH_WEIGHT and name != "nose":
                k = CLOTH_WEIGHT[joint] * body.L * ppm
                yp += k * (cloth["hang"] + cloth["sway"] * math.sin(2 * math.pi * cloth["freq"] * tt
                                                                     + cloth["phase"]))
                xp += cloth_noise.normal(0, cloth["jitter"] * w / 1280 * CLOTH_WEIGHT[joint])
                yp += cloth_noise.normal(0, cloth["jitter"] * w / 1280 * CLOTH_WEIGHT[joint])
                v = float(np.clip(v - cloth["vis_drop"] * CLOTH_WEIGHT[joint], 0.05, 1))
            lm[name] = [round(float(xp), 2), round(float(yp), 2), round(v, 3)]
            world[name] = [round(facing * float(X - hipm[0]), 4), round(-float(Y - hipm[1]), 4),
                           round(z, 4)]
        frames.append({"i": fi, "t": round(tt, 4), "lm": lm, "world": world})

    labels = {
        "rep_count": n_reps,
        "reps": [{k: v for k, v in r.items() if not k.startswith("_")} for r in reps],
        "fatigue_from_rep": fatigue_from,
        "params": {"height_m": round(body.height, 3), "camera": cam,
                   "near_side": "right" if near == "r" else "left",
                   "px_per_m": round(ppm, 2), "clothing": "baggy" if cloth else "fitted",
                   "cloth": cloth, "faults_mode": faults if isinstance(faults, str)
                   else list(faults)},
    }
    landmarks = {"source": f"synth_{idx:05d}", "fps": fps, "w": w, "h": h,
                 "n_frames": n_frames,
                 "detected_frames": sum(f["lm"] is not None for f in frames),
                 "frames": frames}
    return {"landmarks": landmarks, "labels": labels}


# ---------------------------------------------------------------- public API

def generate(n: int = 500, seed: int = 0, faults="mixed", camera: str = "random",
             clothing: str = "fitted") -> list[dict]:
    """Simulate `n` athletes. faults: "mixed" | "none" | code | [codes]. camera: "random" | "ideal".
    clothing: "fitted" | "baggy" | "random" (half baggy). Clothing uses its own random stream,
    so the same seed gives the same athletes and reps whatever the clothing.

    Returns [{"landmarks": <pose.extract schema>, "labels": {"rep_count", "reps": [{"n",
    "start", "bottom", "end", "faults"}], "fatigue_from_rep", "params"}}].
    """
    if clothing not in ("fitted", "baggy", "random"):
        raise ValueError(f"clothing must be fitted, baggy or random, not {clothing!r}")
    rng = np.random.default_rng(seed)
    cloth_rng = np.random.default_rng([seed, 7919])
    return [_athlete(rng, i, faults, camera, _cloth_plan(cloth_rng, clothing)) for i in range(n)]


def export(samples: list[dict], out_dir, with_imu: bool = False) -> str:
    """Write samples to <out_dir>/synth_dataset.zip (JSONL + frames.csv + reps.csv + README)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synth_dataset.zip"
    jsonl, imu_lines = io.StringIO(), io.StringIO()
    frames_csv, reps_csv = io.StringIO(), io.StringIO()
    fw = csv.writer(frames_csv)
    fw.writerow(["sample_id", "i", "t"] + [f"{n}_{c}" for n in NAMES for c in ("x", "y", "vis")])
    rw = csv.writer(reps_csv)
    rw.writerow(["sample_id", "n", "start", "bottom", "end", "faults"])
    counts = {c: 0 for c in FAULT_CODES}
    total_reps = 0
    for s in samples:
        sid = s["landmarks"]["source"]
        jsonl.write(json.dumps({"id": sid, **s}) + "\n")
        for f in s["landmarks"]["frames"]:
            vals = [""] * (3 * len(NAMES)) if f["lm"] is None else \
                [v for n in NAMES for v in f["lm"][n]]
            fw.writerow([sid, f["i"], f["t"]] + vals)
        for r in s["labels"]["reps"]:
            rw.writerow([sid, r["n"], r["start"], r["bottom"], r["end"], ";".join(r["faults"])])
            total_reps += 1
            for c in r["faults"]:
                counts[c] += 1
        if with_imu:
            import imu
            imu_lines.write(json.dumps({"id": sid, "imu": imu.simulate(s["landmarks"])}) + "\n")

    readme = _datasheet(len(samples), total_reps, counts, with_imu)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("samples.jsonl", jsonl.getvalue())
        z.writestr("frames.csv", frames_csv.getvalue())
        z.writestr("reps.csv", reps_csv.getvalue())
        z.writestr("README.md", readme)
        if with_imu:
            z.writestr("imu.jsonl", imu_lines.getvalue())
    return str(path)


def _datasheet(n, total_reps, counts, with_imu):
    rows = "\n".join(f"| {c} | {k} |" for c, k in counts.items())
    imu_note = ("\n`imu.jsonl`: simulated wearable sensors (chest strap, waist clip, smartwatch): "
                "accelerometer m/s² incl. gravity + gyroscope deg/s, see imu.py.\n") if with_imu else ""
    return f"""# Simulated push-up dataset (null-hack synth.py)

{n} simulated athletes, {total_reps} labelled reps. Side-view camera, 2D keypoints in
pixels (x right, y down) in the same schema as `pose.extract`, plus hip-centred 3D
world coordinates in metres.

Randomised per athlete: height 1.55–1.95 m and segment proportions, rep count, tempo,
depth, top holds, fault injection, fatigue drift, camera resolution / fps / yaw (0–25°
off side view) / roll / distance, keypoint jitter, dropped detections.

Files: `samples.jsonl` (one athlete per line: landmarks + labels), `frames.csv`
(sample_id, frame, time, x/y/visibility per joint), `reps.csv` (ground-truth rep
windows and fault codes).
{imu_note}
| Fault | Reps |
|---|---|
{rows}

Fault definitions: SHALLOW elbow never < 95° · NO_LOCKOUT elbow never > 155° ·
HIP_SAG / HIP_PIKE hip > 12° below / above the shoulder–ankle line · HEAD_DROP head
angle well below body line · FAST_DESCENT descent < 0.4 s.
"""


def render_preview(sample: dict, t: float):
    """Plotly figure of the simulated skeleton (near side) at time t."""
    import plotly.graph_objects as go

    lms = sample["landmarks"]
    frame = min(lms["frames"], key=lambda f: abs(f["t"] - t))
    fig = go.Figure()
    if frame["lm"]:
        near = sample["labels"]["params"]["near_side"][0]
        for a, b in BONES:
            pa, pb = frame["lm"][f"{near}_{a}"], frame["lm"][f"{near}_{b}"]
            fig.add_trace(go.Scatter(x=[pa[0], pb[0]], y=[pa[1], pb[1]], mode="lines+markers",
                                     line={"width": 5}, showlegend=False))
    fig.update_xaxes(range=[0, lms["w"]])
    fig.update_yaxes(range=[lms["h"], 0], scaleanchor="x")
    fig.update_layout(title=f"{lms['source']} t={frame['t']:.2f}s", height=400,
                      margin={"l": 10, "r": 10, "t": 40, "b": 10})
    return fig


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    data = generate(n=int(sys.argv[2]) if len(sys.argv) > 2 else 50, seed=0)
    print(export(data, out))
