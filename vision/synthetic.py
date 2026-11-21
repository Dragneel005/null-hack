"""Simulated push-ups: a 2D kinematic body model that emits contract-valid landmarks
with known ground truth (rep timings and injected faults).

Side view, person faces right, feet on the left. Pixel coordinates, y down.
The model is driven by the elbow angle; the shoulder is placed where an arm of that
bend meets a rigid ankle->shoulder body line, so the motion is physically consistent.
"""
import math

import numpy as np

from .landmarks import NAMES

# Body dimensions in pixels (roughly a 1.75 m person filmed at 1280x720).
L_BODY = 600.0      # ankle -> shoulder
HIP_FRAC = 0.55     # hip position along ankle -> shoulder
U_ARM = 150.0       # shoulder -> elbow
F_ARM = 140.0       # elbow -> wrist
GROUND_Y = 620.0
ANKLE_X = 220.0

E_TOP, E_BOTTOM, E_HALF = 172.0, 72.0, 118.0   # elbow angles (deg)


def _arm_reach(elbow_deg):
    e = math.radians(elbow_deg)
    return math.sqrt(U_ARM**2 + F_ARM**2 - 2 * U_ARM * F_ARM * math.cos(e))


def _circle_intersections(p0, r0, p1, r1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = np.linalg.norm(p1 - p0)
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = math.sqrt(max(r0**2 - a**2, 0.0))
    mid = p0 + a * (p1 - p0) / d
    perp = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / d
    return mid + h * perp, mid - h * perp


def _perp_offset_for_angle(a1, a2, dev_deg):
    """Offset of a point between two segments (lengths a1, a2) giving the requested bend."""
    lo, hi = 0.0, min(a1, a2)
    for _ in range(50):
        o = (lo + hi) / 2
        bend = math.degrees(math.atan(o / a1) + math.atan(o / a2))
        lo, hi = (o, hi) if bend < dev_deg else (lo, o)
    return (lo + hi) / 2


# wrist sits directly under the shoulder at the top position
_WRIST = np.array([ANKLE_X + math.sqrt(L_BODY**2 - _arm_reach(E_TOP)**2), GROUND_Y])
_ANKLE = np.array([ANKLE_X, GROUND_Y])


def _pose(elbow_deg, sag_deg=0.0):
    """Return dict of joint -> (x, y) for one visible side."""
    d = _arm_reach(elbow_deg)
    s1, s2 = _circle_intersections(_WRIST, d, _ANKLE, L_BODY)
    shoulder = s1 if s1[1] < s2[1] else s2                       # the higher one
    e1, e2 = _circle_intersections(_WRIST, F_ARM, shoulder, U_ARM)
    elbow = e1 if e1[0] < e2[0] else e2                          # points back toward the feet

    axis = (shoulder - _ANKLE) / L_BODY
    normal_down = np.array([axis[1], -axis[0]])
    if normal_down[1] < 0:
        normal_down = -normal_down
    hip = _ANKLE + axis * L_BODY * HIP_FRAC
    if sag_deg:
        off = _perp_offset_for_angle(L_BODY * HIP_FRAC, L_BODY * (1 - HIP_FRAC), abs(sag_deg))
        hip = hip + normal_down * off * (1 if sag_deg > 0 else -1)   # >0 sag, <0 pike
    knee = (_ANKLE + hip) / 2
    ear = shoulder + axis * 90 - normal_down * 15
    heel = _ANKLE + np.array([-25.0, -5.0])
    toe = _ANKLE + np.array([20.0, 0.0])
    hand = _WRIST + np.array([25.0, 0.0])
    return {
        "shoulder": shoulder, "elbow": elbow, "wrist": _WRIST, "hip": hip, "knee": knee,
        "ankle": _ANKLE, "ear": ear, "eye": ear + axis * 25, "heel": heel,
        "foot_index": toe, "index": hand, "pinky": hand, "thumb": hand,
    }


def _ease(x):  # 0 -> 1 smoothly
    return 0.5 - 0.5 * math.cos(math.pi * x)


def generate_pushups(reps=10, fps=30, rep_seconds=1.6, pause_seconds=0.3,
                     half_reps=(), hip_sag_from_rep=None, hip_sag_deg=20.0,
                     pike_reps=(), jitter_px=2.0, seed=0, w=1280, h=720):
    rng = np.random.default_rng(seed)
    lead = 0.5
    total = lead + reps * (rep_seconds + pause_seconds) + lead
    n_frames = int(total * fps)

    truth = []
    for n in range(1, reps + 1):
        start = lead + (n - 1) * (rep_seconds + pause_seconds)
        issues = []
        if n in half_reps:
            issues.append("SHALLOW")
        if hip_sag_from_rep is not None and n >= hip_sag_from_rep:
            issues.append("HIP_SAG")
        if n in pike_reps:
            issues.append("PIKE")
        truth.append({"n": n, "start": round(start, 3),
                      "bottom": round(start + rep_seconds / 2, 3),
                      "end": round(start + rep_seconds, 3), "issues": issues})

    frames = []
    for i in range(n_frames):
        t = (i + 1) / fps
        elbow_deg, sag = E_TOP, 0.0
        for r in truth:
            if r["start"] <= t <= r["end"]:
                phase = (t - r["start"]) / rep_seconds          # 0..1
                depth = 1 - abs(2 * phase - 1)                  # 0 -> 1 -> 0
                bottom = E_HALF if "SHALLOW" in r["issues"] else E_BOTTOM
                elbow_deg = E_TOP - (E_TOP - bottom) * _ease(depth)
                if "HIP_SAG" in r["issues"]:
                    sag = hip_sag_deg * _ease(depth)
                if "PIKE" in r["issues"]:
                    sag = -hip_sag_deg * _ease(depth)
        pose = _pose(elbow_deg, sag)
        lm = {}
        for name in NAMES:
            if name == "nose":
                p, vis = pose["ear"] + np.array([20.0, 10.0]), 0.9
            else:
                side, joint = name.split("_", 1)
                base = joint if joint in pose else ("eye" if "eye" in joint else "ear")
                p = pose[base]
                vis = 0.95 if side == "l" else 0.35
                if side == "r":
                    p = p + np.array([6.0, -4.0])   # far side, mostly hidden
            p = p + rng.normal(0, jitter_px, 2) if jitter_px else p
            lm[name] = [round(float(p[0]), 2), round(float(p[1]), 2), vis]
        frames.append({"t": round(t, 4), "lm": lm})

    return {
        "version": 1, "fps": fps, "w": w, "h": h,
        "side": "left", "confidence": 0.95, "message": None,
        "source": "synthetic",
        "frames": frames,
        "truth": {"rep_count": reps, "reps": truth},
    }
