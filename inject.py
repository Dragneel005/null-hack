"""Simulated fault injection on REAL pose tracks ("real motion, simulated fault").

`inject_fault(landmarks, code, strength=1.0) -> landmarks` returns a deep copy of a pose.extract track
where the 2D keypoints are geometrically edited so the athlete shows a known form fault:

- HIP_SAG / HIP_PIKE: hips (and knees, half as much) pushed below / above the shoulder-ankle line;
  ear and nose are rotated with the trunk so the head angle stays as it was.
- SHALLOW: every elbow angle is compressed toward 180 deg (elbow moved toward the shoulder-wrist line),
  so the bottom reads ~115-120 deg instead of ~45 deg.
- NO_LOCKOUT: elbow angles above 120 deg are compressed so the top only reaches ~140 deg.
- HEAD_DROP: ear and nose rotated down about the shoulder by ~30 deg.

The real timing, noise and tracking artefacts are untouched, so this gives real-world clips with known
labels: used by the app ("simulate a fault on this real clip") and by benchmark.sim_to_real.
"""
from __future__ import annotations

import copy
import math

import numpy as np

FAULT_OPTIONS: dict[str, str] = {
    "none": "None (real form)",
    "HIP_SAG": "Hip sag",
    "HIP_PIKE": "Hip pike",
    "SHALLOW": "Shallow depth",
    "NO_LOCKOUT": "No lockout",
    "HEAD_DROP": "Head drop",
}


def _down_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    nd = float(np.hypot(*d))
    if nd < 1e-9:
        return np.array([0.0, 1.0])
    d = d / nd
    n = np.array([-d[1], d[0]])
    return -n if n[1] < 0 else n


def _rot(p: np.ndarray, c: np.ndarray, ang: float) -> np.ndarray:
    ca, sa = math.cos(ang), math.sin(ang)
    v = p - c
    return c + np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]])


def _signed_rot(v0: np.ndarray, v1: np.ndarray) -> float:
    return math.atan2(v0[0] * v1[1] - v0[1] * v1[0], v0[0] * v1[0] + v0[1] * v1[1])


def _angle(s: np.ndarray, e: np.ndarray, w: np.ndarray) -> float:
    u, v = s - e, w - e
    den = float(np.hypot(*u) * np.hypot(*v))
    if den < 1e-9:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v) / den))))


def _set_elbow_angle(s: np.ndarray, e: np.ndarray, w: np.ndarray, target: float) -> np.ndarray:
    """Move the elbow along the perpendicular to the shoulder-wrist line until angle(s,e,w) == target."""
    d = w - s
    dd = float(d @ d)
    if dd < 1e-9:
        return e
    m = s + d * float((e - s) @ d) / dd          # foot of the perpendicular from e onto line s-w
    if float(np.hypot(*(m - e))) < 1e-6:
        return e
    lo, hi = -6.0, 0.999                         # angle increases monotonically with lam on this range
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _angle(s, e + mid * (m - e), w) < target:
            lo = mid
        else:
            hi = mid
    return e + 0.5 * (lo + hi) * (m - e)


def inject_fault(landmarks: dict, code: str, strength: float = 1.0) -> dict:
    """Deep-copied landmarks with a simulated form fault applied to every frame (both body sides)."""
    out = copy.deepcopy(landmarks)
    if not code or code == "none" or code not in FAULT_OPTIONS:
        return out
    out["source"] = f"{landmarks.get('source', 'clip')} + simulated {code}"
    out["injected_fault"] = code
    for fr in out.get("frames", []):
        lm = fr.get("lm")
        if not lm:
            continue
        P = {k: np.array(v[:2], float) for k, v in lm.items()}
        for sd in ("l_", "r_"):
            try:
                sh, hp, an = P[sd + "shoulder"], P[sd + "hip"], P[sd + "ankle"]
            except KeyError:
                continue
            torso = float(np.hypot(*(sh - hp)))
            if code in ("HIP_SAG", "HIP_PIKE"):
                k = (0.25 if code == "HIP_SAG" else -0.30) * strength * torso
                n = _down_normal(sh, an)
                new_hp = hp + k * n
                ang = _signed_rot(sh - hp, sh - new_hp)
                P[sd + "hip"] = new_hp
                if sd + "knee" in P:
                    P[sd + "knee"] = P[sd + "knee"] + 0.5 * k * n
                for h in (sd + "ear",) + (("nose",) if sd == "l_" else ()):
                    if h in P:
                        P[h] = _rot(P[h], sh, ang)
            elif code == "HEAD_DROP":
                n = _down_normal(hp, sh)
                d = sh - hp
                # rotate toward the "below the trunk line" side by 30 deg
                test = _rot(sh + d, sh, math.radians(10))
                sgn = 1.0 if float((test - sh) @ n) > 0 else -1.0
                for h in (sd + "ear",) + (("nose",) if sd == "l_" else ()):
                    if h in P:
                        P[h] = _rot(P[h], sh, sgn * math.radians(30 * strength))
            elif code in ("SHALLOW", "NO_LOCKOUT") and sd + "elbow" in P and sd + "wrist" in P:
                e, w = P[sd + "elbow"], P[sd + "wrist"]
                a = _angle(sh, e, w)
                if math.isnan(a):
                    continue
                if code == "SHALLOW":
                    tgt = 180.0 - (180.0 - a) * (1.0 - 0.55 * strength)
                else:
                    tgt = a if a < 120 else 120.0 + (a - 120.0) * max(0.0, 1.0 - 0.6 * strength)
                if abs(tgt - a) > 0.05:
                    P[sd + "elbow"] = _set_elbow_angle(sh, e, w, tgt)
        for k2, v in P.items():
            lm[k2] = [round(float(v[0]), 2), round(float(v[1]), 2)] + list(lm[k2][2:])
    return out


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    import analysis

    p = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "samples/landmarks_demo_side.json")
    lm0 = json.loads(p.read_text())
    for c in FAULT_OPTIONS:
        a = analysis.analyze(inject_fault(lm0, c))
        print(f"{c:<11} reps {a['summary']['rep_count']}  issues {a['summary']['issue_counts']}  "
              f"avg score {a['summary']['avg_score']}")
