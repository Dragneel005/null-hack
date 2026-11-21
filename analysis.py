"""Stage 2: landmarks -> reps, angles, faults, fatigue (CONTRACT.md §2).

    analyze(landmarks, strict=False, thresholds=None) -> analysis

1. activity.detect finds where the person is actually in push-up posture (trims setup and
   standing up; rejects clips that aren't push-ups).
2. Per frame on the camera-facing side: elbow angle, hip deviation from the
   shoulder-ankle line (signed, + = sag), head angle; lightly smoothed.
3. Reps = elbow-angle valleys with >= 25° prominence, so shallow reps and reps without
   lockout are still counted. A rep runs from the start of the descent to the end of the ascent.
4. Hips, head and lockout are judged against the athlete's OWN top-of-rep position,
   not a perfect straight line, so loose clothing (which drags the tracked hip point
   down) doesn't read as sagging. A hip line that is already bent at the top is reported
   as a warning instead.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import find_peaks

import activity

STANDARD = {False: ("Test standard (90°)", {"depth_max_elbow": 90, "lockout_min_elbow": 160,
                                            "hip_tolerance": 15}),
            True: ("Strict (chest near floor)", {"depth_max_elbow": 75, "lockout_min_elbow": 165,
                                                 "hip_tolerance": 12})}
MIN_PROMINENCE = 25.0      # deg of elbow bend that makes a rep
HEAD_DROP_DEG = 25.0       # head angle this far below its top-of-rep baseline
FAST_DESCENT_S = 0.4         # whole descent; judged on the 10-90 % part of it:
FAST_DESCENT_1090_S = 0.25   # ~0.6 of a smooth descent, robust to where smoothing puts the start
CONSTANT_SAG_DEG = 10.0
LOCKOUT_SLACK_DEG = 6.0      # pose models under-read straight arms; compare to the athlete's best


# ---------------------------------------------------------------- per-frame series

def _ang(a, b, c):
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    return float("nan") if n == 0 else math.degrees(
        math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))))


def _hip_dev(lm, s):
    """Signed degrees the hip is off the shoulder-ankle line: + below (sag), - above (pike)."""
    sh, hip, an = lm[f"{s}_shoulder"], lm[f"{s}_hip"], lm[f"{s}_ankle"]
    dev = 180.0 - _ang(sh, hip, an)
    ex, ey = sh[0] - an[0], sh[1] - an[1]
    cross = ex * (hip[1] - an[1]) - ey * (hip[0] - an[0])       # image y points down
    below = cross > 0 if ex > 0 else cross < 0
    return dev if below else -dev


def _smooth(x, window):
    """NaN-aware centered moving average."""
    if window <= 1:
        return x
    half = window // 2
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) and not np.isnan(x[i]):
            out[i] = seg.mean()
    return out


def _fill(x):
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    return np.interp(idx, idx[ok], x[ok]) if ok.any() else x


# ---------------------------------------------------------------- reps

def _find_reps(t, elbow, in_pushup, fps):
    e = _fill(np.where(in_pushup, elbow, np.nan))
    if np.isnan(e).all():
        return []
    valleys, _ = find_peaks(-e, prominence=MIN_PROMINENCE, distance=max(1, int(0.5 * fps)))
    valleys = [v for v in valleys if in_pushup[v]]
    reps, bounds = [], [0] + valleys + [len(e) - 1]
    for k, v in enumerate(valleys):
        left, right = bounds[k] if k else 0, bounds[k + 2]
        li = left + int(np.argmax(e[left:v + 1]))                  # top before
        ri = v + int(np.argmax(e[v:right + 1]))                    # top after
        top_l, top_r = e[li], e[ri]
        s = v
        while s > li and e[s - 1] < top_l - 3:                     # start of descent
            s -= 1
        end = v
        while end < ri and e[end + 1] < top_r - 3:                 # end of ascent
            end += 1
        s, end = max(li, s - 1), min(ri, end + 1)
        reps.append((s, int(v), end))
    return reps


def fatigue_from(reps):
    """First rep n from which every remaining rep (at least 2) has an issue, after a clean start."""
    flags = [bool(r["issues"]) for r in reps]
    for k in range(1, len(flags) - 1):
        if not flags[k - 1] and all(flags[k:]):
            return reps[k]["n"]
    return None


def _severity(excess):
    return "low" if excess < 5 else ("med" if excess < 10 else "high")


# ---------------------------------------------------------------- public

def analyze(landmarks: dict, strict: bool = False, thresholds: dict | None = None) -> dict:
    standard, th = STANDARD[bool(strict)]
    th = {**th, **(thresholds or {})}
    frames = landmarks["frames"]
    fps = float(landmarks.get("fps") or 30.0)
    t = np.array([f["t"] for f in frames], dtype=float)
    act = activity.detect(landmarks)
    warnings = [] if act["is_pushup"] else [f"Not counted as a push-up set: {act['reason']}."]

    base = {"exercise": "pushup", "standard": standard, "thresholds": th,
            "activity": {k: v for k, v in act.items() if k != "in_pushup"}}
    if not any(f.get("lm") for f in frames):
        return {**base, "side": "left", "confidence": 0.0, "confidence_ok": False,
                "series": {"t": t.round(3).tolist(), "elbow": [None] * len(frames),
                           "hip_dev": [None] * len(frames)},
                "reps": [], "summary": {"rep_count": 0, "clean_reps": 0, "fatigue_from_rep": None},
                "warnings": warnings or ["No person detected."]}

    s = activity._side(frames)
    nan = float("nan")
    elbow = np.array([_ang(f["lm"][f"{s}_shoulder"], f["lm"][f"{s}_elbow"], f["lm"][f"{s}_wrist"])
                      if f.get("lm") else nan for f in frames])
    hip = np.array([_hip_dev(f["lm"], s) if f.get("lm") else nan for f in frames])
    head = np.array([_ang(f["lm"][f"{s}_ear"], f["lm"][f"{s}_shoulder"], f["lm"][f"{s}_hip"])
                     if f.get("lm") else nan for f in frames])
    w = max(3, int(round(0.15 * fps)) | 1)
    elbow, hip, head = _smooth(elbow, w), _smooth(hip, w), _smooth(head, w)
    in_pushup = np.array(act["in_pushup"], dtype=bool)

    vis = [f["lm"][f"{s}_{j}"][2] for f in frames if f.get("lm")
           for j in ("shoulder", "elbow", "wrist", "hip", "ankle")]
    detected = sum(1 for f in frames if f.get("lm")) / len(frames)
    confidence = round(float(np.mean(vis)) * detected, 3)

    rep_idx = _find_reps(t, elbow, in_pushup, fps) if act["is_pushup"] else []

    # athlete's own line at the top of their reps (baggy-clothes / body-shape baseline)
    top_mask = in_pushup & (elbow > np.nanpercentile(elbow[in_pushup], 70)) if in_pushup.any() \
        else ~np.isnan(elbow)
    hip_base = float(np.nanmedian(hip[top_mask])) if top_mask.any() else 0.0
    head_base = float(np.nanmedian(head[top_mask])) if top_mask.any() else 180.0
    if abs(hip_base) > CONSTANT_SAG_DEG:
        warnings.append(f"Hips are {abs(hip_base):.0f}° {'below' if hip_base > 0 else 'above'} the "
                        f"shoulder-ankle line even at the top. That's either a constant "
                        f"{'sag' if hip_base > 0 else 'pike'} or loose clothing; judged relative to it.")
    hip_rel = hip - hip_base
    best_lockout = float(np.nanpercentile(elbow[in_pushup], 95)) if in_pushup.any() else 180.0

    reps = []
    for n, (a, b, c) in enumerate(rep_idx, 1):
        seg = slice(a, c + 1)
        mn, mx = float(np.nanmin(elbow[seg])), float(np.nanmax(elbow[seg]))
        hseg = hip_rel[seg]
        hi_i, lo_i = a + int(np.nanargmax(hseg)), a + int(np.nanargmin(hseg))
        max_dev = float(hip_rel[hi_i]) if abs(hip_rel[hi_i]) >= abs(hip_rel[lo_i]) else float(hip_rel[lo_i])
        issues = []
        if mn > th["depth_max_elbow"]:
            issues.append({"code": "SHALLOW", "t": float(t[b]), "value": round(mn, 1),
                           "severity": _severity(mn - th["depth_max_elbow"]),
                           "message": f"Only reached {mn:.0f}° at the elbow (target {th['depth_max_elbow']}°)"})
        if mx < min(th["lockout_min_elbow"], best_lockout - LOCKOUT_SLACK_DEG):
            issues.append({"code": "NO_LOCKOUT", "t": float(t[c]), "value": round(mx, 1),
                           "severity": _severity(th["lockout_min_elbow"] - mx),
                           "message": f"Arms only straightened to {mx:.0f}° (target {th['lockout_min_elbow']}°)"})
        if hip_rel[hi_i] > th["hip_tolerance"]:
            v = float(hip_rel[hi_i])
            issues.append({"code": "HIP_SAG", "t": float(t[hi_i]), "value": round(v, 1),
                           "severity": _severity(v - th["hip_tolerance"]),
                           "message": f"Hips {v:.0f}° below the shoulder-ankle line"})
        if hip_rel[lo_i] < -th["hip_tolerance"]:
            v = float(-hip_rel[lo_i])
            issues.append({"code": "HIP_PIKE", "t": float(t[lo_i]), "value": round(v, 1),
                           "severity": _severity(v - th["hip_tolerance"]),
                           "message": f"Hips {v:.0f}° above the shoulder-ankle line"})
        head_min_i = a + int(np.nanargmin(head[seg]))
        drop = head_base - float(head[head_min_i])
        if drop > HEAD_DROP_DEG:
            issues.append({"code": "HEAD_DROP", "t": float(t[head_min_i]), "value": round(drop, 1),
                           "severity": _severity(drop - HEAD_DROP_DEG),
                           "message": f"Head dropped {drop:.0f}° below your body line"})
        descent = float(t[b] - t[a])
        rng_deg = float(elbow[a] - elbow[b])
        d = elbow[a:b + 1]
        hi_cross = a + int(np.argmax(d < elbow[a] - 0.1 * rng_deg)) if rng_deg > 0 else a
        lo_cross = a + int(np.argmax(d < elbow[b] + 0.1 * rng_deg)) if rng_deg > 0 else b
        d1090 = float(t[lo_cross] - t[hi_cross])
        if d1090 < FAST_DESCENT_1090_S:
            est = d1090 / 0.6
            issues.append({"code": "FAST_DESCENT", "t": float(t[b]), "value": round(est, 2),
                           "severity": "med" if est < 0.3 else "low",
                           "message": f"Dropped in about {est:.2f} s (aim for about 1 s)"})
        reps.append({"n": n, "start": round(float(t[a]), 3), "bottom": round(float(t[b]), 3),
                     "end": round(float(t[c]), 3), "min_elbow": round(mn, 1), "max_elbow": round(mx, 1),
                     "max_hip_dev": round(max_dev, 1), "descent_s": round(descent, 2),
                     "ascent_s": round(float(t[c] - t[b]), 2), "issues": issues})

    def clean(x):
        return [None if np.isnan(v) else round(float(v), 2) for v in x]

    if confidence < 0.6:
        warnings.append("Tracking confidence is low: film from the side, whole body in frame.")
    return {**base, "side": "left" if s == "l" else "right", "confidence": confidence,
            "confidence_ok": confidence >= 0.6, "hip_baseline": round(hip_base, 1),
            "series": {"t": t.round(3).tolist(), "elbow": clean(elbow), "hip_dev": clean(hip_rel)},
            "reps": reps,
            "summary": {"rep_count": len(reps), "clean_reps": sum(not r["issues"] for r in reps),
                        "fatigue_from_rep": fatigue_from(reps)},
            "warnings": warnings}


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/landmarks_demo_side.json"
    a = analyze(json.load(open(path)))
    print(a["summary"], a["warnings"])
    for r in a["reps"]:
        print(r["n"], r["start"], r["bottom"], r["end"], r["min_elbow"], r["max_elbow"], r["max_hip_dev"],
              [i["code"] for i in r["issues"]])
