"""Is this clip a push-up set, and where are the push-ups in it?

    detect(landmarks) -> {"is_pushup", "confidence", "reason", "segments": [(t0, t1)],
                          "cycles", "in_pushup": [bool per frame]}

A frame is in push-up posture when the body (shoulder->ankle) is near horizontal, the
wrists are below the shoulders and down at floor level with the feet. Posture runs of
>= 0.6 s are segments (this trims getting into position and standing up). A clip is a
push-up set if a segment contains at least one elbow bend-and-straighten cycle (a descent
that never comes back up counts, with low confidence); posture without either is a plank hold. Works on pose.extract and synth landmarks (pixels, y down).
"""
from __future__ import annotations

import math

MAX_TILT_DEG = 40          # body line vs horizontal
FLOOR_TOL = 0.35           # |wrist_y - ankle_y| / body length
MIN_SEGMENT_S = 0.6
MERGE_GAP_S = 0.4
CYCLE_DOWN, CYCLE_UP = 125.0, 145.0


def _side(frames):
    ok = [f["lm"] for f in frames if f.get("lm")]
    vis = {s: sum(lm[f"{s}_{j}"][2] for lm in ok for j in ("shoulder", "elbow", "wrist", "ankle"))
           for s in "lr"}
    return max(vis, key=vis.get)


def _angle(a, b, c):
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    return 180.0 if n == 0 else math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))))


def _posture(lm, s):
    sh, an, wr = lm[f"{s}_shoulder"], lm[f"{s}_ankle"], lm[f"{s}_wrist"]
    dx, dy = sh[0] - an[0], sh[1] - an[1]
    body = math.hypot(dx, dy) or 1.0
    tilt = math.degrees(math.atan2(abs(dy), abs(dx)))
    hands_down = wr[1] > sh[1]
    on_floor = abs(wr[1] - an[1]) / body < FLOOR_TOL
    return tilt, hands_down and on_floor and tilt < MAX_TILT_DEG


def _runs(flags, t):
    segs, start = [], None
    for i, ok in enumerate(flags):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            segs.append([t[start], t[i - 1]])
            start = None
    if start is not None:
        segs.append([t[start], t[-1]])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= MIN_SEGMENT_S]


def detect(landmarks: dict) -> dict:
    frames = landmarks["frames"]
    t = [f["t"] for f in frames]
    if not any(f.get("lm") for f in frames):
        return {"is_pushup": False, "confidence": 0.0, "reason": "no person detected",
                "segments": [], "cycles": 0, "in_pushup": [False] * len(frames)}
    s = _side(frames)
    flags, tilts = [], []
    for f in frames:
        if not f.get("lm"):
            flags.append(False)
            continue
        tilt, ok = _posture(f["lm"], s)
        flags.append(ok)
        tilts.append(tilt)
    segments = _runs(flags, t)

    cycles, descents, in_seg_ok, in_seg_n = 0, 0, 0, 0
    for a, b in segments:
        state = "up"
        for f, ok in zip(frames, flags):
            if not (a <= f["t"] <= b):
                continue
            in_seg_n += 1
            in_seg_ok += ok
            if not f.get("lm"):
                continue
            e = _angle(f["lm"][f"{s}_shoulder"], f["lm"][f"{s}_elbow"], f["lm"][f"{s}_wrist"])
            if state == "up" and e < CYCLE_DOWN:
                state, descents = "down", descents + 1
            elif state == "down" and e > CYCLE_UP:
                state, cycles = "up", cycles + 1
    in_pushup = [any(a <= ti <= b for a, b in segments) for ti in t]

    if not segments:
        median_tilt = sorted(tilts)[len(tilts) // 2]
        reason = ("body is upright, not horizontal" if median_tilt >= MAX_TILT_DEG
                  else "body is horizontal but hands are not on the floor under the shoulders")
        return {"is_pushup": False, "confidence": 0.1, "reason": reason, "segments": [],
                "cycles": 0, "in_pushup": in_pushup}
    stability = in_seg_ok / max(in_seg_n, 1)
    if cycles == 0 and descents:
        return {"is_pushup": True, "confidence": round(0.4 * stability, 3),
                "reason": f"push-up position with {descents} incomplete rep(s): went down, never came back up",
                "segments": segments, "cycles": 0, "in_pushup": in_pushup}
    if cycles == 0:
        return {"is_pushup": False, "confidence": round(0.3 * stability, 3),
                "reason": "plank hold, no reps", "segments": segments, "cycles": 0,
                "in_pushup": in_pushup}
    conf = round(0.5 * stability + 0.5 * min(1.0, cycles / 3), 3)
    dur = sum(b - a for a, b in segments)
    return {"is_pushup": True, "confidence": conf,
            "reason": f"push-up posture for {dur:.1f} s with {cycles} bend-and-straighten cycles",
            "segments": segments, "cycles": cycles, "in_pushup": in_pushup}


if __name__ == "__main__":
    import json
    import sys
    for path in sys.argv[1:] or ["samples/landmarks_demo_side.json", "samples/landmarks_demo_army.json"]:
        r = detect(json.load(open(path)))
        print(path, {k: v for k, v in r.items() if k != "in_pushup"})
