"""Skeleton overlay on the user's video (CONTRACT.md §5).

    render(video_path, landmarks, analysis=None, physics=None, out_path=None, progress=None) -> str

Draws the camera-side skeleton (green), turns the joints involved in an issue red while
that issue is happening, and adds a HUD (rep counter, elbow angle, current issue).
Returns a browser-playable H.264 mp4 path. Frames with no detected person pass through.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import cv2

import video

BONES = [("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
         ("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "foot"),
         ("ankle", "foot")]
ISSUE_JOINTS = {"HIP_SAG": {"hip"}, "HIP_PIKE": {"hip"}, "PIKE": {"hip"}, "SHALLOW": {"elbow"},
                "NO_LOCKOUT": {"elbow"}, "HEAD_DROP": {"ear"}, "FAST_DESCENT": {"shoulder", "elbow"}}
ISSUE_HALF_WINDOW_S = 0.35          # when an issue only has a time "t"
GREEN, RED, WHITE, DARK = (90, 210, 90), (40, 40, 235), (255, 255, 255), (25, 25, 25)


def _side(landmarks, analysis):
    if analysis and analysis.get("side") in ("left", "right"):
        return analysis["side"][0]
    ok = [f["lm"] for f in landmarks["frames"] if f.get("lm")]
    vis = {s: sum(lm[f"{s}_{j}"][2] for lm in ok for j in ("shoulder", "elbow", "hip"))
           for s in "lr"}
    return max(vis, key=vis.get)


def _issue_windows(analysis):
    out = []
    for rep in (analysis or {}).get("reps", []):
        for iss in rep.get("issues", []):
            t = float(iss.get("t", rep.get("bottom", 0)))
            t0 = float(iss.get("t_start", t - ISSUE_HALF_WINDOW_S))
            t1 = float(iss.get("t_end", t + ISSUE_HALF_WINDOW_S))
            out.append((t0, t1, ISSUE_JOINTS.get(iss.get("code"), set()),
                        iss.get("message") or iss.get("msg") or iss.get("code", "")))
    return out


def _angle(a, b, c):
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    return 0.0 if n == 0 else math.degrees(math.acos(max(-1, min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / n))))


def _draw(img, lm, s, red, hud_lines):
    h = img.shape[0]
    k = max(h / 720, 0.5)
    pt = {j: (int(round(lm[f"{s}_{j}"][0])), int(round(lm[f"{s}_{j}"][1])))
          for bone in BONES for j in bone}
    for a, b in BONES:
        cv2.line(img, pt[a], pt[b], RED if (a in red or b in red) else GREEN,
                 max(2, int(4 * k)), cv2.LINE_AA)
    for j, p in pt.items():
        cv2.circle(img, p, int((9 if j in red else 6) * k), RED if j in red else WHITE, -1,
                   cv2.LINE_AA)
    y = int(36 * k)
    for text, color in hud_lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, max(1, int(2 * k)))
        cv2.rectangle(img, (int(12 * k), y - th - int(8 * k)), (int(24 * k) + tw, y + int(8 * k)),
                      DARK, -1)
        cv2.putText(img, text, (int(18 * k), y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, color,
                    max(1, int(2 * k)), cv2.LINE_AA)
        y += int(th + 22 * k)


def render(video_path, landmarks, analysis=None, physics=None, out_path=None, progress=None) -> str:
    """Draw the overlay; returns the path of a browser-playable mp4 (default <video>_overlay.mp4)."""
    out_path = Path(out_path or Path(video_path).with_name(Path(video_path).stem + "_overlay.mp4"))
    frames = landmarks["frames"]
    s = _side(landmarks, analysis)
    windows = _issue_windows(analysis)
    starts = sorted(r["start"] for r in (analysis or {}).get("reps", []))
    total = len(starts)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or landmarks.get("fps", 30.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_name(out_path.stem + "_raw.mp4")     # same folder: no cross-drive moves
    writer = None
    for i, f in enumerate(frames):
        ok, img = cap.read()
        if not ok:
            break
        if writer is None:
            hh, ww = img.shape[:2]
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
        lm = f.get("lm")
        if lm:
            t = f["t"]
            active = [w for w in windows if w[0] <= t <= w[1]]
            red = set().union(*(w[2] for w in active)) if active else set()
            elbow = _angle(lm[f"{s}_shoulder"], lm[f"{s}_elbow"], lm[f"{s}_wrist"])
            hud = [(f"Rep {sum(st <= t for st in starts)}/{total}" if total else "Tracking", WHITE),
                   (f"Elbow {elbow:.0f} deg", WHITE)]
            hud += [(w[3], RED) for w in active[:2]]
            _draw(img, lm, s, red, hud)
        writer.write(img)
        if progress and i % 15 == 0:
            progress(min(1.0, (i + 1) / len(frames)))
    cap.release()
    if writer is None:
        raise ValueError(f"no readable frames in {video_path}")
    writer.release()
    web = video.to_browser_mp4(str(raw))
    shutil.move(web, out_path)
    os.remove(raw)
    if progress:
        progress(1.0)
    return str(out_path)
