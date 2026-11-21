"""Stage 1 — Vision: video file -> landmarks dict (see CONTRACT.md).

Runs MediaPipe Pose Landmarker (Tasks API, VIDEO mode) on every frame,
keeps the most prominent person, smooths jitter, and returns pixel-space
landmarks + 3D world landmarks (metres, hip-centred) per frame.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
MODELS = {k: ROOT / "models" / f"pose_landmarker_{k}.task" for k in ("lite", "full", "heavy")}

# MediaPipe's 33 landmark indices -> the names we use everywhere.
NAMES = {
    0: "nose", 7: "l_ear", 8: "r_ear",
    11: "l_shoulder", 12: "r_shoulder", 13: "l_elbow", 14: "r_elbow",
    15: "l_wrist", 16: "r_wrist", 23: "l_hip", 24: "r_hip",
    25: "l_knee", 26: "r_knee", 27: "l_ankle", 28: "r_ankle",
    29: "l_heel", 30: "r_heel", 31: "l_foot", 32: "r_foot",
}


def _pick_person(poses, w, h):
    """Choose the most prominent person: biggest bounding box x visibility."""
    best, best_score = None, -1.0
    for lms in poses:
        xs = np.array([l.x for l in lms]) * w
        ys = np.array([l.y for l in lms]) * h
        vis = np.array([getattr(l, "visibility", 1.0) or 0.0 for l in lms])
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        score = area * float(vis.mean())
        if score > best_score:
            best, best_score = lms, score
    return best


def smooth(frames: list[dict], window: int = 5) -> list[dict]:
    """Centered moving average over valid frames (per landmark, x/y only)."""
    if window <= 1:
        return frames
    half = window // 2
    out = []
    for i, f in enumerate(frames):
        if f["lm"] is None:
            out.append(f)
            continue
        nf = {**f, "lm": {}}
        for name, (x, y, v) in f["lm"].items():
            xs, ys = [], []
            for j in range(max(0, i - half), min(len(frames), i + half + 1)):
                g = frames[j]["lm"]
                if g is not None and name in g:
                    xs.append(g[name][0]); ys.append(g[name][1])
            nf["lm"][name] = [float(np.mean(xs)), float(np.mean(ys)), v]
        out.append(nf)
    return out


def extract(video_path: str, model: str = "full", max_seconds: float = 90.0,
            max_width: int = 960, smooth_window: int = 5, progress=None) -> dict:
    """Run pose estimation over a video. `progress(fraction)` is called if given."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps < 240.0):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(max_seconds * fps)
    if total > 0:
        limit = min(limit, total)

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS[model])),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frames, w, h, idx = [], None, None, 0
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        while idx < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if w is None:
                h, w = frame.shape[:2]
                scale = min(1.0, max_width / w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ts_ms = int(idx * 1000 / fps)  # must be strictly increasing
            res = landmarker.detect_for_video(image, ts_ms)
            rec = {"i": idx, "t": round(idx / fps, 4), "lm": None, "world": None}
            if res.pose_landmarks:
                k = 0
                if len(res.pose_landmarks) > 1:
                    chosen = _pick_person(res.pose_landmarks, w, h)
                    k = next(j for j, p in enumerate(res.pose_landmarks) if p is chosen)
                lms = res.pose_landmarks[k]
                rec["lm"] = {
                    n: [float(lms[i].x * w), float(lms[i].y * h),
                        float(getattr(lms[i], "visibility", 1.0) or 0.0)]
                    for i, n in NAMES.items()
                }
                if res.pose_world_landmarks and len(res.pose_world_landmarks) > k:
                    wl = res.pose_world_landmarks[k]
                    rec["world"] = {n: [float(wl[i].x), float(wl[i].y), float(wl[i].z)]
                                    for i, n in NAMES.items()}
            frames.append(rec)
            idx += 1
            if progress and limit and idx % 10 == 0:
                progress(min(1.0, idx / limit))
    cap.release()
    if w is None:
        raise ValueError("Video contained no readable frames")
    detected = sum(1 for f in frames if f["lm"] is not None)
    return {
        "source": os.path.basename(str(video_path)),
        "fps": float(fps), "w": int(w), "h": int(h),
        "n_frames": len(frames), "detected_frames": detected,
        "frames": smooth(frames, smooth_window),
    }



def render_overlay(video_path: str, landmarks: dict, analysis: dict | None, out_path: str | None = None,
                   progress=None) -> str:
    """Alias used by app.py; see overlay.render (CONTRACT.md §5)."""
    import overlay
    return overlay.render(video_path, landmarks, analysis, out_path=out_path, progress=progress)


if __name__ == "__main__":
    import time
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "landmarks.json"
    t0 = time.time()
    data = extract(src, model=os.environ.get("POSE_MODEL", "full"))
    Path(dst).write_text(json.dumps(data))
    print(f"{data['n_frames']} frames, {data['detected_frames']} with a person, "
          f"{data['w']}x{data['h']} @ {data['fps']:.1f}fps -> {dst} in {time.time()-t0:.1f}s")
