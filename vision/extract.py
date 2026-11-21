"""Video -> landmarks.json (+ simulated IMU) using MediaPipe Pose Landmarker (Tasks API).

Note: the legacy `mp.solutions.pose` API is removed in mediapipe >= 0.10.30; this uses
`mp.tasks.vision.PoseLandmarker` in VIDEO mode. See vision/RESEARCH.md.
"""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .contract import validate_landmarks
from .imu import add_imu
from .landmarks import NAMES
from .quality import assess
from .smoothing import smooth_frames
from .timestamps import frame_timestamps_ms

MODELS_DIR = Path(__file__).parent / "models"
MODEL_URLS = {
    v: [f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}"
        f"/float16/latest/pose_landmarker_{v}.task",
        f"https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-{v.title()}"
        f"/resolve/main/pose_landmarker_{v}.task"]
    for v in ("lite", "full", "heavy")
}


def ensure_model(variant: str = "full") -> Path:
    path = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if path.exists() and path.stat().st_size > 0:
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in MODEL_URLS[variant]:
        try:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as e:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError("could not download pose model:\n" + "\n".join(errors))


def open_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)   # apply phone rotation metadata
    return cap


def run_pose(video_path, model_variant="full", max_seconds=30.0):
    """Return (raw_frames, fps, w, h, m_per_px) with pixel-coordinate landmarks."""
    import cv2
    import mediapipe as mp

    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model(model_variant))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    )
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps)
    stamps = frame_timestamps_ms(max_frames + 1, fps)

    frames, scales, w, h = [], [], 0, 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i in range(max_frames):
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]                      # trust the decoded (rotated) shape
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamps[i])
            t = round((i + 1) / fps, 4)
            if not res.pose_landmarks:
                frames.append({"t": t, "lm": None})
                continue
            pts = res.pose_landmarks[0]
            lm = {name: [round(p.x * w, 2), round(p.y * h, 2), round(p.visibility, 3)]
                  for name, p in zip(NAMES, pts)}
            frames.append({"t": t, "lm": lm})
            if res.pose_world_landmarks:
                wl = res.pose_world_landmarks[0]
                for a, b in ((11, 27), (12, 28)):     # shoulder -> ankle, metres vs pixels
                    m = np.linalg.norm([wl[a].x - wl[b].x, wl[a].y - wl[b].y])
                    px = np.linalg.norm([(pts[a].x - pts[b].x) * w, (pts[a].y - pts[b].y) * h])
                    if px > 50:
                        scales.append(m / px)
    cap.release()
    m_per_px = float(np.median(scales)) if scales else None
    return frames, fps, w, h, m_per_px


def build_document(raw_frames, fps, w, h, m_per_px=None, smooth_window=5,
                   imu_noise=True, seed=0, source="video"):
    frames = smooth_frames(raw_frames, window=smooth_window)
    q = assess(frames)
    doc = {"version": 1, "fps": fps, "w": w, "h": h, "side": q["side"],
           "confidence": q["confidence"], "detected_fraction": q["detected_fraction"],
           "message": q["message"], "source": source, "frames": frames}
    if m_per_px:
        doc["m_per_px"] = m_per_px
    doc = add_imu(doc, noise=imu_noise, seed=seed)
    validate_landmarks(doc)
    return doc


def extract(video_path, out_dir, model_variant="full", max_seconds=30.0, imu_noise=True):
    start = time.time()
    raw, fps, w, h, m_per_px = run_pose(video_path, model_variant, max_seconds)
    doc = build_document(raw, fps, w, h, m_per_px, imu_noise=imu_noise,
                         source=Path(video_path).name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "landmarks.json").write_text(json.dumps(doc))
    doc["_seconds"] = round(time.time() - start, 1)
    return doc
