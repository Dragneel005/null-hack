"""Skeleton overlay video, re-encoded to browser-playable H.264.

pip's opencv-python cannot encode H.264, so we write mp4v and re-encode with the
ffmpeg binary bundled in imageio-ffmpeg (libx264, yuv420p, +faststart).
"""
import subprocess
import tempfile
from pathlib import Path

from .geometry import angle
from .landmarks import SIDE_BONES

GREEN, RED, WHITE = (80, 220, 80), (40, 40, 230), (255, 255, 255)


def _pt(v):
    return int(round(v[0])), int(round(v[1]))


def draw_frame(img, lm, side, red_joints=()):
    import cv2
    s = side[0]
    for a, b in SIDE_BONES:
        cv2.line(img, _pt(lm[f"{s}_{a}"]), _pt(lm[f"{s}_{b}"]), GREEN, 4, cv2.LINE_AA)
    for joint in {j for bone in SIDE_BONES for j in bone}:
        color = RED if joint in red_joints else WHITE
        cv2.circle(img, _pt(lm[f"{s}_{joint}"]), 7 if joint in red_joints else 5, color, -1,
                   cv2.LINE_AA)
    elbow = angle(lm[f"{s}_shoulder"], lm[f"{s}_elbow"], lm[f"{s}_wrist"])
    body = angle(lm[f"{s}_shoulder"], lm[f"{s}_hip"], lm[f"{s}_ankle"])
    cv2.putText(img, f"elbow {elbow:5.1f}  body {body:5.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    return img


def reencode_h264(src, dst):
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def render_overlay(video_path, doc, out_path, highlights=None):
    """highlights: optional {frame_index: ["hip", ...]} joints to paint red (from Person 2)."""
    import cv2
    from .extract import open_video

    highlights = highlights or {}
    cap = open_video(video_path)
    tmp = Path(tempfile.mkdtemp()) / "raw.mp4"
    writer = None
    for i, frame in enumerate(doc["frames"]):
        ok, img = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), doc["fps"], (w, h))
        if frame["lm"] is not None:
            draw_frame(img, frame["lm"], doc["side"], highlights.get(i, ()))
        writer.write(img)
    cap.release()
    if writer is None:
        raise RuntimeError("no frames rendered")
    writer.release()
    reencode_h264(tmp, out_path)
    return Path(out_path)
