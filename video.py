"""Video file handling — the only module that knows about codecs and containers.

    save_upload(data, suffix, directory=None) -> (path, digest)
    to_browser_mp4(path) -> path   # H.264 / yuv420p / faststart; no-op if already H.264
    grab_frame(path, t, max_width=960) -> JPEG bytes | None

pip's opencv-python cannot encode H.264, so re-encoding uses an ffmpeg binary:
imageio-ffmpeg's bundled one if installed, else ffmpeg on PATH.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "outputs" / "uploads"          # outputs/ is gitignored
H264 = {"avc1", "h264", "H264", "AVC1"}


def save_upload(data: bytes, suffix: str = ".mp4", directory: str | Path | None = None):
    """Store uploaded bytes under a content hash, so re-uploads hit the same file/cache key."""
    digest = hashlib.sha1(data).hexdigest()[:16]
    d = Path(directory) if directory else UPLOADS
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{digest}{suffix if suffix.startswith('.') else '.' + suffix}"
    if not path.exists():
        path.write_bytes(data)
    return str(path), digest


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall back to PATH
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg")
        return exe


def _fourcc(path: str) -> str:
    cap = cv2.VideoCapture(str(path))
    code = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    cap.release()
    return code


def to_browser_mp4(path: str) -> str:
    """Return a browser-playable copy of `path` (<stem>_web.mp4), or `path` if it already is."""
    if Path(path).suffix.lower() == ".mp4" and _fourcc(path) in H264:
        return str(path)
    out = Path(path).with_name(Path(path).stem + "_web.mp4")
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(path), "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",          # libx264 needs even sizes
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return str(out)


def grab_frame(path: str, t: float, max_width: int = 960) -> bytes | None:
    """JPEG of the frame at time t (seconds), downscaled to max_width; None if t is past the end."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = int(round(t * fps))
    if n and idx >= n:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None
