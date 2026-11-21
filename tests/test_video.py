"""video.py: the one module that touches video files (uploads, codecs, frame grabs)."""
import cv2
import numpy as np
import pytest

import video


def _fourcc(path):
    cap = cv2.VideoCapture(str(path))
    code = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    cap.release()
    return code


@pytest.fixture
def mp4v_clip(tmp_path):
    """2 s, 30 fps clip in OpenCV's default (not browser-playable) mp4v codec."""
    path = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(60):
        img = np.full((240, 320, 3), i * 4, np.uint8)
        wr.write(img)
    wr.release()
    return path


def test_save_upload_is_content_addressed(tmp_path):
    p1, d1 = video.save_upload(b"same bytes", ".mp4", directory=tmp_path)
    p2, d2 = video.save_upload(b"same bytes", ".mp4", directory=tmp_path)
    p3, d3 = video.save_upload(b"other", ".mov", directory=tmp_path)
    assert p1 == p2 and d1 == d2 and d1 != d3
    assert open(p1, "rb").read() == b"same bytes" and p3.endswith(".mov")


def test_to_browser_mp4_reencodes_mp4v_to_h264(mp4v_clip):
    assert _fourcc(mp4v_clip) not in ("avc1", "h264")
    out = video.to_browser_mp4(str(mp4v_clip))
    assert out != str(mp4v_clip) and _fourcc(out) in ("avc1", "h264")
    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 60
    cap.release()


def test_to_browser_mp4_leaves_h264_files_alone(mp4v_clip):
    once = video.to_browser_mp4(str(mp4v_clip))
    assert video.to_browser_mp4(once) == once


def test_grab_frame_returns_jpeg_bytes_at_time(mp4v_clip):
    jpg = video.grab_frame(str(mp4v_clip), 1.0)
    assert jpg[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert abs(int(img.mean()) - 30 * 4) <= 6          # frame 30 has brightness 120


def test_grab_frame_past_the_end_returns_none(mp4v_clip):
    assert video.grab_frame(str(mp4v_clip), 99.0) is None
