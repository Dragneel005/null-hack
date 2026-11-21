import json

import pytest

from vision.contract import validate_landmarks
from vision.extract import build_document
from vision.synthetic import generate_pushups


def test_build_document_from_raw_frames_is_contract_valid_with_imu():
    raw = generate_pushups(reps=3, seed=2)
    raw["frames"][10]["lm"] = None                      # a dropped detection
    doc = build_document(raw["frames"], raw["fps"], raw["w"], raw["h"])
    validate_landmarks(doc)
    assert doc["side"] == "left" and doc["message"] is None
    assert set(doc["imu"]["sensors"]) == {"chest", "hip", "wrist"}
    json.dumps(doc)                                     # serialisable


def test_overlay_renders_a_browser_playable_h264_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    import numpy as np

    from vision.overlay import render_overlay

    doc = generate_pushups(reps=1, jitter_px=0)
    src = tmp_path / "in.mp4"
    wr = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), doc["fps"],
                         (doc["w"], doc["h"]))
    for _ in doc["frames"]:
        wr.write(np.zeros((doc["h"], doc["w"], 3), np.uint8))
    wr.release()

    out = render_overlay(src, doc, tmp_path / "overlay.mp4", highlights={20: ["hip"]})
    cap = cv2.VideoCapture(str(out))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert fourcc in ("avc1", "h264") and n == len(doc["frames"])
