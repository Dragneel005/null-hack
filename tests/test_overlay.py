"""overlay.py: skeleton + fault highlights drawn on the user's video, browser-playable."""
import cv2
import numpy as np
import pytest

import overlay
import synth


@pytest.fixture
def clip(tmp_path):
    s = synth.generate(n=1, seed=0, faults="none", camera="ideal")[0]
    lms = s["landmarks"]
    path = tmp_path / "in.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), lms["fps"],
                         (lms["w"], lms["h"]))
    for _ in lms["frames"]:
        wr.write(np.zeros((lms["h"], lms["w"], 3), np.uint8))
    wr.release()
    rep = s["labels"]["reps"][0]
    analysis = {"side": s["labels"]["params"]["near_side"],
                "reps": [{"n": 1, "start": rep["start"], "bottom": rep["bottom"], "end": rep["end"],
                          "issues": [{"code": "HIP_SAG", "t": rep["bottom"], "value": 16,
                                      "severity": "med", "message": "Hips sagged 16°"}]}]}
    return path, lms, analysis, tmp_path


def _frame(path, i):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    cap.release()
    assert ok
    return img


def _joint_px(img, lms, i, name):
    x, y = (int(round(v)) for v in lms["frames"][i]["lm"][name][:2])
    patch = img[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).mean(axis=0)
    return patch  # B, G, R


def test_output_is_browser_playable_and_keeps_every_frame(clip):
    path, lms, analysis, tmp = clip
    out = overlay.render(str(path), lms, analysis, out_path=str(tmp / "o.mp4"))
    cap = cv2.VideoCapture(out)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert fourcc in ("avc1", "h264") and n == lms["n_frames"]


def test_faulty_joint_turns_red_only_during_the_issue(clip):
    path, lms, analysis, tmp = clip
    out = overlay.render(str(path), lms, analysis, out_path=str(tmp / "o.mp4"))
    hip = f"{analysis['side'][0]}_hip"
    during = int(round(analysis["reps"][0]["issues"][0]["t"] * lms["fps"]))
    before = 2
    b, g, r = _joint_px(_frame(out, during), lms, during, hip)
    assert r > 150 and g < 110, "hip should be red during HIP_SAG"
    b, g, r = _joint_px(_frame(out, before), lms, before, hip)
    assert r > 150 and g > 150, "hip should be white when there is no issue"


def test_frames_without_a_person_are_passed_through(clip):
    path, lms, analysis, tmp = clip
    for f in lms["frames"][:5]:
        f["lm"] = None
    out = overlay.render(str(path), lms, analysis, out_path=str(tmp / "o.mp4"))
    assert _frame(out, 1).max() < 40          # untouched black frame (codec noise only)


def test_works_without_analysis(clip):
    path, lms, _, tmp = clip
    assert overlay.render(str(path), lms, None, out_path=str(tmp / "o.mp4"))
