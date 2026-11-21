import math

from vision.landmarks import NAMES
from vision.quality import assess
from vision.smoothing import smooth_frames
from vision.timestamps import frame_timestamps_ms


# ---- timestamps -------------------------------------------------------------

def test_timestamps_are_strictly_increasing_ints_even_at_odd_fps():
    ts = frame_timestamps_ms(1000, 2000.0)     # 0.5 ms/frame would collide when truncated
    assert all(isinstance(t, int) for t in ts)
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_timestamps_at_30fps():
    assert frame_timestamps_ms(4, 30) == [0, 33, 66, 100]


# ---- helpers ----------------------------------------------------------------

def _frame(t, x, vis_l=0.9, vis_r=0.3):
    return {"t": t, "lm": {n: [x, 100.0, vis_l if n.startswith("l_") else
                               (vis_r if n.startswith("r_") else 0.9)] for n in NAMES}}


# ---- smoothing --------------------------------------------------------------

def test_smoothing_removes_a_single_frame_spike_and_keeps_length():
    frames = [_frame(i / 30, 100.0) for i in range(9)]
    frames[4]["lm"]["l_elbow"][0] = 150.0
    out = smooth_frames(frames, window=5)
    assert len(out) == 9
    assert abs(out[4]["lm"]["l_elbow"][0] - 110.0) < 1e-6   # (100*4 + 150) / 5


def test_smoothing_skips_missing_frames_instead_of_propagating():
    frames = [_frame(i / 30, 100.0) for i in range(5)]
    frames[2]["lm"] = None
    out = smooth_frames(frames, window=3)
    assert out[2]["lm"] is None
    assert out[1]["lm"]["l_hip"][0] == 100.0


def test_smoothing_does_not_modify_input():
    frames = [_frame(i / 30, float(i)) for i in range(5)]
    smooth_frames(frames, window=3)
    assert frames[0]["lm"]["l_hip"][0] == 0.0


# ---- quality ----------------------------------------------------------------

def test_picks_the_more_visible_side():
    assert assess([_frame(i, 0, vis_l=0.2, vis_r=0.9) for i in range(10)])["side"] == "right"
    assert assess([_frame(i, 0, vis_l=0.9, vis_r=0.2) for i in range(10)])["side"] == "left"


def test_good_clip_has_high_confidence_and_no_message():
    q = assess([_frame(i, 0, vis_l=0.92) for i in range(10)])
    assert math.isclose(q["confidence"], 0.92, abs_tol=1e-6)
    assert q["message"] is None


def test_poorly_visible_body_gets_an_actionable_message():
    q = assess([_frame(i, 0, vis_l=0.3, vis_r=0.2) for i in range(10)])
    assert q["confidence"] < 0.5
    assert "side" in q["message"].lower()


def test_mostly_missing_pose_gets_a_message():
    frames = [_frame(i, 0) for i in range(10)]
    for f in frames[:7]:
        f["lm"] = None
    assert assess(frames)["message"] is not None
