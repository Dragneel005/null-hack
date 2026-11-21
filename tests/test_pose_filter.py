"""pose.smooth: single-frame jumps (loose fabric, occlusion) are removed; real motion is kept."""
import pytest

pytest.importorskip("mediapipe")
import pose  # noqa: E402


def _frames(xs):
    return [{"i": i, "t": i / 30, "lm": None if x is None else {"l_hip": [x, 100.0, 0.9]}, "world": None}
            for i, x in enumerate(xs)]


def test_single_frame_spike_is_removed():
    xs = [100.0] * 15
    xs[7] = 160.0
    out = pose.smooth(_frames(xs), window=5)
    assert abs(out[7]["lm"]["l_hip"][0] - 100.0) < 1.0


def test_real_movement_is_preserved():
    xs = [100.0 + 4 * i for i in range(20)]          # steady 4 px/frame motion
    out = pose.smooth(_frames(xs), window=5)
    assert all(abs(o["lm"]["l_hip"][0] - x) < 1e-6 for o, x in zip(out[3:-3], xs[3:-3]))


def test_missing_frames_stay_missing():
    xs = [100.0] * 10
    xs[4] = None
    out = pose.smooth(_frames(xs), window=5)
    assert out[4]["lm"] is None and out[5]["lm"]["l_hip"][0] == 100.0
