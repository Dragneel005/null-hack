"""activity.py: is this clip a push-up set, and where are the push-ups in it?"""
import copy
import math

import activity
import synth


def _sample(seed=0, **kw):
    return synth.generate(n=1, seed=seed, faults="none", camera="ideal", **kw)[0]


def _rotate(landmarks, deg):
    """Rotate every landmark around the first ankle: turns a plank into a standing body."""
    out = copy.deepcopy(landmarks)
    f0 = next(f for f in out["frames"] if f["lm"])
    side = "r" if f0["lm"]["r_ankle"][2] > f0["lm"]["l_ankle"][2] else "l"
    ax, ay = f0["lm"][f"{side}_ankle"][:2]
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    for f in out["frames"]:
        if f["lm"]:
            for k, (x, y, v) in f["lm"].items():
                dx, dy = x - ax, y - ay
                f["lm"][k] = [ax + c * dx - s * dy, ay + s * dx + c * dy, v]
    return out


def test_simulated_pushup_sets_are_recognised():
    for seed in range(4):
        r = activity.detect(_sample(seed)["landmarks"])
        assert r["is_pushup"], r["reason"]
        assert r["confidence"] > 0.6


def test_same_arm_motion_while_upright_is_not_a_pushup():
    r = activity.detect(_rotate(_sample(1)["landmarks"], -75))
    assert not r["is_pushup"]
    assert "upright" in r["reason"] or "horizontal" in r["reason"]


def test_plank_hold_without_reps_is_not_a_pushup_set():
    s = _sample(2)
    lms = s["landmarks"]
    hold = next(f for f in lms["frames"] if f["t"] < s["labels"]["reps"][0]["start"] - 0.1)
    lms["frames"] = [{**copy.deepcopy(hold), "i": i, "t": i / lms["fps"]} for i in range(90)]
    r = activity.detect(lms)
    assert not r["is_pushup"] and "no reps" in r["reason"]


def test_segments_cover_the_reps_and_exclude_standing_up():
    s = _sample(3)
    lms = s["landmarks"]
    standing = _rotate(lms, -75)["frames"][:45]           # 1.5 s standing before the set
    n = len(standing)
    lms["frames"] = [{**f, "i": i, "t": i / lms["fps"]} for i, f in enumerate(standing)] + \
        [{**f, "i": f["i"] + n, "t": f["t"] + n / lms["fps"]} for f in lms["frames"]]
    r = activity.detect(lms)
    assert r["is_pushup"]
    seg_start = r["segments"][0][0]
    first_rep = s["labels"]["reps"][0]["start"] + n / lms["fps"]
    assert 1.2 <= seg_start <= first_rep
    last_rep_end = s["labels"]["reps"][-1]["end"] + n / lms["fps"]
    assert r["segments"][-1][1] >= last_rep_end - 0.1


def test_empty_or_missing_people_are_handled():
    lms = _sample(0)["landmarks"]
    for f in lms["frames"]:
        f["lm"] = None
    r = activity.detect(lms)
    assert not r["is_pushup"] and "no person" in r["reason"]


def test_a_descent_that_never_comes_back_up_still_counts_as_pushup_footage():
    s = _sample(4)
    lms = s["landmarks"]
    cut = s["labels"]["reps"][0]["bottom"] + 0.2          # clip ends while at the bottom
    lms["frames"] = [f for f in lms["frames"] if f["t"] <= cut]
    r = activity.detect(lms)
    assert r["is_pushup"] and r["cycles"] == 0
    assert "incomplete" in r["reason"] and r["confidence"] < 0.6
