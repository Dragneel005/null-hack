"""analysis.py: landmarks -> reps, faults, fatigue (CONTRACT.md §2)."""
import json
from pathlib import Path

import pytest

import analysis
import synth

ROOT = Path(__file__).resolve().parent.parent
CODES = {"SHALLOW", "NO_LOCKOUT", "HIP_SAG", "HIP_PIKE", "HEAD_DROP", "FAST_DESCENT"}


def _codes(rep):
    return {i["code"] for i in rep["issues"]}


def _match(truth, pred):
    """truth rep -> predicted rep with the most time overlap (or None)."""
    out = {}
    for t in truth:
        best, ov = None, 0.0
        for p in pred:
            o = min(t["end"], p["end"]) - max(t["start"], p["start"])
            if o > ov:
                best, ov = p, o
        out[t["n"]] = best
    return out


def test_output_has_every_field_the_app_and_coach_read():
    s = synth.generate(n=1, seed=0, faults="mixed", camera="ideal")[0]
    a = analysis.analyze(s["landmarks"])
    for k in ("exercise", "standard", "side", "confidence", "thresholds", "series", "reps", "summary"):
        assert k in a
    assert set(a["thresholds"]) >= {"depth_max_elbow", "lockout_min_elbow", "hip_tolerance"}
    n = s["landmarks"]["n_frames"]
    assert all(len(a["series"][k]) == n for k in ("t", "elbow", "hip_dev"))
    for r in a["reps"]:
        assert {"n", "start", "bottom", "end", "min_elbow", "max_elbow", "max_hip_dev", "issues"} <= set(r)
        for i in r["issues"]:
            assert {"code", "t", "value", "severity", "message"} <= set(i) and i["code"] in CODES
    assert set(a["summary"]) >= {"rep_count", "clean_reps", "fatigue_from_rep"}
    json.dumps(a)


def test_counts_clean_reps_exactly():
    for s in synth.generate(n=6, seed=1, faults="none", camera="ideal"):
        a = analysis.analyze(s["landmarks"])
        assert a["summary"]["rep_count"] == s["labels"]["rep_count"]
        assert a["summary"]["clean_reps"] == s["labels"]["rep_count"]


@pytest.mark.parametrize("code", sorted(CODES))
def test_each_fault_is_found_on_the_right_reps(code):
    s = synth.generate(n=1, seed=3, faults=[code], camera="ideal")[0]
    a = analysis.analyze(s["landmarks"])
    assert a["summary"]["rep_count"] == s["labels"]["rep_count"]
    m = _match(s["labels"]["reps"], a["reps"])
    for t in s["labels"]["reps"]:
        assert m[t["n"]] is not None
        assert (code in _codes(m[t["n"]])) == (code in t["faults"]), f"rep {t['n']}"


def test_baggy_clothes_do_not_create_false_hip_sag():
    for s in synth.generate(n=8, seed=21, faults="none", camera="ideal", clothing="baggy"):
        a = analysis.analyze(s["landmarks"])
        assert not any("HIP_SAG" in _codes(r) for r in a["reps"])


def test_fatigue_is_the_first_rep_of_a_persistent_breakdown():
    s = synth.generate(n=1, seed=0, faults="none", camera="ideal")[0]
    a = analysis.analyze(s["landmarks"])
    assert a["summary"]["fatigue_from_rep"] is None
    reps = [{"n": i, "issues": [] if i < 4 else [{"code": "HIP_SAG"}]} for i in range(1, 8)]
    assert analysis.fatigue_from(reps) == 4
    assert analysis.fatigue_from([{"n": 1, "issues": [{"code": "SHALLOW"}]}, {"n": 2, "issues": []}]) is None


def test_strict_standard_demands_more_depth():
    s = synth.generate(n=1, seed=2, faults="none", camera="ideal")[0]
    loose, strict = analysis.analyze(s["landmarks"]), analysis.analyze(s["landmarks"], strict=True)
    assert strict["thresholds"]["depth_max_elbow"] < loose["thresholds"]["depth_max_elbow"]
    assert strict["standard"] != loose["standard"]


def test_not_a_pushup_gives_no_reps_and_says_why():
    s = synth.generate(n=1, seed=0, faults="none", camera="ideal")[0]
    for f in s["landmarks"]["frames"]:
        f["lm"] = None
    a = analysis.analyze(s["landmarks"])
    assert a["summary"]["rep_count"] == 0 and a["warnings"]


def test_real_beach_clip_has_six_reps():
    lms = json.loads((ROOT / "samples" / "landmarks_demo_side.json").read_text())
    a = analysis.analyze(lms)
    assert a["summary"]["rep_count"] == 6


def test_real_baggy_clothes_clip_has_fifteen_reps_and_ignores_kneeling_after():
    """Side-view set in a loose hoodie and trousers; 15 reps counted by hand, then she kneels up."""
    lms = json.loads((ROOT / "samples" / "landmarks_baggy_15reps.json").read_text())
    a = analysis.analyze(lms)
    assert a["summary"]["rep_count"] == 15
    hand_counted_bottoms = [1.0, 2.6, 4.1, 5.5, 7.0, 8.4, 9.9, 11.4, 12.9, 14.3, 15.9, 17.4, 19.1, 20.8, 22.5]
    for r, b in zip(a["reps"], hand_counted_bottoms):
        assert abs(r["bottom"] - b) < 0.4, (r["n"], r["bottom"], b)
    assert a["reps"][-1]["end"] < 24.0                 # kneeling from ~24 s is not a rep
    assert not any(i["code"] == "HIP_SAG" for r in a["reps"] for i in r["issues"])
