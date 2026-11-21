"""synth.py: simulated athletes -> labelled dataset in the pose.extract schema."""
import json
import zipfile

import numpy as np
import pytest

import synth

NAMES = ["nose", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
         "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
         "l_heel", "r_heel", "l_foot", "r_foot"]
CODES = {"SHALLOW", "NO_LOCKOUT", "HIP_SAG", "HIP_PIKE", "HEAD_DROP", "FAST_DESCENT"}


def _angle(a, b, c):
    ba, bc = np.subtract(a[:2], b[:2]), np.subtract(c[:2], b[:2])
    cos = ba @ bc / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _side(lms):
    ok = [f["lm"] for f in lms["frames"] if f["lm"]]
    vis = {s: np.mean([lm[f"{s}_{j}"][2] for lm in ok for j in ("shoulder", "elbow", "hip")])
           for s in "lr"}
    return max(vis, key=vis.get)


def _series(lms, a, b, c):
    s = _side(lms)
    return [(f["t"], _angle(f["lm"][f"{s}_{a}"], f["lm"][f"{s}_{b}"], f["lm"][f"{s}_{c}"]))
            for f in lms["frames"] if f["lm"]]


def _count(elbow, down=100, up=150):
    n, state = 0, "up"
    for _, e in elbow:
        if state == "up" and e < down:
            state = "down"
        elif state == "down" and e > up:
            state, n = "up", n + 1
    return n


def _hip_dev(lm, s):
    """Signed degrees: + = hip below the shoulder-ankle line (sag), - = above (pike)."""
    sh, hip, an = (np.array(lm[f"{s}_{j}"][:2]) for j in ("shoulder", "hip", "ankle"))
    dev = 180 - _angle(sh, hip, an)
    line_y = an[1] + (sh[1] - an[1]) * (hip[0] - an[0]) / (sh[0] - an[0])
    return dev if hip[1] > line_y else -dev       # image y points down


def _in_rep(series, rep):
    return [v for t, v in series if rep["start"] <= t <= rep["end"]]


# ---------------------------------------------------------------- schema

def test_samples_follow_the_pose_extract_schema():
    for s in synth.generate(n=3, seed=1):
        lms, labels = s["landmarks"], s["labels"]
        for key in ("source", "fps", "w", "h", "n_frames", "detected_frames", "frames"):
            assert key in lms
        assert lms["n_frames"] == len(lms["frames"])
        assert lms["detected_frames"] == sum(f["lm"] is not None for f in lms["frames"])
        assert [f["i"] for f in lms["frames"]] == list(range(lms["n_frames"]))
        f = next(f for f in lms["frames"] if f["lm"])
        assert set(f["lm"]) == set(NAMES) and set(f["world"]) == set(NAMES)
        assert all(len(v) == 3 for v in f["lm"].values())
        assert labels["rep_count"] == len(labels["reps"])
        assert all(set(r["faults"]) <= CODES for r in labels["reps"])
        json.dumps(s)


def test_generation_is_reproducible_from_the_seed():
    a, b = synth.generate(n=2, seed=5), synth.generate(n=2, seed=5)
    assert json.dumps(a) == json.dumps(b)
    assert json.dumps(a) != json.dumps(synth.generate(n=2, seed=6))


# ---------------------------------------------------------------- ground truth is real

def test_clean_reps_are_countable_and_match_labels():
    for s in synth.generate(n=5, seed=2, faults="none", camera="ideal"):
        assert _count(_series(s["landmarks"], "shoulder", "elbow", "wrist")) == \
            s["labels"]["rep_count"]


@pytest.mark.parametrize("code", sorted(CODES))
def test_each_injected_fault_is_measurable_in_the_labelled_reps(code):
    s = synth.generate(n=1, seed=3, faults=[code], camera="ideal")[0]
    lms, reps = s["landmarks"], s["labels"]["reps"]
    faulty = [r for r in reps if code in r["faults"]]
    clean = [r for r in reps if not r["faults"]]
    assert faulty and clean, "need both faulty and clean reps to compare"
    side = _side(lms)
    elbow = _series(lms, "shoulder", "elbow", "wrist")
    head = _series(lms, "ear", "shoulder", "hip")
    hip = [(f["t"], _hip_dev(f["lm"], side)) for f in lms["frames"] if f["lm"]]

    for r in faulty:
        if code == "SHALLOW":
            assert min(_in_rep(elbow, r)) > 95
        if code == "NO_LOCKOUT":
            assert max(_in_rep(elbow, r)) < 155
        if code == "HIP_SAG":
            assert max(_in_rep(hip, r)) > 12
        if code == "HIP_PIKE":
            assert min(_in_rep(hip, r)) < -12
        if code == "HEAD_DROP":
            assert min(_in_rep(head, r)) < 150
        if code == "FAST_DESCENT":
            assert r["bottom"] - r["start"] < 0.4
    for r in clean:
        assert min(_in_rep(elbow, r)) < 95
        assert max(abs(v) for v in _in_rep(hip, r)) < 8
        assert r["bottom"] - r["start"] > 0.4


def test_random_cameras_keep_the_body_in_frame():
    for s in synth.generate(n=6, seed=4, camera="random"):
        lms = s["landmarks"]
        pts = np.array([v[:2] for f in lms["frames"] if f["lm"] for v in f["lm"].values()])
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < lms["w"]) & (pts[:, 1] >= 0) & (pts[:, 1] < lms["h"])
        assert inside.mean() > 0.98


# ---------------------------------------------------------------- export

def test_export_writes_a_zip_with_jsonl_csv_and_datasheet(tmp_path):
    samples = synth.generate(n=2, seed=7)
    path = synth.export(samples, tmp_path)
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        assert {"samples.jsonl", "frames.csv", "reps.csv", "README.md"} <= names
        lines = z.read("samples.jsonl").decode().strip().splitlines()
        assert len(lines) == 2 and "labels" in json.loads(lines[0])
        header = z.read("frames.csv").decode().splitlines()[0]
        assert header.startswith("sample_id,i,t,") and "l_elbow_x" in header


# ---------------------------------------------------------------- clothing

def test_baggy_clothes_shift_torso_points_without_changing_the_truth():
    fitted = synth.generate(n=2, seed=9, faults="none", camera="ideal", clothing="fitted")
    baggy = synth.generate(n=2, seed=9, faults="none", camera="ideal", clothing="baggy")
    for f_s, b_s in zip(fitted, baggy):
        assert f_s["labels"]["reps"] == b_s["labels"]["reps"]          # same athlete, same reps
        assert b_s["labels"]["params"]["clothing"] == "baggy"
        side = f_s["labels"]["params"]["near_side"][0]
        pairs = [(f["lm"], b["lm"]) for f, b in zip(f_s["landmarks"]["frames"], b_s["landmarks"]["frames"])
                 if f["lm"] and b["lm"]]
        hip_drop = np.mean([b[f"{side}_hip"][1] - f[f"{side}_hip"][1] for f, b in pairs])
        assert hip_drop > 3                                               # hangs below the real hip (px)
        assert np.mean([b[f"{side}_hip"][2] for _, b in pairs]) < np.mean([f[f"{side}_hip"][2] for f, _ in pairs])
        assert all(f[f"{side}_wrist"] == b[f"{side}_wrist"] for f, b in pairs)   # hands unaffected


def test_existing_seeds_are_unchanged_by_the_clothing_option():
    assert json.dumps(synth.generate(n=2, seed=5)) == json.dumps(synth.generate(n=2, seed=5, clothing="fitted"))
