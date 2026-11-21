from vision.geometry import angle
from vision.synthetic import generate_pushups


def _side_angles(doc, a, b, c):
    s = doc["side"][0]  # "l" or "r"
    out = []
    for f in doc["frames"]:
        lm = f["lm"]
        out.append((f["t"], angle(lm[f"{s}_{a}"], lm[f"{s}_{b}"], lm[f"{s}_{c}"])))
    return out


def _count_reps(elbow, down=90, up=160):
    """Independent hysteresis counter: a rep = went below `down`, then back above `up`."""
    reps, state = 0, "up"
    for _, e in elbow:
        if state == "up" and e < down:
            state = "down"
        elif state == "down" and e > up:
            state, reps = "up", reps + 1
    return reps


def test_generates_requested_number_of_full_reps():
    doc = generate_pushups(reps=10, seed=1)
    assert _count_reps(_side_angles(doc, "shoulder", "elbow", "wrist")) == 10


def test_half_reps_do_not_reach_depth():
    doc = generate_pushups(reps=5, half_reps=[2, 4], jitter_px=0)
    assert _count_reps(_side_angles(doc, "shoulder", "elbow", "wrist")) == 3


def test_hip_sag_appears_only_from_the_requested_rep():
    doc = generate_pushups(reps=8, hip_sag_from_rep=6, hip_sag_deg=20, jitter_px=0)
    body = dict(_side_angles(doc, "shoulder", "hip", "ankle"))
    truth = {r["n"]: r for r in doc["truth"]["reps"]}
    dev_at_bottom = {n: 180 - body[min(body, key=lambda t: abs(t - r["bottom"]))]
                     for n, r in truth.items()}
    assert all(dev_at_bottom[n] < 5 for n in range(1, 6))
    assert all(dev_at_bottom[n] > 15 for n in range(6, 9))
    assert "HIP_SAG" in truth[7]["issues"] and "HIP_SAG" not in truth[2]["issues"]


def test_ground_truth_lists_every_rep_in_time_order():
    doc = generate_pushups(reps=4, half_reps=[3])
    reps = doc["truth"]["reps"]
    assert [r["n"] for r in reps] == [1, 2, 3, 4]
    assert all(r["start"] < r["bottom"] < r["end"] for r in reps)
    assert reps[2]["issues"] == ["SHALLOW"]
    assert doc["truth"]["rep_count"] == 4
