"""Plain-assert tests for analysis.py. Run: .venv/bin/python tests/test_analysis.py"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402

NAMES = ["nose", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
         "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel", "l_foot", "r_foot"]


def _pose(theta, sag=0.0, head_drop=0.0, upright=False):
    """2D side-view push-up skeleton (facing left, y down) for elbow angle theta (deg)."""
    s = 300.0  # px per metre
    u, f = 0.30 * s, 0.28 * s
    floor = 400.0
    W = np.array([200.0, floor])
    d = math.sqrt(u * u + f * f - 2 * u * f * math.cos(math.radians(theta)))
    S = W + np.array([0.0, -d])
    # elbow: circle intersection, placed toward the feet (+x)
    a = (u * u - f * f + d * d) / (2 * d)
    h = math.sqrt(max(u * u - a * a, 0.0))
    E = S + np.array([h, a])
    An = np.array([200.0 + 1.35 * s, floor - 0.08 * s])
    dirv = (An - S) / np.linalg.norm(An - S)
    nrm = np.array([-dirv[1], dirv[0]])
    if nrm[1] < 0:
        nrm = -nrm
    L = np.linalg.norm(An - S)
    H = S + dirv * L * 0.38 + nrm * math.tan(math.radians(sag)) * L * 0.38
    back = -dirv  # from shoulder towards head
    rot = math.radians(head_drop)
    c, si = math.cos(rot), math.sin(rot)
    # rotate 'back' toward the floor side of the body line by head_drop
    hd = back * c + nrm * si
    Ear = S + hd * 0.2 * s
    Kn = (H + An) / 2
    if upright:
        S, E, W = np.array([300, 100.0]), np.array([300, 160.0]), np.array([300, 220.0])
        H, Kn, An, Ear = np.array([300, 250.0]), np.array([300, 330.0]), np.array([300, 410.0]), np.array([300, 60.0])
    pts = {"shoulder": S, "elbow": E, "wrist": W, "hip": H, "knee": Kn, "ankle": An, "heel": An + [5, 0],
           "foot": An + [-5, 5], "ear": Ear}
    lm = {}
    for side, vis in (("l_", 0.95), ("r_", 0.5)):
        for k, v in pts.items():
            lm[side + k] = [float(v[0]), float(v[1]), vis]
    lm["nose"] = [float(Ear[0] - 15), float(Ear[1] + 5), 0.9]
    return lm


def make_clip(reps=5, fps=30.0, top=168.0, bottom=70.0, desc=1.0, asc=0.8, hold_top=1.0, hold_bot=0.2,
              per_rep=None, upright=False, mirror=False, drop_every=0, w=640, h=480):
    """per_rep: optional list of dicts overriding bottom/top/desc/asc/sag/head_drop per rep."""
    seq = []  # (theta, sag, head)
    seq += [(top, 0, 0)] * int(hold_top * fps)
    for r in range(reps):
        o = (per_rep[r] if per_rep and r < len(per_rep) else {}) or {}
        b, tp = o.get("bottom", bottom), o.get("top", top)
        de, as_ = o.get("desc", desc), o.get("asc", asc)
        sag, hdp = o.get("sag", 0.0), o.get("head", 0.0)
        nd, na = max(2, int(de * fps)), max(2, int(as_ * fps))
        prev = seq[-1][0]
        for k in range(nd):
            x = (1 - math.cos(math.pi * k / nd)) / 2
            seq.append((prev + (b - prev) * x, sag * x, hdp * x))
        seq += [(b, sag, hdp)] * int(o.get("hold_bot", hold_bot) * fps)
        for k in range(na):
            x = (1 - math.cos(math.pi * k / na)) / 2
            seq.append((b + (tp - b) * x, sag * (1 - x), hdp * (1 - x)))
        seq += [(tp, 0, 0)] * int(o.get("hold", hold_top) * fps)
    frames = []
    for i, (th, sg, hdp) in enumerate(seq):
        lm = _pose(th, sg, hdp, upright)
        if mirror:
            lm = {k: [w - v[0], v[1], v[2]] for k, v in lm.items()}
        if drop_every and i % drop_every in (0, 1, 2):
            lm = None
        frames.append({"i": i, "t": round(i / fps, 4), "lm": lm, "world": None})
    return {"source": "synthetic", "fps": fps, "w": w, "h": h, "n_frames": len(frames),
            "detected_frames": sum(f["lm"] is not None for f in frames), "frames": frames}


def check_schema(res, n_frames):
    for k in ["exercise", "side", "confidence", "confidence_ok", "warnings", "thresholds", "series", "reps", "summary"]:
        assert k in res, k
    for k in ["t", "elbow", "body_line", "hip_dev", "head", "phase"]:
        assert len(res["series"][k]) == n_frames, k
    for k in ["rep_count", "clean_reps", "avg_score", "fatigue_from_rep", "issue_counts", "avg_descent_s",
              "avg_ascent_s", "total_time_s"]:
        assert k in res["summary"], k
    for r in res["reps"]:
        for k in ["n", "start", "bottom", "end", "min_elbow", "max_elbow", "max_hip_sag", "max_hip_pike",
                  "min_head", "descent_s", "ascent_s", "score", "issues"]:
            assert k in r, k
        assert r["start"] < r["bottom"] < r["end"]
        for i in r["issues"]:
            assert i["code"] in A.FAULT_CODES and i["severity"] in ("low", "med", "high")
    json.dumps(res)  # must be JSON serialisable


def codes(res, n):
    return {i["code"] for i in res["reps"][n - 1]["issues"]}


def main():
    # helpers
    assert abs(A.angle([1, 0], [0, 0], [0, 1]) - 90) < 1e-6
    assert set(A.FAULT_INFO) == set(A.FAULT_CODES)
    assert all({"title", "cue", "why"} <= set(v) for v in A.FAULT_INFO.values())

    # real sample: ground truth 6 reps
    side = json.loads((ROOT / "samples/landmarks_demo_side.json").read_text())
    r = A.analyze(side)
    check_schema(r, len(side["frames"]))
    print(f"[real side]  reps={r['summary']['rep_count']} side={r['side']} conf={r['confidence']} "
          f"ok={r['confidence_ok']} avg_score={r['summary']['avg_score']} issues={r['summary']['issue_counts']}")
    assert r["summary"]["rep_count"] == 6
    assert r["confidence_ok"] and r["side"] == "left"
    assert r["summary"]["avg_score"] >= 85

    # real messy sample: must not crash
    army = json.loads((ROOT / "samples/landmarks_demo_army.json").read_text())
    r = A.analyze(army)
    check_schema(r, len(army["frames"]))
    print(f"[real army]  reps={r['summary']['rep_count']} warnings={len(r['warnings'])}: {r['warnings']}")
    assert any("cut" in w for w in r["warnings"])

    # synthetic clean, slow + fast, mirrored, dropouts
    for name, kw, exp in [
        ("clean slow", dict(reps=5), 5),
        ("fast", dict(reps=10, desc=0.6, asc=0.45, hold_top=0.1, hold_bot=0.0), 10),
        ("mirrored", dict(reps=5, mirror=True), 5),
        ("dropouts", dict(reps=6, drop_every=17), 6),
    ]:
        clip = make_clip(**kw)
        r = A.analyze(clip)
        check_schema(r, len(clip["frames"]))
        print(f"[synth {name:10s}] reps={r['summary']['rep_count']} (exp {exp}) issues={r['summary']['issue_counts']}")
        assert r["summary"]["rep_count"] == exp, (name, r["summary"]["rep_count"])
        assert not r["summary"]["issue_counts"], (name, r["summary"]["issue_counts"])

    # fault injection (and mirrored version must give identical fault codes)
    per = [{}, {"sag": 25}, {"sag": -25}, {"bottom": 118}, {"top": 135}, {"desc": 0.25, "hold_bot": 0.0},
           {"head": 75}, {}]
    expect = {2: "HIP_SAG", 3: "HIP_PIKE", 4: "SHALLOW", 5: "NO_LOCKOUT", 6: "FAST_DESCENT", 7: "HEAD_DROP"}
    for mirror in (False, True):
        clip = make_clip(reps=8, per_rep=per, mirror=mirror)
        r = A.analyze(clip)
        got = {k: codes(r, k) for k in range(1, r["summary"]["rep_count"] + 1)}
        print(f"[synth faults mirror={mirror}] reps={r['summary']['rep_count']} per-rep={got}")
        assert r["summary"]["rep_count"] == 8
        assert not got[1] and not got[8], got
        for k, c in expect.items():
            assert c in got[k], (k, c, got[k])
        assert r["reps"][0]["score"] > r["reps"][1]["score"]

    # fatigue: reps 6+ slow down and get shallower
    per = [{}] * 5 + [{"desc": 2.2, "asc": 2.0, "bottom": 100}] * 4
    r = A.analyze(make_clip(reps=9, per_rep=per))
    print(f"[synth fatigue] reps={r['summary']['rep_count']} fatigue_from_rep={r['summary']['fatigue_from_rep']}")
    assert r["summary"]["rep_count"] == 9 and r["summary"]["fatigue_from_rep"] == 6
    r = A.analyze(make_clip(reps=9))
    assert r["summary"]["fatigue_from_rep"] is None

    # not a push-up (upright) and empty input
    r = A.analyze(make_clip(reps=4, upright=True))
    print(f"[synth upright] ok={r['confidence_ok']} reps={r['summary']['rep_count']} warnings={r['warnings']}")
    assert not r["confidence_ok"] and r["summary"]["rep_count"] == 0
    r = A.analyze({"fps": 30, "w": 640, "h": 480, "frames": []})
    assert not r["confidence_ok"] and r["summary"]["rep_count"] == 0
    nobody = {"fps": 30, "w": 640, "h": 480, "frames": [{"i": i, "t": i / 30, "lm": None, "world": None} for i in range(60)]}
    r = A.analyze(nobody)
    check_schema(r, 60)
    assert not r["confidence_ok"]

    # custom thresholds are respected
    r = A.analyze(side, thresholds={"depth_deg": 40})
    assert r["summary"]["issue_counts"].get("SHALLOW") == 6 and r["thresholds"]["depth_deg"] == 40
    print("ALL ANALYSIS TESTS PASSED")


if __name__ == "__main__":
    main()
