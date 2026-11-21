"""Plain-assert tests for physics.py. Run: .venv/bin/python tests/test_physics.py"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import physics  # noqa: E402

SIDE = json.loads((ROOT / "samples" / "landmarks_demo_side.json").read_text())
ARMY = json.loads((ROOT / "samples" / "landmarks_demo_army.json").read_text())


def stub_analysis(lms):
    """Minimal rep windows from wrist-shoulder vertical distance (if analysis.py unavailable)."""
    t, d = [], []
    for f in lms["frames"]:
        t.append(f["t"])
        lm = f["lm"]
        d.append(np.nan if not lm else lm["l_wrist"][1] - lm["l_shoulder"][1])
    d = np.array(d)
    hi, lo = np.nanpercentile(d, 90), np.nanpercentile(d, 10)
    down_thr, up_thr = lo + 0.35 * (hi - lo), lo + 0.75 * (hi - lo)
    reps, state, start, bottom = [], "up", t[0], None
    for ti, v in zip(t, d):
        if np.isnan(v):
            continue
        if state == "up" and v < down_thr:
            state, bottom = "down", ti
        elif state == "up" and v > up_thr:
            start = ti
        elif state == "down" and v > up_thr:
            reps.append({"n": len(reps) + 1, "start": start, "bottom": bottom, "end": ti,
                         "descent_s": bottom - start, "ascent_s": ti - bottom})
            state, start = "up", ti
    return {"side": "left", "reps": reps, "series": {}}


def get_analysis(lms):
    try:
        import analysis
        return analysis.analyze(lms), "analysis.py"
    except Exception as e:  # noqa: BLE001
        return stub_analysis(lms), f"stub ({e})"


def vals(ph, key):
    return np.array([np.nan if v is None else v for v in ph["series"][key]], float)


def test_side():
    an, src = get_analysis(SIDE)
    ph = physics.simulate(SIDE, an, 75, 1.75)
    s = ph["summary"]
    print(f"[side] analysis={src} reps={len(an['reps'])} px_per_m={ph['px_per_m']}")
    print(f"[side] summary={s}")
    for k in ("t", "com_x_m", "com_y_m", "hand_load_frac", "hand_force_N", "elbow_torque_Nm",
              "shoulder_torque_Nm", "lumbar_moment_Nm", "stress"):
        assert len(ph["series"][k]) == len(SIDE["frames"]), k
    assert 0.6 <= s["avg_hand_load_frac"] <= 0.8, s["avg_hand_load_frac"]
    frac = vals(ph, "hand_load_frac")
    comy = vals(ph, "com_y_m")
    top = frac[comy > np.nanpercentile(comy, 70)]
    bot = frac[comy < np.nanpercentile(comy, 10)]
    print(f"[side] hand load top={np.nanmedian(top):.3f} bottom={np.nanmedian(bot):.3f} (lit ~0.64 / ~0.75)")
    assert np.nanmedian(bot) > np.nanmedian(top)
    st = vals(ph, "stress")
    assert np.nanmin(st) >= 0 and np.nanmax(st) <= 100
    assert len(ph["reps"]) == len(an["reps"]) >= 5
    for r in ph["reps"]:
        assert 400 < r["peak_hand_force_N"] < 900, r
        assert 100 < r["work_J"] < 400, r
        assert r["power_W"] > 0
    assert 40 < s["equivalent_bench_kg"] < 90
    assert s["kcal_est"] > 0 and isinstance(ph["model_notes"], str) and len(ph["model_notes"]) > 200
    json.dumps(ph)  # serialisable, no NaN leaks
    assert "NaN" not in json.dumps(ph)
    return ph, an


def test_mirror(ph_side, an):
    """Flip the clip horizontally: physics must be invariant to facing direction."""
    m = copy.deepcopy(SIDE)
    W = m["w"]
    for f in m["frames"]:
        if f["lm"]:
            f["lm"] = {k: [W - v[0], v[1], v[2]] for k, v in f["lm"].items()}
    ph = physics.simulate(m, an, 75, 1.75)
    a, b = ph["summary"]["avg_hand_load_frac"], ph_side["summary"]["avg_hand_load_frac"]
    print(f"[mirror] avg_hand_load_frac {a} vs {b}")
    assert abs(a - b) < 1e-3
    assert abs(ph["summary"]["peak_shoulder_torque_Nm"] - ph_side["summary"]["peak_shoulder_torque_Nm"]) < 0.5


def test_mass_scaling(an):
    a = physics.simulate(SIDE, an, 60, 1.75)["summary"]["peak_hand_force_N"]
    b = physics.simulate(SIDE, an, 90, 1.75)["summary"]["peak_hand_force_N"]
    print(f"[mass] peak hand force 60kg={a} 90kg={b}")
    assert abs(b / a - 1.5) < 0.01


def test_army():
    an, src = get_analysis(ARMY)
    ph = physics.simulate(ARMY, an)
    s = ph["summary"]
    print(f"[army] analysis={src} reps={len(ph['reps'])} summary={s}")
    assert 0.4 < s["avg_hand_load_frac"] < 0.95
    assert "NaN" not in json.dumps(ph)


def test_empty():
    lms = {"fps": 30.0, "w": 640, "h": 480, "frames": [{"i": i, "t": i / 30, "lm": None, "world": None} for i in range(10)]}
    ph = physics.simulate(lms, None)
    assert ph["summary"]["frames_modelled"] == 0 and all(v is None for v in ph["series"]["stress"])
    print("[empty] ok")


def test_helpers(ph, an):
    fb = physics.free_body(SIDE, ph, 10.0)
    assert fb and fb["hand_force_N"] > 0 and len(fb["segments"]) == 14
    tot = sum(sg["mass_kg"] for sg in fb["segments"])
    assert abs(tot - 75) < 0.1, tot
    fd = physics.figure_data(ph)
    assert len(fd["t"]) == len(fd["value"]) > 1000
    physics.figure(ph, an)
    print(f"[helpers] free_body ok (segment mass sum {tot:.2f} kg), figure ok")


if __name__ == "__main__":
    ph, an = test_side()
    test_mirror(ph, an)
    test_mass_scaling(an)
    test_army()
    test_empty()
    test_helpers(ph, an)
    print("ALL PHYSICS TESTS PASSED")
