"""Plain-assert tests for coach.py. Run: .venv/bin/python tests/test_coach.py"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("ANTHROPIC_API_KEY", None)
import coach as C  # noqa: E402

KEYS = {"source", "headline", "grade", "summary", "top_fixes", "strengths", "per_rep", "next_session"}


def rep(n, score=92, issues=(), min_elbow=80.0, d=1.2, a=0.9):
    t0 = 3.0 * n
    return {"n": n, "start": t0, "bottom": t0 + d, "end": t0 + d + a, "min_elbow": min_elbow,
            "max_elbow": 168.0, "max_hip_sag": 4.0, "max_hip_pike": 0.0, "min_head": 160.0,
            "descent_s": d, "ascent_s": a, "score": score, "issues": list(issues)}


def iss(code, n, sev="med", value=17.0):
    return {"code": code, "t": 3.0 * n + 1.0, "t_start": 3.0 * n + 0.8, "t_end": 3.0 * n + 1.4,
            "value": value, "severity": sev, "msg": f"{code} on rep {n}"}


def check_schema(r):
    assert KEYS <= set(r), set(r)
    assert r["source"] in ("rules", "claude")
    assert len(r["top_fixes"]) <= 2
    for f in r["top_fixes"]:
        assert {"title", "cue", "why", "reps"} <= set(f)
    json.dumps(r)


# 1. perfect set
perfect = {"confidence": 0.95, "confidence_ok": True, "reps": [rep(i) for i in range(1, 7)],
           "summary": {"rep_count": 6, "clean_reps": 6, "avg_score": 93.0, "fatigue_from_rep": None,
                       "issue_counts": {}, "avg_descent_s": 1.2, "avg_ascent_s": 0.9}}
r = C.coach(perfect)
check_schema(r)
assert r["grade"] == "A" and r["top_fixes"] == [] and r["source"] == "rules"
assert "6" in r["headline"] and len(r["per_rep"]) == 6 and r["strengths"]
print("perfect:", r["grade"], "|", r["headline"])

# 2. hip sag + fatigue, with physics
reps = [rep(i) for i in range(1, 5)] + [
    rep(5, 70, [iss("HIP_SAG", 5, "med", 16)]), rep(6, 62, [iss("HIP_SAG", 6, "high", 22), iss("FAST_DESCENT", 6, "low", 0.35)], d=0.35),
    rep(7, 55, [iss("HIP_SAG", 7, "high", 25), iss("SHALLOW", 7, "med", 108)], min_elbow=108)]
sag = {"confidence": 0.9, "confidence_ok": True, "reps": reps,
       "summary": {"rep_count": 7, "clean_reps": 4, "avg_score": 81.0, "fatigue_from_rep": 5,
                   "issue_counts": {"HIP_SAG": 3, "FAST_DESCENT": 1, "SHALLOW": 1}, "avg_descent_s": 1.0, "avg_ascent_s": 0.9}}
phys = {"body_mass_kg": 75, "reps": [{"n": i, "peak_hand_force_N": 560 + 5 * i, "peak_elbow_torque_Nm": 40,
                                      "peak_lumbar_Nm": 30.0 if i < 5 else 40.2, "work_J": 180} for i in range(1, 8)],
        "summary": {"equivalent_bench_kg": 48, "peak_hand_force_N": 600}}
r = C.coach(sag, phys)
check_schema(r)
assert r["top_fixes"][0]["title"] == "Keep your hips in line", r["top_fixes"]
assert r["top_fixes"][0]["reps"] == [5, 6, 7]
assert "34%" in r["top_fixes"][0]["why"], r["top_fixes"][0]["why"]
assert "rep 5" in r["summary"] and r["grade"] == "B", (r["grade"], r["summary"])
assert any("fatigue" in p["note"] for p in r["per_rep"] if p["n"] == 5)
print("hip sag:", r["grade"], "|", r["headline"], "|", r["top_fixes"][0]["why"][-110:])
print("  fix2:", r["top_fixes"][1]["title"], r["top_fixes"][1]["reps"])

# 3. zero reps
r = C.coach({"confidence": 0.8, "confidence_ok": True, "reps": [], "summary": {"rep_count": 0}})
check_schema(r)
assert r["grade"] == C.NO_GRADE and "side" in r["summary"].lower()
print("zero reps:", r["headline"])

# 4. low confidence, no reps
r = C.coach({"confidence": 0.2, "confidence_ok": False, "warnings": ["Low visibility of ankles"], "reps": []})
check_schema(r)
assert "Low visibility" in r["summary"] and "landscape" in r["summary"]
print("low conf:", r["headline"])

# 4b. low confidence but reps found -> caveat in summary
lc = dict(perfect, confidence=0.4, confidence_ok=False)
r = C.coach(lc)
assert "approximate" in r["summary"]

# 5. LLM path with bogus key -> falls back to rules
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-invalid-test"
r = C.coach(sag, phys, use_llm=True)
os.environ.pop("ANTHROPIC_API_KEY")
assert r["source"] == "rules" and "llm_error" in r
print("llm fallback:", r["llm_error"][:80])

# 6. validator accepts a good LLM json, rejects a bad one
rules = C._rules_report(sag, phys)
good = {"headline": "7 reps, 4 clean", "grade": "b", "summary": "x.", "next_session": "3x5",
        "top_fixes": [{"title": "t", "cue": "c", "why": "w", "reps": [5, 6, 99]}], "strengths": ["s"],
        "per_rep": [{"n": i, "note": "ok"} for i in range(1, 8)]}
v = C._validate(C._extract_json("```json\n" + json.dumps(good) + "\n```"), sag, rules)
assert v["grade"] == "B" and v["top_fixes"][0]["reps"] == [5, 6] and v["source"] == "claude"
try:
    C._validate({"grade": "Z"}, sag, rules)
    raise AssertionError("should fail")
except ValueError:
    pass

# 7. markdown + note
md = C.coach_markdown(C.coach(sag, phys))
assert "Top fixes" in md and "Rep 7" in md
assert "1 frame" in C.llm_comparison_note()

# 8. real pipeline if available
try:
    import analysis as A
    lms = json.loads((ROOT / "samples" / "landmarks_demo_side.json").read_text())
    an = A.analyze(lms)
    ph = None
    try:
        import physics as P
        ph = P.simulate(lms, an)
    except Exception as e:  # noqa: BLE001
        print("  (physics unavailable:", e, ")")
    r = C.coach(an, ph, use_llm=False)
    check_schema(r)
    print("real demo_side:", r["grade"], "|", r["headline"])
except ImportError:
    print("real pipeline: analysis.py not available yet, skipped")

print("ALL COACH TESTS PASSED")
