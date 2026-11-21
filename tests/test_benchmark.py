"""benchmark.py: score an analyzer against simulated ground truth."""
import math

import benchmark
import synth

SAMPLES = synth.generate(n=6, seed=11, faults="mixed", camera="ideal")
TRUTH = {s["landmarks"]["source"]: s["labels"] for s in SAMPLES}


def _as_analysis(labels, drop_code=None, extra_rep=False):
    reps = [{"n": r["n"], "start": r["start"], "bottom": r["bottom"], "end": r["end"],
             "issues": [{"code": c, "t": r["bottom"]} for c in r["faults"] if c != drop_code]}
            for r in labels["reps"]]
    if extra_rep:
        last = reps[-1]["end"]
        reps.append({"n": len(reps) + 1, "start": last + 5, "bottom": last + 5.5, "end": last + 6,
                     "issues": [{"code": "HIP_SAG", "t": last + 5.5}]})
    return {"reps": reps, "summary": {"rep_count": len(reps)}}


def oracle(landmarks):
    return _as_analysis(TRUTH[landmarks["source"]])


def test_perfect_analyzer_scores_one_everywhere():
    m = benchmark.evaluate_analyzer(SAMPLES, analyze=oracle)
    assert m["rep_count_exact"] == 1.0 and m["rep_count_mae"] == 0
    assert m["rep_recall"] == 1.0 and m["rep_precision"] == 1.0
    for code, f in m["faults"].items():
        if f["support"]:
            assert f["precision"] == 1.0 and f["recall"] == 1.0, code


def test_blind_analyzer_finds_nothing():
    m = benchmark.evaluate_analyzer(SAMPLES, analyze=lambda lm: {"reps": [], "summary": {"rep_count": 0}})
    assert m["rep_recall"] == 0.0 and m["rep_count_exact"] == 0.0
    assert all(f["recall"] == 0.0 for f in m["faults"].values() if f["support"])


def test_missing_one_fault_type_only_hurts_that_faults_recall():
    m = benchmark.evaluate_analyzer(
        SAMPLES, analyze=lambda lm: _as_analysis(TRUTH[lm["source"]], drop_code="HIP_SAG"))
    assert m["faults"]["HIP_SAG"]["recall"] == 0.0
    others = [f for c, f in m["faults"].items() if c != "HIP_SAG" and f["support"]]
    assert others and all(f["recall"] == 1.0 for f in others)


def test_phantom_rep_hurts_precision_and_count_but_not_recall():
    m = benchmark.evaluate_analyzer(
        SAMPLES, analyze=lambda lm: _as_analysis(TRUTH[lm["source"]], extra_rep=True))
    assert m["rep_recall"] == 1.0 and m["rep_precision"] < 1.0
    assert m["rep_count_mae"] == 1.0
    assert m["faults"]["HIP_SAG"]["precision"] < 1.0


def test_crashing_analyzer_is_reported_not_raised():
    def boom(lm):
        raise ValueError("bad clip")
    m = benchmark.evaluate_analyzer(SAMPLES[:2], analyze=boom)
    assert m["errors"] == 2 and m["n_clips"] == 2


def test_metrics_are_broken_down_by_condition():
    m = benchmark.evaluate_analyzer(SAMPLES, analyze=oracle, by=lambda s: "all")
    assert m["by_condition"]["all"]["rep_count_exact"] == 1.0
    assert math.isclose(m["by_condition"]["all"]["n_clips"], len(SAMPLES))
