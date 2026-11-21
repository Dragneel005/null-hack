"""Plain-assert tests for benchmark.py. Run: env -u PYTHONPATH .venv/bin/python tests/test_benchmark.py"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark  # noqa: E402
import synth  # noqa: E402


def main() -> None:
    # matching
    pred = [{"start": 0.0, "bottom": 1.0, "end": 2.0}, {"start": 5.0, "bottom": 6.0, "end": 7.0}]
    truth = [{"start": 0.1, "bottom": 1.1, "end": 2.1}, {"start": 3.0, "bottom": 3.5, "end": 4.0},
             {"start": 5.2, "bottom": 6.0, "end": 7.1}]
    assert benchmark.match_reps(pred, truth) == [(0, 0), (1, 2)]
    assert benchmark.match_reps([], truth) == []

    t0 = time.time()
    S = synth.generate(60, seed=11)
    m = benchmark.evaluate_analyzer(S)
    print(f"evaluate_analyzer(60): {time.time() - t0:.1f}s  exact {m['rep_count_exact_acc']}  "
          f"MAE {m['rep_count_mae']}  rep F1 {m['rep_detection']['f1']}  macro fault F1 {m['macro_fault_f1']}")
    for f, v in m["per_fault"].items():
        print(f"  {f:<13} P {v['precision']}  R {v['recall']}  F1 {v['f1']}  n={v['support']}")
    assert m["n"] == 60 and set(m["per_fault"]) == set(benchmark.FAULTS)
    assert m["rep_count_exact_acc"] >= 0.85, m["rep_count_exact_acc"]
    assert m["rep_detection"]["f1"] >= 0.9
    json.dumps(m)

    # perfect clean side-view clips: counts must be exact, no faults predicted on clean reps mostly
    C = synth.generate(10, seed=5, faults="none", camera="side", noise=0.3)
    mc = benchmark.evaluate_analyzer(C)
    print(f"clean side-view: exact {mc['rep_count_exact_acc']}")
    assert mc["rep_count_exact_acc"] >= 0.9

    t0 = time.time()
    real = [json.loads(p.read_text()) for p in sorted((ROOT / "samples").glob("landmarks_*.json"))]
    s2r = benchmark.sim_to_real(S, real)
    print(f"sim_to_real: {time.time() - t0:.1f}s train {s2r['n_train_reps']} held-out {s2r['n_heldout_reps']}  "
          f"ML {s2r['heldout_exact_acc_ml']} vs rules {s2r['heldout_exact_acc_rules']}  "
          f"real agreement {s2r['real_agreement_rate']} on {s2r['real_reps']} reps")
    assert s2r["n_train_reps"] > 50 and s2r["real_reps"] >= 6
    assert all(r["ml_sim_trained"] == "clean" for r in s2r["real_predictions"] if "beach" in r["clip"]), \
        s2r["real_predictions"]
    json.dumps(s2r)
    print("ALL BENCHMARK TESTS PASSED")


if __name__ == "__main__":
    main()
