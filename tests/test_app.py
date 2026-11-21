"""Headless smoke test for app.py via streamlit AppTest. Run: .venv/bin/python tests/test_app.py"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> None:
    t0 = time.time()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    print(f"load ok in {time.time() - t0:.1f}s, tabs={len(at.tabs)}, errors={[e.value for e in at.error]}")

    demo = [b for b in at.button if b.key == "demo_btn"]
    assert demo, "demo button missing"
    t0 = time.time()
    demo[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    print(f"demo pipeline ok in {time.time() - t0:.1f}s; errors={[e.value[:200] for e in at.error]}")
    print("metric/markdown blocks:", len(at.markdown), "dataframes:", len(at.dataframe))
    cur = at.session_state["current"]
    assert cur["analysis"]["summary"]["rep_count"] == 6, cur["analysis"]["summary"]

    # plain rerun keeps the demo source
    at.run()
    assert at.session_state["current"]["name"] == "demo_side.mp4"

    # simulated fault on the real clip -> red faults appear (overlay off to keep the test quick)
    tog = [t for t in at.toggle if "overlay" in str(t.label).lower()]
    if tog:
        tog[0].set_value(False)
    t0 = time.time()
    at.selectbox(key="sim_fault").set_value("HIP_SAG").run()
    assert not at.exception, [e.value for e in at.exception]
    s2 = at.session_state["current"]["analysis"]["summary"]
    print(f"sim fault HIP_SAG in {time.time() - t0:.1f}s: reps {s2['rep_count']} issues {s2['issue_counts']}")
    assert s2["rep_count"] == 6 and s2["issue_counts"].get("HIP_SAG") == 6
    at.selectbox(key="sim_fault").set_value("none").run()

    # precomputed benchmark loads instantly
    pre = [b for b in at.button if b.key == "bench_pre_btn"]

    gen = [b for b in at.button if b.key == "gen_btn"]
    if not pre:
        print("(precomputed benchmark button appears after generating)")
    if gen:
        at.number_input[0].set_value(12)
        t0 = time.time()
        gen[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        print(f"sim generate ok in {time.time() - t0:.1f}s; errors={[e.value[:200] for e in at.error]}")
        pre = [b for b in at.button if b.key == "bench_pre_btn"]
        if pre:
            t0 = time.time()
            pre[0].click().run()
            assert not at.exception, [e.value for e in at.exception]
            print(f"precomputed benchmark shown in {time.time() - t0:.1f}s")
        b = [b for b in at.button if b.key == "bench_btn"]
        if b:
            t0 = time.time()
            b[0].click().run()
            assert not at.exception, [e.value for e in at.exception]
            print(f"benchmark ok in {time.time() - t0:.1f}s; errors={[e.value[:200] for e in at.error]}")
            out = at.session_state["bench"]
            assert "eval" in out and "eval_err" not in out, out.get("eval_err")
            assert "s2r_err" not in out, out.get("s2r_err")
            print("  eval exact acc:", out["eval"].get("rep_count_exact_acc"),
                  "| s2r:", {k: v for k, v in out.get("s2r", {}).items() if not isinstance(v, (dict, list))})
    else:
        print("synth not ready: Sim Lab shows placeholder")
    print("ALL OK")


if __name__ == "__main__":
    main()
