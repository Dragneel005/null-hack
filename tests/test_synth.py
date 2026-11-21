"""Plain-assert tests for synth.py. Run: .venv/bin/python tests/test_synth.py"""
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import synth  # noqa: E402


def angle(a, b, c):
    a, b, c = (np.asarray(v[:2], float) for v in (a, b, c))
    v1, v2 = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))


def main():
    t0 = time.time()
    S = synth.generate(50, seed=7)
    dt = time.time() - t0
    print(f"generate(50): {dt:.2f}s")
    real = json.loads((ROOT / "samples" / "landmarks_demo_side.json").read_text())
    rf = real["frames"][0]

    # --- schema matches pose.extract output
    for s in S:
        lm = s["landmarks"]
        assert set(lm) == set(real), (set(lm) ^ set(real))
        assert lm["n_frames"] == len(lm["frames"])
        assert lm["detected_frames"] == sum(f["lm"] is not None for f in lm["frames"])
        for k in ("fps", "w", "h"):
            assert type(lm[k]) is type(real[k]), k
        for f in lm["frames"][:50]:
            assert set(f) == set(rf)
            if f["lm"] is not None:
                assert set(f["lm"]) == set(rf["lm"]) and set(f["world"]) == set(rf["world"])
                assert all(len(v) == 3 and all(isinstance(x, float) for x in v) for v in f["lm"].values())
                assert all(0 <= v[2] <= 1 for v in f["lm"].values())
        # timings: t = i / fps, monotonic
        ts = np.array([f["t"] for f in lm["frames"]])
        assert np.allclose(ts, np.arange(len(ts)) / lm["fps"], atol=1e-3)
        lab = s["labels"]
        assert lab["rep_count"] == len(lab["reps"]) >= 3
        assert len(lab["truth"]["elbow"]) == lm["n_frames"] == len(lab["truth"]["phase"])
        prev_end = -1
        for r in lab["reps"]:
            assert prev_end <= r["start"] < r["bottom"] < r["end"] <= ts[-1] + 1e-6, r
            prev_end = r["end"]
            assert set(r["faults"]) <= set(synth.FAULTS)
    print("schema/timing OK on 50 samples")

    # --- rep labels vs elbow-angle minima (truth series): one local min per rep, at 'bottom'
    errs, depth_ok = [], 0
    for s in S:
        el = np.array(s["labels"]["truth"]["elbow"]); fps = s["landmarks"]["fps"]
        for r in s["labels"]["reps"]:
            a, b = int(round(r["start"] * fps)), int(round(r["end"] * fps)) + 1
            i = a + int(np.argmin(el[a:b]))
            errs.append(abs(i / fps - r["bottom"]))
            assert el[int(round(r["bottom"] * fps))] - el[a:b].min() < 0.5   # flat bottom pauses -> near-ties
            depth_ok += abs(el[a:b].min() - r["min_elbow"]) < 0.2
            # outside reps (top holds) elbow stays near lockout of neighbours
        assert all(abs(r["max_elbow"] - r["min_elbow"]) >= 20 for r in s["labels"]["reps"])
    nrep = sum(len(s["labels"]["reps"]) for s in S)
    print(f"bottom-time error: mean {np.mean(errs)*1000:.1f} ms, max {np.max(errs)*1000:.1f} ms; "
          f"min_elbow consistent {depth_ok}/{nrep}")
    assert np.max(errs) < 0.2 and depth_ok == nrep

    # --- fault labels consistent with thresholds
    for s in S:
        for r in s["labels"]["reps"]:
            assert ("SHALLOW" in r["faults"]) == (r["min_elbow"] > 100)
            assert ("HIP_SAG" in r["faults"]) == (r["max_hip_sag"] > 12)
            assert ("FAST_DESCENT" in r["faults"]) == (r["descent_s"] < 0.40)

    # --- geometry: noise-free perfect side view -> 2D elbow angle ~= 3D truth, body in frame
    C = synth.generate(10, seed=3, camera="side", noise=0.0)
    d = []
    for s in C:
        lm, el = s["landmarks"], s["labels"]["truth"]["elbow"]
        side = "l" if s["labels"]["near_side"] == "left" else "r"
        for i in range(0, lm["n_frames"], 5):
            f = lm["frames"][i]["lm"]
            d.append(angle(f[f"{side}_shoulder"], f[f"{side}_elbow"], f[f"{side}_wrist"]) - el[i])
            # toes & wrists planted (y constant within ~2px after smoothing)
        ys = np.array([fr["lm"][f"{side}_wrist"][1] for fr in lm["frames"]])
        assert ys.std() < 2.0, ys.std()
    print(f"2D-vs-3D elbow angle on clean side view: mean |err| {np.mean(np.abs(d)):.1f} deg, p95 {np.percentile(np.abs(d), 95):.1f}")
    assert np.percentile(np.abs(d), 95) < 20 and np.mean(np.abs(d)) < 8

    inframe = []
    for s in S:
        lm = s["landmarks"]
        pts = np.array([[v[0], v[1]] for f in lm["frames"] if f["lm"] for v in f["lm"].values()])
        inframe.append(np.mean((pts[:, 0] >= 0) & (pts[:, 0] <= lm["w"]) & (pts[:, 1] >= 0) & (pts[:, 1] <= lm["h"])))
    print(f"keypoints inside image: mean {np.mean(inframe)*100:.1f}%, worst clip {np.min(inframe)*100:.1f}%")
    assert np.mean(inframe) > 0.97

    # --- speed
    t0 = time.time(); big = synth.generate(500, seed=0); dt500 = time.time() - t0
    st = synth.dataset_stats(big)
    print(f"generate(500): {dt500:.1f}s, {st['total_frames']} frames, {st['total_reps']} reps, "
          f"clean reps {st['clean_rep_frac']:.0%}, fault reps {st['fault_rep_counts']}")
    assert dt500 < 25

    # --- export, preview, animation
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time(); zp = synth.export(S[:20], td)
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            assert names == {"dataset.jsonl", "frames.csv", "labels.csv", "DATASHEET.md"}, names
            lines = z.read("dataset.jsonl").decode().strip().split("\n")
            assert len(lines) == 20 and json.loads(lines[0])["landmarks"]["frames"]
            lab = z.read("labels.csv").decode().strip().split("\n")
            assert len(lab) - 1 == sum(len(s["labels"]["reps"]) for s in S[:20])
            fr = z.read("frames.csv").decode().strip().split("\n")
            assert len(fr) - 1 == sum(s["landmarks"]["n_frames"] for s in S[:20])
            print(f"export(20): {Path(zp).stat().st_size/1e6:.1f} MB zip in {time.time()-t0:.1f}s")
    fig = synth.render_preview(S[0])
    assert len(fig.data) >= 3
    anim = synth.render_animation(S[0])
    assert len(anim.frames) > 10
    print(f"preview traces {len(fig.data)}, animation frames {len(anim.frames)}")

    # --- informational: analyzer agreement
    try:
        import analysis
        exact, within1, rows = 0, 0, []
        for s in S[:20]:
            a = analysis.analyze(s["landmarks"])
            pred, true = a["summary"]["rep_count"], s["labels"]["rep_count"]
            exact += pred == true; within1 += abs(pred - true) <= 1
            rows.append((true, pred))
        print(f"analysis.analyze rep count on 20 sims: exact {exact}/20, within±1 {within1}/20; (true,pred) {rows}")
    except ImportError:
        print("analysis.py not available — skipped agreement check")
    print("ALL SYNTH TESTS PASSED")


if __name__ == "__main__":
    main()
