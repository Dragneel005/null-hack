"""Plain-assert test for overlay.py. Run: .venv/bin/python tests/test_overlay.py"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import cv2
import overlay


def fake_analysis(lm):
    """Contract-shaped analysis built from raw landmarks (used if analysis.py is absent)."""
    fr = lm["frames"]
    t = [f["t"] for f in fr]
    elbow, body, hip = [], [], []
    for f in fr:
        L = f["lm"]
        if not L:
            elbow.append(None); body.append(None); hip.append(None); continue
        elbow.append(overlay._angle(L["l_shoulder"], L["l_elbow"], L["l_wrist"]))
        body.append(overlay._angle(L["l_shoulder"], L["l_hip"], L["l_ankle"]))
        hip.append(180 - body[-1])
    e = np.array([x if x is not None else np.nan for x in elbow])
    reps, state, start, bottom, mn = [], "up", None, None, 999
    for i, a in enumerate(e):
        if np.isnan(a):
            continue
        if state == "up" and a < 120:
            state, start, mn = "down", t[max(0, i - 15)], a
        if state == "down":
            if a < mn:
                mn, bottom = a, t[i]
            if a > 150:
                n = len(reps) + 1
                issues = []
                if n == 2:
                    issues.append({"code": "HIP_SAG", "t": bottom, "t_start": bottom - 0.6, "t_end": bottom + 0.6,
                                   "value": 16.0, "severity": "high", "msg": "Hips sagged 16° below the shoulder-ankle line"})
                if n == 4:
                    issues.append({"code": "SHALLOW", "t": bottom, "t_start": bottom - 0.5, "t_end": bottom + 0.5,
                                   "value": 104.0, "severity": "med", "msg": "Only reached 104° at the elbow — go lower"})
                reps.append({"n": n, "start": start, "bottom": bottom, "end": t[i], "min_elbow": mn, "max_elbow": a,
                             "max_hip_sag": 5, "max_hip_pike": 0, "min_head": 160, "descent_s": 1, "ascent_s": 1,
                             "score": 100 - 25 * len(issues) - (8 if n == 4 else 0), "issues": issues})
                state = "up"
    phase = [None if x is None else ("down" if x < 150 else "top") for x in elbow]
    return {"exercise": "pushup", "side": "left", "confidence": 0.9, "confidence_ok": True, "warnings": [],
            "thresholds": {"depth_deg": 95},
            "series": {"t": t, "elbow": elbow, "body_line": body, "hip_dev": hip, "head": [None] * len(t), "phase": phase},
            "reps": reps, "summary": {"rep_count": len(reps)}}


def main():
    lm = json.loads((ROOT / "samples" / "landmarks_demo_side.json").read_text())
    try:
        import analysis as A
        an = A.analyze(lm); src = "analysis.py"
    except Exception as ex:
        an = fake_analysis(lm); src = f"fake ({type(ex).__name__})"
    try:
        import physics as P
        ph = P.simulate(lm, an); psrc = "physics.py"
    except Exception:
        e = an["series"]["elbow"]
        ph = {"series": {"stress": [None if x is None else float(np.clip((180 - x) / 1.1, 0, 100)) for x in e]}}
        psrc = "fake"
    print(f"analysis: {src}, physics: {psrc}, reps={len(an['reps'])}")

    # robustness: null frames, video longer than landmarks
    lm2 = dict(lm); lm2["frames"] = [dict(f) for f in lm["frames"][:300]]
    for k in range(50, 80):
        lm2["frames"][k]["lm"] = None
    out2 = overlay.render(str(ROOT / "samples/demo_side.mp4"), lm2, an, ph, out_path=str(ROOT / "outputs/test_overlay_short.mp4"))
    n2 = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_frames", "-of", "csv=p=0", out2], capture_output=True, text=True).stdout.strip())
    assert n2 == 300, n2
    # analysis None also works
    img_none = overlay.snapshot(str(ROOT / "samples/demo_side.mp4"), lm, None, 5.0)
    assert img_none.ndim == 3

    prog = []
    t0 = time.time()
    out = overlay.render(str(ROOT / "samples/demo_side.mp4"), lm, an, ph,
                         out_path=str(ROOT / "outputs/test_overlay.mp4"), progress=prog.append)
    dt = time.time() - t0
    assert os.path.exists(out) and os.path.getsize(out) > 10000
    assert prog and prog[-1] == 1.0 and all(a <= b for a, b in zip(prog, prog[1:]))
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=codec_name,pix_fmt,nb_frames", "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
    assert info.startswith("h264,yuv420p"), info
    print(f"render ok: {out} ({os.path.getsize(out)/1e6:.1f} MB, {dt:.1f}s, {info})")

    # snapshot at the worst rep's bottom
    rep = min(an["reps"], key=lambda r: r["score"]) if an["reps"] else None
    t = rep["bottom"] if rep else 5.0
    img = overlay.snapshot(str(ROOT / "samples/demo_side.mp4"), lm, an, t, physics=ph)
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    cv2.imwrite(str(ROOT / "outputs/test_overlay_snapshot.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if rep:
        img2 = overlay.snapshot(str(ROOT / "samples/demo_side.mp4"), lm, an, rep["end"] + 0.2, physics=ph)
        cv2.imwrite(str(ROOT / "outputs/test_overlay_flash.png"), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    print(f"snapshot ok at t={t:.2f}s (worst rep {rep['n'] if rep else '-'})")
    print("ALL OVERLAY TESTS PASSED")


if __name__ == "__main__":
    main()
