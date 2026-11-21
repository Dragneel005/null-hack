"""FormSim end-to-end pipeline (headless): video -> pose -> analysis -> physics -> coach -> overlay.

    .venv/bin/python pipeline.py samples/demo_side.mp4 [--mass 75] [--height 1.75] [--model full] [--llm]

Writes outputs/<stem>_landmarks.json, <stem>_analysis.json, <stem>_physics.json, <stem>_report.json/.md
and outputs/<stem>_overlay.mp4. For the bundled demo clips, the saved MediaPipe track in samples/ is reused
(same output as pose.extract with model="full") unless --fresh is given.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SAMPLES = ROOT / "samples"


def _precomputed(video: Path) -> Optional[Path]:
    p = SAMPLES / f"landmarks_{video.stem}.json"
    try:
        same_dir = video.resolve().parent == SAMPLES.resolve()
    except OSError:
        same_dir = False
    return p if same_dir and p.exists() else None


def run_pipeline(video_path: str, body_mass_kg: float = 75.0, height_m: float = 1.75, model: str = "full",
                 use_llm: bool = False, progress: Optional[Callable[[str, float], None]] = None,
                 out_dir: Optional[str] = None, render_overlay: bool = True, reuse_landmarks: bool = True) -> dict:
    """Run the full FormSim pipeline on a video.

    progress(stage, fraction) is called with stage in {"pose", "analysis", "physics", "coach", "overlay"}.
    Returns {"landmarks", "analysis", "physics", "report", "overlay_path", "timings"}.
    """
    import analysis as _analysis
    import coach as _coach
    import physics as _physics

    def prog(stage: str, f: float) -> None:
        if progress:
            progress(stage, float(min(1.0, max(0.0, f))))

    video = Path(video_path)
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t0 = time.time()
    pre = _precomputed(video) if (reuse_landmarks and model == "full") else None
    if pre is not None:
        landmarks = json.loads(pre.read_text())
    else:
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import pose as _pose
        landmarks = _pose.extract(str(video), model=model, progress=lambda f: prog("pose", f))
    prog("pose", 1.0)
    timings["pose"] = round(time.time() - t0, 2)

    t0 = time.time()
    an = _analysis.analyze(landmarks)
    prog("analysis", 1.0)
    timings["analysis"] = round(time.time() - t0, 2)

    t0 = time.time()
    ph = _physics.simulate(landmarks, an, body_mass_kg=body_mass_kg, height_m=height_m)
    prog("physics", 1.0)
    timings["physics"] = round(time.time() - t0, 2)

    t0 = time.time()
    report = _coach.coach(an, ph, use_llm=use_llm)
    prog("coach", 1.0)
    timings["coach"] = round(time.time() - t0, 2)

    overlay_path = None
    if render_overlay:
        t0 = time.time()
        import overlay as _overlay
        overlay_path = _overlay.render(str(video), landmarks, an, ph, out_path=str(out / f"{video.stem}_overlay.mp4"),
                                       progress=lambda f: prog("overlay", f))
        prog("overlay", 1.0)
        timings["overlay"] = round(time.time() - t0, 2)

    return {"landmarks": landmarks, "analysis": an, "physics": ph, "report": report,
            "overlay_path": overlay_path, "timings": timings}


def save_outputs(res: dict, stem: str, out_dir: Optional[str] = None) -> dict:
    """Write JSON artefacts + markdown report. Returns {name: path}."""
    import coach as _coach
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k in ("landmarks", "analysis", "physics", "report"):
        p = out / f"{stem}_{k}.json"
        p.write_text(json.dumps(res[k], default=float))
        paths[k] = str(p)
    md = out / f"{stem}_report.md"
    md.write_text(_coach.coach_markdown(res["report"]))
    paths["report_md"] = str(md)
    if res.get("overlay_path"):
        paths["overlay"] = res["overlay_path"]
    return paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--mass", type=float, default=75.0)
    ap.add_argument("--height", type=float, default=1.75)
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--llm", action="store_true", help="use Claude coach if ANTHROPIC_API_KEY is set")
    ap.add_argument("--fresh", action="store_true", help="re-run pose tracking even for demo clips")
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()

    last = {"stage": None}

    def _p(stage: str, f: float) -> None:
        if stage != last["stage"]:
            print(f"  [{stage}]", flush=True)
            last["stage"] = stage

    t0 = time.time()
    res = run_pipeline(a.video, a.mass, a.height, a.model, a.llm, progress=_p,
                       render_overlay=not a.no_overlay, reuse_landmarks=not a.fresh)
    paths = save_outputs(res, Path(a.video).stem)
    an, ph, rp = res["analysis"], res["physics"], res["report"]
    s, ps = an["summary"], ph["summary"]
    print(f"\n{Path(a.video).name}: {s['rep_count']} reps ({s['clean_reps']} clean), avg score {s['avg_score']}, "
          f"side {an['side']}, confidence {an['confidence']} ok={an['confidence_ok']}")
    for w in an["warnings"]:
        print("  warning:", w)
    print(f"physics: avg hand load {ps.get('avg_hand_load_frac')} BW, peak hand {ps.get('peak_hand_force_N')} N, "
          f"work {ps.get('total_work_J')} J, ≈ bench {ps.get('equivalent_bench_kg')} kg")
    print(f"coach [{rp.get('source')}]: {rp.get('grade')} — {rp.get('headline')}")
    for f in rp.get("top_fixes", []):
        print(f"  fix: {f['title']} (reps {f.get('reps')})")
    print("timings:", res["timings"], f"total {time.time() - t0:.1f}s")
    for k, v in paths.items():
        print(f"  {k}: {v}")
