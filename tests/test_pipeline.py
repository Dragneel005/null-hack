"""End-to-end pipeline test on both demo clips. Run: env -u PYTHONPATH .venv/bin/python tests/test_pipeline.py"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402


def main() -> None:
    for name in ("demo_side", "demo_army"):
        t0 = time.time()
        stages = []
        res = pipeline.run_pipeline(str(ROOT / "samples" / f"{name}.mp4"),
                                    progress=lambda s, f: stages.append(s), out_dir=str(ROOT / "outputs" / "test"))
        an, ph, rp = res["analysis"], res["physics"], res["report"]
        assert {"pose", "analysis", "physics", "coach", "overlay"} <= set(stages)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=codec_name,pix_fmt", "-of", "csv=p=0", res["overlay_path"]],
                               capture_output=True, text=True).stdout.strip()
        frac = ph["summary"]["avg_hand_load_frac"]
        print(f"{name}: {an['summary']['rep_count']} reps, hand load {frac}, grade {rp['grade']} "
              f"'{rp['headline']}', overlay {probe}, {time.time() - t0:.1f}s")
        assert probe == "h264,yuv420p", probe
        assert 0.55 <= frac <= 0.85, frac
        json.dumps({k: res[k] for k in ("analysis", "physics", "report")})
        if name == "demo_side":
            assert an["summary"]["rep_count"] == 6 and rp["grade"] in "AB"
            assert len(ph["reps"]) == 6
        else:
            assert rp["grade"] in ("A", "B", "C", "D", "N/A")
    print("ALL PIPELINE TESTS PASSED")


if __name__ == "__main__":
    main()
