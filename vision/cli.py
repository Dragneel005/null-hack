"""python -m vision.cli clip.mp4 --out out/   ->  out/landmarks.json + out/overlay.mp4"""
import argparse
from pathlib import Path

from .extract import extract
from .overlay import render_overlay


def main(argv=None):
    p = argparse.ArgumentParser(description="Video -> landmarks.json (+ simulated IMU) + overlay")
    p.add_argument("video")
    p.add_argument("--out", default="out")
    p.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--no-imu-noise", action="store_true")
    a = p.parse_args(argv)

    doc = extract(a.video, a.out, a.model, a.max_seconds, imu_noise=not a.no_imu_noise)
    detected = sum(f["lm"] is not None for f in doc["frames"])
    print(f"frames {len(doc['frames'])} (pose in {detected}), side {doc['side']}, "
          f"confidence {doc['confidence']:.2f}, {doc['_seconds']}s")
    if doc["message"]:
        print("WARNING:", doc["message"])
    print("wrote", Path(a.out) / "landmarks.json")
    if not a.no_overlay:
        print("wrote", render_overlay(a.video, doc, Path(a.out) / "overlay.mp4"))


if __name__ == "__main__":
    main()
