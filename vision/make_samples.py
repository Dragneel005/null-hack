"""Write synthetic landmarks.json samples (with ground truth) to vision/samples/.

    python -m vision.make_samples
"""
import json
from pathlib import Path

from .contract import validate_landmarks
from .synthetic import generate_pushups

SAMPLES = {
    "pushups_clean_10.json": dict(reps=10),
    "pushups_fatigue_sag_from_7.json": dict(reps=10, hip_sag_from_rep=7, hip_sag_deg=20),
    "pushups_mixed_faults.json": dict(reps=8, half_reps=[2, 5], pike_reps=[7], seed=3),
}


def main(out_dir=Path(__file__).parent / "samples"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, kwargs in SAMPLES.items():
        doc = generate_pushups(**kwargs)
        try:
            from .imu import add_imu
            doc = add_imu(doc, seed=kwargs.get("seed", 0))
        except ImportError:
            pass
        validate_landmarks(doc)
        (out_dir / name).write_text(json.dumps(doc))
        print(f"wrote {out_dir / name}  ({len(doc['frames'])} frames)")


if __name__ == "__main__":
    main()
