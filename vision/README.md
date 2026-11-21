# vision/ — Person 1: video → landmarks.json (+ simulated IMU) + overlay.mp4

## Setup (Windows, Python 3.10+)

```powershell
cd C:\Users\Lelouch\Desktop\nullHack
python -m venv .venv
.venv\Scripts\activate
pip install -r vision\requirements.txt
python -m pytest -q vision\tests
```

## Run on a clip

```powershell
python -m vision.cli path\to\clip.mp4 --out out
# -> out\landmarks.json, out\overlay.mp4 (H.264, plays in browser / Streamlit)
# --model lite if it's slow, --no-overlay to skip the video
```

The pose model downloads itself to `vision/models/` on first run.

Film **from the side**, phone in landscape, 2–3 m away, whole body in frame.

## For Person 2 / Person 3

- Ready-made data with ground truth (no camera needed): `vision/samples/*.json`, regenerate with `python -m vision.make_samples`.
- Contract: `vision/tasks/prd-vision-stage.md`. Validate with `vision.contract.validate_landmarks(doc)`.
- Angle helper: `from vision.geometry import angle` → `angle(lm["l_shoulder"], lm["l_elbow"], lm["l_wrist"])`.
- Coordinates are pixels, y down. Use `doc["side"]` (`"left"`/`"right"`) to pick the `l_`/`r_` landmarks.
- If `doc["message"]` is not null, show it instead of feedback (bad filming).
- Paint fault joints red: `vision.overlay.render_overlay(video, doc, "overlay.mp4", highlights={frame_idx: ["hip"]})`.

## Simulated wearable (the "physical AI" part)

`doc["imu"]["sensors"]` has `chest`, `hip`, `wrist`: what a chest strap, waist clip and
smartwatch would have recorded — accelerometer (m/s², gravity included, sensor frame and
world frame) and gyroscope (deg/s), with realistic noise and bias. Each video therefore
yields a labelled multimodal dataset: pose + IMU + per-rep labels.

`python -m vision.make_samples` also simulates the *person* (`vision/synthetic.py`): a
kinematic push-up model with injectable faults (half reps, hip sag from rep N, pike) and
exact ground truth — unlimited labelled training data without filming anyone.
