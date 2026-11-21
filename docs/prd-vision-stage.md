# PRD: Vision Stage (Person 1)

## Introduction

The NullHacks theme is *physical AI: simulate data*. Our product turns an ordinary
phone video of a push-up set into (a) a per-rep form-coaching report and (b) a
labelled human-motion dataset that includes **simulated wearable IMU data**
(accelerometer + gyroscope), i.e. training data for a smartwatch/chest-strap
coach without owning the hardware.

The vision stage is the first link of the pipeline:

```
video -> [VISION] -> landmarks.json (+ imu) + overlay.mp4 -> [BIOMECH] -> analysis.json -> [PRODUCT/LLM] -> report
```

It measures; it never judges form (that is Person 2's job).

## Goals

- Convert a <= 30 s side-view push-up clip into `landmarks.json` in under ~1 min on a laptop CPU.
- Give Person 2 a contract-valid sample `landmarks.json` within 30 min, before real code is finished.
- Emit a browser-playable skeleton overlay video for the UI.
- Emit a simulated IMU stream (chest, hip, wrist) aligned to the same timestamps.
- Refuse to guess: report low confidence with an actionable message when the body is not visible.

## User Stories

### US-001: Angle and geometry helpers
**Description:** As Person 2, I want a tested `angle(a, b, c)` so every form check uses the same maths.

**Acceptance Criteria:**
- [ ] `angle` returns degrees at joint b; 90 for a right angle, 180 for collinear points
- [ ] Never NaN for coincident points (returns 0 or 180, not a crash)
- [ ] pytest passes

### US-002: Landmark smoothing
**Description:** As Person 2, I want jitter removed so noise does not create fake reps.

**Acceptance Criteria:**
- [ ] Centered moving average, default window 5, configurable
- [ ] Output length equals input length; edges handled without shrinking
- [ ] Missing frames (NaN) are ignored in the average, not propagated
- [ ] pytest passes

### US-003: Side selection and confidence check
**Description:** As a user, I want the app to tell me when my filming is bad instead of inventing feedback.

**Acceptance Criteria:**
- [ ] Picks `left` or `right` by higher mean visibility of shoulder/elbow/wrist/hip/ankle
- [ ] `confidence` = mean visibility of the chosen side's key joints over detected frames (0..1)
- [ ] If confidence < 0.5 or < 50 % of frames have a pose, emits message: "Film from the side, whole body in frame, 2–3 m away"
- [ ] pytest passes

### US-004: Pose extraction to landmarks.json
**Description:** As Person 2, I want landmarks for every frame in a fixed format so I can build rep detection.

**Acceptance Criteria:**
- [ ] Uses MediaPipe Tasks `PoseLandmarker` in VIDEO mode (legacy `mp.solutions` is removed in mediapipe >= 0.10.30)
- [ ] Timestamps passed to `detect_for_video` are strictly increasing ints (`int(i*1000/fps)`, bumped if equal)
- [ ] Landmarks written in **pixel** coordinates (x*w, y*h) so angles are correct on 16:9
- [ ] Frames with no pose are written with `"lm": null`
- [ ] Output validates against the contract (`validate_landmarks`)

### US-005: Skeleton overlay video
**Description:** As a user, I want to see my skeleton drawn on my video.

**Acceptance Criteria:**
- [ ] Draws the chosen side's limbs with `cv2.line` / `cv2.circle`
- [ ] Re-encoded to H.264 / yuv420p / faststart via imageio-ffmpeg so it plays in a browser
- [ ] Optional per-frame highlight list (joints to paint red) so Person 2/3 can mark faults later

### US-006: Simulated wearable IMU
**Description:** As a pitch, I want each video to also yield the accelerometer/gyro signal a wearable would have recorded, so the output is a multimodal training dataset.

**Acceptance Criteria:**
- [ ] Virtual sensors at chest (shoulder), hip, wrist
- [ ] Positions converted px -> metres with a scale from body segment length (world landmarks)
- [ ] Accelerometer = specific force (a - g) in m/s², gravity included (a stationary sensor reads ~9.81 up)
- [ ] Gyro = angular rate of the sensor's segment (torso / forearm) in deg/s
- [ ] Optional seeded Gaussian noise + bias to mimic a real IMU
- [ ] A stationary sensor test reads |accel| ≈ 9.81 and gyro ≈ 0
- [ ] pytest passes

### US-007: CLI
**Description:** As the team, I want one command to run the stage.

**Acceptance Criteria:**
- [ ] `python -m vision.cli clip.mp4 --out out/` writes `landmarks.json` and `overlay.mp4`
- [ ] Prints side, confidence, frames processed, seconds taken

## Functional Requirements

- FR-1: Input any OpenCV-readable video; cap processing at 30 s by default (`--max-seconds`).
- FR-2: Request auto-rotation (`CAP_PROP_ORIENTATION_AUTO=1`) and trust the decoded frame shape for w/h.
- FR-3: Output `landmarks.json` matching the contract below.
- FR-4: Landmark names are the 33 MediaPipe names in snake_case with `l_`/`r_` prefixes (e.g. `l_shoulder`).
- FR-5: Never block downstream: a sample `landmarks.json` is committed at `vision/samples/`.

### landmarks.json contract (v1)

```json
{
  "version": 1,
  "fps": 30, "w": 1920, "h": 1080,
  "side": "left", "confidence": 0.92, "message": null,
  "frames": [
    {"t": 0.033, "lm": {"l_shoulder": [x_px, y_px, vis], "...": []}}
  ],
  "imu": {
    "rate_hz": 30, "units": {"acc": "m/s^2", "gyro": "deg/s"}, "simulated": true,
    "sensors": {"chest": {"t": [], "acc": [[ax, ay]], "gyro": [g]}, "hip": {}, "wrist": {}}
  }
}
```

Coordinates: image pixels, origin top-left, y down. IMU axes: x = forward (direction the
person faces in the image), y = up, in the camera's sagittal plane.

## Non-Goals

- No form judgement, rep counting or thresholds (Person 2).
- No UI, charts or LLM (Person 3).
- No real wearable / Bluetooth.
- No live webcam mode; no front-view checks (elbow flare); single person only.

## Technical Considerations

- `mediapipe==1.2.3` (Python 3.9–3.14, Windows wheels, CPU). Model: `pose_landmarker_full.task`, fall back to `lite` if slow.
- `opencv-python` cannot encode H.264 — write `mp4v`, then re-encode with ffmpeg from `imageio-ffmpeg`.
- Pure functions (geometry, smoothing, side, confidence, IMU, contract) are unit-tested with pytest; the MediaPipe wrapper gets a smoke test only.
- See `vision/RESEARCH.md` for sources.

## Success Metrics

- Sample `landmarks.json` delivered to Person 2 within 30 min.
- Real clip -> outputs in < 60 s for a 30 s clip.
- On the team's labelled clips, confidence < 0.5 for the "bad angle" clip and > 0.7 for good clips.

## Open Questions

- Model variant (full vs lite) — decide after timing a real clip.
- Wearable sensor placement shown in the pitch (wrist watch vs chest strap) — Person 3.
