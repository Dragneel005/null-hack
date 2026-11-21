# Module contract (everyone codes against this)

Pipeline:  `video → pose.extract → analysis.analyze → physics.simulate → coach.feedback → overlay.render → app.py`
Video I/O: `video.py` (save_upload, to_browser_mp4, grab_frame) — the only module that touches codecs.

**Batch 1 decision (16:10):** one format per seam. Analysis + report formats below are the ones `app.py`, `coach.py` and
`samples/sample_analysis.json` already use; the richer fields from the first draft are kept as *optional*. Landmarks = the
19-name `l_shoulder` format of `pose.extract` / `synth`. Fault codes everywhere: `SHALLOW NO_LOCKOUT HIP_SAG HIP_PIKE HEAD_DROP FAST_DESCENT`.
Theme layer: `synth.py` (simulated athletes → labeled dataset) + `benchmark.py` (score our analyzer on simulated ground truth, sim-to-real classifier).

Python 3.10, deps in `requirements.txt`, venv at `.venv/` (`.venv/bin/python`). Run app: `.venv/bin/streamlit run app.py`.
All modules: pure functions, no global state, no prints on import, type hints, docstrings. Each module has a `if __name__ == "__main__":` demo that runs on `samples/landmarks_demo_side.json`.

Camera assumption: **side view** push-ups. Image coords: x right, **y down** (pixels).

## 1. `pose.py` (DONE) — `extract(video_path, model="full", max_seconds=90, max_width=960, smooth_window=5, progress=None) -> landmarks`

```jsonc
landmarks = {
  "source": "demo_side.mp4", "fps": 30.0, "w": 640, "h": 480,
  "n_frames": 1543, "detected_frames": 1543,
  "frames": [
    {"i": 0, "t": 0.0,
     "lm":    {"l_shoulder": [x_px, y_px, visibility_0to1], ...} | null,   // null = no person
     "world": {"l_shoulder": [x_m, y_m, z_m], ...} | null}                 // MediaPipe world coords, metres, hip-centred, y down
  ]
}
```
Landmark names: `nose l_ear r_ear l_shoulder r_shoulder l_elbow r_elbow l_wrist r_wrist l_hip r_hip l_knee r_knee l_ankle r_ankle l_heel r_heel l_foot r_foot`.
Real samples: `samples/landmarks_demo_side.json` (beach silhouette, 640x480, 51 s, **ground truth = 6 slow full reps**, good form, long holds at top between reps) and `samples/landmarks_demo_army.json` (20 s, 1280x720, has camera cuts / multiple people — robustness test, hand-release push-ups, ~2-3 reps visible).

## 2. `analysis.py` — `analyze(landmarks, strict=False, thresholds=None) -> analysis`
`app.py` imports it as `rules` (`import analysis as rules`). `strict=True` = chest-near-floor standard, False = 90° test standard.

```jsonc
analysis = {
  "exercise": "pushup",
  "standard": "Test standard (90°)",          // human-readable, shown in the UI
  "side": "left" | "right",
  "confidence": 0.0-1.0,                      // mean visibility of key joints on chosen side x fraction of frames detected; UI warns < 0.6
  "thresholds": {"depth_max_elbow": 90, "lockout_min_elbow": 160, "hip_tolerance": 15},   // REQUIRED keys (charts use them)
  "series": {"t": [...], "elbow": [...deg], "hip_dev": [...deg signed: + = sag, - = pike]},  // REQUIRED, same length, null where no person
  "reps": [{
    "n": 1, "start": 0.4, "bottom": 1.1, "end": 1.8,     // seconds
    "min_elbow": 84.0, "max_elbow": 165.0,
    "max_hip_dev": 6.0,                                   // signed deg, largest |hip_dev| in the rep
    "issues": [{"code": "HIP_SAG", "t": 1.1, "value": 16.0, "severity": "low"|"med"|"high",
                "message": "Hips 16° below the shoulder-ankle line",
                "t_start": 0.9, "t_end": 1.4}]            // t_start/t_end optional (overlay uses ±0.35 s around t otherwise)
    // optional extras: "descent_s", "ascent_s", "score", "min_head"
  }],
  "summary": {"rep_count": 10, "clean_reps": 6, "fatigue_from_rep": 7 | null
              /* optional: "avg_score", "issue_counts", "avg_descent_s", "avg_ascent_s", "total_time_s" */},
  "warnings": ["..."]                                     // optional
}
```
Fault codes: `SHALLOW` (min elbow never < depth threshold), `NO_LOCKOUT` (max elbow never > lockout threshold), `HIP_SAG` / `HIP_PIKE`
(|hip_dev| > hip_tolerance), `HEAD_DROP` (ear-shoulder-hip well below body line), `FAST_DESCENT` (descent < ~0.4 s).
Rep counting: hysteresis on smoothed elbow angle (down < ~100°, up > ~150°) with a min-rep-duration guard; must handle slow reps
with long top holds. Pick side with higher visibility (per clip). Test against `synth.generate(...)` ground truth.

## 3. `physics.py` — `simulate(landmarks, analysis, body_mass_kg=75.0, height_m=1.75) -> physics`
Rigid multi-segment body model (de Leva / Dempster segment mass fractions & COM locations) from the 2D side-view joints, scaled pixel→metres using body height. Static + dynamic (COM acceleration) equilibrium on two supports (hands, feet).
```jsonc
physics = {
  "body_mass_kg": 75, "height_m": 1.75, "px_per_m": 210.0,
  "series": {"t": [...], "com_x_m": [...], "com_y_m": [...],
             "hand_load_frac": [...0-1 fraction of body weight on hands],
             "hand_force_N": [...], "elbow_torque_Nm": [...], "shoulder_torque_Nm": [...],
             "lumbar_moment_Nm": [...], "stress": [...0-100 combined joint stress index]},
  "reps": [{"n": 1, "peak_hand_force_N": 560, "peak_elbow_torque_Nm": 40, "peak_shoulder_torque_Nm": 55,
            "peak_lumbar_Nm": 30, "stress_peak": 62, "work_J": 180, "power_W": 150}],
  "summary": {"avg_hand_load_frac": 0.68, "peak_hand_force_N": 600, "total_work_J": 1100, "kcal_est": 1.3,
              "equivalent_bench_kg": 48},     // hand force ≈ bench press of X kg
  "model_notes": "plain-English explanation of the model and its assumptions"
}
```
Null in series where no person. Must be physically sane: hand load ≈ 0.6-0.75 of body weight in plank (literature: ~64% top, ~75% bottom).

## 4. `coach.py` (Pary) — `feedback(analysis, frames=None) -> report`
```jsonc
report = {"source": "llm" | "fallback", "headline": "6 reps, 5 clean — strong set", "summary": "2-3 sentences",
          "priorities": [{"issue": "Hips sagging", "evidence": "Reps 7–10 (from 0:12), worst 16°", "why": "...", "cue": "..."}],  // max 2
          "positives": ["..."], "rep_notes": {"7": "note"}, "frame_check": "ok" | "suspect: ..." | "not checked",
          "llm_error": "..."}   // only when the LLM failed and it fell back
```
`frames` = JPEG bytes of the worst moments (`video.grab_frame`). Rule-based fallback always works offline; with
`ANTHROPIC_API_KEY` it asks Claude for the same JSON, grounded only in the numbers.

## 5. `overlay.py` (DONE) — `render(video_path, landmarks, analysis=None, physics=None, out_path=None, progress=None) -> str`
Skeleton of the camera side (green), joints of an active issue red (HIP_SAG/PIKE→hip, SHALLOW/NO_LOCKOUT→elbow, HEAD_DROP→ear,
FAST_DESCENT→shoulder+elbow), HUD: rep counter, elbow angle, current issue message. Returns a browser-playable H.264 mp4
(default `<video>_overlay.mp4`). `pose.render_overlay(video_path, landmarks, analysis, out_path)` is an alias used by `app.py`.

## 5a. `video.py` (DONE) — video file I/O
`save_upload(data, suffix, directory=None) -> (path, digest)` (content-addressed, under `outputs/uploads/`) ·
`to_browser_mp4(path) -> path` (H.264/yuv420p/faststart via imageio-ffmpeg; returns `path` unchanged if already H.264) ·
`grab_frame(path, t, max_width=960) -> JPEG bytes | None`.

## 6. `synth.py` — simulated athletes (THEME)
- `generate(n=500, seed=0, faults="mixed", camera="random") -> list[sample]` — procedural kinematic push-up generator: sample anthropometrics (height, segment proportions), rep count, tempo, depth, fault injection (`SHALLOW, NO_LOCKOUT, HIP_SAG, HIP_PIKE, HEAD_DROP, FAST_DESCENT`, with magnitudes), fatigue drift, camera (distance, height, yaw off side-view, fps, resolution), keypoint noise / jitter / dropouts. Forward kinematics in 3D → project to 2D pixels.
- Each sample: `{"landmarks": <same schema as pose.extract output>, "labels": {"rep_count": int, "reps": [{"n","start","end","faults": [codes]}], "params": {...}}}`.
- `export(samples, out_dir) -> zip_path` (JSONL + per-frame CSV + README/datasheet).
- `render_preview(sample, t) -> plotly Figure` or frames, so the app can show simulated skeletons.

## 7. `benchmark.py` (after build)
- `evaluate_analyzer(samples) -> metrics` — run `analysis.analyze` on simulated clips with known truth → rep-count accuracy, per-fault precision/recall.
- `sim_to_real(samples, real_landmarks_list) -> metrics` — train sklearn classifier on per-rep features from simulation only, apply to real clip reps.

## 8. `app.py` — Streamlit UI
Tabs: **Analyze** (upload or demo clip → overlay video, scorecards, charts, coach) · **Physics** · **Sim Lab** (generate dataset, preview, download, benchmark) · **About/Pitch**.

## 6a. `synth.py` status (DONE — vision branch)
`generate(n, seed, faults="mixed"|"none"|code|[codes], camera="random"|"ideal")`, `export(samples, out_dir, with_imu=False)`, `render_preview(sample, t)`.
Labels per rep also include `"bottom"` (s); labels also have `"fatigue_from_rep"` and `"params"` (height, camera, near_side, px_per_m).
Speed: ~0.2 s per athlete → use n≈50–100 in the app, not 500. Tests: `tests/test_synth.py` (every fault code is measurable in its labelled reps; clean reps are countable with down<100/up>150).
NO_LOCKOUT reps top out at 146–152°, so an "up > 150" counter may merge them with the next rep — that's the realistic hard case for the benchmark.

## 9. `imu.py` (DONE — THEME: simulated wearable) — `simulate(landmarks, side=None, noise=True, seed=0, smooth_window=None, height_m=1.75) -> imu`
Virtual chest strap (shoulder), waist clip (hip), smartwatch (wrist). Works on real `pose.extract` output and on `synth` samples. Scale from `world` landmarks (fallback: height).
```jsonc
imu = {"rate_hz": 30, "simulated": true, "noise": true, "side": "left", "m_per_px": 0.0058,
       "units": {"acc": "m/s^2", "gyro": "deg/s", "t": "s"},
       "axes": {"world": "x forward (facing), y up", "sensor": "x along segment, y perpendicular"},
       "sensors": {"chest"|"hip"|"wrist": {"placement": "shoulder", "segment": ["hip","shoulder"],
                   "t": [...], "acc": [[ax, ay]...],        // sensor frame, includes gravity (+9.81 at rest)
                   "acc_world": [[ax, ay]...],              // world frame, includes gravity
                   "gyro": [...]}}}                         // segment angular rate
```
Same length as `landmarks.frames` (gaps interpolated). Tests: `tests/test_imu.py` (rest reads 9.81 m/s², known oscillation/rotation reproduce textbook values).
Run tests: `python -m pytest -q tests`.
