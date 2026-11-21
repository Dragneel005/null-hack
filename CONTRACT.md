# Module contract (everyone codes against this)

Pipeline:  `video → pose.extract → analysis.analyze → physics.simulate → coach.coach → overlay.render → app.py`
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

## 2. `analysis.py` — `analyze(landmarks, thresholds=None) -> analysis`

```jsonc
analysis = {
  "exercise": "pushup", "side": "left" | "right",
  "confidence": 0.0-1.0,                 // mean visibility of key joints on chosen side × fraction of frames detected
  "confidence_ok": true,                 // false -> UI shows "film from the side, whole body in frame"
  "warnings": ["..."],                   // human-readable (low visibility, not side view, too short, ...)
  "thresholds": {...},                   // effective thresholds used
  "series": {                            // per-frame, same length as landmarks.frames; null where no person
    "t": [...], "elbow": [...deg], "body_line": [...deg, 180 = straight shoulder-hip-ankle],
    "hip_dev": [...deg signed: + = sag (hip below line), - = pike (hip above line)],
    "head": [...deg ear-shoulder-hip], "phase": ["up"|"down"|"bottom"|"top"|null ...]
  },
  "reps": [{
    "n": 1, "start": 0.4, "bottom": 1.1, "end": 1.8,          // seconds
    "min_elbow": 84.0, "max_elbow": 165.0,
    "max_hip_sag": 6.0, "max_hip_pike": 0.0, "min_head": 160.0,
    "descent_s": 0.7, "ascent_s": 0.6,
    "score": 0-100,
    "issues": [{"code": "HIP_SAG", "t": 1.1, "t_start": 0.9, "t_end": 1.4,
                "value": 16.0, "severity": "low"|"med"|"high", "msg": "Hips sagged 16° below the shoulder-ankle line"}]
  }],
  "summary": {
    "rep_count": 6, "clean_reps": 5, "avg_score": 88.0,
    "fatigue_from_rep": 7 | null,        // first rep where form/tempo degrades persistently vs first reps
    "issue_counts": {"HIP_SAG": 2, ...},
    "avg_descent_s": 0.9, "avg_ascent_s": 0.8, "total_time_s": 40.0
  }
}
```
**Fault codes (shared with synth.py):** `SHALLOW` (min elbow never < depth threshold, ~95°), `NO_LOCKOUT` (max elbow never > ~155°), `HIP_SAG`, `HIP_PIKE` (|hip_dev| > ~12-15°), `HEAD_DROP` (head angle well below body line), `FAST_DESCENT` (descent < ~0.4 s).
Rep counting: hysteresis state machine on smoothed elbow angle (down < ~100°, up > ~150°), with min-rep-duration guard; must work for both slow reps with long top holds and fast reps. Pick side with higher visibility (per clip, not per frame).

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

## 4. `coach.py` — `coach(analysis, physics=None, use_llm=True) -> report`
```jsonc
report = {"source": "rules" | "claude",
          "headline": "6 reps, 5 clean — strong set", "grade": "A"|"B"|"C"|"D",
          "summary": "2-3 sentences",
          "top_fixes": [{"title": "Keep hips in line", "cue": "squeeze glutes, ribs down", "why": "...", "reps": [7, 8]}],   // max 2
          "strengths": ["..."], "per_rep": [{"n": 1, "note": "..."}],
          "next_session": "concrete plan, e.g. 3x6 with 2s descent"}
```
Rules-based always works offline. If `use_llm` and `ANTHROPIC_API_KEY` is set, ask Claude (anthropic SDK) for the same JSON, grounded only in the numbers; on any error fall back to rules.

## 5. `overlay.py` — `render(video_path, landmarks, analysis, physics=None, out_path=None, progress=None) -> str`
Draws skeleton (green OK, joints/segments red during an issue window), HUD: rep counter, elbow angle, current issue text, stress bar. Writes browser-playable **H.264 yuv420p mp4** (OpenCV write → `ffmpeg -vcodec libx264 -pix_fmt yuv420p -movflags +faststart`). Also `snapshot(video_path, landmarks, analysis, t) -> np.ndarray RGB` for worst-rep thumbnails.

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
