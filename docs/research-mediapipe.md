# Vision pipeline research: MediaPipe Pose Landmarker and OpenCV (checked 2026-11-21)

Pipeline: video file -> PoseLandmarker (Tasks API, VIDEO mode) -> landmarks.json + skeleton overlay mp4.
If a claim has no primary source, it is marked **UNVERIFIED**.

## Q1. mediapipe on PyPI, Python versions, Windows, legacy `mp.solutions`
- The latest release is **mediapipe 1.2.3**, released 2026-11-11. The PyPI classifiers list **Python 3.9 to 3.14**. The wheels are tagged `py3-none`, so they don't depend on a specific CPython version. Builds exist for `win_amd64` (25.7 MB), `win_arm64`, `manylinux_2_28` x86_64/aarch64, and `macosx_11_0_arm64`. There is **no macOS x86_64 wheel** in 1.2.3. Source: https://pypi.org/project/mediapipe/
- From the GitHub release notes: v1.1.0 "Add Python 3.13 and 3.14 support". v0.10.33 notes that GPU (OpenGL) is disabled automatically on Windows builds, so **use the CPU delegate on Windows**. v0.10.32 and v1.1.0 also include Windows build fixes. Source: https://github.com/google-ai-edge/mediapipe/releases
- **The legacy `mp.solutions` (including `mp.solutions.pose`) has been removed.** In GitHub issue #6192, a user upgraded 0.10.21 -> 0.10.31 and `mediapipe.solutions` was gone. A community member says the removal started in 0.10.30, and maintainer schmidt-sebastian said to "migrate to MediaPipe Tasks". The same thread says DrawingUtils was added to Tasks in 0.10.32, but also that the linked docs and examples are outdated. Source: https://github.com/google-ai-edge/mediapipe/issues/6192
  - Practical advice: write the skeleton drawing yourself with `cv2.line` / `cv2.circle`. Don't rely on `mp.solutions.drawing_utils`.
  - Older tutorials use `mp.solutions.pose`, and that code fails with AttributeError on current versions.
- Known Windows install issues: none found in primary sources beyond the build fixes and the GPU note above. **UNVERIFIED** (from general experience, not a source): older Python versions, 32-bit Python, or a missing MSVC runtime can cause DLL load errors.

## Q2. Model .task URLs
The Python guide links to Hugging Face. Source: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- Lite: https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Lite/resolve/main/pose_landmarker_lite.task
- Full: https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Full/resolve/main/pose_landmarker_full.task
- Heavy: https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Heavy/resolve/main/pose_landmarker_heavy.task

The overview page still lists the GCS URLs. Source: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
- https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
- https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

Notes on the models:
- All three bundles use float16 weights. The detector input is 224x224 and the landmarker input is 256x256 (overview page).
- I could not confirm that any of these URLs downloads: the sandbox proxy blocked the requests. Test one on the dev laptop first.
- The current page has **no latency benchmarks**.
- **UNVERIFIED** (general knowledge): lite is fastest and heavy is most accurate and slowest. For a 3-hour build on a laptop CPU, start with **full**. Move to lite if it runs too slow, or to heavy for the demo if time allows. All three run offline from video, so speed only affects processing time.

## Q3. PoseLandmarker in VIDEO mode
The option names and defaults below come from the overview table: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker

| Option | Default |
|---|---|
| `running_mode` | IMAGE |
| `num_poses` | 1 |
| `min_pose_detection_confidence` | 0.5 |
| `min_pose_presence_confidence` | 0.5 |
| `min_tracking_confidence` | 0.5 |
| `output_segmentation_masks` | False |
| `result_callback` | only used in LIVE_STREAM mode |

The aliases `mp.tasks.BaseOptions` and so on, the `detect_for_video(mp_image, timestamp_ms)` call, and `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)` all come from the Python guide: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python

Timestamp rules from the guide:
- Timestamps are in **milliseconds**.
- VIDEO mode needs **monotonic** timestamps and processes every frame in order without dropping any.
- **UNVERIFIED**: the timestamp must be an int and strictly increasing; a repeated or decreasing timestamp raises an error. The code below makes them strictly increasing anyway.

```python
import json, cv2, mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
)

cap = cv2.VideoCapture("input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frames, i, last_ts = [], 0, -1
with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # OpenCV gives BGR; SRGB expects RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(round(i * 1000.0 / fps))
        ts = max(ts, last_ts + 1); last_ts = ts              # force strictly increasing int ms
        res = landmarker.detect_for_video(mp_image, ts)
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        frames.append({"frame": i, "t_ms": ts, "landmarks": None if lms is None else
                       [{"x": p.x, "y": p.y, "z": p.z, "visibility": p.visibility,
                         "presence": p.presence} for p in lms]})
        i += 1
cap.release()
json.dump({"fps": fps, "frames": frames}, open("landmarks.json", "w"))
```

## Q4. Result structure
Source for this section: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker (and the /python page)
- `result.pose_landmarks` is a list with one entry per pose, and each pose has 33 landmarks.
  - `x` and `y` are normalized to image width and height, in the range 0.0 to 1.0. Multiply by width and height to get pixels.
  - `z` is depth relative to the midpoint of the hips.
  - Each landmark also has `visibility` and `presence`.
- `result.pose_world_landmarks` uses the same 33 points in **meters, with the hip midpoint as the origin**, and also has visibility and presence.
  - Use these for angles that don't depend on the camera. Use normalized coordinates for the overlay.
- `result.segmentation_masks` is only returned when `output_segmentation_masks=True`.

| ID | Name | ID | Name |
|---|---|---|---|
| 0 | nose | 17 | left_pinky |
| 1 | left_eye_inner | 18 | right_pinky |
| 2 | left_eye | 19 | left_index |
| 3 | left_eye_outer | 20 | right_index |
| 4 | right_eye_inner | 21 | left_thumb |
| 5 | right_eye | 22 | right_thumb |
| 6 | right_eye_outer | 23 | left_hip |
| 7 | left_ear | 24 | right_hip |
| 8 | right_ear | 25 | left_knee |
| 9 | mouth_left | 26 | right_knee |
| 10 | mouth_right | 27 | left_ankle |
| 11 | left_shoulder | 28 | right_ankle |
| 12 | right_shoulder | 29 | left_heel |
| 13 | left_elbow | 30 | right_heel |
| 14 | right_elbow | 31 | left_foot_index |
| 15 | left_wrist | 32 | right_foot_index |
| 16 | right_wrist | | |

For side-view push-ups, the joints you need are:
- Elbow angle: points 11-13-15 or 12-14-16.
- Body line: shoulder-hip-ankle, points 11-23-27 or 12-24-28.

Pick the side facing the camera by comparing the mean `visibility` of each side.

## Q5. Browser-playable MP4 output
- **The pip `opencv-python` wheels do not include an H.264 (x264) encoder.** The maintainer gave the reason as "no encoding support for x264 because it's under GPL license". Notes from the thread:
  - The `H264` fourcc falls back to `avc1`, but the encoder is still missing.
  - The conda-forge `opencv` package reportedly supports `avc1`.
  - One user observed that the Windows and macOS wheels can load an OpenH264 DLL placed in the working directory. This was not confirmed by a maintainer and is not a reliable fix.
  - Source: https://github.com/opencv/opencv-python/issues/100
- Plan to use:
  1. Write the overlay with `cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))`. `mp4v` is MPEG-4 Part 2, which most browsers don't play. **UNVERIFIED**: that `mp4v` always works in the pip wheel and is not browser-playable.
  2. Re-encode it with ffmpeg: `ffmpeg -y -i tmp.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart out.mp4`. **UNVERIFIED**: I didn't fetch a primary source for these flags, but they are standard ffmpeg options. yuv420p is needed for browser and QuickTime compatibility, and +faststart moves the moov atom to the front.
- **imageio-ffmpeg is a good pip-only way to get an ffmpeg binary.**
  - Version 0.8.1 (2026-11-03) needs Python 3.10 or later, and its platform wheels "contain the binary executable of ffmpeg" for Windows x86-64/ARM64/x86-32, Linux glibc 2.28+, and macOS. Source: https://pypi.org/project/imageio-ffmpeg/
  - `imageio_ffmpeg.get_ffmpeg_exe()` returns the path. It checks, in order: the IMAGEIO_FFMPEG_EXE environment variable, the bundled binary, a conda ffmpeg, then the system ffmpeg. Source: https://github.com/imageio/imageio-ffmpeg
  - Its `write_frames` defaults to the codec `libx264`, which strongly suggests the bundled build includes libx264. **UNVERIFIED** as an explicit statement.
  - Usage: `subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)`

## Q6. Phone video rotation in cv2.VideoCapture
- `CAP_PROP_ORIENTATION_META = 48` (read-only): "Frame rotation defined by stream meta". Applies to the FFmpeg and AVFoundation back-ends only.
- `CAP_PROP_ORIENTATION_AUTO = 49`: "if true - rotates output frames of CvCapture considering video file's metadata". Applies to FFmpeg and AVFoundation only, and links to OpenCV issue #15499.
- Sources: https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html and https://raw.githubusercontent.com/opencv/opencv/4.x/modules/videoio/include/opencv2/videoio.hpp
- **UNVERIFIED**:
  - That auto-rotation is ON by default for the FFmpeg backend in recent 4.x.
  - The version that introduced it. From memory it was around 4.5.x; I could not confirm this because the versioned docs were unreachable.
- Defensive approach:
  1. Call `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)` explicitly.
  2. Log `cap.get(cv2.CAP_PROP_ORIENTATION_META)`.
  3. Check the first frame's shape against the expected portrait or landscape orientation.
- Because landmarks are normalized to the decoded frame, rotation only matters if the overlay and the analysis disagree about which way is up.
