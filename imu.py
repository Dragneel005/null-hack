"""Theme layer — simulated wearable IMU from landmarks (real pose.extract or synth clips).

Puts virtual sensors on the body — chest strap (shoulder), waist clip (hip), smartwatch
(wrist) — and computes what a real accelerometer + gyroscope would have recorded, so
every video (or simulated athlete) also yields wearable training data.

2D, in the camera's sagittal plane:
  world frame:  x = direction the athlete faces, y = up, metres
  sensor frame: x along the sensor's body segment, y perpendicular (rotated world frame)
Accelerometers measure specific force f = a - g, so a sensor at rest reads +9.81 m/s²
"up". Gyro = angular rate of the segment in deg/s (+ = pitching up). Optional seeded
noise + bias mimic a consumer IMU.

    simulate(landmarks, side=None, noise=True, seed=0, smooth_window=None, height_m=1.75)
    (smooth_window=None -> ~0.3 s moving average; double differentiation amplifies pose jitter)
      -> {"rate_hz", "simulated": True, "side", "m_per_px", "units", "axes",
          "sensors": {"chest"|"hip"|"wrist": {"placement", "segment", "t": [...],
                      "acc": [[x, y]...], "acc_world": [[x, y]...], "gyro": [...]}}}
"""
from __future__ import annotations

import numpy as np

G = 9.81
SENSORS = {   # name: (joint the sensor sits on, segment that orients it)
    "chest": ("shoulder", ("hip", "shoulder")),
    "hip": ("hip", ("hip", "shoulder")),
    "wrist": ("wrist", ("elbow", "wrist")),
}
NOISE = {"acc_std": 0.05, "acc_bias": 0.15, "gyro_std": 0.6, "gyro_bias": 1.0}


def _series(frames, name, key="lm"):
    return np.array([f[key][name][:2] if f.get(key) else [np.nan, np.nan] for f in frames],
                    dtype=float)


def _fill(arr):
    out, idx = arr.copy(), np.arange(len(arr))
    for k in range(arr.shape[1]):
        ok = ~np.isnan(arr[:, k])
        if ok.any():
            out[:, k] = np.interp(idx, idx[ok], arr[ok, k])
    return out


def _smooth(arr, window):
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid")
                     for k in range(arr.shape[1])], axis=1)


def _pick_side(frames):
    ok = [f["lm"] for f in frames if f.get("lm")]
    vis = {s: np.mean([lm[f"{s}_{j}"][2] for lm in ok for j in ("shoulder", "elbow", "hip")])
           for s in "lr"}
    return "left" if vis["l"] >= vis["r"] else "right"


def _m_per_px(frames, s, height_m):
    px = np.linalg.norm(_series(frames, f"{s}_shoulder") - _series(frames, f"{s}_ankle"), axis=1)
    if any(f.get("world") for f in frames):
        m = np.linalg.norm(_series(frames, f"{s}_shoulder", "world")
                           - _series(frames, f"{s}_ankle", "world"), axis=1)
        r = m / px
        r = r[np.isfinite(r) & (px > 20)]
        if len(r):
            return float(np.median(r))
    px = px[np.isfinite(px)]
    return 0.78 * height_m / float(np.median(px))      # shoulder->ankle ≈ 0.78 x height


def simulate(landmarks: dict, side: str | None = None, noise: bool = True, seed: int | None = 0,
             smooth_window: int | None = None, height_m: float = 1.75) -> dict:
    """Simulated chest/hip/wrist IMU for a landmarks dict. Same length as landmarks.frames."""
    frames = landmarks["frames"]
    base = {"rate_hz": landmarks["fps"], "simulated": True, "noise": bool(noise),
            "units": {"acc": "m/s^2", "gyro": "deg/s", "t": "s"},
            "axes": {"world": "x forward (facing), y up",
                     "sensor": "x along segment, y perpendicular"}}
    if len(frames) < 3 or not any(f.get("lm") for f in frames):
        return {**base, "side": side, "m_per_px": None, "sensors": {}}

    side = side or _pick_side(frames)
    if smooth_window is None:
        smooth_window = max(5, int(round(0.3 * landmarks["fps"])) | 1)
    s = side[0]
    t = np.array([f["t"] for f in frames], dtype=float)
    scale = _m_per_px(frames, s, height_m)
    sh, an = _series(frames, f"{s}_shoulder"), _series(frames, f"{s}_ankle")
    facing = 1.0 if np.nanmedian(sh[:, 0]) >= np.nanmedian(an[:, 0]) else -1.0

    def world(joint):
        p = _smooth(_fill(_series(frames, f"{s}_{joint}")), smooth_window)
        return np.stack([facing * p[:, 0] * scale, -p[:, 1] * scale], axis=1)

    rng = np.random.default_rng(seed)
    sensors = {}
    for name, (joint, (seg_a, seg_b)) in SENSORS.items():
        pos = world(joint)
        acc = np.gradient(np.gradient(pos, t, axis=0), t, axis=0)
        f_world = acc + np.array([0.0, G])
        seg = world(seg_b) - world(seg_a)
        theta = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        gyro = np.degrees(np.gradient(theta, t))
        c, si = np.cos(theta), np.sin(theta)
        f_sensor = np.stack([c * f_world[:, 0] + si * f_world[:, 1],
                             -si * f_world[:, 0] + c * f_world[:, 1]], axis=1)
        if noise:
            f_sensor = (f_sensor + rng.normal(0, NOISE["acc_bias"], 2)
                        + rng.normal(0, NOISE["acc_std"], f_sensor.shape))
            gyro = gyro + rng.normal(0, NOISE["gyro_bias"]) + rng.normal(0, NOISE["gyro_std"],
                                                                         len(gyro))
        sensors[name] = {"placement": joint, "segment": [seg_a, seg_b],
                         "t": [round(float(v), 4) for v in t],
                         "acc": np.round(f_sensor, 4).tolist(),
                         "acc_world": np.round(f_world, 4).tolist(),
                         "gyro": np.round(gyro, 3).tolist()}
    return {**base, "side": side, "m_per_px": round(scale, 6), "sensors": sensors}


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/landmarks_demo_side.json"
    out = simulate(json.load(open(path)))
    for k, v in out["sensors"].items():
        acc = np.array(v["acc_world"])
        print(f"{k:6s} side={out['side']} |a-g| peak {np.abs(acc[:, 1] - G).max():.2f} m/s², "
              f"gyro peak {np.abs(v['gyro']).max():.1f} deg/s")
