"""Simulated wearable IMU from pose landmarks.

Places virtual sensors on the body (chest strap at the shoulder, waist clip at the
hip, smartwatch at the wrist) and computes what a real accelerometer + gyroscope
would have read during the movement.

Frames (2D, camera's sagittal plane):
  * world:  x = direction the person faces, y = up, metres.
  * sensor: x along the sensor's body segment, y perpendicular (rotated world frame).
Accelerometers measure specific force f = a - g; with g = (0, -9.81) a sensor at
rest reads +9.81 m/s^2 "up". Gyro = angular rate of the segment, deg/s, + = pitch up.
"""
import copy

import numpy as np

G = 9.81
DEFAULT_BODY_LEN_M = 1.40     # shoulder -> ankle for a ~1.75 m adult

SENSORS = {
    # name: (joint the sensor sits on, segment that defines its orientation)
    "chest": ("shoulder", ("hip", "shoulder")),
    "hip":   ("hip",      ("hip", "shoulder")),
    "wrist": ("wrist",    ("elbow", "wrist")),
}
NOISE = {"acc_std": 0.05, "acc_bias": 0.15, "gyro_std": 0.6, "gyro_bias": 1.0}


def _series(frames, name):
    """(N, 2) array of pixel x/y for one landmark; NaN where no pose."""
    return np.array([f["lm"][name][:2] if f["lm"] is not None else [np.nan, np.nan]
                     for f in frames], dtype=float)


def _fill_gaps(arr):
    out = arr.copy()
    idx = np.arange(len(arr))
    for k in range(arr.shape[1]):
        ok = ~np.isnan(arr[:, k])
        if ok.any():
            out[:, k] = np.interp(idx, idx[ok], arr[ok, k])
    return out


def _moving_avg(arr, window):
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(arr.shape[1])],
                    axis=1)


def _m_per_px(doc, frames, s, body_len_m):
    if doc.get("m_per_px"):
        return float(doc["m_per_px"])
    d = np.linalg.norm(_series(frames, f"{s}_shoulder") - _series(frames, f"{s}_ankle"), axis=1)
    d = d[~np.isnan(d)]
    return body_len_m / float(np.median(d)) if len(d) else 0.002


def add_imu(doc: dict, noise: bool = True, seed: int | None = 0,
            smooth_window: int = 5, body_len_m: float = DEFAULT_BODY_LEN_M) -> dict:
    """Return a copy of a landmarks doc with an `imu` block of simulated sensor data."""
    out = copy.deepcopy(doc)
    frames = doc["frames"]
    s = doc["side"][0]
    t = np.array([f["t"] for f in frames], dtype=float)
    if len(frames) < 3 or all(f["lm"] is None for f in frames):
        out["imu"] = {"rate_hz": doc["fps"], "simulated": True, "sensors": {}}
        return out

    scale = _m_per_px(doc, frames, s, body_len_m)
    sh, an = _series(frames, f"{s}_shoulder"), _series(frames, f"{s}_ankle")
    facing = 1.0 if np.nanmedian(sh[:, 0]) >= np.nanmedian(an[:, 0]) else -1.0

    def world(name):  # metres, x forward, y up
        p = _moving_avg(_fill_gaps(_series(frames, f"{s}_{name}")), smooth_window)
        return np.stack([facing * p[:, 0] * scale, -p[:, 1] * scale], axis=1)

    rng = np.random.default_rng(seed)
    sensors = {}
    for sname, (joint, (seg_a, seg_b)) in SENSORS.items():
        pos = world(joint)
        vel = np.gradient(pos, t, axis=0)
        acc = np.gradient(vel, t, axis=0)
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
            gyro = gyro + rng.normal(0, NOISE["gyro_bias"]) + rng.normal(0, NOISE["gyro_std"], len(gyro))

        sensors[sname] = {
            "placement": joint, "segment": [seg_a, seg_b],
            "t": [round(float(v), 4) for v in t],
            "acc": np.round(f_sensor, 4).tolist(),
            "acc_world": np.round(f_world, 4).tolist(),
            "gyro": np.round(gyro, 3).tolist(),
        }

    out["imu"] = {
        "rate_hz": doc["fps"], "simulated": True, "noise": bool(noise),
        "m_per_px": round(scale, 6),
        "units": {"acc": "m/s^2", "gyro": "deg/s", "t": "s"},
        "axes": {"world": "x forward, y up", "sensor": "x along segment, y perpendicular"},
        "sensors": sensors,
    }
    return out
