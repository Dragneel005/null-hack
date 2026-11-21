import math

import numpy as np

from vision.imu import G, add_imu
from vision.landmarks import NAMES
from vision.synthetic import generate_pushups

FPS = 60


def _doc(positions_at):
    """positions_at(t) -> dict joint(no prefix) -> (x_px, y_px). Left side visible."""
    frames = []
    for i in range(FPS * 3):
        t = i / FPS
        pos = positions_at(t)
        lm = {}
        for n in NAMES:
            j = n.split("_", 1)[1] if n[:2] in ("l_", "r_") else "shoulder"
            x, y = pos.get(j, pos["shoulder"])
            lm[n] = [x, y, 0.9 if n.startswith("l_") else 0.3]
        frames.append({"t": t, "lm": lm})
    return {"version": 1, "fps": FPS, "w": 1280, "h": 720, "side": "left",
            "confidence": 0.9, "message": None, "frames": frames, "m_per_px": 0.002}


STILL = {"shoulder": (700, 300), "hip": (500, 400), "ankle": (200, 600),
         "elbow": (700, 450), "wrist": (700, 600)}


def _interior(arr):
    return np.asarray(arr)[FPS // 2: -FPS // 2]


def test_stationary_sensors_read_gravity_and_no_rotation():
    imu = add_imu(_doc(lambda t: STILL), noise=False)["imu"]
    for name in ("chest", "hip", "wrist"):
        acc = _interior(imu["sensors"][name]["acc"])
        assert np.allclose(np.linalg.norm(acc, axis=1), G, atol=1e-6)
        assert np.allclose(_interior(imu["sensors"][name]["gyro"]), 0, atol=1e-6)


def test_vertical_oscillation_gives_expected_peak_acceleration():
    amp_px, freq = 50.0, 1.0           # 0.1 m amplitude at 1 Hz -> peak 0.1*(2*pi)^2 = 3.95 m/s^2
    w = 2 * math.pi * freq

    def pos(t):
        dy = amp_px * math.sin(w * t)
        return {k: (x, y + dy) for k, (x, y) in STILL.items()}

    acc_world = _interior(add_imu(_doc(pos), noise=False, smooth_window=1)
                          ["imu"]["sensors"]["chest"]["acc_world"])
    vertical = acc_world[:, 1] - G
    assert math.isclose(vertical.max(), 0.1 * w**2, rel_tol=0.05)


def test_rotating_torso_gives_expected_gyro_rate():
    rate_deg = 30.0                    # torso pitches up at 30 deg/s around the hip

    def pos(t):
        a = math.radians(20 + rate_deg * t)
        hip = np.array([500.0, 400.0])
        shoulder = hip + 220 * np.array([math.cos(a), -math.sin(a)])   # y down in image
        return {**STILL, "hip": tuple(hip), "shoulder": tuple(shoulder)}

    gyro = _interior(add_imu(_doc(pos), noise=False)["imu"]["sensors"]["chest"]["gyro"])
    assert np.allclose(gyro, rate_deg, atol=0.5)


def test_noise_is_reproducible_with_a_seed_and_changes_the_signal():
    doc = _doc(lambda t: STILL)
    a = add_imu(doc, noise=True, seed=7)["imu"]["sensors"]["wrist"]["acc"]
    b = add_imu(doc, noise=True, seed=7)["imu"]["sensors"]["wrist"]["acc"]
    clean = add_imu(doc, noise=False)["imu"]["sensors"]["wrist"]["acc"]
    assert a == b and a != clean


def test_pushup_chest_sensor_has_one_vertical_cycle_per_rep():
    doc = add_imu(generate_pushups(reps=6, jitter_px=0), noise=False)
    imu = doc["imu"]
    assert imu["simulated"] is True and len(imu["sensors"]["chest"]["t"]) == len(doc["frames"])
    vy = np.asarray(imu["sensors"]["chest"]["acc_world"])[:, 1] - G
    # each rep has one deceleration-at-bottom peak (upward acceleration) above 0.5 m/s^2
    above = vy > 0.5
    rising_edges = np.sum(above[1:] & ~above[:-1])
    assert rising_edges == 6
