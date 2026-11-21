"""imu.py: simulated wearable sensors from landmarks (pose.extract / synth schema)."""
import math

import numpy as np

import imu
import synth

FPS = 60
NAMES = synth.NAMES
STILL = {"shoulder": (700, 300), "hip": (500, 400), "ankle": (200, 600),
         "elbow": (700, 450), "wrist": (700, 600)}


def _landmarks(positions_at, seconds=3):
    """positions_at(t) -> joint -> (x_px, y_px); left side visible; 500 px per metre."""
    frames = []
    for i in range(FPS * seconds):
        t = i / FPS
        pos = positions_at(t)
        lm, world = {}, {}
        for n in NAMES:
            j = n[2:] if n[1] == "_" else "shoulder"
            x, y = pos.get(j, pos["shoulder"])
            lm[n] = [x, y, 0.9 if n.startswith("l_") else 0.3]
            world[n] = [x / 500, y / 500, 0.0]
        frames.append({"i": i, "t": t, "lm": lm, "world": world})
    return {"source": "test", "fps": FPS, "w": 1280, "h": 720, "n_frames": len(frames),
            "detected_frames": len(frames), "frames": frames}


def _mid(arr):
    return np.asarray(arr)[FPS // 2: -FPS // 2]


def test_stationary_sensors_read_gravity_and_no_rotation():
    out = imu.simulate(_landmarks(lambda t: STILL), noise=False)
    assert set(out["sensors"]) == {"chest", "hip", "wrist"}
    for s in out["sensors"].values():
        assert np.allclose(np.linalg.norm(_mid(s["acc"]), axis=1), imu.G, atol=1e-6)
        assert np.allclose(_mid(s["gyro"]), 0, atol=1e-6)


def test_scale_comes_from_world_landmarks():
    assert math.isclose(imu.simulate(_landmarks(lambda t: STILL))["m_per_px"], 1 / 500,
                        rel_tol=1e-6)


def test_vertical_oscillation_gives_expected_peak_acceleration():
    w = 2 * math.pi * 1.0                      # 50 px = 0.1 m at 1 Hz -> 0.1 * w^2 = 3.95 m/s^2

    def pos(t):
        return {k: (x, y + 50 * math.sin(w * t)) for k, (x, y) in STILL.items()}

    acc = _mid(imu.simulate(_landmarks(pos), noise=False, smooth_window=1)
               ["sensors"]["chest"]["acc_world"])
    assert math.isclose((acc[:, 1] - imu.G).max(), 0.1 * w ** 2, rel_tol=0.05)


def test_rotating_torso_gives_expected_gyro_rate():
    def pos(t):
        a = math.radians(20 + 30 * t)          # pitches up at 30 deg/s
        return {**STILL, "shoulder": (500 + 220 * math.cos(a), 400 - 220 * math.sin(a))}

    gyro = _mid(imu.simulate(_landmarks(pos), noise=False)["sensors"]["chest"]["gyro"])
    assert np.allclose(gyro, 30, atol=0.5)


def test_noise_is_seeded():
    lms = _landmarks(lambda t: STILL)
    a = imu.simulate(lms, seed=7)["sensors"]["wrist"]["acc"]
    assert a == imu.simulate(lms, seed=7)["sensors"]["wrist"]["acc"]
    assert a != imu.simulate(lms, noise=False)["sensors"]["wrist"]["acc"]


def test_simulated_athlete_chest_strap_sees_one_push_per_rep():
    s = synth.generate(n=1, seed=0, faults="none", camera="ideal")[0]
    out = imu.simulate(s["landmarks"], noise=False)
    assert len(out["sensors"]["chest"]["t"]) == s["landmarks"]["n_frames"]
    vy = np.asarray(out["sensors"]["chest"]["acc_world"])[:, 1] - imu.G
    above = vy > 0.5                           # upward acceleration lobe at each bottom
    assert int(np.sum(above[1:] & ~above[:-1])) == s["labels"]["rep_count"]


def test_handles_dropped_frames_and_empty_clips():
    lms = _landmarks(lambda t: STILL)
    for f in lms["frames"][10:20]:
        f["lm"] = f["world"] = None
    assert len(imu.simulate(lms)["sensors"]["hip"]["acc"]) == lms["n_frames"]
    for f in lms["frames"]:
        f["lm"] = f["world"] = None
    assert imu.simulate(lms)["sensors"] == {}
