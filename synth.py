"""Theme layer — simulated athletes for physical AI (see CONTRACT.md §6).

A procedural 3D kinematic push-up model that generates *labeled* synthetic
pose data in exactly the same schema as `pose.extract`:

1. Anthropometrics: height 1.5-2.0 m, segment lengths as fractions of height
   (Drillis & Contini / de Leva) with per-athlete variation.
2. Kinematics: hands and toes planted on the floor, body pivots about the toes.
   Each rep is a tempo profile (top hold -> descent -> bottom pause -> ascent)
   with cosine easing on the elbow angle. For every frame the shoulder position
   is solved by intersecting the "arm circle" (shoulder-wrist distance implied by
   the elbow angle, law of cosines) with the "body circle" (ankle-shoulder
   distance implied by the hip bend), then the elbow is placed with 3D 2-link IK.
3. Faults with magnitudes (SHALLOW, NO_LOCKOUT, HIP_SAG, HIP_PIKE, HEAD_DROP,
   FAST_DESCENT) and fatigue drift (form degrades later in the set). Per-rep
   ground-truth labels are derived from the *simulated* joint angles using
   fixed, documented thresholds.
4. Camera: pinhole, side-ish view with random yaw, elevation, distance, roll,
   fov, fps, resolution; projection to pixels, Gaussian + temporal keypoint
   noise, camera shake, visibility (lower on the far side), dropouts, glitches,
   and the same 5-frame smoothing that `pose.extract` applies.

numpy only; ~500 samples in a few seconds.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

NAMES = ["nose", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
         "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
         "l_heel", "r_heel", "l_foot", "r_foot"]
IDX = {n: i for i, n in enumerate(NAMES)}
FAULTS = ["SHALLOW", "NO_LOCKOUT", "HIP_SAG", "HIP_PIKE", "HEAD_DROP", "FAST_DESCENT"]

# Label thresholds applied to the simulated ground-truth angles/timings (per rep).
# Kept identical to analysis.DEFAULT_THRESHOLDS so simulator labels and the analyzer speak the same language.
LABEL_THRESHOLDS = {
    "SHALLOW": "min_elbow > 100 deg",
    "NO_LOCKOUT": "max_elbow < 150 deg",
    "HIP_SAG": "max hip bend (sag, hip below shoulder-ankle line) > 12 deg",
    "HIP_PIKE": "max hip bend (pike, hip above line) > 15 deg",
    "HEAD_DROP": "min head angle (ear-shoulder-hip) < 124 deg",
    "FAST_DESCENT": "descent duration < 0.40 s",
}
LABEL_LIMITS = {"SHALLOW": 100.0, "NO_LOCKOUT": 150.0, "HIP_SAG": 12.0, "HIP_PIKE": 15.0,
                "HEAD_DROP": 124.0, "FAST_DESCENT": 0.40}   # HEAD_DROP = 180 - analysis head_drop_deg (56)

# Segment lengths as fraction of stature (Drillis & Contini 1966; de Leva 1996).
SEGMENTS = {
    "upper_arm": 0.186, "forearm": 0.146, "thigh": 0.245, "shank": 0.246,
    "trunk": 0.288, "neck_ear": 0.105, "foot": 0.152, "ankle_h": 0.039,
    "shoulder_half_w": 0.100, "hip_half_w": 0.058,
}

SKELETON = [("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
            ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("l_shoulder", "l_hip"),
            ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_heel"),
            ("l_heel", "l_foot"), ("l_ankle", "l_foot"), ("l_ear", "l_shoulder"),
            ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("r_shoulder", "r_hip"),
            ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_heel"),
            ("r_heel", "r_foot"), ("r_ankle", "r_foot"), ("r_ear", "r_shoulder"),
            ("nose", "l_ear"), ("nose", "r_ear")]

RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080), (960, 540), (720, 1280)]
FPS_CHOICES = [24.0, 30.0, 60.0]


# ----------------------------------------------------------------------------- helpers
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _ease_knots(t: np.ndarray, kt: np.ndarray, kv: np.ndarray) -> np.ndarray:
    """Piecewise cosine-eased interpolation through knots (kt, kv)."""
    k = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
    dt = np.maximum(kt[k + 1] - kt[k], 1e-9)
    u = np.clip((t - kt[k]) / dt, 0.0, 1.0)
    e = 0.5 - 0.5 * np.cos(np.pi * u)
    return kv[k] + (kv[k + 1] - kv[k]) * e


def _smooth_noise(rng, n: int, dims: tuple, alpha: float) -> np.ndarray:
    """Low-pass (EMA) Gaussian noise with unit-ish std, shape (n, *dims)."""
    win = max(1, int(round((1 + alpha) / (1 - alpha))))      # box filter ~ same bandwidth as EMA
    w = rng.normal(size=(n + 2 * win, *dims))
    c = np.cumsum(np.concatenate([np.zeros((1, *dims)), w]), axis=0)
    out = (c[win:win + n + win] - c[:n + win])[:n] / win
    return out * np.sqrt(win)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis 0 (edge-truncated), like pose.smooth."""
    if window <= 1 or len(x) < 2:
        return x
    half = window // 2
    c = np.cumsum(np.concatenate([np.zeros((1, *x.shape[1:])), x]), axis=0)
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    shape = (-1,) + (1,) * (x.ndim - 1)
    return (c[hi] - c[lo]) / (hi - lo).reshape(shape)


# ----------------------------------------------------------------------------- athlete
def _sample_athlete(rng) -> dict:
    H = float(rng.uniform(1.50, 2.00))
    seg = {k: float(v * H * np.clip(rng.normal(1.0, 0.035), 0.92, 1.08)) for k, v in SEGMENTS.items()}
    bmi = float(rng.uniform(19, 30))
    return {"height_m": round(H, 3), "body_mass_kg": round(bmi * H * H, 1), "bmi": round(bmi, 1),
            "segments_m": {k: round(v, 4) for k, v in seg.items()},
            "hand_half_w_m": round(seg["shoulder_half_w"] * float(rng.uniform(1.0, 1.6)), 4),
            "hand_offset_m": round(float(rng.uniform(-0.03, 0.09)), 4),   # hands forward of shoulders
            "elbow_flare": round(float(rng.uniform(0.1, 0.55)), 3)}       # 0 tucked .. 0.55 moderate flare


def _choose_faults(rng, faults) -> list[str]:
    if faults in (None, "none", "clean"):
        return []
    if isinstance(faults, str) and faults in FAULTS:
        return [faults]
    if isinstance(faults, (list, tuple)):
        return [f for f in faults if f in FAULTS]
    # "mixed": ~35% clean, else 1-2 faults (no contradictory combos)
    if rng.random() < 0.35:
        return []
    k = 1 if rng.random() < 0.7 else 2
    chosen: list[str] = []
    for f in rng.permutation(FAULTS):
        if len(chosen) >= k:
            break
        if {f, *chosen} >= {"HIP_SAG", "HIP_PIKE"} or {f, *chosen} >= {"SHALLOW", "NO_LOCKOUT"}:
            continue
        chosen.append(str(f))
    return chosen


def _plan_reps(rng, faults: list[str], fatigue: bool) -> tuple[list[dict], dict]:
    """Per-rep targets (angles, timings) with fault magnitudes and fatigue drift."""
    n = int(rng.integers(3, 13))
    base = {
        "min_elbow": rng.uniform(40, 85), "max_elbow": rng.uniform(162, 177),
        "hip": rng.normal(0, 2.5), "head": rng.uniform(140, 165),   # calibrated: real MediaPipe good form ~131-136 at bottom
        "descent": rng.uniform(0.6, 1.9), "ascent": rng.uniform(0.5, 1.5),
        "bottom": rng.uniform(0.0, 0.35), "top": rng.uniform(0.1, 1.6),
    }
    mags = {
        "SHALLOW": rng.uniform(112, 135), "NO_LOCKOUT": rng.uniform(122, 142),
        "HIP_SAG": rng.uniform(15, 35), "HIP_PIKE": -rng.uniform(15, 35),
        "HEAD_DROP": rng.uniform(98, 122), "FAST_DESCENT": rng.uniform(0.18, 0.35),
    }
    onset = {f: int(rng.integers(0, max(1, n // 2))) for f in faults}  # fault may appear mid-set
    fat_from = int(rng.integers(max(1, n // 3), n)) if fatigue and n >= 4 else None
    fat_rate = float(rng.uniform(0.15, 0.4))
    reps = []
    for i in range(n):
        fat = 0.0 if fat_from is None or i < fat_from else min(1.0, (i - fat_from + 1) * fat_rate)
        prog = 0.65 + 0.35 * i / max(1, n - 1)          # faults worsen later in the set
        j = lambda s: float(rng.normal(0, s))              # rep-to-rep variability
        r = {
            "min_elbow": base["min_elbow"] + 14 * fat + j(2.5),
            "max_elbow": base["max_elbow"] - 6 * fat + j(1.5),
            "hip": base["hip"] + 7 * fat + j(1.0),
            "head": base["head"] - 8 * fat + j(2.0),
            "descent": base["descent"] * (1 + j(0.07)),
            "ascent": base["ascent"] * (1 + 0.7 * fat + j(0.07)),
            "bottom": max(0.0, base["bottom"] + j(0.05)),
            "top": max(0.05, base["top"] * (1 + j(0.2)) + 0.5 * fat),
            "fatigue": round(fat, 3),
        }
        for f in faults:
            if i < onset[f]:
                continue
            s = prog * (1 + 0.25 * fat)
            if f == "SHALLOW":
                r["min_elbow"] = mags[f] * (0.9 + 0.1 * s) + j(2)
            elif f == "NO_LOCKOUT":
                r["max_elbow"] = mags[f] - 6 * (s - 0.65) + j(1.5)
            elif f in ("HIP_SAG", "HIP_PIKE"):
                r["hip"] = mags[f] * s + j(1.0)
            elif f == "HEAD_DROP":
                r["head"] = mags[f] - 8 * (s - 0.65) + j(2)
            elif f == "FAST_DESCENT":
                r["descent"] = mags[f] * (1 + j(0.08))
        r["min_elbow"] = float(np.clip(r["min_elbow"], 35, 140))
        r["max_elbow"] = float(np.clip(r["max_elbow"], max(r["min_elbow"] + 25, 125), 179))
        r["descent"] = float(max(0.15, r["descent"]))
        r["ascent"] = float(max(0.3, r["ascent"]))
        reps.append(r)
    meta = {"n_reps": n, "fault_onset_rep": {f: onset[f] + 1 for f in faults},
            "fault_magnitudes": {f: round(float(mags[f]), 2) for f in faults},
            "fatigue_from_rep": None if fat_from is None else fat_from + 1,
            "fatigue_rate": round(fat_rate, 3) if fat_from is not None else 0.0,
            "base_tempo": {k: round(float(base[k]), 3) for k in ("descent", "ascent", "bottom", "top")}}
    return reps, meta


# ----------------------------------------------------------------------------- kinematics
def _kinematics(ath: dict, reps: list[dict], fps: float, rng):
    """Build knot timeline and solve 3D joint positions per frame.

    World frame: X forward (toward head), Y up, Z lateral (person's left = -Z). Toes at origin.
    Returns t, P (T,19,3), truth dict, rep label timings.
    """
    s = ath["segments_m"]
    lu, lf, lt, ls, ltr = s["upper_arm"], s["forearm"], s["thigh"], s["shank"], s["trunk"]
    L_leg = lt + ls
    lead = float(rng.uniform(0.4, 1.5))
    # knots: (time, elbow, hip_dev, head, phase-of-following-segment)
    kt, ke, kh, kd, kp = [0.0], [reps[0]["max_elbow"]], [reps[0]["hip"] * 0.6], [reps[0]["head"] * 0.3 + 170 * 0.7], []
    t = lead
    kt.append(t); ke.append(reps[0]["max_elbow"]); kh.append(reps[0]["hip"] * 0.6); kd.append(kd[-1]); kp.append("top")
    rep_t = []
    for i, r in enumerate(reps):
        sag = r["hip"]
        top_hip = sag * (0.6 if sag > 0 else 0.9)
        t0 = t
        t += r["descent"]; kt.append(t); ke.append(r["min_elbow"]); kh.append(sag); kd.append(r["head"]); kp.append("down")
        tb = t + r["bottom"] / 2
        if r["bottom"] > 0.02:
            t += r["bottom"]; kt.append(t); ke.append(r["min_elbow"]); kh.append(sag); kd.append(r["head"]); kp.append("bottom")
        t += r["ascent"]; kt.append(t); ke.append(r["max_elbow"]); kh.append(top_hip); kd.append(170 * 0.6 + r["head"] * 0.4); kp.append("up")
        rep_t.append((t0, tb, t))
        nxt = reps[i + 1] if i + 1 < len(reps) else r
        hold = r["top"] if i + 1 < len(reps) else float(rng.uniform(0.4, 1.5))
        t += hold; kt.append(t); ke.append(nxt["max_elbow"]); kh.append(nxt["hip"] * (0.6 if nxt["hip"] > 0 else 0.9))
        kd.append(kd[-1]); kp.append("top")
    kt, ke, kh, kd = map(np.asarray, (kt, ke, kh, kd))
    T = int(np.floor(kt[-1] * fps)) + 1
    tt = np.arange(T) / fps
    elbow = _ease_knots(tt, kt, ke) + 0.4 * _smooth_noise(rng, T, (), 0.8)
    hipdev = _ease_knots(tt, kt, kh) + 0.6 * _smooth_noise(rng, T, (), 0.9)
    head = _ease_knots(tt, kt, kd) + 1.0 * _smooth_noise(rng, T, (), 0.9)
    seg_idx = np.clip(np.searchsorted(kt, tt, side="right") - 1, 0, len(kp) - 1)
    phase = [kp[k] for k in seg_idx]

    # --- fixed supports
    ankle = np.array([0.03, s["ankle_h"] + 0.06]) * np.array([ath["height_m"] / 1.75, 1.0])  # on toes
    sw, hw, hipw = s["shoulder_half_w"], ath["hand_half_w_m"], s["hip_half_w"]
    dz = hw - sw
    wrist_y = 0.035
    # where would the shoulder be at full extension with a straight body -> place hands there
    d_full = np.sqrt(max((lu + lf) ** 2 - dz * dz, 1e-6))
    L_as0 = L_leg + ltr
    sy0 = wrist_y + d_full
    sx0 = ankle[0] + np.sqrt(max(L_as0 ** 2 - (sy0 - ankle[1]) ** 2, 1e-6))
    W = np.array([sx0 + ath["hand_offset_m"], wrist_y])

    # --- body circle radius from hip bend (interior angle 180-|dev|)
    g = np.radians(180.0 - np.abs(hipdev))
    L_as = np.sqrt(L_leg ** 2 + ltr ** 2 - 2 * L_leg * ltr * np.cos(g))
    # --- arm circle radius from elbow angle (3D), projected to sagittal plane
    phi = np.radians(elbow)
    d3 = np.sqrt(lu ** 2 + lf ** 2 - 2 * lu * lf * np.cos(phi))
    d_sag = np.sqrt(np.maximum(d3 ** 2 - dz ** 2, 1e-6))
    # circle-circle intersection: centres A (ankle) r=L_as, W (wrist) r=d_sag -> upper solution
    AW = W - ankle
    D = np.linalg.norm(AW)
    a = (L_as ** 2 - d_sag ** 2 + D ** 2) / (2 * D)
    h = np.sqrt(np.maximum(L_as ** 2 - a ** 2, 0.0))
    ex = AW / D
    ey = np.array([-ex[1], ex[0]])
    if ey[1] < 0:
        ey = -ey
    S2 = ankle + a[:, None] * ex + h[:, None] * ey                      # (T,2) shoulder sagittal
    # hip: apex of triangle ankle-hip-shoulder; below the line for sag (+), above for pike (-)
    u_as = _unit(S2 - ankle)
    cos_al = np.clip((L_leg ** 2 + L_as ** 2 - ltr ** 2) / (2 * L_leg * L_as), -1, 1)
    al = np.arccos(cos_al) * np.where(hipdev >= 0, -1.0, 1.0)
    ca, sa = np.cos(al), np.sin(al)
    u_leg = np.stack([u_as[:, 0] * ca - u_as[:, 1] * sa, u_as[:, 0] * sa + u_as[:, 1] * ca], 1)
    Hp2 = ankle + L_leg * u_leg
    K2 = ankle + ls * u_leg + np.array([0, 0.004]) * 0  # straight knees
    n_up = np.stack([-u_leg[:, 1], u_leg[:, 0]], 1)
    n_up = np.where(n_up[:, 1:2] < 0, -n_up, n_up)
    heel2 = ankle + 0.06 * n_up * (ath["height_m"] / 1.75) - 0.015 * u_leg
    toe2 = np.zeros_like(heel2)
    # head: rotate trunk direction (hip->shoulder) about the shoulder by (180-head) downwards
    u_tr = _unit(S2 - Hp2)
    drop = np.radians(180.0 - head)
    cd, sd = np.cos(-drop), np.sin(-drop)
    u_nk = np.stack([u_tr[:, 0] * cd - u_tr[:, 1] * sd, u_tr[:, 0] * sd + u_tr[:, 1] * cd], 1)
    ear2 = S2 + s["neck_ear"] * u_nk
    n_dn = np.stack([u_nk[:, 1], -u_nk[:, 0]], 1)                       # face points down
    nose2 = ear2 + 0.06 * u_nk * (ath["height_m"] / 1.75) + 0.05 * n_dn

    T_ = len(tt)
    P = np.zeros((T_, 19, 3))

    def put(name, xy, z):
        P[:, IDX[name], 0] = xy[..., 0]
        P[:, IDX[name], 1] = xy[..., 1]
        P[:, IDX[name], 2] = z

    for side, sg in (("l", -1.0), ("r", 1.0)):
        put(f"{side}_shoulder", S2, sg * sw)
        put(f"{side}_hip", Hp2, sg * hipw)
        put(f"{side}_knee", K2, sg * hipw * 0.95)
        put(f"{side}_ankle", np.broadcast_to(ankle, (T_, 2)), sg * hipw * 0.9)
        put(f"{side}_heel", heel2, sg * hipw * 0.9)
        put(f"{side}_foot", toe2, sg * hipw * 1.0)
        put(f"{side}_wrist", np.broadcast_to(W, (T_, 2)), sg * hw)
        put(f"{side}_ear", ear2, sg * 0.075 * ath["height_m"] / 1.75)
        # 3D 2-link IK for the elbow: pole = backwards (-X), up, and flared outwards
        Sx = P[:, IDX[f"{side}_shoulder"]]
        Wx = P[:, IDX[f"{side}_wrist"]]
        dv = Wx - Sx
        dd = np.linalg.norm(dv, axis=1, keepdims=True)
        u = dv / dd
        pole = np.array([-1.0, 0.35, sg * ath["elbow_flare"]])
        v = _unit(pole[None, :] - (u @ pole)[:, None] * u)
        aa = (lu ** 2 - lf ** 2 + dd ** 2) / (2 * dd)
        hh = np.sqrt(np.maximum(lu ** 2 - aa ** 2, 0.0))
        P[:, IDX[f"{side}_elbow"]] = Sx + aa * u + hh * v
    put("nose", nose2, 0.0)

    truth = {"elbow": elbow, "hip_dev": hipdev, "head": head, "phase": phase}
    return tt, P, truth, rep_t


# ----------------------------------------------------------------------------- camera
def _sample_camera(rng, mode: str) -> dict:
    W_, H_ = RESOLUTIONS[int(rng.choice(len(RESOLUTIONS), p=[0.25, 0.35, 0.2, 0.1, 0.1]))]
    fps = float(rng.choice(FPS_CHOICES, p=[0.2, 0.6, 0.2]))
    if mode == "side":
        yaw, elev, roll, frac = 0.0, 5.0, 0.0, 0.7
    else:
        yaw = float(rng.uniform(-25, 25))
        elev = float(rng.uniform(-3, 25))
        roll = float(rng.normal(0, 1.5))
        frac = float(rng.uniform(0.45, 0.78))
    return {"w": int(W_), "h": int(H_), "fps": fps, "yaw_deg": round(yaw, 2),
            "elevation_deg": round(elev, 2), "roll_deg": round(roll, 2),
            "hfov_deg": round(float(rng.uniform(55, 75)), 2), "body_frac_of_width": round(frac, 3),
            "side": "left" if rng.random() < 0.6 else "right",          # which side of the body faces camera
            "handheld": bool(rng.random() < 0.4 and mode != "side"),
            "offset_frac": [round(float(rng.normal(0, 0.06)), 3), round(float(rng.normal(-0.04, 0.05)), 3)]}


def _project(P: np.ndarray, cam: dict, H: float):
    """Pinhole projection. Returns pixels (T,19,2), depth (T,19), camera axes, centre."""
    target = np.array([0.42 * H, 0.22, 0.0])
    sgn = -1.0 if cam["side"] == "left" else 1.0           # person's left is -Z
    yaw, el = np.radians(cam["yaw_deg"]), np.radians(cam["elevation_deg"])
    dir_h = np.array([np.sin(yaw), 0.0, sgn * np.cos(yaw)])
    f = (cam["w"] / 2) / np.tan(np.radians(cam["hfov_deg"]) / 2)
    dist = (0.95 * H) * f / (cam["body_frac_of_width"] * cam["w"])
    C = target + dist * (np.cos(el) * dir_h + np.array([0, np.sin(el), 0]))
    fwd = _unit(target - C)
    right = _unit(np.cross(fwd, np.array([0.0, 1.0, 0.0])))
    down = np.cross(fwd, right)
    r = np.radians(cam["roll_deg"])
    right, down = np.cos(r) * right + np.sin(r) * down, -np.sin(r) * right + np.cos(r) * down
    rel = P - C
    xc, yc, zc = rel @ right, rel @ down, rel @ fwd
    cx = cam["w"] / 2 - cam["offset_frac"][0] * cam["w"]
    cy = cam["h"] / 2 - cam["offset_frac"][1] * cam["h"]
    uv = np.stack([cx + f * xc / zc, cy + f * yc / zc], -1)
    return uv, zc, (right, down, fwd), float(dist), float(f)


# ----------------------------------------------------------------------------- labels
def _rep_labels(tt, truth, rep_t) -> list[dict]:
    out = []
    for n, (t0, tb, t1) in enumerate(rep_t, 1):
        m = (tt >= t0) & (tt <= t1)
        if not m.any():
            continue
        el, hd, hp = truth["elbow"][m], truth["head"][m], truth["hip_dev"][m]
        mi = int(np.argmin(el))
        rep = {"n": n, "start": round(float(t0), 3), "bottom": round(float(tt[m][mi]), 3), "end": round(float(t1), 3),
               "min_elbow": round(float(el.min()), 1), "max_elbow": round(float(el.max()), 1),
               "max_hip_sag": round(float(max(0.0, hp.max())), 1), "max_hip_pike": round(float(max(0.0, -hp.min())), 1),
               "min_head": round(float(hd.min()), 1),
               "descent_s": 0.0, "ascent_s": 0.0}
        out.append(rep)
    return out


def _finalize_labels(reps_lab, plan, rep_t) -> None:
    for rep, r, (t0, tb, t1) in zip(reps_lab, plan, rep_t):
        rep["descent_s"] = round(r["descent"], 3)
        rep["ascent_s"] = round(r["ascent"], 3)
        rep["fatigue"] = r["fatigue"]
        f = []
        L = LABEL_LIMITS
        if rep["min_elbow"] > L["SHALLOW"]: f.append("SHALLOW")
        if rep["max_elbow"] < L["NO_LOCKOUT"]: f.append("NO_LOCKOUT")
        if rep["max_hip_sag"] > L["HIP_SAG"]: f.append("HIP_SAG")
        if rep["max_hip_pike"] > L["HIP_PIKE"]: f.append("HIP_PIKE")
        if rep["min_head"] < L["HEAD_DROP"]: f.append("HEAD_DROP")
        if rep["descent_s"] < L["FAST_DESCENT"]: f.append("FAST_DESCENT")
        rep["faults"] = f


def _ang2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorised analysis.angle: angle ABC (deg) for (T,2) arrays."""
    u, v = a - b, c - b
    den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u, v) / np.maximum(den, 1e-9), -1, 1)))


def _below(origin: np.ndarray, toward: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Vectorised sign of analysis._signed_dev: +1 if point is below the origin->toward line (image y down)."""
    d = toward - origin
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n = np.where(n[:, 1:2] < 0, -n, n)
    return np.where(np.einsum("ij,ij->i", point - origin, n) >= 0, 1.0, -1.0)


def _add_2d_labels(reps_lab: list[dict], tt: np.ndarray, uv: np.ndarray, near: str) -> None:
    """Add what a *perfect* 2D tracker would measure (noise-free projection, analyzer geometry).

    `faults` stay the 3D ground truth (what the athlete really did). `faults_2d` / `*_2d` are the same rules
    applied to the projected 2D angles, so `faults != faults_2d` is pure camera-projection error (elbow flare,
    camera yaw/elevation), separate from tracking noise and analyzer bugs.
    """
    g = lambda nm: uv[:, IDX[f"{near}_{nm}"]]  # noqa: E731
    sh, el, wr, hp, an, ear = g("shoulder"), g("elbow"), g("wrist"), g("hip"), g("ankle"), g("ear")
    elbow = _ang2d(sh, el, wr)
    hip = (180.0 - _ang2d(sh, hp, an)) * _below(sh, an, hp)
    head_dev = (180.0 - _ang2d(ear, sh, hp)) * _below(hp, sh, ear)
    L = LABEL_LIMITS
    for rep in reps_lab:
        m = (tt >= rep["start"] - 1e-6) & (tt <= rep["end"] + 1e-6)
        if not m.any():
            rep["faults_2d"] = list(rep["faults"])
            continue
        r2 = {"min_elbow_2d": round(float(elbow[m].min()), 1), "max_elbow_2d": round(float(elbow[m].max()), 1),
              "max_hip_sag_2d": round(float(max(0.0, hip[m].max())), 1),
              "max_hip_pike_2d": round(float(max(0.0, -hip[m].min())), 1),
              "min_head_2d": round(float(180.0 - max(0.0, head_dev[m].max())), 1)}
        rep.update(r2)
        f = []
        if r2["min_elbow_2d"] > L["SHALLOW"]: f.append("SHALLOW")
        if r2["max_elbow_2d"] < L["NO_LOCKOUT"]: f.append("NO_LOCKOUT")
        if r2["max_hip_sag_2d"] > L["HIP_SAG"]: f.append("HIP_SAG")
        if r2["max_hip_pike_2d"] > L["HIP_PIKE"]: f.append("HIP_PIKE")
        if r2["min_head_2d"] < L["HEAD_DROP"]: f.append("HEAD_DROP")
        if "FAST_DESCENT" in rep["faults"]: f.append("FAST_DESCENT")
        rep["faults_2d"] = f


# ----------------------------------------------------------------------------- sample
def _one(rng, sid: int, faults, camera: str, noise: float) -> dict:
    ath = _sample_athlete(rng)
    chosen = _choose_faults(rng, faults)
    fatigue = bool(rng.random() < 0.5)
    plan, meta = _plan_reps(rng, chosen, fatigue)
    cam = _sample_camera(rng, camera)
    fps = cam["fps"]
    tt, P, truth, rep_t = _kinematics(ath, plan, fps, rng)
    T = len(tt)
    uv, depth, (right, down, fwd), dist, focal = _project(P, cam, ath["height_m"])
    uv_clean = uv.copy()               # noise-free 2D projection: what a perfect 2D keypoint tracker would see

    # --- sensor model
    nz = noise if camera != "side" else noise * 0.5
    sig = nz * float(rng.uniform(0.0025, 0.006)) * cam["w"]
    near = "l" if cam["side"] == "left" else "r"
    far_mask = np.array([n.startswith(("l_", "r_")) and not n.startswith(near + "_") for n in NAMES])
    sig_k = np.where(far_mask, 2.0, 1.0) * sig
    uv = uv + sig_k[None, :, None] * (0.6 * rng.normal(size=uv.shape) + 0.8 * _smooth_noise(rng, T, (19, 2), 0.7))
    if cam["handheld"]:
        uv = uv + nz * 0.01 * cam["w"] * _smooth_noise(rng, T, (1, 2), 0.97)
    yaw_open = abs(cam["yaw_deg"]) / 25.0
    vis = np.where(far_mask, rng.uniform(0.35, 0.7) + 0.2 * yaw_open, rng.uniform(0.9, 0.99))
    vis = vis[None, :] + 0.04 * _smooth_noise(rng, T, (19,), 0.8)
    for nm in ("nose", "l_ear", "r_ear"):
        vis[:, IDX[nm]] += 0.25 if not nm.startswith(("l_", "r_")) else 0.0
    # glitches: rare single-keypoint jumps with low visibility
    if nz > 0:
        g = rng.random((T, 19)) < 0.002 * nz
        uv[g] += rng.normal(0, 0.05 * cam["w"], size=(int(g.sum()), 2))
        vis[g] *= 0.4
    out = (uv[..., 0] < 0) | (uv[..., 0] > cam["w"]) | (uv[..., 1] < 0) | (uv[..., 1] > cam["h"])
    vis[out] *= 0.3
    vis = np.clip(vis, 0.0, 1.0)
    uv = _moving_average(uv, 5)                 # same as pose.extract(smooth_window=5)
    # dropouts: whole frames with no person
    present = np.ones(T, bool)
    if nz > 0 and rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 5))):
            a = int(rng.integers(0, T)); present[a:a + int(rng.integers(1, int(0.3 * fps) + 2))] = False

    # MediaPipe-style world landmarks: hip-centred, metres, camera-aligned axes (x right, y down)
    hipc = 0.5 * (P[:, IDX["l_hip"]] + P[:, IDX["r_hip"]])
    rel = P - hipc[:, None, :]
    world = np.stack([rel @ right, rel @ down, rel @ fwd], -1) + rng.normal(0, 0.012 * nz + 1e-4, size=P.shape)

    lm_arr = np.concatenate([uv, vis[..., None]], -1)
    lm_l = np.round(lm_arr, 2).tolist()
    w_l = np.round(world, 4).tolist()
    t_l = np.round(tt, 4).tolist()
    frames = []
    for i in range(T):
        if present[i]:
            frames.append({"i": i, "t": t_l[i], "lm": dict(zip(NAMES, lm_l[i])), "world": dict(zip(NAMES, w_l[i]))})
        else:
            frames.append({"i": i, "t": t_l[i], "lm": None, "world": None})

    reps_lab = _rep_labels(tt, truth, rep_t)
    _finalize_labels(reps_lab, plan, rep_t)
    _add_2d_labels(reps_lab, tt, uv_clean, "l" if cam["side"] == "left" else "r")
    landmarks = {"source": f"synth_{sid:05d}", "fps": fps, "w": cam["w"], "h": cam["h"],
                 "n_frames": T, "detected_frames": int(present.sum()), "frames": frames}
    labels = {
        "rep_count": len(reps_lab), "reps": reps_lab,
        "faults_injected": chosen,
        "fatigue_from_rep": meta["fatigue_from_rep"],
        "near_side": "left" if near == "l" else "right",
        "truth": {"elbow": np.round(truth["elbow"], 2).tolist(), "hip_dev": np.round(truth["hip_dev"], 2).tolist(),
                  "head": np.round(truth["head"], 2).tolist(), "phase": truth["phase"]},
        "params": {"athlete": ath, "plan": meta,
                   "camera": {**cam, "distance_m": round(dist, 3), "focal_px": round(focal, 1),
                              "noise_px_sigma": round(sig, 2), "noise_scale": noise}},
    }
    return {"id": sid, "landmarks": landmarks, "labels": labels}


def generate(n: int = 500, seed: int = 0, faults="mixed", camera: str = "random",
             noise: float = 1.0) -> list[dict]:
    """Generate `n` simulated push-up athletes.

    faults: "mixed" | "none" | a fault code | list of codes.  camera: "random" | "side" (clean, lateral).
    noise: scale for keypoint noise / glitches / dropouts (0 = perfect).
    Each sample: {"id", "landmarks": <pose.extract schema>, "labels": {...}}.
    """
    import gc
    rng = np.random.default_rng(seed)
    was = gc.isenabled()
    gc.disable()                      # millions of small floats/lists: GC passes dominate otherwise
    try:
        return [_one(np.random.default_rng(rng.integers(2 ** 63)), i, faults, camera, noise) for i in range(n)]
    finally:
        if was:
            gc.enable()


# ----------------------------------------------------------------------------- export
def _datasheet(samples: list[dict]) -> str:
    st = dataset_stats(samples)
    fc = "\n".join(f"| {k} | {v} | {LABEL_THRESHOLDS[k]} |" for k, v in st["fault_rep_counts"].items())
    return f"""# FormSim synthetic push-up dataset — DATASHEET

Generated by `synth.py` (procedural 3D kinematic model + pinhole camera + sensor noise).

## Contents
- `dataset.jsonl` — one sample per line: `{{"id", "landmarks", "labels"}}`. `landmarks` has exactly the
  schema of `pose.extract` (19 named keypoints, pixel `[x, y, visibility]`, `world` = metres, hip-centred,
  camera-aligned, y down; `null` frames = person dropout).
- `frames.csv` — long-ish table: sample_id, frame, t, present, then `<name>_x, <name>_y, <name>_vis` (pixels).
- `labels.csv` — one row per rep with ground-truth timings, angles and fault labels.

## Size
{st['n_samples']} samples, {st['total_reps']} reps, {st['total_frames']} frames, {st['total_duration_s']:.0f} s of motion.
Clean reps: {st['clean_rep_frac']*100:.0f}%.

## Generation parameters
- Athlete: height U(1.50, 2.00) m; segment lengths = stature fraction (Drillis & Contini / de Leva:
  upper arm .186, forearm .146, thigh .245, shank .246, trunk .288) x N(1, 0.035) clipped to ±8%;
  hand width 1.0-1.6x shoulder width; hands -3..+9 cm forward of shoulders; elbow flare 0.1-0.55 (lateral pole component; tucked to moderate).
- Kinematics: toes & hands planted; body pivots about toes. Per frame the shoulder is the intersection of
  the arm circle (wrist-shoulder distance from elbow angle via law of cosines) and the body circle
  (ankle-shoulder distance from hip bend); elbow by 3D two-link IK with a backward/flared pole vector.
- Tempo per rep: top hold U(0.1,1.6) s, descent U(0.6,1.9) s, bottom pause U(0,0.35) s, ascent U(0.5,1.5) s,
  cosine easing; 3-12 reps per clip; rep-to-rep jitter.
- Clean form: min elbow U(40,85)°, max elbow U(162,177)°, hip bend N(0,2.5)°, head angle U(140,165)°
  (head range calibrated to real MediaPipe output: good form reads ~131-136° at the bottom).
- Faults ("mixed": 35% clean clips, else 1-2 faults; may start mid-set, worsen later in set):
  SHALLOW min elbow 112-135°, NO_LOCKOUT max elbow 122-142°, HIP_SAG +15..35°, HIP_PIKE -15..-35°,
  HEAD_DROP head angle 98-122°, FAST_DESCENT descent 0.18-0.35 s. Labels use the same thresholds as the analyzer.
- Fatigue (50% of clips): from a random rep, depth shrinks (+14°), lockout drops, hips sag (+7°),
  head drops, ascent slows (+70%), top holds lengthen.
- Camera: pinhole, view from the athlete's left (60%) or right side, yaw U(-25,25)°, elevation U(-3,25)°,
  roll N(0,1.5)°, hfov U(55,75)°, body spans 45-78% of frame width; resolutions {sorted(st['resolutions'])};
  fps 24/30/60.
- Sensor: Gaussian + temporally-correlated keypoint noise (σ = 0.25-0.6% of width, 2x on far-side limbs),
  handheld shake (40%), visibility 0.9-1.0 near side / 0.35-0.9 far side, rare glitches, whole-frame
  dropouts (50% of clips), then 5-frame centered moving average (as `pose.extract`).

## Label definitions (per rep, from simulated ground-truth 3D angles)
`faults` = 3D ground truth (what the athlete really did). Each rep also carries `faults_2d` and `*_2d` angles:
the same rules on the noise-free 2D projection (what a perfect side-view keypoint tracker would see).
Where they differ, the label is invisible/ambiguous in 2D (elbow flare, camera yaw) — the projection gap.
| code | reps labelled | rule |
|---|---|---|
{fc}

Rep boundaries: `start` = descent begins, `bottom` = minimum elbow angle, `end` = ascent completes.
`labels.truth` holds per-frame ground truth: elbow angle, signed hip deviation (+ sag), head angle, phase.
`labels.params` holds every sampled generation parameter (athlete, plan, camera).

## Intended use / limitations
Training & benchmarking side-view push-up form analysers and pose-based classifiers (physical-AI data).
Kinematic (not dynamic) model; symmetric left/right; no clothing/occlusion rendering, no pixels — keypoint
level only; noise model approximates MediaPipe behaviour but is not fitted to it.
"""


def export(samples: list[dict], out_dir: str, include_frames: bool = True) -> str:
    """Write a zip with dataset.jsonl, frames.csv, labels.csv, DATASHEET.md. Returns zip path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zp = out / f"formsim_synth_{len(samples)}.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr("dataset.jsonl", "\n".join(json.dumps(s, separators=(",", ":")) for s in samples) + "\n")
        # labels.csv
        cols = ["sample_id", "n", "start", "bottom", "end", "min_elbow", "max_elbow", "max_hip_sag",
                "max_hip_pike", "min_head", "descent_s", "ascent_s", "fatigue"] + FAULTS
        rows = [",".join(cols)]
        for s in samples:
            for r in s["labels"]["reps"]:
                rows.append(",".join([str(s["id"])] + [str(r[c]) for c in cols[1:13]] +
                                     ["1" if f in r["faults"] else "0" for f in FAULTS]))
        z.writestr("labels.csv", "\n".join(rows) + "\n")
        if include_frames:
            hdr = ["sample_id", "frame", "t", "present"] + [f"{n}_{c}" for n in NAMES for c in ("x", "y", "vis")]
            buf = io.StringIO()
            buf.write(",".join(hdr) + "\n")
            for s in samples:
                fr = s["landmarks"]["frames"]
                for f in fr:
                    if f["lm"] is None:
                        buf.write(f"{s['id']},{f['i']},{f['t']},0" + "," * (3 * len(NAMES)) + "\n")
                    else:
                        lm = f["lm"]
                        buf.write(f"{s['id']},{f['i']},{f['t']},1," +
                                  ",".join(f"{lm[n][0]:.1f},{lm[n][1]:.1f},{lm[n][2]:.2f}" for n in NAMES) + "\n")
            z.writestr("frames.csv", buf.getvalue())
        z.writestr("DATASHEET.md", _datasheet(samples))
    return str(zp)


# ----------------------------------------------------------------------------- stats & preview
def dataset_stats(samples: list[dict]) -> dict:
    """Aggregate distributions for charts."""
    reps = [r for s in samples for r in s["labels"]["reps"]]
    fault_rep = {f: sum(f in r["faults"] for r in reps) for f in FAULTS}
    fault_clip = {f: sum(any(f in r["faults"] for r in s["labels"]["reps"]) for s in samples) for f in FAULTS}
    res, fps = {}, {}
    for s in samples:
        k = f"{s['landmarks']['w']}x{s['landmarks']['h']}"
        res[k] = res.get(k, 0) + 1
        fk = str(int(s["landmarks"]["fps"]))
        fps[fk] = fps.get(fk, 0) + 1
    return {
        "n_samples": len(samples), "total_reps": len(reps),
        "total_frames": int(sum(s["landmarks"]["n_frames"] for s in samples)),
        "total_duration_s": float(sum(s["landmarks"]["n_frames"] / s["landmarks"]["fps"] for s in samples)),
        "clean_rep_frac": float(np.mean([not r["faults"] for r in reps])) if reps else 0.0,
        "fault_rep_counts": fault_rep, "fault_clip_counts": fault_clip,
        "rep_counts": [s["labels"]["rep_count"] for s in samples],
        "heights_m": [s["labels"]["params"]["athlete"]["height_m"] for s in samples],
        "yaw_deg": [s["labels"]["params"]["camera"]["yaw_deg"] for s in samples],
        "elevation_deg": [s["labels"]["params"]["camera"]["elevation_deg"] for s in samples],
        "min_elbow": [r["min_elbow"] for r in reps], "max_elbow": [r["max_elbow"] for r in reps],
        "descent_s": [r["descent_s"] for r in reps], "ascent_s": [r["ascent_s"] for r in reps],
        "max_hip_sag": [r["max_hip_sag"] for r in reps], "max_hip_pike": [r["max_hip_pike"] for r in reps],
        "resolutions": res, "fps": fps,
        "fatigue_clips": sum(s["labels"]["fatigue_from_rep"] is not None for s in samples),
    }


def _frame_at(sample: dict, t: float | None) -> int:
    lm = sample["landmarks"]
    if t is None:
        reps = sample["labels"]["reps"]
        t = reps[0]["bottom"] if reps else 0.0
    i = int(round(t * lm["fps"]))
    i = max(0, min(lm["n_frames"] - 1, i))
    fr = lm["frames"]
    for d in range(0, lm["n_frames"]):          # nearest frame with a person
        for j in (i - d, i + d):
            if 0 <= j < lm["n_frames"] and fr[j]["lm"] is not None:
                return j
    return i


def _skeleton_traces(lm: dict, near: str):
    import plotly.graph_objects as go
    traces = []
    for side, col, wd in ((("r" if near == "l" else "l"), "rgba(120,140,170,0.55)", 3), (near, "#FC4C02", 5)):
        xs, ys = [], []
        for a, b in SKELETON:
            if a[0] == side and a[1] == "_" or (a == "nose" and b[0] == side):
                xs += [lm[a][0], lm[b][0], None]; ys += [lm[a][1], lm[b][1], None]
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=wd), hoverinfo="skip",
                                 name=("near side" if side == near else "far side")))
    traces.append(go.Scatter(x=[lm[n][0] for n in NAMES], y=[lm[n][1] for n in NAMES], mode="markers",
                             marker=dict(size=7, color=[lm[n][2] for n in NAMES], colorscale="Viridis",
                                         cmin=0, cmax=1, line=dict(width=0)),
                             text=[f"{n} vis={lm[n][2]:.2f}" for n in NAMES], hoverinfo="text", name="keypoints"))
    return traces


def _title(sample, i):
    lab = sample["labels"]
    t = sample["landmarks"]["frames"][i]["t"]
    rep = next((r for r in lab["reps"] if r["start"] <= t <= r["end"]), None)
    el = lab["truth"]["elbow"][i]
    s = f"Sim athlete #{sample['id']} · t={t:.2f}s · elbow {el:.0f}° · {lab['truth']['phase'][i]}"
    if rep:
        s += f" · rep {rep['n']}: {', '.join(rep['faults']) or 'clean'}"
    return s


def _floor_trace(sample):
    import plotly.graph_objects as go
    fr = [f for f in sample["landmarks"]["frames"] if f["lm"]]
    pts = [(f["lm"]["l_foot"], f["lm"]["l_wrist"]) for f in fr[:1]]
    if not pts:
        return []
    (a, b) = pts[0]
    return [go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", line=dict(color="#8d6e63", width=2, dash="dot"),
                       hoverinfo="skip", name="floor")]


def render_preview(sample: dict, t: float | None = None):
    """Plotly figure of the projected 2D skeleton at time t (default: bottom of rep 1)."""
    import plotly.graph_objects as go
    lmd = sample["landmarks"]
    i = _frame_at(sample, t)
    near = "l" if sample["labels"]["near_side"] == "left" else "r"
    fig = go.Figure(_floor_trace(sample) + _skeleton_traces(lmd["frames"][i]["lm"], near))
    _layout(fig, lmd, _title(sample, i))
    return fig


def _layout(fig, lmd, title):
    fig.update_layout(title=dict(text=title, font=dict(size=13)), template="plotly_white", height=420,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                      shapes=[dict(type="rect", x0=0, y0=0, x1=lmd["w"], y1=lmd["h"],
                                   line=dict(color="#999", width=1))])
    fig.update_xaxes(range=[-0.02 * lmd["w"], 1.02 * lmd["w"]], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[1.02 * lmd["h"], -0.02 * lmd["h"]], scaleanchor="x", scaleratio=1,
                     showgrid=False, zeroline=False, visible=False)


def render_animation(sample: dict, max_frames: int = 150):
    """Animated plotly figure (play button + frame slider) of the simulated clip."""
    import plotly.graph_objects as go
    lmd = sample["landmarks"]
    near = "l" if sample["labels"]["near_side"] == "left" else "r"
    T = lmd["n_frames"]
    step = max(1, int(np.ceil(T / max_frames)))
    idx = [i for i in range(0, T, step) if lmd["frames"][i]["lm"] is not None]
    floor = _floor_trace(sample)
    frames = [go.Frame(data=floor + _skeleton_traces(lmd["frames"][i]["lm"], near), name=str(i),
                       layout=go.Layout(title_text=_title(sample, i))) for i in idx]
    fig = go.Figure(data=frames[0].data, frames=frames)
    _layout(fig, lmd, _title(sample, idx[0]))
    dur = int(1000 * step / lmd["fps"])
    fig.update_layout(
        height=470,
        updatemenus=[dict(type="buttons", x=0.0, y=-0.02, xanchor="left", yanchor="top", direction="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="❚❚", method="animate",
                                        args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
        sliders=[dict(x=0.12, y=-0.02, len=0.88, currentvalue=dict(prefix="t = ", suffix=" s"),
                      steps=[dict(label=f"{lmd['frames'][i]['t']:.1f}", method="animate",
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for i in idx])])
    return fig


if __name__ == "__main__":
    import time
    t0 = time.time()
    S = generate(50, seed=1)
    st = dataset_stats(S)
    real = json.loads((Path(__file__).resolve().parent / "samples" / "landmarks_demo_side.json").read_text())
    same = set(real) == set(S[0]["landmarks"]) and set(real["frames"][0]["lm"]) == set(NAMES)
    print(f"{st['n_samples']} samples, {st['total_reps']} reps, {st['total_frames']} frames "
          f"in {time.time()-t0:.2f}s; schema matches real: {same}")
    print("fault reps:", st["fault_rep_counts"], f"clean {st['clean_rep_frac']:.0%}")
