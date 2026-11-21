"""Stage 3 — Physics: landmarks + analysis -> joint loads (see CONTRACT.md §3).

A 2D sagittal-plane rigid-body model of the push-up:

* The body is split into de Leva (1996) segments (head+neck, trunk, upper arm,
  forearm, hand, thigh, shank, foot) with published mass fractions and
  centre-of-mass (COM) locations along each segment. Limbs on the camera side
  are mirrored onto the far side (bilateral symmetry), so arms/legs count twice.
* Pixels are converted to metres with a per-clip scale derived from the
  athlete's height: in a side view the ankle->knee->hip->shoulder chain equals
  (0.818 - 0.039) = 0.779 x standing height (Drillis & Contini / Winter
  anthropometric ratios: shoulder height 0.818 H, ankle height 0.039 H).
  We take a high percentile of that chain over well-tracked frames
  (foreshortening only ever shortens it) and use px_per_m = L_px / (0.779 H).
* Two supports: hands (wrist landmark) and feet (toe landmark). Moment balance
  about the toes, including vertical COM acceleration, gives the hand reaction
      F_hand = m (g + a_y) (x_com - x_toe) / (x_wrist - x_toe)
  (signed horizontal distances, so it works for athletes facing either way).
  a_y comes from Savitzky-Golay double differentiation of the COM height,
  done separately inside each continuous tracking run (camera cuts split runs)
  and clamped to +/- 0.6 g.
* Elbow and shoulder net joint moments: inverse statics up the arm chain from
  the (vertical) hand reaction at the wrist, including forearm/hand and upper
  arm weights. Reported per arm, as magnitudes.
* Lumbar moment: net moment about the hip of the upper-body free body
  (head + trunk + both arms, gravity minus hand support).
* Stress index 0-100: weighted sum of each moment as a fraction of a typical
  maximal voluntary capacity scaled by body mass (elbow 0.9 Nm/kg, shoulder
  1.1 Nm/kg, lumbar 2.6 Nm/kg); the lumbar term is amplified by posture
  deviation (|hip_dev|) because a sagging/piking spine carries the same moment
  on passive tissue instead of the core muscles.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

G = 9.81

# de Leva (1996) adjusted Zatsiorsky-Seluyanov parameters (male).
# mass fraction of body mass, COM location as fraction from proximal joint.
SEGMENTS = {
    "head":      {"mass": 0.0694, "com": 0.20},   # ear -> shoulder (head COM sits just below/behind the ear)
    "trunk":     {"mass": 0.4346, "com": 0.5138},  # shoulder -> hip (CERV-MIDH ratio applied to joint centres)
    "upper_arm": {"mass": 0.0271, "com": 0.5772},  # shoulder -> elbow
    "forearm":   {"mass": 0.0162, "com": 0.4574},  # elbow -> wrist
    "hand":      {"mass": 0.0061, "com": 0.0},     # at wrist (rests on the floor / handle)
    "thigh":     {"mass": 0.1416, "com": 0.4095},  # hip -> knee
    "shank":     {"mass": 0.0433, "com": 0.4459},  # knee -> ankle
    "foot":      {"mass": 0.0137, "com": 0.50},    # heel/ankle -> toe midpoint
}
CHAIN_FRAC = 0.818 - 0.039  # ankle->shoulder chain as fraction of standing height

# Reference capacities (Nm per kg body mass) used to normalise the stress index.
CAP_ELBOW = 0.9
CAP_SHOULDER = 1.1
CAP_LUMBAR = 2.6
W_ELBOW, W_SHOULDER, W_LUMBAR = 0.35, 0.40, 0.25
STRESS_GAIN = 100.0
MUSCLE_EFFICIENCY = 0.22  # mechanical -> metabolic

KEYS = ("ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "foot")

MODEL_NOTES = (
    "FormSim models your body as a chain of rigid segments seen from the side "
    "(head, trunk, upper arms, forearms, hands, thighs, shanks, feet) using the "
    "de Leva (1996) mass and centre-of-mass tables, scaled to your body mass. "
    "Video pixels are converted to metres from your height: the ankle-knee-hip-"
    "shoulder chain is about 78% of standing height. Your body is treated as a "
    "plank resting on two supports, hands and toes. Balancing moments about the "
    "toes - including how fast your centre of mass is accelerating up or down - "
    "tells us how much of your weight the hands carry (normally ~65% at the top, "
    "more at the bottom). That hand force is then walked up the arm to get the "
    "torque the triceps (elbow) and chest/front-shoulder (shoulder) must produce, "
    "and the upper body is balanced about the hip to get the moment your core "
    "must resist. A 0-100 stress index compares these torques with typical "
    "maximal strength for your body mass, with the core term amplified when your "
    "hips sag or pike. Work is the energy spent lifting (and accelerating) your "
    "centre of mass. Assumptions: 2D side view, left/right symmetric, hand force "
    "vertical (no friction), segment inertia ignored, so joint torques are "
    "estimates (+/-20-30%) - great for comparing reps and athletes, not a lab "
    "force plate."
)


# ----------------------------------------------------------------------------- helpers

def _pick_side(landmarks: dict, analysis: dict | None) -> str:
    side = (analysis or {}).get("side")
    if side in ("left", "right"):
        return side[0]
    tot = {"l": 0.0, "r": 0.0}
    for f in landmarks.get("frames", []):
        if f.get("lm"):
            for s in "lr":
                tot[s] += sum(f["lm"].get(f"{s}_{k}", [0, 0, 0])[2] for k in ("shoulder", "elbow", "wrist", "hip", "ankle"))
    return "l" if tot["l"] >= tot["r"] else "r"


def _get(lm: dict, s: str, key: str, other: str) -> tuple[float, float, float] | None:
    """Landmark on chosen side; fall back to other side if this one is poorly seen."""
    a = lm.get(f"{s}_{key}")
    b = lm.get(f"{other}_{key}")
    if a is None:
        return tuple(b) if b else None
    if b is not None and a[2] < 0.3 and b[2] > a[2] + 0.2:
        return tuple(b)
    return tuple(a)


def _frame_points(lm: dict | None, s: str) -> dict[str, np.ndarray] | None:
    if not lm:
        return None
    o = "r" if s == "l" else "l"
    pts = {}
    for k in KEYS:
        p = _get(lm, s, k, o)
        if p is None:
            return None
        pts[k] = np.array(p[:2], float)
        pts[k + "_v"] = p[2]
    if lm.get(f"{s}_ear") is None and lm.get("nose"):
        pts["ear"] = np.array(lm["nose"][:2], float)
    return pts


def _segment_coms(p: dict[str, np.ndarray]) -> list[tuple[str, float, np.ndarray]]:
    """(name, mass_fraction, com_px) for every body segment, both sides counted."""
    def along(a, b, f):
        return p[a] + f * (p[b] - p[a])
    toe_mid = 0.5 * (p["heel"] + p["foot"])
    out = [
        ("head", SEGMENTS["head"]["mass"], along("ear", "shoulder", SEGMENTS["head"]["com"])),
        ("trunk", SEGMENTS["trunk"]["mass"], along("shoulder", "hip", SEGMENTS["trunk"]["com"])),
    ]
    for _ in range(2):  # bilateral symmetry
        out += [
            ("upper_arm", SEGMENTS["upper_arm"]["mass"], along("shoulder", "elbow", SEGMENTS["upper_arm"]["com"])),
            ("forearm", SEGMENTS["forearm"]["mass"], along("elbow", "wrist", SEGMENTS["forearm"]["com"])),
            ("hand", SEGMENTS["hand"]["mass"], p["wrist"].copy()),
            ("thigh", SEGMENTS["thigh"]["mass"], along("hip", "knee", SEGMENTS["thigh"]["com"])),
            ("shank", SEGMENTS["shank"]["mass"], along("knee", "ankle", SEGMENTS["shank"]["com"])),
            ("foot", SEGMENTS["foot"]["mass"], toe_mid),
        ]
    return out


def _runs(valid: np.ndarray, jump: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous index runs of valid frames, split at camera-cut jumps."""
    runs, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif v and jump[i]:
            runs.append((start, i))
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def _sg_derivs(y: np.ndarray, dt: float, win_s: float = 0.30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Savitzky-Golay smoothed value, 1st and 2nd derivative (falls back gracefully on short runs)."""
    n = len(y)
    win = int(round(win_s / dt)) | 1
    win = max(5, win)
    if n < 5:
        return y.copy(), np.zeros(n), np.zeros(n)
    if win > n:
        win = n if n % 2 else n - 1
    poly = min(3, win - 2)
    try:
        from scipy.signal import savgol_filter
        ys = savgol_filter(y, win, poly, mode="interp")
        v = savgol_filter(y, win, poly, deriv=1, delta=dt, mode="interp")
        a = savgol_filter(y, win, poly, deriv=2, delta=dt, mode="interp")
    except Exception:  # pragma: no cover - numpy fallback
        k = np.ones(win) / win
        ys = np.convolve(np.pad(y, win // 2, mode="edge"), k, mode="valid")
        v = np.gradient(ys, dt)
        a = np.gradient(v, dt)
    return ys, v, a


def _nanpeak(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else 0.0


def _r(x: float, nd: int = 1) -> float:
    return float(round(float(x), nd))


def _to_list(a: np.ndarray, valid: np.ndarray, nd: int = 3) -> list:
    return [None if not v or not np.isfinite(x) else round(float(x), nd) for x, v in zip(a, valid)]


# ----------------------------------------------------------------------------- main

def simulate(landmarks: dict, analysis: dict | None = None,
             body_mass_kg: float = 75.0, height_m: float = 1.75) -> dict[str, Any]:
    """Run the rigid-body push-up model; returns the physics dict of CONTRACT §3."""
    m = float(body_mass_kg)
    H = float(height_m)
    frames = landmarks.get("frames", [])
    n = len(frames)
    fps = float(landmarks.get("fps") or 30.0)
    dt = 1.0 / fps
    s = _pick_side(landmarks, analysis)
    t = np.array([f.get("t", f.get("i", i) * dt) for i, f in enumerate(frames)], float)

    pts = [_frame_points(f.get("lm"), s) for f in frames]
    valid = np.array([p is not None for p in pts], dtype=bool)

    # ---- scale px -> m (robust over frames)
    chain = np.full(n, np.nan)
    for i, p in enumerate(pts):
        if p is None:
            continue
        vis = min(p["ankle_v"], p["knee_v"], p["hip_v"], p["shoulder_v"])
        if vis < 0.5:
            continue
        chain[i] = (np.linalg.norm(p["knee"] - p["ankle"]) + np.linalg.norm(p["hip"] - p["knee"])
                    + np.linalg.norm(p["shoulder"] - p["hip"]))
    good = chain[np.isfinite(chain)]
    if good.size == 0:
        good = np.array([np.nanmax([np.linalg.norm(p["shoulder"] - p["ankle"]) for p in pts if p is not None] or [200.0])])
    px_per_m_global = float(np.percentile(good, 75)) / (CHAIN_FRAC * H)

    # ---- per-frame geometry (metres, y UP, x as in image)
    com_px = np.full((n, 2), np.nan)
    for i, p in enumerate(pts):
        if p is None:
            continue
        segs = _segment_coms(p)
        tot = sum(mf for _, mf, _ in segs)
        com_px[i] = sum(mf * c for _, mf, c in segs) / tot

    # camera cuts / person switches: a key joint (or the COM) jumps more than 7% of the
    # body chain in one frame (real push-up motion is < ~3%), or the gap is > 0.3 s.
    chain_ref = float(np.median(good))
    jump = np.zeros(n, bool)
    last = None
    for i in range(n):
        if not valid[i]:
            continue
        if last is not None:
            if i - last > max(3, int(0.3 * fps)):
                jump[i] = True
            else:
                d = max(np.linalg.norm(pts[i][k] - pts[last][k]) for k in ("shoulder", "hip", "ankle", "wrist"))
                d = max(d, float(np.linalg.norm(com_px[i] - com_px[last])))
                if d > 0.07 * chain_ref * (i - last):
                    jump[i] = True
        last = i
    runs = _runs(valid, jump)

    # per-run scale (different shots in an edited clip can have different zoom)
    ppm = np.full(n, px_per_m_global)
    for a, b in runs:
        c = chain[a:b]
        c = c[np.isfinite(c)]
        if c.size >= 15:
            ppm[a:b] = float(np.percentile(c, 75)) / (CHAIN_FRAC * H)

    # frames whose body chain is far from the run's typical size are not trustworthy
    for a, b in runs:
        c = chain[a:b]
        med = np.nanmedian(c) if np.isfinite(c).any() else np.nan
        if np.isfinite(med):
            bad = np.isfinite(c) & (np.abs(c - med) > 0.3 * med)
            for j in np.nonzero(bad)[0]:
                pts[a + j] = None
                valid[a + j] = False

    # ground line (image y of the lowest support) per run, for COM height
    ground = np.full(n, np.nan)
    for a, b in runs:
        ys = [max(pts[i]["foot"][1], pts[i]["wrist"][1], pts[i]["heel"][1]) for i in range(a, b) if pts[i] is not None] or [0.0]
        ground[a:b] = float(np.median(ys))

    com_x = com_px[:, 0] / ppm
    com_y = (ground - com_px[:, 1]) / ppm  # height above support level, y up

    # ---- COM kinematics per run
    com_y_s = np.full(n, np.nan)
    vy = np.zeros(n)
    ay = np.zeros(n)
    vx = np.zeros(n)
    AY_MAX = 0.35 * G                   # push-ups never exceed ~0.2 g at the COM; beyond = tracking glitch
    for a, b in runs:
        ys_, v_, a_ = _sg_derivs(com_y[a:b], dt)
        _, vx_, _ = _sg_derivs(com_x[a:b], dt)
        if b - a < 0.6 * fps:           # short fragments (dropouts) -> double differentiation is just noise
            v_, a_, vx_ = np.zeros_like(v_), np.zeros_like(a_), np.zeros_like(vx_)
        com_y_s[a:b], vy[a:b], ay[a:b], vx[a:b] = ys_, v_, a_, vx_
    ay_glitch = np.abs(ay) >= AY_MAX
    ay = np.clip(ay, -AY_MAX, AY_MAX)
    vy = np.clip(vy, -2.0, 2.0)
    vx = np.clip(vx, -2.0, 2.0)

    # ---- forces & moments
    hand_frac = np.full(n, np.nan)
    static_frac = np.full(n, np.nan)     # share of body weight on the hands ignoring COM acceleration
    hand_F = np.full(n, np.nan)
    foot_F = np.full(n, np.nan)
    elbow_M = np.full(n, np.nan)
    shoulder_M = np.full(n, np.nan)
    lumbar_M = np.full(n, np.nan)
    hip_dev = _hip_dev_series(analysis, pts)

    m_fa = (SEGMENTS["forearm"]["mass"] + SEGMENTS["hand"]["mass"]) * m
    m_ua = SEGMENTS["upper_arm"]["mass"] * m
    for i, p in enumerate(pts):
        if p is None:
            continue
        k = ppm[i]
        x = {key: p[key][0] / k for key in KEYS}
        lever = x["wrist"] - x["foot"]
        if abs(lever) < 0.15 * H:  # degenerate geometry (not a push-up pose)
            continue
        raw = (com_x[i] - x["foot"]) / lever
        if not (0.3 <= raw <= 1.0) or ay_glitch[i]:
            continue                     # COM not between feet and hands (not a plank) or a tracking glitch
        frac = float(raw)
        static_frac[i] = frac
        F = m * (G + ay[i]) * frac
        F = max(F, 0.0)
        hand_frac[i] = F / (m * G)
        hand_F[i] = F
        foot_F[i] = m * (G + ay[i]) - F
        Fa = 0.5 * F  # per arm
        segs = _segment_coms(p)
        segx = {}
        for name, mf, c in segs:
            segx.setdefault(name, c[0] / k)
        x_fa = (SEGMENTS["forearm"]["mass"] * segx["forearm"] + SEGMENTS["hand"]["mass"] * x["wrist"]) / (
            SEGMENTS["forearm"]["mass"] + SEGMENTS["hand"]["mass"])
        # moment about joint (z = x * Fy, y up): hand force +Fa at wrist, weights -m g at COMs
        me = Fa * (x["wrist"] - x["elbow"]) - m_fa * G * (x_fa - x["elbow"])
        ms = (Fa * (x["wrist"] - x["shoulder"]) - m_fa * G * (x_fa - x["shoulder"])
              - m_ua * G * (segx["upper_arm"] - x["shoulder"]))
        elbow_M[i] = abs(me)
        shoulder_M[i] = abs(ms)
        # upper body about hip: head + trunk + 2 arms, hand support F at wrist
        ml = F * (x["wrist"] - x["hip"])
        for name, mf, c in segs:
            if name in ("head", "trunk", "upper_arm", "forearm", "hand"):
                ml -= mf * m * (G + ay[i]) * (c[0] / k - x["hip"])
        lumbar_M[i] = abs(ml)

    # ---- stress index
    post = 1.0 + np.clip(np.abs(np.nan_to_num(hip_dev, nan=0.0)) / 15.0, 0.0, 2.0)
    stress = STRESS_GAIN * (W_ELBOW * elbow_M / (CAP_ELBOW * m)
                            + W_SHOULDER * shoulder_M / (CAP_SHOULDER * m)
                            + W_LUMBAR * post * lumbar_M / (CAP_LUMBAR * m))
    stress = np.clip(stress, 0.0, 100.0)

    ok = valid & np.isfinite(hand_F)

    # ---- mechanical energy (potential + kinetic) for work
    E = m * G * com_y_s + 0.5 * m * (vx ** 2 + vy ** 2)
    dE = np.zeros(n)
    for a, b in runs:
        if b - a > 1:
            dE[a + 1:b] = np.diff(E[a:b])
    dE = np.nan_to_num(dE)
    pos_work = np.clip(dE, 0.0, None)
    # ignore sub-noise jitter (< 2 mm of COM rise per frame equivalent)
    pos_work[pos_work < m * G * 0.0005] = 0.0

    # ---- per rep
    reps_out = []
    for r in (analysis or {}).get("reps", []) or []:
        a = int(np.searchsorted(t, float(r.get("start", 0.0)), "left"))
        b = int(np.searchsorted(t, float(r.get("end", 0.0)), "right"))
        sl = slice(a, max(a + 1, b))
        sel = ok[sl]
        if not sel.any():
            continue
        work = float(pos_work[sl][ok[sl]].sum())
        asc = r.get("ascent_s")
        if not asc:
            asc = max(float(r.get("end", 0)) - float(r.get("bottom", r.get("start", 0))), 0.2)
        reps_out.append({
            "n": int(r.get("n", len(reps_out) + 1)),
            "peak_hand_force_N": _r(_nanpeak(hand_F[sl])),
            "peak_static_hand_frac": _r(_nanpeak(np.where(ok[sl], static_frac[sl], np.nan)), 3),
            "peak_elbow_torque_Nm": _r(_nanpeak(elbow_M[sl])),
            "peak_shoulder_torque_Nm": _r(_nanpeak(shoulder_M[sl])),
            "peak_lumbar_Nm": _r(_nanpeak(lumbar_M[sl])),
            "stress_peak": _r(_nanpeak(np.where(ok[sl], stress[sl], np.nan))),
            "work_J": _r(work),
            "power_W": _r(work / max(float(asc), 0.2)),
            "com_drop_m": _r(_nanpeak(com_y_s[sl]) - np.nanmin(np.where(ok[sl], com_y_s[sl], np.nan)), 3),
        })

    hf = hand_frac[ok]
    total_work = float(sum(r["work_J"] for r in reps_out)) if reps_out else float(pos_work[ok].sum())
    def _pk(key_rep: str, arr: np.ndarray) -> float:
        if reps_out:  # rep windows exclude getting-into-position transients
            return max(r[key_rep] for r in reps_out)
        return _nanpeak(arr[ok]) if ok.any() else 0.0

    peak_F = _pk("peak_hand_force_N", hand_F)
    # static equivalent load: body-weight share on the hands at the worst point of a rep (no dynamic spike)
    bench = m * max(r["peak_static_hand_frac"] for r in reps_out) if reps_out else None
    summary = {
        "avg_hand_load_frac": _r(float(np.median(hf)) if hf.size else 0.0, 3),
        "mean_hand_load_frac": _r(float(np.mean(hf)) if hf.size else 0.0, 3),
        "peak_hand_force_N": _r(peak_F),
        "peak_elbow_torque_Nm": _r(_pk("peak_elbow_torque_Nm", elbow_M)),
        "peak_shoulder_torque_Nm": _r(_pk("peak_shoulder_torque_Nm", shoulder_M)),
        "peak_lumbar_Nm": _r(_pk("peak_lumbar_Nm", lumbar_M)),
        "stress_peak": _r(_pk("stress_peak", np.where(ok, stress, np.nan))),
        "total_work_J": _r(total_work),
        "kcal_est": _r(total_work / MUSCLE_EFFICIENCY / 4184.0, 2),
        "equivalent_bench_kg": None if bench is None else _r(bench),
        "frames_modelled": int(ok.sum()),
    }
    if not reps_out:   # no valid push-up reps: whole-clip peaks would be transients / non-plank poses
        for k_ in ("peak_hand_force_N", "peak_elbow_torque_Nm", "peak_shoulder_torque_Nm", "peak_lumbar_Nm",
                   "stress_peak", "equivalent_bench_kg"):
            summary[k_] = None
        summary["note"] = "No valid push-up reps detected: peak loads not computed."

    return {
        "body_mass_kg": m, "height_m": H, "px_per_m": _r(px_per_m_global, 1),
        "side": "left" if s == "l" else "right",
        "series": {
            "t": [round(float(x), 4) for x in t],
            "com_x_m": _to_list(com_x, ok),
            "com_y_m": _to_list(com_y_s, ok),
            "hand_load_frac": _to_list(hand_frac, ok),
            "hand_force_N": _to_list(hand_F, ok, 1),
            "foot_force_N": _to_list(foot_F, ok, 1),
            "elbow_torque_Nm": _to_list(elbow_M, ok, 2),
            "shoulder_torque_Nm": _to_list(shoulder_M, ok, 2),
            "lumbar_moment_Nm": _to_list(lumbar_M, ok, 2),
            "stress": _to_list(stress, ok, 1),
            "com_x_px": _to_list(com_px[:, 0], ok, 1),
            "com_y_px": _to_list(com_px[:, 1], ok, 1),
        },
        "reps": reps_out,
        "summary": summary,
        "model_notes": MODEL_NOTES,
    }


def _hip_dev_series(analysis: dict | None, pts: list) -> np.ndarray:
    """Signed hip deviation (deg) from analysis, or computed from shoulder-hip-ankle geometry."""
    n = len(pts)
    src = ((analysis or {}).get("series") or {}).get("hip_dev")
    if src is not None and len(src) == n:
        return np.array([np.nan if v is None else float(v) for v in src])
    out = np.full(n, np.nan)
    for i, p in enumerate(pts):
        if p is None:
            continue
        a, h, b = p["shoulder"], p["hip"], p["ankle"]
        v1, v2 = a - h, b - h
        ang = math.degrees(math.atan2(abs(v1[0] * v2[1] - v1[1] * v2[0]), float(v1 @ v2)))
        out[i] = 180.0 - ang
    return out


# ----------------------------------------------------------------------------- figure helpers

def free_body(landmarks: dict, physics: dict, t: float) -> dict | None:
    """Free-body-diagram data (pixel coords) at time t for overlays/plots.

    Returns {"com": [x,y], "hand": [x,y], "foot": [x,y], "hand_force_N", "foot_force_N",
    "segments": [{"name", "com": [x,y], "mass_kg"}]} or None if no person at t.
    """
    frames = landmarks.get("frames", [])
    ts = physics["series"]["t"]
    if not frames or not ts:
        return None
    i = int(np.argmin(np.abs(np.array(ts) - t)))
    s = (physics.get("side") or "left")[0]
    p = _frame_points(frames[i].get("lm"), s)
    hf = physics["series"]["hand_force_N"][i]
    if p is None or hf is None:
        return None
    segs = _segment_coms(p)
    return {
        "i": i, "t": ts[i],
        "com": [physics["series"]["com_x_px"][i], physics["series"]["com_y_px"][i]],
        "hand": p["wrist"].tolist(), "foot": p["foot"].tolist(),
        "hand_force_N": hf, "foot_force_N": physics["series"]["foot_force_N"][i],
        "stress": physics["series"]["stress"][i],
        "segments": [{"name": nme, "com": c.tolist(), "mass_kg": round(mf * physics["body_mass_kg"], 2)}
                     for nme, mf, c in segs],
    }


def figure_data(physics: dict) -> dict[str, list]:
    """Long-form table {t, quantity, value} convenient for plotly express line charts."""
    rows = {"t": [], "quantity": [], "value": []}
    ser = physics["series"]
    for q in ("hand_force_N", "elbow_torque_Nm", "shoulder_torque_Nm", "lumbar_moment_Nm", "stress"):
        for t, v in zip(ser["t"], ser[q]):
            if v is not None:
                rows["t"].append(t)
                rows["quantity"].append(q)
                rows["value"].append(v)
    return rows


def figure(physics: dict, analysis: dict | None = None):
    """Plotly figure: hand force (+% body weight) and joint torques over time, rep windows shaded."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ser = physics["series"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Hand reaction force (both hands)", "Joint torques per side"))
    fig.add_trace(go.Scatter(x=ser["t"], y=ser["hand_force_N"], name="Hand force (N)", line=dict(width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=ser["t"], y=ser["elbow_torque_Nm"], name="Elbow (Nm)"), 2, 1)
    fig.add_trace(go.Scatter(x=ser["t"], y=ser["shoulder_torque_Nm"], name="Shoulder (Nm)"), 2, 1)
    fig.add_trace(go.Scatter(x=ser["t"], y=ser["lumbar_moment_Nm"], name="Lumbar (Nm)"), 2, 1)
    bw = physics["body_mass_kg"] * G
    fig.add_hline(y=bw, line_dash="dot", annotation_text="body weight", row=1, col=1)
    for r in (analysis or {}).get("reps", []) or []:
        fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor="gray", opacity=0.08, line_width=0)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40), legend=dict(orientation="h"))
    return fig


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "samples" / "landmarks_demo_side.json"
    lms = json.loads(path.read_text())
    try:
        import analysis as _an
        an = _an.analyze(lms)
    except Exception as e:  # analysis not ready -> no rep windows
        print(f"(analysis unavailable: {e})")
        an = None
    ph = simulate(lms, an)
    print(json.dumps(ph["summary"], indent=2))
    for r in ph["reps"]:
        print(r)
