"""FormSim biomechanics analysis: landmarks -> reps, faults, tempo, fatigue.

Implements CONTRACT section 2: ``analyze(landmarks, thresholds=None) -> analysis``.

Pipeline
--------
1. Pick the camera-facing side (left/right) for the whole clip by mean visibility.
2. Per-frame joint angles in pixel space: elbow, body line, signed hip deviation,
   head angle (+ signed head drop).
3. Segment the clip into continuous tracks: split on missing frames (> max_gap_s)
   and on camera cuts / person switches (sudden jump of the body centre).
4. Rep counting per segment: extra smoothing (median + moving average) of the elbow
   angle, then a hysteresis state machine with a *relative* excursion requirement
   plus absolute gates (works for slow reps with long top holds and fast reps,
   and still counts shallow / no-lockout reps so they can be flagged).
5. Per-rep metrics, faults (shared codes with synth.py), score, tempo.
6. Fatigue onset vs. the first reps, summary, warnings and confidence.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

FAULT_CODES = ["SHALLOW", "NO_LOCKOUT", "HIP_SAG", "HIP_PIKE", "HEAD_DROP", "FAST_DESCENT"]

DEFAULT_THRESHOLDS: dict[str, float] = {
    # --- rep counting (hysteresis on smoothed elbow angle) ---
    "count_down_deg": 150.0,      # elbow must drop below this to be a candidate bottom (loose: shallow reps still count)
    "count_up_deg": 120.0,        # ...and climb back above this to complete the rep (loose: no-lockout reps still count)
    "min_excursion_deg": 25.0,    # minimum drop from top / rise from bottom (relative hysteresis)
    "min_rep_s": 0.3,             # min duration start->end of a rep (trimmed span; fast reps are ~0.4-0.5 s)
    "max_rep_s": 15.0,            # longer "reps" are ignored (resting on the floor etc.)
    "smooth_s": 0.2,              # extra smoothing window for counting
    # --- form faults ---
    "depth_deg": 100.0,           # SHALLOW if min elbow never goes below this (shared with synth labels)
    "lockout_deg": 150.0,         # NO_LOCKOUT if max elbow never exceeds this (MediaPipe rarely reads >165)
    "hip_sag_deg": 12.0,          # HIP_SAG if the hip bends (180 - shoulder-hip-ankle) this far, hip below line
    "hip_pike_deg": 15.0,         # HIP_PIKE if the hip bends this far with the hip above the line
    "head_drop_deg": 56.0,        # HEAD_DROP if neck bends this far (180 - ear-shoulder-hip) with ear below body line
                                  # (good-form real clip peaks ~43-48 at the bottom: MediaPipe ear/shoulder offset; margin for camera height)
    "fast_descent_s": 0.4,        # FAST_DESCENT if lowering phase shorter than this
    # --- quality / robustness ---
    "min_visibility": 0.35,       # frames with lower mean key-joint visibility are ignored
    "max_gap_s": 0.5,             # longer gaps without a person split the clip
    "cut_jump": 0.15,             # body-centre jump (fraction of body length per ~5 frames) => camera cut
    "plank_max_incline_deg": 55.0,  # shoulder->ankle line steeper than this = not a push-up position
    "side_view_max_ratio": 0.5,   # shoulder width / torso length above this => not a side view
    "min_confidence": 0.5,
    # --- fatigue ---
    "fatigue_score_drop": 12.0,
    "fatigue_tempo_ratio": 1.4,
    "fatigue_depth_deg": 12.0,
    # --- leeway (baggy clothes / landmark noise) ---
    "leeway_scale": 1.0,          # multiplies LEEWAY below; 0 = strict (every threshold breach is a fault)
    "sustain_s": 0.2,             # hip / head faults must be held this long; single-frame landmark jumps ignored
}

# Leeway: a fault that only exceeds its threshold by less than this margin is reported as a *minor* note
# (rep["minor_issues"]) instead of an issue: it does not cost score and the rep still counts as clean.
# Loose / baggy clothing hides the real joint, so MediaPipe's elbow and hip landmarks sit a few degrees
# off (straight arms in wide sleeves typically read ~140-150 deg), and borderline readings are not
# trustworthy. Clearly bad form is still flagged (e.g. arms at 135 deg, hips sagging 15+ deg).
# Sized on the baggy-clothes upload (lockout reads 142-148 deg) while keeping the synthetic benchmark's macro
# fault F1 at its old level (~0.83); FAST_DESCENT needs none (the descent walk below bridges wobble stalls).
# HEAD_DROP gets no extra leeway: its threshold already carries a wide margin for the ear/shoulder offset
# (an injected 30 deg head drop on the lite model only reads ~57-62 deg), so leeway would hide real drops.
LEEWAY: dict[str, float] = {"SHALLOW": 5.0, "NO_LOCKOUT": 8.0, "HIP_SAG": 3.0, "HIP_PIKE": 3.0,
                            "HEAD_DROP": 0.0, "FAST_DESCENT": 0.0}
_FAULT_TH = {"SHALLOW": "depth_deg", "NO_LOCKOUT": "lockout_deg", "HIP_SAG": "hip_sag_deg",
             "HIP_PIKE": "hip_pike_deg", "HEAD_DROP": "head_drop_deg", "FAST_DESCENT": "fast_descent_s"}

FAULT_INFO: dict[str, dict[str, str]] = {
    "SHALLOW": {
        "title": "Go deeper",
        "cue": "Lower until your chest is a fist from the floor and elbows reach ~90 degrees.",
        "why": "Partial reps skip the stretched position where the chest and triceps do the most work, "
               "so strength gains in the full range are smaller.",
    },
    "NO_LOCKOUT": {
        "title": "Finish each rep at the top",
        "cue": "Press all the way up until your arms are straight and shoulder blades spread.",
        "why": "Stopping short of lockout cuts the triceps' final range and the serratus 'push-up plus', "
               "which protects shoulder health.",
    },
    "HIP_SAG": {
        "title": "Keep hips in line",
        "cue": "Squeeze your glutes and pull ribs down, like a plank that moves.",
        "why": "Sagging hips hyper-extend the lower back and shift load onto the lumbar spine instead of the core.",
    },
    "HIP_PIKE": {
        "title": "Lower your hips",
        "cue": "Push hips down until shoulders, hips and ankles form one straight line.",
        "why": "Piking shortens the lever, unloading the chest and making the rep easier than it looks.",
    },
    "HEAD_DROP": {
        "title": "Keep a neutral neck",
        "cue": "Look at the floor slightly ahead of your hands; lead with the chest, not the chin.",
        "why": "Dropping the head strains the neck and often hides a shallow rep (head reaches the floor, chest doesn't).",
    },
    "FAST_DESCENT": {
        "title": "Control the descent",
        "cue": "Count '1-2' on the way down.",
        "why": "Dropping fast uses momentum instead of muscle and loads the shoulder joint abruptly at the bottom.",
    },
}

# penalty weights for the per-rep score (multiplied by severity factor)
_PENALTY = {"SHALLOW": 22.0, "NO_LOCKOUT": 14.0, "HIP_SAG": 22.0, "HIP_PIKE": 15.0,
            "HEAD_DROP": 8.0, "FAST_DESCENT": 10.0}
_SEV_FACTOR = {"low": 0.5, "med": 0.8, "high": 1.0}

_KEY = ["shoulder", "elbow", "wrist", "hip", "ankle"]


# ----------------------------------------------------------------------------- geometry
def angle(a, b, c) -> float:
    """Angle ABC in degrees (at vertex b) for 2D/3D points. NaN if degenerate."""
    a, b, c = (np.asarray(p, dtype=float)[:2] for p in (a, b, c))
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(u @ v) / (nu * nv), -1.0, 1.0))))


def _signed_dev(origin, toward, point) -> float:
    """Angle (deg) between origin->toward and origin->point, + if point lies *below*
    the line in image coords (y down), - if above. Independent of facing direction."""
    o, t, p = (np.asarray(x, dtype=float)[:2] for x in (origin, toward, point))
    d, q = t - o, p - o
    nd, nq = np.linalg.norm(d), np.linalg.norm(q)
    if nd < 1e-9 or nq < 1e-9:
        return float("nan")
    d = d / nd
    n = np.array([-d[1], d[0]])          # a unit normal
    if n[1] < 0:                         # make it point down (image y+)
        n = -n
    along = float(q @ d)
    perp = float(q @ n)
    return float(np.degrees(math.atan2(perp, along)))


# ----------------------------------------------------------------------------- smoothing
def _interp_nans(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over NaN runs no longer than max_gap (edges untouched)."""
    x = x.copy()
    n = len(x)
    isn = np.isnan(x)
    if not isn.any() or isn.all():
        return x
    i = 0
    while i < n:
        if isn[i]:
            j = i
            while j < n and isn[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                x[i:j] = np.linspace(x[i - 1], x[j], j - i + 2)[1:-1]
            i = j
        else:
            i += 1
    return x


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """NaN-aware median(5) followed by centred moving average of size win."""
    n = len(x)
    if n == 0:
        return x.copy()
    out = x.copy()
    h = 2
    med = np.full(n, np.nan)
    for i in range(n):
        w = x[max(0, i - h): i + h + 1]
        w = w[~np.isnan(w)]
        if len(w) and not np.isnan(x[i]):
            med[i] = float(np.median(w))
    out = med
    if win > 1:
        k = win // 2
        res = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(out[i]):
                continue
            w = out[max(0, i - k): i + k + 1]
            w = w[~np.isnan(w)]
            res[i] = float(np.mean(w))
        out = res
    return out


# ----------------------------------------------------------------------------- helpers
def _pick_side(frames: list) -> tuple[str, float]:
    best, best_v = "left", -1.0
    for side, p in (("left", "l_"), ("right", "r_")):
        vals = [np.mean([f["lm"][p + j][2] for j in _KEY]) for f in frames if f.get("lm")]
        v = float(np.mean(vals)) if vals else 0.0
        if v > best_v:
            best, best_v = side, v
    return best, max(best_v, 0.0)


def _sev(excess: float, med: float, high: float) -> str:
    return "high" if excess >= high else "med" if excess >= med else "low"


def _r(x, nd=1):
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), nd)


def _sustained(x: np.ndarray, w: int) -> np.ndarray:
    """Level held for >= w frames: rolling min over a w-window (NaN-aware), aligned to window start.
    max(_sustained(x)) ignores spikes shorter than w frames (landmark jumps on loose clothing)."""
    n = len(x)
    if w <= 1 or n == 0:
        return x.copy()
    w = min(w, n)
    out = np.full(n, np.nan)
    for i in range(n - w + 1):
        seg = x[i:i + w]
        if np.isfinite(seg).sum() >= max(1, w // 2):
            out[i] = float(np.nanmin(seg))
    return out


def _window(mask: np.ndarray, peak: int) -> tuple[int, int]:
    """Contiguous True run in mask containing index peak."""
    a = b = peak
    while a - 1 >= 0 and mask[a - 1]:
        a -= 1
    while b + 1 < len(mask) and mask[b + 1]:
        b += 1
    return a, b


# ----------------------------------------------------------------------------- main
def analyze(landmarks: dict, thresholds: Optional[dict] = None) -> dict:
    """Analyse push-up landmarks (pose.extract output) into reps, faults, scores, fatigue.

    See CONTRACT.md section 2 for the exact output schema.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update({k: float(v) for k, v in thresholds.items() if v is not None})
    # a threshold the caller set explicitly is applied strictly (no leeway on top of it)
    custom = {k for k, v in (thresholds or {}).items() if v is not None}
    leeway = {c: (0.0 if _FAULT_TH[c] in custom else LEEWAY[c] * th["leeway_scale"]) for c in LEEWAY}
    frames = landmarks.get("frames") or []
    fps = float(landmarks.get("fps") or 30.0)
    n = len(frames)
    warnings: list[str] = []

    side, side_vis = _pick_side(frames)
    p = "l_" if side == "left" else "r_"
    t_arr = np.array([float(f.get("t", i / fps)) for i, f in enumerate(frames)]) if n else np.zeros(0)

    elbow = np.full(n, np.nan); body = np.full(n, np.nan); hip_dev = np.full(n, np.nan)
    head = np.full(n, np.nan); head_dev = np.full(n, np.nan); incline = np.full(n, np.nan)
    sw_ratio = np.full(n, np.nan); blen = np.full(n, np.nan)
    centre = np.full((n, 2), np.nan)
    present = np.zeros(n, bool)
    valid = np.zeros(n, bool)

    for i, f in enumerate(frames):
        lm = f.get("lm")
        if not lm:
            continue
        present[i] = True
        vis = float(np.mean([lm[p + j][2] for j in _KEY]))
        if vis < th["min_visibility"]:
            continue
        valid[i] = True
        sh, el, wr = lm[p + "shoulder"], lm[p + "elbow"], lm[p + "wrist"]
        hp, an = lm[p + "hip"], lm[p + "ankle"]
        ear = lm.get(p + "ear")
        elbow[i] = angle(sh, el, wr)
        body[i] = angle(sh, hp, an)
        # signed hip bend = 180 - interior shoulder-hip-ankle angle; + = sag (hip below the line), - = pike
        dev_side = _signed_dev(sh, an, hp)
        hip_dev[i] = math.copysign(180.0 - body[i], dev_side) if not math.isnan(body[i]) else float("nan")
        if ear is not None:
            head[i] = angle(ear, sh, hp)
            # signed neck bend: + = ear below the hip->shoulder line (head dropped), - = looking up
            sgn = 1.0 if _signed_dev(hp, sh, ear) >= 0 else -1.0
            head_dev[i] = sgn * (180.0 - head[i])
        dx, dy = an[0] - sh[0], an[1] - sh[1]
        incline[i] = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9))
        torso = math.hypot(hp[0] - sh[0], hp[1] - sh[1])
        sw = math.hypot(lm["l_shoulder"][0] - lm["r_shoulder"][0], lm["l_shoulder"][1] - lm["r_shoulder"][1])
        sw_ratio[i] = sw / torso if torso > 1e-6 else np.nan
        blen[i] = math.hypot(dx, dy)
        pts = np.array([lm[p + j][:2] for j in _KEY], float)
        centre[i] = pts.mean(axis=0)

    # ---------------------------------------------------------------- camera cuts / segments
    k = 5
    cuts = np.zeros(n, bool)
    for i in range(k, n):
        if valid[i] and valid[i - k]:
            scale = max(np.nanmax([blen[i], blen[i - k]]), 1e-6)
            jump = float(np.linalg.norm(centre[i] - centre[i - k])) / scale
            sc = blen[i] / max(blen[i - k], 1e-6)
            if jump > th["cut_jump"] or sc > 1.6 or sc < 1 / 1.6:
                cuts[i] = True
    max_gap = max(1, int(round(th["max_gap_s"] * fps)))
    seg_id = np.full(n, -1)
    sid, last_valid, cut_run = -1, -10 ** 9, False
    n_cuts = 0
    for i in range(n):
        if not valid[i]:
            continue
        new = (i - last_valid) > max_gap
        if cuts[i] and not cut_run:
            new = True
            n_cuts += 1
        cut_run = bool(cuts[i])
        if new:
            sid += 1
        seg_id[i] = sid
        last_valid = i
    if n_cuts:
        warnings.append(f"Detected {n_cuts} camera cut(s) / person switch(es); rep tracking was reset at each.")

    # ---------------------------------------------------------------- smoothing per segment
    win = max(1, int(round(th["smooth_s"] * fps)) | 1)
    e_s = np.full(n, np.nan); hip_s = np.full(n, np.nan); head_s = np.full(n, np.nan)
    e_f = np.full(n, np.nan)   # lightly filtered elbow (median only) for tempo timing: no temporal smearing
    segments = []
    for s in range(sid + 1):
        idx = np.where(seg_id == s)[0]
        if len(idx) < 3:
            continue
        a, b = int(idx[0]), int(idx[-1]) + 1
        segments.append((a, b))
        for src, dst in ((elbow, e_s), (hip_dev, hip_s), (head_dev, head_s)):
            x = src[a:b].copy()
            x[seg_id[a:b] != s] = np.nan
            x = _interp_nans(x, max_gap)
            dst[a:b] = _smooth(x, win if src is elbow else 3)
            if src is elbow:
                e_f[a:b] = _smooth(x, 1)

    # ---------------------------------------------------------------- rep detection
    exc, cd, cu = th["min_excursion_deg"], th["count_down_deg"], th["count_up_deg"]
    rep_spans = []  # (start, arrive, bottom, leave, end)
    partial_reps = 0
    for (a, b) in segments:
        e = e_s[a:b]
        bottoms = []
        state, run_max, run_min, min_i = "up", -1.0, 999.0, -1
        for j in range(len(e)):
            v = e[j]
            if np.isnan(v):
                continue
            if state == "up":
                run_max = max(run_max, v)
                if v < run_max - exc and v < cd:
                    state, run_min, min_i = "down", v, j
            else:
                if v < run_min:
                    run_min, min_i = v, j
                if v > run_min + exc and v > cu:
                    bottoms.append(min_i)
                    state, run_max = "up", v
        if state == "down":
            partial_reps += 1
        for q, bj in enumerate(bottoms):
            left = bottoms[q - 1] if q > 0 else 0
            right = bottoms[q + 1] if q + 1 < len(bottoms) else len(e) - 1
            # don't let the rep window run into standing up / getting into position
            inc = incline[a:b]
            pm = th["plank_max_incline_deg"]
            l2 = bj
            while l2 > left and (np.isnan(inc[l2 - 1]) or inc[l2 - 1] <= pm):
                l2 -= 1
            left = l2
            r2 = bj
            while r2 < right and (np.isnan(inc[r2 + 1]) or inc[r2 + 1] <= pm):
                r2 += 1
            right = r2
            # counting used the heavily smoothed series; tempo boundaries use the fine one
            ef = e_f[a:b]
            if np.isfinite(ef[left:right + 1]).sum() >= 3:
                win_b = max(1, win // 2)
                lo_b, hi_b = max(left, bj - win_b), min(right, bj + win_b) + 1
                if np.isfinite(ef[lo_b:hi_b]).any():
                    bj = lo_b + int(np.nanargmin(ef[lo_b:hi_b]))
                e = np.where(np.isfinite(ef), ef, e_s[a:b])
            lseg, rseg = e[left:bj + 1], e[bj:right + 1]
            lp = left + int(np.nanargmax(lseg))
            rp = bj + int(np.nanargmax(rseg))
            vmin = e[bj]
            tol_l = max(3.0, 0.08 * (e[lp] - vmin))
            tol_r = max(3.0, 0.08 * (e[rp] - vmin))
            st = lp
            for j in range(bj, lp - 1, -1):
                if not np.isnan(e[j]) and e[j] >= e[lp] - tol_l:
                    st = j
                    break
            en = rp
            for j in range(bj, rp + 1):
                if not np.isnan(e[j]) and e[j] >= e[rp] - tol_r:
                    en = j
                    break
            # the rep starts when the descent actually begins (not somewhere in a drifting top hold):
            # walk out from the bottom while the angle is still clearly changing or below half-amplitude
            # (a short stall mid-way, e.g. landmark wobble on a loose sleeve, does not end the walk: up to
            # ~0.2 s of "flat" frames are bridged if the movement continues after them)
            vthr = 12.5 / fps
            bridge = max(1, int(round(0.2 * fps)))

            def _moving(jn, jp, ref):
                return (not np.isnan(e[jn]) and not np.isnan(e[jp])
                        and (e[jn] - e[jp] > vthr or e[jn] < vmin + 0.5 * (ref - vmin)))

            j = bj
            while j - 1 > lp:
                if _moving(j - 1, j, e[lp]):
                    j -= 1
                    continue
                nxt_j = next((m for m in range(j - 2, max(lp, j - 1 - bridge), -1)
                              if not np.isnan(e[m]) and e[m] - e[j] > 2 * vthr * (j - m)
                              and e[m] < e[lp] - tol_l), None)
                if nxt_j is None:
                    break
                j = nxt_j
            st = max(st, j)
            j = bj
            while j + 1 < rp:
                if _moving(j + 1, j, e[rp]):
                    j += 1
                    continue
                nxt_j = next((m for m in range(j + 2, min(rp, j + 1 + bridge))
                              if not np.isnan(e[m]) and e[m] - e[j] > 2 * vthr * (m - j)
                              and e[m] < e[rp] - tol_r), None)
                if nxt_j is None:
                    break
                j = nxt_j
            en = min(en, j)
            btol = max(2.0, 0.04 * (e[lp] - vmin))
            arr = bj
            while arr - 1 > st and not np.isnan(e[arr - 1]) and e[arr - 1] <= vmin + btol:
                arr -= 1
            lea = bj
            while lea + 1 < en and not np.isnan(e[lea + 1]) and e[lea + 1] <= vmin + btol:
                lea += 1
            rep_spans.append((a + st, a + arr, a + bj, a + lea, a + en))

    if partial_reps:
        warnings.append(f"{partial_reps} partial rep(s) were cut off (camera cut, lost tracking or clip end) "
                        "and not counted.")

    # guards: duration and plank posture at the bottom
    reps_idx = []
    rejected_posture = 0
    rejected_dur = 0
    for (st, arr, bj, lea, en) in rep_spans:
        dur = t_arr[en] - t_arr[st]
        if dur < th["min_rep_s"] or dur > th["max_rep_s"]:
            rejected_dur += dur >= 0.15       # degenerate zero-length spans (clip edges) are not worth a warning
            continue
        inc = incline[bj]
        if not np.isnan(inc) and inc > th["plank_max_incline_deg"]:
            rejected_posture += 1
            continue
        reps_idx.append((st, arr, bj, lea, en))
    if rejected_dur:
        warnings.append(f"Ignored {rejected_dur} elbow bend(s) that were too quick (< {th['min_rep_s']:.1f}s) "
                        f"or too slow (> {th['max_rep_s']:.0f}s) to be a push-up rep.")
    if rejected_posture:
        warnings.append(f"Ignored {rejected_posture} elbow bend(s) while the body was upright (not in push-up position).")

    # ---------------------------------------------------------------- per-rep metrics & faults
    reps = []
    for r_i, (st, arr, bj, lea, en) in enumerate(reps_idx, start=1):
        pad = int(round(0.3 * fps))
        # lockout = how far the arms extend when finishing this rep (up to the next rep's start)
        nxt = reps_idx[r_i][0] if r_i < len(reps_idx) else n
        # (window reaches the next rep's start, max 1 s: a wobbly top - e.g. sleeves flapping - can push the
        # detected rep end before the arms are actually straight)
        a2, b2 = bj, max(bj + 1, min(n, en + int(round(1.0 * fps)) + 1, nxt + 1))
        sl = slice(st, en + 1)
        min_el = float(np.nanmin(elbow[sl])) if np.isfinite(elbow[sl]).any() else float(e_s[bj])
        # ...but never past a camera cut, and not frames after the rep where the person is getting up
        # (straight arms while standing/kneeling must not hide a missing lockout)
        top = elbow[a2:b2].copy()
        idx2 = np.arange(a2, b2)
        top[(seg_id[a2:b2] != seg_id[bj]) | ((idx2 > en) & (incline[a2:b2] > th["plank_max_incline_deg"]))] = np.nan
        max_el = float(np.nanmax(top)) if np.isfinite(top).any() else float(e_s[en])
        # hip / head faults use the level *held* for sustain_s (not a one-frame landmark jump)
        w_sus = max(1, int(round(th["sustain_s"] * fps)))
        hs = hip_s[sl]
        sag_sus, pike_sus = _sustained(hs, w_sus), _sustained(-hs, w_sus)
        max_sag = float(max(0.0, np.nanmax(sag_sus))) if np.isfinite(sag_sus).any() else 0.0
        max_pike = float(max(0.0, np.nanmax(pike_sus))) if np.isfinite(pike_sus).any() else 0.0
        min_head = float(np.nanmin(head[sl])) if np.isfinite(head[sl]).any() else float("nan")
        hd = head_s[sl]
        hd_sus = _sustained(hd, w_sus)
        max_head_drop = float(max(0.0, np.nanmax(hd_sus))) if np.isfinite(hd_sus).any() else 0.0
        descent = float(t_arr[arr] - t_arr[st])
        ascent = float(t_arr[en] - t_arr[lea])
        issues = []
        minor = []   # within LEEWAY of the threshold: shown as a note, rep stays clean, no score cost

        def add(code, ti, ta, tb, value, sev, msg, excess=None):
            item = {"code": code, "t": _r(t_arr[ti], 2), "t_start": _r(t_arr[ta], 2),
                    "t_end": _r(t_arr[tb], 2), "value": _r(value), "severity": sev, "msg": msg}
            if excess is not None and excess < leeway.get(code, 0.0):
                item["severity"] = "low"
                item["minor"] = True
                minor.append(item)
            else:
                issues.append(item)

        if min_el > th["depth_deg"]:
            ex = min_el - th["depth_deg"]
            add("SHALLOW", bj, arr, lea, min_el, _sev(ex, 10, 25),
                f"Only reached {min_el:.0f}° elbow bend (target < {th['depth_deg']:.0f}°)", ex)
        if max_el < th["lockout_deg"]:
            ex = th["lockout_deg"] - max_el
            add("NO_LOCKOUT", en, max(lea, en - pad), en, max_el, _sev(ex, 10, 20),
                f"Arms only extended to {max_el:.0f}° at the top (target > {th['lockout_deg']:.0f}°)", ex)
        if max_sag > th["hip_sag_deg"]:
            pk = st + int(np.nanargmax(hs))
            wa, wb = _window(np.nan_to_num(hip_s, nan=-99) > th["hip_sag_deg"], pk)
            add("HIP_SAG", pk, max(wa, st), min(wb, en), max_sag, _sev(max_sag - th["hip_sag_deg"], 5, 12),
                f"Hips sagged {max_sag:.0f}° below the shoulder-ankle line", max_sag - th["hip_sag_deg"])
        if max_pike > th["hip_pike_deg"]:
            pk = st + int(np.nanargmin(hs))
            wa, wb = _window(np.nan_to_num(hip_s, nan=99) < -th["hip_pike_deg"], pk)
            add("HIP_PIKE", pk, max(wa, st), min(wb, en), max_pike, _sev(max_pike - th["hip_pike_deg"], 5, 12),
                f"Hips piked {max_pike:.0f}° above the shoulder-ankle line", max_pike - th["hip_pike_deg"])
        if max_head_drop > th["head_drop_deg"]:
            pk = st + int(np.nanargmax(hd))
            wa, wb = _window(np.nan_to_num(head_s, nan=-99) > th["head_drop_deg"], pk)
            add("HEAD_DROP", pk, max(wa, st), min(wb, en), max_head_drop,
                _sev(max_head_drop - th["head_drop_deg"], 8, 18),
                f"Head dropped {max_head_drop:.0f}° below the body line", max_head_drop - th["head_drop_deg"])
        if descent < th["fast_descent_s"]:
            add("FAST_DESCENT", arr, st, arr, descent,
                _sev(th["fast_descent_s"] - descent, 0.1, 0.2),
                f"Descent took only {descent:.2f}s (aim for > {th['fast_descent_s']:.1f}s)",
                th["fast_descent_s"] - descent)

        score = 100.0 - sum(_PENALTY[i["code"]] * _SEV_FACTOR[i["severity"]] for i in issues)
        reps.append({
            "n": r_i, "start": _r(t_arr[st], 2), "bottom": _r(t_arr[bj], 2), "end": _r(t_arr[en], 2),
            "min_elbow": _r(min_el), "max_elbow": _r(max_el),
            "max_hip_sag": _r(max_sag), "max_hip_pike": _r(max_pike),
            "min_head": _r(min_head), "descent_s": _r(descent, 2), "ascent_s": _r(ascent, 2),
            "score": _r(max(0.0, min(100.0, score))), "issues": issues, "minor_issues": minor,
        })

    # ---------------------------------------------------------------- phase series
    phase: list[Optional[str]] = [("top" if valid[i] else None) for i in range(n)]
    for (st, arr, bj, lea, en) in reps_idx:
        for i in range(st, en + 1):
            if not valid[i]:
                continue
            phase[i] = "down" if i < arr else "bottom" if i <= lea else "up"

    # ---------------------------------------------------------------- fatigue
    fatigue_from = _fatigue(reps, th)

    # ---------------------------------------------------------------- quality checks
    det_frac = float(present.sum()) / n if n else 0.0
    confidence = float(side_vis * det_frac)
    duration = float(t_arr[-1] - t_arr[0]) if n > 1 else 0.0
    if n == 0 or not valid.any():
        warnings.append("No person detected - film from the side with your whole body in frame.")
    else:
        if side_vis < 0.6:
            warnings.append(f"Low joint visibility on the {side} side ({side_vis:.2f}) - keep the whole body in frame.")
        if det_frac < 0.7:
            warnings.append(f"Person only detected in {det_frac*100:.0f}% of frames.")
    side_ok = posture_ok = True
    if valid.any():
        plank = valid & (incline <= th["plank_max_incline_deg"])
        plank_frac = float(plank.sum()) / max(1, valid.sum())
        if plank_frac < 0.3:
            posture_ok = False
            warnings.append("Body is mostly upright - this doesn't look like a push-up "
                            f"(horizontal in {plank_frac*100:.0f}% of frames). If the video plays sideways, "
                            "rotate it so the floor is at the bottom and upload again.")
        ref = plank if plank.any() else valid
        swr = float(np.nanmedian(sw_ratio[ref])) if np.isfinite(sw_ratio[ref]).any() else 0.0
        if swr > th["side_view_max_ratio"]:
            side_ok = False
            warnings.append(f"Camera doesn't look side-on (shoulder width is {swr:.2f}x torso length) - "
                            "film from the side for accurate angles.")
        elif plank.any():
            # partial: a chunk of the clip is filmed from another angle
            bad = np.isfinite(sw_ratio) & plank & (sw_ratio > th["side_view_max_ratio"])
            if bad.sum() > 0.2 * plank.sum():
                warnings.append(f"{bad.sum()/plank.sum()*100:.0f}% of the push-up frames are not side-on.")
    if duration < 2.0:
        warnings.append(f"Clip is very short ({duration:.1f}s).")
    if not reps and valid.any():
        warnings.append("No complete push-up reps detected." + (
            " The clip ends or cuts away before you push back up - film at least one full rep (down and up)."
            if partial_reps else ""))
    confidence_ok = bool(confidence >= th["min_confidence"] and side_ok and posture_ok and valid.any())

    # ---------------------------------------------------------------- summary
    issue_counts: dict[str, int] = {}
    for r in reps:
        for code in {i["code"] for i in r["issues"]}:
            issue_counts[code] = issue_counts.get(code, 0) + 1
    summary = {
        "rep_count": len(reps),
        "clean_reps": sum(1 for r in reps if not r["issues"]),
        "avg_score": _r(np.mean([r["score"] for r in reps])) if reps else None,
        "fatigue_from_rep": fatigue_from,
        "issue_counts": issue_counts,
        "minor_issue_counts": {c: sum(1 for r in reps if any(i["code"] == c for i in r.get("minor_issues", [])))
                               for c in FAULT_CODES
                               if any(i["code"] == c for r in reps for i in r.get("minor_issues", []))},
        "avg_descent_s": _r(np.mean([r["descent_s"] for r in reps]), 2) if reps else None,
        "avg_ascent_s": _r(np.mean([r["ascent_s"] for r in reps]), 2) if reps else None,
        "total_time_s": _r(reps[-1]["end"] - reps[0]["start"], 2) if reps else 0.0,
    }

    def ser(x, nd=1):
        return [None if not valid[i] or np.isnan(x[i]) else round(float(x[i]), nd) for i in range(n)]

    return {
        "exercise": "pushup", "side": side,
        "confidence": round(confidence, 3), "confidence_ok": confidence_ok,
        "warnings": warnings, "thresholds": th,
        "series": {
            "t": [round(float(v), 4) for v in t_arr],
            "elbow": ser(elbow), "body_line": ser(body), "hip_dev": ser(hip_dev), "head": ser(head),
            "phase": phase,
        },
        "reps": reps,
        "summary": summary,
    }


def _fatigue(reps: list, th: dict) -> Optional[int]:
    """First rep from which form/tempo/depth degrade persistently vs. the first reps."""
    if len(reps) < 4:
        return None
    nb = 3 if len(reps) >= 6 else 2
    base = reps[:nb]
    b_score = float(np.mean([r["score"] for r in base]))
    b_dur = float(np.mean([r["end"] - r["start"] for r in base]))
    b_depth = float(np.mean([r["min_elbow"] for r in base]))
    bad = []
    for r in reps:
        dur = r["end"] - r["start"]
        worse = (r["score"] < b_score - th["fatigue_score_drop"]
                 or dur > b_dur * th["fatigue_tempo_ratio"]
                 or r["min_elbow"] > b_depth + th["fatigue_depth_deg"])
        bad.append(worse)
    for k in range(nb, len(reps)):
        rest = bad[k:]
        if not bad[k]:
            continue
        if len(rest) >= 2 and sum(rest) / len(rest) >= 0.67:
            return reps[k]["n"]
        if len(rest) == 1 and reps[k]["score"] < b_score - 2 * th["fatigue_score_drop"]:
            return reps[k]["n"]
    return None


def format_table(analysis: dict) -> str:
    """Plain-text per-rep table for CLI/debug."""
    lines = [f"{'n':>2} {'start':>6} {'bottom':>6} {'end':>6} {'minE':>5} {'maxE':>5} {'sag':>5} "
             f"{'pike':>5} {'head':>5} {'desc':>5} {'asc':>5} {'score':>5}  issues"]
    for r in analysis["reps"]:
        lines.append(f"{r['n']:>2} {r['start']:>6} {r['bottom']:>6} {r['end']:>6} {r['min_elbow']:>5} "
                     f"{r['max_elbow']:>5} {r['max_hip_sag']:>5} {r['max_hip_pike']:>5} {str(r['min_head']):>5} "
                     f"{r['descent_s']:>5} {r['ascent_s']:>5} {r['score']:>5}  "
                     + ",".join(f"{i['code']}({i['severity']})" for i in r["issues"]))
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "samples" / "landmarks_demo_side.json")
    res = analyze(json.loads(Path(path).read_text()))
    print(f"{path}: side={res['side']} confidence={res['confidence']} ok={res['confidence_ok']}")
    for w in res["warnings"]:
        print("  WARNING:", w)
    print(format_table(res))
    print("summary:", json.dumps(res["summary"]))
