"""Stage 5 — Overlay: video + landmarks + analysis (+ physics) -> annotated H.264 mp4.

Draws a coach-style overlay on every frame:
  * skeleton (analysed side bold, other side faint), joint dots, dashed
    shoulder->ankle "body line" reference,
  * segments turn orange/red while a form fault is active
    (HIP_SAG/HIP_PIKE -> trunk + legs, SHALLOW/NO_LOCKOUT/FAST_DESCENT -> arm,
    HEAD_DROP -> neck),
  * live elbow-angle arc gauge at the elbow,
  * HUD panel: rep counter, elbow angle, body-line angle, phase,
  * fault banner, vertical joint-stress bar (physics), rep-score flash,
  * bottom timeline: elbow trace for the whole clip, rep bands, fault ticks, playhead.

Public API (see CONTRACT.md section 5):
  render(video_path, landmarks, analysis, physics=None, out_path=None, progress=None) -> str
  snapshot(video_path, landmarks, analysis, t, physics=None) -> np.ndarray (RGB)
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- palette (BGR)
GREEN = (110, 225, 70)
ORANGE = (0, 165, 255)
RED = (60, 60, 240)
WHITE = (245, 245, 245)
GREY = (170, 170, 170)
DARK = (25, 22, 20)
CYAN = (230, 200, 60)
ACCENT = (255, 190, 80)          # light blue-ish accent for gauges

# Segment groups -> pairs of landmark suffixes (side prefix added later).
SEGMENTS = {
    "neck": [("ear", "shoulder")],
    "arm": [("shoulder", "elbow"), ("elbow", "wrist")],
    "trunk": [("shoulder", "hip")],
    "legs": [("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "foot"), ("ankle", "foot")],
}
FAULT_GROUPS = {
    "HIP_SAG": ("trunk", "legs"), "HIP_PIKE": ("trunk", "legs"),
    "SHALLOW": ("arm",), "NO_LOCKOUT": ("arm",), "FAST_DESCENT": ("arm",),
    "HEAD_DROP": ("neck",),
}
FAULT_LABEL = {
    "HIP_SAG": "HIP SAG", "HIP_PIKE": "HIP PIKE", "SHALLOW": "SHALLOW",
    "NO_LOCKOUT": "NO LOCKOUT", "FAST_DESCENT": "FAST DESCENT", "HEAD_DROP": "HEAD DROP",
}
SEV_RANK = {"low": 0, "med": 1, "high": 2}
MSG_LINGER_S = 0.6       # keep the fault text a bit after its window for readability
FLASH_S = 1.4            # rep-score flash duration
VIS_MIN = 0.3            # hide landmarks below this visibility


# ---------------------------------------------------------------- text (PIL sprites w/ cv2 fallback)
_FONT_CANDIDATES = {
    True: ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
           "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
           "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
           "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"],
    False: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"],
}


@lru_cache(maxsize=32)
def _font(size: int, bold: bool):
    try:
        from PIL import ImageFont
    except Exception:
        return None
    for p in _FONT_CANDIDATES[bold]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size)
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _text_sprite(text: str, size: int, color: tuple, bold: bool):
    """Return (bgr uint8, alpha float32 HxWx1) sprite of rendered text, or None."""
    font = _font(size, bold)
    if font is None:
        return None
    from PIL import Image, ImageDraw
    l, t, r, b = font.getbbox(text)
    w, h = max(1, r - l + 4), max(1, b - t + 4)
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=font, fill=255)
    a = (np.asarray(img, dtype=np.float32) / 255.0)[..., None]
    bgr = np.empty((h, w, 3), np.uint8)
    bgr[:] = color
    return bgr, a


def _text_size(text: str, size: int, bold: bool = True) -> tuple[int, int]:
    sp = _text_sprite(text, size, WHITE, bold)
    if sp is not None:
        return sp[0].shape[1], sp[0].shape[0]
    fs = size / 30.0
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, max(1, int(size / 15)))
    return w, h + 4


def _put(img: np.ndarray, text: str, org: tuple[int, int], size: int,
         color=WHITE, bold: bool = True, shadow: bool = True, alpha: float = 1.0) -> None:
    """Draw text with its top-left at org (crisp TrueType if available)."""
    x, y = int(org[0]), int(org[1])
    sp = _text_sprite(text, size, tuple(int(c) for c in color), bold)
    if sp is None:  # Hershey fallback (no degree glyph)
        text = text.replace("°", " deg")
        fs, th = size / 30.0, max(1, int(size / 15))
        (_, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        if shadow:
            cv2.putText(img, text, (x + 1, y + h + 1), cv2.FONT_HERSHEY_DUPLEX, fs, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y + h), cv2.FONT_HERSHEY_DUPLEX, fs, color, th, cv2.LINE_AA)
        return
    if shadow:
        _blit(img, _text_sprite(text, size, (0, 0, 0), bold), x + max(1, size // 18), y + max(1, size // 18), 0.6 * alpha)
    _blit(img, sp, x, y, alpha)


def _blit(img, sprite, x, y, alpha=1.0):
    bgr, a = sprite
    H, W = img.shape[:2]
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sb = bgr[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x] * alpha
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - sa) + sb * sa).astype(np.uint8)


# ---------------------------------------------------------------- drawing primitives
def _rounded_box(img, x0, y0, x1, y1, r, color=DARK, alpha=0.62, border=None, bw=1):
    """Semi-transparent filled rounded rectangle (blended only inside its ROI)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    r = int(max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    w, h = x1 - x0, y1 - y0
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
    if border is not None:
        # thin border along the rounded outline
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img[y0:y1, x0:x1], cnts, -1, border, bw, cv2.LINE_AA)


def _dashed(img, p, q, color, th, dash):
    p, q = np.array(p, float), np.array(q, float)
    L = float(np.linalg.norm(q - p))
    if L < 1:
        return
    n = max(1, int(L / (2 * dash)))
    for k in range(n + 1):
        a = p + (q - p) * min(1.0, (2 * k * dash) / L)
        b = p + (q - p) * min(1.0, (2 * k * dash + dash) / L)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, th, cv2.LINE_AA)


def _lerp_color(c1, c2, f):
    f = float(min(1.0, max(0.0, f)))
    return tuple(int(a + (b - a) * f) for a, b in zip(c1, c2))


def _stress_color(v):
    v = 0.0 if v is None else v / 100.0
    return _lerp_color(GREEN, ORANGE, v / 0.6) if v < 0.6 else _lerp_color(ORANGE, RED, (v - 0.6) / 0.4)


def _score_color(s):
    return GREEN if s >= 85 else ORANGE if s >= 65 else RED


# ---------------------------------------------------------------- data helpers
def _angle(a, b, c):
    """Angle ABC in degrees (2D)."""
    v1 = np.array(a[:2], float) - np.array(b[:2], float)
    v2 = np.array(c[:2], float) - np.array(b[:2], float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1))))


def _choose_side(landmarks, analysis) -> str:
    side = (analysis or {}).get("side")
    if side in ("left", "right"):
        return side[0]
    tot = {"l": 0.0, "r": 0.0}
    for f in landmarks.get("frames", []):
        lm = f.get("lm")
        if lm:
            for s in "lr":
                tot[s] += sum(lm.get(f"{s}_{j}", [0, 0, 0])[2] for j in ("shoulder", "elbow", "wrist", "hip", "ankle"))
    return "l" if tot["l"] >= tot["r"] else "r"


def _series(d, key, n):
    s = ((d or {}).get("series") or {}).get(key)
    if not isinstance(s, (list, tuple)):
        return [None] * n
    s = list(s)[:n]
    return s + [None] * (n - len(s))


def _issue_windows(analysis):
    """Flatten all rep issues into [(t0, t1, issue, rep_n)]."""
    out = []
    for rep in (analysis or {}).get("reps", []) or []:
        for iss in rep.get("issues", []) or []:
            t = iss.get("t", rep.get("bottom", rep.get("start", 0.0)))
            t0 = iss.get("t_start", t - 0.3 if t is not None else rep.get("start", 0.0))
            t1 = iss.get("t_end", t + 0.3 if t is not None else rep.get("end", 0.0))
            if t0 is None or t1 is None:
                continue
            out.append((float(t0), float(t1), iss, rep.get("n")))
    return out


class _Ctx:
    """Precomputed, per-clip drawing context (shared by render and snapshot)."""

    def __init__(self, landmarks, analysis, physics, vw, vh, fps):
        self.lm = landmarks
        self.an = analysis or {}
        self.ph = physics
        frames = landmarks.get("frames", []) or []
        self.n = len(frames)
        self.frames = frames
        self.fps = float(fps or landmarks.get("fps") or 30.0)
        lw, lh = float(landmarks.get("w") or vw), float(landmarks.get("h") or vh)
        self.sx, self.sy = vw / lw, vh / lh
        self.W, self.H = vw, vh
        self.side = _choose_side(landmarks, analysis)
        self.other = "r" if self.side == "l" else "l"
        self.u = max(0.6, min(vw, vh) / 480.0)          # resolution scale unit
        n = self.n
        self.t = [f.get("t", i / self.fps) for i, f in enumerate(frames)]
        self.elbow = _series(self.an, "elbow", n)
        self.body = _series(self.an, "body_line", n)
        self.phase = _series(self.an, "phase", n)
        self.stress = _series(physics, "stress", n) if physics else None
        # fill missing elbow/body angles from raw landmarks so the HUD always works
        for i, f in enumerate(frames):
            lm = f.get("lm")
            if not lm:
                continue
            s = self.side
            if self.elbow[i] is None and all(f"{s}_{j}" in lm for j in ("shoulder", "elbow", "wrist")):
                self.elbow[i] = _angle(lm[f"{s}_shoulder"], lm[f"{s}_elbow"], lm[f"{s}_wrist"])
            if self.body[i] is None and all(f"{s}_{j}" in lm for j in ("shoulder", "hip", "ankle")) and not analysis:
                self.body[i] = _angle(lm[f"{s}_shoulder"], lm[f"{s}_hip"], lm[f"{s}_ankle"])
        self.reps = sorted((self.an.get("reps") or []), key=lambda r: r.get("end", 0))
        self.issues = _issue_windows(self.an)
        self.duration = self.t[-1] if self.t else 0.0
        self._timeline = None

    # -- per-frame state
    def active_issues(self, t, linger=0.0):
        return [(iss, n) for t0, t1, iss, n in self.issues if t0 <= t <= t1 + linger]

    def reps_done(self, t):
        return sum(1 for r in self.reps if r.get("end", 1e9) <= t)

    def flash_rep(self, t):
        for r in self.reps:
            e = r.get("end")
            if e is not None and e <= t < e + FLASH_S:
                return r, (t - e) / FLASH_S
        return None, 0.0

    def pt(self, lm, name):
        p = lm.get(name)
        if p is None or (len(p) > 2 and p[2] is not None and p[2] < VIS_MIN):
            return None
        return (int(round(p[0] * self.sx)), int(round(p[1] * self.sy)))


# ---------------------------------------------------------------- frame composition
def _draw_skeleton(img, ctx: _Ctx, i: int, t: float):
    f = ctx.frames[i]
    lm = f.get("lm")
    if not lm:
        return
    u = ctx.u
    active = ctx.active_issues(t)
    bad = {}  # group -> colour
    for iss, _ in active:
        col = RED if iss.get("severity") == "high" else ORANGE
        for g in FAULT_GROUPS.get(iss.get("code"), ()):
            if bad.get(g) != RED:
                bad[g] = col

    # faint far side
    o = ctx.other
    for g, pairs in SEGMENTS.items():
        for a, b in pairs:
            p, q = ctx.pt(lm, f"{o}_{a}"), ctx.pt(lm, f"{o}_{b}")
            if p and q:
                cv2.line(img, p, q, (150, 150, 150), max(1, int(2 * u)), cv2.LINE_AA)

    s = ctx.side
    # dashed reference body line shoulder -> ankle
    sh, an_ = ctx.pt(lm, f"{s}_shoulder"), ctx.pt(lm, f"{s}_ankle")
    if sh and an_:
        _dashed(img, sh, an_, (235, 235, 235), max(1, int(1.2 * u)), max(4, int(7 * u)))

    th = max(2, int(round(5 * u)))
    for g, pairs in SEGMENTS.items():
        col = bad.get(g, GREEN)
        for a, b in pairs:
            p, q = ctx.pt(lm, f"{s}_{a}"), ctx.pt(lm, f"{s}_{b}")
            if p and q:
                cv2.line(img, p, q, (20, 20, 20), th + max(2, int(2 * u)), cv2.LINE_AA)  # outline
                cv2.line(img, p, q, col, th, cv2.LINE_AA)
    # joints
    jr = max(3, int(round(5 * u)))
    for j in ("ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "foot"):
        p = ctx.pt(lm, f"{s}_{j}")
        if p:
            cv2.circle(img, p, jr + max(1, int(u)), (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, p, jr, WHITE, -1, cv2.LINE_AA)
    for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        p = ctx.pt(lm, f"{o}_{j}")
        if p:
            cv2.circle(img, p, max(2, jr // 2), (200, 200, 200), -1, cv2.LINE_AA)

    # elbow arc gauge (the real joint angle, coloured by depth)
    e, sp_, wr = ctx.pt(lm, f"{s}_elbow"), ctx.pt(lm, f"{s}_shoulder"), ctx.pt(lm, f"{s}_wrist")
    ang = ctx.elbow[i]
    if e and sp_ and wr and ang is not None:
        a1 = math.degrees(math.atan2(sp_[1] - e[1], sp_[0] - e[0]))
        a2 = math.degrees(math.atan2(wr[1] - e[1], wr[0] - e[0]))
        d = (a2 - a1 + 540) % 360 - 180
        r = int(28 * u)
        depth_thr = float(((ctx.an.get("thresholds") or {}).get("depth_deg")) or 95)
        col = GREEN if ang <= depth_thr else ACCENT if ang < 150 else (220, 220, 220)
        ov = img.copy()
        cv2.ellipse(ov, e, (r, r), 0, a1, a1 + d, col, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.28, img, 0.72, 0, dst=img)
        cv2.ellipse(img, e, (r, r), 0, a1, a1 + d, col, max(2, int(2.5 * u)), cv2.LINE_AA)
        # label on the outside of the joint
        mid = math.radians(a1 + d / 2)
        lx, ly = e[0] - math.cos(mid) * (r + 8 * u), e[1] - math.sin(mid) * (r + 8 * u)
        txt = f"{ang:.0f}°"
        tw, tht = _text_size(txt, int(16 * u))
        lx, ly = int(lx - tw / 2), int(ly - tht / 2)
        _rounded_box(img, lx - 4 * u, ly - 2 * u, lx + tw + 4 * u, ly + tht + 2 * u, int(6 * u), DARK, 0.65)
        _put(img, txt, (lx, ly), int(16 * u), col, shadow=False)


def _draw_hud(img, ctx: _Ctx, i: int, t: float):
    u = ctx.u
    W, H = ctx.W, ctx.H
    pad = int(10 * u)
    # -------- main panel (top-left)
    pw, ph = int(150 * u), int(132 * u)
    x0, y0 = pad, pad
    _rounded_box(img, x0, y0, x0 + pw, y0 + ph, int(12 * u), DARK, 0.62, border=(90, 90, 90), bw=1)
    _put(img, "REPS", (x0 + int(12 * u), y0 + int(8 * u)), int(12 * u), GREY)
    done = ctx.reps_done(t)
    _put(img, str(done), (x0 + int(10 * u), y0 + int(20 * u)), int(46 * u), WHITE)
    ph_ = ctx.phase[i]
    if ph_:
        arrow = {"down": "DOWN", "up": "UP", "bottom": "BOTTOM", "top": "TOP"}.get(ph_, str(ph_).upper())
        tw, _ = _text_size(arrow, int(11 * u))
        bx = x0 + pw - tw - int(18 * u)
        _rounded_box(img, bx - 6 * u, y0 + 10 * u, bx + tw + 6 * u, y0 + 28 * u, int(8 * u), (70, 60, 50), 0.9)
        _put(img, arrow, (bx, y0 + int(12 * u)), int(11 * u), ACCENT, shadow=False)
    ang = ctx.elbow[i]
    body = ctx.body[i]
    yy = y0 + int(80 * u)
    _put(img, "ELBOW", (x0 + int(12 * u), yy), int(11 * u), GREY)
    _put(img, "BODY LINE", (x0 + int(78 * u), yy), int(11 * u), GREY)
    _put(img, f"{ang:.0f}°" if ang is not None else "--", (x0 + int(12 * u), yy + int(14 * u)), int(22 * u), WHITE)
    bcol = WHITE
    if body is not None:
        bcol = GREEN if body >= 165 else ORANGE if body >= 155 else RED
    _put(img, f"{body:.0f}°" if body is not None else "--", (x0 + int(78 * u), yy + int(14 * u)), int(22 * u), bcol)

    # -------- stress bar (right edge)
    if ctx.stress is not None:
        bw, bh = int(16 * u), int(150 * u)
        bx1 = W - pad
        bx0 = bx1 - bw
        by0 = pad + int(22 * u)
        _rounded_box(img, bx0 - 8 * u, pad, bx1 + 2 * u - 2, by0 + bh + int(30 * u), int(10 * u), DARK, 0.62)
        lbl = "LOAD"
        tw, _ = _text_size(lbl, int(10 * u))
        _put(img, lbl, (int((bx0 + bx1) / 2 - tw / 2), pad + int(5 * u)), int(10 * u), GREY)
        cv2.rectangle(img, (bx0, by0), (bx1, by0 + bh), (60, 60, 60), -1, cv2.LINE_AA)
        v = ctx.stress[i]
        if v is not None:
            v = float(min(100.0, max(0.0, v)))
            fill = int(bh * v / 100.0)
            # gradient fill
            for k in range(fill):
                yk = by0 + bh - 1 - k
                cv2.line(img, (bx0, yk), (bx1, yk), _stress_color(100.0 * k / bh), 1)
            vt = f"{v:.0f}"
            tw, _ = _text_size(vt, int(12 * u))
            _put(img, vt, (int((bx0 + bx1) / 2 - tw / 2), by0 + bh + int(6 * u)), int(12 * u), _stress_color(v))
        for k in (0.25, 0.5, 0.75):
            yk = by0 + int(bh * (1 - k))
            cv2.line(img, (bx0, yk), (bx0 + bw // 3, yk), (200, 200, 200), 1, cv2.LINE_AA)

    # -------- fault banner (bottom centre, above timeline)
    act = ctx.active_issues(t, linger=MSG_LINGER_S)
    tl_h = int(46 * u)
    if act:
        iss, _ = max(act, key=lambda x: SEV_RANK.get(x[0].get("severity"), 0))
        col = RED if iss.get("severity") == "high" else ORANGE
        code = FAULT_LABEL.get(iss.get("code"), str(iss.get("code", "")))
        msg = str(iss.get("msg", ""))
        fs = int(14 * u)
        tag = f" {code} "
        tw1, th1 = _text_size(tag, fs)
        tw2, _ = _text_size(msg, int(13 * u), bold=False)
        maxw = W - 2 * pad
        while tw1 + tw2 + 30 * u > maxw and len(msg) > 8:
            msg = msg[:-4].rstrip() + "…"
            tw2, _ = _text_size(msg, int(13 * u), bold=False)
        tot = tw1 + tw2 + int(26 * u)
        bx0 = int((W - tot) / 2)
        by1 = H - tl_h - pad
        by0 = by1 - th1 - int(14 * u)
        _rounded_box(img, bx0, by0, bx0 + tot, by1, int(10 * u), DARK, 0.72, border=col, bw=max(1, int(1.5 * u)))
        _rounded_box(img, bx0 + 6 * u, by0 + 5 * u, bx0 + 6 * u + tw1 + 4 * u, by1 - 5 * u, int(7 * u), col, 0.95)
        _put(img, tag, (bx0 + int(8 * u), by0 + int(7 * u)), fs, WHITE, shadow=False)
        _put(img, msg, (bx0 + int(16 * u) + tw1, by0 + int(8 * u)), int(13 * u), WHITE, bold=False)

    # -------- rep-score flash (top centre)
    rep, prog = ctx.flash_rep(t)
    if rep is not None and rep.get("score") is not None:
        alpha = 1.0 if prog < 0.6 else max(0.0, 1 - (prog - 0.6) / 0.4)
        pop = 1.0 + 0.25 * max(0.0, 1 - prog / 0.15)  # small "pop" at start
        sc = float(rep["score"])
        col = _score_color(sc)
        big = int(34 * u * pop)
        s1, s2 = f"REP {rep.get('n', '?')}", f"{sc:.0f}"
        w1, h1 = _text_size(s1, int(14 * u))
        w2, h2 = _text_size(s2, big)
        tot = max(w1, w2) + int(36 * u)
        cx = W // 2
        y0b = pad
        _rounded_box(img, cx - tot / 2, y0b, cx + tot / 2, y0b + h1 + h2 + int(14 * u), int(12 * u),
                     DARK, 0.7 * alpha, border=col if alpha > 0.5 else None, bw=max(1, int(2 * u)))
        _put(img, s1, (int(cx - w1 / 2), y0b + int(5 * u)), int(14 * u), GREY, alpha=alpha)
        _put(img, s2, (int(cx - w2 / 2), y0b + h1 + int(6 * u)), big, col, alpha=alpha)

    # -------- timeline (bottom)
    _draw_timeline(img, ctx, t, tl_h, pad)

    # -------- no-person hint
    if not ctx.frames[i].get("lm"):
        msg = "no person detected"
        tw, th = _text_size(msg, int(14 * u))
        _rounded_box(img, W / 2 - tw / 2 - 10 * u, H / 2 - th / 2 - 6 * u, W / 2 + tw / 2 + 10 * u,
                     H / 2 + th / 2 + 6 * u, int(8 * u), DARK, 0.6)
        _put(img, msg, (int(W / 2 - tw / 2), int(H / 2 - th / 2)), int(14 * u), GREY)


def _build_timeline(ctx: _Ctx, w: int, h: int):
    """Static timeline layer (BGR + alpha) — elbow trace, rep bands, fault ticks."""
    u = ctx.u
    layer = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    dur = max(ctx.duration, 1e-6)

    def X(t):
        return int(round(t / dur * (w - 1)))

    mask = np.zeros((h, w), np.uint8)
    r = int(8 * u)
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    layer[:] = DARK
    alpha[:] = (mask * 0.62).astype(np.uint8)
    # rep bands
    for rep in ctx.reps:
        s, e = rep.get("start"), rep.get("end")
        if s is None or e is None:
            continue
        col = _score_color(float(rep.get("score", 100) or 0))
        x0, x1 = X(s), max(X(s) + 1, X(e))
        band = layer[2:h - 2, x0:x1].astype(np.float32)
        layer[2:h - 2, x0:x1] = (band * 0.6 + np.array(col, np.float32) * 0.4).astype(np.uint8)
        alpha[2:h - 2, x0:x1] = np.maximum(alpha[2:h - 2, x0:x1], 190)
        lbl = str(rep.get("n", ""))
        sp = _text_sprite(lbl, int(10 * u), WHITE, True)
        if sp is not None and x1 - x0 > sp[0].shape[1]:
            xx, yy = (x0 + x1) // 2 - sp[0].shape[1] // 2, 2
            _blit(layer, sp, xx, yy)
            alpha[yy:yy + sp[0].shape[0], xx:xx + sp[0].shape[1]] = 230
    # elbow trace (60..180 deg mapped to height)
    pts = []
    for tt, a in zip(ctx.t, ctx.elbow):
        if a is None:
            if len(pts) > 1:
                cv2.polylines(layer, [np.array(pts, np.int32)], False, WHITE, max(1, int(1.3 * u)), cv2.LINE_AA)
                cv2.polylines(alpha, [np.array(pts, np.int32)], False, 255, max(1, int(1.3 * u)), cv2.LINE_AA)
            pts = []
            continue
        yv = h - 4 - (min(180, max(40, a)) - 40) / 140.0 * (h - 8 - 12 * u)
        pts.append((X(tt), int(yv)))
    if len(pts) > 1:
        cv2.polylines(layer, [np.array(pts, np.int32)], False, WHITE, max(1, int(1.3 * u)), cv2.LINE_AA)
        cv2.polylines(alpha, [np.array(pts, np.int32)], False, 255, max(1, int(1.3 * u)), cv2.LINE_AA)
    # fault ticks
    for t0, t1, iss, _ in ctx.issues:
        col = RED if iss.get("severity") == "high" else ORANGE
        x0, x1 = X(t0), max(X(t0) + 2, X(t1))
        cv2.rectangle(layer, (x0, h - int(5 * u)), (x1, h - 2), col, -1)
        alpha[h - int(5 * u):h - 1, x0:x1 + 1] = 255
    return layer, (alpha.astype(np.float32) / 255.0)[..., None]


def _draw_timeline(img, ctx: _Ctx, t, h, pad):
    W, H = ctx.W, ctx.H
    w = W - 2 * pad
    if ctx._timeline is None:
        ctx._timeline = _build_timeline(ctx, w, h)
    x0, y0 = pad, H - pad - h
    _blit(img, ctx._timeline, x0, y0)
    px = x0 + int(round(t / max(ctx.duration, 1e-6) * (w - 1)))
    cv2.line(img, (px, y0 - int(3 * ctx.u)), (px, y0 + h + int(2 * ctx.u)), ACCENT, max(2, int(2 * ctx.u)), cv2.LINE_AA)
    cv2.circle(img, (px, y0 - int(3 * ctx.u)), max(3, int(4 * ctx.u)), ACCENT, -1, cv2.LINE_AA)


def _annotate(frame_bgr: np.ndarray, ctx: _Ctx, i: int) -> np.ndarray:
    t = ctx.t[i]
    _draw_skeleton(frame_bgr, ctx, i, t)
    _draw_hud(frame_bgr, ctx, i, t)
    return frame_bgr


# ---------------------------------------------------------------- public API
def _ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    return "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else None


MAX_OUT_W = 960


def _stride(landmarks: dict, vfps: float) -> int:
    """Video frames per landmark frame (pose.extract subsamples high-fps clips)."""
    st = landmarks.get("frame_stride")
    if st:
        return max(1, int(st))
    lf = float(landmarks.get("fps") or 0)
    if lf > 1 and 1 < vfps < 240 and vfps / lf > 1.5:
        return max(1, int(round(vfps / lf)))
    return 1


def render(video_path: str, landmarks: dict, analysis: dict | None, physics: dict | None = None,
           out_path: str | None = None, progress=None) -> str:
    """Render the annotated video and return the path to a browser-playable H.264 mp4.

    Stops at min(video frames, landmark frames). `progress(fraction)` is called during drawing
    (0..0.95) and encoding (1.0).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = float(landmarks.get("fps") or (vfps if 1 < vfps < 240 else 30.0))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or landmarks.get("w") or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or landmarks.get("h") or 0)
    vtotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_lm = len(landmarks.get("frames") or [])
    if n_lm == 0:
        cap.release()
        raise ValueError("landmarks contain no frames")
    stride = _stride(landmarks, vfps)
    limit = min(n_lm, vtotal // stride) if vtotal > 0 else n_lm

    if out_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(OUT_DIR / f"{Path(str(video_path)).stem}_overlay.mp4")
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    ctx = None
    fd, tmp = tempfile.mkstemp(suffix=".mp4", dir=str(Path(out_path).parent))
    os.close(fd)
    writer = None
    written = 0
    try:
        for i in range(limit):
            ok, frame = cap.read()
            if not ok:
                break
            for _ in range(stride - 1):
                cap.grab()
            if ctx is None:
                vh, vw = frame.shape[:2]
                ctx = _Ctx(landmarks, analysis, physics, vw, vh, fps)
                sc = min(1.0, MAX_OUT_W / vw)          # big phone videos: encode at <= 960 px wide
                ow, oh = (int(round(vw * sc)) // 2 * 2, int(round(vh * sc)) // 2 * 2) if sc < 1 else (vw, vh)
                writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
                if not writer.isOpened():
                    raise RuntimeError("cv2.VideoWriter could not open temp file")
            img = _annotate(frame, ctx, i)
            if (ow, oh) != (vw, vh):
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
            writer.write(img)
            written += 1
            if progress and (i % 10 == 0 or i == limit - 1):
                progress(min(0.95, 0.95 * (i + 1) / limit))
    except BaseException:
        # interrupted (e.g. Streamlit rerun raised inside progress()) -> don't leak the temp file
        cap.release()
        if writer is not None:
            writer.release()
            writer = None
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if written == 0:
        os.remove(tmp)
        raise ValueError("Video contained no readable frames")

    ff = _ffmpeg()
    if ff is None:
        warnings.warn("ffmpeg not found; output is mp4v (may not play in browsers)")
        shutil.move(tmp, out_path)
    else:
        cmd = [ff, "-y", "-loglevel", "error", "-i", tmp,
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        try:
            os.remove(tmp)
        except OSError:
            pass
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({res.returncode}): {res.stderr[-500:]}")
    if progress:
        progress(1.0)
    return out_path


def snapshot(video_path: str, landmarks: dict, analysis: dict | None, t: float,
             physics: dict | None = None) -> np.ndarray:
    """Return the annotated frame (RGB uint8 ndarray) closest to time t seconds."""
    frames = landmarks.get("frames") or []
    if not frames:
        raise ValueError("landmarks contain no frames")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        vfps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = float(landmarks.get("fps") or (vfps if 1 < vfps < 240 else 30.0))
        vtotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        last = min(len(frames), vtotal) - 1 if vtotal > 0 else len(frames) - 1
        stride = _stride(landmarks, vfps)
        if vtotal > 0:
            last = min(len(frames), vtotal // stride) - 1
        i = int(min(max(0, round(float(t) * fps)), max(0, last)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
        ok, frame = cap.read()
        if not ok:  # seeking can fail on some containers: read sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(i * stride + 1):
                ok, fr = cap.read()
                if not ok:
                    break
                frame = fr
            if frame is None:
                raise ValueError("Could not read frame")
    finally:
        cap.release()
    vh, vw = frame.shape[:2]
    ctx = _Ctx(landmarks, analysis, physics, vw, vh, fps)
    out = _annotate(frame, ctx, i)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import json
    import sys
    import time

    lm = json.loads((ROOT / "samples" / "landmarks_demo_side.json").read_text())
    an = ph = None
    try:
        import analysis as _a
        an = _a.analyze(lm)
        try:
            import physics as _p
            ph = _p.simulate(lm, an)
        except Exception as ex:  # physics optional
            print("physics unavailable:", ex)
    except Exception as ex:
        print("analysis unavailable:", ex)
    t0 = time.time()
    out = render(str(ROOT / "samples" / "demo_side.mp4"), lm, an, ph,
                 out_path=sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"wrote {out} in {time.time() - t0:.1f}s")
