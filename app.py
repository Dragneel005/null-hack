"""FormSim — physical AI coach & simulator (Streamlit UI).

Run:  .venv/bin/streamlit run app.py

Pipeline: video -> pose.extract -> analysis.analyze -> physics.simulate -> coach.coach -> overlay.render
Theme layer: synth.generate (simulated labeled athletes) + benchmark (analyzer vs simulated ground truth).
Every module is imported defensively so the UI never crashes if one is missing or broken.
"""
from __future__ import annotations

import os

# Keep MediaPipe / TF / glog noise out of the console before anything imports them.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import hashlib
import importlib
import inspect
import json
import math
import shutil
import tempfile
import time
from datetime import datetime
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "outputs"
for _d in ("uploads", "cache", "overlays", "datasets", "reports"):
    (OUT / _d).mkdir(parents=True, exist_ok=True)

ACCENT = "#C6FF3D"
CYAN = "#3DE0FF"
RED = "#FF5470"
AMBER = "#FFB547"
MUTED = "#8B98A5"
BG = "#0A0E13"
CARD = "#131A23"

DEMOS = {
    "Side view — 6 slow reps (beach)": SAMPLES / "demo_side.mp4",
    "Robustness test — camera cuts & bystanders": SAMPLES / "demo_army.mp4",
}

FAULT_LABELS = {
    "SHALLOW": "Shallow depth",
    "NO_LOCKOUT": "No lockout",
    "HIP_SAG": "Hip sag",
    "HIP_PIKE": "Hip pike",
    "HEAD_DROP": "Head drop",
    "FAST_DESCENT": "Fast descent",
}

st.set_page_config(
    page_title="FormSim — physical AI coach & simulator",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, rgba(198,255,61,0.07), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(61,224,255,0.06), transparent 60%), {BG}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
section[data-testid="stSidebar"] {{ background: #0D131A; border-right: 1px solid #1C2530; }}
.fs-hero {{ padding: 1.4rem 1.6rem; border-radius: 18px; border: 1px solid #1E2833;
            background: linear-gradient(135deg, rgba(198,255,61,0.10), rgba(61,224,255,0.05) 55%, rgba(19,26,35,0.9));
            margin-bottom: 1.1rem; position: relative; overflow: hidden; }}
.fs-hero h1 {{ font-size: 2.35rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; color: #F4F8FB; }}
.fs-hero h1 span {{ color: {ACCENT}; }}
.fs-hero p {{ color: #A9B6C2; margin: .35rem 0 0 0; font-size: 1.02rem; }}
.fs-pill {{ display:inline-block; padding: .18rem .6rem; border-radius: 999px; font-size: .74rem; font-weight: 600;
            border: 1px solid #2A3644; color: #C9D4DE; margin-right: .35rem; margin-top: .7rem; background: rgba(10,14,19,.5); }}
.fs-pill.lime {{ border-color: rgba(198,255,61,.5); color: {ACCENT}; }}
.fs-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .75rem; margin: .4rem 0 1rem 0; }}
.fs-card {{ background: {CARD}; border: 1px solid #1E2833; border-radius: 14px; padding: .85rem 1rem; }}
.fs-card .lbl {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: {MUTED}; font-weight: 600; }}
.fs-card .val {{ font-size: 1.85rem; font-weight: 800; color: #F4F8FB; line-height: 1.15; margin-top: .15rem;
                 font-variant-numeric: tabular-nums; }}
.fs-card .val.lime {{ color: {ACCENT}; }}
.fs-card .val.red {{ color: {RED}; }}
.fs-card .val.amber {{ color: {AMBER}; }}
.fs-card .sub {{ font-size: .78rem; color: {MUTED}; margin-top: .1rem; }}
.fs-coach {{ background: linear-gradient(160deg, #141C26, #10161E); border: 1px solid #22303D; border-radius: 16px;
             padding: 1.1rem 1.25rem; }}
.fs-coach h3 {{ margin: 0 0 .3rem 0; font-size: 1.3rem; font-weight: 750; color: #F4F8FB; }}
.fs-coach .summary {{ color: #B7C3CE; font-size: .96rem; }}
.fs-grade {{ float: right; width: 64px; height: 64px; border-radius: 16px; display:flex; align-items:center;
             justify-content:center; font-size: 2.1rem; font-weight: 900; color: {BG}; margin-left: 1rem; }}
.fs-fix {{ border-left: 3px solid {ACCENT}; padding: .45rem .8rem; margin: .55rem 0; background: rgba(198,255,61,.04);
           border-radius: 0 10px 10px 0; }}
.fs-fix b {{ color: #F4F8FB; }}
.fs-fix .cue {{ font-family: 'JetBrains Mono', monospace; color: {ACCENT}; font-size: .86rem; }}
.fs-fix .why {{ color: #9FB0BF; font-size: .86rem; }}
.fs-src {{ font-size: .72rem; color: {MUTED}; }}
.fs-note {{ color: #A9B6C2; font-size: .92rem; }}
.fs-section {{ font-size: 1.05rem; font-weight: 700; color: #E6EDF3; margin: 1.1rem 0 .4rem 0; letter-spacing: -0.01em; }}
.fs-section span {{ color: {ACCENT}; }}
.fs-banner {{ border: 1px solid rgba(255,181,71,.45); background: rgba(255,181,71,.08); color: #FFD9A0;
              padding: .7rem 1rem; border-radius: 12px; margin-bottom: .8rem; }}
.fs-ph {{ border: 1px dashed #2A3644; border-radius: 14px; padding: 1.1rem; color: {MUTED}; text-align: center; }}
div[data-testid="stTabs"] button[role="tab"] p {{ font-size: 1.0rem; font-weight: 650; }}
div[data-testid="stTabs"] button[aria-selected="true"] p {{ color: {ACCENT}; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ color: {BG}; font-weight: 700; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(19,26,35,0.6)",
    font=dict(family="Inter, sans-serif", color="#C9D4DE", size=12),
    margin=dict(l=10, r=10, t=70, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
)


# ----------------------------------------------------------------------------------------------
# Defensive module loading
# ----------------------------------------------------------------------------------------------
def load_module(name: str) -> tuple[Any, str | None]:
    """Import a sibling module; return (module|None, error text|None). Never raises."""
    try:
        return importlib.import_module(name), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


MODS: dict[str, Any] = {}
MOD_ERR: dict[str, str] = {}
for _name in ("pose", "analysis", "physics", "coach", "overlay", "synth", "benchmark", "inject", "leaderboard"):
    _m, _err = load_module(_name)
    MODS[_name] = _m
    if _err:
        MOD_ERR[_name] = _err


def mod(name: str) -> Any:
    return MODS.get(name)


def placeholder(name: str, what: str = "") -> None:
    err = MOD_ERR.get(name, "not found")
    st.markdown(
        f"<div class='fs-ph'>◇ <b>{name}.py</b> module not ready yet {('— ' + what) if what else ''}"
        f"<br><span style='font-size:.8rem'>{err}</span></div>",
        unsafe_allow_html=True,
    )


def call_flex(fn: Callable, *args, **kwargs):
    """Call fn passing only the kwargs its signature accepts (modules are built in parallel)."""
    try:
        sig = inspect.signature(fn)
        if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        pass
    return fn(*args, **kwargs)


def show_exception(where: str, e: BaseException) -> None:
    st.error(f"{where} hit an error: {type(e).__name__}: {e}")
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


def to_jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def dumps(o: Any) -> str:
    return json.dumps(to_jsonable(o), indent=2, default=str)


def fmt(v: Any, nd: int = 1, suffix: str = "", none: str = "—") -> str:
    if v is None:
        return none
    try:
        f = float(v)
        if math.isnan(f):
            return none
        return f"{f:.{nd}f}{suffix}" if nd > 0 else f"{int(round(f))}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def cards(items: list[tuple[str, str, str, str]]) -> None:
    """items: (label, value, sub, colour-class)."""
    html = "".join(
        f"<div class='fs-card'><div class='lbl'>{l}</div><div class='val {c}'>{v}</div>"
        f"<div class='sub'>{s}</div></div>"
        for l, v, s, c in items
    )
    st.markdown(f"<div class='fs-grid'>{html}</div>", unsafe_allow_html=True)


def section(title: str, accent: str = "") -> None:
    st.markdown(f"<div class='fs-section'>{title} <span>{accent}</span></div>", unsafe_allow_html=True)


def arr(series: dict, key: str) -> np.ndarray:
    """Series list (with None) -> float array with NaN."""
    vals = (series or {}).get(key) or []
    return np.array([np.nan if v is None else float(v) for v in vals], dtype=float)


# ----------------------------------------------------------------------------------------------
# Caching helpers
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def file_hash_cached(path: str, mtime: float, size: int) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def file_hash(path: Path) -> str:
    stt = path.stat()
    return file_hash_cached(str(path), stt.st_mtime, stt.st_size)


def precomputed_landmarks(video: Path) -> Path | None:
    p = SAMPLES / f"landmarks_{video.stem}.json"
    return p if video.parent.resolve() == SAMPLES.resolve() and p.exists() else None


def get_landmarks(video: Path, vhash: str, model: str, progress: Callable[[float], None] | None) -> tuple[dict, str]:
    """Pose track. Demo clips: cached on disk by (hash, model). User uploads: kept only in this
    browser session's memory (privacy: nothing derived from an uploaded video is written to disk)."""
    is_sample = video.parent.resolve() == SAMPLES.resolve()
    mem = st.session_state.setdefault("_lm_mem", {})
    mkey = f"{vhash}_{model}"
    if mkey in mem:
        return mem[mkey], "session cache"
    cache = OUT / "cache" / f"{vhash}_{model}.json"
    if is_sample and cache.exists():
        return json.loads(cache.read_text()), "disk cache"
    pre = precomputed_landmarks(video)
    if pre is not None:   # demo clips always use the shipped track (no 1-2 min MediaPipe run on stage)
        return json.loads(pre.read_text()), "precomputed demo track"
    if mod("pose") is None:
        raise RuntimeError(f"pose module unavailable: {MOD_ERR.get('pose')}")
    lm = call_flex(mod("pose").extract, str(video), model=model, progress=progress)
    if is_sample:
        try:
            cache.write_text(json.dumps(lm))
        except OSError:
            pass
    else:
        mem.clear()   # keep at most one upload's track in memory
        mem[mkey] = lm
    return lm, "MediaPipe Pose Landmarker"


def demo_hashes() -> set[str]:
    """Content hashes of the shipped sample clips (so they can't be submitted to the leaderboard)."""
    out = set()
    for p in (SAMPLES.iterdir() if SAMPLES.exists() else []):
        if p.suffix.lower() in (".mp4", ".mov", ".webm", ".avi", ".m4v"):
            try:
                out.add(file_hash(p))
            except OSError:
                pass
    return out


@st.cache_data(show_spinner=False)
def _load_json_cached(path: str, mtime: float) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def load_precomputed_bench() -> dict | None:
    """samples/benchmark_200.json written by `python benchmark.py 200 samples/benchmark_200.json`."""
    p = SAMPLES / "benchmark_200.json"
    if not p.exists():
        return None
    d = _load_json_cached(str(p), p.stat().st_mtime)
    return d if d and "eval" in d and "s2r" in d else None


# ----------------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------------
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
with st.sidebar:
    st.markdown(f"<h2 style='margin-bottom:0'>◆ Form<span style='color:{ACCENT}'>Sim</span></h2>"
                "<div class='fs-src'>physical AI coach & simulator</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Athlete**")
    body_mass = st.slider("Body mass (kg)", 40.0, 140.0, 75.0, 0.5)
    height_m = st.slider("Height (m)", 1.40, 2.10, 1.75, 0.01)
    st.markdown("**Vision**")
    pose_model = st.radio("Pose model", ["full", "lite"], horizontal=True,
                          help="full = more accurate, lite = ~2x faster on CPU")
    st.markdown("**Coach**")
    use_claude = st.toggle("Use Claude coach", value=HAS_KEY, disabled=not HAS_KEY,
                           help=None if HAS_KEY else "Set ANTHROPIC_API_KEY to enable the Claude coach; "
                                                     "the offline rules coach is used otherwise.")
    if not HAS_KEY:
        st.caption("No ANTHROPIC_API_KEY — using the offline rules coach (same JSON schema).")
    st.markdown("**Demo clip**")
    demo_choice = st.selectbox("Demo clip", list(DEMOS.keys()), label_visibility="collapsed")
    _inj_opts = getattr(MODS.get("inject"), "FAULT_OPTIONS", {"none": "None (real form)"})
    sim_fault = st.selectbox("⚗ Simulate a fault on this real clip", list(_inj_opts),
                             format_func=lambda k: _inj_opts.get(k, k), key="sim_fault",
                             help="Real motion, simulated fault: the tracked joints are geometrically edited "
                                  "(e.g. hips pushed below the body line) before analysis — watch the analyzer, "
                                  "physics and coach react.")
    render_overlay = st.toggle("Render overlay video", value=True,
                               help="Draw skeleton + HUD onto the video (takes a few seconds per clip).")
    st.divider()
    ready = [n for n in MODS if MODS[n] is not None]
    st.markdown("<div class='fs-src'>Modules: " + " ".join(
        f"<span style='color:{ACCENT if MODS[n] else RED}'>●</span>&nbsp;{n}" for n in MODS) + "</div>",
        unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------------------------
def _hero_stats() -> str:
    pre = load_precomputed_bench()
    if not pre:
        return ""
    ev, s2 = pre["eval"], pre["s2r"]
    try:
        ml, ru = float(s2["heldout_exact_acc_ml"]), float(s2["heldout_exact_acc_rules"])
        stats = [(f"{pre.get('n')}", "simulated athletes"),
                 (f"{float(ev['rep_count_exact_acc']) * 100:.0f}%", "rep-count accuracy"),
                 (f"{ml * 100:.0f}% vs {ru * 100:.0f}%", "sim-trained AI vs hand rules (unseen athletes)"),
                 ("0", "real labels used to train")]
    except Exception:  # noqa: BLE001
        return ""
    cells = "".join(f"<div><div style='font-size:1.35rem;font-weight:800;color:{ACCENT}'>{v}</div>"
                    f"<div class='fs-src'>{k}</div></div>" for v, k in stats)
    return (f"<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin-top:.9rem'>"
            f"{cells}</div>")


st.markdown(
    f"""
<div class='fs-hero'>
  <h1>Form<span>Sim</span> — push-up form coach</h1>
  <p>Film a side-view push-up set. We count every rep, tell you how many were clean, grade your form and
  give you the fixes that matter — then put you on the leaderboard.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------------
def run_pipeline(video: Path, vhash: str, display_name: str, private: bool = False) -> dict:
    """Run pose -> analysis -> physics -> coach -> overlay with st.status progress. Returns result dict.
    private=True (user uploads): the overlay is rendered next to the temp video, never into outputs/."""
    res: dict[str, Any] = {"video": str(video), "name": display_name, "hash": vhash, "errors": {}, "timings": {}}
    with st.status(f"Analyzing **{display_name}** …", expanded=True) as status:
        # 1. Pose
        t0 = time.time()
        st.write("① Pose tracking — MediaPipe Pose Landmarker (33 keypoints → 19 joints)")
        bar = st.progress(0.0)
        try:
            lm, how = get_landmarks(video, vhash, pose_model, lambda f: bar.progress(min(1.0, float(f))))
            bar.progress(1.0)
            if sim_fault != "none" and mod("inject") is not None:
                lm = mod("inject").inject_fault(lm, sim_fault)
                how += f" + simulated {sim_fault}"
                st.write(f"   ⚗ simulated fault injected into the real track: {_inj_opts.get(sim_fault, sim_fault)}")
            res["landmarks"] = lm
            res["pose_how"] = how
            st.write(f"   ✓ {lm.get('detected_frames')}/{lm.get('n_frames')} frames with a person "
                     f"· {lm.get('w')}×{lm.get('h')} @ {fmt(lm.get('fps'), 0)} fps · {how}")
        except Exception as e:  # noqa: BLE001
            res["errors"]["pose"] = e
            status.update(label="Pose tracking failed", state="error")
            return res
        res["timings"]["pose"] = time.time() - t0

        # 2. Analysis
        t0 = time.time()
        st.write("② Biomechanics — rep state machine, joint angles, fault rules, tempo, fatigue")
        if mod("analysis"):
            try:
                res["analysis"] = mod("analysis").analyze(lm)
                s = res["analysis"].get("summary", {})
                st.write(f"   ✓ {s.get('rep_count')} reps · {s.get('clean_reps')} clean · avg score "
                         f"{fmt(s.get('avg_score'), 0)}")
            except Exception as e:  # noqa: BLE001
                res["errors"]["analysis"] = e
                st.write(f"   ✗ analysis error: {e}")
        else:
            st.write("   ◇ analysis module not ready")
        res["timings"]["analysis"] = time.time() - t0

        an = res.get("analysis")
        # 3. Physics
        t0 = time.time()
        st.write("③ Physics — segment masses (de Leva), COM dynamics, support reactions, joint torques")
        if mod("physics") and an:
            try:
                res["physics"] = call_flex(mod("physics").simulate, lm, an,
                                           body_mass_kg=body_mass, height_m=height_m)
                ps = res["physics"].get("summary", {})
                st.write(f"   ✓ avg hand load {fmt((ps.get('avg_hand_load_frac') or 0) * 100, 0)}% BW · "
                         f"peak {fmt(ps.get('peak_hand_force_N'), 0)} N")
            except Exception as e:  # noqa: BLE001
                res["errors"]["physics"] = e
                st.write(f"   ✗ physics error: {e}")
        else:
            st.write("   ◇ physics skipped")
        res["timings"]["physics"] = time.time() - t0

        # 4. Coach
        t0 = time.time()
        st.write("④ Coach — " + ("Claude, grounded only in the measured numbers" if use_claude
                                 else "offline rules engine"))
        if mod("coach") and an:
            try:
                res["coach"] = call_flex(mod("coach").coach, an, res.get("physics"), use_llm=bool(use_claude))
                st.write(f"   ✓ grade {res['coach'].get('grade')} · source: {res['coach'].get('source')}")
            except Exception as e:  # noqa: BLE001
                res["errors"]["coach"] = e
                st.write(f"   ✗ coach error: {e}")
        else:
            st.write("   ◇ coach skipped")
        res["timings"]["coach"] = time.time() - t0

        # 5. Overlay
        t0 = time.time()
        if render_overlay:
            st.write("⑤ Overlay — skeleton, fault highlights, HUD → H.264 mp4")
            if mod("overlay") and an:
                code_v = int(sum((ROOT / f).stat().st_mtime for f in ("analysis.py", "physics.py", "overlay.py")
                                 if (ROOT / f).exists())) % 100000
                # body mass/height don't change the drawing enough to justify a 20 s re-render
                okey = f"{vhash}_{pose_model}_{sim_fault}_{code_v}"
                out_path = (video.parent if private else OUT / "overlays") / f"{okey}.mp4"
                if not private and out_path.exists() and out_path.stat().st_size > 1000:
                    res["overlay"] = str(out_path)
                    st.write("   ✓ overlay (cached)")
                else:
                    obar = st.progress(0.0)
                    try:
                        p = call_flex(mod("overlay").render, str(video), lm, an, res.get("physics"),
                                      out_path=str(out_path),
                                      progress=lambda f: obar.progress(min(1.0, float(f))))
                        obar.progress(1.0)
                        res["overlay"] = str(p or out_path)
                        st.write("   ✓ overlay rendered")
                    except Exception as e:  # noqa: BLE001
                        res["errors"]["overlay"] = e
                        st.write(f"   ✗ overlay error: {e}")
            else:
                st.write("   ◇ overlay skipped")
        res["timings"]["overlay"] = time.time() - t0
        bad = [k for k in res["errors"]]
        status.update(label=f"Done — {display_name}" + (f" (issues in: {', '.join(bad)})" if bad else ""),
                      state="error" if "pose" in bad or "analysis" in bad else "complete", expanded=False)
    return res


def worst_reps(an: dict) -> list[tuple[dict, list, float]]:
    """Up to 3 lowest-scoring reps with their issues and the time to freeze a snapshot at."""
    out = []
    for r in sorted(an.get("reps", []), key=lambda r: (r.get("score") or 100))[:3]:
        iss = sorted(r.get("issues") or [], key=lambda i: {"high": 0, "med": 1}.get(i.get("severity"), 2))
        t = iss[0].get("t") if iss and iss[0].get("t") is not None else r.get("bottom", r.get("start", 0))
        out.append((r, iss, float(t or 0)))
    return out


def run_upload_pipeline(data: bytes, suffix: str, vhash: str, display_name: str) -> dict:
    """Privacy wrapper for user uploads: the video lives in a temp dir only while it's analyzed.
    The overlay + worst-rep snapshots are read into memory (this session only), then everything is deleted."""
    tmp = Path(tempfile.mkdtemp(prefix="formsim_"))
    try:
        vid = tmp / f"clip{suffix}"
        vid.write_bytes(data)
        res = run_pipeline(vid, vhash, display_name, private=True)
        ov = res.get("overlay")
        if ov and Path(ov).exists():
            res["overlay_bytes"] = Path(ov).read_bytes()
        res["overlay"] = None
        an = res.get("analysis")
        snaps: dict[float, Any] = {}
        if an and res.get("landmarks") and mod("overlay") is not None and hasattr(mod("overlay"), "snapshot"):
            for _r, _iss, t in worst_reps(an):
                try:
                    snaps[t] = mod("overlay").snapshot(str(vid), res["landmarks"], an, t)
                except Exception:  # noqa: BLE001
                    pass
        res["snapshots"] = snaps
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["video"] = ""
    res["upload"] = True
    return res


def result_key(vhash: str) -> str:
    return f"{vhash}|{pose_model}|{body_mass}|{height_m}|{bool(use_claude)}|{render_overlay}|{sim_fault}"


# ----------------------------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------------------------
def angle_chart(an: dict) -> go.Figure:
    s = an.get("series", {})
    t = arr(s, "t")
    elbow = arr(s, "elbow")
    body = arr(s, "body_line")
    th = an.get("thresholds", {}) or {}
    depth = float(th.get("depth_deg", th.get("depth", 95)) or 95)
    lock = float(th.get("lockout_deg", th.get("lockout", 155)) or 155)
    fig = go.Figure()
    # Good zones
    fig.add_hrect(y0=0, y1=depth, fillcolor=ACCENT, opacity=0.06, line_width=0,
                  annotation_text="depth zone", annotation_position="bottom left",
                  annotation_font_color=ACCENT, annotation_font_size=10)
    fig.add_hrect(y0=lock, y1=185, fillcolor=CYAN, opacity=0.05, line_width=0,
                  annotation_text="lockout / straight body", annotation_position="top left",
                  annotation_font_color=CYAN, annotation_font_size=10)
    # Rep bands
    for i, r in enumerate(an.get("reps", [])):
        if r.get("start") is None or r.get("end") is None:
            continue
        sc = r.get("score", 100) or 0
        col = ACCENT if sc >= 80 else (AMBER if sc >= 60 else RED)
        fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=col, opacity=0.07, line_width=0)
        fig.add_annotation(x=(r["start"] + r["end"]) / 2, y=192, text=f"R{r.get('n', i + 1)} · {fmt(sc, 0)}",
                           showarrow=False, font=dict(size=10, color=col))
    fig.add_trace(go.Scatter(x=t, y=elbow, name="Elbow angle", line=dict(color=ACCENT, width=2.4),
                             connectgaps=False, hovertemplate="%{y:.0f}°"))
    if np.isfinite(body).any():
        fig.add_trace(go.Scatter(x=t, y=body, name="Body line (shoulder-hip-ankle)",
                                 line=dict(color=CYAN, width=1.8), hovertemplate="%{y:.0f}°"))
    # Issue markers
    xs, ys, txt, cols = [], [], [], []
    for r in an.get("reps", []):
        for iss in r.get("issues", []) or []:
            ti = iss.get("t")
            if ti is None or len(t) == 0:
                continue
            j = int(np.nanargmin(np.abs(t - ti))) if np.isfinite(t).any() else 0
            yv = elbow[j] if j < len(elbow) and np.isfinite(elbow[j]) else depth
            xs.append(ti); ys.append(yv)
            txt.append(f"Rep {r.get('n')}: {iss.get('msg', iss.get('code'))}")
            cols.append({"high": RED, "med": AMBER}.get(iss.get("severity"), "#FFE066"))
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Faults",
                                 marker=dict(symbol="x", size=11, color=cols, line=dict(width=1, color=BG)),
                                 text=txt, hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, height=380, title="Joint angles over time")
    fig.update_yaxes(title="degrees", range=[30, 198], gridcolor="#1E2833")
    fig.update_xaxes(title="time (s)", gridcolor="#1E2833")
    return fig


def line_fig(t: np.ndarray, traces: list[tuple[str, np.ndarray, str]], title: str, ytitle: str,
             height: int = 300, reps: list | None = None, fill: bool = False) -> go.Figure:
    fig = go.Figure()
    for r in reps or []:
        if r.get("start") is not None and r.get("end") is not None:
            fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor="#FFFFFF", opacity=0.03, line_width=0)
    for name, y, col in traces:
        fig.add_trace(go.Scatter(x=t, y=y, name=name, line=dict(color=col, width=2),
                                 fill="tozeroy" if fill else None,
                                 fillcolor=f"rgba{tuple(int(col[i:i + 2], 16) for i in (1, 3, 5)) + (0.12,)}"
                                 if fill else None))
    fig.update_layout(**PLOT_LAYOUT, height=height, title=title)
    fig.update_yaxes(title=ytitle, gridcolor="#1E2833")
    fig.update_xaxes(title="time (s)", gridcolor="#1E2833")
    return fig


SKELETON = [("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
            ("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "foot"), ("ankle", "foot")]


def side_prefix(lm_frame: dict, preferred: str | None = None) -> str:
    if preferred in ("left", "right"):
        return preferred[0]
    def vis(p):
        return sum(lm_frame.get(f"{p}_{j}", [0, 0, 0])[2] for j in ("shoulder", "elbow", "wrist", "hip", "ankle"))
    return "l" if vis("l") >= vis("r") else "r"


def skeleton_xy(lm_frame: dict, p: str) -> tuple[list, list]:
    xs, ys = [], []
    for a, b in SKELETON:
        A, B = lm_frame.get(f"{p}_{a}"), lm_frame.get(f"{p}_{b}")
        if A and B:
            xs += [A[0], B[0], None]
            ys += [A[1], B[1], None]
    return xs, ys


def force_diagram(lm: dict, an: dict | None, ph: dict, idx: int, mass: float) -> go.Figure:
    """Side-view free-body diagram for one frame: skeleton, COM, weight and support reactions."""
    frames = lm.get("frames", [])
    fr = frames[idx] if 0 <= idx < len(frames) else None
    fig = go.Figure()
    W, H = lm.get("w", 640), lm.get("h", 480)
    if not fr or not fr.get("lm"):
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="no person in this frame",
                           showarrow=False, font=dict(color=MUTED))
        fig.update_layout(**PLOT_LAYOUT, height=420)
        return fig
    L = fr["lm"]
    p = side_prefix(L, (an or {}).get("side"))
    xs, ys = skeleton_xy(L, p)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#DDE6EE", width=5),
                             marker=dict(size=8, color=ACCENT), name="body", hoverinfo="skip"))
    s = ph.get("series", {})
    frac = arr(s, "hand_load_frac")
    hf = frac[idx] if idx < len(frac) and np.isfinite(frac[idx]) else 0.68
    g = 9.81
    BW = mass * g
    hand_N, feet_N = hf * BW, (1 - hf) * BW
    wrist = L.get(f"{p}_wrist")
    foot = L.get(f"{p}_foot") or L.get(f"{p}_ankle")
    sh, hip = L.get(f"{p}_shoulder"), L.get(f"{p}_hip")
    # COM: from physics (metres) if mappable, else weighted between supports by load share
    com = None
    if wrist and foot:
        com = [foot[0] + hf * (wrist[0] - foot[0]), (sh[1] + hip[1]) / 2 if sh and hip else (wrist[1] + foot[1]) / 2]
    pts = [v for k, v in L.items() if k.startswith(p + "_")] or list(L.values())
    allx = [v[0] for v in pts]
    ally = [v[1] for v in pts]
    span = max(max(allx) - min(allx), max(ally) - min(ally), 1.0)
    scale = 0.38 * span / max(BW, 1)  # px per N, relative to body size so arrows read well
    arrows = []
    if wrist:
        arrows.append((wrist[0], wrist[1], wrist[0], wrist[1] - hand_N * scale, ACCENT,
                       f"Hands {hand_N:.0f} N ({hf * 100:.0f}% BW)"))
    if foot:
        arrows.append((foot[0], foot[1], foot[0], foot[1] - feet_N * scale, CYAN,
                       f"Feet {feet_N:.0f} N ({(1 - hf) * 100:.0f}%)"))
    if com:
        arrows.append((com[0], com[1], com[0], com[1] + BW * scale, RED, f"Weight {BW:.0f} N"))
        fig.add_trace(go.Scatter(x=[com[0]], y=[com[1]], mode="markers+text", text=["COM"],
                                 textposition="top center", marker=dict(size=14, color=RED, symbol="circle-dot"),
                                 name="centre of mass", hoverinfo="skip"))
    for x0, y0, x1, y1, col, label in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=3.5, arrowcolor=col, text="")
        fig.add_annotation(x=x1, y=y1, text=f"<b>{label}</b>", showarrow=False, font=dict(color=col, size=12),
                           yshift=-14 if y1 < y0 else 14, bgcolor="rgba(10,14,19,0.7)")
    # Torque readouts at joints
    for key, joint, col in (("elbow_torque_Nm", "elbow", AMBER), ("shoulder_torque_Nm", "shoulder", AMBER),
                            ("lumbar_moment_Nm", "hip", AMBER)):
        v = arr(s, key)
        J = L.get(f"{p}_{joint}")
        if J and idx < len(v) and np.isfinite(v[idx]):
            fig.add_annotation(x=J[0], y=J[1], text=f"{joint} {abs(v[idx]):.0f} N·m", showarrow=False, yshift=-22,
                               font=dict(color=col, size=11), bgcolor="rgba(10,14,19,0.6)")
    pad = 0.12 * span
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"}, height=430, showlegend=False,
                      title=f"Free-body diagram · t = {fr.get('t', 0):.2f} s")
    fig.update_xaxes(range=[min(allx) - pad, max(allx) + pad], visible=False)
    fig.update_yaxes(range=[max(ally) + 0.42 * span, min(ally) - 0.38 * span], visible=False,
                     scaleanchor="x", scaleratio=1)
    return fig


def skeleton_animation(sample_lm: dict, max_frames: int = 90) -> go.Figure:
    """Fallback animated skeleton preview built directly from a landmarks dict."""
    frames = [f for f in sample_lm.get("frames", [])]
    if not frames:
        return go.Figure()
    step = max(1, len(frames) // max_frames)
    frames = frames[::step]
    first = next((f for f in frames if f.get("lm")), None)
    if first is None:
        return go.Figure()
    p = side_prefix(first["lm"])
    W, H = sample_lm.get("w", 640), sample_lm.get("h", 480)

    def trace(f):
        if not f.get("lm"):
            return go.Scatter(x=[], y=[], mode="lines+markers")
        xs, ys = skeleton_xy(f["lm"], p)
        return go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=ACCENT, width=5),
                          marker=dict(size=7, color="#F4F8FB"))

    fig = go.Figure(data=[trace(first)],
                    frames=[go.Frame(data=[trace(f)], name=str(k)) for k, f in enumerate(frames)])
    dur = 1000 * (frames[1]["t"] - frames[0]["t"]) if len(frames) > 1 else 50
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": False}, height=380, showlegend=False,
                      title="Simulated athlete (2D projection)",
                      updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.05, xanchor="left",
                                        buttons=[dict(label="▶ Play", method="animate",
                                                      args=[None, dict(frame=dict(duration=max(20, dur), redraw=True),
                                                                       fromcurrent=True, mode="immediate")]),
                                                 dict(label="❚❚", method="animate",
                                                      args=[[None], dict(mode="immediate")])])])
    fig.update_xaxes(range=[0, W], visible=False)
    fig.update_yaxes(range=[H, 0], visible=False, scaleanchor="x")
    return fig


# ----------------------------------------------------------------------------------------------
# Report markdown
# ----------------------------------------------------------------------------------------------
def report_markdown(res: dict) -> str:
    an, ph, rp = res.get("analysis") or {}, res.get("physics") or {}, res.get("coach") or {}
    s, ps = an.get("summary", {}), ph.get("summary", {})
    lines = [f"# FormSim report — {res.get('name')}", ""]
    if rp:
        lines += [f"## {rp.get('headline', '')}  (grade {rp.get('grade', '?')})", "", rp.get("summary", ""), ""]
    lines += ["## Set summary", "",
              f"- Reps: **{s.get('rep_count')}** ({s.get('clean_reps')} clean), avg score {fmt(s.get('avg_score'), 0)}",
              f"- Tempo: descent {fmt(s.get('avg_descent_s'), 2, ' s')}, ascent {fmt(s.get('avg_ascent_s'), 2, ' s')}",
              f"- Fatigue onset: rep {s.get('fatigue_from_rep') or 'not detected'}",
              f"- Issues: {', '.join(f'{k}×{v}' for k, v in (s.get('issue_counts') or {}).items()) or 'none'}"]
    if ps and ps.get("peak_hand_force_N") is not None:
        lines += [f"- Avg hand load: {fmt((ps.get('avg_hand_load_frac') or 0) * 100, 0)}% body weight; "
                  f"peak hand force {fmt(ps.get('peak_hand_force_N'), 0)} N ≈ bench press "
                  f"{fmt(ps.get('equivalent_bench_kg'), 0)} kg; work {fmt(ps.get('total_work_J'), 0)} J"]
    if rp.get("top_fixes"):
        lines += ["", "## Top fixes", ""]
        for f in rp["top_fixes"]:
            lines += [f"- **{f.get('title')}** — cue: _{f.get('cue')}_. {f.get('why', '')} (reps {f.get('reps')})"]
    if rp.get("strengths"):
        lines += ["", "## Strengths", ""] + [f"- {x}" for x in rp["strengths"]]
    lines += ["", "## Per rep", "", "| Rep | Score | Min elbow | Max elbow | Hip sag | Descent | Issues |",
              "|---|---|---|---|---|---|---|"]
    for r in an.get("reps", []):
        lines.append(f"| {r.get('n')} | {fmt(r.get('score'), 0)} | {fmt(r.get('min_elbow'), 0)}° | "
                     f"{fmt(r.get('max_elbow'), 0)}° | {fmt(r.get('max_hip_sag'), 0)}° | "
                     f"{fmt(r.get('descent_s'), 2)} s | "
                     f"{', '.join(i.get('code', '') for i in r.get('issues', [])) or '—'} |")
    if rp.get("next_session"):
        lines += ["", "## Next session", "", rp["next_session"]]
    lines += ["", f"_Generated by FormSim · coach source: {rp.get('source', 'n/a')}_"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Tab bodies
# ----------------------------------------------------------------------------------------------
GRADE_COL = {"A": ACCENT, "B": CYAN, "C": AMBER, "D": RED}


def coach_card(rp: dict) -> None:
    from html import escape as e
    g = e(str(rp.get("grade", "?")))
    parts = [f"<div class='fs-coach'><div class='fs-grade' style='background:{GRADE_COL.get(g, MUTED)}'>{g}</div>",
             f"<div class='fs-src'>AI COACH · source: {e(str(rp.get('source', '?')))}</div>",
             f"<h3>{e(str(rp.get('headline', '')))}</h3>",
             f"<div class='summary'>{e(str(rp.get('summary', '')))}</div>"]
    for f in (rp.get("top_fixes") or [])[:2]:
        reps = f" · reps {', '.join(map(str, f.get('reps') or []))}" if f.get("reps") else ""
        parts.append(f"<div class='fs-fix'><b>{e(str(f.get('title', '')))}</b>{reps}"
                     f"<div class='cue'>“{e(str(f.get('cue', '')))}”</div>"
                     f"<div class='why'>{e(str(f.get('why', '')))}</div></div>")
    strengths = "".join(f"<li>{e(str(x))}</li>" for x in (rp.get("strengths") or [])[:3])
    if strengths:
        parts.append(f"<div style='margin-top:.5rem;color:#C9D4DE'><b>Strengths</b>"
                     f"<ul style='margin:.2rem 0'>{strengths}</ul></div>")
    if rp.get("next_session"):
        parts.append(f"<div style='margin-top:.4rem;color:#C9D4DE'><b>Next session:</b> "
                     f"{e(str(rp['next_session']))}</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def scorecard_df(an: dict) -> pd.DataFrame:
    rows = []
    for r in an.get("reps", []):
        rows.append({
            "Rep": r.get("n"),
            "Score": r.get("score"),
            "Depth (min elbow °)": r.get("min_elbow"),
            "Lockout (max elbow °)": r.get("max_elbow"),
            "Hip sag °": r.get("max_hip_sag"),
            "Hip pike °": r.get("max_hip_pike"),
            "Descent s": r.get("descent_s"),
            "Ascent s": r.get("ascent_s"),
            "Issues": ", ".join(FAULT_LABELS.get(i.get("code"), i.get("code", "")) for i in r.get("issues", []) or [])
                      or "✓ clean",
        })
    return pd.DataFrame(rows)


def style_scorecard(df: pd.DataFrame):
    def score_col(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        c = ACCENT if v >= 80 else (AMBER if v >= 60 else RED)
        return f"background-color: {c}22; color: {c}; font-weight: 700"

    def issue_col(v):
        return f"color: {ACCENT}" if str(v).startswith("✓") else f"color: {AMBER}"

    num_fmt = {c: ("{:.2f}" if c.endswith(" s") else "{:.0f}")
               for c in df.columns if c not in ("Rep", "Issues")}
    sty = df.style.format(num_fmt, na_rep="—")
    try:
        sty = sty.map(score_col, subset=["Score"]).map(issue_col, subset=["Issues"])
    except AttributeError:  # older pandas
        sty = sty.applymap(score_col, subset=["Score"]).applymap(issue_col, subset=["Issues"])
    return sty


@st.cache_data(show_spinner=False, max_entries=32)
def snapshot_cached(video: str, key: str, t: float, _lm: dict, _an: dict):
    return mod("overlay").snapshot(video, _lm, _an, t)


def tab_analyze() -> None:
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    with c1:
        up = st.file_uploader("Upload a side-view push-up video", type=["mp4", "mov", "webm", "avi", "m4v"],
                              help="Phone on the floor, side-on, whole body in frame. Up to 300 MB.")
    with c2:
        demo_clicked = st.button(f"▶ Run demo: {demo_choice.split(' — ')[0]}", type="primary",
                                 width="stretch", key="demo_btn")
        st.caption("Uses the demo clip chosen in the sidebar.")
    u1, u2 = st.columns([1, 2], vertical_alignment="center")
    with u1:
        st.text_input("Username for the leaderboard (optional)", key="lb_user", max_chars=20,
                      placeholder="e.g. pushup_pete")
    with u2:
        lbm = mod("leaderboard")
        st.checkbox(getattr(lbm, "CONSENT_TEXT", "This is me, and I agree to my username and score being shown "
                                                 "on the public leaderboard"), key="lb_consent")
    st.caption("🔒 " + getattr(lbm, "PRIVACY_TEXT", "We don't keep your video — only your username and score."))

    upload_bytes = None
    if up is not None:
        upload_bytes = up.getvalue()
        h = hashlib.sha1(upload_bytes).hexdigest()[:16]
        if st.session_state.get("_last_upload_hash") != h:
            st.session_state["_last_upload_hash"] = h
            # privacy: the upload is NOT written to disk here; run_upload_pipeline uses a temp file and deletes it
            st.session_state["source"] = {"path": "", "hash": h, "name": up.name, "upload": True,
                                          "suffix": Path(up.name).suffix.lower() or ".mp4"}
    # ?demo=1 in the URL auto-runs the demo once (handy for a shareable judge link)
    auto = st.query_params.get("demo") and not st.session_state.get("_auto_demo_done")
    if auto:
        st.session_state["_auto_demo_done"] = True
    if demo_clicked or auto:
        p = DEMOS[demo_choice]
        if not p.exists():
            st.error(f"Demo clip missing: {p}")
        else:
            st.session_state["source"] = {"path": str(p), "hash": file_hash(p), "name": p.name}

    src = st.session_state.get("source")
    if not src:
        st.markdown(
            "<div class='fs-ph' style='padding:2.2rem'>Upload a clip or hit <b>Run demo</b> — you'll get your rep "
            "count, clean reps, a form grade, an annotated video and your top coaching tips.</div>",
            unsafe_allow_html=True)
        return
    is_upload = bool(src.get("upload"))
    if is_upload and upload_bytes is not None and st.session_state.get("_last_upload_hash") != src["hash"]:
        upload_bytes = None

    key = result_key(src["hash"])
    cache = st.session_state.setdefault("results", {})
    if key not in cache:
        if is_upload:
            if upload_bytes is None:
                st.info("Your video isn't kept on our side — upload it again to re-analyze with these settings.")
                return
            # only the latest upload's result is kept (in memory) for this session
            for k in [k for k, v in cache.items() if v.get("upload")]:
                cache.pop(k, None)
            cache[key] = run_upload_pipeline(upload_bytes, src.get("suffix", ".mp4"), src["hash"], src["name"])
        else:
            cache[key] = run_pipeline(Path(src["path"]), src["hash"], src["name"])
    res = cache[key]
    st.session_state["current"] = res

    if "pose" in res.get("errors", {}):
        st.error("We couldn't read this video file. Please upload an .mp4 / .mov straight from your phone camera "
                 "(H.264 or HEVC), filmed side-on with your whole body in frame.")
    for stage, e in res.get("errors", {}).items():
        with st.expander(f"Technical details ({stage} stage)", expanded=False):
            st.code(f"{type(e).__name__}: {str(e).replace(str(ROOT), '.')}\n\n" +
                    "".join(traceback.format_exception(type(e), e, e.__traceback__)).replace(str(ROOT), "."))
    if "landmarks" not in res:
        return
    lm, an, ph, rp = res["landmarks"], res.get("analysis"), res.get("physics"), res.get("coach")
    if an is None:
        if mod("analysis") is None:
            placeholder("analysis", "rep counting and fault detection will appear here")
        return

    from html import escape as _esc
    if not an.get("confidence_ok", True):
        warns = "<br>".join(f"• {_esc(str(w))}" for w in an.get("warnings", []))
        conf = float(an.get("confidence") or 0)
        thr = float((an.get("thresholds") or {}).get("min_confidence", 0.5))
        head = (f"Low tracking confidence ({fmt(conf * 100, 0)}%)." if conf < thr
                else "Couldn't score this clip.")
        st.markdown(f"<div class='fs-banner'><b>{head}</b> For best results film from the side with the whole body "
                    f"in frame.<br>{warns}</div>", unsafe_allow_html=True)

    # ---- result card: reps / clean reps / grade / form score
    s = an.get("summary", {})
    avg = s.get("avg_score")
    avg_cls = "lime" if (avg or 0) >= 80 else ("amber" if (avg or 0) >= 60 else "red")
    grade = str((rp or {}).get("grade") or "—")
    cards([
        ("Reps", fmt(s.get("rep_count"), 0), f"{fmt(s.get('total_time_s'), 0, ' s')} set", "lime"),
        ("Clean reps", fmt(s.get("clean_reps"), 0), f"of {fmt(s.get('rep_count'), 0)} · pass every form check", ""),
        ("Grade", grade, "AI coach", {"A": "lime", "B": "", "C": "amber", "D": "red"}.get(grade, "")),
        ("Avg form score", fmt(avg, 0), "out of 100", avg_cls),
    ])

    left, right = st.columns([1.15, 1])
    with left:
        if res.get("overlay_bytes"):
            st.video(res["overlay_bytes"], format="video/mp4")
        elif res.get("overlay") and Path(res["overlay"]).exists():
            st.video(res["overlay"])
        elif is_upload:
            if upload_bytes is not None:
                st.video(upload_bytes)
        else:
            st.video(src["path"])
            if render_overlay and mod("overlay") is None:
                st.caption("overlay.py not ready — showing the raw clip.")
    with right:
        if rp:
            coach_card(rp)
        elif mod("coach") is None:
            placeholder("coach", "AI coach report")

    leaderboard_submit(res, src)

    # ---- everything secondary lives in expanders
    with st.expander("📈 Details — tempo, fatigue, tracking, joint angles", expanded=False):
        if an.get("warnings") and an.get("confidence_ok", True):
            st.caption("Notes: " + " · ".join(an["warnings"]))
        fat = s.get("fatigue_from_rep")
        cards([
            ("Fatigue from", f"rep {fat}" if fat else "none", "form/tempo drift vs. first reps", "amber" if fat else ""),
            ("Tempo", f"{fmt(s.get('avg_descent_s'), 1)} / {fmt(s.get('avg_ascent_s'), 1)}", "descent / ascent (s)", ""),
            ("Tracking", fmt((an.get("confidence") or 0) * 100, 0, "%"),
             f"{an.get('side', '?')} side · {res.get('pose_how', '')}", "" if an.get("confidence_ok", True) else "red"),
        ])
        st.plotly_chart(angle_chart(an), width="stretch", key="angle_chart")

    df = scorecard_df(an)
    with st.expander("📋 Per-rep scorecard", expanded=False):
        if len(df):
            st.dataframe(style_scorecard(df), width="stretch", hide_index=True)
            if rp and rp.get("per_rep"):
                st.markdown("**Coach notes per rep**")
                for pr in rp["per_rep"]:
                    st.markdown(f"**Rep {pr.get('n')}** — {pr.get('note', '')}")
        elif any("camera cut" in w for w in an.get("warnings", [])):
            items = "".join(f"<li>{_esc(str(w))}</li>" for w in an.get("warnings", []))
            st.markdown(f"<div class='fs-coach'><b style='color:{CYAN}'>Robustness report</b> — this clip is an "
                        f"edited montage, not one continuous set. Instead of inventing reps, FormSim detected the "
                        f"edits and refused to count partial reps:<ul style='margin:.4rem 0 0 0;color:#B7C3CE'>"
                        f"{items}</ul><div class='fs-src'>Tracking stayed on the athlete at "
                        f"{fmt((an.get('confidence') or 0) * 100, 0)}% confidence.</div></div>",
                        unsafe_allow_html=True)
        else:
            st.info("No reps detected. Make sure the clip shows full push-ups from the side.")

    # Worst-rep snapshots
    reps = an.get("reps", [])
    if reps:
        any_issue = any(r.get("issues") for r in reps)
        with st.expander("📸 " + ("Worst reps — frozen at the fault" if any_issue
                                   else "Rep snapshots — at the bottom position"), expanded=False):
            if mod("overlay") is None or not hasattr(mod("overlay"), "snapshot"):
                placeholder("overlay", "snapshot thumbnails")
            else:
                worst = worst_reps(an)
                cols = st.columns(len(worst))
                for col, (r, iss, t) in zip(cols, worst):
                    with col:
                        try:
                            if is_upload:
                                img = (res.get("snapshots") or {}).get(t)
                                if img is None:
                                    raise RuntimeError("not available")
                            else:
                                img = snapshot_cached(src["path"], f"{key}|{t}", t, lm, an)
                            st.image(img, width="stretch")
                        except Exception as e:  # noqa: BLE001
                            st.caption(f"snapshot failed: {e}")
                        st.caption(f"Rep {r.get('n')} · score {fmt(r.get('score'), 0)} · "
                                   + (iss[0].get("msg", iss[0].get("code")) if iss else "clean"))

    with st.expander("⬇ Downloads", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        stem = Path(src["name"]).stem
        d1.download_button("Analysis JSON", dumps(an), f"{stem}_analysis.json", "application/json",
                           width="stretch")
        d2.download_button("Landmarks JSON", json.dumps(to_jsonable(lm)), f"{stem}_landmarks.json",
                           "application/json", width="stretch")
        if res.get("overlay_bytes"):
            d3.download_button("Overlay MP4", res["overlay_bytes"], f"{stem}_formsim.mp4", "video/mp4",
                               width="stretch", on_click="ignore")
        elif res.get("overlay") and Path(res["overlay"]).exists():
            _ov = res["overlay"]
            d3.download_button("Overlay MP4", lambda: Path(_ov).read_bytes(), f"{stem}_formsim.mp4", "video/mp4",
                               width="stretch", on_click="ignore")
        else:
            d3.button("Overlay MP4", disabled=True, width="stretch")
        d4.download_button("Report (Markdown)", report_markdown(res), f"{stem}_report.md", "text/markdown",
                           width="stretch", type="primary")


def _md_safe(name: str) -> str:
    """Escape markdown control chars in a (validated) username."""
    import re as _re
    return _re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~])", r"\\\1", str(name))


def leaderboard_submit(res: dict, src: dict) -> None:
    """The 'Submit to leaderboard' step under the result card (anti-cheat gated, consent required)."""
    lbm = mod("leaderboard")
    if lbm is None:
        return
    an = res.get("analysis") or {}
    section("🏆 Leaderboard", "")
    done = st.session_state.setdefault("lb_done", {})
    sub_key = f"{src.get('hash')}|{sim_fault}"
    if sub_key in done:
        name, rank, n = done[sub_key]
        if rank:
            st.success(f"Submitted as **{_md_safe(name)}** — you're **#{rank}** of {n} on the leaderboard. "
                       "See the Leaderboard tab.")
        else:
            st.success(f"Submitted as **{_md_safe(name)}**.")
        return
    is_demo = (not src.get("upload")) or src.get("hash") in demo_hashes()
    consent = bool(st.session_state.get("lb_consent"))
    ok, why = lbm.can_submit(an, consent, is_demo=is_demo, simulated=sim_fault != "none")
    if ok:
        ok, name = lbm.validate_username(st.session_state.get("lb_user", ""))
        why = "" if ok else f"Username: {name} (enter it above)"
    if not ok:
        st.caption(f"Can't submit this one: {why}")
        return
    s = an.get("summary") or {}
    st.caption(f"Ranked by clean reps — you'd enter with **{int(s.get('clean_reps') or 0)} clean** of "
               f"{int(s.get('rep_count') or 0)} reps.")
    if st.button(f"🏆 Submit to leaderboard as {name}", type="primary", key="lb_submit"):
        try:
            lbm.submit(name, int(s.get("clean_reps") or 0), int(s.get("rep_count") or 0), s.get("avg_score"),
                       (res.get("coach") or {}).get("grade"))
            rank, n = lbm.rank_of(name)
            done[sub_key] = (name, rank, n)
            st.success(f"Submitted as **{_md_safe(name)}** — you're **#{rank}** of {n} on the leaderboard.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't save your score: {e}")


def tab_leaderboard() -> None:
    lbm = mod("leaderboard")
    if lbm is None:
        placeholder("leaderboard", "public leaderboard")
        return
    st.markdown("### 🏆 Push-up leaderboard")
    st.caption("Ranked by **clean reps** (reps that pass every form check — half reps don't count), then average "
               "form score, then whoever got there first. Best entry per username. No login needed: analyze your "
               "own clip on the Analyze tab, tick the consent box and hit Submit.")
    try:
        rows = lbm.top(20)
    except Exception as e:  # noqa: BLE001
        st.error(f"Leaderboard unavailable: {e}")
        return
    if not rows:
        st.markdown("<div class='fs-ph' style='padding:2rem'>No scores yet — be the first!</div>",
                    unsafe_allow_html=True)
    else:
        df = pd.DataFrame([{
            "Rank": r["rank"], "Username": r["username"], "Clean reps": r["clean_reps"],
            "Total reps": r["total_reps"],
            "Avg form score": None if r.get("avg_score") is None else round(float(r["avg_score"])),
            "Date": datetime.fromtimestamp(float(r["ts"])).strftime("%Y-%m-%d"),
        } for r in rows])
        st.dataframe(df, hide_index=True, width="stretch")
    st.caption("🔒 " + lbm.PRIVACY_TEXT)


def tab_physics() -> None:
    res = st.session_state.get("current")
    if mod("physics") is None:
        placeholder("physics", "joint load simulation")
        return
    if not res or not res.get("physics"):
        st.markdown("<div class='fs-ph' style='padding:2rem'>Run an analysis in the <b>Analyze</b> tab first — "
                    "the physics engine simulates forces for that clip.</div>", unsafe_allow_html=True)
        return
    ph, an, lm = res["physics"], res.get("analysis") or {}, res["landmarks"]
    s, ps = ph.get("series", {}), ph.get("summary", {})
    st.markdown(f"<div class='fs-note'>Rigid multi-segment body ({fmt(ph.get('body_mass_kg'), 0)} kg, "
                f"{fmt(ph.get('height_m'), 2)} m) reconstructed from the 2D joints, scaled at "
                f"{fmt(ph.get('px_per_m'), 0)} px/m. Static + dynamic equilibrium on two supports.</div>",
                unsafe_allow_html=True)
    if not (an.get("summary") or {}).get("rep_count") or ps.get("peak_hand_force_N") is None:
        st.markdown("<div class='fs-banner'><b>No valid push-up reps — peak loads not computed.</b> The traces below "
                    "are shown for frames that look like a plank only.</div>", unsafe_allow_html=True)
    else:
        cards([
            ("Avg hand load", fmt((ps.get("avg_hand_load_frac") or 0) * 100, 0, "%"), "of body weight (lit. 64-75%)", "lime"),
            ("Peak hand force", fmt(ps.get("peak_hand_force_N"), 0, " N"), "both hands, incl. dynamics", ""),
            ("≈ Bench press", fmt(ps.get("equivalent_bench_kg"), 0, " kg"), "static load at the bottom", "lime"),
            ("Total work", fmt(ps.get("total_work_J"), 0, " J"), f"≈ {fmt(ps.get('kcal_est'), 1)} kcal mech.", ""),
        ])
    t = arr(s, "t")
    reps = an.get("reps", [])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(line_fig(t, [("Hand load % BW", arr(s, "hand_load_frac") * 100, ACCENT)],
                                 "Load on hands (% body weight)", "%", reps=reps, fill=True),
                        width="stretch", key="ph_load")
    with c2:
        st.plotly_chart(line_fig(t, [("Elbow", np.abs(arr(s, "elbow_torque_Nm")), ACCENT),
                                     ("Shoulder", np.abs(arr(s, "shoulder_torque_Nm")), CYAN),
                                     ("Lumbar", np.abs(arr(s, "lumbar_moment_Nm")), RED)],
                                 "Joint torques", "N·m", reps=reps), width="stretch", key="ph_torque")
    st.plotly_chart(line_fig(t, [("Stress index", arr(s, "stress"), AMBER)], "Combined joint stress index (0-100)",
                             "index", height=240, reps=reps, fill=True), width="stretch", key="ph_stress")

    c3, c4 = st.columns([1.1, 1])
    with c3:
        frames = lm.get("frames", [])
        if frames:
            tmax = float(frames[-1].get("t", 0))
            # default to the deepest point of rep 1 for a dramatic first view
            default_t = float(reps[0].get("bottom", 0)) if reps and reps[0].get("bottom") is not None else 0.0
            tsel = st.slider("Scrub time (s)", 0.0, max(tmax, 0.01), min(default_t, tmax), 0.05, key=f"ph_t_{res.get('hash')}")
            fps = float(lm.get("fps") or 30)
            idx = int(min(len(frames) - 1, max(0, round(tsel * fps))))
            st.plotly_chart(force_diagram(lm, an, ph, idx, float(ph.get("body_mass_kg") or body_mass)),
                            width="stretch", key="ph_fbd")
    with c4:
        section("Per-rep peaks", "")
        prs = ph.get("reps", [])
        if prs:
            df = pd.DataFrame([{
                "Rep": r.get("n"), "Hand N": r.get("peak_hand_force_N"), "Elbow N·m": r.get("peak_elbow_torque_Nm"),
                "Shoulder N·m": r.get("peak_shoulder_torque_Nm"), "Lumbar N·m": r.get("peak_lumbar_Nm"),
                "Stress": r.get("stress_peak"), "Work J": r.get("work_J"), "Power W": r.get("power_W")} for r in prs])
            st.dataframe(df.style.format({c: "{:.0f}" for c in df.columns if c != "Rep"}, na_rep="—")
                         .background_gradient(subset=["Stress"], cmap="YlOrRd", vmin=0, vmax=100)
                         if "Stress" in df else df, width="stretch", hide_index=True)
        else:
            st.caption("No per-rep physics.")
        with st.expander("How the physics model works", expanded=False):
            st.markdown(ph.get("model_notes") or "_no notes_")


def sim_generate(n: int, seed: int, faults: str, camera: str, noise: float = 1.0):
    return call_flex(mod("synth").generate, n=n, seed=seed, faults=faults, camera=camera, noise=noise)


def tab_simlab() -> None:
    st.markdown(f"<div class='fs-coach' style='margin-bottom:1rem'><b style='color:{ACCENT}'>Why simulate?</b> "
                "Physical-AI labs need labeled motion data; we generate unlimited labeled athletes with ground "
                "truth — every rep, every fault, every camera angle known exactly.</div>", unsafe_allow_html=True)
    if mod("synth") is None:
        placeholder("synth", "synthetic athlete generator")
        return
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1.2, 1, 1, 0.8, 1.2], vertical_alignment="bottom")
    n = c1.number_input("Athletes", 1, 300, 200, 10, help="~1 s per 60 athletes to simulate. The sim-trained model "
                                                          "needs ~150+ athletes to beat the hand-written rules.")
    fault_opts = {"Mixed (realistic gym)": "mixed", "Clean form only": "none",
                  **{f"Only: {v}": k for k, v in FAULT_LABELS.items()}}
    faults_lbl = c2.selectbox("Fault mix", list(fault_opts), help="mixed ≈ 35% clean athletes, the rest get 1-2 "
                                                                   "injected faults with random magnitudes")
    faults = fault_opts[faults_lbl]
    camera = c3.selectbox("Camera", ["random", "side"], help="random = distance, height, yaw ±25°, fps, "
                                                             "resolution, handheld shake; side = canonical side view")
    noise = c4.slider("Sensor noise", 0.0, 2.0, 1.0, 0.1, help="keypoint jitter, glitches and dropouts (0 = perfect)")
    seed = c5.number_input("Seed", 0, 99999, 7, 1)
    go_btn = c6.button("⚙ Generate dataset", type="primary", width="stretch", key="gen_btn")
    if go_btn:
        with st.spinner(f"Simulating {n} athletes …"):
            t0 = time.time()
            st.session_state["sim"] = sim_generate(int(n), int(seed), faults, camera, float(noise))
            st.session_state["sim_meta"] = {"n": int(n), "seed": int(seed), "faults": faults, "camera": camera,
                                            "noise": float(noise), "secs": time.time() - t0}
            st.session_state.pop("sim_zip", None)
            st.session_state.pop("bench", None)
    samples = st.session_state.get("sim")
    if not samples:
        st.markdown("<div class='fs-ph' style='padding:1.6rem'>Pick a size and hit <b>Generate dataset</b>.</div>",
                    unsafe_allow_html=True)
        return
    meta = st.session_state.get("sim_meta", {})
    labels = [smp.get("labels", {}) for smp in samples]
    rep_counts = [lb.get("rep_count", len(lb.get("reps", []))) for lb in labels]
    fault_counts: dict[str, int] = {}
    total_reps = 0
    for lb in labels:
        for r in lb.get("reps", []):
            total_reps += 1
            for f in r.get("faults", []) or []:
                fault_counts[f] = fault_counts.get(f, 0) + 1
    total_frames = sum(len(smp.get("landmarks", {}).get("frames", [])) for smp in samples)
    clean = total_reps - sum(1 for lb in labels for r in lb.get("reps", []) if r.get("faults"))
    cards([
        ("Athletes", f"{len(samples)}", f"seed {meta.get('seed')} · {fmt(meta.get('secs'), 1)} s to simulate", "lime"),
        ("Labeled reps", f"{total_reps}", f"{clean} clean · {total_reps - clean} with faults", ""),
        ("Frames", f"{total_frames:,}", "19 keypoints each", ""),
        ("Fault labels", f"{sum(fault_counts.values())}", f"{len(fault_counts)} fault classes", "amber"),
    ])
    g1, g2, g3 = st.columns(3)
    with g1:
        fc = sorted(fault_counts.items(), key=lambda kv: -kv[1])
        fig = go.Figure(go.Bar(x=[v for _, v in fc], y=[FAULT_LABELS.get(k, k) for k, _ in fc], orientation="h",
                               marker_color=AMBER))
        fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"}, height=280, title="Fault labels")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch", key="sim_faults")
    with g2:
        fig = go.Figure(go.Histogram(x=rep_counts, marker_color=ACCENT, nbinsx=15))
        fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"}, height=280, title="Reps per athlete")
        st.plotly_chart(fig, width="stretch", key="sim_reps")
    with g3:
        params = pd.json_normalize([lb.get("params", {}) for lb in labels], sep=".")
        num_cols = [c for c in params.columns if pd.api.types.is_numeric_dtype(params[c]) and params[c].nunique() > 1]
        pref = [c for c in num_cols if any(k in c.lower() for k in ("height", "yaw", "fps", "tempo", "depth"))]
        opts = pref + [c for c in num_cols if c not in pref]
        if opts:
            pc = st.selectbox("Parameter distribution", opts, key="sim_param", label_visibility="collapsed")
            fig = go.Figure(go.Histogram(x=params[pc], marker_color=CYAN, nbinsx=20))
            fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"}, height=240, title=pc)
            st.plotly_chart(fig, width="stretch", key="sim_param_fig")
        else:
            st.caption("No numeric params in labels.")

    section("Preview a simulated athlete", "")
    p1, p2 = st.columns([1.3, 1])
    with p1:
        k = st.slider("Athlete #", 0, len(samples) - 1, 0, key=f"sim_k_{len(samples)}_{meta.get('seed')}") if len(samples) > 1 else 0
        smp = samples[k]
        fig = None
        synth = mod("synth")
        try:
            if hasattr(synth, "render_animation"):
                fig = synth.render_animation(smp)
            elif hasattr(synth, "render_preview"):
                fig = call_flex(synth.render_preview, smp, None)
        except Exception as e:  # noqa: BLE001
            st.caption(f"synth preview failed ({e}); using built-in renderer")
            fig = None
        if not isinstance(fig, go.Figure):
            fig = skeleton_animation(smp.get("landmarks", {}))
        else:
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        try:  # dark-theme the play/pause buttons and frame slider
            if fig.layout.updatemenus:
                fig.update_layout(updatemenus=[{**um.to_plotly_json(), "bgcolor": CARD, "bordercolor": "#2A3644",
                                                "showactive": False, "font": {"color": ACCENT}} for um in fig.layout.updatemenus])
            if fig.layout.sliders:
                fig.update_layout(sliders=[{**sl.to_plotly_json(), "bgcolor": "#1E2833", "bordercolor": "#2A3644",
                                            "activebgcolor": ACCENT, "font": {"color": "#C9D4DE"}}
                                           for sl in fig.layout.sliders])
        except Exception:  # noqa: BLE001
            pass
        st.plotly_chart(fig, width="stretch", key=f"sim_anim_{k}")
    with p2:
        lb = smp.get("labels", {})
        st.markdown(f"**Ground truth:** {lb.get('rep_count')} reps")
        rows = [{"Rep": r.get("n"), "Start s": r.get("start"), "End s": r.get("end"),
                 "Faults": ", ".join(r.get("faults") or []) or "✓ clean"} for r in lb.get("reps", [])]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=220)
        with st.expander("Simulation parameters"):
            st.json(to_jsonable(lb.get("params", {})), expanded=False)

    real_vs_sim_chart(smp)

    e1, e2 = st.columns(2)
    with e1:
        if st.button("📦 Build dataset export (JSONL + CSV + datasheet)", width="stretch", key="exp_btn"):
            try:
                out_dir = OUT / "datasets" / f"formsim_{meta.get('faults')}_{meta.get('camera')}_s{meta.get('seed')}_n{len(samples)}"
                with st.spinner("Writing dataset …"):
                    st.session_state["sim_zip"] = str(mod("synth").export(samples, str(out_dir)))
            except Exception as e:  # noqa: BLE001
                show_exception("export", e)
        zp = st.session_state.get("sim_zip")
        if zp and Path(zp).exists():
            st.download_button(f"⬇ Download {Path(zp).name} ({Path(zp).stat().st_size / 1e6:.1f} MB)",
                               lambda: Path(zp).read_bytes(), Path(zp).name, "application/zip", type="primary",
                               width="stretch", on_click="ignore")

    section("Benchmark", "· score our analyzer against simulated ground truth")
    bench = mod("benchmark")
    if bench is None and mod("analysis") is None:
        placeholder("benchmark", "analyzer benchmark")
        return
    if bench is None:
        st.caption("benchmark.py not ready — using the built-in quick benchmark (rep count + clip-level faults).")
    nb = st.slider("Benchmark on N simulated athletes", 5, max(5, len(samples)), min(len(samples), 60), key=f"bench_n_{len(samples)}") \
        if len(samples) > 5 else len(samples)
    b1, b2 = st.columns([1, 1])
    pre = load_precomputed_bench()
    if pre and b2.button(f"⚡ Load precomputed benchmark ({pre.get('n')} athletes, instant)", key="bench_pre_btn",
                         width="stretch"):
        st.session_state["bench"] = {"eval": pre["eval"], "s2r": pre["s2r"], "precomputed": pre.get("n")}
    if b1.button(f"🏁 Benchmark on {nb} athletes + train sim-to-real on {min(len(samples), 250)} "
                 f"(~{max(5, int(nb * 0.25 + min(len(samples), 250) * 0.27))} s)",
                 type="primary", key="bench_btn", width="stretch"):
        out: dict[str, Any] = {}
        with st.spinner("Running analyzer on simulated clips …"):
            try:
                out["eval"] = (bench.evaluate_analyzer(samples[:nb]) if bench is not None
                               else quick_benchmark(samples[:nb]))
            except Exception as e:  # noqa: BLE001
                out["eval_err"] = e
            if bench is not None and hasattr(bench, "sim_to_real"):
                try:
                    real = [json.loads(p.read_text()) for p in sorted(SAMPLES.glob("landmarks_*.json"))]
                    out["s2r"] = bench.sim_to_real(samples, real)
                except Exception as e:  # noqa: BLE001
                    out["s2r_err"] = e
        st.session_state["bench"] = out
    out = st.session_state.get("bench")
    if out:
        if out.get("precomputed"):
            st.caption(f"Showing the precomputed benchmark: {out['precomputed']} simulated athletes (seed 1, mixed "
                       f"faults, random cameras), generated by `python benchmark.py 200`.")
        if "eval_err" in out:
            show_exception("evaluate_analyzer", out["eval_err"])
        if "s2r_err" in out:
            show_exception("sim_to_real", out["s2r_err"])
        if "eval" in out and ("s2r" in out or "s2r_err" in out) and bench is not None:
            render_benchmark(out["eval"], out.get("s2r") or {})
        elif "eval" in out:
            render_metrics(out["eval"], "Analyzer vs ground truth")


@st.cache_data(show_spinner=False)
def _real_elbow_trace() -> tuple[list, list]:
    p = SAMPLES / "landmarks_demo_side.json"
    if not p.exists() or mod("analysis") is None:
        return [], []
    ser = mod("analysis").analyze(json.loads(p.read_text()))["series"]
    return ser["t"], ser["elbow"]


def real_vs_sim_chart(smp: dict) -> None:
    """Elbow-angle trace of the real demo athlete vs the selected simulated athlete (same axes)."""
    if mod("analysis") is None:
        return
    try:
        rt, re_ = _real_elbow_trace()
        ser = mod("analysis").analyze(smp["landmarks"])["series"]
        truth = (smp.get("labels", {}).get("truth") or {}).get("elbow")
        fps = float(smp["landmarks"].get("fps") or 30)
    except Exception as e:  # noqa: BLE001
        st.caption(f"real-vs-sim chart unavailable: {e}")
        return
    section("Real vs simulated", "· does simulated motion look like real motion?")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rt, y=re_, name="Real athlete (demo clip, MediaPipe)", line=dict(color=MUTED, width=2)))
    fig.add_trace(go.Scatter(x=ser["t"], y=ser["elbow"], name="Simulated athlete (as measured by analyzer)",
                             line=dict(color=ACCENT, width=2)))
    if truth:
        fig.add_trace(go.Scatter(x=[i / fps for i in range(len(truth))], y=truth, name="Simulated ground truth (3D)",
                                 line=dict(color=CYAN, width=1, dash="dot")))
    fig.add_hrect(y0=0, y1=100, fillcolor=ACCENT, opacity=0.04, line_width=0)
    fig.update_layout(**PLOT_LAYOUT, height=300, title="Elbow angle over time (deg)")
    fig.update_xaxes(title="time (s)")
    st.plotly_chart(fig, width="stretch", key="real_vs_sim")
    st.caption("Same analyzer, same units: the simulated athlete's measured trace (lime) vs the real athlete (grey). "
               "The gap between lime and dotted cyan is camera projection + tracking noise, which the simulator "
               "models on purpose.")


def render_benchmark(ev: dict, s2: dict) -> None:
    """Judge-facing benchmark view: hero cards, ML-vs-rules bars, honest caveats; raw metrics in an expander."""
    section("Benchmark results", "· simulated ground truth, zero real labels")
    ml, ru = s2.get("heldout_exact_acc_ml"), s2.get("heldout_exact_acc_rules")
    rd = (ev.get("rep_detection") or {}).get("f1")
    ri = s2.get("real_injected") or {}
    items = [("Rep-count accuracy", fmt((ev.get("rep_count_exact_acc") or 0) * 100, 0, "%"),
              f"exact count on {ev.get('n')} simulated athletes", "lime"),
             ("Rep detection F1", fmt(rd, 2), "rep boundaries vs ground truth", "lime")]
    if ml is not None and ru is not None:
        d = (ml - ru) * 100
        items.append(("Sim-trained AI vs rules", f"{ml * 100:.0f}% vs {ru * 100:.0f}%",
                      f"{'+' if d >= 0 else ''}{d:.0f} pts · exact fault set on unseen athletes",
                      "lime" if d > 0 else "amber"))
    if ri.get("reps"):
        items.append(("Real reps + injected fault", f"{ri['ml_detect_rate'] * 100:.0f}% vs {ri['rules_detect_rate'] * 100:.0f}%",
                      f"AI vs rules detection · {ri['reps']} real reps, known label", "lime"))
    elif s2.get("real_reps"):
        items.append(("Real clip", f"{fmt((s2.get('real_agreement_rate') or 0) * s2['real_reps'], 0)}/{s2['real_reps']}",
                      "reps where AI and rules agree", ""))
    cards(items)

    hp = s2.get("heldout_per_fault") or {}
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if hp:
            fs = list(hp)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Sim-trained model", x=[FAULT_LABELS.get(f, f) for f in fs],
                                 y=[hp[f]["ml_f1"] for f in fs], marker_color=ACCENT))
            fig.add_trace(go.Bar(name="Hand-written 2D rules", x=[FAULT_LABELS.get(f, f) for f in fs],
                                 y=[hp[f]["rules_f1"] for f in fs], marker_color=MUTED))
            fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"}, barmode="group", height=320,
                              title="Fault F1 on unseen simulated athletes")
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, width="stretch", key="bench_mlrules")
    with c2:
        bc = ev.get("by_camera") or {}
        pg = ev.get("projection_gap") or {}
        rows = [{"Camera": k, "Clips": v.get("clips"), "Rep count": v.get("rep_count_exact_acc"),
                 "Rules fault F1": v.get("macro_fault_f1")} for k, v in bc.items()]
        if rows:
            st.markdown("**Rule engine by camera angle**")
            st.dataframe(pd.DataFrame(rows).style.format({"Rep count": "{:.0%}", "Rules fault F1": "{:.2f}"},
                                                         na_rep="—"), width="stretch", hide_index=True)
        msg = ("Hand-written 2D rules degrade when the camera is off-side (yaw ≥ 10°)")
        if pg.get("label_disagree_rate") is not None:
            msg += (f", and even a perfect 2D tracker disagrees with the 3D truth on "
                    f"{pg['label_disagree_rate'] * 100:.0f}% of reps (elbow flare hides depth: 2D reads "
                    f"{fmt(pg.get('min_elbow_2d_minus_3d_mean_deg'), 0)}° shallower on average, even side-on)")
        st.caption(msg + ". This is exactly why we train a model on simulated data with 3D ground truth.")

    if ri.get("rows"):
        st.markdown("**Real motion, simulated fault** — clean real reps with a geometrically injected fault "
                    "(known label)")
        st.dataframe(pd.DataFrame(ri["rows"]), width="stretch", hide_index=True)
    with st.expander("Per-fault precision / recall of the rule engine"):
        st.caption("Rule engine (2D angles) vs simulator 3D ground truth, matched reps. Over-firing on SHALLOW / "
                   "HEAD_DROP mostly comes from 2D projection, not tracking noise.")
        pf = ev.get("per_fault") or {}
        if pf:
            df = pd.DataFrame(pf).T[["precision", "recall", "f1", "support"]]
            df.index = [FAULT_LABELS.get(i, i) for i in df.index]
            st.dataframe(df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}, na_rep="—"),
                         width="stretch")
    real = [r for r in (s2.get("real_predictions") or []) if r.get("rep") is not None]
    if real:
        with st.expander("Sim-trained model on the real demo clip"):
            st.dataframe(pd.DataFrame(real), width="stretch", hide_index=True)
    with st.expander("Raw metrics JSON"):
        st.json(to_jsonable({"analyzer": ev, "sim_to_real": s2}), expanded=False)


def quick_benchmark(samples: list) -> dict:
    """Fallback when benchmark.py is missing: analyzer vs simulated ground truth (clip-level)."""
    an_mod = mod("analysis")
    exact, within1, abs_err = 0, 0, []
    tp: dict[str, int] = {f: 0 for f in FAULT_LABELS}
    fp, fn = dict(tp), dict(tp)
    for smp in samples:
        truth = smp.get("labels", {})
        a = an_mod.analyze(smp["landmarks"])
        n_true = int(truth.get("rep_count", len(truth.get("reps", []))))
        n_pred = int(a.get("summary", {}).get("rep_count", len(a.get("reps", []))))
        exact += n_pred == n_true
        within1 += abs(n_pred - n_true) <= 1
        abs_err.append(abs(n_pred - n_true))
        t_f = {f for r in truth.get("reps", []) for f in (r.get("faults") or [])}
        p_f = {i.get("code") for r in a.get("reps", []) for i in (r.get("issues") or [])}
        for f in FAULT_LABELS:
            tp[f] += (f in t_f) and (f in p_f)
            fp[f] += (f not in t_f) and (f in p_f)
            fn[f] += (f in t_f) and (f not in p_f)
    n = max(len(samples), 1)
    per_fault = {}
    for f in FAULT_LABELS:
        prec = tp[f] / (tp[f] + fp[f]) if tp[f] + fp[f] else None
        rec = tp[f] / (tp[f] + fn[f]) if tp[f] + fn[f] else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        per_fault[f] = {"precision": prec, "recall": rec, "f1": f1, "support_clips": tp[f] + fn[f]}
    return {"n_clips": len(samples), "rep_count_accuracy": exact / n, "rep_count_within_1": within1 / n,
            "rep_count_mae": float(np.mean(abs_err)) if abs_err else 0.0, "per_fault": per_fault,
            "note": "built-in quick benchmark: fault metrics are clip-level (fault present anywhere in clip)"}


def render_metrics(m: Any, title: str) -> None:
    """Generic renderer for benchmark metric dicts (schema decided by benchmark.py)."""
    section(title, "")
    if not isinstance(m, dict):
        st.write(m)
        return
    scalars = {k: v for k, v in m.items() if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)}
    if scalars:
        items = []
        for k, v in list(scalars.items())[:8]:
            v = float(v)
            pct = (0 <= v <= 1) and any(w in k.lower() for w in ("acc", "prec", "rec", "f1", "rate", "frac", "auc", "within"))
            items.append((k.replace("_", " "), f"{v * 100:.0f}%" if pct else (f"{v:.0f}" if v == int(v) else f"{v:.2f}"),
                          "", "lime" if pct and v >= 0.8 else ""))
        cards(items)
    for k, v in m.items():
        if k in scalars:
            continue
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            df = pd.DataFrame(v).T
            df.index.name = k
            num = df.select_dtypes("number").columns
            sty = df.style.format({c: ("{:.0f}" if (df[c].dropna() % 1 == 0).all() and df[c].max() > 1 else "{:.2f}")
                                   for c in num}, na_rep="—")
            try:
                sty = sty.background_gradient(subset=[c for c in num if df[c].between(0, 1).all()], cmap="Greens",
                                              vmin=0, vmax=1)
            except Exception:  # noqa: BLE001
                pass
            st.markdown(f"**{k.replace('_', ' ')}**")
            st.dataframe(sty, width="stretch")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            st.markdown(f"**{k.replace('_', ' ')}**")
            st.dataframe(pd.DataFrame(v), width="stretch", hide_index=True)
        elif isinstance(v, str):
            st.caption(f"{k.replace('_', ' ')}: {v}")
        elif isinstance(v, go.Figure):
            st.plotly_chart(v, width="stretch")
    with st.expander("Raw metrics JSON"):
        st.json(to_jsonable(m), expanded=False)


def tab_about() -> None:
    pre = load_precomputed_bench()
    proof = ""
    if pre:
        try:
            proof = (f" On {pre['n']} unseen-athlete tests the sim-trained model matches the exact fault set "
                     f"{pre['s2r']['heldout_exact_acc_ml'] * 100:.0f}% of the time vs "
                     f"{pre['s2r']['heldout_exact_acc_rules'] * 100:.0f}% for hand-written rules, with "
                     f"{pre['eval']['rep_count_exact_acc'] * 100:.0f}% rep-count accuracy.")
        except Exception:  # noqa: BLE001
            proof = ""
    st.markdown(
        f"<div class='fs-coach' style='margin-bottom:1rem'>"
        f"<b style='color:{ACCENT}'>Problem.</b> Physical-AI motion models need huge amounts of <i>labeled</i> human "
        f"movement, and nobody labels every rep, fault and camera angle by hand.<br>"
        f"<b style='color:{ACCENT}'>Our simulator.</b> Forward-kinematic athletes with sampled anthropometrics, "
        f"injected form faults and fatigue, randomised cameras and realistic keypoint noise — every label known "
        f"exactly.<br><b style='color:{ACCENT}'>Proof.</b> We benchmark a real-video analyzer against simulated "
        f"ground truth and train a fault model on simulation only, then run it on real video.{proof}</div>",
        unsafe_allow_html=True)
    _hs = _hero_stats()
    if _hs:
        st.markdown(_hs, unsafe_allow_html=True)
    st.markdown("### How it works")
    dot = f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; pad=0.2;
  node [shape=box, style="rounded,filled", fillcolor="#131A23", color="#2A3644", fontcolor="#E6EDF3",
        fontname="Helvetica", fontsize=11, margin="0.18,0.10"];
  edge [color="#5B6B7A", arrowsize=0.7];
  video [label="📱 Side-view video", color="{ACCENT}"];
  pose [label="Pose tracking\\nMediaPipe · 19 joints"];
  an [label="Biomechanics\\nreps · faults · tempo · fatigue"];
  ph [label="Physics sim\\nsegment masses · COM\\nsupport forces · torques"];
  coach [label="AI coach\\nClaude / rules"];
  ov [label="Annotated video\\n+ report", color="{ACCENT}"];
  synth [label="Synthetic athletes\\nFK + camera + noise", color="{CYAN}"];
  ds [label="Labeled dataset\\nJSONL · CSV · datasheet", color="{CYAN}"];
  bench [label="Benchmark\\n& sim-to-real", color="{CYAN}"];
  video -> pose -> an -> ph -> coach -> ov;
  an -> ov;
  synth -> ds;
  synth -> bench [label=" ground truth", fontcolor="#8B98A5", fontsize=9];
  an -> bench [style=dashed, label=" scored", fontcolor="#8B98A5", fontsize=9];
}}"""
    st.graphviz_chart(dot, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Why not just ask a video LLM?")
        note = None
        if mod("coach") and hasattr(mod("coach"), "llm_comparison_note"):
            try:
                note = mod("coach").llm_comparison_note()
            except Exception:  # noqa: BLE001
                note = None
        st.markdown(note or (
            "- **Measured, not guessed** — every number (angles, tempo, newtons) comes from tracked joints and a "
            "physics model; the LLM only explains it, so it can't hallucinate a rep count.\n"
            "- **Frame-accurate** — 30 fps per-joint analysis catches a 0.3 s hip sag a sampled video LLM misses.\n"
            "- **Physics a language model can't see** — hand force, joint torques and load share need body mass and "
            "segment dynamics.\n"
            "- **Verifiable** — we benchmark the analyzer on simulated athletes with known ground truth.\n"
            "- **Cheap & private** — runs on a laptop CPU, offline; Claude is optional."))
    with c2:
        st.markdown("#### Tech stack")
        st.markdown(
            "- **Vision:** MediaPipe Pose Landmarker (Tasks API), OpenCV, ffmpeg H.264\n"
            "- **Biomechanics:** hysteresis rep state machine, joint-angle rules, fatigue drift detection\n"
            "- **Physics:** de Leva segment parameters, 2-support static + dynamic equilibrium, joint moments\n"
            "- **Coach:** Claude (Anthropic SDK) grounded in measured numbers, offline rules fallback\n"
            "- **Simulation:** procedural 3D forward kinematics, fault injection, fatigue, camera randomisation, "
            "keypoint noise & dropouts\n"
            "- **ML:** scikit-learn sim-to-real classifier · **UI:** Streamlit + Plotly")
        st.markdown("#### Team")
        st.markdown("Built at **NullHacks Helsinki** — FormSim team.")
    with st.expander("Module status"):
        for n, m in MODS.items():
            st.markdown(f"- `{n}.py` — " + ("ready" if m else f"not ready ({MOD_ERR.get(n)})"))


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def _safe(fn: Callable, label: str) -> None:
    try:
        fn()
    except Exception as _e:  # noqa: BLE001
        show_exception(label, _e)


def tab_under_the_hood() -> None:
    st.caption("The engine behind the coach: joint-load physics, our synthetic-athlete simulator and how it all fits.")
    sub = st.tabs(["⚛ Physics", "🧪 Sim Lab", "ⓘ How it works"])
    for _t, _fn, _label in zip(sub, (tab_physics, tab_simlab, tab_about), ("Physics", "Sim Lab", "How it works")):
        with _t:
            _safe(_fn, _label)


tabs = st.tabs(["◆ Analyze", "🏆 Leaderboard", "⚙ Under the hood"])
for _tab, _fn, _label in zip(tabs, (tab_analyze, tab_leaderboard, tab_under_the_hood),
                             ("Analyze", "Leaderboard", "Under the hood")):
    with _tab:
        _safe(_fn, _label)
