"""Form Coach: Streamlit UI.

    streamlit run app.py

Upload a side-view push-up set (or try the demo). Every stage falls back to samples/ if it
isn't available, and the page says which numbers are mocked.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import coach  # noqa: E402
import pose  # noqa: E402
import video  # noqa: E402

try:
    import analysis as rules  # noqa: E402
except ImportError:  # not written yet -> run_pipeline falls back to samples/sample_analysis.json
    class rules:  # noqa: N801
        @staticmethod
        def analyze(*_args, **_kwargs):
            raise NotImplementedError("analysis.py not written yet")

try:
    import imu  # noqa: E402
except ImportError:
    imu = None

ROOT = Path(__file__).resolve().parent
SAMPLE_LANDMARKS = ROOT / "samples" / "sample_landmarks.json"
SAMPLE_ANALYSIS = ROOT / "samples" / "sample_analysis.json"
SEVERITY_RANK = {"high": 3, "med": 2, "low": 1}
ISSUE_NAMES = {code: v[0] for code, v in coach.CUES.items()}
ISSUE_NAMES.setdefault("HIP_PIKE", "Hips piking up")

# dark theme (matches .streamlit/config.toml and the README)
INK, MUTED, LINE = "#E8EDF2", "#8C99A6", "#27333F"
CARD, PANEL, BG = "#151E27", "#1B2631", "#0E151C"
GREEN, RED, AMBER, BLUE = "#3DBE74", "#F0525A", "#F2B33D", "#7FB2E5"

st.set_page_config(page_title="Form Coach", page_icon="🟢", layout="wide")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display:none; }}
[data-testid="stFileUploaderDropzone"] {{ background:{CARD}; border:1.5px dashed #3A4856; border-radius:14px; }}
[data-testid="stExpander"] details {{ background:{CARD}; border-radius:12px; border-color:{LINE}; }}
[data-testid="stStatusWidget"], [data-testid="stExpander"] summary {{ color:{INK}; }}
.stApp {{ background: radial-gradient(1200px 500px at 15% -10%, #16222E 0%, {BG} 60%); }}
.stTabs [data-baseweb="tab"] p {{ font-weight:500; }}
.block-container {{ max-width: 1240px; padding-top: 1.6rem; padding-bottom: 4rem; }}
h1, h2, h3, h4 {{ font-family: 'Archivo', 'IBM Plex Sans', sans-serif; letter-spacing: -0.01em; color: {INK}; }}

.fc-brand {{ display:flex; align-items:baseline; gap:.75rem; margin-bottom:.15rem; }}
.fc-mark {{ font-family:'Archivo',sans-serif; font-weight:800; font-size:2.1rem; letter-spacing:-0.03em; color:{INK}; }}
.fc-dot {{ width:.7rem; height:.7rem; border-radius:50%; background:{GREEN}; display:inline-block;
           box-shadow: 1.1rem 0 0 {RED}; margin-right:1.3rem; transform: translateY(-.15rem); }}
.fc-tag {{ color:{MUTED}; font-size:1rem; }}
.fc-eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.08em;
               text-transform:uppercase; color:{MUTED}; margin: 1.4rem 0 .45rem; }}

[data-testid="stMetric"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:14px 18px; }}
[data-testid="stMetricLabel"] p {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.06em;
                                   text-transform:uppercase; color:{MUTED}; }}
[data-testid="stMetricValue"] {{ font-family:'Archivo',sans-serif; font-weight:800; color:{INK};
                                 font-variant-numeric: tabular-nums; }}

div[class*="st-key-rep_"], div[class*="st-key-rep_"] > div {{ width:100%; }}
div[class*="st-key-rep_"] button {{
  width:100%; min-height:58px; border-radius:10px; border:1px solid {LINE}; background:{CARD};
  box-shadow: inset 0 4px 0 {GREEN}; color:{INK}; padding:.5rem .25rem .3rem;
}}
div[class*="st-key-rep_"] button p {{ font-family:'IBM Plex Mono',monospace; font-size:.8rem; line-height:1.3;
  white-space:pre-line; margin:0; }}
div[class*="st-key-rep_bad_"] button {{ box-shadow: inset 0 4px 0 {RED}; background:#2A1C22; }}
div[class*="st-key-rep_"] button:hover {{ border-color:{INK}; color:{INK}; }}
div[class*="st-key-rep_"] button:focus-visible {{ outline:3px solid {BLUE}; outline-offset:2px; }}

.fc-card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:18px 20px; }}
.fc-headline {{ font-family:'Archivo',sans-serif; font-weight:800; font-size:1.35rem; color:{INK}; margin:0 0 .35rem; }}
.fc-fix {{ border-left:4px solid {RED}; padding:.55rem .8rem; margin:.7rem 0; background:{PANEL}; border-radius:0 10px 10px 0; }}
.fc-fix b {{ color:{INK}; }}
.fc-cue {{ display:inline-block; margin-top:.35rem; font-family:'IBM Plex Mono',monospace; font-size:.82rem;
           background:#12301F; color:#8BE3B0; padding:.18rem .5rem; border-radius:6px; }}
.fc-small {{ color:{MUTED}; font-size:.85rem; }}
.fc-steps {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:1.2rem; }}
.fc-step {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; }}
.fc-step-n {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:{MUTED}; letter-spacing:.06em; }}
.fc-step-t {{ font-family:'Archivo',sans-serif; font-weight:800; font-size:1.05rem; color:{INK}; margin:.2rem 0 .25rem; }}
@media (max-width: 760px) {{ .fc-steps {{ grid-template-columns:1fr; }} }}
.fc-pill {{ display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.72rem; padding:.12rem .5rem;
            border-radius:99px; border:1px solid {LINE}; color:{MUTED}; margin-right:.35rem; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- pipeline

def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _worst_frames(path: str, analysis: dict, k: int = 2) -> list[bytes]:
    """JPEGs of the k worst issues, sent to the LLM as a visual double-check."""
    issues = [i for r in analysis.get("reps", []) for i in r.get("issues", [])]
    issues.sort(key=lambda i: (SEVERITY_RANK.get(i.get("severity"), 0), abs(i.get("value", 0))), reverse=True)
    frames = [video.grab_frame(path, i["t"]) for i in issues[:k]]
    return [f for f in frames if f]


def run_pipeline(video_path: str | None, strict: bool, fast: bool = False) -> dict:
    """Demo mode uses samples/. A real upload never falls back to sample numbers: if tracking or
    analysis fails, the result carries an error state instead."""
    mocked, errors = [], []
    holder = st.empty()

    def stage(name, fn, fallback):
        try:
            return fn()
        except NotImplementedError:
            mocked.append(name)
        except Exception:
            mocked.append(name)
            errors.append((name, traceback.format_exc()))
        return fallback()

    with holder.status("Analysing your set…", expanded=True) as status:
        st.write("Tracking your joints…")
        if video_path:
            bar = st.progress(0.0, text="Tracking your joints, frame by frame")
            try:
                landmarks = pose.extract(video_path, frame_step=2 if fast else 1,
                                         progress=lambda f: bar.progress(
                                             min(f, 1.0), text=f"Tracking your joints, frame by frame ({f:.0%})"))
            except Exception:
                holder.empty()
                return {"error": "unreadable", "detail": traceback.format_exc()}
            bar.empty()
        else:  # demo mode
            landmarks = _load(SAMPLE_LANDMARKS)
            mocked.append("demo clip")

        st.write("Measuring every rep…")
        if video_path:
            try:
                analysis = rules.analyze(landmarks, strict=strict)
            except Exception:
                holder.empty()
                return {"error": "analysis", "detail": traceback.format_exc()}
        else:
            analysis = stage("analysis", lambda: rules.analyze(landmarks, strict=strict),
                             lambda: _load(SAMPLE_ANALYSIS))

        overlay = None
        if video_path:
            st.write("Drawing the skeleton…")
            out = str(Path(video_path).with_name(Path(video_path).stem + "_overlay.mp4"))
            src = stage("overlay", lambda: pose.render_overlay(video_path, landmarks, analysis, out),
                        lambda: video_path)
            overlay = video.to_browser_mp4(src)

        wearable = None
        if imu is not None:
            try:
                wearable = imu.simulate(landmarks)
            except Exception:
                errors.append(("wearable", traceback.format_exc()))

        st.write("Writing your coaching notes…")
        frames = _worst_frames(overlay, analysis) if overlay and "overlay" not in mocked else None
        fb = coach.feedback(analysis, frames)
        status.update(label="Analysis complete", state="complete", expanded=False)
    holder.empty()

    return {"analysis": analysis, "feedback": fb, "video": overlay, "mocked": mocked,
            "errors": errors, "imu": wearable}


# ---------------------------------------------------------------- charts

def _rep_bands(a: dict) -> pd.DataFrame:
    rows = [{"start": r["start"], "end": r["end"], "rep": r["n"],
             "issues": ", ".join(ISSUE_NAMES.get(i["code"], i["code"]) for i in r["issues"])}
            for r in a["reps"] if r["issues"]]
    return pd.DataFrame(rows or [{"start": 0, "end": 0, "rep": 0, "issues": ""}])


MONO_STACK = "IBM Plex Mono, ui-monospace, Menlo, Consolas, monospace"
SANS_STACK = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"


def _axis():
    return alt.Axis(labelFont=MONO_STACK, titleFont=SANS_STACK, labelColor=MUTED, titleColor=MUTED,
                    titleFontWeight=500, gridColor="#1F2A35", domainColor=LINE, tickColor=LINE)


def _set_window(a: dict) -> list[float]:
    """x-range of the set itself (first rep start -1 s .. last rep end +1 s)."""
    t = [v for v in a["series"]["t"] if v is not None]
    if a["reps"]:
        return [max(0.0, a["reps"][0]["start"] - 1), a["reps"][-1]["end"] + 1]
    return [min(t), max(t)] if t else [0, 1]


def _series_df(a: dict) -> pd.DataFrame:
    lo, hi = _set_window(a)
    df = pd.DataFrame(a["series"])
    return df[(df["t"] >= lo) & (df["t"] <= hi)]


def elbow_chart(a: dict) -> alt.LayerChart:
    th = a["thresholds"]
    s = _series_df(a)
    xs = alt.Scale(domain=_set_window(a), nice=False)
    rects = alt.Chart(_rep_bands(a)).mark_rect(opacity=0.18, color=RED).encode(
        x=alt.X("start:Q", scale=xs), x2="end:Q", tooltip=["rep", "issues"])
    line = alt.Chart(s).mark_line(color=GREEN, strokeWidth=2, clip=True).encode(
        x=alt.X("t:Q", title="Time (s)", axis=_axis(), scale=xs),
        y=alt.Y("elbow:Q", title="Elbow angle (°)", scale=alt.Scale(domain=[30, 185]), axis=_axis()))
    refs = pd.DataFrame({"y": [th["depth_max_elbow"], th["lockout_min_elbow"]],
                         "label": ["Depth target", "Lockout target"]})
    rule = alt.Chart(refs).mark_rule(strokeDash=[5, 4], color="#5B6875").encode(y="y:Q", tooltip=["label", "y"])
    return (rects + line + rule).properties(height=210).configure_view(stroke=None).configure(background="transparent")


def hip_chart(a: dict) -> alt.LayerChart:
    tol = a["thresholds"]["hip_tolerance"]
    s = _series_df(a)
    xs = alt.Scale(domain=_set_window(a), nice=False)
    ys = alt.Scale(domain=[-35, 35], nice=False)
    line = alt.Chart(s).mark_line(color=BLUE, strokeWidth=2, clip=True).encode(
        x=alt.X("t:Q", title="Time (s)", axis=_axis(), scale=xs),
        y=alt.Y("hip_dev:Q", title="Hips (°)", scale=ys,
                axis=alt.Axis(values=[-30, -15, 0, 15, 30], labelFont=MONO_STACK, titleFont=SANS_STACK,
                              labelColor=MUTED, titleColor=MUTED, gridColor="#1F2A35", domainColor=LINE,
                              tickColor=LINE)))
    refs = pd.DataFrame({"y": [tol, -tol], "label": [f"Sag limit +{tol}°", f"Pike limit -{tol}°"]})
    rule = alt.Chart(refs).mark_rule(strokeDash=[5, 4], color="#5B6875").encode(
        y=alt.Y("y:Q", scale=ys), tooltip=["label"])
    return (line + rule).properties(height=190).configure_view(stroke=None).configure(background="transparent")


def imu_chart(w: dict) -> alt.Chart:
    rows = []
    for name, label in (("chest", "Chest strap"), ("wrist", "Smartwatch")):
        s = w["sensors"].get(name)
        if not s:
            continue
        for t, acc in zip(s["t"][::2], s["acc_world"][::2]):
            rows.append({"t": t, "acc": (acc[0] ** 2 + acc[1] ** 2) ** 0.5, "sensor": label})
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_line(strokeWidth=1.6).encode(
        x=alt.X("t:Q", title="Time (s)", axis=_axis()),
        y=alt.Y("acc:Q", title="Acceleration (m/s², 9.81 = still)", scale=alt.Scale(zero=False), axis=_axis()),
        color=alt.Color("sensor:N", scale=alt.Scale(range=[AMBER, BLUE]),
                        legend=alt.Legend(orient="top", title=None, labelFont=MONO_STACK)),
    ).properties(height=220).configure_view(stroke=None).configure(background="transparent")


# ---------------------------------------------------------------- UI pieces

def _jump_to(t: float):
    st.session_state.jump = max(float(t) - 0.5, 0.0)


def _on_rep_select():
    rows = st.session_state["rep_table"]["selection"]["rows"]
    if rows:
        _jump_to(st.session_state.rep_starts[rows[0]])


def header():
    st.markdown(
        '<div class="fc-brand"><span class="fc-dot"></span><span class="fc-mark">Form Coach</span></div>'
        '<div class="fc-tag">Film a push-up set from the side. Every rep gets measured, then you get the '
        'two fixes that matter most.</div>', unsafe_allow_html=True)


def rep_ribbon(reps: list[dict]):
    st.markdown('<div class="fc-eyebrow">Every rep · tap one to jump the video there</div>',
                unsafe_allow_html=True)
    per_row = 8 if len(reps) > 8 else max(len(reps), 1)
    for start in range(0, len(reps), per_row):
        chunk = reps[start:start + per_row]
        cols = st.columns(per_row, gap="small")
        for col, r in zip(cols, chunk):
            codes = [i["code"] for i in r["issues"]]
            short = {"SHALLOW": "depth", "NO_LOCKOUT": "lockout", "HIP_SAG": "sag", "HIP_PIKE": "pike",
                     "HEAD_DROP": "head", "FAST_DESCENT": "fast"}
            label = f"{r['n']}\n" + (short.get(codes[0], codes[0].lower()) if codes else "✓")
            key = f"rep_{'bad_' if codes else ''}{r['n']}"
            col.button(label, key=key, on_click=_jump_to, args=(r["start"],),
                       help=(" · ".join(i.get("message", i["code"]) for i in r["issues"]) or "Clean rep")
                       + f" — {_mmss(r['start'])}")


def coach_card(fb: dict):
    who = "Claude" if fb.get("source") == "llm" else "rule-based coach"
    fixes = "".join(
        f'<div class="fc-fix"><b>{n}. {p.get("issue", "")}</b><br>'
        f'<span class="fc-small">{p.get("evidence", "")}</span>'
        + (f'<br><span class="fc-small">{p["why"]}</span>' if p.get("why") else "")
        + (f'<br><span class="fc-cue">{p["cue"]}</span>' if p.get("cue") else "") + "</div>"
        for n, p in enumerate(fb.get("priorities", []), 1))
    good = ""
    if fb.get("positives"):
        good = '<div class="fc-small" style="margin-top:.6rem"><b style="color:#8BE3B0">What went well</b> · ' \
               + " · ".join(fb["positives"]) + "</div>"
    st.markdown(
        f'<div class="fc-card"><div class="fc-eyebrow" style="margin-top:0">Coach · {who}</div>'
        f'<div class="fc-headline">{fb.get("headline", "Your feedback")}</div>'
        f'<div>{fb.get("summary", "")}</div>{fixes}{good}</div>', unsafe_allow_html=True)
    if str(fb.get("frame_check", "")).startswith("suspect"):
        st.warning(f"Tracking check: {fb['frame_check']}")
    if fb.get("llm_error"):
        st.caption(f"LLM unavailable, used the rule-based coach ({fb['llm_error']})")


FILM_TIPS = ("Film from the side with the phone in landscape on the floor, 2–3 m away, so your whole "
             "body from head to heels stays in frame. Start filming before the first rep.")


def state_card(title: str, body: str, tone: str = "neutral"):
    edge = {"error": RED, "warn": AMBER, "neutral": BLUE}[tone]
    st.markdown(f'<div class="fc-card" role="status" style="box-shadow: inset 4px 0 0 {edge}">'
                f'<div class="fc-headline">{title}</div><div class="fc-small" style="font-size:.95rem">'
                f'{body}</div></div>', unsafe_allow_html=True)


def render_error(res: dict):
    if res["error"] == "unreadable":
        state_card("We couldn't read this video",
                   "The file didn't decode as a video we can track. Export it as MP4 (H.264) or MOV from your "
                   "phone's gallery and upload it again.", "error")
    else:
        state_card("We couldn't analyse this set",
                   "Tracking worked but measuring the reps failed. Try the upload again; if it keeps failing, "
                   "trim the clip to just the set.", "error")
    with st.expander("Technical details"):
        st.code(res.get("detail", ""))


def render_no_reps(a: dict):
    act = a.get("activity", {})
    reason = act.get("reason", "")
    if "no person" in reason:
        title, body = "No one in frame", "We couldn't find a person in this clip. " + FILM_TIPS
    elif "upright" in reason or "hands are not" in reason:
        title, body = ("This doesn't look like a push-up", "We saw someone, but never in a push-up position "
                       "(body horizontal, legs straight, hands on the floor). " + FILM_TIPS)
    elif "plank" in reason:
        title, body = ("Plank hold, no reps", "You held the position well, but your elbows never bent and "
                       "straightened. Upload a set with at least one full rep.")
    else:
        title, body = "No complete reps found", "We couldn't find a full rep in this clip. " + FILM_TIPS
    state_card(title, body, "warn")


def render(res: dict):
    if res.get("error"):
        render_error(res)
        return
    a, fb = res["analysis"], res["feedback"]
    reps, summ = a["reps"], a["summary"]

    if res["mocked"]:
        st.info("You're viewing the demo set. Upload your own side-view clip to analyse it.", icon="ℹ️")
    if not reps:
        render_no_reps(a)
        return
    for name, tb in res["errors"]:
        with st.expander(f"{name} failed and fell back to sample data"):
            st.code(tb)
    for w in a.get("warnings", []):
        if not w.startswith("Not counted as a push-up set"):
            st.warning(w)
    if a.get("confidence", 1) < 0.6:
        st.error("We can't see you clearly enough to judge form. Film from the side, landscape, with your "
                 "whole body from head to heels in frame, then upload again.")

    total, clean, fat = summ["rep_count"], summ["clean_reps"], summ.get("fatigue_from_rep")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reps", total)
    c2.metric("Clean reps", f"{clean}/{total}")
    c3.metric("Form score", f"{round(100 * clean / total) if total else 0}%")
    c4.metric("Fatigue", f"From rep {fat}" if fat else ("None" if total else "—"))
    st.markdown(f'<span class="fc-pill">{a.get("standard", "")}</span>'
                f'<span class="fc-pill">tracking {a.get("confidence", 0):.0%}</span>'
                f'<span class="fc-pill">{a.get("side", "")} side to camera</span>', unsafe_allow_html=True)

    if reps:
        rep_ribbon(reps)

    st.markdown('<div class="fc-eyebrow">Your set</div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")
    with left:
        if res["video"]:
            st.video(res["video"], start_time=int(st.session_state.get("jump", 0)), autoplay=True, muted=True)
        else:
            st.markdown('<div class="fc-card fc-small">Demo mode has no video. Upload a clip to see your '
                        'skeleton overlay here, with the faulty joints turning red as they happen.</div>',
                        unsafe_allow_html=True)
    with right:
        coach_card(fb)

    tab_charts, tab_reps, tab_wear = st.tabs(["Angles", "Rep by rep", "Wearable (simulated)"])
    with tab_charts:
        st.altair_chart(elbow_chart(a), width="stretch")
        st.altair_chart(hip_chart(a), width="stretch")
        st.caption("Hips are measured against your own line at the top of each rep, so loose clothing "
                   "doesn't read as sagging.")
    with tab_reps:
        st.session_state.rep_starts = [r["start"] for r in reps]
        table = pd.DataFrame([{
            "Rep": r["n"], "Time": _mmss(r["start"]),
            "Bottom elbow °": r["min_elbow"], "Top elbow °": r["max_elbow"],
            "Hips off line °": r["max_hip_dev"], "": "✅" if not r["issues"] else "⚠️",
            "Issues": " · ".join(i.get("message", i["code"]) for i in r["issues"]),
            "Coach note": fb.get("rep_notes", {}).get(str(r["n"]), ""),
        } for r in reps])
        st.caption("Click a row to jump the video to that rep.")
        st.dataframe(table, hide_index=True, key="rep_table", on_select=_on_rep_select,
                     selection_mode="single-row", width="stretch")
    with tab_wear:
        if res.get("imu") and res["imu"].get("sensors"):
            st.altair_chart(imu_chart(res["imu"]), width="stretch")
            st.caption("What a chest strap and a smartwatch would have recorded during this set, simulated "
                       "from the video: every clip doubles as wearable training data.")
        else:
            st.caption("No wearable simulation for this clip.")

    if st.session_state.get("debug"):
        with st.expander("Raw analysis JSON"):
            st.json({k: v for k, v in a.items() if k != "series"})
        with st.expander("Raw feedback JSON"):
            st.json(fb)


# ---------------------------------------------------------------- page

header()

st.markdown('<div class="fc-eyebrow">Analyse a set</div>', unsafe_allow_html=True)
opt_col, gear_col = st.columns([5, 1], vertical_alignment="center")
with opt_col:
    standard = st.segmented_control("Judge against", ["Test standard (90°)", "Strict (chest near floor)"],
                                    default="Test standard (90°)", label_visibility="collapsed")
    strict = (standard or "").startswith("Strict")
with gear_col:
    with st.popover("Settings", width="stretch"):
        st.toggle("Fast tracking (2× faster)", key="fast",
                  help="Runs the pose model on every other frame and fills the gaps. Rep counts stay the "
                       "same; borderline form calls (e.g. lockout within a few degrees) can differ.")
        st.toggle("Show raw JSON", key="debug")
        st.caption(f"Coach: Claude (`{coach.MODEL}`)" if os.getenv("ANTHROPIC_API_KEY")
                   else "Coach: rule-based. Add ANTHROPIC_API_KEY to .env to coach with Claude.")
up_col, demo_col = st.columns([5, 1], vertical_alignment="bottom")
with up_col:
    upload = st.file_uploader("Side-view video of one push-up set (mp4, mov, under ~60 s)",
                              type=["mp4", "mov", "m4v", "avi", "webm"])
with demo_col:
    if st.button("Try the demo", width="stretch", type="secondary"):
        st.session_state.demo = True

with st.expander("How to film for the best results"):
    st.markdown(
        "- **Side view**, phone in **landscape**, about 2–3 m away at hip height\n"
        "- Whole body in frame, **head to heels**, the whole time\n"
        "- One person in shot; loose clothes are fine\n"
        "- Start filming before the first rep; getting up afterwards is ignored automatically")

key, video_path = None, None
if upload is not None:
    st.session_state.demo = False
    video_path, digest = video.save_upload(upload.getvalue(), Path(upload.name).suffix.lower() or ".mp4")
    key = (digest, strict, st.session_state.get("fast", False))
elif st.session_state.get("demo"):
    key = ("demo", strict)

if key and st.session_state.get("result_key") != key:
    st.session_state.jump = 0.0
    st.session_state.result = run_pipeline(video_path, strict, st.session_state.get("fast", False))
    st.session_state.result_key = key

if key and st.session_state.get("result"):
    render(st.session_state.result)
else:
    st.markdown(
        '<div class="fc-steps">'
        '<div class="fc-step"><div class="fc-step-n">STEP 1</div><div class="fc-step-t">Film from the side</div>'
        '<div class="fc-small">Phone in landscape on the floor, 2–3 m away, head to heels in frame. '
        'Baggy clothes are fine.</div></div>'
        '<div class="fc-step"><div class="fc-step-n">STEP 2</div><div class="fc-step-t">Upload the set</div>'
        '<div class="fc-small">We track 19 joints in every frame and find each rep, ignoring getting into '
        'position and standing up.</div></div>'
        '<div class="fc-step"><div class="fc-step-n">STEP 3</div><div class="fc-step-t">Fix what matters</div>'
        '<div class="fc-small">See every rep scored, watch your skeleton turn red where form breaks, and get '
        'the two cues worth fixing first.</div></div></div>', unsafe_allow_html=True)
