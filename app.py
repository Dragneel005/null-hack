"""Form Coach: Streamlit UI. OWNER: Person 3.

    streamlit run app.py

Works from minute one: any pipeline stage that isn't implemented yet (or crashes) falls
back to data/sample_*.json, and a yellow banner says which stages are mocked.
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

from formcoach import coach, pose, rules, video  # noqa: E402

ROOT = Path(__file__).resolve().parent
SAMPLE_LANDMARKS = ROOT / "data" / "sample_landmarks.json"
SAMPLE_ANALYSIS = ROOT / "data" / "sample_analysis.json"
SEVERITY_RANK = {"high": 3, "med": 2, "low": 1}
ISSUE_NAMES = {code: v[0] for code, v in coach.CUES.items()}

st.set_page_config(page_title="Form Coach", page_icon="💪", layout="wide")


# ---------------------------------------------------------------- pipeline

def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _worst_frames(path: str, analysis: dict, k: int = 2) -> list[bytes]:
    """JPEGs of the k worst issues, to send to the LLM for a visual double-check."""
    issues = [i for r in analysis.get("reps", []) for i in r.get("issues", [])]
    issues.sort(key=lambda i: (SEVERITY_RANK.get(i.get("severity"), 0), abs(i.get("value", 0))), reverse=True)
    frames = [video.grab_frame(path, i["t"]) for i in issues[:k]]
    return [f for f in frames if f]


def run_pipeline(video_path: str | None, strict: bool) -> dict:
    mocked, errors = [], []

    def stage(name, fn, fallback):
        try:
            return fn()
        except NotImplementedError:
            mocked.append(name)
        except Exception:
            mocked.append(name)
            errors.append((name, traceback.format_exc()))
        return fallback()

    with st.status("Analysing your set…", expanded=True) as status:
        st.write("Tracking your joints…")
        if video_path:
            landmarks = stage("pose", lambda: pose.extract(video_path), lambda: _load(SAMPLE_LANDMARKS))
        else:  # demo mode
            landmarks = _load(SAMPLE_LANDMARKS)
            mocked.append("pose")

        st.write("Measuring every rep…")
        analysis = stage("rules", lambda: rules.analyze(landmarks, strict=strict), lambda: _load(SAMPLE_ANALYSIS))

        overlay = None
        if video_path:
            st.write("Drawing the skeleton…")
            out = str(Path(video_path).with_name(Path(video_path).stem + "_overlay.mp4"))
            src = stage("overlay", lambda: pose.render_overlay(video_path, landmarks, analysis, out), lambda: video_path)
            overlay = video.to_browser_mp4(src)

        st.write("Writing your coaching notes…")
        frames = _worst_frames(overlay, analysis) if overlay and "overlay" not in mocked else None
        fb = coach.feedback(analysis, frames)
        status.update(label="Analysis complete", state="complete", expanded=False)

    return {"analysis": analysis, "feedback": fb, "video": overlay, "mocked": mocked, "errors": errors}


# ---------------------------------------------------------------- UI pieces

def _jump_to(t: float):
    st.session_state.jump = max(float(t) - 0.5, 0.0)


def _on_rep_select():
    rows = st.session_state["rep_table"]["selection"]["rows"]
    if rows:
        _jump_to(st.session_state.rep_starts[rows[0]])


def elbow_chart(a: dict) -> alt.LayerChart:
    th = a["thresholds"]
    s = pd.DataFrame(a["series"])
    bad = pd.DataFrame([
        {"start": r["start"], "end": r["end"], "rep": r["n"],
         "issues": ", ".join(ISSUE_NAMES.get(i["code"], i["code"]) for i in r["issues"])}
        for r in a["reps"] if r["issues"]
    ] or [{"start": 0, "end": 0, "rep": 0, "issues": ""}])
    lines = pd.DataFrame({"y": [th["depth_max_elbow"], th["lockout_min_elbow"]],
                          "label": ["Depth target", "Lockout target"]})
    rects = alt.Chart(bad).mark_rect(opacity=0.15, color="#e4572e").encode(
        x="start:Q", x2="end:Q", tooltip=["rep", "issues"])
    line = alt.Chart(s).mark_line(color="#2b7bba").encode(
        x=alt.X("t:Q", title="Time (s)"),
        y=alt.Y("elbow:Q", title="Elbow angle (°)", scale=alt.Scale(domain=[40, 185])))
    rule = alt.Chart(lines).mark_rule(strokeDash=[5, 4], color="#555").encode(y="y:Q", tooltip=["label", "y"])
    return (rects + line + rule).properties(height=220)


def hip_chart(a: dict) -> alt.LayerChart:
    tol = a["thresholds"]["hip_tolerance"]
    s = pd.DataFrame(a["series"])
    band = alt.Chart(pd.DataFrame({"lo": [-tol], "hi": [tol]})).mark_rect(opacity=0.12, color="#3cb371").encode(
        y="lo:Q", y2="hi:Q")
    line = alt.Chart(s).mark_line(color="#8e44ad").encode(
        x=alt.X("t:Q", title="Time (s)"),
        y=alt.Y("hip_dev:Q", title="Hips off line (°, + = sag)", scale=alt.Scale(domain=[-30, 30])))
    return (band + line).properties(height=180)


def coach_card(fb: dict):
    who = "Claude" if fb.get("source") == "llm" else "rule-based coach"
    st.subheader(fb.get("headline", "Your feedback"))
    st.write(fb.get("summary", ""))
    for n, p in enumerate(fb.get("priorities", []), 1):
        with st.container(border=True):
            st.markdown(f"**{n}. {p.get('issue', '')}**  \n{p.get('evidence', '')}")
            if p.get("why"):
                st.caption(p["why"])
            if p.get("cue"):
                st.success(f"Cue: {p['cue']}")
    if fb.get("positives"):
        st.markdown("**What went well:** " + " · ".join(fb["positives"]))
    if str(fb.get("frame_check", "")).startswith("suspect"):
        st.warning(f"Tracking check: {fb['frame_check']}")
    st.caption(f"Coach: {who}" + (f" (LLM error: {fb['llm_error']})" if fb.get("llm_error") else ""))


def render(res: dict):
    a, fb = res["analysis"], res["feedback"]
    reps, summ = a["reps"], a["summary"]

    if res["mocked"]:
        st.warning("Mock data in use for: **" + ", ".join(res["mocked"]) + "**. "
                   "Numbers below come from data/sample_*.json, not your video.")
    for name, tb in res["errors"]:
        with st.expander(f"⚠️ {name} crashed (fell back to mock data)"):
            st.code(tb)
    if a.get("confidence", 1) < 0.6:
        st.error("We can't see you clearly enough to judge form. Film from the side, landscape, "
                 "with your whole body from head to heels in frame, then upload again.")

    total, clean, fat = summ["rep_count"], summ["clean_reps"], summ.get("fatigue_from_rep")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reps", total)
    c2.metric("Clean reps", f"{clean}/{total}")
    c3.metric("Form score", f"{round(100 * clean / total) if total else 0}%")
    c4.metric("Form held until", f"Rep {fat - 1}" if fat else ("All reps" if clean == total else "—"))
    st.caption(f"Judged against: {a.get('standard', '')} · tracking confidence {a.get('confidence', 0):.0%}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        if res["video"]:
            st.video(res["video"], start_time=int(st.session_state.get("jump", 0)), autoplay=True, muted=True)
        else:
            st.info("Demo mode: no video loaded. Upload a clip to see your overlay here.")
        flagged = [r for r in reps if r["issues"]]
        if flagged:
            st.caption("Jump to a flagged rep:")
            cols = st.columns(min(len(flagged), 6))
            for col, r in zip(cols * 3, flagged):
                col.button(f"Rep {r['n']} · {_mmss(r['bottom'])}", key=f"jump{r['n']}",
                           on_click=_jump_to, args=(r["start"],), width="stretch")
    with right:
        coach_card(fb)

    st.markdown("#### Every rep, measured")
    st.altair_chart(elbow_chart(a), width="stretch")
    st.altair_chart(hip_chart(a), width="stretch")

    st.session_state.rep_starts = [r["start"] for r in reps]
    table = pd.DataFrame([{
        "Rep": r["n"],
        "Time": _mmss(r["start"]),
        "Bottom elbow °": r["min_elbow"],
        "Top elbow °": r["max_elbow"],
        "Hips off line °": r["max_hip_dev"],
        "": "✅" if not r["issues"] else "⚠️",
        "Issues": " · ".join(i.get("message", i["code"]) for i in r["issues"]),
        "Coach note": fb.get("rep_notes", {}).get(str(r["n"]), ""),
    } for r in reps])
    st.caption("Click a row to jump the video to that rep.")
    st.dataframe(table, hide_index=True, key="rep_table", on_select=_on_rep_select,
                 selection_mode="single-row", width="stretch")

    if st.session_state.get("debug"):
        with st.expander("Raw analysis JSON"):
            st.json({k: v for k, v in a.items() if k != "series"})
        with st.expander("Raw feedback JSON"):
            st.json(fb)


# ---------------------------------------------------------------- page

with st.sidebar:
    st.header("Settings")
    standard = st.radio("Judge against", ["Test standard (90°)", "Strict (chest near floor)"])
    strict = standard.startswith("Strict")
    st.toggle("Show raw JSON", key="debug")
    st.divider()
    if os.getenv("ANTHROPIC_API_KEY"):
        st.caption(f"Coach: Claude (`{coach.MODEL}`)")
    else:
        st.caption("Coach: rule-based (no ANTHROPIC_API_KEY in .env)")

st.title("💪 Form Coach")
st.write("Upload a push-up set filmed from the side. We measure every rep, then coach you on what matters most.")

with st.expander("How to film (read this first)"):
    st.markdown(
        "- **Side view**, phone in **landscape**, about 2–3 m away\n"
        "- Whole body in frame, **head to heels**, the whole time\n"
        "- Good light, one person in shot, plain background if you can\n"
        "- Keep it under 30 seconds"
    )

up_col, demo_col = st.columns([4, 1], vertical_alignment="bottom")
with up_col:
    upload = st.file_uploader("Your video", type=["mp4", "mov", "m4v", "avi", "webm"])
with demo_col:
    if st.button("Try the demo", width="stretch"):
        st.session_state.demo = True

key, video_path = None, None
if upload is not None:
    st.session_state.demo = False
    video_path, digest = video.save_upload(upload.getvalue(), Path(upload.name).suffix.lower() or ".mp4")
    key = (digest, strict)
elif st.session_state.get("demo"):
    key = ("demo", strict)

if key and st.session_state.get("result_key") != key:
    st.session_state.jump = 0.0
    st.session_state.result = run_pipeline(video_path, strict)
    st.session_state.result_key = key

if key and st.session_state.get("result"):
    render(st.session_state.result)
