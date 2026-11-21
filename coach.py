"""OWNER: Person 3 (Product). Analysis dict -> coaching feedback dict.

The LLM never judges form. It only turns measured issues into coaching, so it can't
invent problems. With no API key, or if the call fails, a rule-based fallback runs so
the demo never breaks.
"""
from __future__ import annotations

import base64
import json
import os
import re
from collections import Counter

MODEL = os.getenv("COACH_MODEL", "claude-sonnet-5-5")

# code -> (issue name, why it matters, cue). Keep in sync with CONTRACT.md issue codes.
CUES = {
    "SHALLOW":      ("Not deep enough", "Partial reps skip the hardest part of the range, where most strength is built.", "Lower until your elbows hit 90°, chest towards the floor."),
    "NO_LOCKOUT":   ("No lockout at the top", "Stopping short keeps tension off the triceps and makes reps not count in a test.", "Push the floor away until your arms are straight."),
    "HIP_SAG":      ("Hips sagging", "A sagging midsection shifts load onto the lower back.", "Squeeze your glutes and pull your ribs down."),
    "HIP_PIKE":     ("Hips piking up", "Raising the hips shortens the range and takes load off the chest.", "Body in one straight plank line, hips level with shoulders."),
    "HEAD_DROP":    ("Head dropping", "Leading with the head strains the neck and fakes depth.", "Keep your gaze slightly ahead of your hands, neck long."),
    "FAST_DESCENT": ("Dropping too fast", "Falling into the bottom loses control and stresses the shoulders.", "Take about one second to lower yourself."),
}

SYSTEM = """You are an expert calisthenics coach reviewing measured push-up data from a
pose-estimation system. Rules:
- Only talk about issues that appear in the data. Never invent problems.
- Cite rep numbers and timestamps (m:ss) as evidence.
- Give at most 2 priorities, most important first (safety > range of motion > tempo).
- Each priority gets one short, memorable cue a coach would shout.
- If fatigue_from_rep is set, mention that form held up until then.
- Be encouraging and concrete. No generic advice.
- If images are attached, they are frames with the detected skeleton drawn on. In
  frame_check, say "ok" if the skeleton sits on the body, otherwise "suspect: <why>".
Reply with ONLY a JSON object with keys: headline, summary, priorities (list of
{issue, evidence, why, cue}), positives (list of strings), rep_notes (object rep
number -> note, only for reps worth commenting on), frame_check."""


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _compact(analysis: dict) -> dict:
    """Drop the per-frame series (charts only) to keep the prompt small."""
    return {k: v for k, v in analysis.items() if k != "series"}


def _rep_range(reps: list[int]) -> str:
    reps = sorted(set(reps))
    if len(reps) > 2 and reps[-1] - reps[0] == len(reps) - 1:
        return f"Reps {reps[0]}–{reps[-1]}"
    if len(reps) == 1:
        return f"Rep {reps[0]}"
    return "Reps " + ", ".join(map(str, reps[:-1])) + f" and {reps[-1]}"


def fallback_feedback(analysis: dict) -> dict:
    """Deterministic feedback from the analysis alone. Also a good baseline to compare the LLM to."""
    reps = analysis.get("reps", [])
    summ = analysis.get("summary", {})
    issues = [(r["n"], i) for r in reps for i in r.get("issues", [])]
    counts = Counter(i["code"] for _, i in issues)

    priorities = []
    for code, _ in counts.most_common(2):
        hits = [(n, i) for n, i in issues if i["code"] == code]
        name, why, cue = CUES.get(code, (code.title(), "", ""))
        first_t = min(i["t"] for _, i in hits)
        worst = max(abs(i.get("value", 0)) for _, i in hits)
        priorities.append({
            "issue": name,
            "evidence": f"{_rep_range([n for n, _ in hits])} (from {_mmss(first_t)}), worst {worst:.0f}°",
            "why": why, "cue": cue,
        })

    total, clean = summ.get("rep_count", len(reps)), summ.get("clean_reps", 0)
    fatigue = summ.get("fatigue_from_rep")
    if not issues:
        headline = f"{total} clean reps. Textbook."
    elif fatigue:
        headline = f"Solid until rep {fatigue - 1}, then form broke down"
    else:
        headline = f"{clean} of {total} reps were clean"

    positives = []
    if not counts.get("SHALLOW"):
        positives.append("Full depth on every rep")
    if not counts.get("NO_LOCKOUT"):
        positives.append("Full lockout at the top")
    if fatigue:
        positives.append(f"Clean form for the first {fatigue - 1} reps")

    return {
        "headline": headline,
        "summary": f"{clean} of {total} reps met the {analysis.get('standard', 'push-up')} standard."
                   + (f" Your form held through rep {fatigue - 1} and broke down after, which points to fatigue"
                      " rather than technique." if fatigue else ""),
        "priorities": priorities,
        "positives": positives,
        "rep_notes": {},
        "frame_check": "not checked",
        "source": "fallback",
    }


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def feedback(analysis: dict, frames: list[bytes] | None = None) -> dict:
    """Coaching feedback for an analysis. frames: optional JPEGs of the worst reps (with overlay)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return fallback_feedback(analysis)
    try:
        import anthropic

        content = [{"type": "text", "text": "Measured push-up analysis:\n" + json.dumps(_compact(analysis))}]
        for jpg in (frames or [])[:3]:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpg).decode()}})

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        out = _parse_json(text)
        out["source"] = "llm"
        return out
    except Exception as e:  # never let the LLM break the demo
        out = fallback_feedback(analysis)
        out["llm_error"] = f"{type(e).__name__}: {e}"
        return out
