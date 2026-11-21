"""Stage 4 — Coach: analysis (+ physics) -> coaching report (see CONTRACT.md §4).

Two engines, same output schema:

* **Rules** (always works, offline, deterministic): grades the set, ranks
  faults by count x severity, writes punchy cues grounded in biomechanics and,
  when available, the physics simulation numbers (lumbar moment, elbow torque,
  hand force) measured on *this* athlete's reps.
* **Claude** (optional): if ``use_llm`` and ``ANTHROPIC_API_KEY`` are set, the
  compact numeric summary is sent to Claude, which must answer with strict JSON
  in the same schema, grounded only in the numbers. Any failure (no key,
  network, timeout, bad JSON, schema mismatch) falls back to the rules engine.

Public API: ``coach(analysis, physics=None, use_llm=True) -> report``,
``coach_markdown(report) -> str``, ``llm_comparison_note() -> str``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_MODEL = "claude-opus-5-5"
LLM_TIMEOUT_S = 30.0
GRADES = ("A", "B", "C", "D")
NO_GRADE = "N/A"  # used only when there is nothing to grade (0 reps / unusable video)

SEV_WEIGHT = {"low": 1.0, "med": 2.0, "high": 3.0}

FILMING_TIPS = ("Film from the side (90° to your body), whole body in frame from hands to feet, "
                "phone in landscape at hip height about 2-3 m away, good light, one person in shot.")

# Coaching library per fault code: title, cue, biomechanics why, physics metric to cite.
FIXES: dict[str, dict[str, str]] = {
    "HIP_SAG": {
        "title": "Keep your hips in line",
        "cue": "Squeeze your glutes and pull your ribs down — move like one solid plank.",
        "why": "When the hips sag the lower back arches and the spine, not the abs, holds you up; "
               "that shifts load onto the lumbar discs and facet joints.",
        "metric": "peak_lumbar_Nm", "metric_name": "lower-back load",
    },
    "HIP_PIKE": {
        "title": "Drop the hips, stay straight",
        "cue": "Hips down to the shoulder-ankle line — think 'long body', heels to head.",
        "why": "Piking shifts weight back to the feet, so the chest and triceps do less work and the "
               "rep looks harder than the training stimulus it gives.",
        "metric": "peak_hand_force_N", "metric_name": "hand force",
    },
    "SHALLOW": {
        "title": "Go deeper",
        "cue": "Chest to a fist's height off the floor — elbows past 90° every rep.",
        "why": "Partial reps skip the bottom range where the chest is stretched and the elbow torque "
               "peaks — the part of the movement that builds strength.",
        "metric": "peak_elbow_torque_Nm", "metric_name": "elbow torque",
    },
    "NO_LOCKOUT": {
        "title": "Finish every rep",
        "cue": "Push the floor away until your arms are straight and shoulder blades spread.",
        "why": "Stopping short of lockout cuts the range of motion and skips serratus/scapular "
               "protraction work that keeps shoulders healthy.",
        "metric": "work_J", "metric_name": "work per rep",
    },
    "HEAD_DROP": {
        "title": "Keep your neck neutral",
        "cue": "Eyes on the floor slightly ahead of your hands — chin tucked, don't reach with the nose.",
        "why": "Dropping the head flexes the neck under load and usually signals you're shortening "
               "the rep by reaching for the floor with your face.",
        "metric": "", "metric_name": "",
    },
    "FAST_DESCENT": {
        "title": "Control the way down",
        "cue": "Two-second descent: 'one-and, two-and', then push.",
        "why": "Dropping fast means the muscles aren't braking the body; the stop at the bottom spikes "
               "joint forces and you lose time-under-tension.",
        "metric": "peak_hand_force_N", "metric_name": "peak hand force",
    },
}
CODE_LABEL = {
    "HIP_SAG": "hip sag", "HIP_PIKE": "hip pike", "SHALLOW": "shallow depth",
    "NO_LOCKOUT": "no lockout", "HEAD_DROP": "head drop", "FAST_DESCENT": "fast descent",
}


# ----------------------------------------------------------------------------- helpers

def _num(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        v = float(x)
        return v if v == v else default  # NaN guard
    except (TypeError, ValueError):
        return default


def _mmss(t: Any) -> str:
    v = _num(t)
    if v is None:
        return "?"
    return f"{int(v // 60)}:{int(v % 60):02d}"


def _rep_list(nums: list[int]) -> str:
    nums = sorted(set(int(n) for n in nums))
    if not nums:
        return ""
    if len(nums) == 1:
        return f"rep {nums[0]}"
    return "reps " + ", ".join(str(n) for n in nums[:-1]) + f" & {nums[-1]}"


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _grade(avg_score: float, clean_ratio: float) -> str:
    if avg_score >= 85 and clean_ratio >= 0.8:
        return "A"
    if avg_score >= 72 and clean_ratio >= 0.5:
        return "B"
    if avg_score >= 55:
        return "C"
    return "D"


def _physics_by_rep(physics: dict | None) -> dict[int, dict]:
    if not physics:
        return {}
    out = {}
    for r in physics.get("reps") or []:
        try:
            out[int(r.get("n"))] = r
        except (TypeError, ValueError):
            continue
    return out


def _physics_evidence(code: str, faulty: list[int], reps: list[dict], prep: dict[int, dict]) -> str:
    """Quantify the cost of a fault with physics: faulty reps vs clean reps on the chosen metric."""
    metric = FIXES.get(code, {}).get("metric")
    name = FIXES.get(code, {}).get("metric_name", "")
    if not metric or not prep:
        return ""
    faulty_set = set(faulty)
    clean = [int(r["n"]) for r in reps if not r.get("issues")]
    bad_vals = [_num(prep.get(n, {}).get(metric)) for n in faulty_set]
    ok_vals = [_num(prep.get(n, {}).get(metric)) for n in clean]
    b, o = _mean(bad_vals), _mean(ok_vals)
    unit = "N·m" if metric.endswith("Nm") else ("N" if metric.endswith("_N") else "J")
    if b is not None and o is not None and o > 1e-6:
        pct = (b - o) / o * 100.0
        if abs(pct) >= 3:
            direction = "raised" if pct > 0 else "cut"
            return (f"Physics sim: on those reps your {name} averaged {b:.0f} {unit} vs {o:.0f} {unit} "
                    f"on clean reps — the {CODE_LABEL.get(code, code.lower())} {direction} it by {abs(pct):.0f}%.")
    if b is not None:
        return f"Physics sim: {name} peaked around {b:.0f} {unit} on those reps."
    return ""


# ----------------------------------------------------------------------------- rules engine

def _empty_report(analysis: dict, reason: str) -> dict:
    warnings = [w.strip().rstrip(".") for w in (analysis.get("warnings") or []) if isinstance(w, str)]
    detail = (" Detected issues: " + "; ".join(warnings[:3]) + ".") if warnings else ""
    return {
        "source": "rules",
        "headline": reason,
        "grade": NO_GRADE,
        "summary": f"We couldn't score this clip reliably.{detail} {FILMING_TIPS}",
        "top_fixes": [{
            "title": "Re-film for a reliable analysis",
            "cue": "Side view, whole body, phone landscape at hip height, 2-3 m away.",
            "why": "The tracker needs to see shoulder, elbow, wrist, hip and ankle on one side to measure "
                   "elbow depth and body line; front views or cropped feet hide exactly those joints.",
            "reps": [],
        }],
        "strengths": [],
        "per_rep": [],
        "next_session": f"Record one set of 5-10 push-ups with this setup: {FILMING_TIPS}",
    }


def _rules_report(analysis: dict, physics: dict | None) -> dict:
    reps = [r for r in (analysis.get("reps") or []) if isinstance(r, dict)]
    summ = analysis.get("summary") or {}
    conf = _num(analysis.get("confidence"), 0.0) or 0.0
    conf_ok = analysis.get("confidence_ok", True)

    if not reps:
        if conf_ok is False:
            return _empty_report(analysis, "Can't see you clearly — let's fix the camera angle")
        return _empty_report(analysis, "No push-up reps detected")

    n = int(summ.get("rep_count") or len(reps))
    clean = int(summ.get("clean_reps") if summ.get("clean_reps") is not None
                else sum(1 for r in reps if not r.get("issues")))
    avg_score = _num(summ.get("avg_score"))
    if avg_score is None:
        avg_score = _mean([_num(r.get("score")) for r in reps]) or 0.0
    clean_ratio = clean / max(n, 1)
    grade = _grade(avg_score, clean_ratio)
    fatigue = summ.get("fatigue_from_rep")
    d_s, a_s = _num(summ.get("avg_descent_s")), _num(summ.get("avg_ascent_s"))
    prep = _physics_by_rep(physics)

    # --- rank faults by sum of severities (count x severity)
    agg: dict[str, dict] = {}
    for r in reps:
        for iss in r.get("issues") or []:
            code = iss.get("code")
            if not code:
                continue
            a = agg.setdefault(code, {"score": 0.0, "reps": [], "worst": None, "times": []})
            a["score"] += SEV_WEIGHT.get(iss.get("severity"), 1.0)
            a["reps"].append(int(r.get("n", 0)))
            a["times"].append(iss.get("t"))
            v = _num(iss.get("value"))
            if v is not None:
                v = abs(v)
                lower_is_worse = code == "FAST_DESCENT"  # value = descent seconds
                if a["worst"] is None or (v < a["worst"] if lower_is_worse else v > a["worst"]):
                    a["worst"] = v
    ranked = sorted(agg.items(), key=lambda kv: (-kv[1]["score"], kv[0]))

    top_fixes = []
    for code, a in ranked[:2]:
        lib = FIXES.get(code, {"title": f"Fix {code.lower()}", "cue": "Slow down and own each rep.",
                               "why": "This fault was flagged repeatedly by the analyzer."})
        rep_nums = sorted(set(a["reps"]))
        first_t = next((t for t in a["times"] if _num(t) is not None), None)
        where = f"Seen on {_rep_list(rep_nums)}" + (f" (first at {_mmss(first_t)})" if first_t is not None else "")
        worst = ""
        if a["worst"] is not None:
            worst = (f", fastest descent {a['worst']:.2f}s" if code == "FAST_DESCENT"
                     else f", worst {a['worst']:.0f}°")
        why = f"{lib['why']} {where}{worst}."
        ev = _physics_evidence(code, rep_nums, reps, prep)
        if ev:
            why += " " + ev
        top_fixes.append({"title": lib["title"], "cue": lib["cue"], "why": why, "reps": rep_nums})

    # --- strengths
    strengths: list[str] = []
    codes = set(agg)
    depth_vals = [_num(r.get("min_elbow")) for r in reps]
    depth = _mean(depth_vals)
    if "SHALLOW" not in codes and depth is not None:
        strengths.append(f"Full depth on every rep — average bottom elbow angle {depth:.0f}°.")
    elif "SHALLOW" not in codes:
        strengths.append("Full depth on every rep.")
    if "NO_LOCKOUT" not in codes:
        strengths.append("Full lockout at the top of every rep.")
    if not codes & {"HIP_SAG", "HIP_PIKE"}:
        strengths.append("Solid plank — hips stayed on the shoulder-ankle line.")
    if "FAST_DESCENT" not in codes and d_s is not None and d_s >= 0.8:
        strengths.append(f"Controlled lowering — {d_s:.1f}s average descent.")
    if fatigue is None and n >= 5:
        strengths.append(f"No fatigue breakdown across {n} reps — form held to the end.")
    if "HEAD_DROP" not in codes:
        strengths.append("Neutral neck throughout.")
    if physics and (physics.get("summary") or {}).get("equivalent_bench_kg"):
        eq = _num(physics["summary"]["equivalent_bench_kg"])
        if eq:
            strengths.append(f"At the bottom of each rep your hands held about {eq:.0f} kg of body weight — "
                             f"comparable to a {eq:.0f} kg bench press.")
    strengths = strengths[:4]

    # --- headline
    verdict = {"A": "strong set", "B": "solid set, small fixes", "C": "work on form", "D": "form needs attention"}[grade]
    headline = f"{n} rep{'s' if n != 1 else ''}, {clean} clean — {verdict}"

    # --- summary sentences
    s = [f"You completed {n} push-up{'s' if n != 1 else ''} with {clean} clean "
         f"({clean_ratio * 100:.0f}%) and an average form score of {avg_score:.0f}/100."]
    if fatigue:
        s.append(f"Form started to break down from rep {fatigue}, a sign of fatigue — that's where your set really ends.")
    elif n >= 3:
        s.append("Form held steady from first rep to last — no fatigue drop-off detected.")
    if d_s is not None and a_s is not None:
        tempo_note = "nicely controlled" if d_s >= 1.0 else ("a bit quick on the way down" if d_s < 0.6 else "reasonable")
        s.append(f"Tempo: {d_s:.1f}s down / {a_s:.1f}s up ({tempo_note}).")
    if top_fixes:
        s.append(f"Biggest win next time: {top_fixes[0]['title'].lower()}.")
    if conf < 0.6 or conf_ok is False:
        s.append(f"Tracking confidence was low ({conf * 100:.0f}%), so treat these numbers as approximate. {FILMING_TIPS}")
    summary = " ".join(s)

    # --- per-rep notes
    per_rep = []
    for r in reps:
        rn = int(r.get("n", 0))
        parts = []
        sc = _num(r.get("score"))
        issues = r.get("issues") or []
        if issues:
            parts.append("; ".join(i.get("msg") or CODE_LABEL.get(i.get("code"), i.get("code", "")) for i in issues))
        else:
            parts.append("Clean rep")
        me = _num(r.get("min_elbow"))
        if me is not None:
            parts.append(f"bottom {me:.0f}°")
        ds, as_ = _num(r.get("descent_s")), _num(r.get("ascent_s"))
        if ds is not None and as_ is not None:
            parts.append(f"{ds:.1f}s down / {as_:.1f}s up")
        pr = prep.get(rn)
        if pr and _num(pr.get("peak_lumbar_Nm")) is not None:
            parts.append(f"lumbar {_num(pr['peak_lumbar_Nm']):.0f} N·m")
        if fatigue and rn == int(fatigue):
            parts.append("fatigue starts here")
        note = parts[0] + (" — " + ", ".join(parts[1:]) if len(parts) > 1 else "") + "."
        if sc is not None:
            note = f"[{sc:.0f}] " + note
        per_rep.append({"n": rn, "note": note})

    # --- next session
    quality_reps = max(1, (int(fatigue) - 1) if fatigue else clean if clean else max(1, n // 2))
    target = max(3, min(quality_reps, n))
    if grade == "A":
        plan = (f"3 sets of {target + 2} with a 3-second descent and 90 s rest — "
                f"add reps, not speed. Re-film the last set to confirm form holds.")
    else:
        focus = top_fixes[0] if top_fixes else None
        cue = f" Focus cue: \"{focus['cue']}\"" if focus else ""
        plan = (f"3 sets of {target} (stop each set before form breaks), 2-second descent, 90 s rest.{cue} "
                f"Once every rep is clean, add 1-2 reps per set.")
    return {
        "source": "rules", "headline": headline, "grade": grade, "summary": summary,
        "top_fixes": top_fixes, "strengths": strengths, "per_rep": per_rep, "next_session": plan,
    }


# ----------------------------------------------------------------------------- LLM engine

SYSTEM_PROMPT = """You are FormSim Coach, an elite strength coach reviewing ONE set of push-ups.
You receive JSON produced by a pose tracker + biomechanics analyzer + rigid-body physics simulation.

Rules:
- Ground everything in the provided numbers. Only mention faults that appear in the data (issue codes on reps). Never invent faults, reps, timestamps or numbers.
- Cite rep numbers and timestamps (m:ss) when you mention a fault.
- At most 2 top_fixes, ranked by (number of reps affected x severity). Each needs a short punchy cue (what to do/feel) and a 'why' rooted in biomechanics; when physics numbers are given, quote them (e.g. "lower-back moment 34% higher on sagging reps").
- Speak directly to the athlete ("you"), encouraging but honest. No medical claims.
- grade: A (avg score >= 85 and >= 80% clean reps), B (>= 72 and >= 50% clean), C (>= 55), D otherwise.
- Output ONLY one JSON object, no markdown fences, no prose, with exactly these keys:
{"headline": str (<= 60 chars, e.g. "6 reps, 5 clean — strong set"),
 "grade": "A"|"B"|"C"|"D",
 "summary": str (2-3 sentences: rep count, clean reps, fatigue rep if any, tempo),
 "top_fixes": [{"title": str, "cue": str, "why": str, "reps": [int]}],
 "strengths": [str],
 "per_rep": [{"n": int, "note": str}],
 "next_session": str (concrete plan: sets x reps, tempo, rest, focus cue)}"""


def _compact_payload(analysis: dict, physics: dict | None, rules: dict) -> dict:
    """Small, LLM-friendly view of the analysis (no per-frame series)."""
    reps = []
    for r in analysis.get("reps") or []:
        item = {k: r.get(k) for k in ("n", "start", "bottom", "end", "min_elbow", "max_elbow", "max_hip_sag",
                                       "max_hip_pike", "min_head", "descent_s", "ascent_s", "score")}
        item = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in item.items()}
        item["issues"] = [{"code": i.get("code"), "t": i.get("t"), "severity": i.get("severity"),
                           "value": i.get("value"), "msg": i.get("msg")} for i in (r.get("issues") or [])]
        reps.append(item)
    payload: dict[str, Any] = {
        "side": analysis.get("side"), "confidence": analysis.get("confidence"),
        "warnings": analysis.get("warnings") or [], "summary": analysis.get("summary") or {}, "reps": reps,
        "rules_engine_grade": rules.get("grade"),
    }
    if physics:
        ps = physics.get("summary") or {}
        payload["physics"] = {
            "body_mass_kg": physics.get("body_mass_kg"), "summary": ps,
            "reps": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in (physics.get("reps") or [])],
        }
    return payload


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _validate(raw: dict, analysis: dict, rules: dict) -> dict:
    """Coerce Claude's JSON into the report schema; raise on anything unusable."""
    valid_reps = {int(r.get("n", 0)) for r in analysis.get("reps") or []}
    grade = str(raw.get("grade", "")).strip().upper()[:1]
    if grade not in GRADES:
        raise ValueError(f"bad grade {raw.get('grade')!r}")
    headline, summary, plan = raw.get("headline"), raw.get("summary"), raw.get("next_session")
    if not all(isinstance(x, str) and x.strip() for x in (headline, summary, plan)):
        raise ValueError("missing headline/summary/next_session")
    fixes = []
    for f in (raw.get("top_fixes") or [])[:2]:
        if not isinstance(f, dict) or not all(isinstance(f.get(k), str) for k in ("title", "cue", "why")):
            raise ValueError("bad top_fixes entry")
        reps = []
        for x in f.get("reps") or []:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if xi in valid_reps:
                reps.append(xi)
        fixes.append({"title": f["title"], "cue": f["cue"], "why": f["why"], "reps": sorted(set(reps))})
    strengths = [s for s in (raw.get("strengths") or []) if isinstance(s, str)][:5]
    per_rep = []
    for p in raw.get("per_rep") or []:
        try:
            pn = int(p.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        if pn in valid_reps and isinstance(p.get("note"), str):
            per_rep.append({"n": pn, "note": p["note"]})
    if len(per_rep) < len(valid_reps):
        per_rep = rules["per_rep"]  # keep complete per-rep notes
    return {"source": "claude", "headline": headline.strip(), "grade": grade, "summary": summary.strip(),
            "top_fixes": fixes, "strengths": strengths or rules["strengths"], "per_rep": per_rep,
            "next_session": plan.strip()}


def _llm_report(analysis: dict, physics: dict | None, rules: dict) -> dict:
    import anthropic  # lazy: keep import cheap and offline-safe

    model = os.environ.get("COACH_MODEL", DEFAULT_MODEL)
    client = anthropic.Anthropic(timeout=LLM_TIMEOUT_S, max_retries=1)
    payload = _compact_payload(analysis, physics, rules)
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "Analysis JSON:\n" + json.dumps(payload, separators=(",", ":"))}],
    )
    if resp.stop_reason not in ("end_turn", "stop_sequence"):
        raise RuntimeError(f"stop_reason={resp.stop_reason}")
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    report = _validate(_extract_json(text), analysis, rules)
    report["model"] = model
    return report


# ----------------------------------------------------------------------------- public API

def coach(analysis: dict, physics: dict | None = None, use_llm: bool = True) -> dict:
    """Build the coaching report (CONTRACT.md §4).

    Rules engine always runs (offline, deterministic). If ``use_llm`` and
    ``ANTHROPIC_API_KEY`` is set and there are reps to discuss, Claude rewrites
    the report from the same numbers; any failure returns the rules report.
    Extra keys: ``llm_error`` (str) when the LLM path was tried and failed.
    """
    analysis = analysis or {}
    rules = _rules_report(analysis, physics)
    if not (use_llm and os.environ.get("ANTHROPIC_API_KEY") and analysis.get("reps")):
        return rules
    try:
        return _llm_report(analysis, physics, rules)
    except Exception as e:  # noqa: BLE001 — any failure -> deterministic fallback
        rules["llm_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rules


def coach_markdown(report: dict) -> str:
    """Render a report as Markdown for Streamlit (st.markdown)."""
    r = report or {}
    src = "Claude" if r.get("source") == "claude" else "FormSim rules engine"
    lines = [f"### Grade {r.get('grade', '?')} · {r.get('headline', '')}", "", r.get("summary", ""), ""]
    fixes = r.get("top_fixes") or []
    if fixes:
        lines.append("#### Top fixes")
        for i, f in enumerate(fixes, 1):
            reps = f" _(reps {', '.join(str(x) for x in f['reps'])})_" if f.get("reps") else ""
            lines.append(f"**{i}. {f.get('title', '')}**{reps}  ")
            lines.append(f"> **Cue:** {f.get('cue', '')}  ")
            lines.append(f"> **Why:** {f.get('why', '')}")
            lines.append("")
    if r.get("strengths"):
        lines.append("#### What you did well")
        lines += [f"- {s}" for s in r["strengths"]]
        lines.append("")
    if r.get("per_rep"):
        lines.append("<details><summary>Rep-by-rep notes</summary>\n")
        lines += [f"- **Rep {p['n']}:** {p['note']}" for p in r["per_rep"]]
        lines.append("\n</details>\n")
    if r.get("next_session"):
        lines += ["#### Next session", r["next_session"], ""]
    lines.append(f"<sub>Coach: {src}</sub>")
    return "\n".join(lines)


def llm_comparison_note() -> str:
    """Short pitch text: why a plain video LLM misses what FormSim measures."""
    return (
        "**Why not just ask a video LLM?** General video models typically sample ~1 frame per second. "
        "A push-up's bottom position lasts ~0.2-0.5 s and a fast rep takes ~1 s, so at 1 fps the model "
        "often never sees the bottom of the rep: it can't count reps reliably, can't measure elbow depth or "
        "a 15° hip sag, and can't time a 0.3 s drop. FormSim tracks every frame (30 fps) with pose "
        "estimation, measures joint angles in degrees, counts reps with a hysteresis state machine, and "
        "runs a rigid-body physics model for joint loads — then the LLM only writes the coaching from "
        "those hard numbers. Measurement first, language second."
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent
    path = sys.argv[1] if len(sys.argv) > 1 else str(root / "samples" / "landmarks_demo_side.json")
    try:
        import analysis as _an
        lms = json.loads(Path(path).read_text())
        an = _an.analyze(lms)
        ph = None
        try:
            import physics as _ph
            ph = _ph.simulate(lms, an)
        except Exception as e:  # noqa: BLE001
            print(f"(physics unavailable: {e})", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"(analysis unavailable: {e}; using a fabricated analysis)", file=sys.stderr)
        an = {"confidence": 0.9, "confidence_ok": True, "reps": [
            {"n": i, "start": 2.0 * i, "bottom": 2.0 * i + 0.9, "end": 2.0 * i + 1.8, "min_elbow": 80.0,
             "max_elbow": 168.0, "descent_s": 0.9, "ascent_s": 0.8, "score": 92 if i < 5 else 64,
             "issues": [] if i < 5 else [{"code": "HIP_SAG", "t": 2.0 * i + 0.9, "severity": "med",
                                          "value": 17.0, "msg": "Hips sagged 17° below the line"}]}
            for i in range(1, 8)],
            "summary": {"rep_count": 7, "clean_reps": 4, "avg_score": 84.0, "fatigue_from_rep": 5,
                        "issue_counts": {"HIP_SAG": 3}, "avg_descent_s": 0.9, "avg_ascent_s": 0.8}}
        ph = None
    rep = coach(an, ph, use_llm=True)
    print(coach_markdown(rep))
