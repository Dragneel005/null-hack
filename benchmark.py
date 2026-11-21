"""Score a push-up analyzer against simulated ground truth (CONTRACT.md §7).

    evaluate_analyzer(samples, analyze=None, by=None) -> metrics

`samples` come from synth.generate (landmarks + labels). `analyze` defaults to
analysis.analyze. Reps are matched to ground truth by time overlap (IoU >= 0.3), then
faults are scored per matched rep, so a correct fault on the wrong rep counts as wrong.

metrics = {
  "n_clips", "errors",
  "rep_count_exact": share of clips with the exact rep count, "rep_count_mae",
  "rep_precision", "rep_recall",
  "faults": {CODE: {"precision", "recall", "f1", "support"}},
  "by_condition": {name: <same metrics, without by_condition>}   # when `by` is given
}
"""
from __future__ import annotations

import traceback

FAULT_CODES = ["SHALLOW", "NO_LOCKOUT", "HIP_SAG", "HIP_PIKE", "HEAD_DROP", "FAST_DESCENT"]
MIN_IOU = 0.3


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _match(truth_reps, pred_reps):
    """Greedy one-to-one matching by IoU. Returns list of (truth_idx, pred_idx)."""
    pairs = sorted(((_iou((t["start"], t["end"]), (p["start"], p["end"])), i, j)
                    for i, t in enumerate(truth_reps) for j, p in enumerate(pred_reps)), reverse=True)
    used_t, used_p, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < MIN_IOU:
            break
        if i not in used_t and j not in used_p:
            used_t.add(i), used_p.add(j), out.append((i, j))
    return out


def _codes(rep):
    return {(i.get("code") if isinstance(i, dict) else i) for i in rep.get("issues", [])}


def _empty():
    return {"n_clips": 0, "errors": 0, "exact": 0, "abs_err": 0.0, "tp_rep": 0, "n_truth": 0,
            "n_pred": 0, "fault": {c: {"tp": 0, "fp": 0, "fn": 0} for c in FAULT_CODES}}


def _accumulate(acc, labels, analysis):
    truth = labels["reps"]
    pred = analysis.get("reps", []) if analysis else []
    n_pred = analysis.get("summary", {}).get("rep_count", len(pred)) if analysis else 0
    acc["exact"] += int(n_pred == labels["rep_count"])
    acc["abs_err"] += abs(n_pred - labels["rep_count"])
    pairs = _match(truth, pred)
    acc["tp_rep"] += len(pairs)
    acc["n_truth"] += len(truth)
    acc["n_pred"] += len(pred)
    matched_t = {i: j for i, j in pairs}
    matched_p = {j for _, j in pairs}
    for i, t in enumerate(truth):
        tc = set(t["faults"])
        pc = _codes(pred[matched_t[i]]) if i in matched_t else set()
        for c in FAULT_CODES:
            f = acc["fault"][c]
            f["tp"] += int(c in tc and c in pc)
            f["fn"] += int(c in tc and c not in pc)
            f["fp"] += int(c in pc and c not in tc)
    for j, p in enumerate(pred):
        if j not in matched_p:
            for c in _codes(p) & set(FAULT_CODES):
                acc["fault"][c]["fp"] += 1


def _finish(acc):
    def div(a, b):
        return round(a / b, 4) if b else 0.0
    n_ok = acc["n_clips"]
    faults = {}
    for c, f in acc["fault"].items():
        p, r = div(f["tp"], f["tp"] + f["fp"]), div(f["tp"], f["tp"] + f["fn"])
        faults[c] = {"precision": p, "recall": r, "f1": div(2 * p * r, p + r),
                     "support": f["tp"] + f["fn"]}
    return {"n_clips": n_ok, "errors": acc["errors"],
            "rep_count_exact": div(acc["exact"], n_ok), "rep_count_mae": div(acc["abs_err"], n_ok),
            "rep_precision": div(acc["tp_rep"], acc["n_pred"]),
            "rep_recall": div(acc["tp_rep"], acc["n_truth"]), "faults": faults}


def evaluate_analyzer(samples, analyze=None, by=None, analyze_kwargs=None) -> dict:
    """Run `analyze(landmarks)` on every simulated clip and score it against the labels.

    by: optional fn(sample) -> condition name (e.g. clothing, camera yaw bucket) for a breakdown.
    A clip whose analysis raises counts as an error and as a miss on every rep.
    """
    if analyze is None:
        import analysis
        analyze = analysis.analyze
    analyze_kwargs = analyze_kwargs or {}
    total, groups, first_error = _empty(), {}, None
    for s in samples:
        try:
            result, err = analyze(s["landmarks"], **analyze_kwargs), False
        except Exception:  # noqa: BLE001 - a crash is a result worth reporting
            result, err = None, True
            first_error = first_error or traceback.format_exc(limit=3)
        targets = [total] + ([groups.setdefault(by(s), _empty())] if by else [])
        for acc in targets:
            acc["n_clips"] += 1
            acc["errors"] += int(err)
            _accumulate(acc, s["labels"], result)
    out = _finish(total)
    if by:
        out["by_condition"] = {k: _finish(v) for k, v in sorted(groups.items())}
    if first_error:
        out["first_error"] = first_error
    return out


def format_report(m: dict) -> str:
    """Plain-text table for the terminal / deck."""
    lines = [f"clips {m['n_clips']} (errors {m['errors']})  rep count exact {m['rep_count_exact']:.0%}  "
             f"MAE {m['rep_count_mae']:.2f}  rep P {m['rep_precision']:.0%} R {m['rep_recall']:.0%}",
             f"{'fault':14s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>5s}"]
    for c, f in m["faults"].items():
        lines.append(f"{c:14s} {f['precision']:6.0%} {f['recall']:6.0%} {f['f1']:6.2f} {f['support']:5d}")
    for k, v in m.get("by_condition", {}).items():
        lines.append(f"  [{k}] count exact {v['rep_count_exact']:.0%}  rep R {v['rep_recall']:.0%}  "
                     f"P {v['rep_precision']:.0%}")
    return "\n".join(lines)


if __name__ == "__main__":
    import synth
    samples = synth.generate(n=60, seed=123)
    print(format_report(evaluate_analyzer(
        samples, by=lambda s: s["labels"]["params"].get("clothing", "fitted"))))
