"""FormSim benchmark: score the rule-based analyzer against simulated ground truth,
and train a sim-only ML fault classifier that is then applied to real clips (sim-to-real).

Implements CONTRACT section 7:
  - ``evaluate_analyzer(samples) -> metrics``
  - ``sim_to_real(samples, real_landmarks_list) -> metrics``

Rep matching: every predicted rep is matched 1:1 to the ground-truth rep with the
largest time overlap (IoU of [start, end]); a match needs IoU >= 0.3 or the
predicted bottom to fall inside the true rep. Fault precision/recall/F1 are then
computed per matched rep (unmatched true reps count as misses, unmatched predicted
reps as false alarms for every fault they report).
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

import analysis as _an

FAULTS = list(_an.FAULT_CODES)
FEATURES = ["min_elbow", "max_elbow", "elbow_range", "max_hip_sag", "max_hip_pike", "hip_dev_signed_max",
            "min_head", "descent_s", "ascent_s", "duration_s", "descent_ratio", "elbow_sd",
            "hip_dev_mean", "body_line_min", "head_drop_max"]


# ----------------------------------------------------------------------------- helpers
def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match_reps(pred: list[dict], truth: list[dict], min_iou: float = 0.3) -> list[tuple[int, int]]:
    """Greedy 1:1 matching of predicted to true reps by time overlap. Returns (pred_idx, true_idx) pairs."""
    cands = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            iou = _iou(float(p["start"]), float(p["end"]), float(t["start"]), float(t["end"]))
            inside = float(t["start"]) <= float(p.get("bottom", -1)) <= float(t["end"])
            if iou >= min_iou or (inside and iou > 0.05):
                cands.append((iou + (0.5 if inside else 0.0), i, j))
    cands.sort(reverse=True)
    used_p, used_t, out = set(), set(), []
    for _, i, j in cands:
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j); out.append((i, j))
    return sorted(out)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
    return {"precision": None if p is None else round(p, 3), "recall": None if r is None else round(r, 3),
            "f1": None if f1 is None else round(f1, 3), "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}


def _pred_faults(rep: dict) -> set:
    return {i["code"] for i in rep.get("issues") or []}


# ----------------------------------------------------------------------------- evaluate
def evaluate_analyzer(samples: list[dict], thresholds: Optional[dict] = None,
                      progress=None) -> dict:
    """Run analysis.analyze on simulated clips and compare to their ground-truth labels.

    Returns {n, rep_count_exact_acc, rep_count_within1_acc, rep_count_mae, rep_detection {precision,
    recall, f1}, timing_mae_s, per_fault {code: {precision, recall, f1, support}}, confusion, worst_clips,
    by_camera, notes, seconds}.
    """
    t0 = time.time()
    n = len(samples)
    exact = within1 = 0
    errs: list[int] = []
    tp_r = fp_r = fn_r = 0
    tp = {f: 0 for f in FAULTS}; fp = dict(tp); fn = dict(tp)
    timing_err: list[float] = []
    depth_err: list[float] = []
    confusion = {"over_count": 0, "under_count": 0, "exact": 0}
    worst: list[dict] = []
    by_cam: dict[str, list[int]] = {}
    cam_cnt: dict[str, dict[str, dict[str, int]]] = {}   # bucket -> fault -> tp/fp/fn
    proj = {"reps": 0, "disagree": 0, "elbow_gap": [], "by_fault": {f: 0 for f in FAULTS}}
    for k, s in enumerate(samples):
        lab = s["labels"]
        truth = lab.get("reps", [])
        a = _an.analyze(s["landmarks"], thresholds)
        pred = a["reps"]
        n_t, n_p = int(lab.get("rep_count", len(truth))), len(pred)
        ok = n_t == n_p
        exact += ok
        within1 += abs(n_t - n_p) <= 1
        errs.append(n_p - n_t)
        confusion["exact" if ok else ("over_count" if n_p > n_t else "under_count")] += 1
        cam = (lab.get("params", {}).get("camera") or {})
        yaw = abs(float(cam.get("yaw_deg", 0.0)))
        bucket = "side (|yaw|<10°)" if yaw < 10 else "oblique (|yaw|≥10°)"
        by_cam.setdefault(bucket, []).append(int(ok))
        if not ok:
            worst.append({"id": s.get("id", k), "true_reps": n_t, "pred_reps": n_p,
                          "faults": ",".join(lab.get("faults_injected") or []) or "clean",
                          "yaw_deg": round(yaw, 1), "fps": cam.get("fps")})
        cc = cam_cnt.setdefault(bucket, {f: {"tp": 0, "fp": 0, "fn": 0} for f in FAULTS})
        for tr_ in truth:
            if "faults_2d" in tr_:
                proj["reps"] += 1
                a3, a2 = set(tr_.get("faults") or []), set(tr_["faults_2d"])
                proj["disagree"] += a3 != a2
                for f in a3 ^ a2:
                    proj["by_fault"][f] += 1
                if "min_elbow_2d" in tr_:
                    proj["elbow_gap"].append(float(tr_["min_elbow_2d"]) - float(tr_["min_elbow"]))
        m = match_reps(pred, truth)
        mp_, mt_ = {i for i, _ in m}, {j for _, j in m}
        tp_r += len(m); fp_r += n_p - len(m); fn_r += n_t - len(m)
        for i, j in m:
            timing_err.append(abs(float(pred[i]["bottom"]) - float(truth[j]["bottom"])))
            depth_err.append(abs(float(pred[i]["min_elbow"]) - float(truth[j]["min_elbow"])))
            pf, tf = _pred_faults(pred[i]), set(truth[j].get("faults") or [])
            for f in FAULTS:
                tp[f] += (f in pf) and (f in tf)
                fp[f] += (f in pf) and (f not in tf)
                fn[f] += (f not in pf) and (f in tf)
                cc[f]["tp"] += (f in pf) and (f in tf)
                cc[f]["fp"] += (f in pf) and (f not in tf)
                cc[f]["fn"] += (f not in pf) and (f in tf)
        for i in set(range(n_p)) - mp_:
            for f in _pred_faults(pred[i]):
                fp[f] += 1
                cc[f]["fp"] += 1
        for j in set(range(n_t)) - mt_:
            for f in truth[j].get("faults") or []:
                if f in fn:
                    fn[f] += 1
                    cc[f]["fn"] += 1
        if progress:
            progress((k + 1) / max(n, 1))
    per_fault = {f: _prf(tp[f], fp[f], fn[f]) for f in FAULTS}
    f1s = [v["f1"] for v in per_fault.values() if v["f1"] is not None]
    worst.sort(key=lambda w: -abs(w["pred_reps"] - w["true_reps"]))
    return {
        "n": n,
        "rep_count_exact_acc": round(exact / n, 3) if n else None,
        "rep_count_within1_acc": round(within1 / n, 3) if n else None,
        "rep_count_mae": round(float(np.mean(np.abs(errs))), 3) if errs else None,
        "rep_count_bias": round(float(np.mean(errs)), 3) if errs else None,
        "rep_detection": _prf(tp_r, fp_r, fn_r),
        "macro_fault_f1": round(float(np.mean(f1s)), 3) if f1s else None,
        "bottom_time_mae_s": round(float(np.mean(timing_err)), 3) if timing_err else None,
        "min_elbow_mae_deg": round(float(np.mean(depth_err)), 2) if depth_err else None,
        "per_fault": per_fault,
        "confusion": confusion,
        "by_camera": {k: {"clips": len(v), "rep_count_exact_acc": round(float(np.mean(v)), 3),
                          "macro_fault_f1": _macro_f1(cam_cnt.get(k, {}))}
                      for k, v in sorted(by_cam.items(), reverse=True)},
        "projection_gap": {
            "reps": proj["reps"],
            "label_disagree_rate": round(proj["disagree"] / proj["reps"], 3) if proj["reps"] else None,
            "min_elbow_2d_minus_3d_mean_deg": round(float(np.mean(proj["elbow_gap"])), 2) if proj["elbow_gap"] else None,
            "flips_by_fault": proj["by_fault"],
            "meaning": ("share of reps whose 3D ground-truth faults differ from what a PERFECT (noise-free) 2D "
                        "tracker would see — elbow flare and camera angle hide or fake faults in 2D. "
                        "This is an upper bound on what any 2D rule engine can get right."),
        },
        "worst_clips": worst[:10],
        "seconds": round(time.time() - t0, 2),
        "notes": ("Analyzer (rules, 2D pixel angles) vs simulator ground truth (3D angles before camera projection, "
                  "noise and dropouts). Part of the rule errors is unavoidable 2D projection error (elbow flare, "
                  "camera yaw) even on a perfect side view — see projection_gap. Reps matched 1:1 by time overlap; fault metrics are per matched rep. "
                  "Label thresholds: " + "; ".join(f"{k}: {v}" for k, v in _label_thresholds().items())),
    }


def _macro_f1(cnt: dict) -> Optional[float]:
    f1s = [_prf(v["tp"], v["fp"], v["fn"])["f1"] for v in cnt.values()]
    f1s = [f for f in f1s if f is not None]
    return round(float(np.mean(f1s)), 3) if f1s else None


def _label_thresholds() -> dict:
    try:
        import synth
        return dict(synth.LABEL_THRESHOLDS)
    except Exception:  # noqa: BLE001
        return {}


# ----------------------------------------------------------------------------- features
def rep_features(analysis: dict) -> list[dict]:
    """Per-rep feature vectors (dict of FEATURES) from an analysis dict (series + reps)."""
    ser = analysis.get("series", {})
    t = np.asarray(ser.get("t", []), float)

    def arr(key):
        return np.array([np.nan if v is None else float(v) for v in ser.get(key, [])], float)

    el, hip, head, body = arr("elbow"), arr("hip_dev"), arr("head"), arr("body_line")
    out = []
    for r in analysis.get("reps", []):
        m = (t >= r["start"]) & (t <= r["end"]) if len(t) else np.zeros(0, bool)

        def stat(x, fn, default=np.nan):
            v = x[m] if len(x) == len(m) else np.zeros(0)
            v = v[np.isfinite(v)]
            return float(fn(v)) if len(v) else default

        hmax = stat(hip, np.max, 0.0); hmin = stat(hip, np.min, 0.0)
        dur = float(r["end"] - r["start"])
        out.append({
            "min_elbow": float(r["min_elbow"]), "max_elbow": float(r["max_elbow"]),
            "elbow_range": float(r["max_elbow"] - r["min_elbow"]),
            "max_hip_sag": float(r["max_hip_sag"]), "max_hip_pike": float(r["max_hip_pike"]),
            "hip_dev_signed_max": hmax if abs(hmax) >= abs(hmin) else hmin,
            "min_head": float(r["min_head"]) if r.get("min_head") is not None else stat(head, np.min, 170.0),
            "descent_s": float(r["descent_s"]), "ascent_s": float(r["ascent_s"]),
            "duration_s": dur, "descent_ratio": float(r["descent_s"]) / max(dur, 1e-3),
            "elbow_sd": stat(el, np.std, 0.0), "hip_dev_mean": stat(hip, np.mean, 0.0),
            "body_line_min": stat(body, np.min, 180.0), "head_drop_max": 180.0 - stat(head, np.min, 180.0),
        })
    return out


def _X(feats: list[dict]) -> np.ndarray:
    X = np.array([[f.get(k, np.nan) for k in FEATURES] for f in feats], float)
    if X.size:
        X = np.where(np.isfinite(X), X, 0.0)
    return X.reshape(-1, len(FEATURES))


# ----------------------------------------------------------------------------- sim to real
def sim_to_real(samples: list[dict], real_landmarks_list: list[dict], seed: int = 0,
                test_frac: float = 0.25, max_clips: int = 250, real_injected: bool = True) -> dict:
    """Train a multi-label fault classifier on SIMULATED reps only and apply it to real clips.

    Features are computed by running analysis.analyze on each simulated clip (so the model sees the
    same noisy 2D measurements it will see on real video); labels are the simulator's ground truth for
    the matched true rep. Split is by athlete (clip) so held-out accuracy measures generalisation.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score

    t0 = time.time()
    samples = samples[:max_clips]
    feats, Y, groups, rules_y = [], [], [], []
    for k, s in enumerate(samples):
        a = _an.analyze(s["landmarks"])
        fv = rep_features(a)
        truth = s["labels"].get("reps", [])
        for i, j in match_reps(a["reps"], truth):
            feats.append(fv[i])
            pf = _pred_faults(a["reps"][i])
            rules_y.append([int(f in pf) for f in FAULTS])     # what analyze() actually reported
            tf = set(truth[j].get("faults") or [])
            Y.append([int(f in tf) for f in FAULTS])
            groups.append(k)
    if len(feats) < 20:
        return {"error": f"only {len(feats)} matched simulated reps — generate more athletes", "n_train_reps": len(feats)}
    X, Y, groups = _X(feats), np.array(Y, int), np.array(groups)
    rng = np.random.default_rng(seed)
    clips = np.unique(groups)
    test_clips = set(rng.choice(clips, size=max(1, int(len(clips) * test_frac)), replace=False).tolist())
    te = np.array([g in test_clips for g in groups])
    tr = ~te

    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                 random_state=seed, n_jobs=-1)
    clf.fit(X[tr], Y[tr])
    P = clf.predict(X[te])
    # rule engine on the same held-out reps: exactly the faults analysis.analyze reported
    R = np.array(rules_y, int)[te]
    held = {}
    for c, f in enumerate(FAULTS):
        if Y[te, c].sum() == 0 and P[:, c].sum() == 0:
            continue
        held[f] = {"ml_f1": round(float(f1_score(Y[te, c], P[:, c], zero_division=0)), 3),
                   "ml_precision": round(float(precision_score(Y[te, c], P[:, c], zero_division=0)), 3),
                   "ml_recall": round(float(recall_score(Y[te, c], P[:, c], zero_division=0)), 3),
                   "rules_f1": round(float(f1_score(Y[te, c], R[:, c], zero_division=0)), 3),
                   "support": int(Y[te, c].sum())}
    exact_ml = float(np.mean((P == Y[te]).all(axis=1)))
    exact_rules = float(np.mean((R == Y[te]).all(axis=1)))
    # final model on all sim data for real-world inference
    clf.fit(X, Y)
    imp = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1])

    real_rows, agree, total = [], 0, 0
    for lm in real_landmarks_list:
        a = _an.analyze(lm)
        fv = rep_features(a)
        if not fv:
            real_rows.append({"clip": lm.get("source", "?"), "rep": None, "rules": "(no reps detected)",
                              "ml_sim_trained": "", "agree": None})
            continue
        Pr = clf.predict(_X(fv))
        proba = clf.predict_proba(_X(fv))
        for r, row, prob in zip(a["reps"], Pr, zip(*proba)):
            rules = sorted(_pred_faults(r))
            ml = [f for f, v in zip(FAULTS, row) if v]
            # probability of fault class (handle single-class estimators)
            conf = []
            for c, pr in enumerate(prob):
                cls = list(clf.estimators_[0].classes_[c]) if hasattr(clf.estimators_[0], "classes_") else [0, 1]
                conf.append(float(pr[-1]) if len(pr) > 1 else float(cls[0]))
            agr = set(rules) == set(ml)
            agree += agr; total += 1
            real_rows.append({"clip": lm.get("source", "?"), "rep": r["n"],
                              "rules": ", ".join(rules) or "clean", "ml_sim_trained": ", ".join(ml) or "clean",
                              "ml_max_fault_prob": round(max(conf) if conf else 0.0, 2), "agree": agr})
    # real motion + simulated fault with a KNOWN label (inject.py): the only labeled real-world test we have
    inj = _real_injected_eval(clf, real_landmarks_list) if real_injected else {}
    return {
        "real_injected": inj,
        "n_train_reps": int(tr.sum()), "n_heldout_reps": int(te.sum()),
        "n_sim_clips": int(len(clips)),
        "heldout_exact_acc_ml": round(exact_ml, 3),
        "heldout_exact_acc_rules": round(exact_rules, 3),
        "real_reps": total, "real_agreement_rate": round(agree / total, 3) if total else None,
        "heldout_per_fault": held,
        "feature_importance": {k: {"importance": round(float(v), 3)} for k, v in imp[:8]},
        "real_predictions": real_rows,
        "model": "RandomForest(200 trees, balanced), one multi-output model; trained on simulated reps only",
        "seconds": round(time.time() - t0, 2),
        "notes": ("Held-out = unseen simulated athletes (split by clip). 'rules' = analysis.py thresholds applied to "
                  "the same measurements. On real clips there are no fault labels, so we report agreement between "
                  "the sim-trained model and the rule engine."),
    }


def _real_injected_eval(clf, real_landmarks_list: list[dict]) -> dict:
    """Inject each fault into every real clip that has reps; score ML and rules on the known labels."""
    try:
        import inject
    except Exception:  # noqa: BLE001
        return {}
    rows, ml_ok, ru_ok, ml_hit, ru_hit, n = [], 0, 0, 0, 0, 0
    for lm in real_landmarks_list:
        base = _an.analyze(lm)
        if not base["reps"] or any(r.get("issues") for r in base["reps"]):
            continue                                  # need a clean real baseline so the label is exactly {code}
        for code in [c for c in inject.FAULT_OPTIONS if c != "none"]:
            a = _an.analyze(inject.inject_fault(lm, code))
            fv = rep_features(a)
            if not fv:
                continue
            Pr = clf.predict(_X(fv))
            mh = rh = me = re_ = 0
            for r, row in zip(a["reps"], Pr):
                ml = {f for f, v in zip(FAULTS, row) if v}
                ru = _pred_faults(r)
                mh += code in ml; rh += code in ru
                me += ml == {code}; re_ += ru == {code}
            k = len(a["reps"])
            n += k; ml_hit += mh; ru_hit += rh; ml_ok += me; ru_ok += re_
            rows.append({"clip": lm.get("source", "?"), "injected": code, "reps": k,
                         "ml_detected": f"{mh}/{k}", "rules_detected": f"{rh}/{k}"})
    if not n:
        return {}
    return {"reps": n, "ml_detect_rate": round(ml_hit / n, 3), "rules_detect_rate": round(ru_hit / n, 3),
            "ml_exact_acc": round(ml_ok / n, 3), "rules_exact_acc": round(ru_ok / n, 3), "rows": rows,
            "notes": "Real clean reps with a geometrically injected fault (inject.py): real motion, known label."}


def _rules_from_features(feats: list[dict], th: dict) -> np.ndarray:
    """Fault vector the rule engine would give from rep features (mirrors analysis.py thresholds)."""
    R = np.zeros((len(feats), len(FAULTS)), int)
    for i, f in enumerate(feats):
        R[i, FAULTS.index("SHALLOW")] = f["min_elbow"] > th["depth_deg"]
        R[i, FAULTS.index("NO_LOCKOUT")] = f["max_elbow"] < th["lockout_deg"]
        R[i, FAULTS.index("HIP_SAG")] = f["max_hip_sag"] > th["hip_sag_deg"]
        R[i, FAULTS.index("HIP_PIKE")] = f["max_hip_pike"] > th["hip_pike_deg"]
        R[i, FAULTS.index("HEAD_DROP")] = f["head_drop_max"] > th["head_drop_deg"]
        R[i, FAULTS.index("FAST_DESCENT")] = f["descent_s"] < th["fast_descent_s"]
    return R


# ----------------------------------------------------------------------------- CLI
def _table(per_fault: dict) -> str:
    lines = [f"{'fault':<14}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}"]
    for f, v in per_fault.items():
        fmt = lambda x: "  —  " if x is None else f"{x:.2f}"  # noqa: E731
        lines.append(f"{f:<14}{fmt(v['precision']):>7}{fmt(v['recall']):>8}{fmt(v['f1']):>7}{v['support']:>9}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    import synth

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t0 = time.time()
    samples = synth.generate(n, seed=1)
    print(f"generated {n} simulated athletes in {time.time() - t0:.1f}s")
    m = evaluate_analyzer(samples)
    print(f"\n== Analyzer vs simulated ground truth ({m['n']} clips, {m['seconds']}s) ==")
    print(f"rep count exact: {m['rep_count_exact_acc']:.1%}  within ±1: {m['rep_count_within1_acc']:.1%}  "
          f"MAE {m['rep_count_mae']}  bias {m['rep_count_bias']}")
    rd = m["rep_detection"]
    print(f"rep detection P {rd['precision']} R {rd['recall']} F1 {rd['f1']}  bottom-time MAE {m['bottom_time_mae_s']}s  "
          f"min-elbow MAE {m['min_elbow_mae_deg']}°")
    print(_table(m["per_fault"]))
    print("by camera:", m["by_camera"])
    print("worst:", m["worst_clips"][:5])
    root = Path(__file__).resolve().parent
    real = [json.loads(p.read_text()) for p in sorted((root / "samples").glob("landmarks_*.json"))]
    s = sim_to_real(samples, real)
    print(f"\n== Sim-to-real ({s.get('n_train_reps')} train / {s.get('n_heldout_reps')} held-out reps, {s.get('seconds')}s) ==")
    print(f"held-out exact-match  ML {s.get('heldout_exact_acc_ml')}  rules {s.get('heldout_exact_acc_rules')}")
    for f, v in s.get("heldout_per_fault", {}).items():
        print(f"  {f:<14} ML f1 {v['ml_f1']:.2f}  rules f1 {v['rules_f1']:.2f}  support {v['support']}")
    print("real-clip agreement:", s.get("real_agreement_rate"))
    ri = s.get("real_injected") or {}
    print("real + injected fault:", {k: v for k, v in ri.items() if k != "rows"})
    for r in ri.get("rows", []):
        print("  ", r)
    if len(sys.argv) > 2:   # save for the app hero / Sim Lab ("precomputed benchmark")
        Path(sys.argv[2]).write_text(json.dumps({"n": n, "seed": 1, "eval": m, "s2r": s}, default=str))
        print("saved", sys.argv[2])
    for r in s.get("real_predictions", []):
        print("  ", r)
