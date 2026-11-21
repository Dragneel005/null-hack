from .landmarks import KEY_JOINTS

MIN_CONFIDENCE = 0.5
MIN_DETECTED_FRACTION = 0.5
FILMING_TIP = ("Film from the side, whole body in frame, phone in landscape "
               "2-3 m away at hip height.")


def assess(frames: list[dict]) -> dict:
    """Pick the camera-facing side and say whether the clip is trustworthy.

    Returns {"side", "confidence", "detected_fraction", "message"}; message is None
    when the clip is good, otherwise a tip the UI can show instead of feedback.
    """
    detected = [f["lm"] for f in frames if f["lm"] is not None]
    if not detected:
        return {"side": "left", "confidence": 0.0, "detected_fraction": 0.0,
                "message": "No person detected. " + FILMING_TIP}

    def side_vis(prefix):
        vals = [lm[f"{prefix}_{j}"][2] for lm in detected for j in KEY_JOINTS]
        return sum(vals) / len(vals)

    vis_l, vis_r = side_vis("l"), side_vis("r")
    side, conf = ("left", vis_l) if vis_l >= vis_r else ("right", vis_r)
    frac = len(detected) / len(frames)

    message = None
    if conf < MIN_CONFIDENCE or frac < MIN_DETECTED_FRACTION:
        message = "Body not clearly visible. " + FILMING_TIP
    return {"side": side, "confidence": round(conf, 4),
            "detected_fraction": round(frac, 4), "message": message}
