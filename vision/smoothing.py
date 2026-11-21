import copy


def smooth_frames(frames: list[dict], window: int = 5) -> list[dict]:
    """Centered moving average of landmark x/y over `window` frames.

    Frames with no pose (lm is None) stay None and are skipped by their neighbours'
    averages. Visibility is left unchanged. Returns new frames; input is not modified.
    """
    half = window // 2
    out = copy.deepcopy(frames)
    for i, frame in enumerate(frames):
        if frame["lm"] is None:
            continue
        neigh = [f["lm"] for f in frames[max(0, i - half): i + half + 1] if f["lm"] is not None]
        for name, val in frame["lm"].items():
            xs = [lm[name][0] for lm in neigh]
            ys = [lm[name][1] for lm in neigh]
            out[i]["lm"][name] = [sum(xs) / len(xs), sum(ys) / len(ys), val[2]]
    return out
