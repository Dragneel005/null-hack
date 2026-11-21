def frame_timestamps_ms(n_frames: int, fps: float) -> list[int]:
    """Integer ms timestamps for MediaPipe VIDEO mode: strictly increasing, as it requires."""
    out, prev = [], -1
    for i in range(n_frames):
        t = int(i * 1000 / fps)
        if t <= prev:
            t = prev + 1
        out.append(t)
        prev = t
    return out
