"""Shared geometry helpers. All points are (x, y) in the same units (pixels)."""
import numpy as np


def angle(a, b, c) -> float:
    """Angle at joint b formed by a-b-c, in degrees (0..180)."""
    ba = np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]
    bc = np.asarray(c, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos = ba @ bc / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
