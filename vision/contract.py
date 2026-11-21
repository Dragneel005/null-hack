"""landmarks.json contract (v1). See vision/tasks/prd-vision-stage.md."""
from .landmarks import NAMES


class ContractError(ValueError):
    pass


def validate_landmarks(doc: dict) -> None:
    """Raise ContractError if `doc` does not match the landmarks.json v1 contract."""
    for key in ("version", "fps", "w", "h", "side", "confidence", "frames"):
        if key not in doc:
            raise ContractError(f"missing top-level key: {key}")
    if doc["side"] not in ("left", "right"):
        raise ContractError(f"side must be 'left' or 'right', got {doc['side']!r}")
    if not 0 <= doc["confidence"] <= 1:
        raise ContractError("confidence must be in 0..1")

    prev_t = float("-inf")
    for i, frame in enumerate(doc["frames"]):
        t = frame.get("t")
        if not isinstance(t, (int, float)) or t <= prev_t:
            raise ContractError(f"frame {i}: t must be a number strictly greater than previous")
        prev_t = t
        lm = frame.get("lm", "missing")
        if lm is None:
            continue
        if not isinstance(lm, dict):
            raise ContractError(f"frame {i}: lm must be an object or null")
        missing = [n for n in NAMES if n not in lm]
        if missing:
            raise ContractError(f"frame {i}: missing landmarks {missing[:3]}...")
        for name in NAMES:
            v = lm[name]
            if not (isinstance(v, (list, tuple)) and len(v) == 3):
                raise ContractError(f"frame {i}: {name} must be [x, y, vis]")
