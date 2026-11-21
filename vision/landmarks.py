"""MediaPipe Pose's 33 landmarks, in index order, with contract names (l_/r_ prefixes)."""

NAMES = [
    "nose",
    "l_eye_inner", "l_eye", "l_eye_outer",
    "r_eye_inner", "r_eye", "r_eye_outer",
    "l_ear", "r_ear",
    "l_mouth", "r_mouth",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_pinky", "r_pinky",
    "l_index", "r_index",
    "l_thumb", "r_thumb",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
    "l_heel", "r_heel",
    "l_foot_index", "r_foot_index",
]
assert len(NAMES) == 33

# Joints a side-view push-up analysis relies on (without the l_/r_ prefix).
KEY_JOINTS = ["shoulder", "elbow", "wrist", "hip", "ankle"]

# Limbs drawn in the overlay, per side.
SIDE_BONES = [
    ("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"),
    ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
    ("ankle", "heel"), ("heel", "foot_index"), ("ankle", "foot_index"),
]
