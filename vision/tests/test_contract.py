import copy

import pytest

from vision.contract import ContractError, validate_landmarks
from vision.synthetic import generate_pushups

GOOD = generate_pushups(reps=2)


def test_generated_document_is_valid():
    validate_landmarks(GOOD)


def test_frame_without_a_pose_is_allowed():
    doc = copy.deepcopy(GOOD)
    doc["frames"][3]["lm"] = None
    validate_landmarks(doc)


@pytest.mark.parametrize("breakage", [
    lambda d: d.pop("fps"),
    lambda d: d.update(side="up"),
    lambda d: d["frames"][0]["lm"].update(l_elbow=[1, 2]),
    lambda d: d["frames"][0]["lm"].pop("l_hip"),
    lambda d: d["frames"][5].update(t=d["frames"][4]["t"]),
])
def test_rejects_broken_documents(breakage):
    doc = copy.deepcopy(GOOD)
    breakage(doc)
    with pytest.raises(ContractError):
        validate_landmarks(doc)
