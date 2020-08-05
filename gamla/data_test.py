import dataclasses
from typing import Any

import frozendict

from gamla import data


def test_freeze_deep():
    original = {"1": {"2": "345", "some-string": ["hello"]}}
    frozen = data.freeze_deep(original)
    # Check values are intact.
    assert frozen == frozendict.frozendict(
        {"1": frozendict.frozendict({"2": "345", "some-string": ("hello",)})},
    )
    # Check hashability.
    {data.freeze_deep(original)}


def test_unfreeze_deep():
    original = data.freeze_deep({"1": {"2": "345", "some-string": ["hello"]}})
    unfrozen = data.unfreeze_deep(original)

    assert unfrozen == {"1": {"2": "345", "some-string": ("hello",)}}


@dataclasses.dataclass(frozen=True)
class MockDataclassA:
    a: int


@dataclasses.dataclass(frozen=True)
class MockDataclassB:
    b: MockDataclassA


def test_match_false():
    assert not data.match(MockDataclassB(MockDataclassA(5)))(
        MockDataclassB(MockDataclassA(4)),
    )


def test_match_true_deep():
    assert data.match(MockDataclassB(MockDataclassA(Any)))(
        MockDataclassB(MockDataclassA(4)),
    )


def test_match_true_shallow():
    assert data.match(MockDataclassB(Any))(MockDataclassB(MockDataclassA(4)))
