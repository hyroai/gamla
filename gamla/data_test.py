import dataclasses
import pickle
from typing import Any

from gamla import data


def test_freeze_deep():
    original = {"1": {"2": "345", "some-string": ["hello"]}}
    frozen = data.freeze_deep(original)
    # Check values are intact.
    assert frozen == data.frozendict(
        {"1": data.frozendict({"2": "345", "some-string": ("hello",)})},
    )
    # Check hashability.
    {data.freeze_deep(original)}


def test_fozendict_serializable():
    fd = data.frozendict({"a": "something", "b": 1})
    fd_str = pickle.dumps(fd)

    fd_clone = pickle.loads(fd_str)

    assert fd == fd_clone


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
