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


def test_frozendict_serializable():
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


def test_csv_to_json():
    assert data.csv_to_json("data_test_example.csv") == [
        {"name": "David", "age": "23"},
        {"name": "Itay", "age": "26"},
    ]


def test_tuple_of_tuples_to_csv():
    assert (
        data.tuple_of_tuples_to_csv((("name", "age"), ("David", "23"), ("Itay", "26")))
        == "name\tage\nDavid\t23\nItay\t26"
    )


def test_tuple_of_tuples_to_csv_custom_separator():
    assert (
        data.tuple_of_tuples_to_csv(
            (("name", "age"), ("David", "23"), ("Itay", "26")),
            " ",
        )
        == "name age\nDavid 23\nItay 26"
    )
