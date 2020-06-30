import frozendict

from gamla import data


def test_freeze_deep():
    frozen = data.freeze_deep({"1": {"2": "345"}})
    # Check values are intact.
    assert frozen == frozendict.frozendict({"1": frozendict.frozendict({"2": "345"})})
    # Check hashability.
    {data.freeze_deep({"1": {"2": "345"}})}
