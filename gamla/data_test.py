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
