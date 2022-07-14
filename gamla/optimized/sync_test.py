from gamla import operator
from gamla.optimized import sync


def test_valfilter_sync():
    assert sync.valfilter(operator.identity)(
        {False: True, True: False},
    ) == {False: True}


def test_pair_right_sync():
    assert sync.pair_right(operator.multiply(2))(4) == (4, 8)


def test_flatten():
    assert sync.flatten([1, "a", frozenset({"something"}), (("hi", 6),)]) == (
        1,
        "a",
        "something",
        "hi",
        6,
    )
