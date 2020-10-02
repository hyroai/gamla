from gamla import functional, transducer
from gamla.functional import reduce
from gamla.functional_generic import compose
from gamla.transducer import transjuxt


def _append_to_iterable(s, current):
    return functional.suffix(current, s)


def _increment(x):
    return x + 1


_incrementing = transducer.mapping(_increment)


def test_mapping():
    assert tuple(reduce(_incrementing(_append_to_iterable), (), [1, 2, 3])) == (2, 3, 4)


def test_composition():
    increment_twice = compose(_incrementing, _incrementing)
    assert tuple(reduce(increment_twice(_append_to_iterable), (), [1, 2, 3])) == (
        3,
        4,
        5,
    )


def test_composition_of_3_functions():
    increment_twice_and_sum = compose(
        _incrementing,
        _incrementing,
        lambda step: lambda s, x: x + s,
    )
    assert reduce(increment_twice_and_sum(_append_to_iterable), 0, [1, 2, 3]) == 12


def test_juxt():
    s1, s2, s3 = reduce(
        transjuxt(
            _incrementing(_append_to_iterable),
            _incrementing(_append_to_iterable),
            lambda s, x: x + s,
        )(lambda s, _: s),
        ((), (), 0),
        [1, 2, 3],
    )
    assert tuple(s1) == (2, 3, 4)
    assert tuple(s2) == (2, 3, 4)
    assert s3 == 6
