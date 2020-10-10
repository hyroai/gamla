from gamla import functional, functional_generic, transducer


def _append_to_iterable(s, current):
    return functional.suffix(current, s)


_increment = transducer.map(lambda x: x + 1)


def test_map():
    assert tuple(
        transducer.transduce(_increment, _append_to_iterable, [], [1, 2, 3]),
    ) == (
        2,
        3,
        4,
    )


def test_composition():
    assert tuple(
        transducer.transduce(
            functional_generic.compose(_increment, _increment),
            _append_to_iterable,
            (),
            [1, 2, 3],
        ),
    ) == (
        3,
        4,
        5,
    )


def test_composition_of_3_functions():
    assert (
        transducer.transduce(
            functional_generic.compose(
                _increment,
                _increment,
                lambda _: lambda s, x: x + s,
            ),
            _append_to_iterable,
            0,
            [1, 2, 3],
        )
        == 12
    )


def test_juxt():
    s1, s2, s3 = transducer.transduce(
        transducer.juxt(
            _increment(_append_to_iterable),
            _increment(_append_to_iterable),
            lambda s, x: x + s,
        ),
        lambda s, _: s,
        ((), (), 0),
        [1, 2, 3],
    )
    assert tuple(s1) == (2, 3, 4)
    assert tuple(s2) == (2, 3, 4)
    assert s3 == 6


def test_groupby():
    assert (
        transducer.transduce(
            transducer.groupby(lambda x: x % 2 == 0, lambda s, _: s + 1, 0),
            lambda s, _: s,
            {},
            [1, 2, 3, 4, 5],
        )
        == {True: 2, False: 3}
    )
