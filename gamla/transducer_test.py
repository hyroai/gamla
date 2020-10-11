from gamla import functional_generic, transducer


def _append_to_tuple(s, current):
    """Highly inefficient, but just for testing purposes."""
    return (*s, current)


_increment = transducer.map(lambda x: x + 1)


def test_map():
    assert transducer.transduce(_increment, _append_to_tuple, [], [1, 2, 3]) == (
        2,
        3,
        4,
    )


def test_composition():
    assert transducer.transduce(
        functional_generic.compose(_increment, _increment),
        _append_to_tuple,
        (),
        [1, 2, 3],
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
            lambda s, _: s,
            0,
            [1, 2, 3],
        )
        == 12
    )


def test_juxt():
    assert (
        transducer.transduce(
            transducer.juxt(
                _increment(_append_to_tuple),
                lambda s, x: x + s,
            ),
            lambda s, _: s,
            ((), 0),
            [1, 2, 3],
        )
        == ((2, 3, 4), 6)
    )


def test_apply_spec():
    assert (
        transducer.transduce(
            transducer.apply_spec(
                {
                    "incremented": _increment(_append_to_tuple),
                    "sum": lambda s, x: x + s,
                },
            ),
            lambda s, _: s,
            {"incremented": (), "sum": 0},
            [1, 2, 3],
        )
        == {"incremented": (2, 3, 4), "sum": 6}
    )


def test_countby():
    assert (
        transducer.transduce(
            transducer.count_by(
                lambda x: ("even",) if x % 2 == 0 else ("odd",),
            ),
            lambda s, _: s,
            {},
            [1, 2, 3, 4, 5],
        )
        == {"even": 2, "odd": 3}
    )
