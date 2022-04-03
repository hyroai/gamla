"""https://www.youtube.com/watch?v=6mTbuzafcII for some explanations about this is about."""

import functools
import itertools
from typing import Any, Callable, Iterable, TypeVar

from gamla import construct, dict_utils, functional_generic, operator
from gamla.optimized import sync

_S = TypeVar("_S")
_X = TypeVar("_X")
Reducer = Callable[[_S, _X], _S]
Transducer = Callable[[Reducer], Reducer]


def transduce(transformation: Transducer, step: Reducer, initial, collection):
    """Reduces over a `collection` using a `step` function to get elements and a `transformation` logic.

    See other transducer functions for usage examples."""
    return functools.reduce(transformation(step), collection, initial)


def _transform_by_key(injector):
    return lambda key, reducer: lambda step: lambda s, x: step(
        injector(s, key, reducer(s[key], x)),
        x,
    )


def _replace_index(the_tuple, index, value):
    return (*the_tuple[:index], value, *the_tuple[index + 1 :])


#: Combines transducers in a `dict` into a transducer that produces a `dict`.
#: >>> transducer.transduce(
#: ...     transducer.apply_spec({  # This will combine the inner stuff into one new transducer.
#: ...         "incremented": _increment(_append_to_tuple),  # This is a transducer.
#: ...         "sum": lambda s, x: x + s,  # This is another transducer.
#: ...     }),
#: ...     lambda s, _: s,
#: ...     {"incremented": (), "sum": 0},
#: ...     [1, 2, 3],
#: ... )
#: {"incremented": (2, 3, 4), "sum": 6}
apply_spec = functional_generic.compose_left(
    dict.items,
    sync.map(
        sync.star(_transform_by_key(lambda x, y, z: dict_utils.add_key_value(y, z)(x))),
    ),
    sync.star(functional_generic.compose),
)

#: Combines transducers in a `tuple` into a transducer that produces a `tuple`.
#: >>> transducer.transduce(
#: ...     transducer.juxt(  # This will combine the inner stuff into one new transducer.
#: ...         _increment(_append_to_tuple),  # This is a transducer.
#: ...         lambda s, x: x + s,  # This is another transducer.
#: ...     ),
#: ...    lambda s, _: s,
#: ...    [(), 0],
#: ...    [1, 2, 3],
#: ... )
#: ((2, 3, 4), 6)
juxt = functional_generic.compose_left(
    operator.pack,
    enumerate,
    sync.map(sync.star(_transform_by_key(_replace_index))),
    sync.star(functional_generic.compose),
)


def map(f: Callable[[Any], Any]):
    """
    Transducer version of `map`.

    >>> transducer.transduce(
        transducer.map(lambda x: x + 1),
        lambda previous, current: (*previous, current),
        (),
        [1, 2, 3],
    )
    (3, 4, 5)
    """
    return lambda step: lambda s, current: step(s, f(current))


def filter(f: Callable[[Any], bool]):
    """
    Transducer version of `filter`.

    >>> transducer.transduce(
        transducer.filter(lambda x: x > 2),
        lambda previous, current: (*previous, current),
        [],
        [1, 2, 3]
    )
    (3,)
    """
    return lambda step: lambda s, x: step(s, x) if f(x) else s


def concat(step: Reducer):
    """Flattens collections using the `step` function into a single collection."""
    return lambda s, x: functools.reduce(step, x, s)


def mapcat(f: Callable[[Any], Iterable[Any]]):
    """Maps over a collection, then flattens the results into a single collection."""
    return functional_generic.compose(map(f), concat)


def groupby(key: Callable[[Any], Any], reducer: Reducer, initial):
    """Like `groupby_many`, just with a key function that returns a single element."""
    return groupby_many(
        functional_generic.compose_left(key, construct.wrap_tuple),
        reducer,
        initial,
    )


def groupby_many(keys: Callable[[Any], Iterable], reducer: Reducer, initial):
    """Given a `keys` function, that maps an element into multiple keys, transduces the collection into a dictionary of key to group of matching elements.

    >>> transducer.transduce(
        transducer.groupby_many(
            lambda x: ("even",) if x % 2 == 0 else ("odd",),
            lambda s, x: (*s, x),
            (),
        ),
        lambda s, _: s,
        {},
        [1, 2, 3, 4, 5],
    )
    {"even": (2, 4), "odd": (1, 3, 5)}
    """
    return functional_generic.compose(
        mapcat(
            functional_generic.compose_left(
                functional_generic.juxt(keys, construct.wrap_tuple),
                sync.star(itertools.product),
            ),
        ),
        lambda step: lambda s, x: step(
            dict_utils.add_key_value(x[0], reducer(s.get(x[0], initial), x[1]))(s),
            x,
        ),
    )


def count_by(keys: Callable[[Any], Iterable]):
    """Given a `keys` function, that maps an element into multiple keys, transduces the collection into a dictionary of key to count.

    >>> transducer.transduce(
        transducer.count_by(
            lambda x: ("even",) if x % 2 == 0 else ("odd",),
        ),
        lambda s, _: s,
        {},
        [1, 2, 3, 4, 5],
    )
    {"even": 2, "odd": 3}
    """
    return groupby_many(keys, lambda s, _: s + 1, 0)
