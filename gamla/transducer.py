import functools
import itertools
from typing import Any, Callable, Iterable, TypeVar

import toolz
from toolz import curried

from gamla import functional, functional_generic

_S = TypeVar("_S")
_X = TypeVar("_X")
Reducer = Callable[[_S, _X], _S]
Transducer = Callable[[Reducer], Reducer]


def transduce(transformation, step, initial, collection):
    return functools.reduce(transformation(step), collection, initial)


def _transform_by_key(injector):
    return lambda key, reducer: lambda step: lambda s, x: step(
        injector(s, key, reducer(s[key], x)),
        x,
    )


def _replace_index(the_tuple, index, value):
    return (*the_tuple[:index], value, *the_tuple[index + 1 :])


apply_spec = functional_generic.compose_left(
    dict.items,
    curried.map(functional.star(_transform_by_key(toolz.assoc))),
    functional.star(functional_generic.compose),
)


juxt = functional_generic.compose_left(
    functional.pack,
    enumerate,
    curried.map(functional.star(_transform_by_key(_replace_index))),
    functional.star(functional_generic.compose),
)


def map(f: Callable[[Any], Any]):
    return lambda step: lambda s, current: step(s, f(current))


def filter(f: Callable[[Any], bool]):
    return lambda step: lambda s, x: step(s, x) if f(x) else s


def concat(step: Reducer):
    return lambda s, x: functools.reduce(step, x, s)


def mapcat(f: Callable[[Any], Iterable[Any]]):
    return functional_generic.compose(map(f), concat)


def groupby(key: Callable[[Any], Any], reducer: Reducer, initial):
    return groupby_many(
        functional_generic.compose_left(key, functional.wrap_tuple),
        reducer,
        initial,
    )


def groupby_many(keys: Callable[[Any], Iterable], reducer: Reducer, initial):
    return functional_generic.compose(
        mapcat(
            functional_generic.compose_left(
                functional_generic.juxt(keys, functional.wrap_tuple),
                functional.star(itertools.product),
            ),
        ),
        lambda step: lambda s, x: step(
            toolz.assoc(s, x[0], reducer(s.get(x[0], initial), x[1])),
            x,
        ),
    )


def count_by(keys: Callable[[Any], Iterable]):
    return groupby_many(keys, lambda s, _: s + 1, 0)
