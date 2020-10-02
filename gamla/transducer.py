import functools
from typing import Callable, Iterable, TypeVar

from gamla import currying, functional, functional_generic

_S = TypeVar("_S")
_X = TypeVar("_X")
Reducer = Callable[[_S, _X], _S]
Transducer = Callable[[Reducer], Reducer]


def _transform_nth(n: int, f: Reducer):
    @currying.curry
    def transform_nth(step, state, x):
        new_state_specific = f(state[n], x)
        new_state = (*state[:n], new_state_specific, *state[n + 1 :])
        final_state = step(new_state, x)
        return final_state

    return transform_nth


def transjuxt(*funcs: Iterable[Transducer]) -> Transducer:
    return functional_generic.pipe(
        funcs,
        enumerate,
        functional_generic.map(functional.star(_transform_nth)),
        functional.star(functional_generic.compose),
    )


@currying.curry
def mapping(f, step: Reducer):
    return lambda s, current: step(s, f(current))


@currying.curry
def filtering(f, step: Reducer):
    return lambda s, x: step(s, x) if f(x) else s


def catting(step: Reducer):
    return lambda s, x: functools.reduce(step, x, s)


def mapcatting(f):
    return functional_generic.compose_left(mapping(f), catting)
