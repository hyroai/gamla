"""Builds an index with an arbitrary amount of steps from an iterable."""
from typing import Callable, Dict, Iterable, Tuple

from gamla import functional_generic, operator

NestedIndex = Tuple[dict, int]

_index_dict = operator.head
_num_of_steps = operator.second


def _return_after_n_calls(n: int, value) -> Callable:
    if n == 0:
        return value

    def return_after_n_calls(_):
        return _return_after_n_calls(n - 1, value)

    return return_after_n_calls


def _dict_to_getter_with_default_recursive(default, num_of_steps: int, d: Dict):
    def dict_to_getter_with_default_recursive(key):
        if key in d:
            if isinstance(d[key], dict):
                return _dict_to_getter_with_default_recursive(
                    default,
                    num_of_steps - 1,
                    d[key],
                )
            else:
                return d[key]
        else:
            return _return_after_n_calls(num_of_steps - 1, default)

    return dict_to_getter_with_default_recursive


def _make_index_dict(steps):
    steps = tuple(steps)
    if not steps:
        return frozenset
    return functional_generic.compose_left(
        operator.head(steps),
        functional_generic.valmap(_make_index_dict(steps[1:])),
    )


def build(steps: Iterable, it) -> NestedIndex:
    steps = tuple(steps)
    return _make_index_dict(steps)(it), len(steps)


def to_query(index: NestedIndex) -> Callable:
    return _dict_to_getter_with_default_recursive(
        frozenset(),
        _num_of_steps(index),
        _index_dict(index),
    )
