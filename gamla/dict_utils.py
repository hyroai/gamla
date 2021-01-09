import functools
import operator
from typing import Any, Callable, Dict, Iterable

from gamla import currying, functional, functional_generic


def itemgetter(attr):
    def itemgetter(obj):
        return obj[attr]

    return itemgetter


def itemgetter_or_none(attr):
    def itemgetter_or_none(obj):
        return obj.get(attr, None)

    return itemgetter_or_none


def itemgetter_with_default(default, attr):
    def itemgetter_with_default(d):
        try:
            return d[attr]
        except (KeyError, IndexError):
            return default

    return itemgetter_with_default


def get_or_identity(d):
    def get_or_identity(x):
        return d.get(x, x)

    return get_or_identity


def get_in(keys):
    def get_in(coll):
        return functools.reduce(operator.getitem, keys, coll)

    return get_in


def get_in_with_default(keys, default):
    getter = get_in(keys)

    def get_in_with_default(x):
        try:
            return getter(x)
        except (KeyError, IndexError, TypeError):
            return default

    return get_in_with_default


def get_in_or_none(keys):
    return get_in_with_default(keys, None)


def get_in_or_none_uncurried(keys, coll):
    return get_in_or_none(keys)(coll)


@currying.curry
def dict_to_getter_with_default(default, d: Dict):
    """Turns a dictionary into a function from key to value or default if key is not there.

    >>> dict_to_getter_with_default(None, {1:1})(1)
    1
    >>> dict_to_getter_with_default(None, {1:1})(2)
    None
    """

    def dict_to_getter_with_default(key):
        return d.get(key, default)

    return dict_to_getter_with_default


def _return_after_n_calls(n, value):
    if n == 0:
        return value

    def return_after_n_calls(_):
        return _return_after_n_calls(n - 1, value)

    return return_after_n_calls


def make_index(
    steps: Iterable[Callable[[Iterable], Dict]],
) -> Callable[[Iterable], Any]:
    """Builds an index with arbitrary amount of steps from an iterable.

    >>> index = dict_utils.make_index(map(gamla.groupby, [gamla.head, gamla.second]))(["uri", "dani"])
    >>> index("d")("a")
    frozenset(["dani"])
    """
    steps = tuple(steps)
    if not steps:
        return frozenset
    return functional_generic.compose_left(
        functional.head(steps),
        functional_generic.valmap(make_index(steps[1:])),
        lambda d: lambda x: d.get(
            x,
            _return_after_n_calls(len(steps) - 1, frozenset()),
        ),
    )
