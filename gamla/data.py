import dataclasses
import itertools
import json
from typing import Callable, Collection, Dict

import dataclasses_json

from gamla import construct, functional_generic, operator
from gamla.optimized import sync


def _immutable(self, *args, **kws):
    raise TypeError("cannot change object - object is immutable")


class frozendict(dict):  # noqa: N801
    def __init__(self, *args, **kwargs):
        self.__setattr__ = _immutable
        super(frozendict, self).__init__(*args, **kwargs)

    def __hash__(self):
        return hash(tuple(self.items()))

    def __gt__(self, other):
        return functional_generic.map_dict(dict.items, operator.identity)(
            self,
        ) > functional_generic.map_dict(dict.items, operator.identity)(other)

    # TODO(nitzo): Disabled since we need to be able to un-serialize with dill/pickle.
    # __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable


def get_encode_config():
    """Display dataclass field as a tuple of JSON strings."""
    return dataclasses.field(
        metadata=dataclasses_json.config(
            encoder=lambda lst: sorted(lst, key=json.dumps, reverse=False),
        ),
    )


def _freeze_nonterminal(v):
    if isinstance(v, Dict):
        return frozendict(v)
    return tuple(v)


#: Freeze recursively a dictionary.
#:
#: >>> freeze_deep({"1": {"2": "345", "some-string": ["hello"]}})
#: data.frozendict(
#:  {"1": data.frozendict({"2": "345", "some-string": ("hello",)})},
#: )
freeze_deep = functional_generic.map_dict(_freeze_nonterminal, operator.identity)


class Enum(frozenset):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


def _do_on_positions(f, predicate: Callable[[int], bool]):
    return sync.compose_left(
        enumerate,
        sync.map(
            sync.ternary(
                sync.compose_left(
                    operator.head,
                    predicate,
                ),
                sync.compose_left(operator.second, f),
                operator.second,
            ),
        ),
    )


def explode(*positions: Collection[int]):
    """Flattens a non homogeneous iterable.

    For an iterable where some positions are iterable and some are not,
    "explodes" the iterable, so that each element appears in a single row, and duplicates the non iterable.

    >>> functional_generic.pipe(
    ...     ["x", ["y1", "y2", "y3"], "z"],
    ...     data.explode(1),
    ...     tuple,
    ... )
    (
        ("x", "y1", "z"),
        ("x", "y2", "z"),
        ("x", "y3", "z"),
    )
    """
    return sync.compose_left(
        _do_on_positions(
            construct.wrap_tuple,
            sync.complement(operator.contains(positions)),
        ),
        sync.star(itertools.product),
    )


def transform_if_not_none(transform: Callable, value):
    """
    Apply a function on a given value if it's not None. Else, return the None value.

    >>> transform_if_not_none(
    ...     functional_generic.when(operator.is_instance, lambda x: x.casefold())),
    ...     "Some Title"
    ... )
    'some title'
    >>> transform_if_not_none(
    ...     functional_generic.when(operator.is_instance, lambda x: x.casefold())),
    ...     None
    ... )

    """
    if value is not None:
        return transform(value)
    return value
