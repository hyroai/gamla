import dataclasses
import functools
import itertools
import json
from typing import Any, Callable, Collection, Dict

import dataclasses_json

from gamla import construct, currying, functional_generic, operator
from gamla.optimized import sync


def _immutable(self, *args, **kws):
    raise TypeError("cannot change object - object is immutable")


class frozendict(dict):  # noqa: N801
    def __init__(self, *args, **kwargs):
        self.__setattr__ = _immutable
        super(frozendict, self).__init__(*args, **kwargs)

    @functools.cached_property
    def _hash(self):
        # Caching matters: a frozendict is re-hashed heavily (set/dict membership
        # during graph composition) and each recompute recurses through nested
        # frozendicts. cached_property fills __dict__ lazily on first access, so it
        # neither runs at construction (values may be unhashable yet never hashed)
        # nor goes stale when dill/pickle repopulates items after construction.
        return hash(tuple(self.items()))

    def __hash__(self):
        # hash() looks __hash__ up on the type and calls it, so __hash__ itself
        # can't be the cached_property — delegate to the cached helper.
        return self._hash

    def __getstate__(self):
        # The cached hash must not be serialized: str hashes are randomized per
        # interpreter (PYTHONHASHSEED), so a cache computed in one process is
        # wrong in another — an unpickled frozendict would stop matching an
        # equal one in dict/set lookups. Dropped here, recomputed on first use.
        state = self.__dict__.copy()
        state.pop("_hash", None)
        return state

    def __gt__(self, other):
        return functional_generic.map_dict(dict.items, operator.identity)(
            self,
        ) > functional_generic.map_dict(dict.items, operator.identity)(other)

    def __repr__(self) -> str:
        return super().__repr__()

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
#: {"1": {"2": "345", "some-string": ("hello",)}}
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


@currying.curry
def transform_if_not_none(transform: Callable, value: Any) -> Any:
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
