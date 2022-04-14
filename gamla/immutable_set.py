from typing import Iterable

import immutables

# Design choices:
# - Using a class to allow for typing
# - The class has no methods to ensure all logic is in the functions below.
# - The wrapped map is kept private.
# - To prevent the user from making subtle mistake, we override `__eq__` to raise an error.
# Corollaries:
# - Will not work with operators ootb, e.g. `in`, `==` or `len`.


class ImmutableSet:
    def __init__(self, inner):
        self._inner = inner

    def __eq__(self, _):
        raise NotImplementedError(
            "Use the functions in this module instead of operators.",
        )


def create(iterable: Iterable) -> ImmutableSet:
    return ImmutableSet(immutables.Map(map(lambda x: (x, None), iterable)))


EMPTY: ImmutableSet = create([])


def equals(s1: ImmutableSet, s2: ImmutableSet) -> bool:
    return s1._inner == s2._inner  # noqa: SF01


def length(set: ImmutableSet) -> int:
    return len(set._inner)  # noqa: SF01


def add(set: ImmutableSet, element) -> ImmutableSet:
    return ImmutableSet(set._inner.set(element, None))  # noqa: SF01


def remove(set: ImmutableSet, element) -> ImmutableSet:
    return ImmutableSet(set._inner.delete(element))  # noqa: SF01


def contains(set: ImmutableSet, element) -> bool:
    return element in set._inner  # noqa: SF01


def union(set1: ImmutableSet, set2: ImmutableSet) -> ImmutableSet:
    smaller, larger = sorted([set1, set2], key=length)
    return ImmutableSet(larger._inner.update(smaller._inner))  # noqa: SF01


def intersection(set1: ImmutableSet, set2: ImmutableSet) -> ImmutableSet:
    smaller, larger = sorted([set1, set2], key=length)
    for element in smaller._inner:  # noqa: SF01
        if not contains(larger, element):
            smaller = remove(smaller, element)
    return smaller
