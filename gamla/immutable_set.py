from typing import Iterable

import immutables


# Just for typing.
class ImmutableSet:
    pass


def create(iterable: Iterable) -> ImmutableSet:
    return immutables.Map(map(lambda x: (x, None), iterable))


EMPTY: ImmutableSet = create([])


def length(set: ImmutableSet) -> int:
    return len(set)  # type: ignore


def add(set: ImmutableSet, element) -> ImmutableSet:
    return set.set(element, None)  # type: ignore


def remove(set: ImmutableSet, element) -> ImmutableSet:
    return set.delete(element)  # type: ignore


def contains(set: ImmutableSet, element) -> bool:
    return element in set  # type: ignore


def union(set1: ImmutableSet, set2: ImmutableSet) -> ImmutableSet:
    smaller, larger = sorted([set1, set2], key=length)
    return larger.update(smaller)  # type: ignore


def intersection(set1: ImmutableSet, set2: ImmutableSet) -> ImmutableSet:
    smaller, larger = sorted([set1, set2], key=length)
    for element in smaller:  # type: ignore
        if not contains(larger, element):
            smaller = remove(smaller, element)
    return smaller
