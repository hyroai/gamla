from typing import Iterable

import immutables


# Just for typing.
class ImmutableSet(immutables.Map):
    pass


def create(iterable: Iterable) -> ImmutableSet:
    return immutables.Map(map(lambda x: (x, None), iterable))


EMPTY: ImmutableSet = create([])


def add(set: ImmutableSet, element) -> ImmutableSet:
    return set.set(element, None)


def remove(set: ImmutableSet, element) -> ImmutableSet:
    return set.delete(element)


def contains(set: ImmutableSet, element) -> bool:
    return element in set


def union(set1: ImmutableSet, set2: ImmutableSet) -> ImmutableSet:
    smaller, larger = sorted([set1, set2], key=len)
    for element in smaller:
        larger = add(larger, element)
    return larger


def intersection(set1: ImmutableSet, set2: ImmutableSet) -> ImmutableSet:
    smaller, larger = sorted([set1, set2], key=len)
    for element in smaller:
        if element not in larger:
            smaller = remove(smaller, element)
    return smaller
