import asyncio
import time

import pytest

from gamla import functional, tree

pytestmark = pytest.mark.asyncio


def test_get_leaves_by_ancestor_predicate():
    fn = tree.get_leaves_by_ancestor_predicate(functional.equals("x"))
    assert fn({"x": {"t": ("1")}}) == ("1",)


def test_get_leaves_by_ancestor_predicate_integer():
    fn = tree.get_leaves_by_ancestor_predicate(functional.less_than(4))
    assert fn({7: {2: ("bla")}}) == ("bla",)


def test_get_leaves_by_ancestor_predicate_no_matches():
    fn = tree.get_leaves_by_ancestor_predicate(functional.equals("x"))
    assert fn({"t": {"t": ("1")}}) == ()


def test_filter_leaves():
    fn = tree.filter_leaves(functional.greater_than(3))
    assert tuple(fn({"t": {"t": (1, 12)}})) == (12,)


def test_map_reduce_tree_async():
    assert (
        tree.map_reduce_tree(
            functional.second,
            lambda x, y: sum(x) + y,
            lambda x: functional.head(x) + 1,
        )((1, ((2, ()), (3, ()))))
        == 9
    )


async def test_map_reduce_tree():
    wait_time = 0.1

    async def increment(x):
        await asyncio.sleep(wait_time)
        return functional.head(x) + 1

    start_time = time.time()
    assert (
        await tree.map_reduce_tree(
            functional.second,
            lambda x, y: sum(x) + y,
            increment,
        )(
            (1, ((2, ()), (3, ()))),
        )
    ) == 9
    assert time.time() - start_time < wait_time * 1.1
