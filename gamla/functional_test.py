import asyncio

import pytest
import toolz
from toolz.curried import operator

from gamla import functional, functional_generic, functional_utils

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def _opposite_async(x):
    await asyncio.sleep(1)
    return not x


def test_do_if():
    assert functional.do_if(lambda _: True, lambda x: 2)(1) == 1


def test_currying():
    @functional_utils.curry
    def f(x, y, z):
        return x + y + z

    assert f(1, 2, 3) == 6
    assert f(1)(2, 3) == 6
    assert f(1, 2)(3) == 6


def test_juxt():
    assert functional_generic.juxt(toolz.identity, lambda x: x + 1)(3) == (3, 4)


def test_juxt_zero_params():
    assert functional_generic.juxt(lambda: 1, lambda: 3)() == (1, 3)


async def test_juxt_zero_params_async():
    async def slow_1():
        await asyncio.sleep(1)
        return 1

    assert await functional_generic.juxt(lambda: 3, slow_1)() == (3, 1)


async def test_juxt_async():
    async def slow_identity(x):
        await asyncio.sleep(1)
        return x

    assert await functional_generic.juxt(toolz.identity, slow_identity)(3) == (3, 3)


def test_anyjuxt():
    assert functional_generic.anyjuxt(lambda x: not x, lambda x: x)(True)


def test_alljuxt():
    assert not functional_generic.alljuxt(lambda x: not x, lambda x: x)(True)


async def test_alljuxt_async():
    assert not await functional_generic.alljuxt(_opposite_async, toolz.identity)(True)


async def test_anyjuxt_async():
    assert await functional_generic.anyjuxt(_opposite_async, toolz.identity)(True)


async def test_anymap():
    assert functional_generic.anymap(_opposite_async, [True, True, False])


async def test_allmap():
    def opposite(x):
        return not x

    assert not functional_generic.allmap(opposite, [True, True, False])


async def test_anymap_async():
    assert await functional_generic.anymap(_opposite_async, [True, True, False])


async def test_allmap_async():
    assert not await functional_generic.allmap(_opposite_async, [True, True, False])


async def test_allmap_in_async_pipe():
    assert not await functional_generic.pipe(
        [True, True, False],
        functional_generic.allmap(_opposite_async),
        # Check that the `pipe` serves a value and not a future.
        functional.check(functional.is_instance(bool), AssertionError),
    )


async def test_anymap_in_pipe():
    assert not functional_generic.pipe(
        [True, True, False], functional_generic.allmap(lambda x: not x),
    )


async def test_itemmap_async_sync_mixed():
    assert await functional_generic.pipe(
        {True: True},
        functional_generic.itemmap(
            functional_generic.compose(tuple, functional_generic.map(_opposite_async)),
        ),
        functional_generic.itemmap(
            functional_generic.compose(tuple, functional_generic.map(lambda x: not x)),
        ),
    ) == {True: True}


async def test_keymap_async_curried():
    assert await functional_generic.keymap(_opposite_async)({True: True}) == {
        False: True,
    }


async def test_valmap_sync_curried():
    assert functional_generic.valmap(lambda x: not x)({True: True}) == {True: False}


async def _is_even_async(x):
    await asyncio.sleep(0.1)
    return x % 2 == 0


async def test_filter_curried_async_sync_mix():
    assert await functional_generic.pipe(
        [1, 2, 3, 4],
        functional_generic.filter(_is_even_async),
        functional_generic.map(lambda x: x + 10),
        tuple,
    ) == (12, 14)


async def test_wrap_str():
    assert toolz.pipe("john", functional.wrap_str("hi {}")) == "hi john"


def test_case_single_predicate():
    assert functional_generic.case_dict({toolz.identity: toolz.identity})(True)


def test_case_multiple_predicates():
    assert not functional_generic.case_dict(
        {operator.not_: toolz.identity, toolz.identity: operator.not_},
    )(True)


def test_case_no_predicate():
    with pytest.raises(functional_generic.NoConditionMatched):
        functional_generic.case_dict(
            {operator.not_: toolz.identity, operator.not_: toolz.identity},
        )(True)


async def test_case_async():
    assert not await functional_generic.case_dict(
        {_opposite_async: toolz.identity, toolz.identity: _opposite_async},
    )(True)


def test_partition_after():
    assert functional.partition_after(lambda x: x == 1, []) == ()
    assert tuple(
        functional.partition_after(lambda x: x == 1, [1, 1, 2, 2, 1, 1, 2, 1, 1, 1]),
    ) == ((1,), (1,), (2, 2, 1), (1,), (2, 1), (1,), (1,))


def test_partition_before():
    assert functional.partition_before(lambda x: x == 1, []) == ()
    assert tuple(
        functional.partition_before(
            lambda x: x == 1, [3, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1],
        ),
    ) == ((3,), (1,), (1, 2, 2), (1,), (1, 2), (1,), (1,), (1,))


async def test_drop_last_while():
    assert tuple(functional.drop_last_while(lambda x: x == 1, [])) == ()
    assert tuple(functional.drop_last_while(lambda x: x == 1, [1])) == ()
    assert tuple(functional.drop_last_while(lambda x: x == 1, [2])) == (2,)
    assert tuple(
        functional.drop_last_while(lambda x: x == 1, [1, 1, 2, 2, 1, 1, 2, 1, 1, 1]),
    ) == (1, 1, 2, 2, 1, 1, 2)


def test_apply_spec():
    assert functional_generic.apply_spec(
        {"identity": toolz.identity, "increment": lambda x: x + 1},
    )(1) == {"identity": 1, "increment": 2}


async def test_apply_spec_async():
    async def async_identity(x):
        await asyncio.sleep(1)
        return x

    assert await functional_generic.apply_spec(
        {"identity": async_identity, "increment": lambda x: x + 1},
    )(1) == {"identity": 1, "increment": 2}


async def test_apply_spec_async_recursive():
    async def async_identity(x):
        await asyncio.sleep(1)
        return x

    f = functional_generic.apply_spec(
        {"identity": {"nested": async_identity}, "increment": lambda x: x + 1},
    )
    assert await f(1) == {"identity": {"nested": 1}, "increment": 2}


async def test_async_bifurcate():
    async def async_sum(x):
        await asyncio.sleep(1)
        return sum(x)

    def gen():
        yield 1
        yield 2
        yield 3

    average = await functional_generic.pipe(
        gen(),
        functional_generic.bifurcate(async_sum, toolz.count),
        functional.star(operator.truediv),
    )

    assert average == 2


def test_average():
    def gen():
        yield 1
        yield 2
        yield 3

    assert functional_generic.average(gen()) == 2


def test_countby_many():
    names = ["alice", "bob", "charlie", "dan", "edith", "frank"]
    assert functional.countby_many(lambda name: (name[0], name[-1]), names) == {
        "a": 1,
        "e": 3,
        "b": 2,
        "c": 1,
        "d": 1,
        "n": 1,
        "h": 1,
        "f": 1,
        "k": 1,
    }
