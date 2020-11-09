import asyncio
import operator
import time

import pytest
import toolz

from gamla import currying, functional, functional_generic

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def _opposite_async(x):
    await asyncio.sleep(0.01)
    return not x


def test_do_if():
    assert functional.do_if(functional.just(True), functional.just(2))(1) == 1


def test_currying():
    @currying.curry
    def f(x, y, z):
        return x + y + z

    assert f(1, 2, 3) == 6
    assert f(1)(2, 3) == 6
    assert f(1, 2)(3) == 6


def test_juxt():
    assert functional_generic.juxt(functional.identity, functional.add(1))(3) == (3, 4)


def test_juxt_zero_params():
    assert functional_generic.juxt(lambda: 1, lambda: 3)() == (1, 3)


async def test_juxt_zero_params_async():
    async def slow_1():
        await asyncio.sleep(0.01)
        return 1

    assert await functional_generic.juxt(lambda: 3, slow_1)() == (3, 1)


async def test_juxt_async():
    async def slow_identity(x):
        await asyncio.sleep(0.01)
        return x

    assert await functional_generic.juxt(functional.identity, slow_identity)(3) == (
        3,
        3,
    )


def test_anyjuxt():
    assert functional_generic.anyjuxt(operator.not_, functional.identity)(True)


def test_alljuxt():
    assert not functional_generic.alljuxt(operator.not_, functional.identity)(True)


async def test_alljuxt_async():
    assert not await functional_generic.alljuxt(_opposite_async, functional.identity)(
        True,
    )


async def test_anyjuxt_async():
    assert await functional_generic.anyjuxt(_opposite_async, functional.identity)(True)


async def test_anymap():
    assert functional_generic.anymap(_opposite_async)([True, True, False])


async def test_allmap():
    def opposite(x):
        return not x

    assert not functional_generic.allmap(opposite)([True, True, False])


async def test_anymap_async():
    assert await functional_generic.anymap(_opposite_async)([True, True, False])


async def test_allmap_async():
    assert not await functional_generic.allmap(_opposite_async)([True, True, False])


async def test_allmap_in_async_pipe():
    assert not await functional_generic.pipe(
        [True, True, False],
        functional_generic.allmap(_opposite_async),
        # Check that the `pipe` serves a value and not a future.
        functional_generic.check(functional.is_instance(bool), AssertionError),
    )


async def test_anymap_in_pipe():
    assert not functional_generic.pipe(
        [True, True, False],
        functional_generic.allmap(operator.not_),
    )


async def test_itemmap_async_sync_mixed():
    assert (
        await functional_generic.pipe(
            {True: True},
            functional_generic.itemmap(
                functional_generic.compose(
                    tuple,
                    functional_generic.curried_map(_opposite_async),
                ),
            ),
            functional_generic.itemmap(
                functional_generic.compose(
                    tuple,
                    functional_generic.curried_map(operator.not_),
                ),
            ),
        )
        == {True: True}
    )


async def test_keymap_async_curried():
    assert await functional_generic.keymap(_opposite_async)({True: True}) == {
        False: True,
    }


async def test_valmap_sync_curried():
    assert functional_generic.valmap(operator.not_)({True: True}) == {True: False}


async def _is_even_async(x):
    await asyncio.sleep(0.1)
    return x % 2 == 0


async def test_filter_curried_async_sync_mix():
    assert (
        await functional_generic.pipe(
            [1, 2, 3, 4],
            functional_generic.curried_filter(_is_even_async),
            functional_generic.curried_map(functional.add(10)),
            tuple,
        )
        == (12, 14)
    )


async def test_wrap_str():
    assert toolz.pipe("john", functional.wrap_str("hi {}")) == "hi john"


def test_case_single_predicate():
    assert functional_generic.case_dict({functional.identity: functional.identity})(
        True,
    )


def test_case_multiple_predicates():
    assert not functional_generic.case_dict(
        {operator.not_: functional.identity, functional.identity: operator.not_},
    )(True)


def test_case_no_predicate():
    with pytest.raises(functional_generic.NoConditionMatched):
        functional_generic.case_dict(
            {
                operator.not_: functional.identity,
                # Can't repeat keys.
                lambda x: not x: functional.identity,
            },
        )(True)


async def test_case_async():
    assert not await functional_generic.case_dict(
        {_opposite_async: functional.identity, functional.identity: _opposite_async},
    )(True)


def test_partition_after():
    assert functional.partition_after(functional.equals(1), []) == ()
    assert (
        tuple(
            functional.partition_after(
                functional.equals(1),
                [1, 1, 2, 2, 1, 1, 2, 1, 1, 1],
            ),
        )
        == ((1,), (1,), (2, 2, 1), (1,), (2, 1), (1,), (1,))
    )


def test_partition_before():
    assert functional.partition_before(functional.equals(1), []) == ()
    assert (
        tuple(
            functional.partition_before(
                functional.equals(1),
                [3, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1],
            ),
        )
        == ((3,), (1,), (1, 2, 2), (1,), (1, 2), (1,), (1,), (1,))
    )


async def test_drop_last_while():
    assert tuple(functional.drop_last_while(functional.equals(1), [])) == ()
    assert tuple(functional.drop_last_while(functional.equals(1), [1])) == ()
    assert tuple(functional.drop_last_while(functional.equals(1), [2])) == (2,)
    assert (
        tuple(
            functional.drop_last_while(
                functional.equals(1),
                [1, 1, 2, 2, 1, 1, 2, 1, 1, 1],
            ),
        )
        == (1, 1, 2, 2, 1, 1, 2)
    )


def test_apply_spec():
    assert (
        functional_generic.apply_spec(
            {"identity": functional.identity, "increment": functional.add(1)},
        )(1)
        == {"identity": 1, "increment": 2}
    )


async def test_apply_spec_async():
    async def async_identity(x):
        await asyncio.sleep(0.01)
        return x

    assert (
        await functional_generic.apply_spec(
            {"identity": async_identity, "increment": functional.add(1)},
        )(1)
        == {"identity": 1, "increment": 2}
    )


async def test_apply_spec_async_recursive():
    async def async_identity(x):
        await asyncio.sleep(0.01)
        return x

    f = functional_generic.apply_spec(
        {"identity": {"nested": async_identity}, "increment": functional.add(1)},
    )
    assert await f(1) == {"identity": {"nested": 1}, "increment": 2}


async def test_async_bifurcate():
    async def async_sum(x):
        await asyncio.sleep(0.01)
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
    assert functional_generic.countby_many(lambda name: (name[0], name[-1]))(names) == {
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


async def test_reduce_async():
    async def slow_addition(x, y):
        asyncio.sleep(0.1)
        return x + y

    assert await (functional_generic.reduce_curried(slow_addition, 0)([1, 2, 3])) == 6


def test_reduce_aync():
    def addition(x, y):
        asyncio.sleep(0.01)
        return x + y

    assert functional_generic.reduce_curried(addition, 0)([1, 2, 3]) == 6


def test_excepts_sync():
    class SomeException(Exception):
        pass

    assert (
        functional_generic.excepts(
            SomeException,
            functional.just(None),
            functional.identity,
        )(1)
        == 1
    )
    assert (
        functional_generic.excepts(
            SomeException,
            functional.just(None),
            functional.make_raise(SomeException),
        )(1)
        is None
    )


async def test_excepts_async():
    class SomeException(Exception):
        pass

    async def slow_raise(x):
        raise SomeException

    async def slow_identity(x):
        await asyncio.sleep(0.01)
        return x

    assert (
        await functional_generic.excepts(
            SomeException,
            functional.just(None),
            slow_identity,
        )(1)
        == 1
    )
    assert (
        await functional_generic.excepts(
            SomeException,
            functional.just(None),
            slow_raise,
        )(1)
        is None
    )


def test_find():
    seq = ({"key": 1}, {"key": 2}, {"key": 3}, {"key": 2})

    assert (
        functional_generic.find(
            functional_generic.compose_left(
                functional.itemgetter("key"),
                functional.equals(2),
            ),
        )(
            iter(seq),
        )
        == {"key": 2}
    )

    assert (
        functional_generic.find(
            functional_generic.compose_left(
                functional.itemgetter("key"),
                functional.equals(4),
            ),
        )(iter(seq))
        is None
    )


def test_find_index():
    seq = ({"key": 1}, {"key": 2}, {"key": 3}, {"key": 2})

    assert (
        functional_generic.find_index(
            functional_generic.compose_left(
                functional.itemgetter("key"),
                functional.equals(2),
            ),
        )(iter(seq))
        == 1
    )

    assert (
        functional_generic.find_index(
            functional_generic.compose_left(
                functional.itemgetter("key"),
                functional.equals(4),
            ),
        )(iter(seq))
        == -1
    )


def test_take_while():
    seq = ({"key": 1}, {"key": 2}, {"key": 3}, {"key": 2})

    assert (
        tuple(
            functional.take_while(
                functional_generic.compose_left(
                    functional.itemgetter("key"),
                    functional.not_equals(3),
                ),
                iter(seq),
            ),
        )
        == ({"key": 1}, {"key": 2})
    )

    assert (
        tuple(
            functional.take_while(
                functional_generic.compose_left(
                    functional.itemgetter("key"),
                    functional.not_equals(4),
                ),
                iter(seq),
            ),
        )
        == ({"key": 1}, {"key": 2}, {"key": 3}, {"key": 2})
    )


def test_take_last_while():
    seq = ({"key": 1}, {"key": 2}, {"key": 3}, {"key": 2}, {"key": 4}, {"key": 5})

    assert (
        tuple(
            functional.take_last_while(
                functional_generic.compose_left(
                    functional.itemgetter("key"),
                    functional.not_equals(2),
                ),
                iter(seq),
            ),
        )
        == ({"key": 4}, {"key": 5})
    )

    assert (
        tuple(
            functional.take_last_while(
                functional_generic.compose_left(
                    functional.itemgetter("key"),
                    functional.not_equals(6),
                ),
                iter(seq),
            ),
        )
        == ({"key": 1}, {"key": 2}, {"key": 3}, {"key": 2}, {"key": 4}, {"key": 5})
    )


def test_compositions_have_name():
    assert (
        functional_generic.compose_left(
            functional.identity,
            functional.identity,
            toolz.unique,
        ).__name__
        == "unique_of_identity_of_identity"
    )


def test_async_compositions_have_name():
    async def async_identity(x):
        asyncio.sleep(1)
        return x

    assert (
        functional_generic.compose_left(
            functional.identity,
            async_identity,
            toolz.unique,
        ).__name__
        == "unique_of_async_identity_of_identity"
    )


def test_attrgetter():
    assert functional.attrgetter("lower")("ASD")() == "asd"


def test_itemgetter():
    assert functional.itemgetter("a")({"a": 1}) == 1


def test_itemgetter_with_default():
    assert functional.itemgetter_with_default(2, "b")({"a": 1}) == 2


def test_itemgetter_or_none():
    assert functional.itemgetter_or_none("b")({"a": 1}) is None


def test_latency():
    start_time = time.time()
    for _ in range(1000):
        functional_generic.pipe(
            True,
            functional_generic.juxt(functional.equals(True), functional.equals(False)),
            toolz.first,
            functional.just("bla"),
            functional.attrgetter("lower"),
        )
    assert time.time() - start_time < 0.1


def test_unique_by():
    assert (
        tuple(
            functional.unique_by(lambda x: x[0])(["a", "ab", "abc", "bc", "c"]),
        )
        == ("a", "bc", "c")
    )
    assert tuple(functional.unique(["a", "a", "a", "bc", "a"])) == ("a", "bc")


def test_get_in():
    assert functional.get_in(["a", "b", "c", 1])({"a": {"b": {"c": [0, 1, 2]}}}) == 1


def test_get_in_or_none():
    assert (
        functional.get_in_or_none(["a", "b", "d", 1])({"a": {"b": {"c": [0, 1, 2]}}})
        is None
    )


def test_get_in_or_none_uncurried():
    assert (
        functional.get_in_or_none_uncurried(
            ["a", "b", "c", 1],
            {"a": {"b": {"c": [0, 1, 2]}}},
        )
        == 1
    )


def test_merge():
    assert (
        functional_generic.merge(
            {"1": 1, "2": 2},
            {"2": 3, "3": 3},
        )
        == {"1": 1, "2": 3, "3": 3}
    )


def test_merge_with():
    assert (
        functional_generic.merge_with(toolz.first)(
            {"1": 1, "2": 2},
            {"2": 3, "3": 3},
        )
        == {"1": 1, "2": 2, "3": 3}
    )


async def test_async_merge_with():
    async def async_first(x):
        await asyncio.sleep(0.01)
        return x[0]

    assert (
        await functional_generic.merge_with(async_first)(
            {"1": 1, "2": 2},
            {"2": 3, "3": 3},
        )
        == {"1": 1, "2": 2, "3": 3}
    )
