import asyncio
import operator
import time

import pytest

from gamla import currying, dict_utils, functional, functional_generic
from gamla.optimized import sync

pytestmark = pytest.mark.asyncio


def _generator():
    yield 1
    yield 2
    yield 5


async def _opposite_async(x):
    await asyncio.sleep(0.01)
    return not x


async def _equals(x, y):
    await asyncio.sleep(0.01)
    return x == y


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
    assert await functional_generic.anymap(_opposite_async)([True, True, False])


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
        sync.check(functional.is_instance(bool), AssertionError),
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


async def test_itemfilter_async_sync_mixed():
    assert (
        await functional_generic.pipe(
            {1: 1, 2: 1, 3: 3},
            functional_generic.itemfilter(functional_generic.star(_equals)),
            functional_generic.itemfilter(sync.star(lambda _, val: val == 1)),
        )
        == {1: 1}
    )


async def test_keymap_async_curried():
    assert await functional_generic.keymap(_opposite_async)({True: True}) == {
        False: True,
    }


async def test_keyfilter_sync_curried():
    assert (
        functional_generic.keyfilter(functional.identity)(
            {False: True, True: False},
        )
        == {True: False}
    )


async def test_valmap_sync_curried():
    assert functional_generic.valmap(operator.not_)({True: True}) == {True: False}


async def test_valfilter_sync_curried():
    assert (
        functional_generic.valfilter(functional.identity)(
            {False: True, True: False},
        )
        == {False: True}
    )


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


def test_wrap_str():
    assert functional_generic.pipe("john", functional.wrap_str("hi {}")) == "hi john"


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
        functional_generic.bifurcate(async_sum, functional.count),
        sync.star(operator.truediv),
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
        await asyncio.sleep(0.1)
        return x + y

    assert await (functional_generic.reduce_curried(slow_addition, 0)([1, 2, 3])) == 6


def test_reduce():
    def addition(x, y):
        return x + y

    assert functional_generic.reduce_curried(addition, 0)([1, 2, 3]) == 6


async def test_scan():
    def addition(x, y):
        return x + y

    assert (functional_generic.scan(addition, 0)([1, 2, 3])) == (
        0,
        1,
        3,
        6,
    )


async def test_scan_async():
    async def slow_addition(x, y):
        await asyncio.sleep(0.01)
        return x + y

    assert await (functional_generic.scan(slow_addition, 0)([1, 2, 3])) == (
        0,
        1,
        3,
        6,
    )


def test_find():
    seq = ({"key": 1}, {"key": 2}, {"key": 3}, {"key": 2})

    assert (
        functional_generic.find(
            functional_generic.compose_left(
                dict_utils.itemgetter("key"),
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
                dict_utils.itemgetter("key"),
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
                dict_utils.itemgetter("key"),
                functional.equals(2),
            ),
        )(iter(seq))
        == 1
    )

    assert (
        functional_generic.find_index(
            functional_generic.compose_left(
                dict_utils.itemgetter("key"),
                functional.equals(4),
            ),
        )(iter(seq))
        == -1
    )


def test_compositions_have_name():
    assert (
        functional_generic.compose_left(
            functional.identity,
            functional.identity,
            functional.unique,
        ).__name__
        == "unique_OF_identity_OF_identity"
    )


def test_async_compositions_have_name():
    async def async_identity(x):
        asyncio.sleep(1)
        return x

    assert (
        functional_generic.compose_left(
            functional.identity,
            async_identity,
            functional.unique,
        ).__name__
        == "unique_OF_async_identity_OF_identity"
    )


def test_attrgetter():
    assert functional.attrgetter("lower")("ASD")() == "asd"


def test_latency():
    start_time = time.time()
    for _ in range(1000):
        functional_generic.pipe(
            True,
            functional_generic.juxt(functional.equals(True), functional.equals(False)),
            functional.head,
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
        functional_generic.merge_with(functional.head)(
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


async def test_async_when():
    async def async_equals_1(x):
        await asyncio.sleep(0.01)
        return x == 1

    assert await functional_generic.when(async_equals_1, functional.just(True))(1)

    assert await functional_generic.when(async_equals_1, functional.just(True))(2) == 2


async def test_unless1():
    assert (
        functional_generic.unless(functional.equals(1), functional.just(True))(1) == 1
    )


async def test_unless2():
    assert functional_generic.unless(functional.equals(1), functional.just(True))(2)


def test_compose_many_to_one():
    assert (
        functional_generic.compose_many_to_one([sum, sum], lambda x, y: x + y)(
            [1, 2, 3],
        )
        == 12
    )


def test_empty_pipe():
    with pytest.raises(functional_generic.PipeNotGivenAnyFunctions):
        functional_generic.pipe(
            [
                1,
                2,
                3,
            ],
        )


def test_add_key_value():
    assert functional.add_key_value("1", "1")({"2": "2"}) == {"1": "1", "2": "2"}


def test_remove_key():
    assert functional.remove_key("1")({"1": 1, "2": 2}) == {"2": 2}


def test_wrap_dict():
    assert functional.wrap_dict("some_key")("some_value") == {"some_key": "some_value"}


def test_groupby():
    names = ["alice", "bob", "charlie", "dan", "edith", "frank"]
    assert functional_generic.groupby(functional.last)(names) == {
        "e": ("alice", "charlie"),
        "b": ("bob",),
        "n": ("dan",),
        "h": ("edith",),
        "k": ("frank",),
    }


def test_groupby_empty():
    assert functional_generic.groupby(functional.last)([]) == {}


def test_take():
    assert tuple(functional.take(2)(["a", "b", "c"])) == ("a", "b")


def test_nth():
    assert functional.nth(1)(["a", "b", "c"]) == "b"


def test_drop():
    assert tuple(functional.drop(2)(["a", "b", "c"])) == ("c",)


def test_drop_last():
    assert tuple(functional.drop_last(1)(["a", "b", "c"])) == ("a", "b")


def test_count_by():
    assert functional_generic.count_by(functional.head)(["aa", "ab", "ac", "bc"]) == {
        "a": 3,
        "b": 1,
    }


def test_add():
    assert functional.add(1)(2) == 3
    assert functional.add(["c"])(["a", "b"]) == ["a", "b", "c"]


def test_assert_that():
    functional.assert_that(functional.equals(2))(2)


def test_assoc_in():
    assert functional.assoc_in({"a": {"b": 1}}, ["a", "b"], 2) == {"a": {"b": 2}}


def test_bottom():
    assert tuple(functional.bottom((3, 2, 1))) == (1, 2, 3)
    assert tuple(functional.bottom((1, 2, 3, 4), lambda x: x % 2 == 0)) == (1, 3, 2, 4)


def test_concat_with():
    assert tuple(functional.concat_with((3, 4), (1, 2))) == (1, 2, 3, 4)


def test_contains():
    assert functional.contains([1, 2, 3])(2)
    assert not functional.contains("David")("x")


def test_side_effect():
    side_result = []
    side_effect = functional_generic.compose_left(
        functional.multiply(2),
        side_result.append,
    )
    assert functional_generic.side_effect(side_effect)(2) == 2
    assert side_result == [4]


async def test_side_effect_async():
    side_result = []
    side_effect = functional_generic.compose_left(
        currying.curry(_equals)(2),
        side_result.append,
    )
    assert await functional_generic.side_effect(side_effect)(2) == 2
    assert side_result == [True]


def test_sliding_window():
    assert list(functional.sliding_window(2)([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_partition_all():
    assert list(functional.partition_all(2)([1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(functional.partition_all(2)([1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_ends_with():
    assert functional.ends_with([1, 2, 3])((0, 1, 2, 3)) is True
    assert functional.ends_with([1, 2, 3])((1, 2)) is False
    assert functional.ends_with([1, 2])((3, 1, 2)) is True
    assert functional.ends_with([1])(()) is False
    assert functional.ends_with([1, 2, 3])((0, 1, 2, 3))
    assert functional.ends_with(range(1, 4))(range(0, 4))
    assert not functional.ends_with([1, 2, 3])((1, 2))
    assert functional.ends_with([1, 2])((3, 1, 2))
    assert not functional.ends_with([1])(())


def test_len_equals():
    assert functional.len_equals(3)(_generator())


def test_not_empty():
    assert functional.empty(_generator()) is False


def test_is_empty():
    assert functional.empty([]) is True


def test_flip():
    assert functional.flip(operator.truediv)(2, 6) == 3.0


def test_nonempty():
    assert functional.nonempty(_generator()) is True


def test_between():
    assert functional.between(1, 100)(30) is True


def test_not_between():
    assert functional.between(2, 5)(14) is False


def test_intersect():
    assert tuple(functional.intersect([[1, 2, 3, 4], [4, 5], [4, 2]])) == (4,)


def test_has_intersection():
    assert functional.have_intersection([(1, 2, 3), (3, 4, 5)]) is True


def test_doesnt_have_intersection():
    assert functional.have_intersection([(1, 2, 3), (4, 5, 6)]) is False


def test_all_n_grams():
    assert set(functional.get_all_n_grams("abc")) == {
        ("a",),
        ("a", "b"),
        (
            "a",
            "b",
            "c",
        ),
        ("b",),
        ("b", "c"),
        ("c",),
    }


def test_all_n_grams_non_textual():
    assert set(functional.get_all_n_grams([1, 2, 3])) == {
        (1,),
        (1, 2),
        (
            1,
            2,
            3,
        ),
        (2,),
        (2, 3),
        (3,),
    }


def test_take_while():
    assert list(functional.take_while(lambda x: x < 7)([1, 2, 9, 2])) == [1, 2]


def test_translate_exception():
    with pytest.raises(ValueError):
        functional_generic.pipe(
            iter([]),
            functional.translate_exception(next, StopIteration, ValueError),
        )
