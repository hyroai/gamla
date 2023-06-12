import asyncio
import functools
import zlib
from typing import Any, Callable

import gamla
from gamla import functional, functional_generic, higher_order

_VALUE_TO_CACHE = "something"
_CACHE_SUCCESS_STATE = {"c3aa999f887e4eb8a1dda68862dcf172a78b5d30": _VALUE_TO_CACHE}


def test_prepare_and_apply():
    def increment(x):
        return x + 1

    def decrement(x):
        return x - 1

    def conditional_transformation(x):
        return increment if x < 10 else decrement

    assert higher_order.prepare_and_apply(conditional_transformation)(15) == 14


async def test_prepare_and_apply_async():
    async def increment(x):
        return x + 1

    async def decrement(x):
        return x - 1

    def conditional_transformation(x):
        return increment if x < 10 else decrement

    assert (
        await higher_order.prepare_and_apply_async(conditional_transformation)(15) == 14
    )


def test_ignore_first():
    def increment(x):
        return x + 1

    assert higher_order.ignore_first_arg(increment)("a", 2) == 3


class _CalledTooManyTimes(Exception):
    pass


def assert_max_called(n: int):
    def decorator(f):
        if asyncio.iscoroutinefunction(f):

            @functools.wraps(f)
            async def wrapped(*args, **kwargs):
                wrapped.count += 1
                if wrapped.count > n:
                    raise _CalledTooManyTimes()
                return await f(*args, **kwargs)

            wrapped.count = 0
            return wrapped

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            wrapped.count += 1
            if wrapped.count > n:
                raise _CalledTooManyTimes()
            return f(*args, **kwargs)

        wrapped.count = 0
        return wrapped

    return decorator


def generate_getter_and_setter(
    d: dict,
) -> tuple[Callable[[str], Any], Callable[[str, Any], Any]]:
    def get_item(key: str):
        return d[key]

    def set_item(key: str, value):
        d[key] = value
        return

    return get_item, set_item


def test_persistent_cache():
    d = {}

    @assert_max_called(1)
    def f(x):
        return x

    cached_function = higher_order.persistent_cache(
        *generate_getter_and_setter(d),
        functional.make_hashed_call_key,
        gamla.identity,
        gamla.identity,
        False,
    )(f)

    assert cached_function(_VALUE_TO_CACHE) == _VALUE_TO_CACHE
    assert cached_function(_VALUE_TO_CACHE) == _VALUE_TO_CACHE
    assert d == _CACHE_SUCCESS_STATE


async def test_persistent_cache_async():
    d = {}

    @assert_max_called(1)
    async def f(x):
        return x

    assert (
        await higher_order.persistent_cache(
            *generate_getter_and_setter(d),
            functional.make_hashed_call_key,
            gamla.identity,
            gamla.identity,
            False,
        )(f)(_VALUE_TO_CACHE)
        == _VALUE_TO_CACHE
    )
    assert d == _CACHE_SUCCESS_STATE


def test_persistent_cache_force():
    d = {}

    @assert_max_called(2)
    def f(x):
        return x

    cached_function = higher_order.persistent_cache(
        *generate_getter_and_setter(d),
        functional.make_hashed_call_key,
        gamla.identity,
        gamla.identity,
        True,
    )(f)

    assert cached_function(_VALUE_TO_CACHE) == _VALUE_TO_CACHE
    assert cached_function(_VALUE_TO_CACHE) == _VALUE_TO_CACHE
    assert d == _CACHE_SUCCESS_STATE
    assert f.count == 2


def test_persistent_cache_zip():
    d = {}

    @assert_max_called(1)
    def f(x):
        return x

    assert (
        higher_order.persistent_cache(
            *generate_getter_and_setter(d),
            functional.make_hashed_call_key,
            functional_generic.compose_left(lambda x: x.encode("utf-8"), zlib.compress),
            functional_generic.compose_left(
                zlib.decompress,
                lambda x: x.decode("utf-8"),
            ),
            False,
        )(f)(_VALUE_TO_CACHE)
        == _VALUE_TO_CACHE
    )
    assert d == {
        "c3aa999f887e4eb8a1dda68862dcf172a78b5d30": b"x\x9c+\xce\xcfM-\xc9\xc8\xccK\x07\x00\x13G\x03\xcf",
    }
