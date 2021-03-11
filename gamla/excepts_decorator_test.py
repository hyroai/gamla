import asyncio

import pytest

from gamla import excepts_decorator, functional

pytestmark = pytest.mark.asyncio


def test_excepts_sync():
    class SomeException(Exception):
        pass

    assert (
        excepts_decorator.excepts(
            SomeException,
            functional.just(None),
            functional.identity,
        )(1)
        == 1
    )
    assert (
        excepts_decorator.excepts(
            SomeException,
            functional.just(None),
            functional.make_raise(SomeException),
        )(1)
        is None
    )


async def test_excepts_async():
    class SomeException(Exception):
        pass

    async def async_raise(x):
        raise SomeException

    async def slow_identity(x):
        await asyncio.sleep(0.01)
        return x

    assert (
        await excepts_decorator.excepts(
            SomeException,
            functional.just(None),
            slow_identity,
        )(1)
        == 1
    )
    assert (
        await excepts_decorator.excepts(
            SomeException,
            functional.just(None),
            async_raise,
        )(1)
        is None
    )
