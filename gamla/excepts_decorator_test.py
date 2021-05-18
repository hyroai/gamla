import asyncio
import dataclasses

import pytest

from gamla import excepts_decorator, functional, functional_generic

pytestmark = pytest.mark.asyncio


@dataclasses.dataclass(frozen=True)
class SomeException(Exception):
    pass


def test_excepts_sync():
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


def test_try_and_excepts_no_exception():
    assert (
        excepts_decorator.try_and_excepts(
            SomeException,
            functional.just(None),
            functional.identity,
        )(1)
        == 1
    )


def test_try_and_excepts_with_exception():
    assert (
        excepts_decorator.try_and_excepts(
            SomeException,
            functional_generic.compose_left(functional.pack, functional.identity),
            functional.make_raise(SomeException),
        )(1)
        == (SomeException(), 1)
    )
