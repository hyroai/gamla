import asyncio
import dataclasses

from gamla import construct, excepts_decorator, functional, functional_generic, operator


@dataclasses.dataclass(frozen=True)
class _SomeError(Exception):
    pass


def test_excepts_sync():
    assert (
        excepts_decorator.excepts(
            _SomeError,
            construct.just(None),
            operator.identity,
        )(1)
        == 1
    )
    assert (
        excepts_decorator.excepts(
            _SomeError,
            construct.just(None),
            functional.make_raise(_SomeError),
        )(1)
        is None
    )


async def test_excepts_async():
    async def async_raise(x):
        raise _SomeError

    async def slow_identity(x):
        await asyncio.sleep(0.01)
        return x

    assert (
        await excepts_decorator.excepts(
            _SomeError,
            construct.just(None),
            slow_identity,
        )(1)
        == 1
    )
    assert (
        await excepts_decorator.excepts(
            _SomeError,
            construct.just(None),
            async_raise,
        )(1)
        is None
    )


def test_try_and_excepts_no_exception():
    assert (
        excepts_decorator.try_and_excepts(
            _SomeError,
            construct.just(None),
            operator.identity,
        )(1)
        == 1
    )


def test_try_and_excepts_with_exception():
    assert excepts_decorator.try_and_excepts(
        _SomeError,
        functional_generic.compose_left(operator.pack, operator.identity),
        functional.make_raise(_SomeError),
    )(1) == (_SomeError(), 1)
