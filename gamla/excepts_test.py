import asyncio

from gamla import excepts, functional


def test_excepts_sync():
    class SomeException(Exception):
        pass

    assert (
        excepts.excepts(
            SomeException,
            functional.just(None),
            functional.identity,
        )(1)
        == 1
    )
    assert (
        excepts.excepts(
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
        await excepts.excepts(
            SomeException,
            functional.just(None),
            slow_identity,
        )(1)
        == 1
    )
    assert (
        await excepts.excepts(
            SomeException,
            functional.just(None),
            slow_raise,
        )(1)
        is None
    )
