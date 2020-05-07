import asyncio

import pytest
import toolz

from gamla import functional_generic, functional

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def _opposite_async(x):
    await asyncio.sleep(1)
    return not x


def test_do_if():
    assert functional.do_if(lambda _: True, lambda x: 2)(1) == 1


def test_currying():
    @functional_generic.curry
    def f(x, y, z):
        return x + y + z

    assert f(1, 2, 3) == 6
    assert f(1)(2, 3) == 6
    assert f(1, 2)(3) == 6


def test_juxt():
    assert functional_generic.juxt(toolz.identity, lambda x: x + 1)(3) == (3, 4)


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
