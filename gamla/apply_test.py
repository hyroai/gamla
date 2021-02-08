import pytest

from gamla import apply_utils, asyncio, functional

pytestmark = pytest.mark.asyncio


async def _opposite_async(x):
    await asyncio.sleep(0.01)
    return not x


def test_apply_method():
    class SomeClass:
        x: int

        def __init__(self, x):
            self.x = x

        def add(self, y):
            return self.x + y

    assert apply_utils.apply_method("add", 1)(SomeClass(2)) == 3


async def test_apply_method_async():
    class SomeClass:
        x: int

        def __init__(self, x):
            self.x = x

        async def mult_async(self, y):
            await asyncio.sleep(0.01)
            return self.x * y

    assert await apply_utils.apply_method_async("mult_async", 2)(SomeClass(2)) == 4


def test_apply():
    assert apply_utils.apply(1)(functional.add(2)) == 3


async def test_apply_async():
    assert not await apply_utils.apply_async(True)(_opposite_async)


async def test_apply_fn_with_args():
    assert apply_utils.apply_fn_with_args(lambda x, y: x + y, 1, 3) == 4
