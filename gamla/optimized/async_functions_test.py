import functools
import inspect

import pytest

from gamla.optimized import async_functions


async def test_double_star():
    async def increment(x):
        return x + 1

    assert await async_functions.double_star(increment)({"x": 2}) == 3


async def _async_fn(x):
    return x


def _sync_fn(x):
    return x


@pytest.mark.parametrize(
    "f",
    [_sync_fn, _async_fn, len, functools.partial(_async_fn, 1)],
)
def test_is_coroutine_function_matches_inspect(f):
    assert async_functions.is_coroutine_function(f) == inspect.iscoroutinefunction(f)
