from gamla.optimized import async_functions


async def test_double_star():
    async def increment(x):
        return x + 1

    assert await async_functions.double_star(increment)({"x": 2}) == 3
