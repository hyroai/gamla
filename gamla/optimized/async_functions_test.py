from gamla.optimized import async_functions


async def test_double_star():
    await async_functions.double_star(lambda x: x + 1)({"x": 2}) == 3
