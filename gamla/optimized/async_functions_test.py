from gamla.optimized import async_functions


def test_double_star():
    async_functions.double_star(lambda x: x + 1)({"x": 2}) == 3
