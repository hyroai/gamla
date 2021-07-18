import pytest

import gamla
from gamla import async_functions, dict_utils, functional_generic, graph_async

pytestmark = pytest.mark.asyncio


async def test_async_graph_traverse_many():
    graph = {1: (1, 2, 3, 5), 2: (4,), 3: (1, 2)}
    res = []
    await graph_async.async_graph_traverse_many(
        functional_generic.compose_left(
            dict_utils.dict_to_getter_with_default((), graph),
            async_functions.to_awaitable,
        ),
        res.append,
        [1],
    )

    assert sorted(res) == [1, 2, 3, 4, 5]


async def test_async_graph_traverse_many_with_async_process_node():
    graph = {1: (1, 2, 3, 5), 2: (4,), 3: (1, 2)}
    res = []

    await graph_async.async_graph_traverse_many(
        functional_generic.compose_left(
            dict_utils.dict_to_getter_with_default((), graph),
            async_functions.to_awaitable,
        ),
        gamla.compose_left(res.append, gamla.to_awaitable),
        [1],
    )

    assert sorted(res) == [1, 2, 3, 4, 5]
