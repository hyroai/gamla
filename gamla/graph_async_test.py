import pytest

from gamla import async_functions, dict_utils, functional_generic, graph_async

pytestmark = pytest.mark.asyncio


async def test_reduce_graph_async():
    assert (
        await graph_async.reduce_graph_async(
            lambda children, current: max(children + (current,)),
            functional_generic.compose_left(
                dict_utils.dict_to_getter_with_default(
                    (),
                    {1: (1, 2, 3, 5), 2: (4,), 3: (1, 2)},
                ),
                async_functions.to_awaitable,
            ),
            set(),
            1,
        )
        == 5
    )
