import pytest

from gamla import async_functions, dict_utils, functional_generic, graph_async

pytestmark = pytest.mark.asyncio


def _reduce_max(children, current):
    return max(children + (current,))


_get_neighbors = dict_utils.dict_to_getter_with_default(
    (),
    {1: (1, 2, 3, 5), 2: (4,), 3: (1, 2)},
)


async def test_reduce_graph_async():
    assert (
        await graph_async.reduce_graph_async(
            _reduce_max,
            functional_generic.compose_left(
                _get_neighbors,
                async_functions.to_awaitable,
            ),
            set(),
            1,
        )
        == 5
    )


async def test_reduce_graph_async_reducer():
    assert (
        await graph_async.reduce_graph_async(
            functional_generic.compose_left(
                _reduce_max,
                async_functions.to_awaitable,
            ),
            functional_generic.compose_left(
                _get_neighbors,
                async_functions.to_awaitable,
            ),
            set(),
            1,
        )
        == 5
    )
