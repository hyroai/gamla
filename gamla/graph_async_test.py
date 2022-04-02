from gamla import async_functions, dict_utils, functional_generic, graph_async, operator


def _reduce_max(children, current):
    return max(children + (current,))


_get_neighbors = dict_utils.dict_to_getter_with_default(
    (),
    {1: (1, 2, 3, 5), 2: (4,), 3: (1, 2)},
)


async def test_reduce_graph_async1():
    set_instance = set()
    assert (
        await graph_async.reduce_graph_async(
            _reduce_max,
            functional_generic.compose_left(
                _get_neighbors,
                async_functions.to_awaitable,
            ),
            set_instance.add,
            operator.contains(set_instance),
            1,
        )
        == 5
    )


async def test_reduce_graph_async2():
    set_instance = set()
    assert (
        await graph_async.reduce_graph_async(
            lambda children, current: sum(children) + current,
            functional_generic.compose_left(
                dict_utils.dict_to_getter_with_default(
                    (),
                    {1: (1, 2, 3, 5), 2: (4,), 3: (1, 2)},
                ),
                async_functions.to_awaitable,
            ),
            set_instance.add,
            operator.contains(set_instance),
            1,
        )
        == 15
    )


async def test_reduce_graph_async_reducer():
    set_instance = set()
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
            set_instance.add,
            operator.contains(set_instance),
            1,
        )
        == 5
    )
