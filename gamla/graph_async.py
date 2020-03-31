import itertools
from typing import Any, AsyncGenerator, Callable, Text, Tuple

import toolz
from toolz import curried

from gamla import functional, functional_async, graph


@toolz.curry
async def agroupby_many(f, it):
    return await functional_async.apipe(
        it,
        functional_async.amap(
            functional_async.acompose_left(
                functional_async.apair_with(f),
                functional.star(lambda x, y: (x, [y])),
                functional.star(itertools.product),
            )
        ),
        toolz.concat,
        graph.edges_to_graph,
    )


@toolz.curry
async def agraph_traverse(
    source: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    key: Callable = toolz.identity,
) -> AsyncGenerator:
    async for s in agraph_traverse_many(
        [source], aget_neighbors=aget_neighbors, key=key
    ):
        yield s


@toolz.curry
async def agraph_traverse_many(
    sources: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    key: Callable = toolz.identity,
) -> AsyncGenerator[Any, None]:
    """BFS over a graph, yielding unique nodes.

    Note: `get_neighbors` must return elements without duplicates."""
    seen = set()
    queue = [*sources]
    while queue:
        current = queue.pop()
        yield current
        seen.add(key(current))

        async for node in aget_neighbors(current):
            if key(node) not in seen:
                queue = [node] + queue


@toolz.curry
async def atraverse_graph_by_radius(
    source: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    radius: int,
    key: Callable = toolz.identity,
) -> AsyncGenerator[Any, None]:
    """Like `agraph_traverse`, but does not traverse farther from given `radius`."""

    async def get_neighbors_limiting_radius(
        current_and_distance: Tuple[Text, int]
    ) -> AsyncGenerator[Tuple[Any, int], None]:
        current, distance = current_and_distance
        if distance < radius:
            async for neighbor in aget_neighbors(current):
                yield neighbor, distance + 1

    async for s in agraph_traverse(
        source=(source, 0),
        aget_neighbors=get_neighbors_limiting_radius,
        key=curried.compose_left(toolz.first, key),
    ):
        yield toolz.first(s)
