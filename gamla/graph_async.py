import itertools
from typing import Any, AsyncGenerator, Callable, Text, Tuple

import toolz
from toolz import curried

from gamla import functional, functional_async, functional_generic, graph


@functional_generic.curry
async def agroupby_many(f, it):
    return await functional_async.pipe(
        it,
        functional_async.map(
            functional_async.compose_left(
                functional_generic.pair_with(f),
                functional.star(lambda x, y: (x, [y])),
                functional.star(itertools.product),
            )
        ),
        toolz.concat,
        graph.edges_to_graph,
    )


@functional_generic.curry
async def agraph_traverse(
    source: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    key: Callable = toolz.identity,
) -> AsyncGenerator:
    async for s in agraph_traverse_many(
        [source], aget_neighbors=aget_neighbors, key=key
    ):
        yield s


@functional_generic.curry
async def agraph_traverse_many(
    sources: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    key: Callable = toolz.identity,
) -> AsyncGenerator[Any, None]:
    """BFS over a graph, yielding unique nodes.

    Note: `get_neighbors` must return elements without duplicates."""
    queue = [*sources]
    seen = set(map(key, queue))

    while queue:
        current = queue.pop()
        yield current
        async for node in aget_neighbors(current):
            if key(node) not in seen:
                seen.add(key(node))
                queue = [node] + queue


@functional_generic.curry
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
