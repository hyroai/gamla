from typing import Any, AsyncGenerator, Awaitable, Callable, Iterable, Set, Text, Tuple

from gamla import async_functions, currying, functional, functional_generic


@currying.curry
async def agraph_traverse(
    source: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    key: Callable = functional.identity,
) -> AsyncGenerator:
    """Gets a graph and a function to get a node's neighbours,
    BFS over it from a single source node, returns an AsyncGenerator of unique nodes.

    >>> g = {'1': ['2', '3'], '2': ['3'], '3': ['4'], '4': []}
    >>> async def get_item(x):
    >>>     for a in g.get(x):
    >>>         yield a
    >>> async def to_list(ag):
    >>>     return [i async for i in ag]
    >>> gamla.run_sync(to_list(gamla.agraph_traverse('1', get_item)))
    ['1', '2', '3', '4']"""
    async for s in agraph_traverse_many(
        [source],
        aget_neighbors=aget_neighbors,
        key=key,
    ):
        yield s


@currying.curry
async def agraph_traverse_many(
    sources: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    key: Callable = functional.identity,
) -> AsyncGenerator[Any, None]:
    """BFS over a graph, yielding unique nodes.
    Use when `aget_neighbors` returns an AsyncGenerator.

    >>> g = {'1': ['2', '3'], '2': ['3'], '3': ['4'], '4': []}
    >>> async def get_item(x):
    >>>     for a in g.get(x):
    >>>         yield a
    >>> async def to_list(ag):
    >>>     return [i async for i in ag]
    >>> gamla.run_sync(to_list(gamla.agraph_traverse_many(['1', '3'], get_item)))
    ['3', '1', '4', '2']

    Note: `aget_neighbors` must return elements without duplicates."""
    queue = [*sources]
    seen = set(map(key, queue))

    while queue:
        current = queue.pop()
        yield current
        async for node in aget_neighbors(current):
            if key(node) not in seen:
                seen.add(key(node))
                queue = [node] + queue


@currying.curry
async def atraverse_graph_by_radius(
    source: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    radius: int,
    key: Callable = functional.identity,
) -> AsyncGenerator[Any, None]:
    """Gets a graph and a function to get a node's neighbours,
    BFS over it from a single source node, returns an AsyncGenerator of unique nodes.
    Does not traverse farther from given `radius`

    >>> g = {'1': ['2', '3'], '2': ['3'], '3': ['4'], '4': []}
    >>> async def get_item(x):
    >>>     for a in g.get(x):
    >>>         yield a
    >>> async def to_list(ag):
    >>>     return [i async for i in ag]
    >>> gamla.run_sync(to_list(gamla.atraverse_graph_by_radius('1', get_item, 1)))
    ['1', '2', '3']"""

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
        key=functional_generic.compose_left(functional.head, key),
    ):
        yield functional.head(s)


_Node = Any


@currying.curry
async def _async_graph_traverse_many_inner(
    seen: Set[_Node],
    get_neighbors: Callable[[_Node], Awaitable[Iterable[_Node]]],
    process_node: Callable[[_Node], None],
    roots: Iterable[_Node],
):
    assert set(roots).isdisjoint(seen)

    await functional_generic.pipe(
        roots,
        functional_generic.juxt(
            seen.update,
            functional_generic.compose_left(
                functional_generic.curried_map(process_node),
                tuple,
            ),
        ),
        async_functions.to_awaitable,
    )

    await functional_generic.pipe(
        roots,
        functional_generic.mapcat(get_neighbors),
        functional_generic.remove(functional.contains(seen)),
        frozenset,
        functional_generic.unless(
            functional.empty,
            _async_graph_traverse_many_inner(seen, get_neighbors, process_node),
        ),
    )


async def async_graph_traverse_many(
    get_neighbors: Callable[[_Node], Awaitable[Iterable[_Node]]],
    process_node: Callable[[_Node], None],
    roots: Iterable[_Node],
):
    """BFS over a graph, calling mapper on unique nodes during iteration.
    Use when get_neighbors is async.

    >>> graph = {1: (1, 2, 3, 5), 2: (4,), 3: (1, 2)}
    >>> result = []
    >>> await graph_async.async_graph_traverse_many(
    >>>  functional_generic.compose_left(
    >>>     dict_utils.dict_to_getter_with_default((), graph),
    >>>     async_functions.to_awaitable,
    >>>  ),
    >>>  res.append,
    >>>  [1],
    >>> )
    [1, 2, 3, 5, 4]"""

    return await _async_graph_traverse_many_inner(
        set(),
        get_neighbors,
        process_node,
        roots,
    )
