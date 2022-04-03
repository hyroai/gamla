from typing import Any, AsyncGenerator, Callable, Hashable, Text, Tuple

from gamla import construct, currying, functional_generic, operator


@currying.curry
async def agraph_traverse(
    source: Any,
    aget_neighbors: Callable[[Any], AsyncGenerator],
    key: Callable = operator.identity,
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
    key: Callable = operator.identity,
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
    key: Callable = operator.identity,
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
        current_and_distance: Tuple[Text, int],
    ) -> AsyncGenerator[Tuple[Any, int], None]:
        current, distance = current_and_distance
        if distance < radius:
            async for neighbor in aget_neighbors(current):
                yield neighbor, distance + 1

    async for s in agraph_traverse(
        source=(source, 0),
        aget_neighbors=get_neighbors_limiting_radius,
        key=functional_generic.compose_left(operator.head, key),
    ):
        yield operator.head(s)


class _IgnoreChild:
    pass


@currying.curry
async def reduce_graph_async(
    reducer: Callable[[Tuple[Any, ...], Hashable], Any],
    get_neighbors: Callable,
    remember: Callable[[Hashable], None],
    is_seen: Callable[[Hashable], bool],
    current: Hashable,
):
    """Reduces a graph from some starting point using async functions.

    >>> set_instance = set()
    >>> await reduce_graph_async(
    ...     lambda children, current: sum(children) + current,
    ...     functional_generic.compose_left(
    ...         dict_utils.dict_to_getter_with_default(
    ...             (),
    ...             {1: (1, 2, 3, 5), 2: (4,), 3: (1, 2)}),
    ...         async_functions.to_awaitable,
    ...     ),
    ...     set_instance.add,
    ...     contains(set_instance)
    ...     1,
    ... )
    15"""
    if is_seen(current):
        return _IgnoreChild()
    # Since we may reach a node from two different branches, at the same time,
    # we have to broadcast to the other branch that we've reached this node, this
    # can't be done in an immutable fashion.
    remember(current)
    return await functional_generic.pipe(
        current,
        get_neighbors,
        functional_generic.curried_map(
            reduce_graph_async(reducer, get_neighbors, remember, is_seen),
        ),
        functional_generic.remove(operator.is_instance(_IgnoreChild)),
        tuple,
        functional_generic.pair_right(construct.just(current)),
        functional_generic.star(reducer),
    )
