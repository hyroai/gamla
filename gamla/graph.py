import itertools
from typing import Any, Callable, Dict, FrozenSet, Iterable, Set, Text, Tuple

import toolz
from toolz import curried

from gamla import currying, functional, functional_generic


@currying.curry
def graph_traverse(
    source: Any,
    get_neighbors: Callable,
    key: Callable = functional.identity,
) -> Iterable:
    yield from graph_traverse_many([source], get_neighbors=get_neighbors, key=key)


@currying.curry
def graph_traverse_many(
    sources: Any,
    get_neighbors: Callable,
    key: Callable = functional.identity,
) -> Iterable:
    """BFS over a graph, yielding unique nodes.
    Note: `get_neighbors` must return elements without duplicates."""
    seen_set: Set = set()
    remember = functional_generic.compose_left(key, seen_set.add)
    should_traverse = functional_generic.compose_left(
        key,
        functional_generic.complement(functional.contains(seen_set)),
    )
    yield from general_graph_traverse_many(
        sources,
        get_neighbors,
        remember,
        should_traverse,
    )


@currying.curry
def general_graph_traverse_many(
    sources: Any,
    get_neighbors: Callable,
    remember: Callable,
    should_traverse: Callable,
) -> Iterable:
    """BFS over a graph, yielding unique nodes.
    Note: `get_neighbors` must return elements without duplicates."""

    queue = [*sources]
    for element in queue:
        remember(element)

    while queue:
        current = queue.pop()
        yield current
        for node in get_neighbors(current):
            if should_traverse(node):
                remember(node)
                queue = [node] + queue


def traverse_graph_by_radius(
    source: Any,
    get_neighbors: Callable,
    radius: int,
) -> Iterable:
    """Like `graph_traverse`, but does not traverse farther from given `radius`."""

    def get_neighbors_limiting_radius(
        current_and_distance: Tuple[Text, int],
    ) -> Iterable[Tuple[Text, int]]:
        current, distance = current_and_distance
        if distance < radius:
            yield from map(
                lambda neighbor: (neighbor, distance + 1),
                get_neighbors(current),
            )

    return map(
        toolz.first,
        graph_traverse(source=(source, 0), get_neighbors=get_neighbors_limiting_radius),
    )


edges_to_graph = functional_generic.compose(
    functional_generic.valmap(
        functional_generic.compose(
            frozenset,
            functional_generic.curried_map(toolz.second),
        ),
    ),
    curried.groupby(toolz.first),
)

graph_to_edges = functional_generic.compose_left(
    functional_generic.keymap(functional.wrap_tuple),
    dict.items,
    curried.mapcat(functional.star(itertools.product)),
)

reverse_graph = functional_generic.compose_left(
    graph_to_edges,
    functional_generic.curried_map(functional_generic.compose_left(reversed, tuple)),
    edges_to_graph,
)

cliques_to_graph = functional_generic.compose_left(
    curried.mapcat(lambda clique: itertools.permutations(clique, r=2)),
    edges_to_graph,
)


def get_connectivity_components(graph: Dict) -> Iterable[FrozenSet]:
    """Graph is assumed to undirected, so each edge must appear both ways."""
    nodes_left = frozenset(graph)
    while nodes_left:
        result = frozenset(
            graph_traverse(
                source=toolz.first(nodes_left),
                get_neighbors=functional_generic.compose(
                    functional_generic.curried_filter(functional.contains(nodes_left)),
                    graph.get,
                ),
            ),
        )
        yield result
        nodes_left -= result


@currying.curry
def groupby_many(f, it):
    """Return a mapping `{y: {x s.t. y in f(x)}}, where x in it. `

    Parameters:
    f (Callable): Key function (given object in collection outputs tuple of keys).
    it (Iterable): Collection.

    Returns:
    Dict[Text, Any]: Dictionary where key has been computed by the `f` key function.

    >>> names = ['alice', 'bob', 'charlie', 'dan', 'edith', 'frank']
    >>> groupby_many(lambda name: (name[0], name[-1]), names)
    {'a': frozenset({'alice'}),
     'e': frozenset({'alice', 'charlie', 'edith'}),
     'b': frozenset({'bob'}),
     'c': frozenset({'charlie'}),
     'd': frozenset({'dan'}),
     'n': frozenset({'dan'}),
     'h': frozenset({'edith'}),
     'f': frozenset({'frank'}),
     'k': frozenset({'frank'})}
    """
    return functional_generic.pipe(
        it,
        curried.mapcat(
            functional_generic.compose_left(
                lambda element: (f(element), [element]),
                functional.star(itertools.product),
            ),
        ),
        edges_to_graph,
    )


@currying.curry
def _has_cycle(sourced, get_neighbors, visited, node):
    if node in sourced:
        return True
    if node in visited:
        return False
    visited.add(node)
    return functional_generic.pipe(
        node,
        get_neighbors,
        functional_generic.anymap(_has_cycle(sourced | {node}, get_neighbors, visited)),
    )


def has_cycle(graph):
    return functional_generic.pipe(
        graph,
        dict.keys,
        functional_generic.curried_map(
            _has_cycle(
                frozenset(),
                lambda node: functional.itemgetter_with_default((), node)(graph),
                set(),
            ),
        ),
        any,
    )
