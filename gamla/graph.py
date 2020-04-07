import itertools
from typing import Any, Callable, Dict, FrozenSet, Iterable, Text, Tuple

import toolz
from toolz import curried
from toolz.curried import operator

from gamla import functional


@toolz.curry
def graph_traverse(
    source: Any, get_neighbors: Callable, key: Callable = toolz.identity
) -> Iterable:
    yield from graph_traverse_many([source], get_neighbors=get_neighbors, key=key)


@toolz.curry
def graph_traverse_many(
    sources: Any, get_neighbors: Callable, key: Callable = toolz.identity
) -> Iterable:
    """BFS over a graph, yielding unique nodes.

    Note: `get_neighbors` must return elements without duplicates."""
    seen = set()
    queue = [*sources]
    while queue:
        current = queue.pop()
        yield current
        seen.add(key(current))
        for node in get_neighbors(current):
            if key(node) not in seen:
                queue = [node] + queue


def traverse_graph_by_radius(
    source: Any, get_neighbors: Callable, radius: int
) -> Iterable:
    """Like `graph_traverse`, but does not traverse farther from given `radius`."""

    def get_neighbors_limiting_radius(
        current_and_distance: Tuple[Text, int]
    ) -> Iterable[Tuple[Text, int]]:
        current, distance = current_and_distance
        if distance < radius:
            yield from map(
                lambda neighbor: (neighbor, distance + 1), get_neighbors(current)
            )

    return map(
        toolz.first,
        graph_traverse(source=(source, 0), get_neighbors=get_neighbors_limiting_radius),
    )


edges_to_graph = toolz.compose(
    curried.valmap(toolz.compose(frozenset, curried.map(toolz.second))),
    curried.groupby(toolz.first),
)

graph_to_edges = toolz.compose_left(
    curried.keymap(lambda x: (x,)),
    dict.items,
    curried.mapcat(functional.star(itertools.product)),
)

reverse_graph = toolz.compose_left(
    graph_to_edges, curried.map(toolz.compose_left(reversed, tuple)), edges_to_graph
)


cliques_to_graph = toolz.compose_left(
    curried.mapcat(lambda clique: itertools.permutations(clique, r=2)), edges_to_graph
)


def get_connectivity_components(graph: Dict) -> Iterable[FrozenSet]:
    """Graph is assumed to undirected, so each edge must appear both ways."""
    nodes_left = frozenset(graph)
    while nodes_left:
        result = frozenset(
            graph_traverse(
                source=toolz.first(nodes_left),
                get_neighbors=toolz.compose(
                    curried.filter(operator.contains(nodes_left)), graph.get
                ),
            )
        )
        yield result
        nodes_left -= result


@toolz.curry
def groupby_many(f, it):
    return toolz.pipe(
        it,
        curried.mapcat(
            toolz.compose_left(
                lambda element: (f(element), [element]),
                functional.star(itertools.product),
            )
        ),
        edges_to_graph,
    )


@toolz.curry
def groupby_many_reduce(key, reducer, seq):
    result = {}
    for element in seq:
        for key_result in key(element):
            result[key_result] = reducer(result.get(key_result, None), element)
    return result


@toolz.curry
def _has_cycle(sourced, get_neighbors, visited, node):
    if node in sourced:
        return True
    if node in visited:
        return False
    visited.add(node)
    return toolz.pipe(
        node,
        get_neighbors,
        functional.anymap(_has_cycle(sourced | {node}, get_neighbors, visited)),
    )


def has_cycle(graph):
    return toolz.pipe(
        graph,
        dict.keys,
        curried.map(_has_cycle(frozenset(), curried.get(seq=graph, default=()), set())),
        any,
    )
