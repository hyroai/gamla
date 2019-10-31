import itertools
from typing import Any, Callable, Dict, FrozenSet, Iterable, Text, Tuple

import toolz
from toolz import curried
from toolz.curried import operator


@toolz.curry
def graph_traverse(source: Any, get_neighbors: Callable) -> FrozenSet:
    """BFS over a graph, yielding unique nodes."""
    seen = set()
    queue = [source]
    while queue:
        current = queue.pop()
        seen.add(current)
        for node in get_neighbors(current):
            if node not in seen:
                queue = [node] + queue
    return frozenset(seen)


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


def cliques_to_graph(cliques: Iterable[Iterable]) -> Dict[Any, FrozenSet[Any]]:
    return toolz.pipe(
        cliques,
        curried.mapcat(lambda clique: itertools.permutations(clique, r=2)),
        edges_to_graph,
    )


def get_connectivity_components(graph: Dict) -> Iterable[FrozenSet]:
    """Graph is assumed to undirected, so each edge must appear both ways."""
    nodes_left = frozenset(graph)
    while nodes_left:
        result = graph_traverse(
            source=toolz.first(nodes_left),
            get_neighbors=toolz.compose(
                curried.filter(operator.contains(nodes_left)), graph.get
            ),
        )
        yield result
        nodes_left -= result
