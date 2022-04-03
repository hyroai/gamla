import itertools
from typing import Any, Callable, Dict, FrozenSet, Iterable, Set, Text, Tuple

from gamla import construct, currying, dict_utils, functional_generic, operator
from gamla.optimized import sync


@currying.curry
def graph_traverse(
    source: Any,
    get_neighbors: Callable,
    key: Callable = operator.identity,
) -> Iterable:
    """Gets a graph and a function to get a node's neighbours, BFS over it from a single source node, return an iterator of unique nodes.

    >>> g = {'1': ['2', '3'], '2': ['3'], '3': ['4'], '4': []}
    >>> list(graph_traverse('1', g.__getitem__))
    ['1', '2', '3', '4']"""
    yield from graph_traverse_many([source], get_neighbors=get_neighbors, key=key)


@currying.curry
def graph_traverse_many(
    sources: Any,
    get_neighbors: Callable,
    key: Callable = operator.identity,
) -> Iterable:
    """Gets a graph and a function to get a node's neighbours, BFS over it starting from multiple sources, return an iterator of unique nodes.

    >>> g = {'1': ['2', '3'], '2': ['3'], '3': ['4'], '4': []}
    >>> list(graph_traverse_many(['1', '3'], g.__getitem__))
    ['3', '1', '4', '2']

    Note: `get_neighbors` must return elements without duplicates."""
    seen_set: Set = set()
    remember = functional_generic.compose_left(key, seen_set.add)
    should_traverse = functional_generic.compose_left(
        key,
        sync.complement(operator.contains(seen_set)),
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
    """Gets a graph, a function to get a node's neighbours, a function to add to the set of seen nodes, and function to know if a node is seen or not.
    BFS over the graph and return an iterator of unique nodes.

    >>> seen_set = set(); key =  len; g = {'one': ['two', 'three'], 'two': ['three'], 'three': ['four'], 'four': []}
    >>> list(general_graph_traverse_many(['one', 'three'], g.__getitem__, functional_generic.compose_left(key, seen_set.add), functional_generic.compose_left(key, sync.complement(operator.contains(seen_set)))))
    ['three', 'one', 'four']

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
    """Traverse over a graph like `graph_traverse`, but up to a given radius.

    >>> g = {'1': ['2', '3'], '2': ['3'], '3': ['4'], '4': []}
    >>> list(traverse_graph_by_radius('1', g.__getitem__, 1))
    ['1', '2', '3']"""

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
        operator.head,
        graph_traverse(source=(source, 0), get_neighbors=get_neighbors_limiting_radius),
    )


#: Gets a sequence of edges and returns a graph made of these edges.
#:
#: >>> graph.edges_to_graph([(1,2), (2, 3), (3, 1), (3, 2)])
#: {1: frozenset({2}), 2: frozenset({3}), 3: frozenset({1, 2})}
edges_to_graph = functional_generic.compose(
    functional_generic.sync.valmap(
        functional_generic.sync.compose(
            frozenset,
            functional_generic.sync.map(operator.second),
        ),
    ),
    sync.groupby(operator.head),
)
#: Gets a graph and returns an iterator of all edges in it.
#:
#: >>> list(graph_to_edges({'1': ['2', '3'], '2': ['3'], '3': ['4'], '4': []}))
#: [('1', '2'), ('1', '3'), ('2', '3'), ('3', '4')]
graph_to_edges = functional_generic.compose_left(
    sync.keymap(construct.wrap_tuple),
    dict.items,
    sync.mapcat(sync.star(itertools.product)),
)

#: Gets a graph and returns the graph with its edges reversed
#:
#: >>> reverse_graph({'1': ['2', '3'], '2': ['3'], '3': ['4'], '4': []})
#: {'2': frozenset({'1'}), '3': frozenset({'1', '2'}), '4': frozenset({'3'})}
reverse_graph = functional_generic.compose_left(
    graph_to_edges,
    functional_generic.curried_map(functional_generic.compose_left(reversed, tuple)),
    edges_to_graph,
)

#: Gets a sequence of nodes (cliques) and returns the bidirectional graph they represent
#:
#: >>> cliques_to_graph([{1, 2}, {3, 4}])
#: {1: frozenset({2}), 2: frozenset({1}), 3: frozenset({4}), 4: frozenset({3})}
cliques_to_graph = functional_generic.compose_left(
    sync.mapcat(lambda clique: itertools.permutations(clique, r=2)),
    edges_to_graph,
)


def get_connectivity_components(graph: Dict) -> Iterable[FrozenSet]:
    """
    Gets a graph and return an iterator of its connectivity components.

    >>> g = cliques_to_graph([{1, 2}, {3, 4}])
    >>> list(get_connectivity_components(g))
    [frozenset({1, 2}), frozenset({3, 4})]

    Note: Graph is assumed to undirected, so each edge must appear both ways."""
    nodes_left = frozenset(graph)
    while nodes_left:
        result = frozenset(
            graph_traverse(
                source=operator.head(nodes_left),
                get_neighbors=functional_generic.compose(
                    sync.filter(operator.contains(nodes_left)),
                    graph.get,
                ),
            ),
        )
        yield result
        nodes_left -= result


@currying.curry
def groupby_many(f: Callable, it: Iterable) -> Dict[Text, Any]:
    """Return a mapping `{y: {x s.t. y in f(x)}}, where x in it. `

    Parameters:
    Key function (gets an object in collection and outputs tuple of keys).
    A Collection.

    Returns a dictionary where key has been computed by the `f` key function.

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
     'k': frozenset({'frank'})}"""
    return functional_generic.pipe(
        it,
        functional_generic.mapcat(
            functional_generic.compose_left(
                lambda element: (f(element), [element]),
                sync.star(itertools.product),
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
    """Gets a graph, returns True if it contains a cycle, False else.

    >>> has_cycle({1: [2], 2: [3], 3: [1]})
    True
    >>> has_cycle({1: [2], 2: [3]})
    False"""
    return functional_generic.pipe(
        graph,
        dict.keys,
        functional_generic.curried_map(
            _has_cycle(
                frozenset(),
                dict_utils.dict_to_getter_with_default((), graph),
                set(),
            ),
        ),
        any,
    )


_non_sources = sync.compose_left(
    dict.values,
    operator.concat,
    frozenset,
)


def find_sources(graph: Dict) -> FrozenSet:
    """Gets a directional graph and returns its sources.

    >>> find_sources({'1': ['2', '3'], '2': ['3'], '3': [], '4': []})
    frozenset({'1', '4'})
    """
    return sync.pipe(
        graph,
        sync.remove(operator.contains(_non_sources(graph))),
        frozenset,
    )
