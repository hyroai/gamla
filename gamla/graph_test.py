from gamla import graph


def test_get_connectivity_components():
    assert list(
        graph.get_connectivity_components({1: [2], 2: [1, 3], 3: [2], 4: [5], 5: [4]})
    ) == [frozenset({1, 2, 3}), frozenset({4, 5})]


def test_cliques_to_graph():
    assert graph.cliques_to_graph([{1, 2}, {2, 3}, {7, 8}]) == {
        1: frozenset({2}),
        2: frozenset({1, 3}),
        3: frozenset({2}),
        7: frozenset({8}),
        8: frozenset({7}),
    }


def test_cycles():
    assert graph.has_cycle({1: [1]})
    assert graph.has_cycle({1: [2], 2: [3], 3: [1]})
    assert graph.has_cycle({1: [2], 2: [3], 3: [4], 4: [2]})
    assert not graph.has_cycle({1: [2]})
    assert not graph.has_cycle({1: [2], 2: [3]})
