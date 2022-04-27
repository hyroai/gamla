from gamla import functional_generic, hierarchical_index, operator


def test_index():
    index = functional_generic.pipe(
        hierarchical_index.build(
            map(functional_generic.groupby, [operator.head, operator.second]),
            ["uri", "dani"],
        ),
        hierarchical_index.to_query,
    )
    assert index("d")("a") == frozenset(["dani"])
    assert index("u")("r") == frozenset(["uri"])
    assert index("u")("x") == frozenset()
    assert index("h")("i") == frozenset()


def test_three_level_index():
    three_level_index = functional_generic.pipe(
        hierarchical_index.build(
            map(
                functional_generic.groupby,
                [operator.head, operator.second, operator.nth(2)],
            ),
            ["uri", "dani"],
        ),
        hierarchical_index.to_query,
    )
    assert three_level_index("d")("a")("n") == frozenset(["dani"])
    assert three_level_index("u")("r")("i") == frozenset(["uri"])
    assert three_level_index("u")("r")("x") == frozenset()
    assert three_level_index("u")("y")("x") == frozenset()
    assert three_level_index("h")("i")("n") == frozenset()
