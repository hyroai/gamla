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
    assert index("h")("i") == frozenset()
