from gamla import functional, tree


def test_get_leaves_by_ancestor_predicate():
    fn = tree.get_leaves_by_ancestor_predicate(functional.equals("x"))
    assert fn({"x": {"t": ("1")}}) == ("1",)


def test_get_leaves_by_ancestor_predicate_integer():
    fn = tree.get_leaves_by_ancestor_predicate(functional.less_than(4))
    assert fn({7: {2: ("bla")}}) == ("bla",)


def test_get_leaves_by_ancestor_predicate_no_matches():
    fn = tree.get_leaves_by_ancestor_predicate(functional.equals("x"))
    assert fn({"t": {"t": ("1")}}) == ()


def test_filter_leaves():
    fn = tree.filter_leaves(functional.greater_than(3))
    assert tuple(fn({"t": {"t": (1, 12)}})) == (12,)
