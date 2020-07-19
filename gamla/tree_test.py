from toolz.curried import operator

from gamla import tree


def test_get_leaves_by_ancestor_predicate():
    fn = tree.get_leaves_by_ancestor_predicate(operator.eq("x"))
    assert fn({"x": {"t": ("1")}}) == ("1",)


def test_get_leaves_by_ancestor_predicate_no_matches():
    fn = tree.get_leaves_by_ancestor_predicate(operator.eq("x"))
    assert fn({"t": {"t": ("1")}}) == ()


def test_filter_leaves():
    fn = tree.filter_leaves(operator.lt(3))
    assert tuple(fn({"t": {"t": (1, 12)}})) == (12,)
