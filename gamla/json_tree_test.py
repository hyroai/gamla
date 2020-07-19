from toolz.curried import operator

from gamla import json_tree


def test_get_leafs_by_ancestor_predicate():
    fn = json_tree.get_leafs_by_ancestor_predicate(operator.eq("x"))
    assert fn({"x": {"t": ("1")}}) == ("1",)


def test_get_leafs_by_ancestor_predicate_no_matches():
    fn = json_tree.get_leafs_by_ancestor_predicate(operator.eq("x"))
    assert fn({"t": {"t": ("1")}}) == ()
