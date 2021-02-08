import dataclasses
from typing import Any, Callable

import toolz
from toolz import curried

from gamla import currying, dict_utils, functional, functional_generic


@currying.curry
def _tree_reduce(get_children, reduce_fn, tree_node):
    return reduce_fn(
        tree_node,
        map(_tree_reduce(get_children, reduce_fn), get_children(tree_node)),
    )


@dataclasses.dataclass(frozen=True)
class KeyValue:
    key: Any
    value: Any


_is_terminal = functional_generic.anyjuxt(
    functional.is_instance(str),
    functional.is_instance(int),
    functional.is_instance(float),
)


def _get_children(element):
    return functional_generic.case_dict(
        {
            _is_terminal: functional.just(()),
            functional.is_instance(tuple): functional.identity,
            functional.is_instance(list): functional.identity,
            functional.is_instance(dict): functional_generic.compose_left(
                dict.items,
                functional.curried_map_sync(functional.star(KeyValue)),
            ),
            functional.is_instance(KeyValue): functional_generic.compose_left(
                lambda x: x.value,
                functional_generic.ternary(
                    _is_terminal,
                    functional.wrap_tuple,
                    _get_children,
                ),
            ),
        },
    )(element)


_MATCHED = "matched"
_UNMATCHED = "unmatched"
_get_matched = dict_utils.itemgetter(_MATCHED)
_get_unmatched = dict_utils.itemgetter(_UNMATCHED)


def _make_matched_unmatched(matched, unmatched):
    return {_MATCHED: matched, _UNMATCHED: unmatched}


_merge_children_as_matched = functional_generic.compose_left(
    curried.mapcat(functional_generic.juxtcat(_get_matched, _get_unmatched)),
    tuple,
    functional_generic.pair_right(functional.just(())),
    functional.star(_make_matched_unmatched),
)

_merge_children = functional_generic.compose_left(
    functional_generic.bifurcate(
        functional_generic.compose_left(curried.mapcat(_get_matched), tuple),
        functional_generic.compose_left(curried.mapcat(_get_unmatched), tuple),
    ),
    functional.star(_make_matched_unmatched),
)


@currying.curry
def _get_anywhere_reducer(predicate: Callable, node, children):
    if _is_terminal(node):
        return _make_matched_unmatched((), (node,))
    if isinstance(node, KeyValue) and predicate(node.key):
        return _merge_children_as_matched(children)
    return _merge_children(children)


def get_leaves_by_ancestor_predicate(predicate: Callable):
    """Gets a predicate, and builds a function that gets a dictionary, potentially nested and returns an iterable of leaf values.
    The values returned are of leafs where some ancestor (possibly indirect) passes the predicate.

    >>> gamla.pipe({"x": {"y": (1, 2, 3)}}, gamla.get_leaves_by_ancestor_predicate(gamla.equals("x")), tuple)
    (1, 2, 3)
    >>> gamla.pipe({"x": {"y": (1, 2, 3)}}, gamla.get_leaves_by_ancestor_predicate(gamla.equals("z")), tuple)
    ()

    Useful for retrieving values from large json objects, where the exact path is unimportant.
    """
    return functional_generic.compose_left(
        _tree_reduce(_get_children, _get_anywhere_reducer(predicate)),
        _get_matched,
    )


@currying.curry
def _filter_leaves_reducer(predicate, node, children):
    if _is_terminal(node) and predicate(node):
        return (node,)
    return toolz.concat(children)


def filter_leaves(predicate: Callable):
    """Gets a predicate, and builds a function that gets a dictionary, potentially nested and returns an iterable of leaf values.
    The values returned are of leafs that pass the predicate.

    >>> gamla.pipe({"x": {"y": (1, 2, 3)}}, gamla.filter_leaves(gamla.greater_than(2)), tuple)
    (3,)

    Useful for retrieving values from large json objects, where the exact path is unimportant.
    """
    return _tree_reduce(_get_children, _filter_leaves_reducer(predicate))


#: Reduce a JSON like tree.
json_tree_reduce = _tree_reduce(_get_children)
