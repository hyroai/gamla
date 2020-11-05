import dataclasses
from typing import Any, Callable

import toolz
from toolz import curried

from gamla import currying, functional, functional_generic


@currying.curry
def _tree_reduce(get_children, reduce_fn, tree_node):
    return reduce_fn(
        tree_node,
        map(_tree_reduce(get_children, reduce_fn), get_children(tree_node)),
    )


@dataclasses.dataclass(frozen=True)
class _KeyValue:
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
                functional.curried_map_sync(functional.star(_KeyValue)),
            ),
            functional.is_instance(_KeyValue): functional_generic.compose_left(
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
_get_matched = functional.itemgetter(_MATCHED)
_get_unmatched = functional.itemgetter(_UNMATCHED)


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
    if isinstance(node, str):
        return _make_matched_unmatched((), (node,))
    if isinstance(node, _KeyValue) and predicate(node.key):
        return _merge_children_as_matched(children)
    return _merge_children(children)


def get_leaves_by_ancestor_predicate(predicate: Callable):
    """Gets leafs with ancestor nodes passing the predicate."""
    return functional_generic.compose_left(
        _tree_reduce(_get_children, _get_anywhere_reducer(predicate)),
        _get_matched,
    )


@currying.curry
def _filter_leaves_reducer(predicate, node, children):
    if _is_terminal(node) and predicate(node):
        return (node,)
    return toolz.concat(children)


def filter_leaves(predicate):
    return _tree_reduce(_get_children, _filter_leaves_reducer(predicate))
