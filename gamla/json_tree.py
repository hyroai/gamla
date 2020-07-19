import dataclasses
from typing import Any, Callable

import toolz
from toolz import curried

from gamla import functional, functional_generic


@functional_generic.curry
def _tree_reduce(get_children, reduce_fn, tree_node):
    return reduce_fn(
        tree_node,
        tuple(map(_tree_reduce(get_children, reduce_fn), get_children(tree_node))),
    )


@dataclasses.dataclass(frozen=True)
class _KeyValue:
    key: Any
    value: Any


def _get_children(element):
    return functional_generic.case_dict(
        {
            functional.is_instance(tuple): toolz.identity,
            functional.is_instance(dict): functional_generic.compose_left(
                dict.items, functional_generic.map(functional.star(_KeyValue)), tuple,
            ),
            functional.is_instance(_KeyValue): functional_generic.compose_left(
                lambda x: x.value,
                functional_generic.ternary(
                    functional.is_instance(str), functional.wrap_tuple, _get_children,
                ),
            ),
            functional.is_instance(str): functional.just(()),
        },
    )(element)


_MATCHED = "matched"
_UNMATCHED = "unmatched"
_get_matched = curried.get(_MATCHED)
_get_unmatched = curried.get(_UNMATCHED)


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


@functional_generic.curry
def _get_anywhere_reducer(predicate: Callable, node, children):
    if isinstance(node, str):
        return _make_matched_unmatched((), (node,))
    if isinstance(node, _KeyValue) and predicate(node.key):
        return _merge_children_as_matched(children)
    return _merge_children(children)


def get_leafs_by_ancestor_predicate(predicate: Callable):
    """Gets leafs with ancestor nodes passing the predicate."""
    return functional_generic.compose_left(
        _tree_reduce(_get_children, _get_anywhere_reducer(predicate)), _get_matched,
    )
