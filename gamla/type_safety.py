import typing
from collections import abc
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

from gamla import construct, operator
from gamla.optimized import sync


def _handle_union_on_left(type1, type2):
    return sync.pipe(
        type1,
        typing.get_args,
        sync.allmap(lambda t: _is_subtype((t, type2))),
    )


def _handle_union_on_right(type1, type2):
    return sync.pipe(
        type2,
        typing.get_args,
        sync.anymap(lambda t: _is_subtype((type1, t))),
    )


_origin_equals = sync.compose_left(operator.equals, sync.before(typing.get_origin))

_handle_union = sync.case_dict(
    {
        sync.compose_left(operator.head, _origin_equals(Union)): sync.star(
            _handle_union_on_left,
        ),
        sync.compose_left(operator.second, _origin_equals(Union)): sync.star(
            _handle_union_on_right,
        ),
    },
)


def _rewrite_optional(x):
    return Union[None, typing.get_args(x)]


def _forward_ref(x):
    def forward_ref(*args, **kwargs):
        return x()(*args, **kwargs)

    return forward_ref


def _iterable_to_union(it):
    it = tuple(it)
    assert it
    if len(it) == 1:
        return it[0]
    return Union[it[0], _iterable_to_union(it[1:])]


_rewrite_typevar = sync.compose_left(
    operator.attrgetter("__constraints__"),
    sync.ternary(operator.empty, construct.just(Any), _iterable_to_union),
)

_handle_generics = sync.alljuxt(
    sync.compose_left(sync.map(typing.get_origin), sync.star(issubclass)),
    sync.compose_left(
        sync.map(typing.get_args),
        sync.star(zip),
        sync.allmap(_forward_ref(lambda: _is_subtype)),
    ),
)


def _handle_callable(c1, c2):
    args1 = typing.get_args(c1)
    args2 = typing.get_args(c2)
    if not args1 and args2:
        return False
    if not args2:
        return True
    input1, output1 = args1
    input2, output2 = args2
    return is_subtype(output1, output2) and (
        Ellipsis in [input1, input2]
        or len(input1) == len(input2)
        and sync.pipe([input1, input2], sync.star(zip), sync.allmap(_is_subtype))
    )


_is_subtype: Callable[[Tuple[Any, Any]], bool] = sync.compose_left(
    sync.map(
        sync.compose_left(
            sync.when(_origin_equals(Optional), _rewrite_optional),
            sync.when(operator.is_instance(TypeVar), _rewrite_typevar),
        ),
    ),
    tuple,
    sync.case_dict(
        {
            sync.allmap(_origin_equals(abc.Callable)): sync.star(_handle_callable),
            operator.inside(Any): sync.compose_left(
                operator.second,
                operator.equals(Any),
            ),
            sync.anymap(_origin_equals(Union)): _handle_union,
            sync.allmap(typing.get_origin): _handle_generics,
            operator.inside(Ellipsis): sync.allmap(operator.equals(Ellipsis)),
            sync.complement(sync.anymap(typing.get_origin)): sync.star(issubclass),
            construct.just(True): construct.just(False),
        },
    ),
)

#: Given two typings, checks if the second is a superset of the first.
is_subtype = sync.compose_left(operator.pack, _is_subtype)


_RETURN_TYPING = "return"


def composable(destination: Callable, origin: Callable, key: Optional[str]) -> bool:
    """Checks if `destination` can be composed after `source`, considering their typing."""
    s = typing.get_type_hints(origin)
    d = typing.get_type_hints(destination)
    if _RETURN_TYPING not in s:
        return True
    if key:
        if key not in d:
            return True
        d = d[key]
    else:
        if _RETURN_TYPING in d:
            del d[_RETURN_TYPING]
        if not d:
            return True
        if len(d) != 1:
            return False
        d = operator.head(d.values())
    return is_subtype(s[_RETURN_TYPING], d)
