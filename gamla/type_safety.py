import typing
from collections import abc
from typing import Any, Callable, Optional, Tuple, Union

from gamla import operator
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
    def _forward_ref(*args, **kwargs):
        return x()(*args, **kwargs)

    return _forward_ref


_handle_generics = sync.alljuxt(
    sync.compose_left(sync.map(typing.get_origin), sync.star(issubclass)),
    sync.compose_left(
        sync.map(typing.get_args),
        sync.star(zip),
        sync.allmap(_forward_ref(lambda: _is_subtype)),
    ),
)


def _handle_callable(args1, output1, args2, output2):
    return is_subtype(output1, output2) and (
        Ellipsis in [args1, args2]
        or len(args1) == len(args2)
        and sync.pipe([args1, args2], sync.star(zip), sync.allmap(_is_subtype))
    )


_is_subtype: Callable[[Tuple[Any, Any]], bool] = sync.compose_left(
    sync.map(sync.when(_origin_equals(Optional), _rewrite_optional)),
    tuple,
    sync.case_dict(
        {
            sync.allmap(_origin_equals(abc.Callable)): sync.compose_left(
                sync.mapcat(typing.get_args),
                sync.star(_handle_callable),
            ),
            operator.inside(Any): sync.compose_left(
                operator.second,
                operator.equals(Any),
            ),
            sync.anymap(_origin_equals(Union)): _handle_union,
            sync.allmap(typing.get_origin): _handle_generics,
            operator.inside(Ellipsis): sync.allmap(operator.equals(Ellipsis)),
            sync.complement(sync.anymap(typing.get_origin)): sync.star(issubclass),
            operator.just(True): operator.just(False),
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
