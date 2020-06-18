import asyncio
import functools
import inspect
from typing import Callable, Tuple

import toolz
from toolz import curried
from toolz.curried import operator

from gamla import functional


def compose_left(*funcs):
    return compose(*reversed(funcs))


_any_is_async = toolz.compose(any, curried.map(asyncio.iscoroutinefunction))


async def to_awaitable(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _acompose(*funcs):
    async def composed(*args, **kwargs):
        for f in reversed(funcs):
            args = [await to_awaitable(f(*args, **kwargs))]
            kwargs = {}
        return toolz.first(args)

    return composed


def compose(*funcs):
    if _any_is_async(funcs):
        composed = _acompose(*funcs)
        # TODO(uri): Far from a perfect id, but should work most of the time.
        # Improve by having higher order functions create meaningful names (e.g. `map`).
        # Copying `toolz` convention.
        composed.__name__ = "_of_".join(map(lambda x: x.__name__, funcs))
        return composed
    return toolz.compose(*funcs)


def _make_amap(f):
    async def amap_inner(it):
        return await asyncio.gather(*curried.map(f, it))

    return amap_inner


def gamla_map(*args):
    if len(args) == 2:
        f, it = args
        if asyncio.iscoroutinefunction(f):
            return _make_amap(f)(it)
        return curried.map(f, it)
    [f] = args
    if asyncio.iscoroutinefunction(f):
        return _make_amap(f)
    return curried.map(f)


map = gamla_map


@toolz.curry
def after(f1, f2):
    return compose(f1, f2)


@toolz.curry
def before(f1, f2):
    return compose_left(f1, f2)


def lazyjuxt(*funcs):
    """Reverts to eager implementation if any of `funcs` is async."""
    if _any_is_async(funcs):

        async def lazyjuxt_inner(value):
            return await toolz.pipe(
                funcs,
                curried.map(toolz.compose_left(functional.apply(value), to_awaitable)),
                functional.star(asyncio.gather),
            )

        return lazyjuxt_inner
    return compose_left(functional.apply, curried.map, functional.apply(funcs))


juxt = compose(after(tuple), lazyjuxt)
alljuxt = compose(after(all), lazyjuxt)
anyjuxt = compose(after(any), lazyjuxt)
juxtcat = compose(after(toolz.concat), lazyjuxt)


def ternary(condition, f_true, f_false):
    if _any_is_async([condition, f_true, f_false]):

        async def ternary_inner_async(*args, **kwargs):
            return (
                await to_awaitable(f_true(*args, **kwargs))
                if await to_awaitable(condition(*args, **kwargs))
                else await to_awaitable(f_false(*args, **kwargs))
            )

        return ternary_inner_async

    def ternary_inner(*args, **kwargs):
        return (
            f_true(*args, **kwargs)
            if condition(*args, **kwargs)
            else f_false(*args, **kwargs)
        )

    return ternary_inner


curried_ternary = ternary


def pipe(val, *funcs):
    return compose_left(*funcs)(val)


def _curry_helper(f, args_so_far, kwargs_so_far, *args, **kwargs):
    f_len_args = inspect.signature(f).parameters
    args_so_far += args
    kwargs_so_far = toolz.merge(kwargs_so_far, kwargs)
    len_so_far = len(args_so_far) + len(kwargs_so_far)
    if len_so_far > len(f_len_args):
        return f(*args_so_far)
    if len_so_far == len(f_len_args):
        return f(*args_so_far, **kwargs_so_far)
    if len_so_far + 1 == len(f_len_args) and asyncio.iscoroutinefunction(f):

        @functools.wraps(f)
        async def curry_inner_async(*args, **kwargs):
            return await f(
                *(args_so_far + args), **(toolz.merge(kwargs_so_far, kwargs))
            )

        return curry_inner_async

    @functools.wraps(f)
    def curry_inner(*args, **kwargs):
        return _curry_helper(f, args_so_far, kwargs_so_far, *args, **kwargs)

    return curry_inner


def _infer_defaults(f):
    params = inspect.signature(f).parameters
    kwargs = {}
    for p in params.values():
        if p.default != p.empty:
            kwargs[p.name] = p.default
    return kwargs


def curry(f):
    @functools.wraps(f)
    def indirection(*args, **kwargs):
        return _curry_helper(f, (), _infer_defaults(f), *args, **kwargs)

    return indirection


def _compose_over_binary_curried(composer):
    def composition_over_binary_curried(*args):
        f = composer(args[0])
        if len(args) == 2:
            return f(args[1])
        return f

    return composition_over_binary_curried


allmap = _compose_over_binary_curried(compose(after(all), gamla_map))
anymap = _compose_over_binary_curried(compose(after(any), gamla_map))


itemmap = _compose_over_binary_curried(
    compose(after(dict), before(dict.items), gamla_map)
)
keymap = _compose_over_binary_curried(
    compose(itemmap, lambda f: juxt(f, toolz.second), before(toolz.first))
)

valmap = _compose_over_binary_curried(
    compose(itemmap, lambda f: juxt(toolz.first, f), before(toolz.second))
)


pair_with = _compose_over_binary_curried(lambda f: juxt(f, toolz.identity))
pair_right = _compose_over_binary_curried(lambda f: juxt(toolz.identity, f))

filter = _compose_over_binary_curried(
    compose(
        after(compose(curried.map(toolz.second), curried.filter(toolz.first))),
        gamla_map,
        pair_with,
    )
)


_first_truthy_index = compose_left(
    enumerate,
    curried.filter(toolz.second),
    curried.map(toolz.first),
    toolz.excepts(StopIteration, toolz.first),
)


class NoConditionMatched(Exception):
    pass


def _case(predicates: Tuple[Callable, ...], mappers: Tuple[Callable, ...]):
    """Case with functions.

    Handles async iff one of the predicates is async.
    Raises `KeyError` if no condition matched.
    """
    return compose_left(
        pair_right(
            compose_left(
                lazyjuxt(*predicates),
                _first_truthy_index,
                functional.check(
                    toolz.complement(operator.eq(None)), NoConditionMatched
                ),
                mappers.__getitem__,
            )
        ),
        functional.star(functional.apply),
    )


def case(predicates_and_mappers: Tuple[Tuple[Callable, Callable], ...]):
    predicates = tuple(map(toolz.first, predicates_and_mappers))
    mappers = tuple(map(toolz.second, predicates_and_mappers))
    return _case(predicates, mappers)


case_dict = compose_left(dict.items, tuple, case)
