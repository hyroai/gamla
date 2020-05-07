import asyncio
import functools
import inspect
from typing import Dict

import toolz
from toolz import curried

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


def _acompose_left(*funcs):
    return _acompose(*reversed(funcs))


def compose(*funcs):
    if _any_is_async(funcs):
        return _acompose(*funcs)
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


@toolz.curry
def apply(value, function):
    return function(value)


def lazyjuxt(*funcs):
    """Reverts to eager implementation if any of `funcs` is async."""
    if _any_is_async(funcs):

        async def lazyjuxt_inner(value):
            return await toolz.pipe(
                funcs,
                curried.map(toolz.compose_left(apply(value), to_awaitable)),
                functional.star(asyncio.gather),
            )

        return lazyjuxt_inner
    return compose_left(apply, curried.map, apply(funcs))


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


# TODO(uri): Currently async only.
@curry
async def filter(func, it):
    results = await to_awaitable(gamla_map(func, it))
    return toolz.pipe(
        zip(results, it), curried.filter(toolz.first), curried.map(toolz.second)
    )


def _compose_over_binary_curried(composer):
    def composition_over_binary_curried(*args):
        if len(args) == 2:
            f, it = args
            return pipe(it, composer(f))
        [f] = args
        return composer(f)

    return composition_over_binary_curried


allmap = _compose_over_binary_curried(compose(after(all), gamla_map))
anymap = _compose_over_binary_curried(compose(after(any), gamla_map))


itemmap = _compose_over_binary_curried(
    compose(after(dict), before(dict.items), gamla_map)
)


@toolz.curry
def keymap(f, d: Dict):
    return itemmap(juxt(compose_left(toolz.first, f), toolz.second), d)


@toolz.curry
def valmap(f, d: Dict):
    return itemmap(juxt(toolz.first, compose_left(toolz.second, f)), d)
