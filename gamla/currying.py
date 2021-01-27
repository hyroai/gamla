import asyncio
import dataclasses
import functools
import inspect
from typing import Callable

import toolz


@dataclasses.dataclass
class _Curry:
    function: Callable
    wrapper: Callable

    def __call__(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except TypeError:
            return functools.wraps(self.function)(
                _Curry(
                    functools.partial(self.function, *args, **kwargs),
                    functools.wraps(self.function),
                ),
            )


def curry(f):
    if asyncio.iscoroutinefunction(f):
        return old_curry(f)
    wrapper = functools.wraps(f)
    return wrapper(_Curry(function=f, wrapper=wrapper))


def _curry_helper(
    is_coroutine, f_len_args, f, args_so_far, kwargs_so_far, *args, **kwargs
):
    args_so_far += args
    kwargs_so_far = {**kwargs_so_far, **kwargs}
    len_so_far = len(args_so_far) + len(kwargs_so_far)
    if len_so_far > len(f_len_args):
        return f(*args_so_far)
    if len_so_far == len(f_len_args):
        return f(*args_so_far, **kwargs_so_far)
    if len_so_far + 1 == len(f_len_args) and is_coroutine:

        @functools.wraps(f)
        async def curry_inner_async(*args, **kwargs):
            return await f(*(args_so_far + args), **{**kwargs_so_far, **kwargs})

        return curry_inner_async

    @functools.wraps(f)
    def curry_inner(*args, **kwargs):
        return _curry_helper(
            is_coroutine, f_len_args, f, args_so_far, kwargs_so_far, *args, **kwargs
        )

    return curry_inner


def _infer_defaults(f):
    params = inspect.signature(f).parameters
    kwargs = {}
    for p in params.values():
        if p.default != p.empty:
            kwargs[p.name] = p.default
    return kwargs


def old_curry(f):
    f_len_args = inspect.signature(f).parameters
    assert (
        len(f_len_args) > 1
    ), f"Curry function must have at least 2 parameters, {f} has {len(f_len_args)}"
    defaults = _infer_defaults(f)
    is_coroutine = asyncio.iscoroutinefunction(f)

    @functools.wraps(f)
    def indirection(*args, **kwargs):
        return _curry_helper(is_coroutine, f_len_args, f, (), defaults, *args, **kwargs)

    return indirection


def addition(x):
    return x + 1


def compose_sync(*funcs):
    @functools.wraps(toolz.last(funcs))
    def composed(*args, **kwargs):
        for f in reversed(funcs):
            args = [f(*args, **kwargs)]
            kwargs = {}
        return toolz.first(args)

    return composed


@curry
def after(f1, f2):
    """Second-order composition of `f1` over `f2`."""
    return compose_sync(f1, f2)


def curried_map(f):
    return curried_map_sync(f)


def identity(x):
    return x


def curried_map_sync(f):
    def curried_map(it):
        for x in it:
            yield f(x)

    return curried_map


def _sync_curried_filter(f):
    def curried_filter(it):
        for x in it:
            if f(x):
                yield x

    return curried_filter


def juxt(*funcs):
    def juxt(*args, **kwargs):
        return tuple(func(*args, **kwargs) for func in funcs)

    return juxt


def pair_with(f):
    return juxt(f, identity)


curried_filter = compose_sync(
    after(
        compose_sync(
            curried_map_sync(toolz.second),
            _sync_curried_filter(toolz.first),
        ),
    ),
    curried_map,
    pair_with,
)

p = curried_filter(lambda x: x != 2)([1, 2, 3])

map_filter_empty = compose_sync(after(curried_filter(identity)), curried_map)


d = tuple(map_filter_empty(lambda x: None if x == 2 else x)([1, 2, 3]))
