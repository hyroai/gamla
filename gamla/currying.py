import asyncio
import functools
import inspect

import toolz


def _curry_helper(
    is_coroutine, f_len_args, f, args_so_far, kwargs_so_far, *args, **kwargs
):
    args_so_far += args
    kwargs_so_far = toolz.merge(kwargs_so_far, kwargs)
    len_so_far = len(args_so_far) + len(kwargs_so_far)
    if len_so_far > len(f_len_args):
        return f(*args_so_far)
    if len_so_far == len(f_len_args):
        return f(*args_so_far, **kwargs_so_far)
    if len_so_far + 1 == len(f_len_args) and is_coroutine:

        @functools.wraps(f)
        async def curry_inner_async(*args, **kwargs):
            return await f(
                *(args_so_far + args), **(toolz.merge(kwargs_so_far, kwargs))
            )

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


def curry(f):
    f_len_args = inspect.signature(f).parameters
    assert len(f_len_args) > 1, f"Curry function must have at least 2 parameters, {f} has {len(f_len_args)}"
    defaults = _infer_defaults(f)
    is_coroutine = asyncio.iscoroutinefunction(f)

    @functools.wraps(f)
    def indirection(*args, **kwargs):
        return _curry_helper(is_coroutine, f_len_args, f, (), defaults, *args, **kwargs)

    return indirection
