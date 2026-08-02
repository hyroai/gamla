"""Asynchronous versions of common functions for optimized use cases."""

import asyncio
import inspect
import types

_CO_COROUTINE = inspect.CO_COROUTINE


def is_coroutine_function(f) -> bool:
    """`inspect.iscoroutinefunction` with fast paths for the common callable types.

    Composition combinators check async-ness on every dynamic dispatch
    (`pipe`, `ternary`, `map`, `juxt`), so the check must be cheap: plain
    functions and bound methods are decided straight from the code-object
    flag, builtins are never coroutine functions, and any other callable
    falls back to `inspect`. Functions tagged with
    `inspect.markcoroutinefunction` (which sets no code flag) are not
    recognized by the fast path.
    """
    t = type(f)
    if t is types.FunctionType:
        return bool(f.__code__.co_flags & _CO_COROUTINE)
    if t is types.BuiltinFunctionType:
        return False
    if t is types.MethodType:
        m = f.__func__
        if type(m) is types.FunctionType:
            return bool(m.__code__.co_flags & _CO_COROUTINE)
    return inspect.iscoroutinefunction(f)


async def to_awaitable(value):
    """Wraps a value in a coroutine.
    If value is a future, it will await it. Otherwise it will simply return the value.
    Useful when we have a mix of coroutines and regular functions.

    >>> run_sync(await to_awaitable(5))
    '5'
    >>> run_sync(await to_awaitable(some_coroutine_that_returns_its_input(5)))
    '5'
    """
    if inspect.isawaitable(value):
        return await value
    return value


def wrap_awaitable(f):
    """Wraps a function with async."""

    async def async_identity(x):
        return x

    return compose_left(f, async_identity)


def compose_left(*funcs):
    async def compose_left_async(*args, **kwargs):
        for f in funcs:
            args = [await to_awaitable(f(*args, **kwargs))]
            kwargs = {}
        return args[0]

    return compose_left_async


def compose(*funcs):
    async def compose(*args, **kwargs):
        for f in reversed(funcs):
            args = [await to_awaitable(f(*args, **kwargs))]
            kwargs = {}
        return args[0]

    return compose


def map(f):
    async def map(it):
        return await asyncio.gather(*(f(x) for x in it))

    return map


def star(f):
    async def star(x):
        return await f(*x)

    return star


def double_star(f):
    async def double_star(x):
        return await f(**x)

    return double_star


def thunk(f, *args, **kwargs):
    async def thunk(*inner_args, **inner_kwargs):
        return await f(*args, **kwargs)(*inner_args, **inner_kwargs)

    return thunk
