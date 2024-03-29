import asyncio
import functools
import inspect


def _curry_helper(
    is_coroutine, signature, f, args_so_far, kwargs_so_far, *args, **kwargs
):
    f_len_args = signature.parameters
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

        curry_inner_async.__signature__ = _update_signature(
            args_so_far,
            kwargs_so_far,
            signature,
        )
        return curry_inner_async

    @functools.wraps(f)
    def curry_inner(*args, **kwargs):
        return _curry_helper(
            is_coroutine, signature, f, args_so_far, kwargs_so_far, *args, **kwargs
        )

    curry_inner.__signature__ = _update_signature(
        args_so_far,
        kwargs_so_far,
        signature,
    )
    return curry_inner


def _update_signature(args_so_far, kwargs_so_far, signature):
    bounded = signature.bind_partial(
        *args_so_far,
        **kwargs_so_far,
    )
    new_signature = signature.replace(
        parameters=[
            v for k, v in signature.parameters.items() if k not in bounded.arguments
        ],
    )
    return new_signature


def _infer_defaults(params):
    kwargs = {}
    for p in params.values():
        if p.default != p.empty:
            kwargs[p.name] = p.default
    return kwargs


def curry(f):
    """Make a function handle partial input, returning a function that expects the rest.

    Warning: uses `inspect` which is slow, so avoid calling this inside a loop.

    >>> def addition(a, b): return a + b
    ... add_3 = gamla.curry(addition)(3)
    ... add_3(7)
    10

    Can also be used as a decorator:
    >>>@gamla.curry
    ... def addition(a, b):
    ...    return a + b

    In case the function is async, the function becomes synchronous until the last argument.
    Although this is not always what you want, it fits the majority of cases.

    >>> @gamla.curry
    ... async def async_addition(a, b):
    ...    await asyncio.sleep(0.1)
    ...    return a + b

    >>> add_3 = async_addition(3)

    >>> await add_3(7)  # Note that `await` is needed here. Must be done inside an async scope.

    The variables can be given with keywords, but mixing keyword and call by order might have unexpected results.
    """
    signature = inspect.signature(f)
    f_len_args = signature.parameters
    assert (
        len(f_len_args) > 1
    ), f"Curry function must have at least 2 parameters, {f} has {len(f_len_args)}"
    defaults = _infer_defaults(f_len_args)
    is_coroutine = asyncio.iscoroutinefunction(f)

    @functools.wraps(f)
    def indirection(*args, **kwargs):
        return _curry_helper(is_coroutine, signature, f, (), defaults, *args, **kwargs)

    return indirection
