import asyncio
from typing import Any, Callable

from gamla import excepts_decorator, functional_generic, operator, sync
from gamla.optimized import async_functions


def prepare_and_apply(f: Callable) -> Callable:
    """Transforms a higher order function to a regular one.

    Uses the given value once to prepare a regular function, then again to call it with.

    >>> def increment(x): return x + 1
    >>> def decrement(x): return x - 1
    >>> def conditional_transformation(x):
    ...     return increment if x < 10 else decrement
    >>> prepare_and_apply(conditional_transformation))(15)
    14
    """

    def prepare_and_apply(value: Any):
        return f(value)(value)

    return prepare_and_apply


def prepare_and_apply_async(f: Callable) -> Callable:
    """Transforms a higher order function to a regular one.

    Uses the given value once to prepare a regular function, then again to call it with.

    >>> async def increment(x): return x + 1
    >>> async def decrement(x): return x - 1
    >>> def conditional_transformation(x):
    ...     return increment if x < 10 else decrement
    >>> prepare_and_apply(conditional_transformation))(15)
    14
    """

    async def prepare_and_apply(value: Any):
        return await async_functions.to_awaitable(
            (await async_functions.to_awaitable(f(value)))(value),
        )

    return prepare_and_apply


def ignore_first(f: Callable) -> Callable:
    if asyncio.iscoroutinefunction(f):

        async def ignore_first_async(_, *args, **kwargs):
            return await f(*args, **kwargs)

        return ignore_first_async

    def ignore_first(_, *args, **kwargs):
        return f(*args, **kwargs)

    return ignore_first


def persistent_cache(
    get_item: Callable[[str], Any],
    set_item: Callable[[str, Any], None],
    make_key: Callable,
) -> Callable:
    def decorator(f: Callable):
        return excepts_decorator.try_and_excepts(
            KeyError,  # type: ignore
            sync.compose_left(
                sync.juxt(ignore_first(make_key), ignore_first(f)),
                functional_generic.side_effect(sync.star(set_item)),
                operator.second,
            ),
            sync.compose_left(make_key, get_item),
        )

    return decorator


#: Make a function act on the first element on incoming input.
on_first = functional_generic.before(operator.head)
#: Make a function act on the second element on incoming input.
on_second = functional_generic.before(operator.second)
