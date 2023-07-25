import asyncio
from typing import Any, Callable

from gamla import construct, excepts_decorator, functional, functional_generic, operator
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


def ignore_first_arg(f: Callable) -> Callable:
    """Ignores the first argument."""
    if asyncio.iscoroutinefunction(f):

        async def ignore_first_arg_async(_, *args, **kwargs):
            return await f(*args, **kwargs)

        return ignore_first_arg_async

    def ignore_first_arg(_, *args, **kwargs):
        return f(*args, **kwargs)

    return ignore_first_arg


def persistent_cache(
    get_item: Callable[[str], Any],
    set_item: Callable[[str, Any], None],
    make_key: Callable[[Any], str],
    force: bool,
) -> Callable:
    """Wraps a function with persistent cache. Gets the item getter and item setter, a function that creates the key,
    and a boolean flag that if set to true forces the cache to refresh.
    """

    def decorator(f: Callable):
        return excepts_decorator.try_and_excepts(
            KeyError,  # type: ignore
            functional_generic.compose_left(
                functional_generic.juxt(
                    ignore_first_arg(make_key),
                    ignore_first_arg(f),
                ),
                functional_generic.side_effect(functional_generic.star(set_item)),
                operator.second,
            ),
            functional_generic.ternary(
                construct.just(force),
                functional.make_raise(KeyError),
                functional_generic.compose_left(make_key, get_item),
            ),
        )

    return decorator


#: Make a function act on the first element on incoming input.
on_first = functional_generic.before(operator.head)
#: Make a function act on the second element on incoming input.
on_second = functional_generic.before(operator.second)
