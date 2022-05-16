from gamla import functional_generic, operator
from gamla.optimized import async_functions


def prepare_and_apply_sync(f):
    """Transforms a higher order function to a regular one.

    Uses the given value once to prepare a regular function, then again to call it with.

    >>> def increment(x): return x + 1
    >>> def decrement(x): return x - 1
    >>> def conditional_transformation(x):
    ...     return increment if x < 10 else decrement
    >>> prepare_and_apply(conditional_transformation))(15)
    14
    """

    def prepare_and_apply(value):
        return f(value)(value)

    return prepare_and_apply


def prepare_and_apply_async(f):
    """Transforms a higher order function to a regular one.

    Uses the given value once to prepare a regular function, then again to call it with.

    >>> async def increment(x): return x + 1
    >>> async def decrement(x): return x - 1
    >>> def conditional_transformation(x):
    ...     return increment if x < 10 else decrement
    >>> prepare_and_apply(conditional_transformation))(15)
    14
    """

    async def prepare_and_apply(value):
        return await async_functions.to_awaitable(
            (await async_functions.to_awaitable(f(value)))(value),
        )

    return prepare_and_apply


prepare_and_apply = functional_generic.choose_by_async(
    prepare_and_apply_sync,
    prepare_and_apply_async,
)

#: Make a function act on the first element on incoming input.
on_first = functional_generic.before(operator.head)
#: Make a function act on the second element on incoming input.
on_second = functional_generic.before(operator.second)
