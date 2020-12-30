def prepare_and_apply(f):
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
        return await f(value)(value)

    return prepare_and_apply
