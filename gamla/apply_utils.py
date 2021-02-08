from typing import Text


def apply(*args, **kwargs):
    """Apply input on function.

    >>> apply(1)(add(2))
    3
    """

    def apply_inner(function):
        return function(*args, **kwargs)

    return apply_inner


def apply_async(*args, **kwargs):
    """Apply input on an async function.

    >>> apply_async(1)(async_add(2))
    3
    """

    async def apply_async(f):
        return await f(*args, **kwargs)

    return apply_async


def apply_fn_with_args(fn, *args):
    """Returns the result of applying `fn(*args)`."""
    return fn(*args)


def apply_method_async(method: Text, *args, **kwargs):
    """Invokes the specified async method on an object with `*args` and `**kwargs`.

    >>> apply_method_async("get", "http://www.someurl.com")(httpx)
    httpx.Response()
    """

    async def apply_method_async(obj):
        return await apply_async(*args, **kwargs)(getattr(obj, method))

    return apply_method_async


def apply_method(method: Text, *args, **kwargs):
    """Invokes the specified method on an object with `*args` and `**kwargs`.

    >>> apply_method("get", "http://www.someurl.com")(requests)
    requests.Response()
    """

    def apply_method(obj):
        return apply(*args, **kwargs)(getattr(obj, method))

    return apply_method


def invoke(f):
    """Performs a call of the input function.
    >>> invoke(lambda: 0)
    0
    """
    return f()
