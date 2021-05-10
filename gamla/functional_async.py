import asyncio
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterable, Type

from gamla import currying
from gamla.optimized import async_functions


def run_sync(f):
    """Runs a coroutine in a synchronous context, blocking until result arrives.

    >>> async def afoo(x):
    >>>     await asyncio.sleep(1)
    >>>     return x
    >>> run_sync(afoo(1))
    1 (after 1 second of waiting)
    """
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(asyncio.ensure_future(f, loop=loop))


@currying.curry
async def amap_ascompleted(
    f: Callable[[Any], Awaitable[Any]], it: Iterable
) -> AsyncGenerator[Any, None]:
    """Returns an AsyncGenerator of the results after applying async `f` to each element of Iterable `it`

    >>> async def amulti(x)
    >>>    return x*2
    >>> async def to_list(ag):
    >>>     return [i async for i in ag]
    >>> run_sync(to_list(amap_ascompleted(amulti, range(4))))
    [6, 0, 2, 4] (In a random order)
    """
    for future in asyncio.as_completed(map(f, it)):
        yield await future


@currying.curry
async def aexcepts(
    exception_type: Type[Exception], func: Callable, handler: Callable, x: Any
):
    """An async functional try/except block: await and return `func` on `x`.
    If fails with `exception_type`, return the reult of running handler on the error.


    >>> async def araise(x):
    >>>     raise ValueError
    >>> run_sync(aexcepts(ValueError, araise, lambda e: e, 5))
    ValueError()

    >>> async def a_just(x):
    >>>     return x
    >>> run_sync(aexcepts(ValueError, a_just, lambda e: e, 5))
    5
    """
    try:
        return await func(x)
    except exception_type as error:
        return handler(error)


@currying.curry
async def mapa(f: Callable, it: AsyncGenerator) -> AsyncGenerator:
    """Returns an AsyncGenerator of the results after applying `f` to each async element of `it`

    >>> async def arange(count):
    >>>     for i in range(count):
    >>>         yield(i)
    >>> async def to_list(ag):
    >>>     return [i async for i in ag]
    >>> run_sync(to_list(mapa(lambda x: x*2, arange(4))))
    [0, 2, 4, 6]
    """
    async for element in it:
        yield f(element)


async def aconcat(async_generator: AsyncGenerator) -> AsyncGenerator:
    """Concat iterables of an async_generator.

    >>> async def many_range(count):
    >>>     for i in range(count):
    >>>         yield range(i, i+1)
    >>> async def to_list(ag):
    >>>     return [i async for i in ag]
    >>> run_sync(to_list(aconcat(many_range(4))))
    [0, 1, 2, 3]
    """
    async for g in async_generator:
        for x in g:
            yield x


def afirst(*funcs: Callable, exception_type: Type[Exception]):
    """Runs given `funcs` serially until getting a succefull result.
    Returns the result of the first function that runs on `x` without raising `exception_type`.
    If all given function raise e`xception_type`, `exception_type` will be raised.

    >>> async def araise(x):
    >>>     raise ValueError
    >>> run_sync(afirst(araise, lambda x: x*x, exception_type=ValueError)(3))
    9
    """

    async def afirst_inner(x: Any):
        for f in funcs:
            try:
                return await async_functions.to_awaitable(f(x))
            except exception_type:
                pass
        raise exception_type

    return afirst_inner
