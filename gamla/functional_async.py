import asyncio
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterable

from gamla import functional_generic


def run_sync(f):
    """Runs a coroutine in a synchronous context, blocking until result arrives."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(asyncio.ensure_future(f, loop=loop))


@functional_generic.curry
async def amap_ascompleted(
    f: Callable[[Any], Awaitable[Any]], it: Iterable
) -> AsyncGenerator[Any, None]:
    for future in asyncio.as_completed(map(f, it)):
        yield await future


@functional_generic.curry
async def aexcepts(exception_type, func, handler, x):
    try:
        return await func(x)
    except exception_type as error:
        return handler(error)


@functional_generic.curry
async def mapa(f, it):
    async for element in it:
        yield f(element)


async def aconcat(async_generators):
    async for g in async_generators:
        for x in g:
            yield x


def afirst(*funcs, exception_type):
    async def afirst_inner(x):
        for f in funcs:
            try:
                return await functional_generic.to_awaitable(f(x))
            except exception_type:
                pass
        raise exception_type

    return afirst_inner
