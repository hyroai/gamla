import asyncio
import inspect
from typing import Dict

import toolz
from toolz import curried


async def apipe(val, *funcs):
    for f in funcs:
        val = f(val)
        if inspect.isawaitable(val):
            val = await val
    return val


def acompose(*funcs):
    async def composed(*args, **kwargs):
        for f in reversed(funcs):
            inp = f(*args, **kwargs)
            if inspect.isawaitable(inp):
                inp = await inp
            args = [inp]
            kwargs = {}
        return inp

    return composed


def acompose_left(*funcs):
    return acompose(*reversed(funcs))


def run_sync(f):
    """Runs a coroutine in a synchronous context, blocking until result arrives."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(asyncio.ensure_future(f, loop=loop))


@toolz.curry
async def amap(f, it):
    return await asyncio.gather(*map(f, it))


@toolz.curry
async def aexcepts(exception_type, func, handler, x):
    try:
        return await func(x)
    except exception_type as error:
        return handler(error)


@toolz.curry
async def mapa(f, it):
    async for element in it:
        yield f(element)


async def aconcat(async_generators):
    async for g in async_generators:
        for x in g:
            yield x


def ajuxt(*funcs):
    async def ajuxt_inner(x):
        results = []
        for f in funcs:
            result = f(x)
            if inspect.isawaitable(result):
                result = await result
            results.append(result)
        return tuple(results)

    return ajuxt_inner


@toolz.curry
async def afilter(func, it):
    it = tuple(it)
    results = await amap(func, it)
    return toolz.pipe(
        zip(it, results), curried.filter(toolz.second), curried.map(toolz.first)
    )


def afirst(*funcs, exception_type):
    async def afirst_inner(x):
        for f in funcs:
            try:
                result = f(x)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except exception_type:
                pass
        raise exception_type

    return afirst_inner


@toolz.curry
async def apair_with(f, element):
    return await f(element), element


@toolz.curry
async def apair_right(f, element):
    return element, await f(element)


@toolz.curry
async def akeymap(f, d: Dict):
    return await aitemmap(ajuxt(acompose_left(toolz.first, f), toolz.second), d)


@toolz.curry
async def avalmap(f, d: Dict):
    return await aitemmap(ajuxt(toolz.first, acompose_left(toolz.second, f)), d)


@toolz.curry
async def aitemmap(f, d: Dict):
    return await apipe(d, dict.items, amap(f), dict)
