import asyncio
import functools
import hashlib
import inspect
import itertools
import json
import logging
from typing import Callable, Dict, Iterable, Text, Type

import gevent
import toolz
from gevent import pool
from toolz import curried


@toolz.curry
def dict_get(dict_obj: Dict, default, key):
    return dict_obj.get(key, default)


def do_if(condition, fun):
    return curried.do(curried_ternary(condition, fun, toolz.identity))


def check(condition, exception):
    return do_if(toolz.complement(condition), make_raise(exception))


def bifurcate(*funcs):
    """Serially runs each function on tee'd copies of `input_generator`."""

    def inner(input_generator):
        return toolz.pipe(
            zip(funcs, itertools.tee(input_generator, len(funcs))),
            curried.map(star(lambda f, generator: f(generator))),
            tuple,
        )

    return inner


def singleize(func: Callable) -> Callable:
    def wrapped(some_input):
        if isinstance(some_input, tuple):
            return func(some_input)
        return toolz.first(func((some_input,)))

    return wrapped


def wrapped_partial(func: Callable, *args, **kwargs) -> Callable:
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


@toolz.curry
def anymap(f: Callable, it: Iterable):
    return any(map(f, it))


def anyjuxt(*funcs):
    return toolz.compose(any, toolz.juxt(*funcs))


@toolz.curry
def allmap(f: Callable, it: Iterable):
    return all(map(f, it))


def alljuxt(*funcs):
    return toolz.compose(all, toolz.juxt(*funcs))


def ignore_input(inner):
    def ignore_and_run(*args, **kwargs):
        return inner()

    return ignore_and_run


def curried_ternary(condition, f_true, f_false):
    def inner(*args, **kwargs):
        return (
            f_true(*args, **kwargs)
            if condition(*args, **kwargs)
            else f_false(*args, **kwargs)
        )

    return inner


def make_raise(exception):
    def inner():
        raise exception

    return ignore_input(inner)


def translate_exception(func, exc1, exc2):
    """`func` is assumed to be unary."""
    return toolz.excepts(exc1, func, make_raise(exc2))


@functools.lru_cache(maxsize=None)
def compute_stable_json_hash(item) -> Text:
    return hashlib.sha1(
        json.dumps(
            json.loads(item.to_json()), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def star(function: Callable) -> Callable:
    return lambda x: function(*x)


@toolz.curry
def _assert_f_output_on_inp(f, inp):
    assert f(inp)


def assert_that(f):
    return curried.do(_assert_f_output_on_inp(f))


_GLOBAL_POOL = pool.Group()


async def apipe(val, *funcs):
    for f in funcs:
        val = f(val)
        if inspect.isawaitable(val):
            val = await val
    return val


def acompose(*funcs):
    async def composed(*args, **kwargs):
        if args:
            inp = toolz.first(args)
        else:
            inp = toolz.first(kwargs.values())
        for f in reversed(funcs):
            inp = f(inp)
            if inspect.isawaitable(inp):
                inp = await inp
        return inp

    return composed


def acompose_left(*funcs):
    return acompose(*reversed(funcs))


async def materialize(async_generator):
    elements = []
    async for element in async_generator:
        elements.append(element)
    return tuple(elements)


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
def pmap(f, it):
    # The `tuple` is for callers convenience (even without it, the pool is eager).
    return tuple(_GLOBAL_POOL.map(f, it))


@toolz.curry
def pfilter(f, it):
    return toolz.pipe(
        it,
        bifurcate(pmap(f), curried.map(toolz.identity)),
        zip,
        curried.filter(toolz.first),
        curried.map(toolz.second),
    )


def pfirst(*funcs, exception_type):
    """Parallel+lazy iterator.

    This is used for when we want to start everything in parallel, but return a value
    on the first successful result.
    """
    failure_value = "gamla-value-signifying-failure"

    def inner(*args, **kwargs):
        return toolz.pipe(
            funcs,
            # Prepare runs for each function on the given input, wrapping them for
            # failure as gevent treats an exception within a greenlet as a real error.
            curried.map(
                lambda f: gevent.spawn(
                    toolz.excepts(exception_type, f, lambda _: failure_value),
                    *args,
                    **kwargs
                )
            ),
            # Materialize to actually start the requests in parallel.
            tuple,
            # Don't wait for all to finish, start examining the requests by order.
            curried.map(lambda promise: promise.get()),
            curried.filter(lambda result: result != failure_value),
            translate_exception(next, StopIteration, exception_type),
        )

    return inner


def first(*funcs, exception_type: Type[Exception]):
    def inner(*args, **kwargs):
        for func in funcs:
            try:
                return func(*args, **kwargs)
            except exception_type:
                pass
        raise exception_type

    return inner


logger = curried.do(logging.info)


def log_text(text: Text):
    return curried.do(lambda _: logging.info(text))
