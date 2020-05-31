import builtins
import cProfile
import dataclasses
import functools
import hashlib
import heapq
import inspect
import itertools
import json
import logging
from concurrent import futures
from typing import Any, Callable, Iterable, Sequence, Text, Type, TypeVar

import heapq_max
import toolz
from toolz import curried
from toolz.curried import operator

do_breakpoint = curried.do(lambda x: builtins.breakpoint())


def do_if(condition, fun):
    def inner_do_if(x):
        if condition(x):
            fun(x)
            return x
        return x

    return inner_do_if


def check(condition, exception):
    return do_if(toolz.complement(condition), make_raise(exception))


def bifurcate(*funcs):
    """Serially runs each function on tee'd copies of `input_generator`."""

    def inner(input_iterable):
        return toolz.pipe(
            zip(funcs, itertools.tee(iter(input_iterable), len(funcs))),
            curried.map(star(lambda f, generator: f(generator))),
            tuple,
        )

    return inner


def singleize(func: Callable) -> Callable:
    def wrapped(some_input):
        if isinstance(some_input, tuple):
            return func(some_input)
        return toolz.first(func((some_input,)))

    async def wrapped_async(some_input):
        if isinstance(some_input, tuple):
            return await func(some_input)
        return toolz.first(await func((some_input,)))

    if inspect.iscoroutinefunction(func):
        return wrapped_async
    return wrapped


def wrapped_partial(func: Callable, *args, **kwargs) -> Callable:
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def ignore_input(inner):
    def ignore_and_run(*args, **kwargs):
        return inner()

    async def ignore_and_run_async(*args, **kwargs):
        return await inner()

    if inspect.iscoroutinefunction(inner):
        return ignore_and_run_async
    return ignore_and_run


def make_raise(exception):
    def inner():
        raise exception

    return ignore_input(inner)


@toolz.curry
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


@toolz.curry
def pmap(f, n_workers, it):
    # The `tuple` is for callers convenience (even without it, the pool is eager).
    return tuple(futures.ThreadPoolExecutor(max_workers=n_workers).map(f, it))


@toolz.curry
def pfilter(f, it):
    return toolz.pipe(
        it,
        bifurcate(pmap(f, None), curried.map(toolz.identity)),
        zip,
        curried.filter(toolz.first),
        curried.map(toolz.second),
    )


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


def log_text(text: Text, level: int = logging.INFO):
    return curried.do(lambda x: logging.log(level, text.format(x)))


def just(x):
    return ignore_input(lambda: x)


# To get a unique caching key for each function invocation, we take `args` and `items()`
# of `kwargs` and sort them (by keys), while also marking the beginning of `kwargs`.
# Inspired by: http://code.activestate.com/recipes/578078/ (python LRU cache
# implementation).
def make_call_key(args, kwargs):
    """Stable id for function calls, can be used for caching."""
    key = args
    if kwargs:
        key += "##kwargs##", tuple(sorted(kwargs.items()))
    return key


@toolz.curry
def top(iterable, key=toolz.identity):
    """Generates elements from max to min."""
    h = []
    for i, value in enumerate(iterable):
        # Use the index as a tie breaker.
        heapq_max.heappush_max(h, (key(value), i, value))
    while h:
        yield toolz.nth(2, heapq_max.heappop_max(h))


@toolz.curry
def bottom(iterable, key=toolz.identity):
    """Generates elements from min to max."""
    h = []
    for i, value in enumerate(iterable):
        # Use the index as a tie breaker.
        heapq.heappush(h, (key(value), i, value))
    while h:
        yield toolz.nth(2, heapq.heappop(h))


def profileit(func):
    def wrapper(*args, **kwargs):
        filename = func.__name__ + ".profile"
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(filename)
        logging.info(f"Saved profiling stats to {filename}")
        return retval

    return wrapper


@toolz.curry
def inside(val, container):
    return val in container


average = toolz.compose_left(
    bifurcate(sum, toolz.count),
    toolz.excepts(ZeroDivisionError, star(operator.truediv), lambda _: 0),
)


@toolz.curry
def len_equals(length: int, seq):
    return len(seq) == length


@toolz.curry
def skip(n, seq):
    for i, x in enumerate(seq):
        if i < n:
            continue
        yield x


def wrap_tuple(x):
    return (x,)


def invoke(x):
    return x()


@toolz.curry
def assoc_in(d, keys, value, factory=dict):
    return update_in(d, keys, lambda x: value, value, factory)


@toolz.curry
def update_in(d, keys, func, default=None, factory=dict):
    ks = iter(keys)
    k = next(ks)

    rv = inner = factory()
    rv.update(d)

    for key in ks:
        if k in d or isinstance(d, list):
            d = d[k]
            if isinstance(d, dict):
                dtemp = {}
                dtemp.update(d)
            elif isinstance(d, list):
                dtemp = []
                dtemp.extend(d)
            else:
                dtemp = factory()
        else:
            d = dtemp = factory()

        inner[k] = inner = dtemp
        k = key

    if k in d:
        inner[k] = func(d[k])
    else:
        inner[k] = func(default)
    return rv


@toolz.curry
def dataclass_transform(
    attr_name: Text, attr_transformer: Callable[[Any], Any], dataclass_instance
):
    return dataclasses.replace(
        dataclass_instance,
        **{
            attr_name: toolz.pipe(
                dataclass_instance, operator.attrgetter(attr_name), attr_transformer
            )
        },
    )


@toolz.curry
def dataclass_replace(attr_name: Text, attr_value: Any, dataclass_instance):
    return dataclasses.replace(dataclass_instance, **{attr_name: attr_value})


_R = TypeVar("_R")
_E = TypeVar("_E")


@toolz.curry
def reduce(
    reducer: Callable[[_R, _E], _R], initial_value: _R, elements: Iterable[_E]
) -> _R:
    return functools.reduce(reducer, elements, initial_value)


@toolz.curry
def suffix(val, it: Iterable):
    return itertools.chain(it, (val,))


@toolz.curry
def prefix(val, it: Iterable):
    return itertools.chain((val,), it)


@toolz.curry
def concat_with(new_it: Iterable, it: Iterable):
    return itertools.chain(it, new_it)


@toolz.curry
def wrap_str(wrapping_string: Text, x: Text) -> Text:
    return wrapping_string.format(x)


@toolz.curry
def apply(value, function):
    return function(value)


@toolz.curry
def drop_last_while(predicate: Callable[[Any], bool], seq: Sequence) -> Sequence:
    return toolz.pipe(
        seq, reversed, toolz.curry(itertools.dropwhile)(predicate), tuple, reversed
    )


@toolz.curry
def partition_when(
    predicate: Callable[[Any], bool], seq: Sequence
) -> Sequence[Sequence]:
    return toolz.reduce(
        lambda a, b: (*a, (b,))
        if not a or predicate(a[-1][-1])
        else (*a[:-1], (*a[-1], b)),
        seq,
        (),
    )
