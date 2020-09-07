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
import random
from concurrent import futures
from typing import Any, Callable, Dict, Iterable, Sequence, Text, TypeVar

import heapq_max
import toolz
from toolz import curried
from toolz.curried import operator

from gamla import functional_utils

do_breakpoint = curried.do(lambda x: builtins.breakpoint())


def do_if(condition, fun):
    def inner_do_if(x):
        if condition(x):
            fun(x)
        return x

    return inner_do_if


def check(condition, exception):
    return do_if(
        toolz.complement(condition), toolz.compose_left(exception, just_raise),
    )


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


def just_raise(exception):
    raise exception


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
            json.loads(item.to_json()), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"),
    ).hexdigest()


def star(function: Callable) -> Callable:
    def star_and_run(x):
        return function(*x)

    async def star_and_run_async(x):
        return await function(*x)

    if inspect.iscoroutinefunction(function):
        return star_and_run_async
    return star_and_run


@toolz.curry
def _assert_f_output_on_inp(f, inp):
    assert f(inp)


def assert_that(f):
    return curried.do(_assert_f_output_on_inp(f))


@toolz.curry
def pmap(f, n_workers, it):
    # The `tuple` is for callers convenience (even without it, the pool is eager).
    return tuple(futures.ThreadPoolExecutor(max_workers=n_workers).map(f, it))


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


@toolz.curry
def len_equals(length: int, seq):
    return len(seq) == length


@toolz.curry
def len_greater(length: int, seq):
    return len(seq) > length


@toolz.curry
def len_smaller(length: int, seq):
    return len(seq) < length


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
    attr_name: Text, attr_transformer: Callable[[Any], Any], dataclass_instance,
):
    return dataclasses.replace(
        dataclass_instance,
        **{
            attr_name: toolz.pipe(
                dataclass_instance, operator.attrgetter(attr_name), attr_transformer,
            ),
        },
    )


@toolz.curry
def dataclass_replace(attr_name: Text, attr_value: Any, dataclass_instance):
    return dataclasses.replace(dataclass_instance, **{attr_name: attr_value})


_R = TypeVar("_R")
_E = TypeVar("_E")


@toolz.curry
def reduce(
    reducer: Callable[[_R, _E], _R], initial_value: _R, elements: Iterable[_E],
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


def apply(*args, **kwargs):
    def apply_inner(function):
        return function(*args, **kwargs)

    return apply_inner


@toolz.curry
def drop_last_while(predicate: Callable[[Any], bool], seq: Sequence) -> Sequence:
    return toolz.pipe(
        seq, reversed, toolz.curry(itertools.dropwhile)(predicate), tuple, reversed,
    )


@toolz.curry
def partition_after(
    predicate: Callable[[Any], bool], seq: Sequence,
) -> Sequence[Sequence]:
    return toolz.reduce(
        lambda a, b: (*a, (b,))
        if not a or predicate(a[-1][-1])
        else (*a[:-1], (*a[-1], b)),
        seq,
        (),
    )


@toolz.curry
def partition_before(
    predicate: Callable[[Any], bool], seq: Sequence,
) -> Sequence[Sequence]:
    return toolz.reduce(
        lambda a, b: (*a, (b,)) if not a or predicate(b) else (*a[:-1], (*a[-1], b)),
        seq,
        (),
    )


def get_all_n_grams(seq):
    return toolz.pipe(
        range(1, len(seq) + 1), curried.mapcat(curried.sliding_window(seq=seq)),
    )


@toolz.curry
def is_instance(the_type, the_value):
    return type(the_value) == the_type


def sample(n: int):
    def sample_inner(population):
        return random.sample(population, n)

    return sample_inner


@functional_utils.curry
def eq_by(f, value_1, value_2):
    return f(value_1) == f(value_2)


eq_str_ignore_case = eq_by(str.lower)


@toolz.curry
def groupby_many_reduce(key: Callable, reducer: Callable, seq: Iterable):
    """
    Group a collection by a key function, when the value is given by a reducer function.

    Parameters:
    key (Callable): Key function (given object in collection outputs key).
    reducer (Callable): Reducer function (given object in collection outputs new value).
    seq (Iterable): Collection.

    Returns:
    Dict[Text, Any]: Dictionary where key has been computed by the `key` function
    and value by the `reducer` function.

    """
    result: Dict[Any, Any] = {}
    for element in seq:
        for key_result in key(element):
            result[key_result] = reducer(result.get(key_result, None), element)
    return result


@functional_utils.curry
def countby_many(f, it):
    """Count elements of a collection by a function which returns a tuple of keys
    for single element.

    Parameters:
    f (Callable): Key function (given object in collection outputs tuple of keys).
    it (Iterable): Collection.

    Returns:
    Dict[Text, Any]: Dictionary where key has been computed by the `f` key function
    and value is the frequency of this key.

    >>> names = ['alice', 'bob', 'charlie', 'dan', 'edith', 'frank']
    >>> countby_many(lambda name: (name[0], name[-1]), names)
    {'a': 1,
     'e': 3,
     'b': 2,
     'c': 1,
     'd': 1,
     'n': 1,
     'h': 1,
     'f': 1,
     'k': 1}
    """
    return toolz.pipe(
        it,
        curried.mapcat(f),
        groupby_many_reduce(toolz.identity, lambda x, y: x + 1 if x else 1),
    )
