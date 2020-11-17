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
import operator
import random
from concurrent import futures
from typing import Any, Callable, Dict, Iterable, Sequence, Text, TypeVar

import heapq_max
import toolz
from toolz import curried

from gamla import currying


def identity(x):
    return x


do_breakpoint = curried.do(lambda x: builtins.breakpoint())


count = toolz.count


def sort_by(key: Callable):
    def sort_by(seq: Iterable):
        return sorted(seq, key=key)

    return sort_by


def sort_by_reversed(key: Callable):
    def sort_by_reversed(seq: Iterable):
        return sorted(seq, key=key, reverse=True)

    return sort_by_reversed


sort = sort_by(identity)
sort_reversed = sort_by_reversed(identity)


def curried_map_sync(f):
    def curried_map(it):
        for x in it:
            yield f(x)

    return curried_map


def pack(*stuff):
    return stuff


def do_if(condition, fun):
    def inner_do_if(x):
        if condition(x):
            fun(x)
        return x

    return inner_do_if


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


@currying.curry
def translate_exception(func, exc1, exc2):
    """`func` is assumed to be unary."""
    return toolz.excepts(exc1, func, make_raise(exc2))


def to_json(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    return json.dumps(obj)


@functools.lru_cache(maxsize=None)
def compute_stable_json_hash(item) -> Text:
    return hashlib.sha1(
        json.dumps(
            json.loads(to_json(item)),
            sort_keys=True,
            separators=(",", ":"),
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


@currying.curry
def _assert_f_output_on_inp(f, inp):
    assert f(inp)


def assert_that(f):
    return curried.do(_assert_f_output_on_inp(f))


@currying.curry
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


@currying.curry
def top(iterable, key=identity):
    """Generates elements from max to min."""
    h = []
    for i, value in enumerate(iterable):
        # Use the index as a tie breaker.
        heapq_max.heappush_max(h, (key(value), i, value))
    while h:
        yield toolz.nth(2, heapq_max.heappop_max(h))


@currying.curry
def bottom(iterable, key=identity):
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


@currying.curry
def inside(val, container):
    return val in container


@currying.curry
def len_equals(length: int, seq):
    return len(seq) == length


@currying.curry
def len_greater(length: int, seq):
    return len(seq) > length


@currying.curry
def len_smaller(length: int, seq):
    return len(seq) < length


@currying.curry
def skip(n, seq):
    for i, x in enumerate(seq):
        if i < n:
            continue
        yield x


def wrap_tuple(x):
    return (x,)


def invoke(x):
    return x()


@currying.curry
def assoc_in(d, keys, value, factory=dict):
    return update_in(d, keys, lambda x: value, value, factory)


@currying.curry
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


@currying.curry
def dataclass_transform(
    attr_name: Text,
    attr_transformer: Callable[[Any], Any],
    dataclass_instance,
):
    return dataclasses.replace(
        dataclass_instance,
        **{
            attr_name: toolz.pipe(
                dataclass_instance,
                attrgetter(attr_name),
                attr_transformer,
            ),
        },
    )


@currying.curry
def dataclass_replace(attr_name: Text, attr_value: Any, dataclass_instance):
    return dataclasses.replace(dataclass_instance, **{attr_name: attr_value})


_R = TypeVar("_R")
_E = TypeVar("_E")


@currying.curry
def reduce(
    reducer: Callable[[_R, _E], _R],
    initial_value: _R,
    elements: Iterable[_E],
) -> _R:
    return functools.reduce(reducer, elements, initial_value)


@currying.curry
def suffix(val, it: Iterable):
    return itertools.chain(it, (val,))


@currying.curry
def prefix(val, it: Iterable):
    return itertools.chain((val,), it)


@currying.curry
def concat_with(new_it: Iterable, it: Iterable):
    return itertools.chain(it, new_it)


@currying.curry
def wrap_str(wrapping_string: Text, x: Text) -> Text:
    return wrapping_string.format(x)


def apply(*args, **kwargs):
    def apply_inner(function):
        return function(*args, **kwargs)

    return apply_inner


@currying.curry
def drop_last_while(predicate: Callable[[Any], bool], seq: Sequence) -> Sequence:
    return toolz.pipe(
        seq,
        reversed,
        currying.curry(itertools.dropwhile)(predicate),
        tuple,
        reversed,
    )


@currying.curry
def partition_after(
    predicate: Callable[[Any], bool],
    seq: Sequence,
) -> Sequence[Sequence]:
    return toolz.reduce(
        lambda a, b: (*a, (b,))
        if not a or predicate(a[-1][-1])
        else (*a[:-1], (*a[-1], b)),
        seq,
        (),
    )


@currying.curry
def partition_before(
    predicate: Callable[[Any], bool],
    seq: Sequence,
) -> Sequence[Sequence]:
    return toolz.reduce(
        lambda a, b: (*a, (b,)) if not a or predicate(b) else (*a[:-1], (*a[-1], b)),
        seq,
        (),
    )


def get_all_n_grams(seq):
    return toolz.pipe(
        range(1, len(seq) + 1),
        curried.mapcat(curried.sliding_window(seq=seq)),
    )


@currying.curry
def is_instance(the_type, the_value):
    return type(the_value) == the_type


def sample(n: int):
    def sample_inner(population):
        return random.sample(population, n)

    return sample_inner


@currying.curry
def eq_by(f, value_1, value_2):
    return f(value_1) == f(value_2)


eq_str_ignore_case = eq_by(str.lower)


@currying.curry
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


@currying.curry
def take_while(pred, seq):
    for x in seq:
        if not pred(x):
            return
        yield x


@currying.curry
def take_last_while(pred, seq):
    return toolz.pipe(
        seq,
        reduce(
            lambda acc, elem: suffix(elem, acc) if pred(elem) else (),
            (),
        ),
    )


def unique_by(f):
    """Return only unique elements of a sequence defined by function f

    >>> tuple(unique_by(['cat', 'mouse', 'dog', 'hen'], f=len))
    ('cat', 'mouse')
    """

    def unique(seq):
        seen = set()
        for item in seq:
            val = f(item)
            if val not in seen:
                seen.add(val)
                yield item

    return unique


unique = unique_by(identity)


def attrgetter(attr):
    def attrgetter(obj):
        return getattr(obj, attr)

    return attrgetter


def itemgetter(attr):
    def itemgetter(obj):
        return operator.getitem(obj, attr)

    return itemgetter


def itemgetter_or_none(attr):
    def itemgetter_or_none(obj):
        return obj.get(attr, None)

    return itemgetter_or_none


def itemgetter_with_default(default, attr):
    return toolz.excepts(
        (KeyError, IndexError),
        itemgetter(attr),
        just(default),
    )


def equals(x):
    def equals(y):
        return x == y

    return equals


def not_equals(x):
    def not_equals(y):
        return x != y

    return not_equals


def contains(x):
    def contains(y):
        return y in x

    return contains


def add(x):
    def add(y):
        return y + x

    return add


def greater_than(x):
    def greater_than(y):
        return y > x

    return greater_than


def greater_equals(x):
    def greater_equals(y):
        return y >= x

    return greater_equals


def less_than(x):
    def less_than(y):
        return y < x

    return less_than


def less_equals(x):
    def less_equals(y):
        return y <= x

    return less_equals


def multiply(x):
    def multiply(y):
        return y * x

    return multiply


def divide_by(x):
    def divide_by(y):
        return y / x

    return divide_by


_GET_IN_EXCEPTIONS = (KeyError, IndexError, TypeError)


def get_in(keys):
    def get_in(coll):
        return functools.reduce(operator.getitem, keys, coll)

    return get_in


def get_in_with_default(keys, default):
    return toolz.excepts(_GET_IN_EXCEPTIONS, get_in(keys), just(default))


def get_in_or_none(keys):
    return get_in_with_default(keys, None)


def get_in_or_none_uncurried(keys, coll):
    return get_in_or_none(keys)(coll)


def interpose(el):
    def interpose_inner(seq):
        return toolz.interpose(el, seq)


def tail(n: int):
    def tail(seq):
        return toolz.tail(n, seq)

    return tail


frequencies = toolz.frequencies 

head = toolz.first
head.__doc__ = """ The first element in a sequence bla

    >>> first('ABC')
    'A'
    """
#: The second element in a sequence >>> second([1,2,3]) '2'
second = toolz.second
last = toolz.last
