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
from typing import Any, Callable, Dict, Iterable, List, Sequence, Text, TypeVar

import heapq_max
import toolz
from toolz import curried

from gamla import currying


def identity(x):
    return x


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


def ignore_input(inner: Callable[[], Any]) -> Callable:
    """
    Returns `inner` function ignoring the provided inputs.

    >>> ignore_input(lambda: 0)(1)
    0
    """

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
def translate_exception(func: Callable, exc1: Exception, exc2: Exception):
    """
    A functional try/except block: if `func` fails with `exc1`, raise `exc2`.

    >>> from gamla import functional_generic
    >>> functional_generic.pipe(iter([]), translate_exception(next, StopIteration, ValueError))
    ValueError
    Note: `func` is assumed to be unary."""
    return toolz.excepts(exc1, func, make_raise(exc2))


def to_json(obj):
    """
    Return a `JSON` representation of a 'dictionary' or an object.

    >>> to_json({"one": 1, "two": 2})
    '{"one": 1, "two": 2}'
    """
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
    """
    Turns a variadic function into an unary one that gets a tuple of args to the original function.

    >>> from gamla import functional_generic
    >>> functional_generic.pipe((2, 3), star(lambda x, y: x + y))
    5
    """

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
    """
    Assert a function `f` on the input.

    >>> assert_that(equals(2))(2)
    2
    """
    return curried.do(_assert_f_output_on_inp(f))


@currying.curry
def pmap(f, n_workers, it):
    # The `tuple` is for callers convenience (even without it, the pool is eager).
    return tuple(futures.ThreadPoolExecutor(max_workers=n_workers).map(f, it))


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
def top(iterable: Iterable, key=identity):
    """Generates elements from max to min.

    >>> tuple(top((1, 3, 2)))
    (3, 2, 1)

    >>> tuple(top(('a', 'aa', 'aaa'), len))
    ('aaa', 'aa', 'a')
    """
    h: List = []
    for i, value in enumerate(iterable):
        # Use the index as a tie breaker.
        heapq_max.heappush_max(h, (key(value), i, value))
    while h:
        yield toolz.nth(2, heapq_max.heappop_max(h))


@currying.curry
def bottom(iterable, key=identity):
    """
    Generates elements from min to max.

    >>> tuple(bottom((3, 2, 1)))
    (1, 2, 3)

    >>> tuple(bottom((1, 2, 3, 4), lambda x: x % 2 == 0))
    (1, 3, 2, 4)
    """
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
    """
    In operator.

    >>> inside(1, [0, 1, 2])
    True

    >>> inside("a", "word")
    False
    """
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


def wrap_tuple(x: Any):
    """
    Wrap an element in a tuple.

    >>> wrap_tuple("hello")
    ('hello',)
    """
    return (x,)


def wrap_frozenset(x):
    """Wraps x with frozenset.

    >>> wrap_frozenset(1)
    frozenset({1})
    """
    return frozenset([x])


@currying.curry
def assoc_in(d, keys, value, factory=dict):
    """
    Associate a value to the input dict given the path "keys".

    >>> assoc_in({"a": {"b": 1}}, ["a", "b"], 2)
    {'a': {'b': 2}}
    """
    return update_in(d, keys, lambda x: value, value, factory)


def add_key_value(key, value):
    """
    Associate a key-value pair to the input dict.

    >>> add_key_value("1", "1")({"2": "2"})
    {'2': '2', '1': '1'}
    """

    def add_key_value(d):
        return assoc_in(d, [key], value)

    return add_key_value


def remove_key(key):
    def remove_key(d):
        updated = d.copy()
        del updated[key]
        return updated

    return remove_key


def wrap_dict(key):
    """
    Wrap a key and a value in a dict (in a curried fashion).

    >>> wrap_dict("one") (1)
    {'one': 1}
    """

    def wrap_dict(value):
        return {key: value}

    return wrap_dict


@currying.curry
def update_in(d: dict, keys: Iterable, func: Callable, default=None, factory=dict):
    """
    Gets a (potentially nested) dictionary, key(s) and a function, and return new `dictionary` d' where d'[key] = func(d[key]).

    >>> inc = lambda x: x + 1
    >>> update_in({'a': 0}, ['a'], inc)
    {'a': 1}
    """
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
def suffix(val: Any, it: Iterable):
    """
    Add a value to the end of an iterable. Return an iterable.

    >>> tuple(suffix(4, (1, 2, 3)))
    (1, 2, 3, 4)
    """
    return itertools.chain(it, (val,))


@currying.curry
def prefix(val: Any, it: Iterable):
    """
    Add a value to the beginning of an iterable. Return an iterable.

    >>> tuple(prefix(1, (2, 3, 4)))
    (1, 2, 3, 4)
    """
    return itertools.chain((val,), it)


@currying.curry
def concat_with(new_it: Iterable, it: Iterable):
    """
    Concat two iterables.

    >>> tuple(concat_with((3, 4), (1, 2)))
    (1, 2, 3, 4)
    """
    return itertools.chain(it, new_it)


@currying.curry
def wrap_str(wrapping_string: Text, x: Text) -> Text:
    """
    Wrap a string in a wrapping string.

    >>> wrap_str("hello {}", "world")
    'hello world'
    """
    return wrapping_string.format(x)


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
    """
    Returns if `the_value` is an instance of `the_type`.

    >>> is_instance(str, "hello")
    True

    >>> is_instance(int, "a")
    False
    """
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

    >>> groupby_many_reduce(head, lambda x, y: x + len(y) if x else len(y), ["hello", "hi", "test", "to"])
    {'h': 7, 't': 6}
    """
    result: Dict[Any, Any] = {}
    for element in seq:
        for key_result in key(element):
            result[key_result] = reducer(result.get(key_result, None), element)
    return result


def unique_by(f):
    """Return only unique elements of a sequence defined by function f

    >>> tuple(unique_by(len)(['cat', 'mouse', 'dog', 'hen']))
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


#: Return only unique elements of a sequence
#:
#: >>> tuple(unique(["cat", "mouse", "dog", "cat"]))
#: ('cat', 'mouse', 'dog')
unique = unique_by(identity)


def attrgetter(attr):
    """
    Access the object attribute by its name `attr`.

    >>> attrgetter("lower")("ASD")()
    'asd'
    """

    def attrgetter(obj):
        return getattr(obj, attr)

    return attrgetter


def equals(x):
    def equals(y):
        return x == y

    return equals


def not_equals(x):
    def not_equals(y):
        return x != y

    return not_equals


def contains(x):
    """
    Contains operator.

    >>> contains([1, 2, 3])(2)
    True

    >>> contains("David")("x")
    False
    """

    def contains(y):
        return y in x

    return contains


def add(x):
    """
    Addition operator.

    >>> add(1)(2)
    3

    >>> add(["c"])(["a", "b"])
    ['a', 'b', 'c']
    """

    def add(y):
        return y + x

    return add


def greater_than(x):
    """
    Greater than operator.

    >>> greater_than(1)(2)
    True

    >>> greater_than(1)(0)
    False
    """

    def greater_than(y):
        return y > x

    return greater_than


def greater_equals(x):
    """Greater than or equal operator.

    >>> greater_equals(1)(1)
    True

    >>> greater_equals(1)(0)
    False
    """

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


def interpose(el):
    """
    Introduces an element between each pair of elements in the input sequence.

    >>> tuple(interpose("a")([1, 2, 3]))
    (1, 'a', 2, 'a', 3)
    """

    def interpose_inner(seq):
        return toolz.interpose(el, seq)

    return interpose_inner


def tail(n: int):
    """
    Get the last n elements of a sequence.

    >>> tail(3) ([1, 2, 3, 4, 5])
    [3, 4, 5]
    """

    def tail(seq: Iterable):
        return toolz.tail(n, seq)

    return tail


def take(n: int):
    """
    Get an iterator for the first n elements of a sequence.

    >>> tuple(take(3) ([1, 2, 3, 4, 5]))
    (1, 2, 3)
    """

    def take(seq):
        return itertools.islice(seq, n)

    return take


def nth(n: int):
    def nth(seq):
        return toolz.nth(n, seq)

    return nth


def drop(n: int):
    def drop(seq):
        return toolz.drop(n, seq)

    return drop


def flip(func: Callable):
    """
    Call the function call with the arguments flipped.

    >>> import operator; flip(operator.truediv)(2, 6)
    3.0
    """

    @currying.curry
    def flip(a, b):
        return func(b, a)

    return flip


frequencies = toolz.frequencies

#: The first element in a sequence.
#:
#:    >>> head('ABC')
#:    'A'
head = toolz.first

#:  The second element in a sequence.
#:
#:    >>> second('ABC')
#:    'B'
second = toolz.second

#: The last element in a sequence.
#:
#:    >>> last('ABC')
#:    'C'
last = toolz.last

#: Determines whether the element is iterable.
#:
#:    >>> isiterable([1, 2, 3])
#:    True
#:
#:    >>> isiterable(5)
#:    False
is_iterable = toolz.isiterable


def sliding_window(n: int):
    """
    A sequence of overlapping subsequences.

    >>> list(sliding_window(2)([1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]
    """

    def sliding_window(seq):
        return toolz.sliding_window(n, seq)

    return sliding_window


def partition_all(n: int):
    """
    Partition all elements of sequence into tuples of length at most n.
    The final tuple may be shorter to accommodate extra elements.

    >>> list(partition_all(2)([1, 2, 3, 4]))
    [(1, 2), (3, 4)]

    >>> list(partition_all(2)([1, 2, 3, 4, 5]))
    [(1, 2), (3, 4), (5,)]
    """

    def partition_all(seq):
        return toolz.partition_all(n, seq)

    return partition_all


def ends_with(expected_tail: Iterable) -> Callable[[Sequence], bool]:
    """
    Returns a predicate that checks if an iterabel ends with another iterable.

    >>> ends_with([1,2,3])((0,1,2,3))
    True
    >>> ends_with([1,2,3])((1,2))
    False
    >>> ends_with([1,2])((3,1,2))
    True
    >>> ends_with([1])(())
    False
    """

    class Nothing:
        pass

    expected_tail_as_tuple = tuple(expected_tail)

    def ends_with(seq: Iterable):
        tail_of_seq = tail(len(expected_tail_as_tuple))(seq)
        for a, b in itertools.zip_longest(
            tail_of_seq,
            expected_tail_as_tuple,
            fillvalue=Nothing(),
        ):
            if a != b:
                return False
        return True

    return ends_with
