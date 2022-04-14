import dataclasses
import functools
import hashlib
import heapq
import inspect
import itertools
import json
import os
import random
from concurrent import futures
from operator import truediv
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Sequence,
    Text,
    Tuple,
    TypeVar,
)

import heapq_max
import immutables
import toolz

from gamla import construct, currying, excepts_decorator, operator
from gamla.optimized import sync


def sort_by(key: Callable):
    """Return a new list containing all items from the iterable in ascending order, sorted by a key.
    >>> sort_by(len)(["hi!", "my", "name", "is"])
    ['my', 'is', 'hi!', 'name']
    """

    def sort_by(seq: Iterable):
        return sorted(seq, key=key)

    return sort_by


def sort_by_reversed(key: Callable):
    """Return a new list containing all items from the iterable in descending order, sorted by a key.
    >>> sort_by_reversed(lambda x: x % 10)([2231, 47, 19, 100])
    [19, 47, 2231, 100]
    """

    def sort_by_reversed(seq: Iterable):
        return sorted(seq, key=key, reverse=True)

    return sort_by_reversed


#: Return a new list containing all items from the iterable in ascending order
#: >>> sort([5,2,4,1])
#: '[1,2,4,5]'
sort = sort_by(operator.identity)

#: Return a new list containing all items from the iterable in descending order
#: >>> sort([5,2,4,1])
#: '[5,4,2,1]'
sort_reversed = sort_by_reversed(operator.identity)


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
        return operator.head(func((some_input,)))

    async def wrapped_async(some_input):
        if isinstance(some_input, tuple):
            return await func(some_input)
        return operator.head(await func((some_input,)))

    if inspect.iscoroutinefunction(func):
        return wrapped_async
    return wrapped


def ignore_input(inner: Callable[[], Any]) -> Callable:
    """Returns `inner` function ignoring the provided inputs.

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
    """Raises the given exception.

    >>> just_raise(KeyError)
    raise exception KeyError
    """
    raise exception


def make_raise(exception):
    """Returns a function that ignores the input and just raises the given exception.

    >>> f = make_raise(KeyError)
    >>> f(3)
    raise exception KeyError
    """

    def inner():
        raise exception

    return ignore_input(inner)


@currying.curry
def translate_exception(func: Callable, exc1: Exception, exc2: Exception):
    """A functional try/except block: if `func` fails with `exc1`, raise `exc2`.

    >>> from gamla import functional_generic
    >>> functional_generic.pipe(iter([]), translate_exception(next, StopIteration, ValueError))
    ValueError
    Note: `func` is assumed to be unary."""
    return excepts_decorator.excepts(exc1, make_raise(exc2), func)


def to_json(obj):
    """Return a `JSON` representation of a 'dictionary' or an object.

    >>> to_json({"one": 1, "two": 2})
    '{"one": 1, "two": 2}'
    """
    if hasattr(obj, "to_json"):
        return obj.to_json()
    return json.dumps(obj)


def compute_stable_json_hash(item) -> Text:
    """Only works on json valid data types."""
    return hashlib.sha1(
        json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ).hexdigest()


@currying.curry
def assert_that_with_message(input_to_message: Callable, f: Callable):
    """Assert a function `f` on the input, printing the output of `input_to_message(input)` if assertion is False.

    >>> assert_that_with_message(just("Input is not 2!"), equals(2))(2)
    2
    >>> assert_that_with_message(just("Input is not 2!"), equals(2))(3)
    "Output is not 2!"
    """

    def assert_that_f(inp):
        assert f(inp), input_to_message(inp)
        return inp

    return assert_that_f


assert_that = assert_that_with_message(construct.just(""))


@currying.curry
def pmap(f, n_workers, it):
    # The `tuple` is for callers convenience (even without it, the pool is eager).
    return tuple(futures.ThreadPoolExecutor(max_workers=n_workers).map(f, it))


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
def top(iterable: Iterable, key=operator.identity):
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
        yield operator.nth(2)(heapq_max.heappop_max(h))


@currying.curry
def bottom(iterable, key=operator.identity):
    """Generates elements from min to max.

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
        yield operator.nth(2)(heapq.heappop(h))


def skip(n: int):
    """Skip the first n elements of a sequence. i.e, Return a generator that yields all elements after the n'th element.
    >>> tuple(skip(3)([i for i in range(6)]))
    (3, 4, 5)
    """

    def skip(seq: Iterable):
        for i, x in enumerate(seq):
            if i < n:
                continue
            yield x

    return skip


@currying.curry
def assoc_in(d, keys, value, factory=dict):
    """Associate a value to the input dict given the path "keys".

    >>> assoc_in({"a": {"b": 1}}, ["a", "b"], 2)
    {'a': {'b': 2}}
    """
    return update_in(d, keys, lambda x: value, value, factory)


@currying.curry
def update_in(d: dict, keys: Iterable, func: Callable, default=None, factory=dict):
    """Gets a (potentially nested) dictionary, key(s) and a function, and return new `dictionary` d' where d'[key] = func(d[key]).

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


def dataclass_transform(
    attr_name: Text,
    attr_transformer: Callable[[Any], Any],
):
    """Return a new instance of the dataclass where new_dataclass_instance.attr_name = attr_transformer(dataclass_instance.attr_name)
    >>> @dataclasses.dataclass(frozen=True)
    ... class C:
    ...    x: int
    >>> c = C(5)
    >>> d = dataclass_transform('x', lambda i: i * 2)(c)
    >>> assert d.x == 10
    """
    transformation = sync.compose_left(
        operator.attrgetter(attr_name),
        attr_transformer,
    )

    def dataclass_transform(dataclass_instance):
        return dataclasses.replace(
            dataclass_instance,
            **{
                attr_name: transformation(dataclass_instance),
            },
        )

    return dataclass_transform


dataclass_transform_attribute = sync.binary_curry(dataclass_transform)


def dataclass_replace(attr_name: Text, attr_value) -> Callable:
    return dataclass_transform(attr_name, lambda _: attr_value)


dataclass_replace_attribute = sync.binary_curry(dataclass_replace)

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
    """Add a value to the end of an iterable. Return an iterable.

    >>> tuple(suffix(4, (1, 2, 3)))
    (1, 2, 3, 4)
    """
    return itertools.chain(it, (val,))


@currying.curry
def prefix(val: Any, it: Iterable):
    """Add a value to the beginning of an iterable. Return an iterable.

    >>> tuple(prefix(1, (2, 3, 4)))
    (1, 2, 3, 4)
    """
    return itertools.chain((val,), it)


@currying.curry
def concat_with(new_it: Iterable, it: Iterable):
    """Concat two iterables.

    >>> tuple(concat_with((3, 4), (1, 2)))
    (1, 2, 3, 4)
    """
    return itertools.chain(it, new_it)


@currying.curry
def drop_last_while(predicate: Callable[[Any], bool], seq: Sequence) -> Sequence:
    return sync.pipe(
        seq,
        reversed,
        currying.curry(itertools.dropwhile)(predicate),
        tuple,
        reversed,
    )


def take_while(predicate: Callable[[Any], bool]):
    """Take elements from an iterable as long as elements pass some predicate.

    >>> list(functional.take_while(lambda x: x < 7)([1, 2, 9, 2]))
    [1, 2]
    """

    def take_while(iterable: Iterable) -> Iterable:
        for x in iterable:
            if not predicate(x):
                break
            yield x

    return take_while


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


def get_all_n_grams(seq: Sequence) -> Iterable[Tuple]:
    for i in range(len(seq)):
        for j in range(i + 1, len(seq) + 1):
            yield tuple(seq[i:j])


@currying.curry
def eq_by(f, value_1, value_2):
    """Check if two values are equal when applying f on both of them."""
    return f(value_1) == f(value_2)


#: Check if two strings are equal, ignoring case.
#: >>> eq_str_ignore_case("HeLlO wOrLd", "hello world")
#: True
eq_str_ignore_case = eq_by(str.lower)


@currying.curry
def groupby_many_reduce(key: Callable, reducer: Callable, seq: Iterable):
    """Group a collection by a key function, when the value is given by a reducer function.

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
unique = unique_by(operator.identity)


def interpose(el):
    """Introduces an element between each pair of elements in the input sequence.

    >>> tuple(interpose("a")([1, 2, 3]))
    (1, 'a', 2, 'a', 3)
    """

    def interpose_inner(seq):
        return toolz.interpose(el, seq)

    return interpose_inner


def take(n: int):
    """Get an iterator for the first n elements of a sequence.

    >>> tuple(take(3) ([1, 2, 3, 4, 5]))
    (1, 2, 3)
    """

    def take(seq):
        return itertools.islice(seq, n)

    return take


def drop(n: int):
    """Drops the first n elements of a sequence.

    >>> tuple(drop(2)([1,2,3,4,5]))
    (3,4,5)
    """

    def drop(seq):
        return itertools.islice(seq, n, None)

    return drop


def drop_last(n: int):
    """Drops the last n elements of a sequence.

    >>> tuple(drop_last(1)([1,2,3,4,5]))
    (1,2,3,4)
    """

    def drop_last(seq):
        return itertools.islice(seq, len(seq) - n)

    return drop_last


def flip(func: Callable):
    """Call the function call with the arguments flipped.

    >>> import operator; flip(operator.truediv)(2, 6)
    3.0
    """

    @currying.curry
    def flip(a, b):
        return func(b, a)

    return flip


def sliding_window(n: int):
    """A sequence of overlapping subsequences.

    >>> list(sliding_window(2)([1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]
    """

    def sliding_window(seq):
        return toolz.sliding_window(n, seq)

    return sliding_window


def partition_all(n: int):
    """Partition all elements of sequence into tuples of length at most n.
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
    """Returns a predicate that checks if an iterabel ends with another iterable.

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
        tail_of_seq = operator.tail(len(expected_tail_as_tuple))(seq)
        for a, b in itertools.zip_longest(
            tail_of_seq,
            expected_tail_as_tuple,
            fillvalue=Nothing(),
        ):
            if a != b:
                return False
        return True

    return ends_with


def intersect(collections: Collection[Collection]) -> Iterable:
    """Intersects a group of collections.

    Caller is responsible to give collections with O(1) containment checks.

    >>> tuple(
            gamla.intersect(
                [
                    [1, 2, 3, 4],
                    [4, 5],
                    [4, 2],
                ]
            )
        )
    (4,)

    """
    first_collection, *rest = collections
    for x in first_collection:
        if all(x in container for container in rest):
            yield x


have_intersection = sync.compose_left(intersect, operator.nonempty)


def function_to_uid(f: Callable) -> str:
    """Returns a unique identifier for the given function."""
    return hashlib.sha1(f.__name__.encode("utf-8")).hexdigest()


#: Directory path of a given function
function_to_directory = sync.compose_left(
    operator.attrgetter("__code__"),
    operator.attrgetter("co_filename"),
    os.path.dirname,
)


def function_and_input_to_identifier(factory) -> Callable:
    """Returns a unique identifier for the given function and input."""

    def inner(args, kwargs) -> str:
        return sync.pipe(
            (
                function_to_uid(factory),
                compute_stable_json_hash(make_call_key(args, kwargs)),
            ),
            sync.filter(operator.identity),
            "-".join,
        )

    return inner


#: Average of an iterable. If the sequence is empty, returns 0.
#: >>> average([1,2,3])
#: 2.0
average = sync.compose_left(
    sync.bifurcate(sum, operator.count),
    excepts_decorator.excepts(
        ZeroDivisionError,
        construct.just(0),
        sync.star(truediv),
    ),
)


def attr_equals(attribute: str, equals_what: Any) -> Callable[[Any], bool]:
    """Returns a function that get an object x and returns whether x.attribute == equals_what

    >>> attr_equals("imag", 5.0)(8 + 5j)
    True
    >>> attr_equals("imag", 5.0)(8)
    False
    """
    return sync.compose_left(
        operator.attrgetter(attribute),
        operator.equals(equals_what),
    )


def sample_with_randint(randint: Callable, k: int):
    """Samples an iterable uniformly in one pass with O(k) memory.

    >>> sample(2)([1, 2, 3])
    frozenset([1,3])
    """

    def reducer(
        index_and_sample: Tuple[int, immutables.Map],
        current,
    ) -> Tuple[int, immutables.Map]:
        index, sample = index_and_sample
        replacement_index = index if index < k else randint(0, index)
        return (
            index + 1,
            sample.set(replacement_index, current) if replacement_index < k else sample,
        )

    return sync.compose_left(
        reduce(reducer, (0, immutables.Map())),
        operator.second,
        immutables.Map.values,
        frozenset,
    )


sample = currying.curry(sample_with_randint)(random.randint)
choice = sync.compose(operator.head, sample_with_randint(random.randint, 1))
