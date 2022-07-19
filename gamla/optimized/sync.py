"""Synchronous versions of common functions for optimized use cases."""
import itertools
from typing import Any, Callable, Iterable, Tuple, Union

from gamla import construct, operator


def packstack(*functions):
    def packstack(values):
        return tuple(f(x) for f, x in zip(functions, values))

    return packstack


def keyfilter(predicate):
    def keyfilter(d):
        new_d = {}
        for k in d:
            if predicate(k):
                new_d[k] = d[k]
        return new_d

    return keyfilter


def valfilter(predicate):
    """Create a function that filters a dict using a predicate over values.
    >>> f = valefilter(lambda k: k > 2)
    >>> f({"a": 1, "b": 2, "c": 3, "d": 4})
    {
    "c": 3,
    "d": 4
    }
    """

    def valfilter(d):
        new_d = {}
        for (k, v) in d.items():
            if predicate(v):
                new_d[k] = d[k]
        return new_d

    return valfilter


def mapcat(f):
    def mapcat(it):
        for i in it:
            yield from f(i)

    return mapcat


def juxtcat(*functions):
    def juxtcat(x):
        for f in functions:
            for result in f(x):
                yield result

    return juxtcat


def map(f):
    def map_curried(it):
        for x in it:
            yield f(x)

    return map_curried


def filter(f):
    def filter(it):
        for x in it:
            if f(x):
                yield x

    return filter


def remove(f):
    def remove(it):
        for x in it:
            if not f(x):
                yield x

    return remove


def juxt(*functions):
    def juxt(*args, **kwargs):
        return tuple(f(*args, **kwargs) for f in functions)

    return juxt


def valmap(mapper):
    def valmap(d):
        new_d = {}
        for k in d:
            new_d[k] = mapper(d[k])
        return new_d

    return valmap


def keymap(mapper):
    def keymap(d):
        new_d = {}
        for k in d:
            new_d[mapper(k)] = d[k]
        return new_d

    return keymap


def groupby(grouper):
    def groupby(it):
        d = {}
        for x in it:
            key = grouper(x)
            if key not in d:
                d[key] = []
            d[key].append(x)
        return d

    return groupby


def groupby_many(grouper):
    def groupby_many(it):
        d = {}
        for x in it:
            for key in grouper(x):
                if key not in d:
                    d[key] = []
                d[key].append(x)
        return d

    return groupby_many


def ternary(condition, f_true, f_false):
    def ternary(*args, **kwargs):
        if condition(*args, **kwargs):
            return f_true(*args, **kwargs)
        return f_false(*args, **kwargs)

    return ternary


def check(condition, exception):
    """Apply function `condition` to value, raise `exception` if return value is `False`-ish or return the value as-is.

    >>> f = check(gamla.greater_than(10), ValueError)
    >>> f(5)
    `ValueError`
    >>> f(15)
    15
    """

    def check(x):
        if condition(x):
            return x
        raise exception

    return check


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def compose(*functions):
    """Compose sync functions to operate in series.

    Returns a function that applies other functions in sequence.

    Functions are applied from right to left so that
    ``compose(f, g, h)(x, y)`` is the same as ``f(g(h(x, y)))``.

    >>> inc = lambda i: i + 1
    >>> compose(str, inc)(3)
    '4'

    """

    def compose(*args, **kwargs):
        for f in reversed(functions):
            args = [f(*args, **kwargs)]
            kwargs = {}
        return args[0]

    return compose


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def compose_left(*functions):
    def compose_left(*args, **kwargs):
        for f in functions:
            x = f(*args, **kwargs)
            args = (x,)
            kwargs = {}
        return x

    return compose_left


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def pipe(x, *functions):
    return compose_left(*functions)(x)


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def anyjuxt(*functions):
    def anyjuxt(x):
        for f in functions:
            if f(x):
                return True
        return False

    return anyjuxt


def alljuxt(*functions):
    def alljuxt(x):
        for f in functions:
            if not f(x):
                return False
        return True

    return alljuxt


def complement(f):
    def complement(*args, **kwargs):
        return not f(*args, **kwargs)

    return complement


def allmap(f):
    def allmap(xs):
        for x in xs:
            if not f(x):
                return False
        return True

    return allmap


def anymap(f):
    def anymap(xs):
        for x in xs:
            if f(x):
                return True
        return False

    return anymap


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def star(f):
    def starred(args):
        return f(*args)

    return starred


def double_star(f):
    def double_star(kwargs):
        return f(**kwargs)

    return double_star


def pair_left(f):
    def pair_left(x):
        return f(x), x

    return pair_left


def pair_right(f):
    """Returns a function that given a value x, returns a tuple of the form: (x, f(x)).

    >>> add_one = pair_right(lambda x: x + 1)
    >>> add_one(3)
    (3, 4)
    """

    def pair_right(x):
        return x, f(x)

    return pair_right


def reduce(f, initial):
    def reduce(it):
        state = initial
        for element in it:
            state = f(state, element)
        return state

    return reduce


def merge_with_reducer(reducer):
    def merge_with_reducer(*dictionaries):
        new_d = {}
        for d in dictionaries:
            for k, v in d.items():
                if k in new_d:
                    new_d[k] = reducer(new_d[k], v)
                else:
                    new_d[k] = v
        return new_d

    return merge_with_reducer


merge = star(merge_with_reducer(lambda _, x: x))


def after(f):
    def after(g):
        return compose_left(g, f)

    return after


def before(f):
    def before(g):
        return compose(g, f)

    return before


juxtduct = compose_left(juxt, after(star(itertools.product)))
mapdict = compose_left(map, after(dict))
mapduct = compose_left(map, after(star(itertools.product)))
maptuple = compose_left(map, after(tuple))


def binary_curry(f):
    def binary_curry(x):
        def binary_curry(y):
            return f(x, y)

        return binary_curry

    return binary_curry


def thunk(f, *args, **kwargs):
    def thunk(*inner_args, **inner_kwargs):
        return f(*args, **kwargs)(*inner_args, **inner_kwargs)

    return thunk


def when(f, g):
    def when(x):
        if f(x):
            return g(x)
        return x

    return when


class NoConditionMatched(Exception):  # noqa
    pass


def case(predicates_and_mappers: Tuple[Tuple[Callable, Callable], ...]):
    def case(*args, **kwargs):
        for predicate, transformation in predicates_and_mappers:
            if predicate(*args, **kwargs):
                return transformation(*args, **kwargs)
        raise NoConditionMatched({"input args": args, "input kwargs": kwargs})

    return case


case_dict = compose_left(dict.items, tuple, case)
stack = compose_left(
    enumerate,
    map(star(lambda i, f: compose(f, lambda x: x[i]))),
    star(juxt),
)

_is_terminal = anyjuxt(*map(operator.is_instance)([str, int, float]))


def map_dict(nonterminal_mapper: Callable, terminal_mapper: Callable) -> Callable:
    recurse = thunk(
        map_dict,
        nonterminal_mapper,
        terminal_mapper,
    )
    return case_dict(
        {
            _is_terminal: terminal_mapper,
            operator.is_instance(dict): compose_left(
                valmap(recurse),
                nonterminal_mapper,
            ),
            operator.is_iterable: compose_left(
                map(recurse),
                tuple,
                nonterminal_mapper,
            ),
            # Other types are considered terminals to support things like `apply_spec`.
            construct.just(True): terminal_mapper,
        },
    )


def bifurcate(*funcs):
    """Serially run each function on tee'd copies of a sequence.
    If the sequence is a generator, it is duplicated so it will not be exhausted (which may incur a substantial memory signature in some cases).
    >>> f = bifurcate(sum, gamla.count)
    >>> seq = map(gamla.identity, [1, 2, 3, 4, 5])
    >>> f(seq)
    (15, 5)
    """
    return compose_left(iter, lambda it: itertools.tee(it, len(funcs)), stack(funcs))


_should_be_concatenated = anyjuxt(
    *map(operator.is_instance)([frozenset, set, tuple, list])
)


def flatten(iterable: Iterable[Union[Any, Iterable]]) -> Iterable[Any]:
    """Flatten a given iterable recursively.
    >>> iter = [1, "a", frozenset({"something"}), (("hi", 6),)]
    >>> flatten(iter)
    (
        1,
        "a",
        "something",
        "hi",
        6,
    )
    """
    if anymap(_should_be_concatenated)(iterable):
        return flatten(
            pipe(
                iterable,
                map(when(complement(_should_be_concatenated), construct.wrap_tuple)),
                operator.concat,
                tuple,
            ),
        )
    return iterable
