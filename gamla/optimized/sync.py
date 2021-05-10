"""Synchronous versions of common functions for optimized use cases."""
import itertools


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


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def star(f):
    def starred(args):
        return f(*args)

    return starred


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def pair_left(f):
    def pair_left(x):
        return f(x), x

    return pair_left


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def reduce(f, initial):
    def reduce(it):
        state = initial
        for element in it:
            state = f(state, element)
        return state

    return reduce


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
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


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
merge = star(merge_with_reducer(lambda _, x: x))


# TODO(uri): This might be used to optimize functions in gamla instead of its generic counterpart.
def after(f):
    def after(g):
        return compose_left(g, f)

    return after


juxtduct = compose_left(juxt, after(star(itertools.product)))
mapdict = compose_left(map, after(dict))
mapduct = compose_left(map, after(star(itertools.product)))
maptuple = compose_left(map, after(tuple))
