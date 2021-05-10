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


# TODO(uri): Anything below this line was not deduplicated.


def remove(f):
    def remove(it):
        for x in it:
            if not f(x):
                yield x

    return remove


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


def compose(*functions):
    def compose(x):
        for f in reversed(functions):
            x = f(x)
        return x

    return compose


def compose_left(*functions):
    def compose_left(*args, **kwargs):
        for f in functions:
            x = f(*args, **kwargs)
            args = (x,)
            kwargs = {}
        return x

    return compose_left


def pipe(x, *functions):
    return compose_left(*functions)(x)


def ternary(c, f, g):
    def ternary(x):
        if c(x):
            return f(x)
        return g(x)

    return ternary


def check(f, exception):
    def check(x):
        if not f(x):
            raise exception
        return x

    return check


def anyjuxt(*functions):
    def anyjuxt(x):
        for f in functions:
            if f(x):
                return True
        return False

    return anyjuxt


def star(f):
    def starred(args):
        return f(*args)

    return starred


def pair_left(f):
    def pair_left(x):
        return f(x), x

    return pair_left


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


def juxt(*functions):
    def juxt(*args, **kwargs):
        return tuple(f(*args, **kwargs) for f in functions)

    return juxt


juxtduct = compose_left(juxt, after(star(itertools.product)))
mapdict = compose_left(map, after(dict))
mapduct = compose_left(map, after(star(itertools.product)))
maptuple = compose_left(map, after(tuple))
