import functools
import operator
from typing import Dict

from gamla import currying


def itemgetter(attr):
    def itemgetter(obj):
        return obj[attr]

    return itemgetter


def itemgetter_or_none(attr):
    def itemgetter_or_none(obj):
        return obj.get(attr, None)

    return itemgetter_or_none


def itemgetter_with_default(default, attr):
    def itemgetter_with_default(d):
        try:
            return d[attr]
        except (KeyError, IndexError):
            return default

    return itemgetter_with_default


def get_or_identity(d):
    def get_or_identity(x):
        return d.get(x, x)

    return get_or_identity


def get_in(keys):
    def get_in(coll):
        return functools.reduce(operator.getitem, keys, coll)

    return get_in


def get_in_with_default(keys, default):
    getter = get_in(keys)

    def get_in_with_default(x):
        try:
            return getter(x)
        except (KeyError, IndexError, TypeError):
            return default

    return get_in_with_default


def get_in_or_none(keys):
    return get_in_with_default(keys, None)


def get_in_or_none_uncurried(keys, coll):
    return get_in_or_none(keys)(coll)


@currying.curry
def dict_to_getter_with_default(default, d: Dict):
    """Turns a dictionary into a function from key to value or default if key is not there.

    >>> dict_to_getter_with_default(None, {1:1})(1)
    1
    >>> dict_to_getter_with_default(None, {1:1})(2)
    None
    """

    def dict_to_getter_with_default(key):
        return d.get(key, default)

    return dict_to_getter_with_default
