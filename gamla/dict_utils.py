import functools
from operator import getitem
from typing import Any, Callable, Dict, Iterable

from gamla import construct, currying, functional, functional_generic, operator
from gamla.optimized import sync


def itemgetter(attr):
    """Retrieves a dictionary entry by its key `attr`.

    >>> itemgetter("a")({"a": 1})
    1
    """

    def itemgetter(obj):
        return obj[attr]

    return itemgetter


def itemgetter_or_none(attr):
    """Retrieves a dictionary entry by its key `attr`. Returns `None` if the key is not there.

    >>> itemgetter_or_none("a")({"a": 1})
    1

    >>> itemgetter_or_none("b")({"a": 1})
    None
    """

    def itemgetter_or_none(obj):
        return obj.get(attr, None)

    return itemgetter_or_none


def itemgetter_with_default(default, attr):
    """Retrieves a dictionary entry by its key `attr`. Returns `default` if the key is not there.

    >>> itemgetter_with_default(0, "a")({"a": 1})
    1
    >>> itemgetter_with_default(0, "b")({"a": 1})
    0
    """

    def itemgetter_with_default(d):
        try:
            return d[attr]
        except (KeyError, IndexError):
            return default

    return itemgetter_with_default


@currying.curry
def get_or_transform(f, d):
    def get_or_transform(x):
        if x in d:
            return d[x]
        return f(x)

    return get_or_transform


get_or_identity = get_or_transform(operator.identity)


def get_in(keys: Iterable):

    """Creates a function that returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.

    >>> get_in(["a", "b", 1])({"a": {"b": [0, 1, 2]}})
    1
    """

    def get_in(coll):
        return functools.reduce(getitem, keys, coll)

    return get_in


def get_in_with_default(keys: Iterable, default):
    """`get_in` function, returning `default` if a key is not there.

    >>> get_in_with_default(["a", "b", 1], 0)({"a": {"b": [0, 1, 2]}})
    1
    >>> get_in_with_default(["a", "c", 1], 0)({"a": {"b": [0, 1, 2]}})
    0
    """
    getter = get_in(keys)

    def get_in_with_default(x):
        try:
            return getter(x)
        except (KeyError, IndexError, TypeError):
            return default

    return get_in_with_default


def get_in_or_none(keys: Iterable):
    """`get_in` function, returning `None` if a key is not there.

    >>> get_in_or_none(["a", "b", 1])({"a": {"b": [0, 1, 2]}})
    1
    >>> get_in_or_none(["a", "c", 1])({"a": {"b": [0, 1, 2]}})
    None
    """
    return get_in_with_default(keys, None)


def get_in_or_none_uncurried(keys: Iterable, coll):
    """Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys and `None` if a key is not there.

    >>> get_in_or_none_uncurried(["a", "b", 1],{"a": {"b": [0, 1, 2]}})
    1
    >>> get_in_or_none_uncurried(["a", "c", 1], {"a": {"b": [0, 1, 2]}})
    None
    """
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


def _return_after_n_calls(n, value):
    if n == 0:
        return value

    def return_after_n_calls(_):
        return _return_after_n_calls(n - 1, value)

    return return_after_n_calls


def make_index(
    steps: Iterable[Callable[[Iterable], Dict]],
) -> Callable[[Iterable], Any]:
    """Builds an index with arbitrary amount of steps from an iterable.

    >>> index = dict_utils.make_index(map(gamla.groupby, [gamla.head, gamla.second]))(["uri", "dani"])
    >>> index("d")("a")
    frozenset(["dani"])
    """
    steps = tuple(steps)
    if not steps:
        return frozenset
    return functional_generic.compose_left(
        operator.head(steps),
        functional_generic.valmap(make_index(steps[1:])),
        lambda d: lambda x: d.get(
            x,
            _return_after_n_calls(len(steps) - 1, frozenset()),
        ),
    )


def add_key_value(key, value):
    """Associate a key-value pair to the input dict.

    >>> add_key_value("1", "1")({"2": "2"})
    {'2': '2', '1': '1'}
    """

    def add_key_value(d):
        return functional.assoc_in(d, [key], value)

    return add_key_value


def remove_key(key):
    """Given a dictionary, return a new dictionary with 'key' removed.
    >>> remove_key("two")({"one": 1, "two": 2, "three": 3})
    {'one': 1, 'three': 3}
    """

    def remove_key(d: dict):
        updated = d.copy()
        del updated[key]
        return updated

    return remove_key


def rename_key(old: str, new: str) -> Callable[[dict], dict]:
    """Rename a key in a dictionary.

    >>> my_dict = {"name": "Danny", "age": 20}
    >>> rename_key("name", "first_name")(my_dict)
    {'first_name': 'Danny', 'age': 20}
    """
    return sync.keymap(sync.when(operator.equals(old), construct.just(new)))


def transform_item(key, f: Callable) -> Callable[[dict], dict]:
    """transform a value of `key` in a dict. i.e given a dict `d`, return a new dictionary `e` s.t e[key] = f(d[key]).

    >>> my_dict = {"name": "Danny", "age": 20}
    >>> transform_item("name", str.upper)(my_dict)
    {'name': 'DANNY', 'age': 20}
    """
    return functional_generic.itemmap(
        functional_generic.when(
            functional_generic.compose_left(operator.head, operator.equals(key)),
            functional_generic.packstack(operator.identity, f),
        ),
    )
