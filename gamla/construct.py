from typing import Any, Dict, Hashable


def wrap_dict(key: Hashable):
    """Wrap a key and a value in a dict (in a curried fashion).

    >>> wrap_dict("one") (1)
    {'one': 1}
    """

    def wrap_dict(value):
        return {key: value}

    return wrap_dict


def wrap_tuple(x: Any):
    """Wrap an element in a tuple.

    >>> wrap_tuple("hello")
    ('hello',)
    """
    return (x,)


def wrap_frozenset(x: Any):
    """Wraps x with frozenset.

    >>> wrap_frozenset(1)
    frozenset({1})
    """
    return frozenset([x])


def wrap_str(wrapping_string: str):
    """Wrap a string in a wrapping string.

    >>> wrap_str("hello {}", "world")
    'hello world'
    """

    def wrap_str(x) -> str:
        return wrapping_string.format(x)

    return wrap_str


def wrap_multiple_str(wrapping_string: str):
    """Wrap multiple values in a wrapping string by passing a dict where the keys are the parameters in the wrapping string and the values are the desired values.

    >>> wrap_multiple_str("hello {first} {second}")({ "first": "happy", "second": "world" })
    'hello happy world'
    """

    def inner(x: Dict[str, str]) -> str:
        return wrapping_string.format(**x)

    return inner


def just(x):
    """Ignores the input upon execution and returns the given argument.
    >>> f = just(1)
    >>> f(2)
    1
    """

    def just(*args, **kwargs):
        del args, kwargs
        return x

    return just
