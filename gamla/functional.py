import functools
import hashlib
import json
import logging
from typing import Callable, Iterable, Text

import toolz
from toolz import curried


def wrapped_partial(func: Callable, *args, **kwargs) -> Callable:
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


@toolz.curry
def anymap(f: Callable, it: Iterable):
    return any(map(f, it))


@toolz.curry
def allmap(f: Callable, it: Iterable):
    return all(map(f, it))


def ignore_input(inner):
    def ignore_and_run(*args, **kwargs):
        return inner()

    return ignore_and_run


def curried_ternary(condition, f_true, f_false):
    def inner(*args, **kwargs):
        return (
            f_true(*args, **kwargs)
            if condition(*args, **kwargs)
            else f_false(*args, **kwargs)
        )

    return inner


def make_raise(exception):
    def inner():
        raise exception

    return ignore_input(inner)


def translate_exception(func, exc1, exc2):
    """`func` is assumed to be unary."""
    return toolz.excepts(exc1, func, make_raise(exc2))


def compute_stable_json_hash(item) -> Text:
    return hashlib.sha1(
        json.dumps(
            json.loads(item.to_json()), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def star(function: Callable) -> Callable:
    return lambda x: function(*x)


@toolz.curry
def _assert_f_output_on_inp(f, inp):
    assert f(inp)


def assert_that(f):
    return curried.do(_assert_f_output_on_inp(f))


def first(*funcs, exception_type):
    def inner(*args, **kwargs):
        for func in funcs:
            try:
                return func(*args, **kwargs)
            except exception_type:
                pass
        raise exception_type

    return inner


logger = curried.do(logging.info)


def log_text(text: Text):
    return curried.do(lambda _: logging.info(text))
