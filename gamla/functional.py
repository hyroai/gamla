import functools
import hashlib
import itertools
import json
import logging
from typing import Callable, Dict, Iterable, Text, Type

import toolz
from gevent import pool
from toolz import curried


@toolz.curry
def dict_get(dict_obj: Dict, default, key):
    return dict_obj.get(key, default)


def _do_if(condition, fun):
    return curried.do(curried_ternary(condition, fun, toolz.identity))


def check(condition, exception):
    return _do_if(toolz.complement(condition), make_raise(exception))


def bifurcate(*funcs):
    """Serially runs each function on tee'd copies of `input_generator`."""

    def inner(input_generator):
        return toolz.pipe(
            zip(funcs, itertools.tee(input_generator, len(funcs))),
            curried.map(star(lambda f, generator: f(generator))),
            tuple,
        )

    return inner


def singleize(func: Callable) -> Callable:
    def wrapped(some_input):
        if isinstance(some_input, tuple):
            return func(some_input)
        return toolz.first(func((some_input,)))

    return wrapped


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


@functools.lru_cache(maxsize=None)
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


@toolz.curry
def pmap(f, it):
    return pool.Group().map(f, it)


def pfirst(*funcs, exception_type):
    value_signifying_failure = "gamla-first-failed"

    def inner(*args, **kwargs):
        return toolz.pipe(
            funcs,
            curried.pmap(
                toolz.excepts(
                    exception_type,
                    lambda func: func(*args, **kwargs),
                    lambda _: value_signifying_failure,
                )
            ),
            curried.filter(lambda x: x != value_signifying_failure),
            next,
        )

    return inner


def first(*funcs, exception_type: Type[Exception], run_parallel: bool = False):
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
