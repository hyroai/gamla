import asyncio

import toolz
from toolz import curried

from gamla import functional, functional_async


def compose_left(*funcs):
    return compose(*reversed(funcs))


_any_async = functional.anymap(asyncio.iscoroutinefunction)


def compose(*funcs):
    if _any_async(funcs):
        return functional_async.acompose(*funcs)
    return toolz.compose(*funcs)


@toolz.curry
def after(f1, f2):
    return compose(f1, f2)


@toolz.curry
def before(f1, f2):
    return compose_left(f1, f2)


def lazyjuxt(*funcs):
    if _any_async(funcs):
        return compose_left(
            functional_async.apply_async,
            functional_async.amap,
            functional_async.apply_async(funcs),
        )
    return compose_left(functional.apply, curried.map, functional.apply(funcs))


alljuxt = compose(after(all), lazyjuxt)


anyjuxt = compose(after(any), lazyjuxt)


juxtcat = compose(after(toolz.concat), lazyjuxt)
