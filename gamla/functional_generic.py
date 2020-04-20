import asyncio

import toolz

from gamla import functional, functional_async


def compose_left(*funcs):
    return compose(*reversed(funcs))


def compose(*funcs):
    if functional.anymap(asyncio.iscoroutinefunction, funcs):
        return functional_async.acompose(*funcs)
    else:
        return toolz.compose(*funcs)
