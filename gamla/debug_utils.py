import asyncio
import builtins
import functools
import logging
from typing import Text

import toolz

from gamla import functional, functional_generic

logger = functional_generic.side_effect(logging.info)


def log_text(text: Text, level: int = logging.INFO):
    return functional_generic.side_effect(lambda x: logging.log(level, text.format(x)))


def _is_generator(iterable):
    return hasattr(iterable, "__iter__") and not hasattr(iterable, "__len__")


#: A util to inspect a pipline by opening a debug prompt.
#: Note:
#: - Materializes generators, as most of the time we are interested in looking into them, so can have unexpected results.
#: - The current value can be referenced by `x` in the debug prompt.
debug = functional_generic.compose_left(
    functional_generic.when(_is_generator, tuple),
    functional_generic.side_effect(lambda x: builtins.breakpoint()),
)

debug_after = functional_generic.after(debug)
debug_before = functional_generic.before(debug)


def _debug_generic(f):
    return functional_generic.compose_left(
        lambda *funcs: toolz.interleave([funcs, [debug] * len(funcs)]),
        functional.star(f),
    )


#: Replace regular `compose` calls with this to get a breakpoint at each step.
debug_compose = _debug_generic(functional_generic.compose)
#: Replace regular `compose_left` calls with this to get a breakpoint at each step.
debug_compose_left = _debug_generic(functional_generic.compose_left)


def debug_exception(f):
    """Debug exception in a pipeline stage by looking at the causal value.

    >>> gamla.pipe(
        "abc",
        gamla.itemgetter("some_key"),  # This would cause an exception.
    )

    >>> gamla.pipe(
        "abc",
        gamla.debug_exception(gamla.itemgetter("some_key")),  # Now we can see the cause of the exception - we expect a `dict` but get a `str`.
    )
    """
    if asyncio.iscoroutinefunction(f):

        async def debug_exception(*x, **kwargs):
            try:
                return await f(*x, **kwargs)
            except Exception as e:
                builtins.breakpoint()
                raise e

    else:

        def debug_exception(*x, **kwargs):  # type: ignore
            try:
                return f(*x, **kwargs)
            except Exception as e:
                builtins.breakpoint()
                raise e

    return functools.wraps(f)(debug_exception)
