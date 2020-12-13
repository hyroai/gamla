import asyncio
import builtins
import logging
from typing import Text

from gamla import functional_generic

logger = functional_generic.side_effect(logging.info)


def log_text(text: Text, level: int = logging.INFO):
    return functional_generic.side_effect(lambda x: logging.log(level, text.format(x)))


do_breakpoint = functional_generic.side_effect(lambda x: builtins.breakpoint())


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

        async def debug_exception(x):
            try:
                return await f(x)
            except Exception as e:
                builtins.breakpoint()
                raise e

    else:

        def debug_exception(x):  # type: ignore
            try:
                return f(x)
            except Exception as e:
                builtins.breakpoint()
                raise e

    return debug_exception
