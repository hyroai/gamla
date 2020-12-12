import asyncio
import builtins
import logging
from typing import Text

from toolz import curried

logger = curried.do(logging.info)


def log_text(text: Text, level: int = logging.INFO):
    return curried.do(lambda x: logging.log(level, text.format(x)))


do_breakpoint = curried.do(lambda x: builtins.breakpoint())


def debug_exception(f):
    if asyncio.iscoroutinefunction(f):

        async def debug_exception(x):
            try:
                return await f(x)
            except Exception as e:
                builtins.breakpoint()
                raise e

    else:

        def debug_exception(value):  # type: ignore
            try:
                return f(value)
            except Exception as e:
                builtins.breakpoint()
                raise e

    return debug_exception
