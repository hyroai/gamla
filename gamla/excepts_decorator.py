import asyncio
import functools
from typing import Any, Callable, Tuple, Union

from gamla import currying


@currying.curry
def excepts(
    exception: Union[Tuple[Exception, ...], Exception],
    handler: Callable[[Exception], Any],
    function: Callable,
):
    if asyncio.iscoroutinefunction(function):

        async def excepts(*args, **kwargs):
            try:
                return await function(*args, **kwargs)
            except exception as error:
                return handler(error)

    else:

        def excepts(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except exception as error:
                return handler(error)

    return functools.wraps(function)(excepts)
