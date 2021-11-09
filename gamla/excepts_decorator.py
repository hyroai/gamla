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


def try_and_excepts(
    exception: Union[Tuple[Exception, ...], Exception],
    handler: Callable,
    function: Callable,
):
    """Same as sync excepts only that the handler gets the original function params after the exception param."""
    if asyncio.iscoroutinefunction(function):

        async def try_and_excepts(*args, **kwargs):
            try:
                return await function(*args, **kwargs)
            except exception as error:
                return await handler(error, *args, **kwargs)

    else:

        def try_and_excepts(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except exception as error:
                return handler(error, *args, **kwargs)

    return functools.wraps(function)(try_and_excepts)
