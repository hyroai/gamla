import asyncio
import functools
import logging
import time

import aiohttp
import requests
import requests.adapters
from requests.packages.urllib3.util import retry

_HTTP_SESSION = aiohttp.ClientSession()


def apost(url, json, timeout, headers=None):
    return _HTTP_SESSION.post(url=url, json=json, timeout=timeout, headers=headers)


def aget(url, timeout, params=None, headers=None, json=None):
    return _HTTP_SESSION.get(
        url=url, timeout=timeout, headers=headers, params=params, json=json
    )


def _log_args(name, elapsed, args, kwargs):
    args_str = str(args)[:50]
    kwargs_str = str(kwargs)[:50]
    logging.info(f"{name}: {elapsed} (args: {args_str}, kwargs: {kwargs_str})")


def _async_timeit(f):
    @functools.wraps(f)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await f(*args, **kwargs)
        _log_args(
            name=f.__name__, elapsed=time.time() - start, args=args, kwargs=kwargs
        )
        return result

    return wrapper


def timeit(f):
    if asyncio.iscoroutinefunction(f):
        return _async_timeit(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        _log_args(
            name=f.__name__, elapsed=time.time() - start, args=args, kwargs=kwargs
        )
        return result

    return wrapper


def requests_with_retry(retries: int = 3) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=retry.Retry(
            total=retries, backoff_factor=0.1, status_forcelist=(500, 502, 504)
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# TODO(uri): move the test as well
def batch_calls(f):
    """Batches single call into one request.

    Turns `f`, a function that gets a `tuple` of independent requests, into a function
    that gets a single request.
    """
    queue = {}

    async def make_call():
        if not queue:
            return
        await asyncio.sleep(0.1)
        if not queue:
            return
        queue_copy = dict(queue)
        queue.clear()
        try:
            for async_result, result in zip(
                queue_copy.values(), await f(tuple(queue_copy))
            ):
                async_result.set_result(result)
        except Exception as exception:
            for async_result in queue_copy.values():
                async_result.set_exception(exception)

    async def wrapped(hashable_input):
        if hashable_input in queue:
            return await queue[hashable_input]
        async_result = asyncio.Future()
        # Check again because of context switch due to the creation of `asyncio.Future`.
        # TODO(uri): Make sure this is needed.
        if hashable_input in queue:
            return await queue[hashable_input]
        queue[hashable_input] = async_result
        if len(queue) == 1:
            asyncio.create_task(make_call())
        return await async_result

    return wrapped
