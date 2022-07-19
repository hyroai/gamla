import asyncio
import functools
import logging
import time
from typing import Callable, Dict, Text, Tuple, Union

import async_timeout
import httpx
import requests
import requests.adapters
import termcolor
from requests.packages.urllib3.util import retry as retry_lib

from gamla import currying, functional
from gamla.optimized import sync


def _color_for_duration(duration: float):
    if duration < 0.1:
        return "green"
    if duration < 1:
        return "yellow"
    return "red"


def _log_finish(req_id: Text, start: float):
    finish = time.time()
    elapsed = finish - start
    logging.info(
        f"{termcolor.colored(req_id, 'white', 'on_grey')} {termcolor.colored(str(round(elapsed, 2)), _color_for_duration(elapsed))} seconds",
    )


def _async_timeit(f):
    @functools.wraps(f)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await f(*args, **kwargs)
        _log_finish(f.__name__, start)
        return result

    return wrapper


def timeit(f):
    """Wraps a function `f` with a timer.
    Logs the start time, and end time (and difference in seconds).

    >>> timed_get_async = timeit(get_async)
    """
    if asyncio.iscoroutinefunction(f):
        return _async_timeit(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        _log_finish(f.__name__, start)
        return result

    return wrapper


def requests_with_retry(retries: int = 3) -> requests.Session:
    """Creates a `requests` object. Request will be attempted `retries` times.
    Will retry on 500, 502 and 504 status codes.

    >>> response = requests_with_retry(3).get("http://someurl.com")
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=retry_lib.Retry(
            total=retries,
            backoff_factor=0.1,
            status_forcelist=(500, 502, 504),
        ),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@currying.curry
def batch_calls(max_batch_size: int, f: Callable):
    """Batches single call into one request.
    Turns `f`, a function that gets a `tuple` of independent requests, into a function
    that gets a single request.
    Each request will be at most of size `max_batched_size`.
    Requests will be performed in time intervals of 0.1s.

    >>> batched_f = batch_calls(5, f)
    """
    queue: Dict = {}

    async def make_call():
        await asyncio.sleep(0.1)
        if not queue:
            return
        slice_from_queue = tuple(functional.take(max_batch_size)(queue))
        promises = tuple(map(queue.__getitem__, slice_from_queue))
        requests = tuple(slice_from_queue)
        for x in slice_from_queue:
            del queue[x]
        try:
            for promise, result in zip(promises, await f(requests)):
                # We check for possible mid-exception or timeouts.
                if promise.done() or promise.cancelled():
                    continue
                promise.set_result(result)
        except Exception as exception:
            for promise in promises:
                # We check for possible mid-exception or timeouts.
                if promise.done() or promise.cancelled():
                    continue
                promise.set_exception(exception)

    @functools.wraps(f)
    async def wrapped(hashable_input):
        if hashable_input in queue:
            return await queue[hashable_input]
        async_result = asyncio.Future()
        # Check again because of context switch
        # due to the creation of `asyncio.Future`.
        # TODO(uri): Make sure this is needed.
        if hashable_input in queue:
            return await queue[hashable_input]
        queue[hashable_input] = async_result
        asyncio.create_task(make_call())
        return await async_result

    return wrapped


def queue_identical_calls(f):
    """Queues identical calls to coroutine `f` and performs each call once.
    Note that pending grows infinitely large, this assumes we cache the results
    anyway after this decorator, so we at most double the memory consumption.

    >>> deduped_get_async = queue_identical_calls(get_async)
    """

    pending = {}

    @functools.wraps(f)
    async def wrapped(*args, **kwargs):
        key = functional.make_call_key(args, kwargs)
        if key not in pending:
            pending[key] = asyncio.Future()
            pending[key].set_result(await f(*args, **kwargs))
        return await asyncio.wait_for(pending[key], timeout=20)

    return wrapped


def make_throttler(limit):
    """Returns a function that can be used on any number of coroutines to make sure only `limit` amount of calls are done in parallel.

    >>> throttler = make_throttler(3)
    >>> throttler(get_async)
    >>> throttler(post_json_async)
    """
    semaphore = None

    def wrap_function(f):
        @functools.wraps(f)
        async def wrap(*args, **kwargs):
            nonlocal semaphore
            # This must be in the inner function so that we avoid creating an event loop before the user has, causing the code to run with two different event loops.
            if not semaphore:
                semaphore = asyncio.Semaphore(limit)
            async with semaphore:
                return await f(*args, **kwargs)

        return wrap

    return wrap_function


@currying.curry
def throttle(limit, f):
    """Wraps a coroutine f assuring only `limit` amount of calls are done in parallel.

    >>> throttled_get_async = throttle(3, get_async)
    """
    return make_throttler(limit)(f)


def timeout(seconds: float):
    """Wraps a coroutine with a timeout (seconds) after
    which it will raise `asyncio.TimeoutError`.

    >>> get_with_timeout = timeout(3.5)(get_async)
    """

    def wrapper(corofunc):
        @functools.wraps(corofunc)
        async def run(*args, **kwargs):
            async with async_timeout.timeout(seconds):
                return await corofunc(*args, **kwargs)

        return run

    return wrapper


@currying.curry
async def get_async_with_headers(headers: Dict[str, str], timeout: float, url: Text):
    """Performs an async GET request to url with the specified timeout (seconds) and headers.

    >>> response = await get_async_with_headers({some_header: some_value}, 30, "https://www.someurl.com")
    """
    async with httpx.AsyncClient() as client:
        return await client.get(url, timeout=timeout, headers=headers)


#: Performs an async GET request to url with the specified timeout (seconds) and headers.
#: Expects the payload to be a json serializable object.
#:
#: >>> response = await get_async(30, "https://www.someurl.com")
get_async = get_async_with_headers({})


@currying.curry
async def head_async_with_headers(headers: Dict[str, str], timeout: float, url: Text):
    """Performs an async HEAD request to url with the specified timeout (seconds) and headers.

    >>> response = await head_async_with_headers(30, "https://www.someurl.com", {})
    """
    async with httpx.AsyncClient() as client:
        return await client.head(url, timeout=timeout, headers=headers)


@currying.curry
async def post_json_with_extra_headers_and_params_async(
    params: Dict[Text, Text],
    extra_headers: Dict[Text, Text],
    timeout: float,
    url: Text,
    payload,
):
    """Performs an http POST request with json data and URL parameters. Additional headers may be specified.
    Expects the params to be a dictionary object and the payload to be a json serializable object.

    >>> response = await post_json_with_extra_headers_and_params_async({"Date": "07/08/2022"}, {"Authorization": "Bearer TOKEN" }, 30, "https://www.someurl.com/post_data", { "name": "Danny" })
    """
    async with httpx.AsyncClient() as client:
        return await client.post(
            url=url,
            params=params,
            json=payload,
            headers=sync.merge(
                [
                    {"content_type": "application/json"},
                    extra_headers,
                ],
            ),
            timeout=timeout,
        )


#: Performs an http POST request of json data with specified headers.
#: Expects the payload to be a json serializable object and the headers to be a dictionary.
#:
#: >>> response = post_json_with_extra_headers_async({"username": "First Last", 30, "https://www.someurl.com/post_data", { "name": "Danny" })
post_json_with_extra_headers_async = post_json_with_extra_headers_and_params_async({})

#: Performs an http POST request of json data.
#: Expects the payload to be a json serializable object.
#:
#: >>> response = post_json_async(30, "https://www.someurl.com/post_data", { "name": "Danny" })
post_json_async = post_json_with_extra_headers_and_params_async({}, {})


@currying.curry
def _retry_with_count(
    exception: Union[Exception, Tuple[Exception, ...]],
    max_times: int,
    times: int,
    wait_seconds: float,
    f: Callable,
):
    async def retry_inner(*args, **kwargs):
        try:
            return await f(*args, **kwargs), max_times - times
        except exception:
            await asyncio.sleep(wait_seconds)
            if not times:
                raise exception
            return await _retry_with_count(
                exception,
                max_times,
                times - 1,
                wait_seconds,
                f,
            )(*args, **kwargs)

    return retry_inner


@currying.curry
def retry_with_count(
    exception: Union[Exception, Tuple[Exception, ...]],
    times: int,
    wait_seconds: float,
    f: Callable,
):
    """Wraps a coroutine to retry on given exceptions and returns tuple of output and count of retries."""
    return _retry_with_count(exception, times, times, wait_seconds, f)


@currying.curry
def retry(
    exception: Union[Exception, Tuple[Exception, ...]],
    times: int,
    wait_seconds: float,
    f: Callable,
):
    """Wraps a coroutine to retry on given exceptions."""

    async def retry_inner(*args, **kwargs):
        output, _ = await retry_with_count(exception, times, wait_seconds, f)(
            *args, **kwargs
        )
        return output

    return retry_inner
