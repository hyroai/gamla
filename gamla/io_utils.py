import asyncio
import datetime
import functools
import logging
import time
from typing import Callable, Dict, Text

import async_timeout
import httpx
import requests
import requests.adapters
import toolz
from requests.packages.urllib3.util import retry

from gamla import currying, functional


def _time_to_readable(time_s: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(time_s)


def _request_id(f: Callable, args, kwargs) -> Text:
    name = f.__name__
    params_str = f", args: {args}, kwargs: {kwargs}"
    together = name + params_str
    if len(together) > 100:
        return name
    return together


def _log_finish(req_id: Text, start: float):
    finish = time.time()
    elapsed = finish - start
    logging.info(
        f"{req_id} finished at {_time_to_readable(finish)}, took {elapsed:.2f}",
    )


def _log_start(req_id: Text) -> float:
    start = time.time()
    logging.info(f"{req_id} started at {_time_to_readable(start)}")
    return start


def _async_timeit(f):
    @functools.wraps(f)
    async def wrapper(*args, **kwargs):
        req_id = _request_id(f, args, kwargs)
        start = _log_start(req_id)
        result = await f(*args, **kwargs)
        _log_finish(req_id, start)
        return result

    return wrapper


def timeit(f):
    if asyncio.iscoroutinefunction(f):
        return _async_timeit(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        req_id = _request_id(f, args, kwargs)
        start = _log_start(req_id)
        result = f(*args, **kwargs)
        _log_finish(req_id, start)
        return result

    return wrapper


def requests_with_retry(retries: int = 3) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=retry.Retry(
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
    """
    queue: Dict = {}

    async def make_call():
        await asyncio.sleep(0.1)
        if not queue:
            return
        slice_from_queue = tuple(toolz.take(max_batch_size, queue))
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
    # Note that pending grows infinitely large, this assumes we cache the results
    # anyway after this decorator, so we at most double the memory consumption.
    pending = {}

    @functools.wraps(f)
    async def wrapped(*args, **kwargs):
        key = functional.make_call_key(args, kwargs)
        if key not in pending:
            pending[key] = asyncio.Future()
            pending[key].set_result(await f(*args, **kwargs))
        return await asyncio.wait_for(pending[key], timeout=20)

    return wrapped


@currying.curry
def throttle(limit, f):
    semaphore = asyncio.Semaphore(limit)

    @functools.wraps(f)
    async def wrapped(*args, **kwargs):
        async with semaphore:
            return await f(*args, **kwargs)

    return wrapped


def timeout(seconds: float):
    def wrapper(corofunc):
        @functools.wraps(corofunc)
        async def run(*args, **kwargs):
            with async_timeout.timeout(seconds):
                return await corofunc(*args, **kwargs)

        return run

    return wrapper


@currying.curry
async def get_async(timeout: float, url: Text):
    async with httpx.AsyncClient() as client:
        return await client.get(url, timeout=timeout)


@currying.curry
async def post_json_with_extra_headers_async(
    extra_headers: Dict[Text, Text], timeout: float, url: Text, payload
):
    """Expects payload to be a json object, and the response to be json as well."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            url=url,
            json=payload,
            headers=toolz.merge({"content_type": "application/json"}, extra_headers),
            timeout=timeout,
        )


post_json_async = post_json_with_extra_headers_async({})
