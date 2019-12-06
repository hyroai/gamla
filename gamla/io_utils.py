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
    if asyncio.iscoroutine(f):
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
