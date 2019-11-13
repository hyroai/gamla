import functools
import logging
import time

import requests
import requests.adapters
import requests.packages.urllib3.util.retry
import toolz
from gevent import pool


def timeit(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(
            f"Elapsed time {f.__name__}: {elapsed} ({str(args)[:100]}, {str(kwargs)[:100]})"
        )
        return result

    return wrapper


def requests_with_retry(retries: int = 3) -> requests.Session:
    session = requests.Session()
    retry = requests.packages.urllib3.util.retry.Retry(
        total=retries, backoff_factor=0.1, status_forcelist=(500, 502, 504)
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# TODO(uri): Remove in favor of `pmap`.
def curried_pmap(f):
    def inner(it):
        return pool.Group().map(f, it)

    return inner


@toolz.curry
def pmap(f, it):
    return pool.Group().map(f, it)
