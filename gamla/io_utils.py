import functools
import logging
import threading
import time
from concurrent import futures

import requests
import requests.adapters
import toolz
from requests.packages.urllib3.util import retry


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
    adapter = requests.adapters.HTTPAdapter(
        max_retries=retry.Retry(
            total=retries, backoff_factor=0.1, status_forcelist=(500, 502, 504)
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def spawn(f):
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(f)


def spawn_later(s, f):
    threading.Timer(s, f).start()


@toolz.curry
def pmap(f, it):
    return tuple(futures.ThreadPoolExecutor().map(f, it))


def make_promise():
    lock = threading.Semaphore()
    lock.acquire()
    resolved_value = None

    def resolve(value):
        nonlocal resolved_value
        resolved_value = value
        lock.release()

    def wait():
        with lock:
            if isinstance(resolved_value, Exception):
                raise resolved_value
            return resolved_value

    return resolve, wait
