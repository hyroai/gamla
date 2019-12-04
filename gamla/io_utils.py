import functools
import logging
import time

import aiohttp
import requests
import requests.adapters
from requests.packages.urllib3.util import retry


async def apost(url, json, timeout):
    async with aiohttp.ClientSession() as session:
        async with session.post(url=url, json=json, timeout=timeout) as response:
            return await response.json()


def timeit(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        elapsed = time.time() - start
        args_str = str(args)[:100]
        kwargs_str = str(kwargs)[:100]
        logging.info(f"Elapsed time {f.__name__}: {elapsed} ({args_str}, {kwargs_str})")
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
