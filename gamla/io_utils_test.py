import asyncio
import random

import pytest

import gamla
from gamla import functional_generic, io_utils

pytestmark = pytest.mark.asyncio


async def test_batch_decorator():
    times_f_called = 0

    @io_utils.batch_calls(100)
    async def slow_identity(inputs):
        nonlocal times_f_called
        times_f_called += 1
        await asyncio.sleep(random.random() * 0.1 * len(inputs))
        return inputs

    inputs = tuple(range(100))
    results = await gamla.pipe(inputs, gamla.map(slow_identity), tuple)
    assert results == inputs
    assert times_f_called < 5


async def test_batch_decorator_errors():
    times_f_called = 0

    @io_utils.batch_calls(100)
    async def slow_identity_with_errors(inputs):
        nonlocal times_f_called
        times_f_called += 1
        await asyncio.sleep(random.random() * 0.1 * len(inputs))
        if times_f_called > 1:
            raise ValueError
        return inputs

    assert (await gamla.pipe((1,), gamla.map(slow_identity_with_errors), tuple)) == (1,)

    with pytest.raises(ValueError):
        await gamla.pipe((1, 2, 3), gamla.map(slow_identity_with_errors), tuple)

    assert times_f_called == 2


async def test_batch_decorator_max_size():
    times_f_called = 0

    @io_utils.batch_calls(10)
    async def slow_identity(inputs):
        nonlocal times_f_called
        times_f_called += 1
        await asyncio.sleep(random.random() * 0.1 * len(inputs))
        return inputs

    inputs = tuple(range(100))
    results = await gamla.pipe(inputs, gamla.map(slow_identity), tuple)
    assert results == inputs
    assert times_f_called == 10


async def test_with_and_without_deduper():
    inputs = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3)
    called = []

    async def identity_with_spying_and_delay(x):
        await asyncio.sleep(0.4)
        called.append(x)
        return x

    # Without.
    assert inputs == tuple(await gamla.map(identity_with_spying_and_delay)(inputs))

    assert len(called) == len(inputs)

    called[:] = []

    # With.
    assert inputs == tuple(
        await gamla.map(io_utils.queue_identical_calls(identity_with_spying_and_delay))(
            inputs,
        ),
    )

    unique = frozenset(inputs)
    assert len(called) == len(unique)
    assert frozenset(called) == unique


async def test_retry():
    class SomeException(Exception):
        pass

    succeeds_in_n_retries = 3

    async def f(x, y):
        nonlocal succeeds_in_n_retries
        if succeeds_in_n_retries == 0:
            return x + y
        succeeds_in_n_retries -= 1
        raise SomeException

    assert await io_utils.retry(SomeException, 3, 0, f)(3, 2) == 5


async def test_retry_raises():
    class SomeException(Exception):
        pass

    succeeds_in_n_retries = 3

    async def f(x, y):
        nonlocal succeeds_in_n_retries
        if succeeds_in_n_retries == 0:
            return x + y
        succeeds_in_n_retries -= 1
        raise SomeException

    with pytest.raises(SomeException):
        await io_utils.retry(SomeException, 2, 0, f)(3, 2)


async def test_throtle():
    factor = 0

    @io_utils.throttle(1)
    async def multiply_with_delay(x):
        nonlocal factor
        factor = factor + 1
        await asyncio.sleep(0.01)
        return x * factor

    assert (
        await gamla.pipe(
            (1, 2, 3),
            functional_generic.curried_map(multiply_with_delay),
            tuple,
        )
        == (1, 4, 9)
    )
