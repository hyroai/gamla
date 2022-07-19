import asyncio
import random

import pytest

from gamla import functional_generic, io_utils


async def test_batch_decorator():
    times_f_called = 0

    @io_utils.batch_calls(100)
    async def slow_identity(inputs):
        nonlocal times_f_called
        times_f_called += 1
        await asyncio.sleep(random.random() * 0.1 * len(inputs))
        return inputs

    inputs = tuple(range(100))
    results = await functional_generic.pipe(
        inputs,
        functional_generic.curried_map(slow_identity),
        tuple,
    )
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

    assert (
        await functional_generic.pipe(
            (1,),
            functional_generic.curried_map(slow_identity_with_errors),
            tuple,
        )
    ) == (1,)

    with pytest.raises(ValueError):
        await functional_generic.pipe(
            (1, 2, 3),
            functional_generic.curried_map(slow_identity_with_errors),
            tuple,
        )

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
    results = await functional_generic.pipe(
        inputs,
        functional_generic.curried_map(slow_identity),
        tuple,
    )
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
    assert inputs == tuple(
        await functional_generic.curried_map(identity_with_spying_and_delay)(inputs),
    )

    assert len(called) == len(inputs)

    called[:] = []

    # With.
    assert inputs == tuple(
        await functional_generic.curried_map(
            io_utils.queue_identical_calls(identity_with_spying_and_delay),
        )(
            inputs,
        ),
    )

    unique = frozenset(inputs)
    assert len(called) == len(unique)
    assert frozenset(called) == unique


async def test_retry():
    class SomeError(Exception):
        pass

    succeeds_in_n_retries = 3

    async def f(x, y):
        nonlocal succeeds_in_n_retries
        if succeeds_in_n_retries == 0:
            return x + y
        succeeds_in_n_retries -= 1
        raise SomeError

    assert await io_utils.retry(SomeError, 3, 0, f)(3, 2) == 5


async def test_retry_with_count():
    class SomeError(Exception):
        pass

    succeeds_in_n_retries = 2

    async def f(x, y):
        nonlocal succeeds_in_n_retries
        if succeeds_in_n_retries == 0:
            return x + y
        succeeds_in_n_retries -= 1
        raise SomeError

    assert await io_utils.retry_with_count(SomeError, 3, 0, f)(3, 2) == (5, 2)


async def test_retry_raises():
    class SomeError(Exception):
        pass

    succeeds_in_n_retries = 3

    async def f(x, y):
        nonlocal succeeds_in_n_retries
        if succeeds_in_n_retries == 0:
            return x + y
        succeeds_in_n_retries -= 1
        raise SomeError

    with pytest.raises(SomeError):
        await io_utils.retry(SomeError, 2, 0, f)(3, 2)


async def test_throtle():
    factor = 0

    @io_utils.throttle(1)
    async def multiply_with_delay(x):
        nonlocal factor
        factor = factor + 1
        await asyncio.sleep(0.01)
        return x * factor

    assert await functional_generic.pipe(
        (1, 2, 3),
        functional_generic.curried_map(multiply_with_delay),
        tuple,
    ) == (1, 4, 9)


async def test_make_throttler():
    total = 0
    throttle = io_utils.make_throttler(1)

    @throttle
    async def inc():
        nonlocal total
        total = total + 1
        await asyncio.sleep(0.01)
        assert total <= 1

    @throttle
    async def dec():
        await asyncio.sleep(0.01)
        nonlocal total
        total = total - 1
        assert total >= 0

    await asyncio.gather(inc(), dec(), inc(), dec())
    assert total == 0
