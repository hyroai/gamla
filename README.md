[![Build Status](https://travis-ci.com/hyroai/gamla.svg?branch=master)](https://travis-ci.com/hyroai/gamla)

גamla is a functional programming library for python.

`pip install gamla`

## Debugging anonymous compositions

`gamla.compose(x, y, z)` produces a new function which doesn't have a proper name. If `x` raises an exception, it is sometimes hard to figure out where this occurred. To overcome this, set the env variable `GAMLA_DEBUG_MODE` (to anything). This will cause the composition functions to give a name for the anonymous composition. The name would be a concatenation of its constituents' names, e.g. `x_of_y_of_z`. The name is given in a way that persists to the stack exceptions.

This is turned on only by flag because it incurs significant overhead so things might get slow.

## Mixing asynchronous and synchronous code

Most functions in this lib will work seamlessly with `async` and regular functions, and allow the developer to focus on the logic instead of deciding where to place an `await`.

For example:

```python
import asyncio

import gamla


def increment(i):
    return i + 1


async def increment_async(i):
    asyncio.sleep(1)
    return i + 1


async def run():
    mixed_composition = gamla.compose_left(increment, increment_async, increment)
    return await mixed_composition(0)  # returns 3!
```

## Releasing a new  version

1. Create a pypi account.
1. Download twine and give it your pypi credentials.
1. Get pypi permissions for the project from its owner.
1. `python setup.py sdist bdist_wheel; twine upload dist/*; rm -rf dist;`
