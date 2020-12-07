[![Build Status](https://travis-ci.com/hyroai/gamla.svg?branch=master)](https://travis-ci.com/hyroai/gamla)

גamla is a performant functional programming library for python which supports `async`.

## Installation

`pip install gamla`

## Debugging anonymous compositions

`gamla.compose(x, y, z)` produces a new function which doesn't have a proper name. If `x` raises an exception, it is sometimes hard to figure out where this occurred. To overcome this, set the env variable `GAMLA_DEBUG_MODE` (to anything) to get more useful exceptions. This is turned on only by flag because it incurs significant overhead so things might get slow.

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

## Migrating from `toolz`

The main problems - `toolz` is slow and does not support `async` functions.

### Why are curried functions and composition in `toolz` slow?

These functions use an expensive `inspect` call to look at a function’s arguments, and doing so at each run.

### Why does `gamla` not suffer from this problem?

Two reasons:

1. It does not have binary signatures on things like `map`, so it doesn’t need to infer anything (these are higher order functions in `gamla`).
1. The `gamla.curry` function eagerly pays for the signature inspection in advance, and remembers its results for future runs.

### Function mapping and common gotchas:

Most functions are drop in replacements. Here are some examples:

- `curried.(filter|map|valmap|itemmap|keymap)` -> `gamla.$1` (make sure the call is with a single argument)
- `toolz.identity` -> `gamla.identity`
- `toolz.contains` -> `gamla.contains`
- `toolz.lt` -> `gamla.greater_than`
- `toolz.gt` -> `gamla.less_than`
- `toolz.ge` -> `gamla.less_equals`
- `toolz.le` -> `gamla.greater_equals`
- `toolz.filter(None) -> gamla.filter(gamla.identity)`
- `toolz.excepts(a, b, c)` -> `gamla.excepts(a, c, b)`
- `toolz.excepts(a, b)` -> `gamla.excepts(a, gamla.just(None), b)` (following the “data-last” currying convention)

## Releasing a new version

1. Create a pypi account.
1. Download twine and give it your pypi credentials.
1. Get pypi permissions for the project from its owner.
1. `python setup.py sdist bdist_wheel; twine upload dist/*; rm -rf dist;`

## How to update gamla documentation after library update

### If a new function was added

1. Go to `docs/api.rst` and add your function name under the relevant module, with an indentation of 3 spaces.
   For example:

```rest
.. currentmodule:: gamla.functional_generic

.. autosummary::
   old_functions
   .
   .
   .
   my_new_function
```

### If README.md was updated

While in gamla directory:

1. Install md-to-rst converter: `pip install m2r`
1. Convert README.md to README.rst: `m2r README.md`
1. Move README.rst to docs/source folder instead of existing one: `mv README.rst docs/source`

### If an existing function was updated

Do nothing. The documentation will update itself.
