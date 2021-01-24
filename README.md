[![Build Status](https://travis-ci.com/hyroai/gamla.svg?branch=master)](https://travis-ci.com/hyroai/gamla)

גamla is a performant functional programming library for python which supports `async`.

## Installation

`pip install gamla`

## Debugging anonymous compositions

### `gamla.debug`

It is sometimes hard to debug pipelines because you can't place ordinary breakpoints. For this `gamla.debug` and `gamla.debug_exception` were created.

`gamla.debug` can be used within pipelines and provide a pdb breakpoint prompt where the value at this position can be referenced by `x`.

```python

def increment(x):
    return x + 1

increment_twice = gamla.compose_left(increment, gamla.debug, increment)

increment_twice(1)
```

The above code will break with `x` being 2.

When you have a long pipeline and want to debug at each step of the way, you can use `gamla.debug_compose` and `gamla.debug_compose_left`.

### `gamla.debug_exception`

In some cases tracking down an exception involves inspecting code that runs many times. Consider the following example:

```python

def increment(x):
    return x + 1

def sometimes_has_a_bug(x):
    if x == 666:
        raise Exception
    return x

increment_with_bug = gamla.map(gamla.compose_left(increment, sometimes_has_a_bug))

tuple(inrement_with_bug(range(1000)))
```

Adding a `gamla.debug` here can be quite tedious, because the code will break many times.

Instead we can use `gamla.debug_exception` to break only in the case the inner function raises, at which case we would get a breakpoint prompt, and be able to inspect the value causing the exception, use the name `x`. This would like this:

`increment_with_bug = gamla.map(gamla.compose_left(increment, gamla.debug_exception(sometimes_has_a_bug)))`

One can also use `gamla.debug_exception` using a decorator.

```python
@gamla.debug_exception
def sometimes_has_a_bug(x):
    if x == 666:
        raise Exception
    return x

```

### Debug mode

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

Increment the version in master, and pypi will automatically update.

## Updating documentation after change in README.md

### While in gamla directory:

1. Install md-to-rst converter: `pip install m2r`
1. Convert README.md to README.rst: `m2r README.md`
1. Move README.rst to docs/source folder instead of existing one: `mv README.rst docs/source`
