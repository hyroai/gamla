[![Build Status](https://travis-ci.com/hyroai/gamla.svg?branch=master)](https://travis-ci.com/hyroai/gamla)

גamla is a performant functional programming library for python which supports mixing `async` and regular functions.

Installation: `pip install gamla`

API reference: <https://gamla.readthedocs.io/>

## Basic example

`gamla` can help you turn this:

```python
import dataclasses

@dataclasses.dataclass
class Person:
    age: int
    name: str

    def is_eligible(self):
        return self.age > 9


def get_names_eligible_for_vaccine(people):
    result = []
    for person in people:
        if person.is_eligible():
            result.append(person.name)
    return result


```

into this:

```python
import dataclasses
from gamla import attrgetter, greater_than, compose_left, filter, map

@dataclasses.dataclass(frozen=True)
class Person:
    age: int
    name: str

is_eligible = compose_left(attrgetter("age"), greater_than(9))
get_names_eligible_for_vaccine = compose_left(filter(is_eligible), map(attrgetter("name")), list)

```

Is this a good thing? that's for you to decide.

The upside:

Functional programming is mainly about how to split your code into composable parts. Composability means that things are easy to move, replace or combine together like lego. It helps you identify recurring patterns (e.g. `filter`), factor them out and reuse them. If your generalizations are good, they free your mind to focus on the new logic. Concretely it saves a lot of code and helps a reader understand what a piece of code is doing. For example, if you are familir with what `filter` is, you don't have to squint and realize that an `if` and a `for` actually do a filtering pattern.

The downside:

Programming in this style in python means some tools won't be so useful (e.g. your debugger, static analysis tools).

## Debugging anonymous compositions

### `gamla.debug`

Classic breakpoints are less useful when working with compositions, as there isn't always a line of code to place the breakpoint on. Instead one can use `gamla.debug` and `gamla.debug_exception`.

`gamla.debug` can be used within pipelines and will provide a `pdb` breakpoint prompt where the value at this position can be referenced by `x`.

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

## Mixing asynchronous and synchronous code

Most functions in this lib will work seamlessly with `async` and regular functions, and allow the developer to focus on the logic instead of deciding where to place an `await`.

For example:

```python
import asyncio

import gamla


def increment(i):
    return i + 1


async def increment_async(i):
    await asyncio.sleep(1)
    return i + 1


async def run():
    mixed_composition = gamla.compose_left(increment, increment_async, increment)
    return await mixed_composition(0)  # returns 3!
```

## Releasing a new version

Increment the version in master, and pypi will automatically update.

## Updating documentation after change in README.md

### While in gamla directory:

1. Install md-to-rst converter: `pip install m2r`
1. Convert README.md to README.rst: `m2r README.md`
1. Move README.rst to docs/source folder instead of existing one: `mv README.rst docs/source`
