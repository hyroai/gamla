import asyncio
import functools
import inspect
import itertools
from typing import Callable, Iterable, Text, Tuple, Type

import toolz
from toolz import curried
from toolz.curried import operator

from gamla import data, functional


def compose_left(*funcs):
    return compose(*reversed(funcs))


_any_is_async = toolz.compose(any, curried.map(asyncio.iscoroutinefunction))


async def to_awaitable(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _acompose(*funcs):
    @functools.wraps(toolz.last(funcs))
    async def async_composed(*args, **kwargs):
        for f in reversed(funcs):
            args = [await to_awaitable(f(*args, **kwargs))]
            kwargs = {}
        return toolz.first(args)

    return async_composed


def compose(*funcs):
    if _any_is_async(funcs):
        composed = _acompose(*funcs)
    else:

        @functools.wraps(toolz.last(funcs))
        def composed(*args, **kwargs):
            for f in reversed(funcs):
                args = [f(*args, **kwargs)]
                kwargs = {}
            return toolz.first(args)

    # TODO(uri): Far from a perfect id, but should work most of the time.
    # Improve by having higher order functions create meaningful names (e.g. `map`).
    # Copying `toolz` convention.
    composed.__name__ = "_of_".join(map(lambda x: x.__name__, funcs))
    return composed


def _make_amap(f):
    async def amap_inner(it):
        return await asyncio.gather(*curried.map(f, it))

    return amap_inner


def gamla_map(*args):
    if len(args) == 2:
        f, it = args
        if asyncio.iscoroutinefunction(f):
            return _make_amap(f)(it)
        return curried.map(f, it)
    [f] = args
    if asyncio.iscoroutinefunction(f):
        return _make_amap(f)
    return curried.map(f)


map = gamla_map


@toolz.curry
def after(f1, f2):
    return compose(f1, f2)


@toolz.curry
def before(f1, f2):
    return compose_left(f1, f2)


def lazyjuxt(*funcs):
    """Reverts to eager implementation if any of `funcs` is async."""
    if _any_is_async(funcs):

        async def lazyjuxt_inner(*args, **kwargs):
            return await toolz.pipe(
                funcs,
                curried.map(
                    compose_left(functional.apply(*args, **kwargs), to_awaitable),
                ),
                functional.star(asyncio.gather),
            )

        return lazyjuxt_inner
    return compose_left(functional.apply, curried.map, functional.apply(funcs))


juxt = compose(after(tuple), lazyjuxt)
alljuxt = compose(after(all), lazyjuxt)
anyjuxt = compose(after(any), lazyjuxt)
juxtcat = compose(after(toolz.concat), lazyjuxt)


def ternary(condition, f_true, f_false):
    if _any_is_async([condition, f_true, f_false]):

        async def ternary_inner_async(*args, **kwargs):
            return (
                await to_awaitable(f_true(*args, **kwargs))
                if await to_awaitable(condition(*args, **kwargs))
                else await to_awaitable(f_false(*args, **kwargs))
            )

        return ternary_inner_async

    def ternary_inner(*args, **kwargs):
        return (
            f_true(*args, **kwargs)
            if condition(*args, **kwargs)
            else f_false(*args, **kwargs)
        )

    return ternary_inner


curried_ternary = ternary


def first(*funcs, exception_type: Type[Exception]):
    if _any_is_async([*funcs]):

        async def inner_async(*args, **kwargs):
            for func in funcs:
                try:
                    return await to_awaitable(func(*args, **kwargs))
                except exception_type:
                    pass
            raise exception_type

        return inner_async

    def inner(*args, **kwargs):
        for func in funcs:
            try:
                return func(*args, **kwargs)
            except exception_type:
                pass
        raise exception_type

    return inner


def pipe(val, *funcs):
    return compose_left(*funcs)(val)


def _compose_over_binary_curried(composer):
    def composition_over_binary_curried(*args):
        f = composer(args[0])
        if len(args) == 2:
            return f(args[1])
        return f

    return composition_over_binary_curried


allmap = _compose_over_binary_curried(compose(after(all), gamla_map))
anymap = _compose_over_binary_curried(compose(after(any), gamla_map))


itemmap = _compose_over_binary_curried(
    compose(after(dict), before(dict.items), gamla_map),
)
keymap = _compose_over_binary_curried(
    compose(itemmap, lambda f: juxt(f, toolz.second), before(toolz.first)),
)

valmap = _compose_over_binary_curried(
    compose(itemmap, lambda f: juxt(toolz.first, f), before(toolz.second)),
)


pair_with = _compose_over_binary_curried(lambda f: juxt(f, toolz.identity))
pair_right = _compose_over_binary_curried(lambda f: juxt(toolz.identity, f))

filter = _compose_over_binary_curried(
    compose(
        after(compose(curried.map(toolz.second), curried.filter(toolz.first))),
        gamla_map,
        pair_with,
    ),
)


_first_truthy_index = compose_left(
    enumerate,
    curried.filter(toolz.second),
    curried.map(toolz.first),
    toolz.excepts(StopIteration, toolz.first),
)


class NoConditionMatched(Exception):
    pass


def _case(predicates: Tuple[Callable, ...], mappers: Tuple[Callable, ...]):
    """Case with functions.

    Handles async iff one of the predicates is async.
    Raises `NoConditionMatched` if no condition matched.
    """
    predicates = (*predicates, functional.just(True))
    mappers = (
        *mappers,
        compose_left(NoConditionMatched, functional.just_raise),
    )
    return compose_left(
        pair_right(
            compose_left(
                lazyjuxt(*predicates), _first_truthy_index, mappers.__getitem__,
            ),
        ),
        functional.star(
            lambda value, transformer: functional.apply(value)(transformer),
        ),
    )


def case(predicates_and_mappers: Tuple[Tuple[Callable, Callable], ...]):
    predicates = tuple(map(toolz.first, predicates_and_mappers))
    mappers = tuple(map(toolz.second, predicates_and_mappers))
    return _case(predicates, mappers)


case_dict = compose_left(dict.items, tuple, case)


async def _await_dict(value):
    if isinstance(value, dict) or isinstance(value, data.frozendict):
        return await pipe(
            value,
            # In case input is a `frozendict`.
            dict,
            valmap(_await_dict),
        )
    if isinstance(value, Iterable):
        return await pipe(value, gamla_map(_await_dict), type(value))
    return await to_awaitable(value)


def map_dict(nonterminal_mapper: Callable, terminal_mapper: Callable):
    def map_dict_inner(value):
        if isinstance(value, dict) or isinstance(value, data.frozendict):
            return toolz.pipe(
                value,
                dict,  # In case input is a `frozendict`.
                curried.valmap(map_dict(nonterminal_mapper, terminal_mapper)),
                nonterminal_mapper,
            )
        if isinstance(value, Iterable) and not isinstance(value, str):
            return toolz.pipe(
                value,
                curried.map(map_dict(nonterminal_mapper, terminal_mapper)),
                type(value),  # Keep the same format as input.
                nonterminal_mapper,
            )
        return terminal_mapper(value)

    if _any_is_async([nonterminal_mapper, terminal_mapper]):
        return compose_left(map_dict_inner, _await_dict)

    return map_dict_inner


def _iterdict(d):
    results = []
    map_dict(toolz.identity, results.append)(d)
    return results


_has_coroutines = compose_left(_iterdict, _any_is_async)


def apply_spec(spec):
    """
    >>> spec = {"len": len, "sum": sum}
    >>> apply_spec(spec)([1,2,3,4,5])
    {'len': 5, 'sum': 15}

    Notes:
    - The dictionary can be nested.
    - Returned function will be async iff any leaf is an async function.
    """
    if _has_coroutines(spec):

        async def apply_spec_async(*args, **kwargs):
            return await map_dict(
                toolz.identity,
                compose_left(functional.apply(*args, **kwargs), to_awaitable),
            )(spec)

        return apply_spec_async
    return compose_left(
        functional.apply,
        lambda applier: map_dict(toolz.identity, applier),
        functional.apply(spec),
    )


# Stacks functions on top of each other, so will run pairwise on the input.
# Similar to juxt, only zips with the incoming iterable.
stack = compose_left(
    enumerate,
    map(functional.star(lambda i, f: compose(f, curried.nth(i)))),
    functional.star(juxt),
)


def bifurcate(*funcs):
    """Serially runs each function on tee'd copies of `input_generator`."""
    return compose_left(iter, lambda it: itertools.tee(it, len(funcs)), stack(funcs))


average = toolz.compose_left(
    bifurcate(sum, toolz.count),
    toolz.excepts(ZeroDivisionError, functional.star(operator.truediv), lambda _: 0),
)


def value_to_dict(key: Text):
    return compose_left(
        functional.wrap_tuple, functional.prefix(key), functional.wrap_tuple, dict,
    )
