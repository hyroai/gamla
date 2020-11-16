import asyncio
import functools
import inspect
import itertools
import logging
import operator
import os
from typing import Callable, Iterable, Mapping, Text, Tuple, Type, TypeVar

from gamla import currying, data, excepts_decorator, functional


def compose_left(*funcs):
    return compose(*reversed(funcs))


def _async_curried_map(f):
    async def async_curried_map(it):
        return await asyncio.gather(*map(f, it))

    return async_curried_map


def curried_map(f):
    if asyncio.iscoroutinefunction(f):
        return _async_curried_map(f)
    return functional.curried_map_sync(f)


def curried_to_binary(f):
    def internal(param1, param2):
        return f(param1)(param2)

    return internal


def _any_is_async(funcs):
    return any(map(asyncio.iscoroutinefunction, funcs))


async def to_awaitable(value):
    if inspect.isawaitable(value):
        return await value
    return value


_DEBUG_MODE = os.environ.get("GAMLA_DEBUG_MODE")


# Copying `toolz` convention.
# TODO(uri): Far from a perfect id, but should work most of the time.
# Improve by having higher order functions create meaningful names (e.g. `map`).
def _get_name_for_function_group(funcs):
    return "_of_".join(map(lambda x: x.__name__, funcs))


def _compose_async(*funcs):
    @functools.wraps(functional.last(funcs))
    async def async_composed(*args, **kwargs):
        for f in reversed(funcs):
            args = [await to_awaitable(f(*args, **kwargs))]
            kwargs = {}
        return functional.head(args)

    return async_composed


def compose_sync(*funcs):
    @functools.wraps(functional.last(funcs))
    def composed(*args, **kwargs):
        for f in reversed(funcs):
            args = [f(*args, **kwargs)]
            kwargs = {}
        return functional.head(args)

    return composed


def compose(*funcs):
    if _any_is_async(funcs):
        composed = _compose_async(*funcs)
    else:
        composed = compose_sync(*funcs)
    name = _get_name_for_function_group(funcs)
    if _DEBUG_MODE:
        logging.info("making call to `inspect` for debug mode")
        frames = inspect.stack()

        def reraise_and_log(e):
            for frame in frames:
                if "gamla" in frame.filename:
                    continue
                raise type(e)(
                    f"Composition involved in exception: {frame.filename}:{frame.lineno}",
                )
            raise e

        composed = excepts_decorator.excepts(Exception, reraise_and_log, composed)
    composed.__name__ = name
    return composed


def compose_many_to_one(incoming: Iterable[Callable], f: Callable):
    return compose_left(juxt(*incoming), functional.star(f))


@currying.curry
def after(f1, f2):
    return compose(f1, f2)


@currying.curry
def before(f1, f2):
    return compose_left(f1, f2)


def lazyjuxt(*funcs):
    """Reverts to eager implementation if any of `funcs` is async."""
    if _any_is_async(funcs):
        funcs = tuple(map(after(to_awaitable), funcs))

        async def lazyjuxt_async(*args, **kwargs):
            return await asyncio.gather(*map(lambda f: f(*args, **kwargs), funcs))

        return lazyjuxt_async

    def lazyjuxt(*args, **kwargs):
        for f in funcs:
            yield f(*args, **kwargs)

    return lazyjuxt


juxt = compose(after(tuple), lazyjuxt)
alljuxt = compose(after(all), lazyjuxt)
anyjuxt = compose(after(any), lazyjuxt)
juxtcat = compose(after(itertools.chain.from_iterable), lazyjuxt)


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


def when(condition, f_true):
    return ternary(condition, f_true, functional.identity)


def unless(condition, f_false):
    return ternary(condition, functional.identity, f_false)


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


allmap = compose(after(all), curried_map)
anymap = compose(after(any), curried_map)


itemmap = compose(after(dict), before(dict.items), curried_map)
keymap = compose(
    itemmap,
    lambda f: juxt(f, functional.second),
    before(functional.head),
)
valmap = compose(
    itemmap,
    lambda f: juxt(functional.head, f),
    before(functional.second),
)


def pair_with(f):
    return juxt(f, functional.identity)


def pair_right(f):
    return juxt(functional.identity, f)


def _sync_curried_filter(f):
    def curried_filter(it):
        for x in it:
            if f(x):
                yield x

    return curried_filter


curried_filter = compose(
    after(
        compose(
            functional.curried_map_sync(functional.second),
            _sync_curried_filter(functional.head),
        ),
    ),
    curried_map,
    pair_with,
)

itemfilter = compose(after(dict), before(dict.items), curried_filter)
keyfilter = compose(
    itemfilter,
    before(functional.head),
)
valfilter = compose(
    itemfilter,
    before(functional.second),
)

complement = after(operator.not_)

remove = compose(curried_filter, complement)


class NoConditionMatched(Exception):
    pass


def _case(predicates: Tuple[Callable, ...], mappers: Tuple[Callable, ...]):
    """Case with functions.

    Handles async iff one of the predicates or one of the mappers is async.
    Raises `NoConditionMatched` if no condition matched.
    """
    predicates = tuple(predicates)
    mappers = tuple(mappers)
    if _any_is_async(mappers + predicates):
        predicates = tuple(map(after(to_awaitable), predicates))
        mappers = tuple(map(after(to_awaitable), mappers))

        async def case_async(*args, **kwargs):
            for is_matched, mapper in zip(
                await asyncio.gather(*map(lambda f: f(*args, **kwargs), predicates)),
                mappers,
            ):
                if is_matched:
                    return await mapper(*args, *kwargs)
            raise NoConditionMatched

        return case_async

    def case(*args, **kwargs):
        for predicate, transformation in zip(predicates, mappers):
            if predicate(*args, **kwargs):
                return transformation(*args, **kwargs)
        raise NoConditionMatched

    return case


def case(predicates_and_mappers: Tuple[Tuple[Callable, Callable], ...]):
    predicates = tuple(map(functional.head, predicates_and_mappers))
    mappers = tuple(map(functional.second, predicates_and_mappers))
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
        return await pipe(value, _async_curried_map(_await_dict), type(value))
    return await to_awaitable(value)


def map_dict(nonterminal_mapper: Callable, terminal_mapper: Callable):
    def map_dict_inner(value):
        if isinstance(value, dict) or isinstance(value, data.frozendict):
            return pipe(
                value,
                dict,  # In case input is a `frozendict`.
                valmap(map_dict(nonterminal_mapper, terminal_mapper)),
                nonterminal_mapper,
            )
        if isinstance(value, Iterable) and not isinstance(value, str):
            return pipe(
                value,
                functional.curried_map_sync(
                    map_dict(nonterminal_mapper, terminal_mapper),
                ),
                type(value),  # Keep the same format as input.
                nonterminal_mapper,
            )
        return terminal_mapper(value)

    if _any_is_async([nonterminal_mapper, terminal_mapper]):
        return compose_left(map_dict_inner, _await_dict)

    return map_dict_inner


def _iterdict(d):
    results = []
    map_dict(functional.identity, results.append)(d)
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
                functional.identity,
                compose_left(functional.apply(*args, **kwargs), to_awaitable),
            )(spec)

        return apply_spec_async
    return compose_left(
        functional.apply,
        lambda applier: map_dict(functional.identity, applier),
        functional.apply(spec),
    )


# Stacks functions on top of each other, so will run pairwise on the input.
# Similar to juxt, only zips with the incoming iterable.
stack = compose_left(
    enumerate,
    functional.curried_map_sync(
        functional.star(lambda i, f: compose(f, lambda x: x[i])),
    ),
    functional.star(juxt),
)


def bifurcate(*funcs):
    """Serially runs each function on tee'd copies of `input_generator`."""
    return compose_left(iter, lambda it: itertools.tee(it, len(funcs)), stack(funcs))


average = compose_left(
    bifurcate(sum, functional.count),
    excepts_decorator.excepts(
        ZeroDivisionError,
        functional.just(0),
        functional.star(operator.truediv),
    ),
)


def value_to_dict(key: Text):
    return compose_left(
        functional.wrap_tuple,
        functional.prefix(key),
        functional.wrap_tuple,
        dict,
    )


_R = TypeVar("_R")
_E = TypeVar("_E")


def reduce_curried(
    reducer: Callable[[_R, _E], _R],
    initial_value: _R,
) -> Callable[[Iterable[_E]], _R]:
    if asyncio.iscoroutinefunction(reducer):

        async def reduce_async(elements):
            state = initial_value
            for element in elements:
                state = await reducer(state, element)
            return state

        return reduce_async

    def reduce(elements):
        state = initial_value
        for element in elements:
            state = reducer(state, element)
        return state

    return reduce


find = compose(
    after(
        excepts_decorator.excepts(
            StopIteration,
            functional.just(None),
            functional.head,
        ),
    ),
    curried_filter,
)

find_index = compose_left(
    before(functional.second),
    find,
    before(enumerate),
    after(ternary(functional.equals(None), functional.just(-1), functional.head)),
)


def check(condition, exception):
    return functional.do_if(
        complement(condition),
        functional.make_raise(exception),
    )


def countby_many(f):
    """Count elements of a collection by a function which returns a tuple of keys
    for single element.

    Parameters:
    f (Callable): Key function (given object in collection outputs tuple of keys).
    it (Iterable): Collection.

    Returns:
    Dict[Text, Any]: Dictionary where key has been computed by the `f` key function
    and value is the frequency of this key.

    >>> names = ['alice', 'bob', 'charlie', 'dan', 'edith', 'frank']
    >>> countby_many(lambda name: (name[0], name[-1]), names)
    {'a': 1,
     'e': 3,
     'b': 2,
     'c': 1,
     'd': 1,
     'n': 1,
     'h': 1,
     'f': 1,
     'k': 1}
    """
    return compose_left(
        curried_map(f),
        functional.groupby_many_reduce(
            functional.identity,
            lambda x, y: x + 1 if x else 1,
        ),
    )


def _inner_merge_with(dicts):
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result:
                result[k].append(v)
            else:
                result[k] = [v]
    return result


map_filter_empty = compose_left(curried_map, after(curried_filter(functional.identity)))


def merge_with(f):
    if asyncio.iscoroutinefunction(f):

        async def merge_with(*dicts):
            result = _inner_merge_with(dicts)
            return await valmap(f)(result)

        return merge_with

    def merge_with(*dicts):
        result = _inner_merge_with(dicts)
        return valmap(f)(result)

    return merge_with


merge = merge_with(functional.last)
concat = itertools.chain.from_iterable
mapcat = compose_left(curried_map, after(concat))
