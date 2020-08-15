import dataclasses
import itertools
import json
from typing import Any, Dict, Optional, Text, Tuple

import dataclasses_json
import toolz
from toolz import curried
from toolz.curried import operator

from gamla import functional, functional_generic


def _immutable(self, *args, **kws):
    raise TypeError("cannot change object - object is immutable")


class frozendict(dict):  # noqa: N801
    def __init__(self, *args, **kwargs):
        self._hash = None
        self.__setattr__ = _immutable
        super(frozendict, self).__init__(*args, **kwargs)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(self.items()))
        return self._hash

    # TODO(nitzo): Disabled since we need to be able to un-serialize with dill/pickle.
    # __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable


def get_encode_config():
    return dataclasses.field(
        metadata=dataclasses_json.config(
            encoder=lambda lst: sorted(lst, key=json.dumps, reverse=False),
        ),
    )


def _freeze_nonterminal(v):
    if isinstance(v, Dict):
        return frozendict(v)
    return tuple(v)


freeze_deep = functional_generic.map_dict(_freeze_nonterminal, toolz.identity)


@toolz.curry
def dict_to_csv(
    table: Dict[Any, Tuple], titles: Optional[Tuple] = None, separator: Text = "\t",
) -> Text:
    return toolz.pipe(
        table,
        dict_to_tuple_of_tuples,
        tuple_of_tuples_to_csv(titles=titles, separator=separator),
    )


dict_to_tuple_of_tuples = toolz.compose_left(
    dict.items,
    curried.map(curried.compose_left(lambda x: (x[0], *x[1]), curried.map(str), tuple)),
    tuple,
)


@toolz.curry
def tuple_of_tuples_to_csv(
    tuple_of_tuples: Tuple[Tuple[Any], ...], separator: Text = "\t",
) -> Text:
    return toolz.pipe(
        tuple_of_tuples,
        curried.map(toolz.compose_left(curried.map(str), tuple, separator.join)),
        "\n".join,
    )


_field_getters = toolz.compose_left(
    dataclasses.fields,
    curried.map(toolz.compose_left(lambda f: f.name, operator.attrgetter)),
    tuple,
)


def match(dataclass_pattern):
    """ creates a function that returns true if input matches dataclass_pattern.
     Use data.Any as wildcard for field value.
     Supports recursive patterns.
     """
    # pattern -> ( (getter,...), pattern) -> ((getter,...), (value,...)) ->
    # ((getter,...), (eq(value),...)) -> alljuxt( compose_left(getter,eq(value)),... )
    return toolz.pipe(
        dataclass_pattern,
        toolz.juxt(_field_getters, itertools.repeat),
        toolz.juxt(
            toolz.first,
            functional.star(
                curried.map(
                    toolz.compose_left(
                        toolz.apply,
                        functional_generic.case(
                            (
                                (
                                    operator.eq(Any),
                                    functional.just(functional.just(True)),
                                ),
                                (dataclasses.is_dataclass, match),
                                (functional.just(True), operator.eq),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        functional.star(curried.map(toolz.compose_left)),
        functional.prefix(lambda dc: type(dc) == type(dataclass_pattern)),
        functional.star(functional_generic.alljuxt),
    )


class Enum(frozenset):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError
