import dataclasses
import json
from typing import Any, Dict, Iterable, Optional, Text, Tuple

import dataclasses_json
import frozendict
import toolz
from toolz import curried


def get_encode_config():
    return dataclasses.field(
        metadata=dataclasses_json.config(
            encoder=lambda lst: sorted(lst, key=json.dumps, reverse=False)
        )
    )


def freeze_deep(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict) or isinstance(value, frozendict.frozendict):
        return toolz.pipe(
            value,
            dict,  # In case input is already a `frozendict`.
            curried.valmap(freeze_deep),
            frozendict.frozendict,
        )
    if isinstance(value, Iterable):
        return toolz.pipe(value, curried.map(freeze_deep), tuple)
    return value


@toolz.curry
def dict_to_csv(
    table: Dict[Any, Tuple], titles: Optional[Tuple] = None, separator: Text = "\t"
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
    tuple_of_tuples: Tuple[Tuple[Any], ...], separator: Text = "\t"
) -> Text:
    return toolz.pipe(
        tuple_of_tuples,
        curried.map(toolz.compose_left(curried.map(str), tuple, separator.join)),
        "\n".join,
    )


class Enum(frozenset):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError
