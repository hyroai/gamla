import dataclasses
import json
from typing import Any, Dict, Iterable, Optional, Text, Tuple

import dataclasses_json
import frozendict
import toolz
from toolz import curried

from gamla import functional


def get_encode_config():
    return dataclasses.field(
        metadata=dataclasses_json.config(
            encoder=lambda lst: sorted(lst, key=json.dumps, reverse=False)
        )
    )


def _freeze_deep_inner(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return freeze_deep(value)
    elif isinstance(value, Iterable):
        return toolz.pipe(
            value,
            curried.map(
                functional.curried_ternary(
                    lambda x: isinstance(x, dict), freeze_deep, toolz.identity
                )
            ),
            tuple,
        )

    return value


def freeze_deep(dict_to_freeze: Dict) -> frozendict.frozendict:
    return toolz.pipe(
        dict_to_freeze,
        dict,  # In case input is already a `frozendict`.
        curried.valmap(_freeze_deep_inner),
        frozendict.frozendict,
    )


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
