import dataclasses
import json
from typing import Any, Dict, Optional, Text, Tuple

import dataclasses_json
import frozendict
import toolz
from toolz import curried

from gamla import functional_generic


def get_encode_config():
    return dataclasses.field(
        metadata=dataclasses_json.config(
            encoder=lambda lst: sorted(lst, key=json.dumps, reverse=False),
        ),
    )


def _freeze_nonterminal(v):
    if isinstance(v, Dict):
        return frozendict.frozendict(v)
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


class Enum(frozenset):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError
