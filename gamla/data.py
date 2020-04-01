import dataclasses
import json
from typing import Dict, Iterable

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
        dict, # In case input is already a `frozendict`.
        curried.valmap(_freeze_deep_inner),
        frozendict.frozendict
    )

