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


def freeze_deep(dict_to_freeze: Dict) -> frozendict.frozendict:
    dict_to_freeze = dict(dict_to_freeze)  # In case input is already a `frozendict`.
    for key in dict_to_freeze:
        if isinstance(dict_to_freeze[key], str):
            continue
        if isinstance(dict_to_freeze[key], dict):
            dict_to_freeze[key] = freeze_deep(dict_to_freeze[key])
        elif isinstance(dict_to_freeze[key], Iterable):
            dict_to_freeze[key] = toolz.pipe(
                dict_to_freeze[key],
                curried.map(
                    functional.curried_ternary(
                        lambda x: isinstance(x, dict), freeze_deep, toolz.identity
                    )
                ),
                tuple,
            )

    return frozendict.frozendict(dict_to_freeze)
