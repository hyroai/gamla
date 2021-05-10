import csv
import dataclasses
import itertools
import json
from typing import Any, Callable, Collection, Dict, List, Text, Tuple

import dataclasses_json

from gamla import currying, functional, functional_generic
from gamla.optimized import sync


def _immutable(self, *args, **kws):
    raise TypeError("cannot change object - object is immutable")


class frozendict(dict):  # noqa: N801
    def __init__(self, *args, **kwargs):
        self.__setattr__ = _immutable
        super(frozendict, self).__init__(*args, **kwargs)

    def __hash__(self):
        return hash(tuple(self.items()))

    # TODO(nitzo): Disabled since we need to be able to un-serialize with dill/pickle.
    # __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable


def get_encode_config():
    """Display dataclass field as a tuple of JSON strings."""
    return dataclasses.field(
        metadata=dataclasses_json.config(
            encoder=lambda lst: sorted(lst, key=json.dumps, reverse=False),
        ),
    )


def _freeze_nonterminal(v):
    if isinstance(v, Dict):
        return frozendict(v)
    return tuple(v)


#: Freeze recursively a dictionary.
#:
#: >>> freeze_deep({"1": {"2": "345", "some-string": ["hello"]}})
#: data.frozendict(
#:     {"1": data.frozendict({"2": "345", "some-string": ("hello",)})},
#: )
freeze_deep = functional_generic.map_dict(_freeze_nonterminal, functional.identity)


@currying.curry
def csv_to_list_of_dicts(csv_file_path, fieldnames=None) -> List:
    """Return a list of dicts given a CSV file path.

    >>> csv_to_list_of_dicts("data_test_example_with_headers.csv")
    [{'name': 'David', 'age': '23'}, {'name': 'Itay', 'age': '26'}]
    >>> csv_to_list_of_dicts("data_test_example_without_headers.csv", ["name", "age"])
    [{'name': 'David', 'age': '23'}, {'name': 'Itay', 'age': '26'}]
    """
    with open(csv_file_path, encoding="utf-8") as csvf:
        return list(csv.DictReader(csvf, fieldnames))


@currying.curry
def tuple_of_tuples_to_csv(
    tuple_of_tuples: Tuple[Tuple[Any], ...],
    separator: Text = "\t",
) -> Text:
    """Return a CSV formatted string given a tuple of tuples. Each element is separated by the character "separator" (default is \t).

    >>> tuple_of_tuples_to_csv((("name", "age"), ("David", "23"), ("Itay", "26")))
    'name\\tage\\nDavid\\t23\\nItay\\t26'
    >>> tuple_of_tuples_to_csv((("name", "age"), ("David", "23"), ("Itay", "26")), " ")
    'name age\\nDavid 23\\nItay 26'
    """
    return functional_generic.pipe(
        tuple_of_tuples,
        functional_generic.curried_map(
            functional_generic.compose_left(
                functional_generic.curried_map(str),
                tuple,
                separator.join,
            ),
        ),
        "\n".join,
    )


class Enum(frozenset):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


def _do_on_positions(f, predicate: Callable[[int], bool]):
    return functional_generic.compose_left(
        enumerate,
        functional_generic.curried_map(
            sync.ternary(
                functional_generic.compose_left(
                    functional.head,
                    predicate,
                ),
                functional_generic.compose_left(functional.second, f),
                functional.second,
            ),
        ),
    )


def explode(*positions: Collection[int]):
    """Flattens a non homogeneous iterable.

    For an iterable where some positions are iterable and some are not,
    "explodes" the iterable, so that each element appears in a single row, and duplicates the non iterable.

    >>> functional_generic.pipe(
        [
            "x",
            [
                "y1",
                "y2",
                "y3",
            ],
            "z",
        ],
        data.explode(1),
        tuple,
    )
    (
        ("x", "y1", "z"),
        ("x", "y2", "z"),
        ("x", "y3", "z"),
    )
    """
    return functional_generic.compose_left(
        _do_on_positions(
            functional.wrap_tuple,
            functional_generic.complement(functional.contains(positions)),
        ),
        functional.star(itertools.product),
    )
