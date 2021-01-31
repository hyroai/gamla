import csv
import dataclasses
import itertools
import json
from typing import Any, Dict, List, Text, Tuple

import dataclasses_json
import toolz
from toolz import curried
from toolz.curried import operator

from gamla import currying, functional, functional_generic


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
    """
    Display dataclass field as a tuple of JSON strings.
    """
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
    """
    Return a list of dicts given a CSV file path.

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
    """
    Return a CSV formatted string given a tuple of tuples. Each element is separated by the character "separator" (default is \t).

    >>> tuple_of_tuples_to_csv((("name", "age"), ("David", "23"), ("Itay", "26")))
    'name\\tage\\nDavid\\t23\\nItay\\t26'
    >>> tuple_of_tuples_to_csv((("name", "age"), ("David", "23"), ("Itay", "26")), " ")
    'name age\\nDavid 23\\nItay 26'
    """
    return functional_generic.pipe(
        tuple_of_tuples,
        curried.map(
            functional_generic.compose_left(
                functional_generic.curried_map(str),
                tuple,
                separator.join,
            ),
        ),
        "\n".join,
    )


_field_getters = functional_generic.compose_left(
    dataclasses.fields,
    functional_generic.curried_map(
        functional_generic.compose_left(
            functional.attrgetter("name"),
            functional.attrgetter,
        ),
    ),
    tuple,
)


def match(dataclass_pattern):
    """
    Create a function that returns true if input matches dataclass_pattern.
    Use data.Any as wildcard for field value.
    Supports recursive patterns.

    >>> @dataclasses.dataclass(frozen=True)
    >>> class MockDataclassA:
    >>>     a: int

    >>> @dataclasses.dataclass(frozen=True)
    >>> class MockDataclassB:
    >>>     b: MockDataclassA

    >>> match(MockDataclassB(MockDataclassA(5)))(
    >>>    MockDataclassB(MockDataclassA(4)),
    >>> )
    False

    >>> match(MockDataclassB(MockDataclassA(Any)))(
    >>>    MockDataclassB(MockDataclassA(4)),
    >>> )
    True

    >>> match(MockDataclassB(Any))(MockDataclassB(MockDataclassA(4)))
    True
    """
    # pattern -> ( (getter,...), pattern) -> ((getter,...), (value,...)) ->
    # ((getter,...), (eq(value),...)) -> alljuxt( compose_left(getter,eq(value)),... )
    return functional_generic.pipe(
        dataclass_pattern,
        functional_generic.juxt(_field_getters, itertools.repeat),
        functional_generic.juxt(
            toolz.first,
            functional.star(
                curried.map(
                    functional_generic.compose_left(
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
        functional.star(
            curried.map(functional_generic.compose_left),
        ),
        functional.prefix(
            functional_generic.compose_left(
                type,
                functional.equals(type(dataclass_pattern)),
            ),
        ),
        functional.star(functional_generic.alljuxt),
    )


class Enum(frozenset):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError
