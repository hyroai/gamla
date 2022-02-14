from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pytest

from gamla import type_safety


def test_no_annotations():
    def f(x):
        pass

    def g(x):
        pass

    assert type_safety.composable(f, g, None)


def test_simple_types():
    def f(x: int) -> str:
        pass

    def g(x: str) -> str:
        pass

    assert type_safety.composable(g, f, None)
    assert not type_safety.composable(f, g, None)


def test_compose_on_key():
    def f(x: int, y: str) -> str:
        pass

    def g(x: str) -> str:
        pass

    assert type_safety.composable(f, g, "y")
    assert not type_safety.composable(f, g, "x")


@pytest.mark.parametrize(
    "subtype,supertype",
    [
        [Callable[[str], str], Callable[[str], str]],
        [Callable[..., str], Callable[[str], str]],
        [Callable[[str], str], Callable[..., str]],
        [Dict[str, int], Dict[str, int]],
        [type(None), Optional[str]],
        [type(None), type(None)],
        [Optional[str], Optional[str]],
        [Optional[str], Union[None, str]],
        [str, Optional[str]],
        [FrozenSet[str], FrozenSet[str]],
        [str, Any],
        [Tuple[str, ...], Tuple[str, ...]],
        [Set[str], Collection[str]],
        [List, Sequence],
        [Union[int, str], Union[int, str]],
        [str, Union[int, str]],
        [Union[List, Set], Collection],
        [Dict, Dict[str, int]],
    ],
)
def test_is_subtype(subtype, supertype):
    assert type_safety.is_subtype(subtype, supertype)


@pytest.mark.parametrize(
    "subtype,supertype",
    [
        [Callable[[str, str], str], Callable[[str], str]],
        [Callable[[int, str], str], Callable[[str, int], str]],
        [Dict[str, str], Dict[str, int]],
        [Optional[str], type(None)],
        [Optional[str], Optional[int]],
        [int, Optional[str]],
        [FrozenSet[int], FrozenSet[str]],
        [str, FrozenSet[str]],
        [Collection, FrozenSet],
        [Tuple[str, ...], Tuple[int, ...]],
        [Union[int, str], int],
        [Any, str],
        [List, Union[int, str]],
        [Union[int, str, List], Union[int, str]],
        [Dict, Tuple],
    ],
)
def test_is_not_subtype(subtype, supertype):
    assert not type_safety.is_subtype(subtype, supertype)
