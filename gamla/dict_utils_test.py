from gamla import dict_utils, functional, functional_generic


def test_itemgetter():
    assert dict_utils.itemgetter("a")({"a": 1}) == 1


def test_itemgetter_with_default():
    assert dict_utils.itemgetter_with_default(2, "b")({"a": 1}) == 2


def test_itemgetter_or_none():
    assert dict_utils.itemgetter_or_none("b")({"a": 1}) is None


def test_get_in():
    assert dict_utils.get_in(["a", "b", "c", 1])({"a": {"b": {"c": [0, 1, 2]}}}) == 1


def test_get_in_or_none():
    assert (
        dict_utils.get_in_or_none(["a", "b", "d", 1])({"a": {"b": {"c": [0, 1, 2]}}})
        is None
    )


def test_get_in_or_none_uncurried():
    assert (
        dict_utils.get_in_or_none_uncurried(
            ["a", "b", "c", 1],
            {"a": {"b": {"c": [0, 1, 2]}}},
        )
        == 1
    )


def test_dict_to_getter_with_default_value_exists():
    assert dict_utils.dict_to_getter_with_default(None, {1: 1})(1) == 1


def test_dict_to_getter_with_default_values_does_not_exist():
    assert dict_utils.dict_to_getter_with_default(None, {1: 1})(2) is None


def test_get_or_identity():
    assert dict_utils.get_or_identity({1: 1})(1) == 1
    assert dict_utils.get_or_identity({1: 1})(2) == 2


def test_make_index():
    index = dict_utils.make_index(
        map(functional_generic.groupby, [functional.head, functional.second]),
    )(["uri", "dani"])
    assert index("d")("a") == frozenset(["dani"])
    assert index("u")("r") == frozenset(["uri"])
    assert index("h")("i") == frozenset()
