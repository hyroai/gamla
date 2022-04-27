from gamla import dict_utils, functional_generic, operator


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


def test_get_or_transform():
    assert dict_utils.get_or_transform(operator.add(1), {1: 1})(1) == 1
    assert dict_utils.get_or_transform(operator.add(1), {1: 1})(3) == 4


def test_get_or_identity():
    assert dict_utils.get_or_identity({1: 1})(1) == 1
    assert dict_utils.get_or_identity({1: 1})(2) == 2


def test_make_index():
    index = dict_utils.make_index(
        map(functional_generic.groupby, [operator.head, operator.second]),
    )(["uri", "dani"])
    assert index("d")("a") == frozenset(["dani"])
    assert index("u")("r") == frozenset(["uri"])
    assert index("h")("i") == frozenset()


def test_rename_key():
    assert dict_utils.rename_key("name", "first_name")(
        {"name": "Danny", "age": 20},
    ) == {"first_name": "Danny", "age": 20}


def test_add_key_value():
    assert dict_utils.add_key_value("1", "1")({"2": "2"}) == {"1": "1", "2": "2"}


def test_remove_key():
    assert dict_utils.remove_key("1")({"1": 1, "2": 2}) == {"2": 2}


def test_transform_item():
    assert dict_utils.transform_item("1", operator.add(1))({"1": 1, "2": 2}) == {
        "1": 2,
        "2": 2,
    }
