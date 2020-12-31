from gamla import url_utils


def test_add_to_query_string1():
    assert (
        url_utils.add_to_query_string(
            {"a": 123},
            "https://www.domain.com/path?param1=param1#anchor",
        )
        == "https://www.domain.com/path?param1=param1&a=123#anchor"
    )


def test_add_to_query_string2():
    assert (
        url_utils.add_to_query_string(
            {"param1": 123},
            "https://www.domain.com/path?param1=param1#anchor",
        )
        == "https://www.domain.com/path?param1=123#anchor"
    )
