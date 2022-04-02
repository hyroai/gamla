from gamla import construct, functional_generic


def test_wrap_str():
    assert functional_generic.pipe("john", construct.wrap_str("hi {}")) == "hi john"


def test_wrap_multiple_str():
    assert (
        functional_generic.pipe(
            {"first": "happy", "second": "world"},
            construct.wrap_multiple_str("hello {first} {second}"),
        )
        == "hello happy world"
    )


def test_wrap_dict():
    assert construct.wrap_dict("some_key")("some_value") == {"some_key": "some_value"}
