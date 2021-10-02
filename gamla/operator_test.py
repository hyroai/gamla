from gamla import operator


def _generator():
    yield 1
    yield 2
    yield 5


def test_nth():
    assert operator.nth(1)(["a", "b", "c"]) == "b"


def test_len_equals():
    assert operator.len_equals(3)(_generator())


def test_not_empty():
    assert operator.empty(_generator()) is False


def test_is_empty():
    assert operator.empty([]) is True


def test_nonempty():
    assert operator.nonempty(_generator()) is True


def test_between():
    assert operator.between(1, 100)(30) is True


def test_not_between():
    assert operator.between(2, 5)(14) is False
