import time

from gamla import immutable_set


def test_add():
    assert immutable_set.equals(
        immutable_set.add(
            immutable_set.create([1, 2, 3]),
            4,
        ),
        immutable_set.create(
            [1, 2, 3, 4],
        ),
    )


def test_remove():
    assert immutable_set.equals(
        immutable_set.remove(
            immutable_set.create([1, 2, 3]),
            2,
        ),
        immutable_set.create([1, 3]),
    )


def test_contains():
    assert immutable_set.contains(immutable_set.create([1, 2, 3]), 3)


def test_not_contains():
    assert not immutable_set.contains(immutable_set.create([1, 2, 3]), 4)


def test_union():
    assert immutable_set.equals(
        immutable_set.union(
            immutable_set.create([1, 2, 3, 4]),
            immutable_set.create([1, 2, 3]),
        ),
        immutable_set.create([1, 2, 3, 4]),
    )


def _is_o_of_1(f, arg1, arg2):
    start = time.perf_counter()
    f(arg1, arg2)
    return time.perf_counter() - start < 0.0001


_large_number = 9999


def test_intersection():
    assert immutable_set.equals(
        immutable_set.intersection(
            immutable_set.create([1, 2]),
            immutable_set.create([2]),
        ),
        immutable_set.create([2]),
    )


def test_performance_sanity():
    assert not _is_o_of_1(
        immutable_set.union,
        immutable_set.create(range(_large_number)),
        immutable_set.create(range(_large_number)),
    )


def test_union_performance():
    assert _is_o_of_1(
        immutable_set.union,
        immutable_set.create(range(_large_number)),
        immutable_set.create(range(_large_number // 64, _large_number // 32)),
    )


def test_intersection_performance():
    assert _is_o_of_1(
        immutable_set.intersection,
        immutable_set.create(range(_large_number)),
        immutable_set.create(range(1)),
    )
