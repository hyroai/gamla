from gamla import immutable_set


def test_add():
    assert immutable_set.add(
        immutable_set.create([1, 2, 3]),
        4,
    ) == immutable_set.create(
        [1, 2, 3, 4],
    )


def test_remove():
    assert immutable_set.remove(
        immutable_set.create([1, 2, 3]),
        2,
    ) == immutable_set.create([1, 3])


def test_contains():
    assert 3 in immutable_set.create([1, 2, 3])


def test_not_contains():
    assert 4 not in immutable_set.create([1, 2, 3])


def test_union():
    assert immutable_set.union(
        immutable_set.create([1, 2, 3, 4]),
        immutable_set.create([1, 2, 3]),
    ) == immutable_set.create([1, 2, 3, 4])


def test_intersection():
    assert immutable_set.intersection(
        immutable_set.create([1, 2]),
        immutable_set.create([2]),
    ) == immutable_set.create([2])
