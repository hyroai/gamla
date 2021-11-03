import pickle

from gamla import data, functional_generic


def test_freeze_deep():
    original = {"1": {"2": "345", "some-string": ["hello"]}}
    frozen = data.freeze_deep(original)
    # Check values are intact.
    assert frozen == data.frozendict(
        {"1": data.frozendict({"2": "345", "some-string": ("hello",)})},
    )
    # Check hashability.
    {data.freeze_deep(original)}


def test_freeze_deep_idempotent():
    assert data.freeze_deep(data.freeze_deep({"a": 1})) == data.freeze_deep({"a": 1})


def test_frozendict_serializable():
    fd = data.frozendict({"a": "something", "b": 1})
    fd_str = pickle.dumps(fd)

    fd_clone = pickle.loads(fd_str)

    assert fd == fd_clone


def test_tuple_of_tuples_to_csv():
    assert (
        data.tuple_of_tuples_to_csv((("name", "age"), ("David", "23"), ("Itay", "26")))
        == "name\tage\nDavid\t23\nItay\t26"
    )


def test_tuple_of_tuples_to_csv_custom_separator():
    assert (
        data.tuple_of_tuples_to_csv(
            (("name", "age"), ("David", "23"), ("Itay", "26")),
            " ",
        )
        == "name age\nDavid 23\nItay 26"
    )


def test_explode():
    assert functional_generic.pipe(
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
    ) == (
        ("x", "y1", "z"),
        ("x", "y2", "z"),
        ("x", "y3", "z"),
    )
