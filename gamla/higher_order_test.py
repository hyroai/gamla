from gamla import functional, higher_order


def test_prepare_and_apply():
    def increment(x):
        return x + 1

    def decrement(x):
        return x - 1

    def conditional_transformation(x):
        return increment if x < 10 else decrement

    assert higher_order.prepare_and_apply(conditional_transformation)(15) == 14


async def test_prepare_and_apply_async():
    async def increment(x):
        return x + 1

    async def decrement(x):
        return x - 1

    def conditional_transformation(x):
        return increment if x < 10 else decrement

    assert (
        await higher_order.prepare_and_apply_async(conditional_transformation)(15) == 14
    )


def test_ignore_first():
    def increment(x):
        return x + 1

    assert higher_order.ignore_first(increment)("a", 2) == 3


def test_persistent_cache():
    d = {}

    def get_item(key: str):
        return d[key]

    def set_item(key: str, value):
        d[key] = value
        return

    def f(x):
        return x

    assert (
        higher_order.persistent_cache(
            get_item,
            set_item,
            functional.make_key("some key"),
        )(f)("something")
        == "something"
    )
