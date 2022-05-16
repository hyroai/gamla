from gamla import higher_order


def test_prepare_and_apply_sync():
    def increment(x):
        return x + 1

    def decrement(x):
        return x - 1

    def conditional_transformation(x):
        return increment if x < 10 else decrement

    assert higher_order.prepare_and_apply_sync(conditional_transformation)(15) == 14


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


async def test_prepare_and_apply():
    def increment(x):
        return x + 1

    def decrement(x):
        return x - 1

    def conditional_transformation(x):
        return increment if x < 10 else decrement

    assert higher_order.prepare_and_apply(conditional_transformation)(15) == 14

    async def async_increment(x):
        return x + 1

    async def async_decrement(x):
        return x - 1

    def async_conditional_transformation(x):
        return async_increment if x < 10 else async_decrement

    assert (
        await higher_order.prepare_and_apply(async_conditional_transformation)(15) == 14
    )
