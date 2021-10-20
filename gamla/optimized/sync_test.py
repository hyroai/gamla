import pytest

from gamla import operator
from gamla.optimized import sync

pytestmark = pytest.mark.asyncio


def test_valfilter_sync():
    assert (
        sync.valfilter(operator.identity)(
            {False: True, True: False},
        )
        == {False: True}
    )


def test_pair_right_sync():
    assert sync.pair_right(operator.multiply(2))(4) == (4, 8)
