import collections
import itertools
from typing import Callable, Iterable, Sequence, Union

concat = itertools.chain.from_iterable


def identity(x):
    return x


def nth(n):
    """Returns the nth element in a sequence.

    >>> nth(1, 'ABC')
    ['B']
    """

    def nth(seq):
        if isinstance(seq, (tuple, list, Sequence)):
            return seq[n]
        return next(itertools.islice(seq, n, None))

    return nth


def count(seq: Iterable) -> int:
    """Counts the number of items in `seq`. Similar to `len` but works on lazy sequences."""
    if hasattr(seq, "__len__"):
        return len(seq)  # type: ignore
    return sum(1 for _ in seq)


def attrgetter(attr):
    """Access the object attribute by its name `attr`.

    >>> attrgetter("lower")("ASD")()
    'asd'
    """

    def attrgetter(obj):
        return getattr(obj, attr)

    return attrgetter


def equals(x):
    def equals(y):
        return x == y

    return equals


def not_equals(x):
    """A functional !=.

    >>> not_equals(2)(2)
    False

    >>> not_equals("David")("Michael")
    True
    """

    def not_equals(y):
        return x != y

    return not_equals


def contains(x):
    """Contains operator.

    >>> contains([1, 2, 3])(2)
    True

    >>> contains("David")("x")
    False
    """

    def contains(y):
        return y in x

    return contains


def add(x):
    """Addition operator.

    >>> add(1)(2)
    3

    >>> add(["c"])(["a", "b"])
    ['a', 'b', 'c']
    """

    def add(y):
        return y + x

    return add


def greater_than(x):
    """Greater than operator.

    >>> greater_than(1)(2)
    True

    >>> greater_than(1)(0)
    False
    """

    def greater_than(y):
        return y > x

    return greater_than


def greater_equals(x):
    """Greater than or equal operator.

    >>> greater_equals(1)(1)
    True

    >>> greater_equals(1)(0)
    False
    """

    def greater_equals(y):
        return y >= x

    return greater_equals


def less_than(x):
    """Less than operator.

    >>> less_than(1)(1)
    False
    """

    def less_than(y):
        return y < x

    return less_than


def less_equals(x):
    """Less than or equal operator.

    >>> less_equals(1)(1)
    True

    >>> less_equals(1)(3)
    False
    """

    def less_equals(y):
        return y <= x

    return less_equals


def multiply(x):
    """Multiply operator.

    >>> multiply(2)(1)
    2
    """

    def multiply(y):
        return y * x

    return multiply


def divide_by(x):
    def divide_by(y):
        return y / x

    return divide_by


def inside(val):
    """A functional `in` operator.

    >>> inside(1)([0, 1, 2])
    True

    >>> inside("a", "word")
    False
    """

    def inside(container):
        return val in container

    return inside


def len_equals(length: int):
    """Measures if the length of a sequence equals to a given length.

    >>> len_equals(3)([0, 1, 2])
    True
    """

    def len_equals(x: Iterable) -> bool:
        return count(x) == length

    return len_equals


def len_greater(length: int):
    """Measures if the length of a sequence is greater than a given length.

    >>> len_greater(2)([0, 1, 2])
    True
    """

    def len_greater(seq):
        return count(seq) > length

    return len_greater


def len_smaller(length: int) -> Callable:
    """Measures if the length of a sequence is smaller than a given length.

    >>> len_smaller(2)([0, 1, 2])
    False
    """

    def len_smaller(seq):
        return count(seq) < length

    return len_smaller


def between(low: int, high: int):
    def between(number: Union[int, float]):
        return low <= number < high

    return between


def empty(seq):
    try:
        next(iter(seq))
    except StopIteration:
        return True
    return False


def nonempty(seq):
    return not empty(seq)


def head(seq):
    """Returns the first element in a sequence.
    >>> first('ABC')
    'A'
    """
    return next(iter(seq))


def second(seq):
    """Returns the second element in a sequence.
    >>> second('ABC')
    'B'
    """
    seq = iter(seq)
    next(seq)
    return next(seq)


def tail(n):
    """Returns the last n elements of a sequence.
    >>> tail(2, [10, 20, 30, 40, 50])
    [40, 50]
    """

    def tail(seq):
        try:
            return seq[-n:]
        except (TypeError, KeyError):
            return tuple(collections.deque(seq, n))

    return tail


def last(seq):
    """Returns the last element in a sequence
    >>> last('ABC')
    'C'
    """
    return tail(1)(seq)[0]


def pack(*stuff):
    """Returns a list generated from the provided input.

    >>> pack(1, 2, 3)
    (1, 2, 3)
    """
    return stuff


def is_instance(the_type):
    """Returns if `the_value` is an instance of `the_type`.

    >>> is_instance(str)("hello")
    True

    >>> is_instance(int)("a")
    False
    """

    def is_instance(the_value):
        return isinstance(the_value, the_type)

    return is_instance


def is_iterable(x):
    """Determines whether the element is iterable.

    >>> isiterable([1, 2, 3])
    True
    >>> isiterable('abc')
    True
    >>> isiterable(5)
    False"""
    try:
        iter(x)
        return True
    except TypeError:
        return False
