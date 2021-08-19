import re
from typing import Text


def replace_in_text(old: Text, new: Text):
    """Return a copy of the string with all occurrences of substring old replaced by new
    >>> txt = "hello world"
    >>> replace_in_text("world", "Jhon")(txt)
    'hello Jhon'
    """

    def replace_in_text(txt: Text):
        return txt.replace(old, new)

    return replace_in_text


def split_text(sep: Text):
    """Return a list of the words in the string, using sep as the delimiter string

    >>> txt = "hello world"
    >>> split_text(" ")(txt)
    ['hello', 'world']
    """

    def split_text(txt: Text):
        return txt.split(sep)

    return split_text


def capitalize(text: Text):
    """Capitalize only the first letter of a given string, unlike
    str.capitalize that transforms all characters after the first
    character to lowercase.

    >>>capitalize("my name is John Smith")
    'My name is John Smith'
    """
    return text[0].upper() + text[1:]


def regex_match(pattern):
    """A curried regex match. Gets a pattern and returns a function that expects a text to match the pattern with.

    >>> regex_match(r"phone:(\d*)")("phone:1234567").group(1)  # noqa: W605
    '1234567'

    """

    def regex_match_inner(text):
        return re.match(pattern, text)

    return regex_match_inner
