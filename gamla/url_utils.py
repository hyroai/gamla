from typing import Dict, Text
from urllib import parse

from gamla import currying
from gamla.optimized import sync


@currying.curry
def add_to_query_string(params_to_add: Dict, url: Text) -> Text:
    """Add params_to_add to the query string part of url

    >>> add_to_query_string({ "param1" : "value"}, "http://domain.com")
    http://domain.com?param1=value
    """
    (scheme, netloc, path, query, fragment) = parse.urlsplit(url)
    return parse.urlunsplit(
        (
            scheme,
            netloc,
            path,
            parse.urlencode(
                sync.merge([parse.parse_qs(query), params_to_add]),
                doseq=True,
            ),
            fragment,
        ),
    )
