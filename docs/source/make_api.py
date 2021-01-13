"""Script for updating api.rst in accordance with changes in the repository"""
import importlib
import inspect
import sys
from typing import Any, Text, Tuple

import gamla

REPO_NAME = sys.argv[1]
repo = importlib.import_module(REPO_NAME)


def _module_filter(module):
    return (
        inspect.ismodule(module)
        and REPO_NAME in str(module)
        and "test" not in str(module)
    )


def get_modules() -> Tuple[Tuple[Text, Any], ...]:
    return tuple(inspect.getmembers(repo, _module_filter))


def _get_function_table_entries(module: Tuple[Text, Any]) -> Text:
    return "".join(
        [
            f"   {o[0]}\n"
            for o in inspect.getmembers(module)
            if inspect.isfunction(o[1]) and o[0][0] != "_"
        ],
    )


def _concat_module_table_string(string_so_far: Text, module: Tuple[Text, Any]) -> Text:
    return "".join(
        gamla.concat_with(
            f"{module[0]}\n{len(module[0]) * '-'}\n\n.. currentmodule:: {REPO_NAME}.{module[0]}\n\n.. autosummary::\n{_get_function_table_entries(module[1])}\n",
            string_so_far,
        ),
    )


def _concat_module_members_string(
    string_so_far: Text,
    module: Tuple[Text, Any],
) -> Text:
    return "".join(
        gamla.concat_with(
            f".. automodule:: {REPO_NAME}.{module[0]}\n   :members:\n\n",
            string_so_far,
        ),
    )


def create_api_string(modules: Tuple[Tuple[Text, Any], ...]) -> Text:
    return gamla.reduce(
        _concat_module_members_string,
        gamla.reduce(_concat_module_table_string, "API\n===\n\n", modules)
        + "Definitions\n-----------\n\n",
        modules,
    )


new_api = open("api.rst", "w")
new_api.write(create_api_string(get_modules()))
new_api.close()
