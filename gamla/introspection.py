import ast
import copy
import logging
from typing import Callable, Dict, Text


def _convert_expression(expr) -> ast.Expression:
    expr.lineno = 0
    expr.col_offset = 0
    return ast.Expression(expr.value, lineno=0, col_offset=0)


# https://stackoverflow.com/a/52361938/378594
def _exec_with_return(code: Text, globals_dict: Dict):
    code_ast = ast.parse(code)

    init_ast = copy.deepcopy(code_ast)
    init_ast.body = code_ast.body[:-1]

    last_ast = copy.deepcopy(code_ast)
    last_ast.body = code_ast.body[-1:]

    logging.warning(
        "Giving true names for composition (this will slow things considerably)",
    )

    exec(compile(init_ast, "<ast>", "exec"), globals_dict)
    if type(last_ast.body[0]) == ast.Expr:
        return eval(
            compile(
                _convert_expression(last_ast.body[0]),
                "<ast>",
                "eval",
            ),
            globals_dict,
        )
    exec(compile(last_ast, "<ast>", "exec"), globals_dict)


def _clean_name_for_function(name: Text):
    # Lambdas appear as <lambda> which is not valid python function name.
    name = name.replace("<", "").replace(">", "")
    if name == "lambda":
        return "some_lambda"
    return name


def rename_async_function(name: Text, f: Callable) -> Callable:
    name = _clean_name_for_function(name)
    return _exec_with_return(
        f"import functools\n@functools.wraps(f)\nasync def {name}(*args, **kwargs): return await f(*args, **kwargs)\n{name}",
        {"f": f},
    )


def rename_function(name: Text, f: Callable) -> Callable:
    name = _clean_name_for_function(name)
    return _exec_with_return(
        f"import functools\n@functools.wraps(f)\ndef {name}(*args, **kwargs): return f(*args, **kwargs)\n{name}",
        {"f": f},
    )
