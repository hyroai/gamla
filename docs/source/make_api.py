import gamla
from gamla import *


def get_modules():
    return [
        currying,
        data,
        dict_utils,
        excepts_decorator,
        functional,
        functional_async,
        functional_generic,
        graph,
        graph_async,
        higher_order,
        io_utils,
        string,
        tree,
        url_utils,
    ]
    # TODO(itay): find a way to get all modules in gamla automatically. this line below can only get filenames
    # return [importlib.import_module(module) for module in os.listdir('/Users/itayzit/projects/gamla/gamla') if module.endswith('.py') and 'test' not in module]


def _get_function_table_entries(module) -> Text:
    return "".join(
        [
            f"   {o[0]}\n"
            for o in inspect.getmembers(module)
            if inspect.isfunction(o[1]) and o[0][0] != "_"
        ],
    )


def _concat_module_table_string(string_so_far: Text, module) -> Text:
    return ''.join(gamla.concat_with(f"{module.__name__[6:]}\n{len(module.__name__[6:]) * '-'}\n\n.. currentmodule:: {module.__name__}\n\n.. autosummary::\n{_get_function_table_entries(module)}\n", string_so_far))


def _concat_module_members_string(string_so_far: Text, module) -> Text:
    return ''.join(gamla.concat_with(f".. automodule:: {module.__name__}\n   :members:\n\n", string_so_far))


def create_api_string(modules: Iterable) -> Text:
    return gamla.reduce(
        _concat_module_members_string,
        gamla.reduce(_concat_module_table_string, "API\n===\n\n", modules)
        + "Definitions\n-----------\n\n",
        modules,
    )


print(create_api_string(get_modules()))
# new_api = open("api.rst", "w")
# new_api.write(create_api_string(get_modules))
# new_api.close()
