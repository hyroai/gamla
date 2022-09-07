import setuptools

with open("README.md", "r") as fh:
    _LONG_DESCRIPTION = fh.read()


setuptools.setup(
    name="gamla",
    version="135",
    python_requires=">=3.9.5",
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=setuptools.find_namespace_packages(),
    install_requires=[
        "async-timeout",
        "dataclasses_json",
        "heapq_max",
        "httpx",
        "immutables",
        "pytest-asyncio",
        "pytest",
        "requests",
        "tabulate",
        "termcolor",
        "toolz",
        "yappi",
    ],
)
