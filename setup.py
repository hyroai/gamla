import setuptools

with open("README.md", "r") as fh:
    _LONG_DESCRIPTION = fh.read()


setuptools.setup(
    name="gamla",
    version="7",
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=setuptools.find_namespace_packages(),
    install_requires=[
        "async-timeout",
        "dataclasses_json",
        "frozendict",
        "heapq_max",
        "httpx",
        "pytest-asyncio",
        "pytest>=5.4.0",
        "requests",
        "toolz",
    ],
)
