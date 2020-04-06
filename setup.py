import setuptools

with open("README.md", "r") as fh:
    _LONG_DESCRIPTION = fh.read()


setuptools.setup(
    name="gamla",
    version="0.0.63",
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=setuptools.find_namespace_packages(),
    install_requires=[
        "async-timeout",
        "dataclasses_json",
        "heapq_max",
        "pytest-asyncio",
        "pytest",
        "requests",
        "toolz",
        "frozendict",
    ],
)
