[![Build Status](https://travis-ci.com/hyroai/gamla.svg?branch=master)](https://travis-ci.com/hyroai/gamla)

Gamla is a functional programming library for python.

`pip install gamla`

## Releasing a new  version

1. Create a pypi account.
1. Download twine and give it your pypi credentials.
1. Get pypi permissions for the project from its owner.
1. `python setup.py sdist bdist_wheel; twine upload dist/*; rm -rf dist;`
