## How to update gamla documentation after library update

### If a new docstring was added
1. Go to docs/api.rst and add your function name under the relevant module, with 3 spaces indentation.
For example:

```rest
.. currentmodule:: gamla.functional_generic

.. autosummary::
   old_functions
   .
   .
   .
   my_new_function
```


### If README.md was updated
While in gamla directory:
1. Install .md-to-.rst converter: ``pip install m2r``
1. Convert README.md to README.rst: ``m2r README.md``
1. Move README.rst to docs/source folder instead of existing one: ``mv README.rst docs/source``

### If an existing function was updated
Do nothing. The documentation will update itself.