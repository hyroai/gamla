API
===

currying
----

.. currentmodule:: gamla.currying

.. autosummary::
   curry

data
----

.. currentmodule:: gamla.data

.. autosummary::
   get_encode_config
   freeze_deep
   dict_to_csv
   csv_to_json
   dict_to_tuple_of_tuples
   tuple_of_tuples_to_csv
   match

functional
----------

.. currentmodule:: gamla.functional

.. autosummary::
   translate_exception
   make_call_key
   top
   bottom
   groupby_many_reduce
   unique_by
   head
   second


functional_async
----------------

.. currentmodule:: gamla.functional_async

.. autosummary::
   run_sync


functional_generic
------------------

.. currentmodule:: gamla.functional_generic

.. autosummary::
   lazyjuxt
   apply_spec
   bifurcate
   countby_many
   groupby
   

graph
-----

.. currentmodule:: gamla.graph

.. autosummary::
   graph_traverse
   graph_traverse_many
   general_graph_traverse_many
   traverse_graph_by_radius
   edges_to_graph
   graph_to_edges
   reverse_graph
   cliques_to_graph
   get_connectivity_components
   groupby_many
   has_cycle


graph_async
-----------
.. currentmodule:: gamla.graph_async

.. autosummary::
   agraph_traverse
   agraph_traverse_many
   agroupby_many
   atraverse_graph_by_radius

io_utils
--------

.. currentmodule:: gamla.io_utils

.. autosummary::
   batch_calls
   get_async
   post_json_async
   post_json_with_extra_headers_async
   queue_identical_calls
   requests_with_retry
   throttle
   timeit
   timeout

transducer
--------

.. currentmodule:: gamla.transducer

.. autosummary::
   apply_spec
   concat
   count_by
   filter
   groupby
   groupby_many
   juxt
   map
   mapcat
   transduce

tree
----

.. currentmodule:: gamla.tree

.. autosummary::
   filter_leaves
   get_leaves_by_ancestor_predicate


Definitions
-----------

.. automodule:: gamla.__init__
   :members:

.. automodule:: gamla.currying
   :members:

.. automodule:: gamla.data
   :members:

.. automodule:: gamla.functional
   :members:

.. automodule:: gamla.functional_async
   :members:

.. automodule:: gamla.functional_generic
   :members:

.. automodule:: gamla.graph
   :members:

.. automodule:: gamla.graph_async
   :members:

.. automodule:: gamla.io_utils
   :members:

.. automodule:: gamla.transducer
   :members:

.. automodule:: gamla.tree
   :members:
