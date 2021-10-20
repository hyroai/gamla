API
===

apply_utils
-----------

.. currentmodule:: gamla.apply_utils

.. autosummary::
   apply
   apply_async
   apply_fn_with_args
   apply_method
   apply_method_async
   invoke

async_functions
---------------

.. currentmodule:: gamla.async_functions

.. autosummary::
   compose
   compose_left
   map
   star
   thunk
   to_awaitable

currying
--------

.. currentmodule:: gamla.currying

.. autosummary::
   curry

data
----

.. currentmodule:: gamla.data

.. autosummary::
   explode
   freeze_deep
   get_encode_config
   tuple_of_tuples_to_csv

debug_utils
-----------

.. currentmodule:: gamla.debug_utils

.. autosummary::
   debug
   debug_after
   debug_before
   debug_compose
   debug_compose_left
   debug_exception
   log_text
   logger
   profileit

dict_utils
----------

.. currentmodule:: gamla.dict_utils

.. autosummary::
   dict_to_getter_with_default
   get_in
   get_in_or_none
   get_in_or_none_uncurried
   get_in_with_default
   get_or_identity
   get_or_transform
   itemgetter
   itemgetter_or_none
   itemgetter_with_default
   make_index

excepts_decorator
-----------------

.. currentmodule:: gamla.excepts_decorator

.. autosummary::
   excepts
   try_and_excepts

functional
----------

.. currentmodule:: gamla.functional

.. autosummary::
   add_key_value
   assert_that
   assoc_in
   average
   bottom
   concat_with
   dataclass_replace
   dataclass_replace_attribute
   dataclass_transform
   dataclass_transform_attribute
   do_if
   drop
   drop_last
   drop_last_while
   ends_with
   eq_by
   eq_str_ignore_case
   flip
   function_to_uid
   get_all_n_grams
   groupby_many_reduce
   have_intersection
   ignore_input
   interpose
   intersect
   is_instance
   is_iterable
   just_raise
   make_call_key
   make_raise
   partition_after
   partition_all
   partition_before
   pmap
   prefix
   reduce
   remove_key
   sample
   singleize
   skip
   sliding_window
   sort
   sort_by
   sort_by_reversed
   sort_reversed
   suffix
   take
   take_while
   to_json
   top
   translate_exception
   unique
   unique_by
   update_in
   wrap_dict
   wrap_frozenset
   wrap_str
   wrap_tuple

functional_async
----------------

.. currentmodule:: gamla.functional_async

.. autosummary::
   aconcat
   aexcepts
   afirst
   amap_ascompleted
   mapa
   run_sync

functional_generic
------------------

.. currentmodule:: gamla.functional_generic

.. autosummary::
   after
   alljuxt
   allmap
   allstack
   any_is_async
   anyjuxt
   anymap
   anystack
   apply_spec
   before
   bifurcate
   case
   case_dict
   complement
   compose
   compose_left
   compose_many_to_one
   count_by
   count_by_many
   countby_many
   curried_filter
   curried_map
   curried_to_binary
   find
   find_index
   first
   frequencies
   groupby
   itemfilter
   itemmap
   juxt
   juxtcat
   keyfilter
   keymap
   lazyjuxt
   map_dict
   map_filter_empty
   mapcat
   merge
   merge_with
   packstack
   pair_right
   pair_with
   pipe
   reduce_curried
   remove
   scan
   side_effect
   stack
   star
   ternary
   unless
   valfilter
   valmap
   value_to_dict
   when

graph
-----

.. currentmodule:: gamla.graph

.. autosummary::
   cliques_to_graph
   edges_to_graph
   find_sources
   general_graph_traverse_many
   get_connectivity_components
   graph_to_edges
   graph_traverse
   graph_traverse_many
   groupby_many
   has_cycle
   reverse_graph
   traverse_graph_by_radius

graph_async
-----------

.. currentmodule:: gamla.graph_async

.. autosummary::
   agraph_traverse
   agraph_traverse_many
   atraverse_graph_by_radius
   reduce_graph_async

higher_order
------------

.. currentmodule:: gamla.higher_order

.. autosummary::
   on_first
   on_second
   prepare_and_apply
   prepare_and_apply_async

io_utils
--------

.. currentmodule:: gamla.io_utils

.. autosummary::
   batch_calls
   get_async
   head_async_with_headers
   make_throttler
   post_json_async
   post_json_with_extra_headers_and_params_async
   post_json_with_extra_headers_async
   queue_identical_calls
   requests_with_retry
   retry
   throttle
   timeit
   timeout

operator
--------

.. currentmodule:: gamla.operator

.. autosummary::
   add
   attrgetter
   between
   contains
   count
   divide_by
   empty
   equals
   greater_equals
   greater_than
   head
   identity
   inside
   just
   last
   len_equals
   len_greater
   len_smaller
   less_equals
   less_than
   multiply
   nonempty
   not_equals
   nth
   pack
   second
   tail

optimized
---------

.. currentmodule:: gamla.optimized

.. autosummary::

string_utils
------------

.. currentmodule:: gamla.string_utils

.. autosummary::
   capitalize
   regex_match
   replace_in_text
   split_text

sync
----

.. currentmodule:: gamla.sync

.. autosummary::
   after
   alljuxt
   allmap
   anyjuxt
   anymap
   before
   bifurcate
   binary_curry
   case
   case_dict
   check
   complement
   compose
   compose_left
   filter
   groupby
   groupby_many
   juxt
   juxtcat
   juxtduct
   keyfilter
   keymap
   map
   mapcat
   mapdict
   mapduct
   maptuple
   merge
   merge_with_reducer
   packstack
   pair_left
   pipe
   reduce
   remove
   stack
   star
   ternary
   thunk
   valmap
   when

tree
----

.. currentmodule:: gamla.tree

.. autosummary::
   filter_leaves
   get_leaves_by_ancestor_predicate
   json_tree_reduce
   map_reduce_tree
   tree_reduce
   tree_reduce_async

type_safety
-----------

.. currentmodule:: gamla.type_safety

.. autosummary::
   composable
   is_subtype

url_utils
---------

.. currentmodule:: gamla.url_utils

.. autosummary::
   add_to_query_string

Definitions
-----------

.. automodule:: gamla.apply_utils
   :members:

.. automodule:: gamla.async_functions
   :members:

.. automodule:: gamla.currying
   :members:

.. automodule:: gamla.data
   :members:

.. automodule:: gamla.debug_utils
   :members:

.. automodule:: gamla.dict_utils
   :members:

.. automodule:: gamla.excepts_decorator
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

.. automodule:: gamla.higher_order
   :members:

.. automodule:: gamla.io_utils
   :members:

.. automodule:: gamla.operator
   :members:

.. automodule:: gamla.optimized
   :members:

.. automodule:: gamla.string_utils
   :members:

.. automodule:: gamla.sync
   :members:

.. automodule:: gamla.tree
   :members:

.. automodule:: gamla.type_safety
   :members:

.. automodule:: gamla.url_utils
   :members:

