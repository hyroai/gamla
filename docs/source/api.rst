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

currying
--------

.. currentmodule:: gamla.currying

.. autosummary::
   curry

data
----

.. currentmodule:: gamla.data

.. autosummary::
   csv_to_list_of_dicts
   freeze_deep
   get_encode_config
   match
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
   itemgetter
   itemgetter_or_none
   itemgetter_with_default
   make_index

excepts_decorator
-----------------

.. currentmodule:: gamla.excepts_decorator

.. autosummary::
   excepts

functional
----------

.. currentmodule:: gamla.functional

.. autosummary::
   add
   add_key_value
   assert_that
   assoc_in
   attrgetter
   bottom
   concat_with
   contains
   count
   curried_map_sync
   dataclass_replace
   dataclass_transform
   divide_by
   do_if
   drop
   drop_last_while
   ends_with
   eq_by
   eq_str_ignore_case
   equals
   flip
   frequencies
   get_all_n_grams
   greater_equals
   greater_than
   groupby_many_reduce
   head
   identity
   ignore_input
   inside
   interpose
   is_instance
   is_iterable
   just
   just_raise
   last
   len_equals
   len_greater
   len_smaller
   less_equals
   less_than
   make_call_key
   make_raise
   multiply
   not_equals
   nth
   pack
   partition_after
   partition_all
   partition_before
   pmap
   prefix
   profileit
   reduce
   remove_key
   sample
   second
   singleize
   skip
   sliding_window
   sort
   sort_by
   sort_by_reversed
   sort_reversed
   star
   suffix
   tail
   take
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
   anyjuxt
   anymap
   anystack
   apply_spec
   average
   before
   bifurcate
   case
   case_dict
   check
   complement
   compose
   compose_left
   compose_many_to_one
   count_by
   countby_many
   curried_filter
   curried_map
   curried_to_binary
   find
   find_index
   first
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
   side_effect
   stack
   ternary
   to_awaitable
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
   post_json_async
   post_json_with_extra_headers_and_params_async
   post_json_with_extra_headers_async
   queue_identical_calls
   requests_with_retry
   throttle
   timeit
   timeout

string
------

.. currentmodule:: gamla.string

.. autosummary::
   capitalize
   replace_in_text
   split_text

tree
----

.. currentmodule:: gamla.tree

.. autosummary::
   filter_leaves
   get_leaves_by_ancestor_predicate
   json_tree_reduce

url_utils
---------

.. currentmodule:: gamla.url_utils

.. autosummary::
   add_to_query_string

Definitions
-----------

.. automodule:: gamla.apply_utils
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

.. automodule:: gamla.string
   :members:

.. automodule:: gamla.tree
   :members:

.. automodule:: gamla.url_utils
   :members:

