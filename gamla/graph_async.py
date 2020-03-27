import itertools

import graph
import toolz

from gamla import functional, functional_async


@toolz.curry
async def agroupby_many(f, it):
    return await functional_async.apipe(
        it,
        functional_async.amap(
            functional_async.acompose_left(
                functional_async.apair_with(f),
                functional.star(lambda x, y: (x, [y])),
                functional.star(itertools.product),
            )
        ),
        toolz.concat,
        graph.edges_to_graph,
    )
