from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast, no_type_check

import awkward as ak
import dask.config
from awkward.typetracer import touch_data
from dask.base import tokenize
from dask.blockwise import Blockwise, fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.layers import AwkwardBlockwiseLayer, AwkwardInputLayer, _dask_uses_tasks
from dask_awkward.lib.utils import _buf_to_col, commit_to_reports, typetracer_nochecks
from dask_awkward.utils import first

if _dask_uses_tasks:
    from dask.blockwise import GraphNode, Task, TaskRef

if TYPE_CHECKING:
    from dask.typing import Key

log = logging.getLogger(__name__)

COLUMN_OPT_FAILED_WARNING_MSG = """The necessary columns optimization failed; exception raised:

{exception} with message {message}.

Please see the FAQ section of the docs for more information:
https://dask-awkward.readthedocs.io/en/stable/more/faq.html

"""


def all_optimizations(dsk: Mapping, keys: Sequence[Key], **_: Any) -> Mapping:
    """Run all optimizations that benefit dask-awkward computations.

    This function will run both dask-awkward specific and upstream
    general optimizations from core dask.

    """
    keys = tuple(flatten(keys))

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(str(id(dsk)), dsk, dependencies=())

    # Perform dask-awkward specific optimizations.
    with typetracer_nochecks():
        dsk = optimize(dsk, keys=keys)
    # Perform Blockwise optimizations for HLG input
    dsk = optimize_blockwise(dsk, keys=keys)
    # fuse nearby layers
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore
    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore

    return dsk


def optimize(dsk: HighLevelGraph, keys: Sequence[Key], **_: Any) -> Mapping:
    """Run optimizations specific to dask-awkward.

    - determine the necessary columns for input layers
    - fuse linear chains of blockwise operations in linear time

    """
    if dask.config.get("awkward.optimization.enabled"):
        which = dask.config.get("awkward.optimization.which")
        if "columns" in which:
            dsk = optimize_columns(dsk, keys)
        if "layer-chains" in which:
            dsk = rewrite_layer_chains(dsk, keys)

    return dsk


def optimize_columns(
    dsk: HighLevelGraph, keys: Sequence[Key], dryrun=False
) -> HighLevelGraph:
    """Run column projection optimization.

    This optimization determines which columns from an
    ``AwkwardInputLayer`` are necessary for a complete computation.

    For example, if a parquet dataset is loaded with fields:
    ``["foo", "bar", "baz.x", "baz.y"]``

    And the following task graph is made:

    >>> ds = dak.from_parquet("/path/to/dataset")
    >>> z = ds["foo"] - ds["baz"]["y"]

    Upon calling z.compute() the AwkwardInputLayer created in the
    from_parquet call will only read the parquet columns ``foo`` and
    ``baz.y``.

    Parameters
    ----------
    dsk : HighLevelGraph
        Task graph to optimize.

    Returns
    -------
    HighLevelGraph
        New, optimized task graph with column-projected ``AwkwardInputLayer``.

    """
    dsk2 = dsk.layers.copy()

    lays = {_[0] for _ in keys if isinstance(_, tuple)}
    all_reps = set()
    for ln in lays:
        if ln in dsk.layers and hasattr(dsk.layers[ln], "meta"):
            m = dsk.layers[ln].meta
            if not isinstance(m, ak._nplikes.typetracer.MaybeNone):
                # maybenone cases should already have been all touched
                # but we could extract the .content here
                touch_data(m)
            rep = getattr(dsk.layers[ln].meta, "_report", ())
            if rep:
                all_reps.update(rep)
    name = tokenize("output", lays)
    commit_to_reports(name, all_reps)
    all_layers = tuple(dsk.layers) + (name,)

    if dryrun:
        out = {}
    for k, lay, cols in _optimize_columns(dsk.layers, all_layers):
        if dryrun:
            out[k] = cols
        else:
            new_lay = lay.project(cols)
            dsk2[k] = new_lay
    if dryrun:
        return out
    return HighLevelGraph(dsk2, dsk.dependencies)


def _optimize_columns(dsk, all_layers):
    for k, lay in dsk.copy().items():
        if not isinstance(lay, AwkwardInputLayer) or not hasattr(lay, "meta"):
            continue
        rep = getattr(lay.meta, "_report", None)
        if not rep:
            continue
        rep = first(rep)  # each meta of an IO layer should have just one report
        cols = rep.data_touched_in(all_layers)
        if cols:
            yield k, lay, cols


def necessary_columns(*args, normalize: bool = True, trim: bool = True):
    """Find the columns in each input layer that are needed by given collections

    Parameters
    ----------
    args: dask-awkward colections or other dask objects baseed on them
    normalize: if True, will transform the internal buffer-oriented representation
        to column names similar to the convention used for instance by parquet. The
        raw representation is the one actually passed to the IO backends during
        optimization, and includes information about which component of a field
        is needed (data, offsets, index, etc.)
    trim: if normalize is True, setting this True will remove parent columns

    Returns
    -------
    dict: the keys are the dask names of IO layers contained in the combined graph,
        and for each there is a set of required columns
    """
    dsk = {}
    keys = []
    for arg in args:
        dsk.update(arg.dask.layers)
        keys.append((arg.name, 0))
    hlg = HighLevelGraph(dsk, {})
    out = optimize_columns(hlg, keys, dryrun=True)
    if normalize:
        for k in list(out):
            # `startswith` to clobber attributes of unnamed root field
            col1 = {_buf_to_col(_) for _ in out[k] if _.startswith("@.")}
            if trim:
                parents = {_.rsplit(".", 1)[0] for _ in col1 if "." in _}
                out[k] = {_ for _ in col1 if _ not in parents}
            else:
                out[k] = col1
        # TODO: remove columns included in children?
    return out


@no_type_check
def rewrite_layer_chains(dsk: HighLevelGraph, keys: Sequence[Key]) -> HighLevelGraph:
    """Smush chains of blockwise layers into a single layer.

    The logic here identifies chains by popping layers (in arbitrary
    order) from a set of all layers in the task graph and walking
    through the dependencies (parent layers) and dependents (child
    layers). If a multi layer chain is discovered we compress it into
    a single layer with the second loop below (for chain in chains;
    that step rewrites the graph). In the chain building logic, if a
    layer exists in the `keys` argument (the keys necessary for the
    compute that we are optimizing for), we shortcircuit the logic to
    ensure we do not chain layers that contain a necessary key inside
    (these layers are called `required_layers` below).

    Parameters
    ----------
    dsk : HighLevelGraph
        Task graph to optimize.
    keys : Any
        Keys that are requested by the compute that is being
        optimized.

    Returns
    -------
    HighLevelGraph
        New, optimized task graph.

    """
    # dask.optimization.fuse_liner for blockwise layers
    import copy

    chains = []
    deps = copy.copy(dsk.dependencies)

    required_layers = {k[0] for k in keys if isinstance(k, tuple)}
    layers = {}
    # find chains; each chain list is at least two keys long
    dependents = dsk.dependents
    all_layers = set(dsk.layers)
    while all_layers:
        layer_key = all_layers.pop()
        layer = dsk.layers[layer_key]
        if not isinstance(layer, AwkwardBlockwiseLayer):
            # shortcut to avoid making comparisons
            layers[layer_key] = layer  # passthrough unchanged
            continue
        children = dependents[layer_key]
        chain = [layer_key]
        current_layer_key = layer_key
        while (
            len(children) == 1
            and dsk.dependencies[first(children)] == {current_layer_key}
            and isinstance(dsk.layers[first(children)], AwkwardBlockwiseLayer)
            and len(dsk.layers[current_layer_key])
            == len(dsk.layers[first(children)])  # SLOW?!
            and current_layer_key not in required_layers
        ):
            # walk forwards
            current_layer_key = first(children)
            chain.append(current_layer_key)
            all_layers.remove(current_layer_key)
            children = dependents[current_layer_key]

        parents = dsk.dependencies[layer_key]
        while (
            len(parents) == 1
            and dependents[first(parents)] == {layer_key}
            and isinstance(dsk.layers[first(parents)], AwkwardBlockwiseLayer)
            and len(dsk.layers[layer_key]) == len(dsk.layers[first(parents)])
            and next(iter(parents)) not in required_layers
        ):
            # walk backwards
            layer_key = first(parents)
            chain.insert(0, layer_key)
            all_layers.remove(layer_key)
            parents = dsk.dependencies[layer_key]
        if len(chain) > 1:
            chains.append(chain)
            layers[chain[-1]] = copy.copy(
                dsk.layers[chain[-1]]
            )  # shallow copy to be mutated
        else:
            layers[layer_key] = layer  # passthrough unchanged

    # do rewrite
    for chain in chains:
        # inputs are the inputs of chain[0]
        # outputs are the outputs of chain[-1]
        # .dsk is composed of the .dsk of each layer
        outkey = chain[-1]
        layer0 = cast(Blockwise, dsk.layers[chain[0]])
        outlayer = layers[outkey]
        numblocks = [nb[0] for nb in layer0.numblocks.values() if nb[0] is not None][0]
        deps[outkey] = deps[chain[0]]
        [deps.pop(ch) for ch in chain[:-1]]

        if _dask_uses_tasks:
            all_tasks = [layer0.task]
        else:
            subgraph = layer0.dsk.copy()
        indices = list(layer0.indices)
        parent = chain[0]

        outlayer.io_deps = layer0.io_deps  # mypy: ignore
        for chain_member in chain[1:]:
            layer = dsk.layers[chain_member]
            for k in layer.io_deps:  # mypy: ignore
                outlayer.io_deps[k] = layer.io_deps[k]

            if _dask_uses_tasks:
                func = layer.task.func
                args = [
                    arg.key if isinstance(arg, GraphNode) else arg
                    for arg in layer.task.args
                ]
                # how to do this with `.substitute(...)`?
                args2 = _recursive_replace(args, layer, parent, indices)
                all_tasks.append(Task(chain_member, func, *args2))
            else:
                func, *args = layer.dsk[chain_member]  # mypy: ignore
                args2 = _recursive_replace(args, layer, parent, indices)
                subgraph[chain_member] = (func,) + tuple(args2)
            parent = chain_member
        outlayer.numblocks = {
            i[0]: (numblocks,) for i in indices if i[1] is not None
        }  # mypy: ignore
        if _dask_uses_tasks:
            outlayer.task = Task.fuse(*all_tasks)
        else:
            outlayer.dsk = subgraph  # mypy: ignore
        if hasattr(outlayer, "_dims"):
            del outlayer._dims
        outlayer.indices = tuple(  # mypy: ignore
            (i[0], (".0",) if i[1] is not None else None) for i in indices
        )
        outlayer.output_indices = (".0",)  # mypy: ignore
        outlayer.inputs = getattr(layer0, "inputs", set())  # mypy: ignore
        if hasattr(outlayer, "_cached_dict"):
            del outlayer._cached_dict  # reset, since original can be mutated
    return HighLevelGraph(layers, deps)


def _recursive_replace(args, layer, parent, indices):
    args2 = []
    for arg in args:
        if isinstance(arg, str) and arg.startswith("__dask_blockwise__"):
            ind = int(arg[18:])
            if layer.indices[ind][1] is None:
                # this is a simple arg
                args2.append(layer.indices[ind][0])
            elif layer.indices[ind][0] == parent:
                # arg refers to output of previous layer
                if _dask_uses_tasks:
                    args2.append(TaskRef(parent))
                else:
                    args2.append(parent)
            else:
                # arg refers to things defined in io_deps
                indices.append(layer.indices[ind])
                arg2 = f"__dask_blockwise__{len(indices) - 1}"
                if _dask_uses_tasks:
                    args2.append(TaskRef(arg2))
                else:
                    args2.append(arg2)
        elif isinstance(arg, list):
            args2.append(_recursive_replace(arg, layer, parent, indices))
        elif isinstance(arg, tuple):
            args2.append(tuple(_recursive_replace(arg, layer, parent, indices)))
        # elif isinstance(arg, dict):
        else:
            args2.append(arg)
    return args2
