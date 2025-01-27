from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

import dask
import toolz
from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import MaterializedLayer
from dask.layers import Layer

from dask_awkward.utils import LazyInputsDict


_dask_uses_tasks = hasattr(dask.blockwise, "Task")

if _dask_uses_tasks:
    from dask.blockwise import Task, TaskRef


class AwkwardBlockwiseLayer(Blockwise):
    """Just like upstream Blockwise, except we override pickling"""

    has_been_unpickled: bool = False

    @classmethod
    def from_blockwise(cls, layer: Blockwise) -> AwkwardBlockwiseLayer:
        ob = object.__new__(cls)
        ob.__dict__.update(layer.__dict__)
        return ob

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Indicator that this layer has been serialised
        state["has_been_unpickled"] = True
        state.pop("meta", None)  # this is a typetracer
        return state

    def __repr__(self) -> str:
        return "Awkward" + super().__repr__()


class ImplementsIOFunction(Protocol):
    def __call__(self, *args, **kwargs): ...


T = TypeVar("T")


class ImplementsReport(ImplementsIOFunction, Protocol):
    @property
    def return_report(self) -> bool: ...


def io_func_implements_projection(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "project")


def io_func_implements_columnar(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "necessary_columns")


def io_func_implements_report(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "return_report")


class AwkwardTokenizable:

    def __init__(self, ret_val, parent_name):
        self.parent_name = parent_name
        self.ret_val = ret_val

    def __dask_tokenize__(self):
        return ("AwkwardTokenizable", self.parent_name)

    def __call__(self, *_, **__):
        return self.ret_val


class AwkwardInputLayer(AwkwardBlockwiseLayer):
    """A layer known to perform IO and produce Awkward arrays

    We specialise this so that we have a way to prune column selection on load
    """

    def __init__(
        self,
        *,
        name: str,
        inputs: Any,
        io_func: ImplementsIOFunction,
        label: str | None = None,
        produces_tasks: bool = False,
        creation_info: dict | None = None,
        annotations: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.inputs = inputs
        self.io_func = io_func
        self.label = label
        self.produces_tasks = produces_tasks
        self.annotations = annotations
        self.creation_info = creation_info

        io_arg_map = BlockwiseDepDict(
            mapping=LazyInputsDict(self.inputs),  # type: ignore
            produces_tasks=self.produces_tasks,
        )

        super_kwargs: dict[str, Any] = {
            "output": self.name,
            "output_indices": "i",
            "indices": [(io_arg_map, "i")],
            "numblocks": {},
            "annotations": None,
        }

        if _dask_uses_tasks:
            super_kwargs["task"] = Task(name, self.io_func, TaskRef(blockwise_token(0)))
        else:
            super_kwargs["dsk"] = {name: (self.io_func, blockwise_token(0))}

        super().__init__(**super_kwargs)

    def __repr__(self) -> str:
        return f"AwkwardInputLayer<{self.output}>"

    @property
    def is_columnar(self) -> bool:
        return io_func_implements_columnar(self.io_func)

    def project(self, columns: list[str]) -> AwkwardInputLayer:
        if hasattr(self.io_func, "project"):
            io_func = self.io_func.project(columns)
        else:
            return self

        return AwkwardInputLayer(
            name=self.name,
            inputs=self.inputs,
            io_func=io_func,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )


class AwkwardMaterializedLayer(MaterializedLayer):
    def __init__(
        self,
        mapping: dict,
        *,
        previous_layer_names: list[str],
        fn: Callable | None = None,
        **kwargs: Any,
    ):
        self.previous_layer_names: list[str] = previous_layer_names
        self.fn = fn
        super().__init__(mapping, **kwargs)


class AwkwardTreeReductionLayer(Layer):
    """Awkward Tree-Reduction Layer
    Parameters
    ----------
    name : str
        Name to use for the constructed layer.
    name_input : str
        Name of the input layer that is being reduced.
    npartitions_input : str
        Number of partitions in the input layer.
    concat_func : callable
        Function used by each tree node to reduce a list of inputs
        into a single output value. This function must accept only
        a list as its first positional argument.
    tree_node_func : callable
        Function used on the output of ``concat_func`` in each tree
        node. This function must accept the output of ``concat_func``
        as its first positional argument.
    finalize_func : callable, optional
        Function used in place of ``tree_node_func`` on the final tree
        node(s) to produce the final output for each split. By default,
        ``tree_node_func`` will be used.
    split_every : int, optional
        This argument specifies the maximum number of input nodes
        to be handled by any one task in the tree. Defaults to 32.
    split_out : int, optional
        This argument specifies the number of output nodes in the
        reduction tree. If ``split_out`` is set to an integer >=1, the
        input tasks must contain data that can be indexed by a ``getitem``
        operation with a key in the range ``[0, split_out)``.
    output_partitions : list, optional
        List of required output partitions. This parameter is used
        internally by Dask for high-level culling.
    tree_node_name : str, optional
        Name to use for intermediate tree-node tasks.
    """

    name: str
    name_input: str
    npartitions_input: int
    concat_func: Callable
    tree_node_func: Callable
    finalize_func: Callable | None
    split_every: int
    split_out: int
    output_partitions: list[int]
    tree_node_name: str
    widths: list[int]
    height: int

    def __init__(
        self,
        name: str,
        name_input: str,
        npartitions_input: int,
        concat_func: Callable,
        tree_node_func: Callable,
        finalize_func: Callable | None = None,
        split_every: int = 32,
        split_out: int | None = None,
        output_partitions: list[int] | None = None,
        tree_node_name: str | None = None,
        annotations: dict[str, Any] | None = None,
    ):
        super().__init__(annotations=annotations)
        self.name = name
        self.name_input = name_input
        self.npartitions_input = npartitions_input
        self.concat_func = concat_func
        self.tree_node_func = tree_node_func
        self.finalize_func = finalize_func
        self.split_every = split_every
        self.split_out = split_out  # type: ignore
        self.output_partitions = (
            list(range(self.split_out or 1))
            if output_partitions is None
            else output_partitions
        )
        self.tree_node_name = tree_node_name or "tree_node-" + self.name

        # Calculate tree widths and height
        # (Used to get output keys without materializing)
        parts = self.npartitions_input
        self.widths = [parts]
        while parts > 1:
            parts = math.ceil(parts / self.split_every)
            self.widths.append(int(parts))
        self.height = len(self.widths)

    def _make_key(self, *name_parts, split=0):
        # Helper function construct a key
        # with a "split" element when
        # bool(split_out) is True
        return name_parts + (split,) if self.split_out else name_parts

    def _define_task(self, input_keys, final_task=False):
        # Define nested concatenation and func task
        if final_task and self.finalize_func:
            outer_func = self.finalize_func
        else:
            outer_func = self.tree_node_func
        return (toolz.pipe, input_keys, self.concat_func, outer_func)

    def _construct_graph(self):
        """Construct graph for a tree reduction."""

        dsk = {}
        if not self.output_partitions:
            return dsk

        # Deal with `bool(split_out) == True`.
        # These cases require that the input tasks
        # return a type that enables getitem operation
        # with indices: [0, split_out)
        # Therefore, we must add "getitem" tasks to
        # select the appropriate element for each split
        name_input_use = self.name_input
        if self.split_out:
            name_input_use += "-split"
            for s in self.output_partitions:
                for p in range(self.npartitions_input):
                    dsk[self._make_key(name_input_use, p, split=s)] = (
                        operator.getitem,
                        (self.name_input, p),
                        s,
                    )

        if self.height >= 2:
            # Loop over output splits
            for s in self.output_partitions:
                # Loop over reduction levels
                for depth in range(1, self.height):
                    # Loop over reduction groups
                    for group in range(self.widths[depth]):
                        # Calculate inputs for the current group
                        p_max = self.widths[depth - 1]
                        lstart = self.split_every * group
                        lstop = min(lstart + self.split_every, p_max)
                        if depth == 1:
                            # Input nodes are from input layer
                            input_keys = [
                                self._make_key(name_input_use, p, split=s)
                                for p in range(lstart, lstop)
                            ]
                        else:
                            # Input nodes are tree-reduction nodes
                            input_keys = [
                                self._make_key(
                                    self.tree_node_name, p, depth - 1, split=s
                                )
                                for p in range(lstart, lstop)
                            ]

                        # Define task
                        if depth == self.height - 1:
                            # Final Node (Use fused `self.tree_finalize` task)
                            assert (
                                group == 0
                            ), f"group = {group}, not 0 for final tree reduction task"
                            dsk[(self.name, s)] = self._define_task(
                                input_keys, final_task=True
                            )
                        else:
                            # Intermediate Node
                            dsk[
                                self._make_key(
                                    self.tree_node_name, group, depth, split=s
                                )
                            ] = self._define_task(input_keys, final_task=False)
        else:
            # Deal with single-partition case
            for s in self.output_partitions:
                input_keys = [self._make_key(name_input_use, 0, split=s)]
                dsk[(self.name, s)] = self._define_task(input_keys, final_task=True)

        return dsk

    def __repr__(self):
        return "DataFrameTreeReduction<name='{}', input_name={}, split_out={}>".format(
            self.name, self.name_input, self.split_out
        )

    def _output_keys(self):
        return {(self.name, s) for s in self.output_partitions}

    def get_output_keys(self):
        if hasattr(self, "_cached_output_keys"):
            return self._cached_output_keys
        else:
            output_keys = self._output_keys()
            self._cached_output_keys = output_keys
        return self._cached_output_keys

    def is_materialized(self):
        return hasattr(self, "_cached_dict")

    @property
    def _dict(self):
        """Materialize full dict representation"""
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            dsk = self._construct_graph()
            self._cached_dict = dsk
        return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        # Start with "base" tree-reduction size
        tree_size = (sum(self.widths[1:]) or 1) * (self.split_out or 1)
        if self.split_out:
            # Add on "split-*" tasks used for `getitem` ops
            return tree_size + self.npartitions_input * len(self.output_partitions)
        return tree_size

    def _keys_to_output_partitions(self, keys):
        """Simple utility to convert keys to output partition indices."""
        splits = set()
        for key in keys:
            try:
                _name, _split = key
            except ValueError:
                continue
            if _name != self.name:
                continue
            splits.add(_split)
        return splits

    def _cull(self, output_partitions):
        return AwkwardTreeReductionLayer(
            self.name,
            self.name_input,
            self.npartitions_input,
            self.concat_func,
            self.tree_node_func,
            finalize_func=self.finalize_func,
            split_every=self.split_every,
            split_out=self.split_out,
            output_partitions=output_partitions,
            tree_node_name=self.tree_node_name,
            annotations=self.annotations,
        )

    def cull(self, keys, all_keys):
        """Cull a DataFrameTreeReduction HighLevelGraph layer"""
        deps = {
            (self.name, 0): {
                (self.name_input, i) for i in range(self.npartitions_input)
            }
        }
        output_partitions = self._keys_to_output_partitions(keys)
        if output_partitions != set(self.output_partitions):
            culled_layer = self._cull(output_partitions)
            return culled_layer, deps
        else:
            return self, deps
