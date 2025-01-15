from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

import dask
from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import MaterializedLayer
from dask.layers import DataFrameTreeReduction

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


class AwkwardTreeReductionLayer(DataFrameTreeReduction): ...
