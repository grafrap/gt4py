# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, cast

import numpy as np

from gt4py import next as gtx
from gt4py.next.modules.translator import (  # type: ignore[import-not-found]
    IDim,
    IndexMap,
    JDim,
    Kolor,
    StructuredRemapSizes,
    pack_edge_field_to_structured,
    pack_sparse_local_field_to_structured,
    pack_vertex_field_to_structured,
    unpack_edge_field,
    unpack_vertex_field_to_unstructured,
)
from gt4py.next.program_processors.program_setup_utils import setup_program
from gt4py.next.program_processors.runners import gtfn as gtfn_runner


class StructuredExecutionWrapper:
    def __init__(
        self,
        operator: Any,
        backend: Any,
        index_map: IndexMap,
        e2v_conn: np.ndarray,
        v2e_conn: np.ndarray,
        allocator: Any,
        offset_provider: dict,
    ) -> None:
        self.index_map = index_map
        self.e2v_conn = e2v_conn
        self.v2e_conn = v2e_conn
        self.allocator = allocator

        # Compile the actual structured GT4Py program under the hood
        self._compiled_program = setup_program(
            operator, backend=backend, offset_provider=offset_provider
        )

    def _is_unstructured(self, field: gtx.Field, axis_name: str) -> bool:
        if not getattr(field, "domain", None):
            return False
        return any(d.value == axis_name for d in field.domain.dims)

    def _pack_argument(self, field: gtx.Field) -> gtx.Field:
        if not getattr(field, "domain", None):
            return field

        np_data = field.asnumpy()

        # 1. Sparse fields (e.g., Sign: [Vertex, V2EDim])
        if self._is_unstructured(field, "Vertex") and np_data.ndim == 2:
            local_dim = field.domain.dims[1]
            struct_np = pack_sparse_local_field_to_structured(
                coeff=np_data,
                connectivity=self.v2e_conn,
                index_map=self.index_map,
                local_dim_name=getattr(local_dim, "value", "V2E"),  # TODO: make this more general
            )
            return gtx.as_field([IDim, JDim, Kolor, local_dim], struct_np, allocator=self.allocator)

        # 2. Standard unstructured fields
        if self._is_unstructured(field, "Vertex"):
            struct_np = pack_vertex_field_to_structured(np_data, self.index_map)
            return gtx.as_field([IDim, JDim, Kolor], struct_np, allocator=self.allocator)

        elif self._is_unstructured(field, "Edge"):
            struct_np = pack_edge_field_to_structured(np_data, self.index_map)
            return gtx.as_field([IDim, JDim, Kolor], struct_np, allocator=self.allocator)

        return field

    def _unpack_to_buffer(
        self, structured_field: gtx.Field, original_unstructured_field: gtx.Field
    ) -> None:
        if not getattr(original_unstructured_field, "domain", None):
            return

        struct_np = structured_field.asnumpy()
        orig_np = original_unstructured_field.asnumpy()

        if self._is_unstructured(original_unstructured_field, "Vertex"):
            unstruct_np = unpack_vertex_field_to_unstructured(struct_np, self.index_map)
        elif self._is_unstructured(original_unstructured_field, "Edge"):
            unstruct_np = unpack_edge_field(struct_np, self.index_map, orig_np.shape[0])
        else:
            return

        np.copyto(orig_np, unstruct_np)

    def __call__(self, **kwargs: Any) -> None:
        structured_kwargs: dict[str, Any] = {}
        out_fields = []

        for arg_name, arg_val in kwargs.items():
            if arg_name == "out":
                if isinstance(arg_val, tuple):
                    out_fields = list(arg_val)
                    structured_kwargs[arg_name] = tuple(self._pack_argument(f) for f in arg_val)
                else:
                    out_fields = [arg_val]
                    structured_kwargs[arg_name] = self._pack_argument(arg_val)
            else:
                structured_kwargs[arg_name] = self._pack_argument(arg_val)

        self._compiled_program(**structured_kwargs)

        if isinstance(kwargs["out"], tuple):
            for orig_f, struct_f in zip(out_fields, structured_kwargs["out"]):
                self._unpack_to_buffer(cast(gtx.Field, struct_f), orig_f)
        else:
            self._unpack_to_buffer(cast(gtx.Field, structured_kwargs["out"]), out_fields[0])


def setup_smart_program(
    operator: Any,
    setup: Any,
    index_map: IndexMap,
    remap_sizes: StructuredRemapSizes,
    allocator: Any,
    offset_provider: dict,
) -> StructuredExecutionWrapper:
    """Factory to create the structured wrapper."""
    structured_backend = gtfn_runner.GTFNBackendFactory(
        cached=True,
        otf_workflow__cached_translation=True,
        otf_workflow__bare_translation__symbolic_domain_sizes={
            "max_i": int(remap_sizes.max_i),
            "max_j": int(remap_sizes.max_j),
        },
    )

    return StructuredExecutionWrapper(
        operator=operator,
        backend=structured_backend,
        index_map=index_map,
        e2v_conn=setup.edges2node_connectivity.asnumpy(),
        v2e_conn=setup.nodes2edge_connectivity.asnumpy(),
        allocator=allocator,
        offset_provider=offset_provider,
    )
