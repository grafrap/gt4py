# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Backward-compatibility shim for cart_unroll.

All logic has been moved to structured_backend_passes.py.
This file re-exports the old names so pass_manager.py and existing
tests continue to work without changes. Do not add new logic here.
"""

from gt4py.next.iterator.transforms.structured_backend_passes import (
    # Main pass classes (old names kept for pass_manager.py compatibility)
    CartesianDomainAndTypeRemapper,
    CartesianReductionUnroller,
    CartUnroll,
    # New canonical names also exported for direct use
    StructuredBackend,
    StructuredTypeRemapper,
    SetAtRemapper,
    BroadcastAxisExpander,
    ThresholdConditionRewriter,
    SymbolicSizeInliner,
    NeighborReductionUnroller,
    ComposedShiftInliner,
    KolorConstantPropagation,
    CanDerefRewriter,
    RewriteCartesianCanDeref,
    ConcatWhereSetAtSplitter,
    TupleOutputSetAtSplitter,
    # Module-level helpers used by cartesian_interceptor.py, tests, and other callers
    _derive_entity_start_bounds_from_mapping,
    _derive_entity_range_bounds_from_mapping,
    _coerce_int,
    _coerce_mapping_rows,
    _bounded_shifted_deref,
    _build_field_concat_where_from_branches,
    _build_concat_where_from_branches,
    _make_lifted_deref_shift,
    _apply_shift_chain,
    _get_axis_name,
    _kolor_from_domain,
    _kolor_from_cw_condition,
    _kolor_cw_condition_to_domain,
    _concat_where_condition_from_domain,
    _expr_uses_edge_to_edge_connectivity,
    _e2c2e_on_local_intermediate,
    _is_unstructured_edge_domain_stmt,
    _detect_setat_entity,
    _EDGE_TO_EDGE_CONNECTIVITIES,
    _ENTITY_START_BOUNDS_CACHE,
    map_dict,
)

__all__ = [
    "CartesianDomainAndTypeRemapper",
    "CartesianReductionUnroller",
    "CartUnroll",
    "ConcatWhereSetAtSplitter",
    "TupleOutputSetAtSplitter",
    "StructuredBackend",
    "StructuredTypeRemapper",
    "SetAtRemapper",
    "BroadcastAxisExpander",
    "ThresholdConditionRewriter",
    "SymbolicSizeInliner",
    "NeighborReductionUnroller",
    "ComposedShiftInliner",
    "KolorConstantPropagation",
    "CanDerefRewriter",
    "RewriteCartesianCanDeref",
    "_derive_entity_start_bounds_from_mapping",
    "_derive_entity_range_bounds_from_mapping",
    "_coerce_int",
    "_coerce_mapping_rows",
    "_EDGE_TO_EDGE_CONNECTIVITIES",
    "_ENTITY_START_BOUNDS_CACHE",
    "map_dict",
]
