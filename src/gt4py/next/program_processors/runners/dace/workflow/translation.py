# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional, Union

import dace
import factory

from gt4py._core import definitions as core_defs
from gt4py.next import common, config
from gt4py.next.instrumentation import metrics
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.iterator.transforms import pass_manager
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.program_processors.runners.dace import (
    lowering as gtx_dace_lowering,
    sdfg_args as gtx_dace_args,
    transformations as gtx_transformations,
)
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon
from gt4py.next.type_system import type_specifications as ts
import os as _os


# DaCe SDFG map parameter name for the Kolor dimension.
# Computed once using the same function the SDFG lowering uses — guaranteed to match actual params.
_KOLOR_MAP_PARAM: str = gtx_dace_lowering.get_map_variable(common.Dimension("Kolor"))

# DaCe SDFG map parameter name for the vertical K dimension.
_K_MAP_PARAM: str = gtx_dace_lowering.get_map_variable(
    common.Dimension("K", kind=common.DimensionKind.VERTICAL)
)

# DaCe SDFG map parameter name for the IDim (warp / unit-stride) horizontal dimension.
_IDIM_MAP_PARAM: str = gtx_dace_lowering.get_map_variable(common.Dimension("IDim"))


def _sequentialize_k_dimension(sdfg: dace.SDFG) -> int:
    """Turn the vertical K dim of each top-level GPU map into a nested Sequential loop.

    Opt-in via DACE_SEQUENTIAL_K=1. After the GPU transformation a top-level map covers
    [IDim, JDim, Kolor, K] as the GPU thread grid (K → blockIdx). This strips K out into a
    nested ``Sequential`` map so the structure becomes a GPU_Device map over [IDim, JDim, Kolor]
    containing a sequential for-loop over K — i.e. each thread owns one horizontal point and
    loops the vertical levels (2.5D). K-independent work (mesh geometry, coefficients, composed
    shift addresses) is then computed once per thread and reused across all K-levels instead of
    re-fetched/re-derived for every (i, j, k). Returns the number of maps transformed.
    """
    from dace.transformation import helpers as dace_helpers

    n_transformed = 0
    for state in sdfg.states():
        candidates = [
            node
            for node in state.nodes()
            if isinstance(node, dace.sdfg.nodes.MapEntry)
            and state.entry_node(node) is None  # top-level only
            and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device
            and _K_MAP_PARAM in node.map.params
            and len(node.map.params) >= 2
        ]
        for map_entry in candidates:
            keep_dims = [
                i for i, p in enumerate(map_entry.map.params) if p != _K_MAP_PARAM
            ]
            # extract_map_dims puts `keep_dims` in the OUTER map and the rest (K) in the inner
            # remainder map.
            outer_entry, inner_k_entry = dace_helpers.extract_map_dims(
                sdfg, map_entry, keep_dims
            )
            outer_entry.map.schedule = dace.dtypes.ScheduleType.GPU_Device
            if inner_k_entry is not None:
                inner_k_entry.map.schedule = dace.dtypes.ScheduleType.Sequential
            n_transformed += 1
    return n_transformed


def _unroll_inner_k_maps(
    sdfg: dace.SDFG, tile_size: int, map_param: str = _K_MAP_PARAM
) -> int:
    """Mark the inner Sequential tiled maps (from LoopBlocking) for unrolling.

    ``map_param`` selects which blocked dimension's inner map to unroll (``_K_MAP_PARAM``
    for the vertical K tile / DACE_KTILE, ``_IDIM_MAP_PARAM`` for the horizontal IDim tile
    / DACE_ITILE — the latter makes each thread process B adjacent IDim columns whose loads
    nvcc fuses into wider/coalesced transactions, cutting LSU instructions / LG-Throttle).

    Opt-in via DACE_KTILE=B. LoopBlocking on the vertical K dim creates an inner
    ``Sequential`` map over B consecutive K-levels per thread but leaves
    ``Map.unroll=False`` (explicit TODO in the transform). Setting ``unroll=True`` makes
    DaCe codegen emit ``#pragma unroll`` — but LoopBlocking's inner range is
    ``[coarse*B : Min(N, coarse*B + B)]``, and that ``Min`` makes the trip count a
    *runtime* value, which defeats the unroll (nvcc keeps a rolled loop with a back-edge,
    so the B neighbour loads stay serialized and there is no ILP win — confirmed in SASS).

    When the total vertical size N is divisible by B, the last block is always full, so
    the ``Min`` is redundant: we rewrite the inner range stop to the constant
    ``coarse*B + (B-1)``. The trip count is then the compile-time constant B and nvcc
    actually unrolls, exposing the B independent loads for latency hiding (the whole point
    of K-tiling). If N is not divisible by B (or N can't be determined), we leave the
    ``Min`` in place (correct, just not unrolled). Returns the number of maps marked.
    """
    n_unrolled = 0
    for state in sdfg.states():
        for node in state.nodes():
            if not (
                isinstance(node, dace.sdfg.nodes.MapEntry)
                and node.map.schedule == dace.dtypes.ScheduleType.Sequential
                and map_param in node.map.params
            ):
                continue
            node.map.unroll = True
            n_unrolled += 1
            # Drop the redundant Min clamp so the trip count is the constant B (=tile_size),
            # which is what lets nvcc honour the #pragma unroll. Only safe when N % B == 0.
            idx = node.map.params.index(map_param)
            start, stop, step = node.map.range.ranges[idx]
            try:
                stop_sym = dace.symbolic.pystr_to_symbolic(str(stop))
                # The inclusive stop is Min(N-1, coarse*B + B - 1); recover N from the
                # integer operand(s) of any Min in the expression.
                n_candidates = [
                    int(a)
                    for m in stop_sym.atoms(dace.symbolic.sympy.Min)
                    for a in m.args
                    if a.is_Integer
                ]
                if n_candidates:
                    # LoopBlocking stores the inner stop as ``Min(N, coarse*B + B) - 1``
                    # (verified), so the Min's integer operand is exactly N (the K size).
                    total_n = max(n_candidates)
                    if total_n % int(tile_size) == 0:
                        new_stop = dace.symbolic.pystr_to_symbolic(str(start)) + (
                            int(tile_size) - 1
                        )
                        node.map.range.ranges[idx] = (start, new_stop, step)
            except Exception:  # noqa: BLE001 — leave the Min in place if anything is unexpected
                pass
    return n_unrolled


def _map_nontransient_output_arrays(
    graph: dace.SDFGState,
    sdfg: dace.SDFG,
    map_node: Union[dace.nodes.MapExit, dace.nodes.MapEntry],
) -> set[str]:
    """Return the set of non-transient (program-output) array names written by a map scope."""
    map_exit = (
        map_node
        if isinstance(map_node, dace.nodes.MapExit)
        else graph.exit_node(map_node)
    )
    outs: set[str] = set()
    for edge in graph.out_edges(map_exit):
        dst = edge.dst
        if isinstance(dst, dace.nodes.AccessNode) and not sdfg.arrays[dst.data].transient:
            outs.add(dst.data)
    return outs


def _kolor_aware_fusion_callback(
    self: Any,
    first_map_node: Union[dace.nodes.MapExit, dace.nodes.MapEntry],
    second_map_entry: dace.nodes.MapEntry,
    graph: dace.SDFGState,
    sdfg: dace.SDFG,
) -> bool:
    """Refuse to fuse maps with different single-element Kolor ranges (or disjoint outputs).

    Per-kolor split stencils produce separate maps per Kolor value (Kolor:[k,k+1)).
    Fusing two such maps with different k values creates a dimensionality mismatch
    in the fusion temporary (InvalidSDFGEdgeError). Fusion is allowed when either
    map has no Kolor parameter, both share the same Kolor start index, or the range
    is symbolic and cannot be evaluated at transform time.

    OPT-IN output fission (DACE_FISSION_OUTPUTS=1): additionally refuse to fuse two maps
    that write *disjoint non-transient program outputs*. Tuple-returning stencils (e.g.
    rbf_nabla4 → (u_vert_out, v_vert_out)) lower to ``make_tuple(as_fieldop_u, as_fieldop_v)``
    (DaCe cannot lower a single tuple-output as_fieldop), i.e. two independent maps writing
    distinct outputs that MapFusionHorizontal would otherwise merge into one register-heavy
    kernel that spills. Keeping them separate roughly halves the live-register footprint per
    kernel. Only fires for disjoint non-transient outputs, so producer→consumer (vertical)
    fusion and shared-output fusion are unaffected.
    """
    import os

    if os.environ.get("DACE_FISSION_OUTPUTS", "0") == "1":
        first_outs = _map_nontransient_output_arrays(graph, sdfg, first_map_node)
        second_outs = _map_nontransient_output_arrays(graph, sdfg, second_map_entry)
        if first_outs and second_outs and first_outs.isdisjoint(second_outs):
            return False  # distinct program outputs — keep as separate kernels

    first_map_entry = (
        first_map_node
        if isinstance(first_map_node, dace.nodes.MapEntry)
        else graph.entry_node(first_map_node)
    )
    first_params = first_map_entry.map.params
    second_params = second_map_entry.map.params

    if _KOLOR_MAP_PARAM not in first_params or _KOLOR_MAP_PARAM not in second_params:
        return True  # One or both maps have no Kolor — allow fusion

    first_kolor_range = first_map_entry.map.range[first_params.index(_KOLOR_MAP_PARAM)]
    second_kolor_range = second_map_entry.map.range[second_params.index(_KOLOR_MAP_PARAM)]

    try:
        if int(first_kolor_range[0]) != int(second_kolor_range[0]):
            return False  # Different Kolor values — refuse fusion
    except (TypeError, ValueError):
        pass  # Symbolic range — cannot determine, allow fusion conservatively

    return True


def _inline_multi_consumer_transients(sdfg: dace.SDFG) -> None:
    """Inline producer maps into each consumer when an intermediate has >1 reader.

    `MapPromoter` only promotes transients that are "single use" (one reader). For
    nabla4-type stencils the E2C2V intermediate (e.g. `gtir_tmp_31`) is written once
    but consumed by two separate maps (boundary zone and interior zone). This helper
    calls `inline_dataflow_into_map` per consumer edge, embedding the producer
    computation as a NestedSDFG inside each consumer. The intermediate is then dead
    and removed by the subsequent `gt_auto_optimize` passes.

    Fixed-point loop: restarts after each successful inline because the SDFG
    structure changes and iterators become stale.
    """
    n_inlined = 1
    while n_inlined > 0:
        n_inlined = 0
        for state in sdfg.states():
            scope_dict = state.scope_dict()
            for map_entry in list(state.nodes()):
                if not isinstance(map_entry, dace.nodes.MapEntry):
                    continue
                if scope_dict.get(map_entry) is not None:
                    continue  # only top-level maps
                for edge in list(state.out_edges(map_entry)):
                    if not edge.src_conn or not edge.src_conn.startswith("OUT_"):
                        continue
                    in_conn = "IN_" + edge.src_conn[4:]
                    in_edges = list(state.in_edges_by_connector(map_entry, in_conn))
                    if len(in_edges) != 1:
                        continue
                    src = in_edges[0].src
                    if not isinstance(src, dace.nodes.AccessNode):
                        continue
                    desc = sdfg.arrays.get(src.data)
                    if desc is None or not desc.transient:
                        continue
                    if state.out_degree(src) <= 1:
                        continue  # single consumer — MapPromoter already handles it
                    result = gtx_transformations.inline_dataflow_into_map(sdfg, state, edge)
                    if result is not None:
                        n_inlined += 1
                        break  # state changed — restart from outer while
                if n_inlined > 0:
                    break  # restart state loop
            if n_inlined > 0:
                break  # restart sdfg-states loop


def find_constant_symbols(
    ir: itir.Program,
    sdfg: dace.SDFG,
    offset_provider_type: common.OffsetProviderType,
    disable_field_origin_on_program_arguments: bool,
    unstructured_horizontal_has_unit_stride: bool,
) -> dict[str, int]:
    """Helper function to find symbols to replace with constant values."""
    constant_symbols: dict[str, int] = {}

    if unstructured_horizontal_has_unit_stride:
        # Search the stride symbols corresponding to the horizontal dimension
        for p in ir.params:
            if isinstance(p.type, ts.FieldType):
                h_dims = [dim for dim in p.type.dims if dim.kind == common.DimensionKind.HORIZONTAL]
                if len(h_dims) == 0:
                    continue
                elif len(h_dims) == 1:
                    dim = h_dims[0]
                else:
                    raise NotImplementedError(
                        f"Unsupported field with multiple horizontal dimensions '{p}'."
                    )
                sdfg_stride_symbol = gtx_dace_args.field_stride_symbol(str(p.id), dim)
                constant_symbols[sdfg_stride_symbol.name] = 1
        # Same for connectivity tables, for which the first dimension is always horizontal
        for offset, conn_type in offset_provider_type.items():
            if (
                isinstance(conn_type, common.NeighborConnectivityType)
                and (conn_id := gtx_dace_args.connectivity_identifier(offset)) in sdfg.arrays
            ):
                assert not sdfg.arrays[conn_id].transient
                assert conn_type.source_dim.kind == common.DimensionKind.HORIZONTAL
                sdfg_stride_symbol = gtx_dace_args.field_stride_symbol(
                    conn_id, conn_type.source_dim, offset_provider_type
                )
                constant_symbols[sdfg_stride_symbol.name] = 1

    if disable_field_origin_on_program_arguments:
        # collect symbols used as range start for all program arguments
        for p in ir.params:
            if isinstance(p.type, ts.TupleType):
                psymbols = [
                    sym
                    for sym in gtx_dace_lowering.flatten_tuple_fields(p.id, p.type)
                    if isinstance(sym.type, ts.FieldType)
                ]
            elif isinstance(p.type, ts.FieldType):
                psymbols = [p]
            else:
                psymbols = []
            for psymbol in psymbols:
                assert isinstance(psymbol.type, ts.FieldType)
                if len(psymbol.type.dims) == 0:
                    # zero-dimensional field
                    continue
                # set all range start symbols to constant value 0
                sdfg_origin_symbols = [
                    gtx_dace_args.range_start_symbol(str(psymbol.id), dim)
                    for dim in psymbol.type.dims
                ]
                constant_symbols |= {sdfg_symbol.name: 0 for sdfg_symbol in sdfg_origin_symbols}

    return constant_symbols


def _structured_field_stride_constants(
    ir: itir.Program,
    sdfg: dace.SDFG,
    symbolic_domain_sizes: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    """Bake structured field strides as compile-time constants from the exact packed-array strides.

    Default ON for the structured backend (opt out with GT4PY_BAKE_STRIDES=0). The structured
    backend passes every field stride
    (``__<field>_<dim>_stride``) as a runtime kernel argument, so the GPU compiler cannot prove the
    warp dimension is unit-stride. It therefore emits register-indirect gather loads (uncoalesced —
    ncu reports ~11% excessive sectors) and spends registers on per-thread address arithmetic, which
    lowers occupancy. Baking the strides folds the constant neighbour-offset address arithmetic to a
    single constant per access -> coalesced warp loads + freed address registers -> higher occupancy.

    The values are the EXACT element strides of the packed arrays, captured by
    ``GenericStructuredWrapper.__call__`` and passed via ``symbolic_domain_sizes['baked_strides']``
    as ``{field_name: [stride_dim0, stride_dim1, ...]}``. This is the only correct source: the
    origin-shift / cache-line padding (``pack_vertex_field_padded``, ``pack_edge_field_compact`` pad
    IDim to a multiple of ``_STRIDE_PAD``) makes strides per-field and not derivable from the grid
    size. Each non-transient SDFG array's symbolic strides are matched **by position** to that
    field's packed strides; only symbols still free in the SDFG are baked. The runtime check
    ``get_array_stride_symbols`` validates every baked constant against the real array stride.
    """
    sds = symbolic_domain_sizes or {}
    baked: dict[str, list[int]] = sds.get("baked_strides") or {}
    baked_r0: dict[str, dict[str, int]] = sds.get("baked_range_starts") or {}
    if not baked and not baked_r0:
        return {}
    free = {str(s) for s in sdfg.free_symbols}
    consts: dict[str, int] = {}

    # Strides: match each non-transient SDFG array's symbolic strides to the packed strides by pos.
    for name, arr in sdfg.arrays.items():
        if getattr(arr, "transient", False):
            continue
        strides_list = baked.get(name)
        if not strides_list:
            continue
        for sym, value in zip(arr.strides, strides_list):
            sym_name = str(sym)
            if sym_name in free:
                consts[sym_name] = int(value)

    # Range-starts (field origins): bake __<field>_<dim>_range_0 from the exact packed-field domain
    # (the only remaining runtime symbols after stride baking). Matched per field+dim by name.
    if baked_r0:
        fields: list[Any] = []
        for p in ir.params:
            if isinstance(p.type, ts.TupleType):
                fields.extend(
                    sym
                    for sym in gtx_dace_lowering.flatten_tuple_fields(p.id, p.type)
                    if isinstance(sym.type, ts.FieldType)
                )
            elif isinstance(p.type, ts.FieldType):
                fields.append(p)
        for f in fields:
            starts = baked_r0.get(str(f.id))
            if not starts:
                continue
            for d in f.type.dims:
                dval = getattr(d, "value", None)
                if dval in starts:
                    sym = gtx_dace_args.range_start_symbol(str(f.id), d)
                    if sym.name in free:
                        consts[sym.name] = int(starts[dval])
    return consts



def make_sdfg_call_async(sdfg: dace.SDFG, gpu: bool) -> None:
    """Configure an SDFG to immediately return once all work has been scheduled.

    This means that `CompiledSDFG.fast_call()` will return immediately after all
    computations have been _scheduled_ on the device. This function only has an effect
    for work that runs on the GPU. Furthermore, all work is scheduled on the
    default stream.

    Todo: Revisit this function once DaCe changes its behaviour in this regard.
    """

    # This is only a problem on GPU.
    # TODO(phimuell): Figuring out what about OpenMP.
    if not gpu:
        return

    assert dace.Config.get("compiler.cuda.max_concurrent_streams") == -1, (
        f"Expected `max_concurrent_streams == -1` but it was `{dace.Config.get('compiler.cuda.max_concurrent_streams')}`."
    )

    # NOTE: We are using the default stream this means that _**currently**_ the launch is
    #   already asynchronous, see [DaCe issue#2120](https://github.com/spcl/dace/issues/2120)
    #   for more. However, DaCe still [generates streams internally](https://github.com/spcl/dace/blob/54c935cfe74a52c5107dc91680e6201ddbf86821/dace/codegen/targets/cuda.py#L467).
    #   Thus to be absolutely sure we will no set all streams, DaCe uses to the default
    #   stream.
    # NOTE: Another important note here is, that it looks as if no synchronization
    #   what soever between states is generated if the default stream is used. This
    #   might lead to problems if a GPU kernel computes something that is needed for a
    #   condition of an interstate edge. However, we should not have that case.
    # TODO(phimuell, edopao): Make sure if this is really the case.
    dace_gpu_backend = dace.Config.get("compiler.cuda.backend")
    assert dace_gpu_backend in ["cuda", "hip"], f"GPU backend '{dace_gpu_backend}' is unknown."
    sdfg.append_init_code(
        f"__dace_gpu_set_all_streams(__state, {dace_gpu_backend}StreamDefault);",
        location="cuda",
    )


def _has_gpu_schedule(sdfg: dace.SDFG) -> bool:
    """Check if any node (e.g. maps) of the given SDFG is scheduled on GPU."""
    return any(
        getattr(node, "schedule", dace.dtypes.ScheduleType.Default) in dace.dtypes.GPU_SCHEDULES
        for node, _ in sdfg.all_nodes_recursive()
    )


def _make_if_region_for_metrics_collection(
    name: str,
    metrics_level: str,
    sdfg: dace.SDFG,
) -> tuple[dace.state.ConditionalBlock, dace.state.SDFGState]:
    """
    Helper function to create a conditional block in the given SDFG, with only one
    branch to be executed if 'metric_level >= metrics.PERFORMANCE' is true.
    """
    if_region = dace.sdfg.state.ConditionalBlock(name)
    sdfg.add_node(if_region, ensure_unique_name=True)
    then_body = dace.sdfg.state.ControlFlowRegion(f"{if_region.label}_collect_metrics", sdfg=sdfg)
    then_state = then_body.add_state(f"{if_region.label}_collect_metrics")
    if_region.add_branch(
        dace.sdfg.state.CodeBlock(f"{metrics_level} >= {metrics.PERFORMANCE}"), then_body
    )
    return if_region, then_state


def _report_copy_kernels(sdfg: dace.SDFG) -> None:
    """Print a greppable ``COPY_KERNEL_DETECTED`` line for each pure data-copy map kernel.

    DaCe emits a separate GPU kernel for every array-to-array copy map (labelled
    ``copy_<src>_<dst>``), as opposed to the compute maps (``map_*_fieldop_*``). These copies are
    pure data movement (no compute) and a prime target for elimination — e.g. multi-output
    stencils such as ``compute_exner_from_rhotheta`` lower to one compute kernel plus two
    ``copy_*`` kernels that the unstructured backend avoids. Reporting them by name lets a whole
    suite run be grepped (``grep COPY_KERNEL_DETECTED``) to find which stencils still materialize
    copy kernels and how many. Only prints when at least one copy kernel is present.
    """
    copy_kernels: set[str] = set()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.label.startswith("copy_"):
            # Normalize "copy_<src>_<dst>_map[<ranges>]" → "copy_<src>_<dst>".
            base = node.map.label.split("[", 1)[0]
            if base.endswith("_map"):
                base = base[: -len("_map")]
            copy_kernels.add(base)
    if copy_kernels:
        print(
            f"COPY_KERNEL_DETECTED: program={sdfg.name} count={len(copy_kernels)} "
            f"kernels={sorted(copy_kernels)}",
            flush=True,
        )


def _instrument_copy_kernels_for_compute_only(sdfg: dace.SDFG) -> bool:
    """Time copy_* GPU kernels via SDFG pre/post timing states (fast_call-compatible).

    DEFAULT-ON (opt out with ``DACE_TIME_COMPUTE_ONLY=0``). Enables the *compute-only timer*:
    decoration.py subtracts the accumulated copy-kernel duration from the chrono wall-time so
    the reported GT4Py Timer metric reflects only the compute kernels.

    Uses the same accumulator pattern as ``add_instrumentation`` (cudaStreamSynchronize +
    std::chrono), rather than GPU_Events + get_latest_report(). GPU_Events requires DaCe to
    flush the instrumentation report to disk, which only happens when the CompiledSDFG is
    finalized — never during fast_call(). Instead, this adds ``SDFG_ARG_METRIC_COPY_TIME`` as
    a non-transient SDFG output array, wraps each copy_* state with a pre/post timing state,
    and writes the elapsed wall time into the array. decoration.py reads it directly after
    fast_call() (no file I/O, no report flushing required).

    Correctness rests on ``max_concurrent_streams == -1`` (default stream only ⇒ sequential
    execution ⇒ total_wall = compute_wall + copy_wall exactly). Returns True iff at least one
    copy state was instrumented.
    """
    if _os.environ.get("DACE_TIME_COMPUTE_ONLY", "1") == "0":
        return False
    if dace.Config.get("compiler.cuda.max_concurrent_streams") != -1:
        return False

    # Only wrap states where ALL top-level maps are copy_* maps.
    # States that mix a compute map (map_*_fieldop_*) with a copy map cannot be timed
    # for copy-only: the pre/post states would bracket both kernels together, causing
    # copy_t ≈ total time and ~99% over-subtraction.
    def _top_level_maps(state):
        return [
            n for n in state.nodes()
            if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) is None
        ]

    copy_states = [
        state for state in sdfg.states()
        if (
            maps := _top_level_maps(state)
        ) and all(m.map.label.startswith("copy_") for m in maps)
    ]
    # Debug: print all states with any copy map and which ones were selected for timing.
    for state in sdfg.states():
        maps = _top_level_maps(state)
        if any(m.map.label.startswith("copy_") for m in maps):
            selected = state in copy_states
            print(
                f"[copy_time_debug] state='{state.label}'  "
                f"maps={[m.map.label for m in maps]}  "
                f"selected_for_timing={selected}",
                flush=True,
            )
    if not copy_states:
        return False

    # Non-transient output array: decoration.py resets to 0.0 before each fast_call(),
    # and reads the total copy time after fast_call(). C++ accumulates into it per copy state.
    copy_time_name = gtx_wfdcommon.SDFG_ARG_METRIC_COPY_TIME
    sdfg.add_array(copy_time_name, [1], dace.float64)

    dace_gpu_backend = dace.Config.get("compiler.cuda.backend")
    assert dace_gpu_backend in ["cuda", "hip"]
    sync_code = f"{dace_gpu_backend}StreamSynchronize({dace_gpu_backend}StreamDefault);"

    for i, copy_state in enumerate(copy_states):
        # Snapshot edges before modifying the graph
        in_edges = list(sdfg.in_edges(copy_state))
        out_edges = list(sdfg.out_edges(copy_state))

        # Transients: start timestamp (ns) + previous accumulated copy time
        start_var = f"__gt_copy_start_ns_{i}"
        prev_var = f"__gt_prev_copy_t_{i}"
        sdfg.add_array(start_var, [1], dace.int64, transient=True)
        sdfg.add_array(prev_var, [1], dace.float64, transient=True)

        pre = sdfg.add_state(f"copy_timing_pre_{i}")
        post = sdfg.add_state(f"copy_timing_post_{i}")

        # Promote pre to start block if copy_state was the start block
        if sdfg.start_block is copy_state:
            sdfg.start_block = sdfg.node_id(pre)

        # Rewire: predecessors → pre → copy_state → post → successors
        for edge in in_edges:
            sdfg.add_edge(edge.src, pre, edge.data)
            sdfg.remove_edge(edge)
        sdfg.add_edge(pre, copy_state, dace.InterstateEdge())
        sdfg.add_edge(copy_state, post, dace.InterstateEdge())
        for edge in out_edges:
            sdfg.add_edge(post, edge.dst, edge.data)
            sdfg.remove_edge(edge)

        # Pre-state: GPU sync + record start timestamp + snapshot current accumulated copy time.
        # Reading copy_time_name here (not in post) keeps the post-state free of read-write
        # conflicts on the same array within one DaCe state.
        ct_read = pre.add_access(copy_time_name)
        start_access = pre.add_access(start_var)
        prev_access = pre.add_access(prev_var)
        pre_tlet = pre.add_tasklet(
            f"copy_pre_{i}",
            inputs={"ct_in"},
            outputs={"t_start", "ct_prev"},
            code=f"""\
{sync_code}
auto __gt_copy_now_{i} = std::chrono::high_resolution_clock::now();
t_start = std::chrono::duration_cast<std::chrono::nanoseconds>(
    __gt_copy_now_{i}.time_since_epoch()).count();
ct_prev = ct_in;
""",
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        pre.add_edge(ct_read, None, pre_tlet, "ct_in", dace.Memlet(f"{copy_time_name}[0]"))
        pre.add_edge(pre_tlet, "t_start", start_access, None, dace.Memlet(f"{start_var}[0]"))
        pre.add_edge(pre_tlet, "ct_prev", prev_access, None, dace.Memlet(f"{prev_var}[0]"))

        # Post-state: GPU sync + compute elapsed + write accumulated copy time.
        start_read = post.add_access(start_var)
        prev_read = post.add_access(prev_var)
        ct_write = post.add_access(copy_time_name)
        post_tlet = post.add_tasklet(
            f"copy_post_{i}",
            inputs={"t_start", "ct_prev"},
            outputs={"ct_out"},
            code=f"""\
{sync_code}
auto __gt_copy_now2_{i} = std::chrono::high_resolution_clock::now();
auto __gt_copy_stop_ns_{i} = std::chrono::duration_cast<std::chrono::nanoseconds>(
    __gt_copy_now2_{i}.time_since_epoch()).count();
ct_out = ct_prev + static_cast<double>(__gt_copy_stop_ns_{i} - t_start) * 1.0e-9;
""",
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        post.add_edge(start_read, None, post_tlet, "t_start", dace.Memlet(f"{start_var}[0]"))
        post.add_edge(prev_read, None, post_tlet, "ct_prev", dace.Memlet(f"{prev_var}[0]"))
        post.add_edge(post_tlet, "ct_out", ct_write, None, dace.Memlet(f"{copy_time_name}[0]"))

    sdfg.validate()
    return True


def add_instrumentation(sdfg: dace.SDFG, gpu: bool) -> None:
    """
    Instrument SDFG with measurement of total execution time.

    We measure the execution time of one GT4Py program by instrumenting the top-level
    SDFG with a cpp timer (std::chrono). This timer measures only the computation
    time, it does not include the overhead of calling the SDFG from Python.

    The execution time is measured in seconds and represented as a 'float64' value.
    It is written to the global array 'SDFG_ARG_METRIC_COMPUTE_TIME'.
    """
    output, _ = sdfg.add_array(gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME, [1], dace.float64)
    start_time, _ = sdfg.add_scalar("gt_start_time", dace.int64, transient=True)
    metrics_level = sdfg.add_symbol(gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL, dace.int32)

    #### 1. Synchronize the CUDA device, in order to wait for kernels completion.
    # Even when the target device is GPU, it can happen that dace emits code without
    # GPU kernels. In this case, the cuda headers are not imported and the SDFG is
    # compiled as plain C++. Therefore, we also check here the schedule of SDFG maps.
    if gpu and _has_gpu_schedule(sdfg):
        dace_gpu_backend = dace.Config.get("compiler.cuda.backend")
        assert dace_gpu_backend in ["cuda", "hip"], f"GPU backend '{dace_gpu_backend}' is unknown."

        # NOTE: We should actually wrap the `DeviceSynchronize` function inside a
        #   `DACE_GPU_CHECK()` macro. However, this only works in GPU context, but
        #   here we are in CPU context. Thus we cannot do it.
        sync_code = f"{dace_gpu_backend}DeviceSynchronize();"
        has_side_effects = True

    else:
        sync_code = "/* The SDFG execution should already be synchronized */"
        has_side_effects = False

    #### 2. Timestamp the SDFG entry point.
    start_block = sdfg.start_block
    entry_if_region, begin_state = _make_if_region_for_metrics_collection(
        "metrics_entry", metrics_level, sdfg
    )
    sdfg.add_edge(entry_if_region, start_block, dace.InterstateEdge())
    sdfg.start_block = sdfg.node_id(entry_if_region)
    assert sdfg.start_block is entry_if_region

    tlet_start_timer = begin_state.add_tasklet(
        "gt_start_timer",
        inputs={},
        outputs={"time"},
        code=f"""\
{sync_code}
auto now = std::chrono::high_resolution_clock::now();
time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
        """,
        language=dace.dtypes.Language.CPP,
        side_effects=has_side_effects,
    )
    begin_state.add_edge(
        tlet_start_timer,
        "time",
        begin_state.add_access(start_time),
        None,
        dace.Memlet(f"{start_time}[0]"),
    )

    #### 3. Collect the SDFG end timestamp and produce the compute metric.
    exit_if_region, end_state = _make_if_region_for_metrics_collection(
        "metrics_exit", metrics_level, sdfg
    )
    for sink_node in sdfg.sink_nodes():
        if sink_node is exit_if_region:
            continue
        sdfg.add_edge(sink_node, exit_if_region, dace.InterstateEdge())
    assert sdfg.in_degree(exit_if_region) > 0

    # Populate the branch that computes the stencil time metric
    tlet_stop_timer = end_state.add_tasklet(
        "gt_stop_timer",
        inputs={"run_cpp_start_time"},
        outputs={"duration"},
        code=f"""\
{sync_code}
auto now = std::chrono::high_resolution_clock::now();
auto run_cpp_end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
duration = static_cast<double>(run_cpp_end_time - run_cpp_start_time) * 1.e-9;
        """,
        language=dace.dtypes.Language.CPP,
        side_effects=has_side_effects,
    )
    end_state.add_edge(
        end_state.add_access(start_time),
        None,
        tlet_stop_timer,
        "run_cpp_start_time",
        dace.Memlet(f"{start_time}[0]"),
    )
    end_state.add_edge(
        tlet_stop_timer,
        "duration",
        end_state.add_access(output),
        None,
        dace.Memlet(f"{output}[0]"),
    )

    if gpu and _has_gpu_schedule(sdfg) and config.ADD_GPU_TRACE_MARKERS:
        sdfg.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(
                node, dace.nodes.MapEntry
            ):  # Add ranges to scopes and maps that are NOT scheduled to the GPU
                node.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS
            elif isinstance(node, dace.sdfg.state.SDFGState):
                node.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS

    # Check SDFG validity after applying the above changes.
    # Normally, we do not call `SDFGState.add_tasklet()` directly, instead we call
    #  the wrapper provided by `DataflowBuilder`, that modifies the tasklet connectors
    #  to avoid name conflicts with program symbols. However, this method is not
    #  available here, so we have to call the underlying DaCe function directly.
    #  We now run `validate()` to make sure that no name conflict was introduced.
    sdfg.validate()


def make_sdfg_call_sync(sdfg: dace.SDFG, gpu: bool) -> None:
    """Process the SDFG such that the call is synchronous.

    This means that `CompiledSDFG.fast_call()` will return only after all computations
    have _finished_ and the results are available. This function only has an effect for
    work that runs on the GPU. Furthermore, all work is scheduled on the default stream.
    """

    if not gpu:
        # This is only a problem on GPU. Dace uses OpenMP on CPU and
        # the OpenMP parallel region creates a synchronization point.
        return
    elif not _has_gpu_schedule(sdfg):
        # Even when the target device is GPU, it can happen that dace
        # emits code without GPU kernels. In this case, the cuda headers
        # are not imported and the SDFG is compiled as plain C++.
        return

    assert dace.Config.get("compiler.cuda.max_concurrent_streams") == -1, (
        f"Expected `max_concurrent_streams == -1` but it was `{dace.Config.get('compiler.cuda.max_concurrent_streams')}`."
    )

    # If we are using the default stream, things are a bit simpler/harder. For some
    #  reasons when using the default stream, DaCe seems to skip _all_ synchronization,
    #  for more see [DaCe issue#2120](https://github.com/spcl/dace/issues/2120).
    #  Thus the `CompiledSDFG.fast_call()` call is truly asynchronous, i.e. just
    #  launches the kernels and then exist. Thus we have to add a synchronization
    #  at the end to have a synchronous call. We can not use `SDFG.append_exit_code()`
    #  because that code is only run at the `exit()` stage, not after a call. Thus we
    #  will generate an SDFGState that contains a Tasklet with the sync call.
    sync_state = sdfg.add_state("sync_state")
    for sink_node in sdfg.sink_nodes():
        if sink_node is sync_state:
            continue
        sdfg.add_edge(sink_node, sync_state, dace.InterstateEdge())
    assert sdfg.in_degree(sync_state) > 0

    # NOTE: Since the synchronization is done through the Tasklet explicitly,
    #   we can disable synchronization for the last state. Might be useless.
    sync_state.nosync = True

    # NOTE: We should actually wrap the `StreamSynchronize` function inside a
    #   `DACE_GPU_CHECK()` macro. However, this only works in GPU context, but
    #   here we are in CPU context. Thus we can not do it.
    dace_gpu_backend = dace.Config.get("compiler.cuda.backend")
    assert dace_gpu_backend in ["cuda", "hip"], f"GPU backend '{dace_gpu_backend}' is unknown."
    sync_state.add_tasklet(
        "sync_tlet",
        inputs=set(),
        outputs=set(),
        code=f"{dace_gpu_backend}StreamSynchronize({dace_gpu_backend}StreamDefault);",
        language=dace.dtypes.Language.CPP,
        side_effects=True,
    )

    # DaCe [still generates a stream](https://github.com/spcl/dace/blob/54c935cfe74a52c5107dc91680e6201ddbf86821/dace/codegen/targets/cuda.py#L467)
    #  despite not using it. Thus to be absolutely sure, we will not set that stream
    #  to the default stream.
    sdfg.append_init_code(
        f"__dace_gpu_set_all_streams(__state, {dace_gpu_backend}StreamDefault);",
        location="cuda",
    )


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        definitions.CompilableProgramDef,
        stages.ProgramSource[code_specs.SDFGCodeSpec],
    ],
    definitions.TranslationStep[code_specs.SDFGCodeSpec],
):
    device_type: core_defs.DeviceType
    apply_common_transform: bool
    auto_optimize: bool
    auto_optimize_args: dict[str, Any] | None
    async_sdfg_call: bool
    unstructured_horizontal_has_unit_stride: bool
    use_metrics: bool

    disable_itir_transforms: bool = False
    disable_field_origin_on_program_arguments: bool = False
    use_max_domain_range_on_unstructured_shift: bool | None = None
    symbolic_domain_sizes: dict[str, Any] | None = None

    def generate_sdfg(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dace.SDFG:
        with gtx_wfdcommon.dace_context(device_type=self.device_type):
            return self._generate_sdfg_without_configuring_dace(*args, **kwargs)

    def _preprocess_program(
        self,
        program: itir.Program,
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
    ) -> itir.Program:
        apply_common_transforms = functools.partial(
            pass_manager.apply_common_transforms,
            offset_provider=offset_provider,
            force_inline_lambda_args=True,
            transform_concat_where_to_as_fieldop=False,
            use_max_domain_range_on_unstructured_shift=self.use_max_domain_range_on_unstructured_shift,
            symbolic_domain_sizes=self.symbolic_domain_sizes,
        )

        new_program = apply_common_transforms(program, unroll_reduce=False)

        if any(
            node.id == "neighbors"
            for node in new_program.pre_walk_values().if_isinstance(itir.SymRef)
        ):
            # if we don't unroll, there may be lifts left in the itir which can't
            # be lowered to SDFG. In this case, just retry with unrolled reductions.
            new_program = apply_common_transforms(program, unroll_reduce=True)

        return new_program

    def _generate_sdfg_without_configuring_dace(
        self,
        ir: itir.Program,
        offset_provider: common.OffsetProvider,
        column_axis: Optional[common.Dimension],
    ) -> dace.SDFG:
        import os as _os

        symbolic_domain_sizes = None
        if _os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
            try:
                from gt4py.next.modules.cartesian_interceptor import get_compile_sds
                symbolic_domain_sizes = get_compile_sds()
            except ImportError:
                pass
            if symbolic_domain_sizes is None:
                try:
                    from gt4py.next.program_processors.codegens.gtfn.gtfn_module import (
                        GTFNTranslationStep,
                    )
                    symbolic_domain_sizes = (
                        GTFNTranslationStep._resolve_symbolic_domain_sizes_from_mesh_metadata()
                    ) or None
                except Exception:
                    pass

        # Classify the PRE-transform GTIR for the automatic dimension-order policy (used at the
        # `_usd_default` site below). This must run before the itir transforms: the structured
        # passes (NeighborReductionUnroller) replace connectivity OffsetLiterals (E2C2V, ...)
        # with cartesian IDim/JDim/Kolor shifts, so gather connectivity is only visible here.
        _pretransform_offsets: set = set()
        if _os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
            for _n in ir.pre_walk_values().if_isinstance(itir.OffsetLiteral):
                if isinstance(_n.value, str):
                    _pretransform_offsets.add(_n.value)

        if not self.disable_itir_transforms:
            if self.apply_common_transform:
                ir = self._preprocess_program(ir, offset_provider)
            else:
                ir = itir_transforms.pre_inline_scalar_params(ir, symbolic_domain_sizes)
                ir = itir_transforms.apply_fieldview_transforms(
                    ir,
                    use_max_domain_range_on_unstructured_shift=self.use_max_domain_range_on_unstructured_shift,
                    offset_provider=offset_provider,
                    unroll_reduce=True,
                    symbolic_domain_sizes=symbolic_domain_sizes,
                )
        pass_manager._print_ir_block("IR after common transforms", ir, enabled=True)
        offset_provider_type = common.offset_provider_to_type(offset_provider)

        if _os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
            structured_dim_entries = {
                "IDim": common.Dimension("IDim"),
                "JDim": common.Dimension("JDim"),
                "Kolor": common.Dimension("Kolor"),
            }
            offset_provider_type = {**offset_provider_type, **structured_dim_entries}

        on_gpu = self.device_type != core_defs.DeviceType.CPU

        sdfg = gtx_dace_lowering.build_sdfg_from_gtir(ir, offset_provider_type, column_axis)
        # Debug dump of the pre-GPU-transformation SDFG — uncomment when needed.
        # sdfg.save("before_gpu_transformation.sdfg")
        constant_symbols = find_constant_symbols(
            ir,
            sdfg,
            offset_provider_type,
            self.disable_field_origin_on_program_arguments,
            self.unstructured_horizontal_has_unit_stride,
        )
        # Bake the structured per-field strides and origins (range_0) as compile-time constants so the
        # GPU compiler coalesces warp loads and folds the neighbour-offset address arithmetic out of
        # registers — a strict, correctness-safe win that raises occupancy. Default ON for the
        # structured backend; opt out with GT4PY_BAKE_STRIDES=0. find_constant_symbols cannot do this
        # itself (it asserts a single horizontal dim; structured fields have three). See
        # _structured_field_stride_constants.
        if (
            _os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1"
            and _os.environ.get("GT4PY_BAKE_STRIDES", "1") != "0"
        ):
            constant_symbols.update(
                _structured_field_stride_constants(ir, sdfg, symbolic_domain_sizes)
            )

        if self.auto_optimize:
            auto_optimize_args = {} if self.auto_optimize_args is None else self.auto_optimize_args
            _is_structured = _os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1"

            # Experiment selector: set DACE_OPT_EXPERIMENT env var to override defaults. Computed
            # UNCONDITIONALLY (both structured and unstructured) so DACE_KTILE / DACE_GPU_LAUNCH_
            # BOUNDS / DACE_OPT_EXPERIMENT reach the unstructured backend too — these are generic
            # DaCe SDFG-level knobs (register/occupancy tuning, K-loop blocking) that don't depend
            # on the structured IDim/JDim/Kolor layout. Previously this whole block, including
            # DACE_KTILE, was nested inside `if _is_structured:`, so setting these env vars for an
            # unstructured (USE_STRUCTURED_BACKEND=0) run was a silent no-op — every "unstructured"
            # config in a comparison sweep compiled to the byte-identical SDFG regardless of which
            # knobs were set, confirmed via .gt4py_cache: 3 differently-flagged unstructured pytest
            # invocations produced only 1 distinct compiled binary.
            #   "D"  → blocking_dim=JDim   (loop tiling on JDim) — STRUCTURED ONLY, no JDim exists
            #          on an unstructured SDFG (Edge/Cell/Vertex + K); skipped when not structured.
            #   "K"  → blocking_dim=KDim   (loop tiling on the vertical K loop)
            #   "E"  → fuse_tasklets=True  (tasklet fusion)
            #   "F"  → scan_loop_unrolling=True  (K-loop unrolling, part of the default)
            #   "G"  → reuse_transients=True
            #   "I"  → inline multi-consumer transients (E2C2V intermediate elimination)
            #   "B"  → gpu_block_size from DACE_GPU_BLOCK_SIZE (default "64,4,1")
            #   "R"  → gpu_maxnreg from DACE_GPU_MAXNREG (default "64") — register cap / occupancy
            #   "L"  → gpu_launch_factor from DACE_GPU_LAUNCH_FACTOR (default "2");
            #          clears the fixed gpu_launch_bounds so the factor takes effect
            #   "P"  → make_persistent=True (persistent transient lifetime, skip per-call alloc)
            #   "M"  → demote_fields from DACE_DEMOTE_FIELDS (comma-separated field names);
            #          only non-transient fields can be demoted — set DACE_DUMP_ARRAYS=1 to
            #          list the SDFG's transient vs non-transient array names first.
            #   "N"  → blocking_only_if_independent_nodes=False (allow blocking in more places)
            #   "none" → disable all extras (pure opt_v2 baseline)
            # Block size is shared by D/K and overridable via DACE_BLOCKING_SIZE (default 8).
            # D and K are mutually exclusive (single blocking_dim); if both are present K wins.
            # DACE_GPU_LAUNCH_BOUNDS overrides the __launch_bounds__ min_blocks argument.
            #   default "256, 8" → 32 registers/thread (heavy spilling for large kernels)
            #   "256, 2"         → 128 registers/thread (less spilling)
            #   "256, 1"         → 255 registers/thread (minimal spilling, lower occupancy)
            #   "0"              → compiler decides (no min_blocks constraint)
            _exp = _os.environ.get("DACE_OPT_EXPERIMENT", "FD")
            _blocking_size = int(_os.environ.get("DACE_BLOCKING_SIZE", "8"))
            _extra: dict = {}
            if _exp != "none":
                if "D" in _exp and _is_structured:
                    _extra["blocking_dim"] = common.Dimension("JDim")
                    _extra["blocking_size"] = _blocking_size
                if "K" in _exp:
                    _extra["blocking_dim"] = common.Dimension(
                        "K", kind=common.DimensionKind.VERTICAL
                    )
                    _extra["blocking_size"] = _blocking_size
                # DACE_BLOCKING_DIMS (e.g. "IDim,JDim") overrides D/K and enables
                # multi-dimensional blocking (2D I+J tiling). blocking_dim becomes a list;
                # LoopBlocking is applied once per dim in auto_optimize. Names: IDim/JDim
                # (HORIZONTAL, structured only) and K/KDim (VERTICAL, both backends).
                _bdims = _os.environ.get("DACE_BLOCKING_DIMS", "").strip()
                if _bdims:
                    _dim_map = {
                        "IDim": common.Dimension("IDim"),
                        "JDim": common.Dimension("JDim"),
                        "K": common.Dimension("K", kind=common.DimensionKind.VERTICAL),
                        "KDim": common.Dimension("K", kind=common.DimensionKind.VERTICAL),
                    }
                    _extra["blocking_dim"] = [
                        _dim_map[d.strip()] for d in _bdims.split(",") if d.strip()
                    ]
                    _extra["blocking_size"] = _blocking_size
                if "E" in _exp:
                    _extra["fuse_tasklets"] = True
                if "F" in _exp:
                    _extra["scan_loop_unrolling"] = True
                    _extra["scan_loop_unrolling_factor"] = 0
                if "G" in _exp:
                    _extra["reuse_transients"] = True
                if "B" in _exp:
                    _bs = _os.environ.get("DACE_GPU_BLOCK_SIZE", "64,4,1")
                    _extra["gpu_block_size"] = tuple(
                        int(x) for x in _bs.split(",")
                    )
                if "R" in _exp:
                    _extra["gpu_maxnreg"] = int(
                        _os.environ.get("DACE_GPU_MAXNREG", "64")
                    )
                if "L" in _exp:
                    _extra["gpu_launch_factor"] = int(
                        _os.environ.get("DACE_GPU_LAUNCH_FACTOR", "2")
                    )
                    # launch_bounds and launch_factor are mutually exclusive; drop the
                    # fixed bounds set below so the factor governs __launch_bounds__.
                    _extra["gpu_launch_bounds"] = None
                if "P" in _exp:
                    _extra["make_persistent"] = True
                if "M" in _exp:
                    _demote = _os.environ.get("DACE_DEMOTE_FIELDS", "").strip()
                    if _demote:
                        _extra["demote_fields"] = [
                            f.strip() for f in _demote.split(",") if f.strip()
                        ]
                if "N" in _exp:
                    _extra["blocking_only_if_independent_nodes"] = False
                if "I" in _exp:
                    _inline_multi_consumer_transients(sdfg)

            # OPT-IN K-tile thread-coarsening (DACE_KTILE=B, off by default): block the
            # vertical K dim into per-thread blocks of B levels via LoopBlocking. Each GPU
            # thread then processes B consecutive K-levels (unrolled after auto-opt) instead
            # of one, overlapping their independent loads (ILP) to hide global-load latency,
            # and hoisting K-invariant geometry/coefficients to once-per-thread. Overrides any
            # D/K-letter blocking. Default unset → no-op (suite/driver byte-for-byte unchanged).
            # K is a shared vertical dimension on both structured and unstructured SDFGs, so this
            # applies identically regardless of USE_STRUCTURED_BACKEND.
            _ktile = _os.environ.get("DACE_KTILE", "0")
            if _ktile != "0":
                _extra["blocking_dim"] = common.Dimension(
                    "K", kind=common.DimensionKind.VERTICAL
                )
                _extra["blocking_size"] = int(_ktile)
                _extra["blocking_only_if_independent_nodes"] = False

            # DACE_GPU_LAUNCH_BOUNDS controls __launch_bounds__(max_threads, min_blocks) — a pure
            # register/occupancy knob, backend-agnostic. See the docstring above for value meanings.
            _launch_bounds = _os.environ.get("DACE_GPU_LAUNCH_BOUNDS", "0")

            if _is_structured:
                # Step 1: constant folding — always safe, no structural SDFG changes.
                if constant_symbols:
                    gtx_transformations.gt_substitute_compiletime_symbols(
                        sdfg, constant_symbols, validate=False
                    )

                # Step 2: gt_auto_optimize with disable_splitting=True for all structured stencils.
                # Only the two propagate_memlets_sdfg calls in Phase 1's splitting block
                # (auto_optimize.py lines 557, 572) are unsafe for per-kolor split stencils
                # (they collapse single-element Kolor:[k,k+1) ranges → InvalidSDFGEdgeError).
                # disable_splitting=True skips exactly that block; all other phases are safe:
                #   Phase 1: MapFusionVertical/Horizontal, MapPromoter, GT4PyMapBufferElimination
                #   Phase 3: GT4PyMoveTaskletIntoMap, RemoveScalarCopies, FuseHorizontalConditionBlocks
                #   Phase 4: gt_set_iteration_order + gt_change_strides (with unit_strides_kind)
                # unit_strides_kind=VERTICAL makes K (stride-1 in [IDim,JDim,Kolor,K]) the
                # innermost CPU loop for cache-coherent memory access.
                #
                # Fallback: for a small subset of stencils, Phase 1 map fusion creates a
                # temporary with wrong dimensionality (scalar vs array mismatch).
                # Phase 3's FuseHorizontalConditionBlocks hardcodes validate=True and catches
                # this as InvalidSDFGEdgeError. We snapshot the SDFG before attempting
                # gt_auto_optimize and restore+apply a safe subset on failure.

                # ⚠️ BROKEN — DO NOT ENABLE. IDim register-tiling (DACE_ITILE=B): the intent is to
                # block the IDim (warp/unit-stride) dim so each thread does B adjacent IDim columns
                # (wider coalesced loads). In practice blocking the warp dim breaks the GPU grid
                # mapping (blockIdx.z overflows the 65535 limit → invalid launch) AND desyncs the
                # neighbour-offset / per-kolor domain bounds (98.5% wrong results). Kept gated off
                # and documented as a dead end; a correct version needs the grid assignment + shift
                # machinery reworked. See Opt.md "Tried & rejected".
                _itile = _os.environ.get("DACE_ITILE", "0")
                if _itile != "0":
                    _idim_dim = common.Dimension("IDim")
                    if _ktile != "0":
                        _extra["blocking_dim"] = [
                            common.Dimension("K", kind=common.DimensionKind.VERTICAL),
                            _idim_dim,
                        ]
                    else:
                        _extra["blocking_dim"] = _idim_dim
                        _extra["blocking_size"] = int(_itile)
                    _extra["blocking_only_if_independent_nodes"] = False

                if _os.environ.get("DACE_DUMP_ARRAYS", "0") == "1":
                    _nt = [n for n, d in sdfg.arrays.items() if not d.transient]
                    _tr = [n for n, d in sdfg.arrays.items() if d.transient]
                    print(f"[DACE_ARRAYS] non-transient ({len(_nt)}): {_nt}", flush=True)
                    print(f"[DACE_ARRAYS] transient ({len(_tr)}): {_tr}", flush=True)

                _hooks = {
                    gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowMapFusionVerticalCallBack:
                        _kolor_aware_fusion_callback,
                    gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowMapFusionHorizontalCallBack:
                        _kolor_aware_fusion_callback,
                }
                # unit_strides_dim controls ONLY the map iteration order (→ GPU grid dim
                # assignment): the listed dims are moved rightmost in reverse-priority order,
                # so the 1st becomes the x-thread/warp dim, the 2nd → threadIdx.y, the 3rd →
                # threadIdx.z, and the rest fold into blockIdx. Strides are set separately
                # (gt_change_strides, HORIZONTAL → IDim stride-1), so this does NOT touch the
                # memory layout.
                #   "IDim"              → params [JDim,Kolor,K,IDim] → x=IDim, y=K, z=(Kolor,JDim)
                #     Best for parallel-K: K in threadIdx.y/blockIdx.y → good occupancy (0.656ms)
                #   "IDim,JDim"         → params [Kolor,K,JDim,IDim] → x=IDim, y=JDim, z=(Kolor,K)
                #   "IDim,JDim,Kolor"   → params [K,Kolor,JDim,IDim] → x=IDim, y=JDim, z=Kolor
                #     Best for seqK: K stripped → Kolor=1 in z → no 87.5% Kolor waste (0.685ms)
                # Override via DACE_UNIT_STRIDES_DIMS (comma-separated dim names).
                # Default "IDim,JDim" (x=IDim warp/stride-1, y=JDim, K folds to grid.z): avoids the
                # ceil(K/8) block.y quantization of plain "IDim" (K=50→56 ⇒ ~11% wasted threads) AND
                # gives a compact IDim×JDim block footprint (~4KB contiguous plane) instead of IDim×K
                # (~53MB strided slab) ⇒ much better L2 locality. The warp axis stays IDim (stride-1,
                # set separately by gt_change_strides), so coalescing is preserved. Measured vs plain
                # "IDim" at 512/K=50, correctness intact: apply_2nd −7.1%, add_analysis_to_vn −6.0%,
                # apply_rayleigh(Cell) −6.2%, horiz_kinetic −1.3%. (The earlier "IDim" default was
                # picked for parallel-K occupancy but never benchmarked against "IDim,JDim".)
                # Use "IDim,JDim,Kolor" with DACE_SEQUENTIAL_K=1 to avoid Kolor thread waste.
                #
                # ⚠️ AUTOMATIC PER-STENCIL POLICY (K=120 full-suite measurement, Opt 22/23):
                # "IDim,JDim" only wins for K-independent, non-gather stencils. Two stencil
                # classes regress and get "IDim" (K stays on threadIdx.y) automatically:
                #   - Koff/scan stencils (vertical dependency): with K in grid.z the k±1 reads
                #     lose all cache locality → rho_virtual +41%, thermo/solver family +25-35%.
                #     Under "IDim", 8 consecutive K-levels share a block → k±1 hits L1/L2.
                #   - E2C2V/C2V gather stencils: nabla2_smag 2.35× worse, rbf_nabla4 2.8× worse
                #     (register-heavy gathers; see Opt.md Opt 21/22). E2C2EO measured neutral —
                #     intentionally NOT in the rule.
                # Detection uses the PRE-transform GTIR offsets (`_pretransform_offsets`, collected
                # at function entry) + `column_axis` (set iff the program contains a scan).
                # An explicit DACE_UNIT_STRIDES_DIMS env override still beats the automatic choice.
                _has_vertical_dep = (
                    "Koff" in _pretransform_offsets or column_axis is not None
                )
                _has_gather = bool({"E2C2V", "C2V"} & _pretransform_offsets)
                if _os.environ.get("DACE_SEQUENTIAL_K", "0") == "1":
                    _usd_default = "IDim,JDim,Kolor"
                elif _has_vertical_dep or _has_gather:
                    _usd_default = "IDim"
                    _reasons = []
                    if "Koff" in _pretransform_offsets:
                        _reasons.append("koff")
                    if column_axis is not None:
                        _reasons.append("scan")
                    if _has_gather:
                        _reasons.append("gather")
                    # Greppable audit line (same style as COPY_KERNEL_DETECTED).
                    print(
                        f"DIM_ORDER: program={sdfg.name} order=IDim reason={'+'.join(_reasons)}",
                        flush=True,
                    )
                else:
                    _usd_default = "IDim,JDim"
                _usd_names = _os.environ.get("DACE_UNIT_STRIDES_DIMS", _usd_default).strip()
                _usd_map = {
                    "IDim": common.Dimension("IDim"),
                    "JDim": common.Dimension("JDim"),
                    "Kolor": common.Dimension("Kolor"),
                    "K": common.Dimension("K", kind=common.DimensionKind.VERTICAL),
                    "KDim": common.Dimension("K", kind=common.DimensionKind.VERTICAL),
                }
                _unit_strides_dim = [
                    _usd_map[n.strip()] for n in _usd_names.split(",") if n.strip()
                ]
                # OPT-IN warp=K (DACE_K_WARP=1, off by default): make K the warp/threadIdx.x
                # dim so the 32 K-levels of one (i,j,kolor) form a warp — K-dependent field
                # loads coalesce and the K-invariant geometry/coefficients become uniform
                # (broadcast) loads, read once per warp instead of once per K (the full version
                # of the K-tile's geometry hoisting). Requires C-order (K-unit-stride) packing
                # (_make_structured_field under the same flag) + VERTICAL unit strides so the
                # SDFG iteration order matches. Overrides DACE_UNIT_STRIDES_DIMS.
                _k_warp = _os.environ.get("DACE_K_WARP", "0") != "0"
                if _k_warp:
                    _unit_strides_dim = [
                        _usd_map[n] for n in ("K", "IDim", "JDim", "Kolor")
                    ]
                # DACE_GPU_LAUNCH_BOUNDS meaning (see the unconditional computation above):
                #   "256, 8"  (old default) → 8 blocks/SM → 32 registers/thread → heavy spilling
                #   "256, 3"               → 3 blocks/SM → ~85 registers/thread
                #   "256, 2"               → 2 blocks/SM → 128 registers/thread → moderate spilling
                #   "256, 1"               → 1 block/SM  → 255 registers/thread → minimal spilling
                #   "0"  (DEFAULT)         → __launch_bounds__(256) only → compiler decides regs
                # DEFAULT "0": the old fixed "256, 8" cap (32 regs) forced heavy register spilling on
                # the register-hungry elementwise/scan stencils (the worst structured performers).
                # Letting the compiler choose eliminates the spill — measured wins: rho_virtual
                # 1.063→0.91 ms (−14%), solve_tridiagonal_w_fwd 0.92→0.56 ms (−39%), both now BEAT
                # unstructured; neutral where there was no spill (divergence, horiz_kinetic_energy).
                # EXCEPTION: register-heavy *gather* kernels (rbf_nabla4 / *_direct) want an explicit
                # cap — the compiler over-allocates regs → 1 block/SM → low occupancy. Override those
                # with DACE_GPU_LAUNCH_BOUNDS="256, 3" (post-baking optimum; see Opt.md Opt 18).
                structured_opt_args: dict = {
                    "unit_strides_dim": _unit_strides_dim,
                    "disable_splitting": True,
                    "validate": False,
                    "optimization_hooks": _hooks,
                    "gpu_launch_bounds": _launch_bounds,
                    **_extra,
                    **(auto_optimize_args or {}),
                }
                if _k_warp:
                    # Treat the vertical K as the unit-stride dim so gt_change_strides /
                    # iteration order agree with the C-order (K stride-1) packing.
                    structured_opt_args["unit_strides_kind"] = (
                        common.DimensionKind.VERTICAL
                    )
                sdfg_snapshot = sdfg.to_json()
                try:
                    gtx_transformations.gt_auto_optimize(
                        sdfg,
                        gpu=on_gpu,
                        constant_symbols=None,  # already substituted above
                        **structured_opt_args,
                    )
                except Exception:
                    # Restore SDFG from snapshot and apply safe fallback:
                    # buffer elimination + iteration order (no map fusion).
                    sdfg = dace.SDFG.from_json(sdfg_snapshot)
                    sdfg.apply_transformations_repeated(
                        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=True),
                        validate=False,
                        validate_all=False,
                    )
                    gtx_transformations.gt_set_iteration_order(
                        sdfg,
                        unit_strides_dim=_unit_strides_dim,
                        validate=False,
                    )
                    if on_gpu:
                        gtx_transformations.gt_gpu_transformation(
                            sdfg, try_removing_trivial_maps=True,
                            gpu_launch_bounds=_launch_bounds,
                        )
                # K-tile (DACE_KTILE=B): unroll the inner Sequential K-block maps that
                # LoopBlocking produced, so the B per-thread K-levels' independent loads overlap.
                if on_gpu and _ktile != "0":
                    _n_ktile = _unroll_inner_k_maps(sdfg, int(_ktile))
                    print(
                        f"[ktile] B={_ktile}, unrolled {_n_ktile} inner-K map(s)", flush=True
                    )
                if on_gpu and _itile != "0":
                    _n_itile = _unroll_inner_k_maps(
                        sdfg, int(_itile), map_param=_IDIM_MAP_PARAM
                    )
                    print(
                        f"[itile] B={_itile}, unrolled {_n_itile} inner-IDim map(s)",
                        flush=True,
                    )
                # OPT-IN (GT4PY_ENABLE_MISR=1, off by default): merge duplicate reads from the
                # same global array at the same offset into one shared scalar transient. This
                # deduplicates SDFG-level reads, but DaCe codegen re-materialises the shared
                # scalars so the final local-variable count is unchanged — measured slightly
                # SLOWER (0.674 vs 0.656 ms) on rbf_nabla4. Kept gated for experimentation.
                if on_gpu and _os.environ.get("GT4PY_ENABLE_MISR", "0") == "1":
                    _n_misr = gtx_transformations.gt_merge_identical_scalar_reads(sdfg)
                    if _n_misr > 0:
                        print(f"[misr] merged {_n_misr} redundant global reads", flush=True)

                # OPT-IN 2.5D vertical loop: strip K out of the GPU grid into a nested
                # sequential per-thread K-loop (see _sequentialize_k_dimension). Targets the
                # redundant per-K address arithmetic (top ALU/ADU pipe) + geometry re-fetch.
                if on_gpu and _os.environ.get("DACE_SEQUENTIAL_K", "0") == "1":
                    # if _os.environ.get("DACE_SEQK_DEBUG", "0") == "1":
                    #     sdfg.save("before_seqk.sdfg")
                    # _n_kseq = _sequentialize_k_dimension(sdfg)
                    # print(f"[seqK] sequentialized K in {_n_kseq} GPU map(s)", flush=True)
                    # if _os.environ.get("DACE_SEQK_DEBUG", "0") == "1":
                    #     sdfg.save("after_seqk.sdfg")
                    # LICM: hoist K-invariant global reads (primal_normal, ptr_coeff, inv_*)
                    # out of the Sequential K-loop into the outer GPU scope so they are read
                    # once per thread instead of once per K-iteration (50× reduction).
                    # Set DACE_LICM_DISABLE=1 to skip LICM (useful to test seqK alone).
                    if _os.environ.get("DACE_LICM_DISABLE", "0") != "1":
                        _n_hoist = gtx_transformations.gt_hoist_k_invariant_reads(sdfg)
                        if _n_hoist > 0:
                            print(f"[seqK] hoisted {_n_hoist} K-invariant reads", flush=True)
            else:
                # Unstructured path: forward the same backend-agnostic knobs computed above
                # (_extra from DACE_OPT_EXPERIMENT sans "D", DACE_GPU_LAUNCH_BOUNDS) — these are
                # generic gt_auto_optimize parameters, not specific to the structured IDim/JDim/
                # Kolor layout. Previously this branch ignored all of them entirely (see the
                # comment on the unconditional computation above).
                #
                # Only forward when EXPLICITLY set via env var — not just whenever _exp/
                # _launch_bounds hold their defaults ("FD"/"0") — so an unstructured run with no
                # env vars set behaves byte-for-byte as before (gpu_launch_bounds=None, no
                # scan_loop_unrolling, etc.). Forwarding the *default* "FD"/"0" unconditionally
                # would have silently changed every existing unstructured stencil's compiled
                # kernel (e.g. gpu_launch_bounds="0" emits an explicit __launch_bounds__(256),
                # which is NOT the same as never emitting __launch_bounds__ at all).
                unstructured_opt_args: dict = dict(auto_optimize_args)
                if "DACE_OPT_EXPERIMENT" in _os.environ:
                    unstructured_opt_args.update(_extra)
                if "DACE_GPU_LAUNCH_BOUNDS" in _os.environ:
                    unstructured_opt_args["gpu_launch_bounds"] = _launch_bounds
                gtx_transformations.gt_auto_optimize(
                    sdfg,
                    gpu=on_gpu,
                    constant_symbols=constant_symbols,
                    **unstructured_opt_args,
                )
                # K-tile (DACE_KTILE=B): unroll the inner Sequential K-block maps, same
                # post-pass as the structured path — see _unroll_inner_k_maps' docstring.
                # _ktile is already "0" (no-op) unless explicitly set to a nonzero value, so no
                # separate explicit-set check is needed here.
                if on_gpu and _ktile != "0":
                    _n_ktile = _unroll_inner_k_maps(sdfg, int(_ktile))
                    print(
                        f"[ktile] (unstructured) B={_ktile}, unrolled {_n_ktile} inner-K map(s)",
                        flush=True,
                    )
        elif on_gpu:
            # Note that `gt_substitute_compiletime_symbols()` will run `gt_simplify()`
            # at entry, in order to avoid some issue in constant propagatation.
            # Besides, `gt_simplify()` will bring the SDFG into a canonical form
            # that the GPU transformations can handle. This is a workaround for
            # an issue with scalar expressions that are promoted to symbolic expressions
            # and computed on the host (CPU), but the intermediate result is written
            # to a GPU global variable (https://github.com/spcl/dace/issues/1773).
            gtx_transformations.gt_substitute_compiletime_symbols(
                sdfg, constant_symbols, validate=True
            )
            gtx_transformations.gt_gpu_transformation(sdfg, try_removing_trivial_maps=True)

        elif len(constant_symbols) != 0:
            # Target CPU without SDFG transformations, but still replace constant symbols.
            # Replacing the SDFG symbols for field origin in global arrays is strictly
            # required by dace orchestration, which runs the translation stage
            # with `disable_field_origin_on_program_arguments=True`. The program
            # decorator used by dace orchestartion cannot handle field origin.
            # It also requires skipping auto-optimize on the `SDFGConvertible` objects,
            # because it targets the full-application SDFG, so we have to explicitly
            # apply `gt_substitute_compiletime_symbols()` here.
            gtx_transformations.gt_substitute_compiletime_symbols(
                sdfg, constant_symbols, validate=True
            )
        # sdfg.save("after_gpu_transformation.sdfg")
        if self.async_sdfg_call:
            make_sdfg_call_async(sdfg, on_gpu)
        else:
            make_sdfg_call_sync(sdfg, on_gpu)

        if self.use_metrics:
            add_instrumentation(sdfg, on_gpu)
            if on_gpu:
                # Opt-in (DACE_TIME_COMPUTE_ONLY=1): time the copy_* kernels so the reported
                # metric can be made compute-only (decoration.py subtracts their duration).
                _instrument_copy_kernels_for_compute_only(sdfg)

        # Report any pure data-copy kernels (e.g. multi-output stencils → 1 compute + N copy
        # kernels) so a suite run can be grepped for COPY_KERNEL_DETECTED. Detection only; does
        # not change the SDFG.
        _report_copy_kernels(sdfg)

        return sdfg

    def __call__(
        self, inp: definitions.CompilableProgramDef
    ) -> stages.ProgramSource[code_specs.SDFGCodeSpec]:
        """Generate DaCe SDFG file from the GTIR definition."""
        program: itir.Program = inp.data
        assert isinstance(program, itir.Program)

        sdfg = self.generate_sdfg(
            program,
            inp.args.offset_provider,  # TODO(havogt): should be offset_provider_type once the transformation don't require run-time info
            inp.args.column_axis,
        )

        import os as _os
        if _os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
            # apply_fieldview_transforms remaps param types inside StructuredTypeRemapper,
            # but that transformation is on a NEW program object not accessible here.
            # Re-apply _cartesian_remapped_type to each arg type so the bindings
            # match the SDFG (which was built with the structured remapped types).
            from gt4py.next.iterator.transforms.structured_backend_passes import (
                StructuredTypeRemapper as _STR,
            )
            arg_types = tuple(
                _STR._cartesian_remapped_type(t) for t in inp.args.args
            )
        else:
            arg_types = inp.args.args

        if self.symbolic_domain_sizes:
            # After StructuredTypeRemapper runs, the SDFG has IDim/JDim/Kolor-shaped
            # arrays, but inp.args.args still carries the original unstructured types.
            # Remap them so the bindings generator sees the correct dimensions.
            from gt4py.next.iterator.transforms.structured_backend_passes import (
                StructuredTypeRemapper,
            )
            arg_types = tuple(
                StructuredTypeRemapper._cartesian_remapped_type(t) or t for t in arg_types
            )

        program_parameters = tuple(
            interface.Parameter(param.id, arg_type)
            for param, arg_type in zip(program.params, arg_types)
        )

        module: stages.ProgramSource[code_specs.SDFGCodeSpec] = stages.ProgramSource(
            entry_point=interface.Function(program.id, program_parameters),
            # Set 'hash=True' to compute the SDFG hash and store it in the JSON.
            #   We compute the hash in order to refresh `cfg_list` on the SDFG,
            #   which makes the JSON serialization stable.
            source_code=sdfg.to_json(hash=True),
            library_deps=tuple(),
            code_spec=code_specs.SDFGCodeSpec(),
        )
        return module


class DaCeTranslationStepFactory(factory.Factory):
    class Meta:
        model = DaCeTranslator

    # Required fields; device_type and auto_optimize are forwarded by DaCeWorkflowFactory.
    # All others need defaults here so that DaCeBackendFactory can be called with only
    # a subset of kwargs (e.g. symbolic_domain_sizes) without TypeError.
    apply_common_transform = False
    auto_optimize_args = None
    async_sdfg_call = False
    unstructured_horizontal_has_unit_stride = False
    use_metrics = True
    symbolic_domain_sizes = None
