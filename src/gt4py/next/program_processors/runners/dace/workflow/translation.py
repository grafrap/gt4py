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


# DaCe SDFG map parameter name for the Kolor dimension.
# Computed once using the same function the SDFG lowering uses — guaranteed to match actual params.
_KOLOR_MAP_PARAM: str = gtx_dace_lowering.get_map_variable(common.Dimension("Kolor"))

# DaCe SDFG map parameter name for the vertical K dimension.
_K_MAP_PARAM: str = gtx_dace_lowering.get_map_variable(
    common.Dimension("K", kind=common.DimensionKind.VERTICAL)
)


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


def _kolor_aware_fusion_callback(
    self: Any,
    first_map_node: Union[dace.nodes.MapExit, dace.nodes.MapEntry],
    second_map_entry: dace.nodes.MapEntry,
    graph: dace.SDFGState,
    sdfg: dace.SDFG,
) -> bool:
    """Refuse to fuse maps with different single-element Kolor ranges.

    Per-kolor split stencils produce separate maps per Kolor value (Kolor:[k,k+1)).
    Fusing two such maps with different k values creates a dimensionality mismatch
    in the fusion temporary (InvalidSDFGEdgeError). Fusion is allowed when either
    map has no Kolor parameter, both share the same Kolor start index, or the range
    is symbolic and cannot be evaluated at transform time.
    """
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
        # get sdfg here sdfg.save()
        # sdfg.save("before_gpu_transformation.sdfg", compress=True)
        constant_symbols = find_constant_symbols(
            ir,
            sdfg,
            offset_provider_type,
            self.disable_field_origin_on_program_arguments,
            self.unstructured_horizontal_has_unit_stride,
        )

        if self.auto_optimize:
            auto_optimize_args = {} if self.auto_optimize_args is None else self.auto_optimize_args

            if _os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
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
                # Experiment selector: set DACE_OPT_EXPERIMENT env var to override defaults:
                #   "D"  → blocking_dim=JDim   (loop tiling on JDim)
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
                    if "D" in _exp:
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
                    # (HORIZONTAL) and K/KDim (VERTICAL).
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
                # so the 1st becomes the x-thread/warp dim, the 2nd → blockIdx.y, and the rest
                # fold into blockIdx.z. Strides are set separately (gt_change_strides, HORIZONTAL
                # → IDim stride-1), so this does NOT touch the memory layout.
                #   default "IDim"       → params [JDim,Kolor,K,IDim] → x=IDim, y=K, z=(Kolor,JDim)
                #   "IDim,JDim"          → params [Kolor,K,JDim,IDim] → x=IDim, y=JDim, z=(Kolor,K)
                # Override via DACE_UNIT_STRIDES_DIMS (comma-separated dim names).
                _usd_names = _os.environ.get("DACE_UNIT_STRIDES_DIMS", "IDim").strip()
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
                # DACE_GPU_LAUNCH_BOUNDS controls __launch_bounds__(max_threads, min_blocks).
                # The second argument (min_blocks) limits registers per thread on the GPU:
                #   "256, 8"  (old default) → 8 blocks/SM → 32 registers/thread → heavy spilling
                #   "256, 2"               → 2 blocks/SM → 128 registers/thread → moderate spilling
                #   "256, 1"               → 1 block/SM  → 255 registers/thread → minimal spilling
                #   "0"                    → __launch_bounds__(256) only → compiler decides
                # Use "0" for register-pressure-heavy stencils (e.g. rbf_nabla4_direct with 237 tmps).
                _launch_bounds = _os.environ.get("DACE_GPU_LAUNCH_BOUNDS", "256, 8")
                structured_opt_args: dict = {
                    "unit_strides_dim": _unit_strides_dim,
                    "disable_splitting": True,
                    "validate": False,
                    "optimization_hooks": _hooks,
                    "gpu_launch_bounds": _launch_bounds,
                    **_extra,
                    **(auto_optimize_args or {}),
                }
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
                # OPT-IN 2.5D vertical loop: strip K out of the GPU grid into a nested
                # sequential per-thread K-loop (see _sequentialize_k_dimension). Targets the
                # redundant per-K address arithmetic (top ALU/ADU pipe) + geometry re-fetch.
                if on_gpu and _os.environ.get("DACE_SEQUENTIAL_K", "0") == "1":
                    _n_kseq = _sequentialize_k_dimension(sdfg)
                    print(f"[seqK] sequentialized K in {_n_kseq} GPU map(s)", flush=True)
            else:
                gtx_transformations.gt_auto_optimize(
                    sdfg,
                    gpu=on_gpu,
                    constant_symbols=constant_symbols,
                    **auto_optimize_args,
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
