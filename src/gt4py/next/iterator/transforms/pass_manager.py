# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from typing import Optional, Protocol, cast

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir, pretty_printer
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import (
    cart_unroll,
    concat_where,
    dead_code_elimination,
    fuse_as_fieldop,
    global_tmps,
    infer_domain,
    infer_domain_ops,
    inline_dynamic_shifts,
    inline_fundefs,
    inline_lifts,
    prune_empty_concat_where,
    remove_broadcast,
    simplify_cart_shifts,
    symbol_ref_utils,
)
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.inline_scalar import InlineScalar
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce
from gt4py.next.iterator.type_system.inference import infer


class GTIRTransform(Protocol):
    def __call__(
        self, _: itir.Program, *, offset_provider: common.OffsetProvider
    ) -> itir.Program: ...


class _FieldviewDebugStats(eve.NodeVisitor):
    def __init__(self) -> None:
        self.cartesian_reduce_nodes: list[itir.FunCall] = []
        self.nested_as_fieldop_nodes: list[itir.FunCall] = []
        self.named_range_nodes: list[itir.FunCall] = []
        self.named_range_arg_debug: list[tuple[str, str, str]] = []
        self.named_range_count = 0

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        if cpm.is_call_to(node, "cartesian_reduce"):
            self.cartesian_reduce_nodes.append(node)
        if cpm.is_call_to(node, "named_range"):
            self.named_range_count += 1
            self.named_range_nodes.append(node)
            axis, start, stop = node.args
            self.named_range_arg_debug.append(
                (
                    f"{type(axis).__name__}:{getattr(axis, 'value', None)}",
                    f"{type(start).__name__}:{getattr(start, 'value', getattr(start, 'name', None))}",
                    f"{type(stop).__name__}:{getattr(stop, 'value', getattr(stop, 'name', None))}",
                )
            )

        if cpm.is_applied_as_fieldop(node):
            if any(
                cpm.is_applied_as_fieldop(arg) for arg in node.args if isinstance(arg, itir.FunCall)
            ):
                self.nested_as_fieldop_nodes.append(node)

        self.generic_visit(node, **kwargs)


def _write_debug_output(text: str, *, stream=None) -> None:
    global _DEBUG_FILE_INITIALIZED

    if stream is not None:
        print(text, file=stream, end="")

    debug_file = "ir_out.txt"  # os.environ.get("GT4PY_PRINT_IR_FILE")
    if debug_file:
        mode = "a" if _DEBUG_FILE_INITIALIZED else "w"
        with open(debug_file, mode, encoding="utf-8") as output:
            output.write(text)
        _DEBUG_FILE_INITIALIZED = True


_DEBUG_FILE_INITIALIZED = False


def _print_ir_block(title: str, ir: itir.Program, *, enabled: bool) -> None:
    if not enabled:
        return

    text = "\n" + "=" * 60 + "\n" + title + "\n" + "=" * 60 + "\n" + f"{ir}\n" + "=" * 60 + "\n\n"
    _write_debug_output(text)


def _debug_dump_fieldview_ir(stage: str, ir: itir.Program) -> None:
    if not os.environ.get("GT4PY_DEBUG_FIELDVIEW_IR"):
        return

    stats = _FieldviewDebugStats()
    stats.visit(ir)

    _write_debug_output(
        f"[GT4PY_DEBUG_FIELDVIEW_IR] stage={stage} "
        f"cartesian_reduce_calls={len(stats.cartesian_reduce_nodes)} "
        f"nested_as_fieldop_calls={len(stats.nested_as_fieldop_nodes)} "
        f"named_range_calls={stats.named_range_count}\n",
        stream=sys.stderr,
    )

    if stats.cartesian_reduce_nodes:
        _write_debug_output(
            "[GT4PY_DEBUG_FIELDVIEW_IR] unresolved cartesian_reduce snippets:\n",
            stream=sys.stderr,
        )
        for node in stats.cartesian_reduce_nodes[:5]:
            _write_debug_output(pretty_printer.pformat(node) + "\n", stream=sys.stderr)

    if stats.nested_as_fieldop_nodes:
        _write_debug_output(
            "[GT4PY_DEBUG_FIELDVIEW_IR] nested as_fieldop snippets:\n",
            stream=sys.stderr,
        )
        for node in stats.nested_as_fieldop_nodes[:5]:
            _write_debug_output(pretty_printer.pformat(node) + "\n", stream=sys.stderr)

    if stats.named_range_nodes:
        _write_debug_output("[GT4PY_DEBUG_FIELDVIEW_IR] named_range snippets:\n", stream=sys.stderr)
        for node in stats.named_range_nodes[:5]:
            _write_debug_output(pretty_printer.pformat(node) + "\n", stream=sys.stderr)
        _write_debug_output(
            "[GT4PY_DEBUG_FIELDVIEW_IR] named_range arg types:\n", stream=sys.stderr
        )
        for axis_dbg, start_dbg, stop_dbg in stats.named_range_arg_debug[:5]:
            _write_debug_output(
                f"axis={axis_dbg} start={start_dbg} stop={stop_dbg}\n", stream=sys.stderr
            )

    if os.environ.get("GT4PY_DEBUG_FIELDVIEW_IR_FULL"):
        _write_debug_output(
            "[GT4PY_DEBUG_FIELDVIEW_IR] full pre-infer-domain IR:\n", stream=sys.stderr
        )
        _write_debug_output(pretty_printer.pformat(ir) + "\n", stream=sys.stderr)


def _apply_unroll_reduce_pipeline(
    ir: itir.Program,
    *,
    offset_provider_type: common.OffsetProviderType,
    uids: utils.IDGeneratorPool,
    use_offset_literal_index: bool = True,
) -> itir.Program:
    for _ in range(10):
        try:
            unrolled = cast(
                itir.Program,
                UnrollReduce.apply(
                    ir,
                    offset_provider_type=offset_provider_type,
                    uids=uids,
                    use_offset_literal_index=use_offset_literal_index,
                ),
            )
            unrolled = cast(itir.Program, CollapseListGet().visit(unrolled))
            unrolled = cast(itir.Program, NormalizeShifts().visit(unrolled))
            # this is required as nested neighbor reductions can contain lifts, e.g.,
            # `neighbors(V2Eₒ, ↑f(...))`
            unrolled = cast(itir.Program, inline_lifts.InlineLifts().visit(unrolled))
            unrolled = cast(itir.Program, NormalizeShifts().visit(unrolled))
        except Exception as e:
            raise RuntimeError("Failed inside _apply_unroll_reduce_pipeline") from e
        if unrolled == ir:
            break
        ir = unrolled
    else:
        raise RuntimeError("Reduction unrolling failed.")
    return ir


def _max_domain_range_sizes(offset_provider: common.OffsetProvider) -> dict[str, itir.Literal]:
    """
    Extract horizontal domain sizes from an `offset_provider`.

    Considers the shape of the neighbor table to get the size of each `source_dim` and the maximum
    value inside the neighbor table to get the size of each `codomain`.
    """
    sizes: dict[str, int] = {}
    for provider in offset_provider.values():
        if common.is_neighbor_connectivity(provider):
            src_dim = provider.__gt_type__().source_dim.value
            codomain_dim = provider.__gt_type__().codomain.value
            sizes[src_dim] = max(sizes.get(src_dim, 0), provider.ndarray.shape[0])
            sizes[codomain_dim] = max(
                sizes.get(codomain_dim, 0),
                int(provider.ndarray.max()) + 1,  # type: ignore[attr-defined] # TODO(havogt): improve typing for NDArrayObject
            )

    sizes_exprs = {k: im.literal_from_value(v) for k, v in sizes.items()}
    return sizes_exprs


def _has_dynamic_domains(ir: itir.Program) -> bool:
    # note: this function does not respect symbol collisions with builtins. As it is a temporary
    # workaround we don't care about this corner case.
    domains = set()
    domains |= ir.walk_values().if_isinstance(itir.SetAt).getattr("domain").to_set()
    for as_fop in (
        ir.walk_values()
        .if_isinstance(itir.FunCall)
        .filter(lambda node: cpm.is_call_to(node, "as_fieldop") and len(node.args) == 2)
    ):
        domains.add(as_fop.args[1])
    return len(symbol_ref_utils.collect_symbol_refs(domains)) > 0


def _process_symbolic_domains_option(
    ir: itir.Program,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]],
    use_max_domain_range_on_unstructured_shift: Optional[bool],
) -> Optional[dict[str, itir.Expr]]:
    """
    Given a program, offset_provider and some configuration options determine how domains are
    inferred.

    The output of this function is used as `symbolic_domain_sizes` argument of domain inference, i.e.
    :func:`infer_domain.infer_program`.

    Right now domains of `as_fieldop` expressions can be inferred either a) using static information
    from the offset provider, or b) they are set to an expression controlled by
    the user and configured in the backend, or c) they are set to the maximum possible domain /
    everywhere (see :func:`_max_domain_range_sizes`)

    Option a) applies when the program is decorated with `static_domains = True` (unless option c)
    is explicitly requested). Then all dynamic domains were replaced with static ones
    which we recognize here. The domain inference then uses this static information which we
    communicate by returning `None`, i.e. no symbolic domain sizes.
    Option b) applies when the user explicitly configured `symbolic_domain_sizes` in the backend.
    In that case we just forward the value.
    Option c) applies when `static_domains = False` or when explicitly configured in the backend
    with `use_max_domain_range_on_unstructured_shift = True`. In that case we determine the
    maximum sizes using :func:`_max_domain_range_sizes` and return them.
    """
    if symbolic_domain_sizes:
        assert not use_max_domain_range_on_unstructured_shift, "Options are mutually exclusive."
        return symbolic_domain_sizes

    if use_max_domain_range_on_unstructured_shift is None:
        use_max_domain_range_on_unstructured_shift = _has_dynamic_domains(ir)
    if use_max_domain_range_on_unstructured_shift:
        # TODO(havogt): ICON4Py uses this codepath as default for now. Once we use the minimal domain range, we should re-enable this warning.
        # if not _has_dynamic_domains(ir):
        #     warnings.warn(
        #         "You are using static domains together with "
        #         "'use_max_domain_range_on_unstructured_shift'. This is "
        #         "likely not what you wanted.",
        #         stacklevel=2,
        #     )  # noqa: ERA001, RUF100
        assert not symbolic_domain_sizes, "Options are mutually exclusive."
        symbolic_domain_sizes = _max_domain_range_sizes(offset_provider)  # type: ignore[assignment]
    return symbolic_domain_sizes


# TODO(tehrengruber): Revisit interface to configure temporary extraction. We currently forward
#  `extract_temporaries` and `temporary_extraction_heuristics` which is inconvenient.
def apply_common_transforms(
    ir: itir.Program,
    *,
    offset_provider: common.OffsetProvider | common.OffsetProviderType,
    extract_temporaries=False,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    #: A dictionary mapping axes names to their length. See :func:`infer_domain.infer_expr` for
    #: more details.
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]] = None,
    # TODO(tehrengruber): Remove this option again as soon as we have the necessary builtins
    #  to work with / translate domains.
    use_max_domain_range_on_unstructured_shift: Optional[bool] = None,
    cartesian_reduce_axis_ranges: Optional[dict[common.Dimension, tuple[int, int]]] = None,
    transform_concat_where_to_as_fieldop: bool = True,
    enable_structured_backend_transforms: bool = False,
) -> itir.Program:
    assert isinstance(ir, itir.Program)
    # TODO(tehrengruber): Allow `common.OffsetProviderType`, but domain inference currently
    #  relies on static information or `symbolic_domain_sizes`.
    assert common.is_offset_provider(offset_provider)

    offset_provider_type = common.offset_provider_to_type(offset_provider)
    print_ir = True
    _print_ir_block("=== FINAL GTIR HANDED TO GTFN BACKEND ===", ir, enabled=print_ir)

    symbolic_domain_sizes = _process_symbolic_domains_option(
        ir, offset_provider, symbolic_domain_sizes, use_max_domain_range_on_unstructured_shift
    )

    uids = utils.IDGeneratorPool()
    ir = MergeLet().visit(ir)
    ir = inline_fundefs.InlineFundefs().visit(ir)
    _print_ir_block("=== GTIR AFTER INLINING FUNDEFS ===", ir, enabled=print_ir)

    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = NormalizeShifts().visit(ir)
    _print_ir_block(
        "=== GTIR AFTER PRUNING UNREFERENCED FUNDEFS AND NORMALIZING SHIFTS ===",
        ir,
        enabled=print_ir,
    )

    # TODO(tehrengruber): Many iterator test contain lifts that need to be inlined, e.g.
    #  test_can_deref. We didn't notice previously as FieldOpFusion did this implicitly everywhere.
    ir = inline_lifts.InlineLifts().visit(ir)

    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
    ir = dead_code_elimination.dead_code_elimination(
        ir, uids=uids, offset_provider_type=offset_provider_type
    )  # domain inference does not support dead-code
    _print_ir_block("=== GTIR AFTER DEAD CODE ELIMINATION ===", ir, enabled=print_ir)
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )  # domain inference does not support dynamic offsets yet

    # ir = cart_unroll.CartUnroll.apply(ir, symbolic_domain_sizes=symbolic_domain_sizes)
    if enable_structured_backend_transforms and os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
        ir = cart_unroll.CartesianDomainAndTypeRemapper.apply(  # type: ignore[assignment]
            ir,
            symbolic_domain_sizes=cast(dict[str, str | int] | None, symbolic_domain_sizes),
            offset_provider=offset_provider,
        )
        _print_ir_block(
            "=== GTIR AFTER CARTESIAN DOMAIN AND TYPE REMAPPING ===", ir, enabled=print_ir
        )
        ir = cart_unroll.CartesianReductionUnroller.apply(ir)  # type: ignore[assignment]
        ir = NormalizeShifts().visit(ir)
        _print_ir_block("=== GTIR AFTER CARTESIAN UNROLLING ===", ir, enabled=print_ir)
        # ir = cart_unroll.KolorConstantPropagation.apply(ir)  # type: ignore[assignment]
        # _print_ir_block("=== GTIR AFTER KOLOR CONSTANT PROPAGATION ===", ir, enabled=print_ir)

    ir = infer_domain_ops.InferDomainOps.apply(ir)
    ir = concat_where.canonicalize_domain_argument(ir)
    _print_ir_block("=== GTIR AFTER CANONICALIZING DOMAIN ARGUMENTS ===", ir, enabled=print_ir)
    # if cartesian_reduce_axis_ranges is None:
    #     cartesian_reduce_axis_ranges = {common.Dimension("Kolor"): (0, 3)}
    # ir = UnrollCartesianReduce.apply(ir, axis_ranges=cartesian_reduce_axis_ranges)
    # _print_ir_block("=== GTIR AFTER UNROLLING CARTESIAN REDUCE ===", ir, enabled=print_ir)
    ir = infer_domain.infer_program(
        ir,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        allow_uninferred=True,
    )
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    ir = remove_broadcast.RemoveBroadcast.apply(ir)
    ir = cast(itir.Program, ConstantFolding.apply(ir))
    # After ConstantFolding some K ranges may now be provably empty (e.g. when vertical_end
    # is small); prune those dead concat_where branches so they don't inflate the IR.
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    _print_ir_block(
        "=== GTIR AFTER COMMON TRANSFORMS BEFORE INFER_DOMAIN ===", ir, enabled=print_ir
    )
    ir = concat_where.transform_to_as_fieldop(ir)
    _print_ir_block("=== GTIR AFTER TRANSFORM AS FIELDOP ===", ir, enabled=print_ir)
    for _ in range(10):
        inlined = ir

        inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
        inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]  # always an itir.Program
        # This pass is required to be in the loop such that when an `if_` call with tuple arguments
        # is constant-folded the surrounding tuple_get calls can be removed.
        inlined = CollapseTuple.apply(
            inlined,
            enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
            uids=uids,
            offset_provider_type=offset_provider_type,
        )  # type: ignore[assignment]  # always an itir.Program
        inlined = InlineScalar.apply(inlined, offset_provider_type=offset_provider_type)
        if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
            inlined = simplify_cart_shifts.SimplifyCartesianShifts.apply(inlined)  # type: ignore[assignment]

        # This pass is required to run after CollapseTuple as otherwise we can not inline
        # expressions like `tuple_get(make_tuple(as_fieldop(stencil)(...)))` where stencil returns
        # a list. Such expressions must be inlined however because no backend supports such
        # field operators right now.
        try:
            inlined = fuse_as_fieldop.FuseAsFieldOp.apply(
                inlined, uids=uids, offset_provider_type=offset_provider_type
            )
        except Exception:
            pass

        # FuseAsFieldOp intersects as_fieldop domains which introduces nested maximum/minimum
        # expressions in the K bounds (e.g. maximum(maximum(X, Y), Y)). Run ConstantFolding
        # immediately after to algebraically simplify them before the next fusion round, otherwise
        # the nesting grows unboundedly over iterations.
        inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

    _print_ir_block("=== GTIR AFTER INLINING LIFTS AND LAMBDAS ===", ir, enabled=print_ir)
    # breaks in test_zero_dim_tuple_arg as trivial tuple_get is not inlined
    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination.apply(
            ir, offset_provider_type=offset_provider_type, uids=uids
        )
        ir = MergeLet().visit(ir)
        ir = InlineLambdas.apply(ir, opcount_preserving=True)

    if extract_temporaries:
        ir = infer(ir, inplace=True, offset_provider_type=offset_provider_type)
        ir = global_tmps.create_global_tmps(
            ir,
            offset_provider=offset_provider,
            symbolic_domain_sizes=symbolic_domain_sizes,
            uids=uids,
        )

    _print_ir_block("=== GTIR AFTER COMMON TRANSFORMS ===", ir, enabled=print_ir)

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps(uids=uids).visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        ir = _apply_unroll_reduce_pipeline(
            ir,
            offset_provider_type=offset_provider_type,
            uids=uids,
        )

    _print_ir_block("=== GTIR AFTER UNROLLING REDUCE ===", ir, enabled=print_ir)

    ir = InlineLambdas.apply(
        ir, opcount_preserving=True, force_inline_lambda_args=force_inline_lambda_args
    )
    ir = NormalizeShifts().visit(ir)
    _print_ir_block("=== GTIR END ===", ir, enabled=print_ir)

    assert isinstance(ir, itir.Program)
    return ir


def apply_fieldview_transforms(
    ir: itir.Program,
    *,
    offset_provider: common.OffsetProvider,
    unroll_reduce: bool = False,
    cartesian_reduce_axis_ranges: Optional[dict[common.Dimension, tuple[int, int]]] = None,
    use_max_domain_range_on_unstructured_shift: Optional[bool] = None,
    symbolic_domain_sizes: Optional[dict[str, str | int]] = None,
) -> itir.Program:
    """Minimal-diff variant of the grafrap_dace `apply_fieldview_transforms`.

    Restored to the original 12-pass pipeline, plus exactly one structured-backend
    block (CartesianDomainAndTypeRemapper + CartesianReductionUnroller, gated on
    USE_STRUCTURED_BACKEND=1) inserted between `inline_dynamic_shifts` and
    `InferDomainOps`. All experimental gates / fusion loops / extra `infer_program`
    calls / domain-attach patches removed. Per-pass IR prints kept for debugging.
    """
    offset_provider_type = common.offset_provider_to_type(offset_provider)

    uids = utils.IDGeneratorPool()
    _print_ir_block("=== FIELDVIEW IR BEFORE TRANSFORMS ===", ir, enabled=True)
    symbolic_domain_sizes = _process_symbolic_domains_option(
        ir,
        offset_provider,
        cast(Optional[dict[str, itir.Expr]], symbolic_domain_sizes),
        use_max_domain_range_on_unstructured_shift,
    )
    _print_ir_block(
        "=== FIELDVIEW IR AFTER PROCESSING DOMAIN OPTIONS ===", ir, enabled=True
    )

    ir = inline_fundefs.InlineFundefs().visit(ir)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    _print_ir_block("=== FIELDVIEW IR AFTER INLINING FUNDEFS ===", ir, enabled=True)

    # required for dead-code-elimination and `prune_empty_concat_where` pass
    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program

    ir = dead_code_elimination.dead_code_elimination(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )
    _print_ir_block("=== FIELDVIEW IR AFTER DEAD CODE ELIMINATION ===", ir, enabled=True)

    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )  # domain inference does not support dynamic offsets yet

    # ── Structured-backend block (user's only addition vs grafrap_dace original) ──
    if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
        ir = NormalizeShifts().visit(ir)
        ir = inline_lifts.InlineLifts().visit(ir)
        ir = cart_unroll.CartesianDomainAndTypeRemapper.apply(  # type: ignore[assignment]
            ir,
            symbolic_domain_sizes=cast(dict[str, str | int] | None, symbolic_domain_sizes),
            offset_provider=offset_provider,
        )
        ir = cart_unroll.CartesianReductionUnroller.apply(ir)  # type: ignore[assignment]
        ir = NormalizeShifts().visit(ir)
        ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
        ir = dead_code_elimination.dead_code_elimination(
            ir, offset_provider_type=offset_provider_type, uids=uids
        )
    _print_ir_block("=== FIELDVIEW IR AFTER CARTESIAN UNROLLING ===", ir, enabled=True)

    ir = infer_domain_ops.InferDomainOps.apply(ir)
    _print_ir_block("=== FIELDVIEW IR AFTER INFERRING DOMAIN OPS ===", ir, enabled=True)

    ir = concat_where.canonicalize_domain_argument(ir)

    ir = ConstantFolding.apply(ir)  # type: ignore[assignment]  # always an itir.Program

    # `keep_existing_domains=True` is critical here: `canonicalize_domain_argument`
    # transforms `concat_where` into `as_fieldop`-with-let-bindings, so the per-branch
    # narrow Kolor domains live inside explicit `as_fieldop(stencil, domain)` nodes.
    # Without this flag, `infer_program` would override those narrow domains with the
    # outer SetAt domain (`Kolor:[0,3)`), and shifts would back-propagate widened source
    # ranges like `Kolor:[2,5)` — OOB on the cell field `dwdz` (Kolor=2). See the
    # docstring on `keep_existing_domains` for the exact rationale.
    ir = infer_domain.infer_program(
        ir,
        symbolic_domain_sizes=symbolic_domain_sizes,
        offset_provider=offset_provider,
        keep_existing_domains=True,
    )
    ir = ConstantFolding.apply(ir)  # type: ignore[assignment]  # always an itir.Program

    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    _print_ir_block("=== FIELDVIEW IR AFTER PRUNING EMPTY CONCAT WHERE ===", ir, enabled=True)

    ir = remove_broadcast.RemoveBroadcast.apply(ir)
    _print_ir_block("=== FINAL FIELDVIEW IR ===", ir, enabled=True)
    return ir
