# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Structured backend passes for mapping unstructured IR to Cartesian (I, J, Kolor) domains.

Passes (in execution order):
  1. StructuredTypeRemapper   — General: remap field types Edge/Cell/Vertex → IDim/JDim/Kolor
  2. SetAtRemapper            — General dispatcher + entity-specialized: remap SetAt domains,
                                per-kolor split (edge: 3, cell: 2, vertex: 1)
  3. BroadcastAxisExpander    — General: expand broadcast axes
  4. ThresholdConditionRewriter — General dispatcher + entity-specialized: rewrite threshold conds
  5. SymbolicSizeInliner      — General: inline symbolic domain sizes
  6. NeighborReductionUnroller — General + edge-specialized kolor peeling: unroll reductions
  7. KolorConstantPropagation — General (optional, disabled): dead kolor branch elimination
  8. CanDerefRewriter         — General: replace can_deref with True
  9. StructuredBackend        — Orchestration: run all passes in order

cart_unroll.py is a backward-compat shim that re-exports the old class names from here.
"""

import copy
import dataclasses
import math
import numbers
import os
from typing import Any, cast

import gt4py.next.iterator.transforms.map_dict as map_dict_module
from gt4py.eve import NodeTranslator
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_specifications as ts

map_dict = map_dict_module.map_dict

# =====================================================================
# Module-level constants
# =====================================================================

#: Unstructured horizontal entity axis names (before structured remapping).
_UNSTRUCTURED_AXES: frozenset[str] = frozenset({"Edge", "Vertex", "Cell"})

#: Structured Cartesian horizontal axis names (after structured remapping).
_STRUCTURED_HORIZONTAL_AXES: frozenset[str] = frozenset({"IDim", "JDim", "Kolor"})

#: Edge-to-edge connectivity tags that span multiple kolors on local intermediates.
_EDGE_TO_EDGE_CONNECTIVITIES: frozenset[str] = frozenset({"E2C2EO", "E2C2E"})

#: Threshold parameter name for the 2nd nudge line (edge entity).
_EDGE_NUDGE_THRESHOLD_ID = "start_2nd_nudge_line_idx_e"

# =====================================================================
# Module-level cache
# =====================================================================

_ENTITY_START_BOUNDS_CACHE: dict[
    tuple[str, int, int, int, int], dict[int, tuple[int, int, int, int]]
] = {}


# =====================================================================
# Coercion helpers
# =====================================================================

def _coerce_int(value: Any) -> int | None:
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_mapping_rows(value: Any, expected_len: int) -> tuple[tuple[int, ...], ...] | None:
    # if isinstance(value, np.ndarray):
    #     if value.ndim != 2 or value.shape[1] < expected_len:
    #         return None
    #     rows = []
    #     for row in value:
    #         parsed = tuple(_coerce_int(item) for item in row[:expected_len])
    #         if any(item is None for item in parsed):
    #             return None
    #         rows.append(cast(tuple[int, ...], parsed))
    #     return tuple(rows)
    
    if not isinstance(value, (list, tuple)):
        return None
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) < expected_len:
            return None
        parsed = tuple(_coerce_int(item) for item in row[:expected_len])
        if any(item is None for item in parsed):
            return None
        rows.append(cast(tuple[int, ...], parsed))
    return tuple(rows)


# =====================================================================
# Mapping-based bounds derivation (unchanged from cart_unroll.py)
# =====================================================================

def _derive_entity_start_bounds_from_mapping(
    entity_name: str,
    *,
    mapping_rows: tuple[tuple[int, ...], ...],
    horizontal_start: int,
    max_i: int,
    max_j: int,
) -> dict[int, tuple[int, int, int, int]] | None:
    cache_key = (entity_name, id(mapping_rows), horizontal_start, max_i, max_j)
    if cache_key in _ENTITY_START_BOUNDS_CACHE:
        return _ENTITY_START_BOUNDS_CACHE[cache_key]

    if horizontal_start < 0:
        horizontal_start = 0
    if horizontal_start >= len(mapping_rows):
        return None

    if entity_name in {"Edge", "Cell"}:
        valid_kolors = {0, 1, 2} if entity_name == "Edge" else {0, 1}
        by_kolor: dict[int, tuple[int, int, int, int]] = {}
        for i_val, j_val, kolor in mapping_rows[horizontal_start:]:
            if i_val < 0 or j_val < 0 or kolor < 0:
                continue
            if kolor not in valid_kolors:
                continue
            prev = by_kolor.get(kolor)
            if prev is None:
                by_kolor[kolor] = (i_val, j_val, i_val, j_val)
            else:
                by_kolor[kolor] = (
                    min(prev[0], i_val), min(prev[1], j_val),
                    max(prev[2], i_val), max(prev[3], j_val),
                )
        if not by_kolor:
            return None
        _ENTITY_START_BOUNDS_CACHE[cache_key] = by_kolor
        return by_kolor

    # Vertex: scalar horizontal_start maps to a single (i,j) lower shell.
    min_i = min_j = max_i_seen = max_j_seen = None
    for row in mapping_rows[horizontal_start:]:
        i_val, j_val = row[0], row[1]
        if i_val < 0 or j_val < 0:
            continue
        min_i = i_val if min_i is None else min(min_i, i_val)
        min_j = j_val if min_j is None else min(min_j, j_val)
        max_i_seen = i_val if max_i_seen is None else max(max_i_seen, i_val)
        max_j_seen = j_val if max_j_seen is None else max(max_j_seen, j_val)

    if min_i is None or min_j is None or max_i_seen is None or max_j_seen is None:
        return None
    bounds = {0: (min_i, min_j, max_i_seen, max_j_seen)}
    _ENTITY_START_BOUNDS_CACHE[cache_key] = bounds
    return bounds


def _derive_entity_range_bounds_from_mapping(
    entity_name: str,
    *,
    mapping_rows: tuple[tuple[int, ...], ...],
    horizontal_start: int,
    horizontal_end: int,
    max_i: int,
    max_j: int,
) -> dict[int, tuple[int, int, int, int]] | None:
    """Return per-kolor bounding boxes for mapping_rows[horizontal_start:horizontal_end]."""
    if horizontal_start < 0:
        horizontal_start = 0
    if horizontal_end > len(mapping_rows):
        horizontal_end = len(mapping_rows)
    if horizontal_start >= horizontal_end:
        return None

    valid_kolors = {0, 1, 2} if entity_name == "Edge" else {0, 1}
    by_kolor: dict[int, tuple[int, int, int, int]] = {}
    for i_val, j_val, kolor in mapping_rows[horizontal_start:horizontal_end]:
        if i_val < 0 or j_val < 0 or kolor < 0 or kolor not in valid_kolors:
            continue
        prev = by_kolor.get(kolor)
        if prev is None:
            by_kolor[kolor] = (i_val, j_val, i_val, j_val)
        else:
            by_kolor[kolor] = (
                min(prev[0], i_val), min(prev[1], j_val),
                max(prev[2], i_val), max(prev[3], j_val),
            )
    return by_kolor if by_kolor else None


# =====================================================================
# Shift / concat-where builders (unchanged from cart_unroll.py)
# =====================================================================

def _apply_shift_chain(arg: ir.Expr, shift_spec: tuple[ir.OffsetLiteral, ...]) -> ir.Expr:
    normalized: list[ir.OffsetLiteral] = []
    for idx, off in enumerate(shift_spec):
        if idx % 2 == 0 and isinstance(off.value, str) and off.value in _STRUCTURED_HORIZONTAL_AXES:
            normalized.append(
                ir.OffsetLiteral(value=common.dimension_to_implicit_offset(off.value))
            )
        else:
            normalized.append(off)
    return im.call(im.call("shift")(*tuple(normalized)))(arg)


def _build_concat_where_from_branches(arg: ir.Expr, branches) -> ir.Expr:
    cond, shift_spec = branches[0]
    expr = _apply_shift_chain(copy.deepcopy(arg), shift_spec)
    if len(branches) == 1:
        return expr
    return im.concat_where(cond, expr, _build_concat_where_from_branches(arg, branches[1:]))


def _make_lifted_deref_shift(
    arg: ir.Expr, shift_spec: tuple[ir.OffsetLiteral, ...], domain: ir.Expr | None = None
) -> ir.Expr:
    it_name = "__cart_unroll_it"
    return im.as_fieldop(
        im.lambda_(it_name)(im.deref(_apply_shift_chain(im.ref(it_name), shift_spec))),
        domain,
    )(copy.deepcopy(arg))


def _needs_edge_shape_bounds(connectivity: str | None) -> bool:
    if connectivity is None:
        return False
    normalized = connectivity.replace("ₒ", "").replace("_o", "")
    return normalized in {"E2V", "E2C2EO"}


def _neutral_element_for_reduce_op(red_op: ir.Expr, red_init: ir.Expr) -> ir.Expr:
    op_name = red_op.id if isinstance(red_op, ir.SymRef) else None
    if op_name in {"plus", "add"}:
        return im.literal("0.0", "float64")
    if op_name in {"multiplies", "mul"}:
        return im.literal("1.0", "float64")
    if op_name in {"maximum", "max"}:
        return im.literal(str(-math.inf), "float64")
    if op_name in {"minimum", "min"}:
        return im.literal(str(math.inf), "float64")
    return copy.deepcopy(red_init)


def _bounded_shifted_deref(
    ref: ir.Expr,
    shift_spec: tuple[ir.OffsetLiteral, ...],
    neutral_element: ir.Expr,
    domain_bounds: dict[str, tuple[ir.Expr, ir.Expr]] | None = None,
) -> ir.Expr:
    shifted_ref = _apply_shift_chain(copy.deepcopy(ref), shift_spec)
    return im.deref(shifted_ref)


def _build_field_concat_where_from_branches(
    arg: ir.Expr,
    branches,
    domain: ir.Expr | None = None,
    inferred_kolor_start: int | None = None,
    apply_edge_shape_bounds: bool = False,
    current_kolor: int | None = None,
) -> ir.Expr:
    def _extract_kolor_interval(cond_expr: ir.Expr | None) -> tuple[int, int] | None:
        if cond_expr is None or not cpm.is_call_to(cond_expr, "cartesian_domain"):
            return None
        for range_expr in cond_expr.args:
            if not (cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3):
                continue
            if _get_axis_name(range_expr.args[0]) != "Kolor":
                continue
            lo, hi = range_expr.args[1], range_expr.args[2]
            if (
                isinstance(lo, ir.OffsetLiteral) and isinstance(lo.value, int)
                and isinstance(hi, ir.OffsetLiteral) and isinstance(hi.value, int)
            ):
                return lo.value, hi.value
        return None

    def _narrow_domain_kolor(domain_expr: ir.Expr, kolor_interval: tuple[int, int]) -> ir.Expr:
        if not cpm.is_call_to(domain_expr, "cartesian_domain"):
            return domain_expr
        lo, hi = kolor_interval
        new_args: list[ir.Expr] = []
        for range_expr in domain_expr.args:
            if not (cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3):
                new_args.append(copy.deepcopy(range_expr))
                continue
            axis = range_expr.args[0]
            if _get_axis_name(axis) == "Kolor":
                new_args.append(im.named_range(
                    cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(axis)),
                    ir.OffsetLiteral(value=lo),
                    ir.OffsetLiteral(value=hi),
                ))
            else:
                new_args.append(copy.deepcopy(range_expr))
        return im.call("cartesian_domain")(*new_args)

    def _kolor_shift(shift_spec: tuple[ir.OffsetLiteral, ...]) -> int:
        kolor_axis_tags = {common.dimension_to_implicit_offset("Kolor"), "Kolor"}
        for idx in range(0, len(shift_spec), 2):
            if idx + 1 >= len(shift_spec):
                break
            axis_lit = shift_spec[idx]
            off_lit = shift_spec[idx + 1]
            if (
                isinstance(axis_lit, ir.OffsetLiteral) and axis_lit.value in kolor_axis_tags
                and isinstance(off_lit, ir.OffsetLiteral) and isinstance(off_lit.value, int)
            ):
                return int(off_lit.value)
        return 0

    def _idim_shift(shift_spec: tuple[ir.OffsetLiteral, ...]) -> int:
        idim_axis_tags = {common.dimension_to_implicit_offset("IDim"), "IDim"}
        for idx in range(0, len(shift_spec), 2):
            if idx + 1 >= len(shift_spec):
                break
            axis_lit = shift_spec[idx]
            off_lit = shift_spec[idx + 1]
            if (
                isinstance(axis_lit, ir.OffsetLiteral) and axis_lit.value in idim_axis_tags
                and isinstance(off_lit, ir.OffsetLiteral) and isinstance(off_lit.value, int)
            ):
                return int(off_lit.value)
        return 0

    def _jdim_shift(shift_spec: tuple[ir.OffsetLiteral, ...]) -> int:
        jdim_axis_tags = {common.dimension_to_implicit_offset("JDim"), "JDim"}
        for idx in range(0, len(shift_spec), 2):
            if idx + 1 >= len(shift_spec):
                break
            axis_lit = shift_spec[idx]
            off_lit = shift_spec[idx + 1]
            if (
                isinstance(axis_lit, ir.OffsetLiteral) and axis_lit.value in jdim_axis_tags
                and isinstance(off_lit, ir.OffsetLiteral) and isinstance(off_lit.value, int)
            ):
                return int(off_lit.value)
        return 0

    def _minus_one_offset(expr: ir.Expr) -> ir.Expr:
        if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
            return ir.OffsetLiteral(value=expr.value - 1)
        return im.minus(copy.deepcopy(expr), ir.OffsetLiteral(value=1))

    def _apply_offset(expr: ir.Expr, n: int) -> ir.Expr:
        """Apply integer offset n to expr; n=0 returns a deepcopy unchanged."""
        if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
            return ir.OffsetLiteral(value=expr.value + n)
        if n == 0:
            return copy.deepcopy(expr)
        if n < 0:
            return im.minus(copy.deepcopy(expr), ir.OffsetLiteral(value=-n))
        return im.plus(copy.deepcopy(expr), ir.OffsetLiteral(value=n))

    def _edge_shape_domain(
        source_kolor: int,
        shift_spec: tuple[ir.OffsetLiteral, ...],
        current_kolor: int | None = None,
    ) -> ir.Expr | None:
        if current_kolor is not None:
            return None
        if domain is None or not cpm.is_call_to(domain, "cartesian_domain"):
            return None
        target_kolor = source_kolor + _kolor_shift(shift_spec)
        if target_kolor not in {0, 1, 2}:
            return None
        id_range = jd_range = None
        for range_expr in domain.args:
            if not (cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3):
                continue
            axis_name = _get_axis_name(range_expr.args[0])
            if axis_name == "IDim":
                id_range = range_expr
            elif axis_name == "JDim":
                jd_range = range_expr
        if id_range is None or jd_range is None:
            return None
        i_lo = copy.deepcopy(id_range.args[1])
        i_hi = copy.deepcopy(id_range.args[2])
        j_lo = copy.deepcopy(jd_range.args[1])
        j_hi = copy.deepcopy(jd_range.args[2])
        # No IDim/JDim clipping needed: the structured backend packs the FULL edge/vertex
        # field (not just the inner domain), so reads at any position within the allocated
        # array are valid. Previous clip attempts made branch transients too small while the
        # outer consumer still iterated over the full outer domain, causing SDFG OOB errors.
        new_ranges: list[ir.Expr] = []
        for range_expr in domain.args:
            if not (cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3):
                continue
            axis_name = _get_axis_name(range_expr.args[0])
            if axis_name == "IDim":
                new_ranges.append(im.named_range(
                    cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(range_expr.args[0])),
                    i_lo, i_hi,
                ))
            elif axis_name == "JDim":
                new_ranges.append(im.named_range(
                    cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(range_expr.args[0])),
                    j_lo, j_hi,
                ))
            else:
                new_ranges.append(copy.deepcopy(range_expr))
        return im.call("cartesian_domain")(*new_ranges)

    def _infer_source_kolor_from_cond(
        cond_expr: ir.Expr | None, fallback: int | None
    ) -> tuple[int | None, int | None]:
        if cond_expr is not None:
            interval = _extract_kolor_interval(cond_expr)
            if interval is None:
                return None, fallback
            lo, hi = interval
            if hi - lo == 1:
                return lo, hi
            return None, fallback
        if fallback is None:
            return None, None
        return fallback, fallback + 1

    if current_kolor is not None:
        inferred = inferred_kolor_start
        fallback_spec = branches[-1][1] if branches else None
        for cond_expr, shift_spec in branches:
            k, next_k = _infer_source_kolor_from_cond(cond_expr, inferred)
            inferred = next_k
            if k == current_kolor:
                branch_domain = domain
                if apply_edge_shape_bounds:
                    ed = _edge_shape_domain(k, shift_spec, current_kolor=current_kolor)
                    if ed is not None:
                        branch_domain = ed
                return _make_lifted_deref_shift(arg, shift_spec, branch_domain)
            fallback_spec = shift_spec
        if fallback_spec is not None:
            branch_domain = domain
            if apply_edge_shape_bounds:
                ed = _edge_shape_domain(current_kolor, fallback_spec, current_kolor=current_kolor)
                if ed is not None:
                    branch_domain = ed
            return _make_lifted_deref_shift(arg, fallback_spec, branch_domain)

    cond, shift_spec = branches[0]
    source_kolor, next_inferred_kolor = _infer_source_kolor_from_cond(cond, inferred_kolor_start)
    branch_domain = domain
    if apply_edge_shape_bounds and source_kolor is not None:
        edge_domain = _edge_shape_domain(source_kolor, shift_spec)
        if edge_domain is not None:
            branch_domain = edge_domain
    # Narrow the Kolor range of `branch_domain` to match the concat_where `cond`.
    # Otherwise the as_fieldop stores the full outer Kolor:[0,3) and `infer_program`
    # back-propagates `[0,3)+shift` to the source — widening past valid field extents
    # and producing OOB reads in DaCe (SIGSEGV on CPU, ILLEGAL_ADDRESS on GPU).
    kolor_interval = _extract_kolor_interval(cond)
    if kolor_interval is not None:
        branch_domain = _narrow_domain_kolor(branch_domain, kolor_interval)
    expr = _make_lifted_deref_shift(arg, shift_spec, branch_domain)

    if len(branches) == 1:
        # Trailing-else branch (cond=None in `map_dict.py`). A bare wide-domain
        # `as_fieldop` here causes GPU OOB at 512×512 — e.g. E2C[1] trailing with
        # shift `_OffKolor:-2` reading cell field at Kolor=output-2 OOBs at output
        # K=0,1 (source K=-2,-1) on cell (Kolor=2). At 26×26 the OOB reads land in
        # adjacent valid memory; at 512×512 they land outside any allocation and
        # CUDA traps it.
        #
        # Naïvely narrowing this branch's as_fieldop domain to Kolor:[k,k+1) breaks
        # CPU numerical verification (`+inf location mismatch`): the let-lifted
        # `concat_where` structure created by `canonicalize_domain_argument` reads
        # the resulting 1-slot temp at *other* kolors that are downstream-masked
        # but still materialized by DaCe — those reads are OOB on the 1-slot temp.
        #
        # Right fix: produce a wider field with valid values at ALL kolors so the
        # consumer's reads are always in-bounds. Wrap as:
        #
        #   concat_where(Kolor:[source_kolor, source_kolor+1),
        #                narrow_trailing_shift,         # the actual trailing computation
        #                literal_zero_fallback)         # NEVER-arg → no field reads
        #
        # The fallback is a constant `0.0` field-op whose lambda doesn't reference its
        # input, so `infer_domain` marks `arg` as NEVER and DaCe's
        # `_parse_fieldop_arg` strips the unused arg. This avoids OOB reads regardless
        # of the source field's Kolor extent (a previous identity-based fallback
        # read `arg` at output position K=0..source_kolor-1, which OOB'd on vertex
        # fields with Kolor=1 — observed in stencils 01/10 E2C2V).
        if source_kolor is not None and source_kolor > 0 and isinstance(branch_domain, ir.FunCall):
            narrow_trailing_domain = _narrow_domain_kolor(
                branch_domain, (source_kolor, source_kolor + 1)
            )
            narrow_trailing_expr = _make_lifted_deref_shift(
                arg, shift_spec, narrow_trailing_domain
            )
            # Use the FULL outer domain (not narrowed to [0, source_kolor)) so that
            # `canonicalize_domain_argument` can access the fallback at any Kolor position.
            # The lambda body is a constant 0.0 (NEVER-arg), so `arg` is never read regardless
            # of domain size — the full domain just ensures no OOB on the transient itself.
            fallback_expr = im.as_fieldop(
                im.lambda_("__cart_trailing_unused")(im.literal("0.0", "float64")),
                domain,
            )(copy.deepcopy(arg))
            kolor_axis = ir.AxisLiteral(
                value="Kolor", kind=common.DimensionKind.HORIZONTAL
            )
            narrow_cond = im.call("cartesian_domain")(
                im.named_range(
                    kolor_axis,
                    ir.OffsetLiteral(value=source_kolor),
                    ir.OffsetLiteral(value=source_kolor + 1),
                )
            )
            return im.concat_where(narrow_cond, narrow_trailing_expr, fallback_expr)
        return expr

    return im.concat_where(
        cond,
        expr,
        _build_field_concat_where_from_branches(
            arg, branches[1:], domain,
            inferred_kolor_start=next_inferred_kolor,
            apply_edge_shape_bounds=apply_edge_shape_bounds,
            current_kolor=current_kolor,
        ),
    )


# =====================================================================
# Domain / entity detection helpers
# =====================================================================

def _get_axis_name(axis_expr: ir.Expr) -> str | None:
    if isinstance(axis_expr, ir.AxisLiteral) and isinstance(axis_expr.value, str):
        return axis_expr.value
    if hasattr(axis_expr, "value") and isinstance(axis_expr.value, str):
        return axis_expr.value
    return None


def _iter_domain_nodes(expr: ir.Expr):
    if cpm.is_call_to(expr, "make_tuple"):
        for a in expr.args:
            yield from _iter_domain_nodes(a)
        return
    yield expr


def _detect_setat_entity(domain_expr: ir.Expr) -> str | None:
    """Return 'Edge', 'Cell', 'Vertex', or None for the horizontal entity in a SetAt domain."""
    for dom in _iter_domain_nodes(domain_expr):
        if not cpm.is_call_to(dom, "unstructured_domain"):
            continue
        for nr in dom.args:
            if _named_range_args(nr) is not None:
                axis_name = _get_axis_name(nr.args[0])
                if axis_name in {"Edge", "Cell", "Vertex"}:
                    return axis_name
    return None


def _is_unstructured_edge_domain_stmt(domain_expr: ir.Expr) -> bool:
    return _detect_setat_entity(domain_expr) == "Edge"


# _EDGE_TO_EDGE_CONNECTIVITIES is defined at module top as part of the constants block.


def _named_range_args(nr: ir.Expr) -> tuple[ir.Expr, ir.Expr, ir.Expr] | None:
    """Return (axis, lo, hi) if nr is a valid named_range call, else None."""
    if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
        return nr.args[0], nr.args[1], nr.args[2]
    return None


def _conn_name_and_is_edge_to_non_edge(key: tuple) -> tuple[str, bool]:
    """Extract connectivity name and determine if it is edge-source.

    All edge-source connectivities (E2C, E2V, E2C2EO, E2C2E, …) have their
    map_dict branches keyed by the OUTPUT EDGE kolor.  Passing current_kolor
    directly selects the correct single shift, avoiding a full 3-branch
    concat_where inside per-kolor SetAts.

    Non-edge-source connectivities (C2E, V2E, …) have branches keyed by the
    source entity's kolor (cell or vertex), which differs from the edge
    current_kolor — so current_kolor must be suppressed for those.

    Note: E2C2EO/E2C2E branches are keyed by OUTPUT edge kolor (verified from
    map_dict.py: Kolor:[0,1)→shift Kolor:+2, Kolor:[1,2)→shift Kolor:-1, etc.),
    so passing current_kolor is correct despite the fact that the READ position
    is at a different kolor than the output.
    """
    conn_name = key[0].value if isinstance(key[0], ir.OffsetLiteral) and isinstance(key[0].value, str) else ""
    normalized = conn_name.rstrip("ₒ")
    is_edge_to_non_edge = normalized.startswith("E")
    return conn_name, is_edge_to_non_edge


def _expr_uses_edge_to_edge_connectivity(node: ir.Node) -> bool:
    for lit in node.pre_walk_values().if_isinstance(ir.OffsetLiteral):
        if isinstance(lit.value, str) and lit.value.rstrip("ₒ") in _EDGE_TO_EDGE_CONNECTIVITIES:
            return True
    return False


def _e2c2e_on_local_intermediate(node: ir.Node) -> bool:
    """Return True if any E2C2EO/E2C2E neighbor access is on a lambda-bound (local) field."""
    lambda_params: set[str] = set()
    for lam in node.pre_walk_values().if_isinstance(ir.Lambda):
        for p in lam.params:
            lambda_params.add(str(p.id))

    def _unwrap_to_symref(expr: ir.Expr) -> ir.SymRef | None:
        if isinstance(expr, ir.SymRef):
            return expr
        if cpm.is_applied_as_fieldop(expr) and len(expr.args) == 1:
            return _unwrap_to_symref(expr.args[0])
        return None

    for fun_call in node.pre_walk_values().if_isinstance(ir.FunCall):
        if not (cpm.is_applied_as_fieldop(fun_call) and len(fun_call.args) == 1):
            continue
        stencil = fun_call.fun.args[0]
        if not isinstance(stencil, ir.Lambda):
            continue
        if not (
            cpm.is_call_to(stencil.expr, "neighbors")
            and len(stencil.expr.args) == 2
            and len(stencil.params) == 1
        ):
            continue
        conn_expr = stencil.expr.args[0]
        if not (isinstance(conn_expr, ir.OffsetLiteral) and isinstance(conn_expr.value, str)):
            continue
        if conn_expr.value.rstrip("ₒ") not in _EDGE_TO_EDGE_CONNECTIVITIES:
            continue
        sym = _unwrap_to_symref(fun_call.args[0])
        if sym is not None and str(sym.id) in lambda_params:
            return True
    return False


def _kolor_from_domain(domain: ir.Expr | None) -> int | None:
    """Return k if domain has a single-kolor Kolor:[k, k+1) range, else None."""
    if domain is None or not cpm.is_call_to(domain, "cartesian_domain"):
        return None
    for nr in domain.args:
        if _named_range_args(nr) is None:
            continue
        if _get_axis_name(nr.args[0]) != "Kolor":
            continue
        lo, hi = nr.args[1], nr.args[2]
        if isinstance(lo, ir.OffsetLiteral) and isinstance(hi, ir.OffsetLiteral):
            if hi.value - lo.value == 1:
                return int(lo.value)
    return None


def _kolor_from_cw_condition(cond: ir.Expr) -> int | None:
    def _walk(expr: ir.Expr) -> int | None:
        if cpm.is_call_to(expr, "and_"):
            for arg in expr.args:
                k = _walk(arg)
                if k is not None:
                    return k
            return None
        return _kolor_from_domain(expr)
    return _walk(cond)


def _kolor_cw_condition_to_domain(cond: ir.Expr, fallback_domain: ir.Expr) -> ir.Expr | None:
    """Convert a concat_where condition back to a cartesian_domain."""
    by_axis: dict[str, ir.Expr] = {}

    def _collect(expr: ir.Expr) -> None:
        if cpm.is_call_to(expr, "and_"):
            for arg in expr.args:
                _collect(arg)
        elif cpm.is_call_to(expr, "cartesian_domain"):
            for nr in expr.args:
                if _named_range_args(nr) is not None:
                    name = _get_axis_name(nr.args[0])
                    if name is not None:
                        by_axis[name] = copy.deepcopy(nr)

    _collect(cond)
    if not by_axis:
        return None

    def _collect_fallback(domain: ir.Expr) -> None:
        if cpm.is_call_to(domain, "cartesian_domain"):
            for nr in domain.args:
                if _named_range_args(nr) is not None:
                    name = _get_axis_name(nr.args[0])
                    if name is not None and name not in _STRUCTURED_HORIZONTAL_AXES:
                        by_axis.setdefault(name, copy.deepcopy(nr))
        elif cpm.is_call_to(domain, "make_tuple"):
            for sub in domain.args:
                _collect_fallback(sub)
                break

    _collect_fallback(fallback_domain)
    canonical = ["IDim", "JDim", "Kolor"]
    ordered = [by_axis[n] for n in canonical if n in by_axis]
    for name, nr in by_axis.items():
        if name not in canonical:
            ordered.append(nr)
    return im.call("cartesian_domain")(*ordered)


# =====================================================================
# Symbolic-size helpers (extracted from old visit_FunCall / visit_SetAt closures)
# =====================================================================

def _pick_symbolic_int(symbolic_domain_sizes: dict[str, Any], *names: str) -> int | None:
    """Return the first integer value found under any of *names* in symbolic_domain_sizes.

    Companion to _pick_size_param: use this when you need an int for arithmetic or as a
    Python-level bound; use _pick_size_param when you need an ir.Expr for embedding in IR nodes.
    Call sites in the mapping path always receive integer-valued dicts (confirmed by H7 tests).
    """
    for name in names:
        value = symbolic_domain_sizes.get(name)
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _horizontal_start_mapping_enabled(symbolic_domain_sizes: dict[str, Any]) -> bool:
    raw = symbolic_domain_sizes.get("use_horizontal_start_mapping")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, numbers.Integral):
        return int(raw) != 0
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _pick_size_param(
    program_param_ids: set[str],
    symbolic_domain_sizes: dict[str, Any],
    *candidates: str,
) -> ir.Expr | None:
    for candidate in candidates:
        if candidate in program_param_ids:
            return im.ref(candidate)
        if candidate in symbolic_domain_sizes:
            symbolic_size = symbolic_domain_sizes[candidate]
            if isinstance(symbolic_size, numbers.Integral):
                return ir.OffsetLiteral(value=int(symbolic_size))
            if isinstance(symbolic_size, str):
                if symbolic_size in program_param_ids:
                    return im.ref(symbolic_size)
                try:
                    return ir.OffsetLiteral(value=int(symbolic_size))
                except ValueError:
                    return im.ref(symbolic_size)
            return im.ensure_expr(symbolic_size)
    return None


def _offset_int_value(expr: ir.Expr) -> int | None:
    if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
        return expr.value
    return None


def _extract_index_value(expr: ir.Expr) -> int | None:
    """Extract a non-negative integer index from ir.Literal or ir.OffsetLiteral.

    Used for tuple_get indices: im.tuple_get produces ir.Literal; manually-constructed
    IR may use ir.OffsetLiteral.
    """
    if isinstance(expr, ir.Literal):
        try:
            return int(expr.value)
        except (ValueError, TypeError):
            return None
    if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
        return expr.value
    return None


def _offset_add(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
    lv, rv = _offset_int_value(lhs), _offset_int_value(rhs)
    if lv is not None and rv is not None:
        return ir.OffsetLiteral(value=lv + rv)
    # Zero-offset fast path: return the other operand.  Both OffsetLiteral and SymRef are
    # value-typed in GT4Py IR (never mutated by callers), so no deepcopy is needed here.
    if rv == 0:
        return lhs
    if lv == 0:
        return rhs
    return im.plus(copy.deepcopy(lhs), copy.deepcopy(rhs))


def _offset_sub(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
    lv, rv = _offset_int_value(lhs), _offset_int_value(rhs)
    if lv is not None and rv is not None:
        return ir.OffsetLiteral(value=lv - rv)
    if rv == 0:
        return lhs
    return im.minus(copy.deepcopy(lhs), copy.deepcopy(rhs))


def _plus_n(expr: ir.Expr, n: int) -> ir.Expr:
    if n == 0:
        return copy.deepcopy(expr)
    if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
        return ir.OffsetLiteral(value=expr.value + n)
    return im.plus(copy.deepcopy(expr), ir.OffsetLiteral(value=n))


def _minus_n(expr: ir.Expr, n: int) -> ir.Expr:
    if n == 0:
        return copy.deepcopy(expr)
    if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
        return ir.OffsetLiteral(value=expr.value - n)
    return im.minus(copy.deepcopy(expr), ir.OffsetLiteral(value=n))


def _minus_one(expr: ir.Expr) -> ir.Expr:
    if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
        return ir.OffsetLiteral(value=expr.value - 1)
    if isinstance(expr, ir.Literal) and str(expr.value).lstrip("-").isdigit():
        return ir.Literal(value=str(int(expr.value) - 1), type=expr.type)
    return im.minus(copy.deepcopy(expr), ir.OffsetLiteral(value=1))


def _entity_kolor_bounds(entity_name: str) -> tuple[ir.Expr, ir.Expr] | None:
    kolor_stops = {"Vertex": 1, "Cell": 2, "Edge": 3}
    if entity_name not in kolor_stops:
        return None
    return ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=kolor_stops[entity_name])


def _entity_start_bounds_from_horizontal_start(
    entity_name: str,
    symbolic_domain_sizes: dict[str, Any],
) -> dict[int, tuple[int, int, int, int]] | None:
    if not _horizontal_start_mapping_enabled(symbolic_domain_sizes):
        return None
    max_i_int = _pick_symbolic_int(symbolic_domain_sizes, "i_max", "domain_i_max", "max_i", "domain_max_i", "nx")
    max_j_int = _pick_symbolic_int(symbolic_domain_sizes, "j_max", "domain_j_max", "max_j", "domain_max_j", "ny")
    if max_i_int is None or max_j_int is None:
        return None

    if entity_name == "Edge":
        mapping_value = symbolic_domain_sizes.get("edge_to_ijk")
        mapping_rows = _coerce_mapping_rows(mapping_value, expected_len=3)
        horizontal_start = _pick_symbolic_int(
            symbolic_domain_sizes, "horizontal_start_edge", "horizontal_start_e", "horizontal_start"
        )
    elif entity_name == "Vertex":
        mapping_value = symbolic_domain_sizes.get("vertex_to_ij")
        mapping_rows = _coerce_mapping_rows(mapping_value, expected_len=2)
        horizontal_start = _pick_symbolic_int(
            symbolic_domain_sizes, "horizontal_start_vertex", "horizontal_start_v", "horizontal_start"
        )
    elif entity_name == "Cell":
        mapping_value = symbolic_domain_sizes.get("cell_to_ijk")
        mapping_rows = _coerce_mapping_rows(mapping_value, expected_len=3)
        horizontal_start = _pick_symbolic_int(
            symbolic_domain_sizes, "horizontal_start_cell", "horizontal_start_c", "horizontal_start"
        )
    else:
        return None

    if mapping_rows is None or horizontal_start is None or horizontal_start < 0:
        return None
    return _derive_entity_start_bounds_from_mapping(
        entity_name,
        mapping_rows=mapping_rows,
        horizontal_start=horizontal_start,
        max_i=max_i_int,
        max_j=max_j_int,
    )


def _mapping_based_axis_bounds(
    entity_name: str,
    axis_name: str,
    symbolic_domain_sizes: dict[str, Any],
    kolor: int | None = None,
) -> tuple[ir.Expr, ir.Expr] | None:
    if axis_name not in {"IDim", "JDim"}:
        return None
    bounds_by_kolor = _entity_start_bounds_from_horizontal_start(entity_name, symbolic_domain_sizes)
    if not bounds_by_kolor:
        return None
    axis_index = 0 if axis_name == "IDim" else 1
    axis_hi_index = 2 if axis_name == "IDim" else 3
    if entity_name == "Edge" and kolor is not None:
        if kolor not in bounds_by_kolor:
            return None
        axis_lo = bounds_by_kolor[kolor][axis_index]
        axis_hi = bounds_by_kolor[kolor][axis_hi_index] + 1
        return ir.OffsetLiteral(value=axis_lo), ir.OffsetLiteral(value=axis_hi)
    axis_los = [pair[axis_index] for pair in bounds_by_kolor.values()]
    axis_his = [pair[axis_hi_index] for pair in bounds_by_kolor.values()]
    return ir.OffsetLiteral(value=min(axis_los)), ir.OffsetLiteral(value=max(axis_his) + 1)


def _cartesian_axis_bounds(
    axis_name: str,
    program_param_ids: set[str],
    symbolic_domain_sizes: dict[str, Any],
) -> tuple[ir.Expr, ir.Expr] | None:
    if axis_name == "IDim":
        lower_base = _pick_size_param(program_param_ids, symbolic_domain_sizes, "i_min", "domain_i_min", "imin")
        upper_base = _pick_size_param(program_param_ids, symbolic_domain_sizes,
                                      "i_max", "domain_i_max", "max_i", "domain_max_i", "nx", "num_i", "ni")
        if upper_base is None:
            return None
        lower = ir.OffsetLiteral(value=0) if lower_base is None else copy.deepcopy(lower_base)
        upper = copy.deepcopy(upper_base)
        return lower, upper
    if axis_name == "JDim":
        lower_base = _pick_size_param(program_param_ids, symbolic_domain_sizes, "j_min", "domain_j_min", "jmin")
        upper_base = _pick_size_param(program_param_ids, symbolic_domain_sizes,
                                      "j_max", "domain_j_max", "max_j", "domain_max_j", "ny", "num_j", "nj")
        if upper_base is None:
            return None
        lower = ir.OffsetLiteral(value=0) if lower_base is None else copy.deepcopy(lower_base)
        upper = copy.deepcopy(upper_base)
        return lower, upper
    if axis_name == "Kolor":
        return ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=3)
    return None


def _entity_cartesian_bounds(
    entity_name: str,
    axis_name: str,
    program_param_ids: set[str],
    symbolic_domain_sizes: dict[str, Any],
    kolor: int | None = None,
) -> tuple[ir.Expr, ir.Expr] | None:
    # S1: cache the mapping result to avoid calling _mapping_based_axis_bounds twice.
    mapping_bounds = _mapping_based_axis_bounds(entity_name, axis_name, symbolic_domain_sizes, kolor=kolor)
    bounds = mapping_bounds if mapping_bounds is not None else _cartesian_axis_bounds(axis_name, program_param_ids, symbolic_domain_sizes)
    if bounds is None:
        return None
    lo, hi = bounds
    # Cell-centered fields are one element smaller than the vertex extent on each axis
    # when falling back to cartesian_axis_bounds (no per-kolor mapping available).
    if entity_name == "Cell" and mapping_bounds is None:
        hi = _offset_sub(hi, ir.OffsetLiteral(value=1))
    return lo, hi


def _is_fully_structured_field(expr: ir.Expr) -> bool:
    type_ = getattr(expr, "type", None)
    if not isinstance(type_, ts.FieldType):
        return False
    dim_names = {dim.value for dim in type_.dims}
    if _STRUCTURED_HORIZONTAL_AXES.isdisjoint(dim_names):
        return False
    return not any(name in _UNSTRUCTURED_AXES for name in dim_names)


def _retarget_self_refs(expr: ir.Expr, target: ir.Expr) -> ir.Expr:
    if not isinstance(target, ir.SymRef) or not isinstance(target.type, ts.FieldType):
        return expr
    target_id = target.id
    target_type = target.type

    def _has_unstructured_axis(type_: ts.TypeSpec | None) -> bool:
        return isinstance(type_, ts.FieldType) and any(
            dim.value in _UNSTRUCTURED_AXES for dim in type_.dims
        )

    class _RetargetSelfRef(NodeTranslator):
        def visit_SymRef(self, n: ir.SymRef, **kws) -> ir.SymRef:
            if n.id == target_id and _has_unstructured_axis(n.type):
                return ir.SymRef(id=n.id, type=target_type)
            return n

    return _RetargetSelfRef().visit(expr)


# =====================================================================
# Pass 1: StructuredTypeRemapper — General
# =====================================================================

@dataclasses.dataclass
class StructuredTypeRemapper(NodeTranslator):
    """General: Remaps field types from unstructured to structured Cartesian types.

    Works for all entity types: Edge → IDim/JDim/Kolor, Cell → IDim/JDim/Kolor,
    Vertex → IDim/JDim/Kolor.
    Also remaps Program param types and validates transform requirements.
    """

    @staticmethod
    def _cartesian_remapped_type(type_: ts.TypeSpec | None) -> ts.TypeSpec | None:
        if not isinstance(type_, ts.FieldType):
            return type_
        if not any(dim.value in _UNSTRUCTURED_AXES for dim in type_.dims):
            return type_
        idim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        jdim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
        new_dims: list[common.Dimension] = []
        for dim in type_.dims:
            if dim.value in _UNSTRUCTURED_AXES:
                new_dims.extend([idim, jdim, kolor])
            else:
                new_dims.append(dim)
        return ts.FieldType(dims=new_dims, dtype=type_.dtype)

    @staticmethod
    def _program_uses_horizontal_unstructured_axis(node: ir.Node) -> bool:
        for fun_call in node.pre_walk_values().if_isinstance(ir.FunCall):
            if not cpm.is_call_to(fun_call, "named_range") or len(fun_call.args) != 3:
                continue
            if _get_axis_name(fun_call.args[0]) in _UNSTRUCTURED_AXES:
                return True
        return False

    @staticmethod
    def _program_uses_offsets(node: ir.Node, names: set[str]) -> bool:
        for lit in node.pre_walk_values().if_isinstance(ir.OffsetLiteral):
            if isinstance(lit.value, str) and lit.value in names:
                return True
        return False

    @staticmethod
    def _infer_symbolic_sizes_from_offset_provider(
        offset_provider: common.OffsetProvider,
    ) -> dict[str, int]:
        if not isinstance(offset_provider, dict):
            return {}
        axis_sizes: dict[str, int] = {}
        for connectivity in offset_provider.values():
            domain = getattr(connectivity, "domain", None)
            shape = getattr(connectivity, "shape", None)
            dims = getattr(domain, "dims", ()) if domain is not None else ()
            if shape is None or not dims:
                continue
            for dim, size in zip(dims, shape):
                dim_name = getattr(dim, "value", None)
                if isinstance(dim_name, str) and isinstance(size, numbers.Integral) and int(size) > 0:
                    axis_sizes.setdefault(dim_name, int(size))
        i_extent = axis_sizes.get("IDim")
        j_extent = axis_sizes.get("JDim")
        if not (isinstance(i_extent, int) and isinstance(j_extent, int)
                and i_extent > 0 and j_extent > 0):
            return {}
        return {"i_min": 0, "i_max": i_extent, "j_min": 0, "j_max": j_extent,
                "max_i": i_extent, "max_j": j_extent}

    @staticmethod
    def _validate_structured_remap_requirements(
        node: ir.Node, symbolic_domain_sizes: dict[str, Any]
    ) -> bool:
        """Return True to proceed with the structured transform, False to skip gracefully."""
        if not isinstance(node, ir.Program):
            return False
        if not StructuredTypeRemapper._program_uses_horizontal_unstructured_axis(node):
            return False
        if os.environ.get("GT4PY_CART_DEBUG", "0") == "1":
            print(f"[structured_backend] {node.id}: proceeding with structured remap")

        program_param_ids = {str(param.id) for param in node.params}

        def _has_any(*names: str) -> bool:
            return any(name in symbolic_domain_sizes or name in program_param_ids for name in names)

        # Signal fires when max_i/max_j are available as symbolic sizes OR program params.
        has_symbolic_structured_sizes = any(
            name in symbolic_domain_sizes or name in program_param_ids
            for name in ("max_i", "max_j", "domain_max_i", "domain_max_j", "nx", "ny")
        )
        if not has_symbolic_structured_sizes:
            return False

        use_mapping = (
            symbolic_domain_sizes.get("use_horizontal_start_mapping") in {True, 1, "1", "true", "on", "yes"}
            and "edge_to_ijk" in symbolic_domain_sizes
        )
        # Mapping is optional: when present it enables kolor-specific bounds.
        # Without mapping, the generic max_i/max_j domain bounds are used.
        # However when symbolic structured sizes are present but mapping is absent,
        # skip gracefully (the stencil hasn't been set up for the structured path yet).
        if any(name in symbolic_domain_sizes for name in ("max_i", "max_j", "domain_max_i", "domain_max_j", "nx", "ny")):
            if not use_mapping:
                return False

        missing_groups: list[str] = []
        if not _has_any("i_max", "domain_i_max", "max_i", "domain_max_i", "nx", "num_i", "ni"):
            missing_groups.append("IDim upper bound")
        if not _has_any("j_max", "domain_j_max", "max_j", "domain_max_j", "ny", "num_j", "nj"):
            missing_groups.append("JDim upper bound")
        if missing_groups:
            raise ValueError(
                "Cartesian unstructured→structured remap requires explicit symbolic domain "
                f"sizes. Missing: {', '.join(missing_groups)}."
            )
        return True

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        symbolic_domain_sizes: dict[str, Any] | None = None,
        offset_provider: common.OffsetProvider | None = None,
    ) -> ir.Node:
        effective = dict(symbolic_domain_sizes or {})
        has_structured_bounds = any(
            key in effective
            for key in ("i_min", "i_max", "j_min", "j_max", "max_i", "max_j",
                        "domain_max_i", "domain_max_j", "nx", "ny")
        )
        if (not has_structured_bounds and offset_provider is not None
                and cls._program_uses_offsets(node, {"E2C2E"})):
            for key, value in cls._infer_symbolic_sizes_from_offset_provider(offset_provider).items():
                effective.setdefault(key, value)
        if not cls._validate_structured_remap_requirements(node, effective):
            return node
        return cls().visit(node, symbolic_domain_sizes=effective)

    def visit_Program(self, node: ir.Program, **kwargs) -> ir.Program:
        new_params = [
            ir.Sym(id=param.id, type=self._cartesian_remapped_type(param.type))
            for param in node.params
        ]
        new_body = [self.visit(stmt, **kwargs) for stmt in node.body]
        return ir.Program(
            id=node.id,
            function_definitions=node.function_definitions,
            params=new_params,
            declarations=node.declarations,
            body=new_body,
        )

    def visit_SymRef(self, node: ir.SymRef, **kwargs) -> ir.SymRef:
        if isinstance(node.type, it_ts.IteratorType) or str(node.id).startswith("__"):
            return ir.SymRef(id=node.id, type=None) if node.type is not None else node
        mapped_type = self._cartesian_remapped_type(node.type)
        return node if mapped_type is node.type else ir.SymRef(id=node.id, type=mapped_type)

    def visit_Sym(self, node: ir.Sym, **kwargs) -> ir.Sym:
        if isinstance(node.type, it_ts.IteratorType) or str(node.id).startswith("__"):
            return ir.Sym(id=node.id, type=None) if node.type is not None else node
        mapped_type = self._cartesian_remapped_type(node.type)
        return node if mapped_type is node.type else ir.Sym(id=node.id, type=mapped_type)

    def visit_Lambda(self, node: ir.Lambda, **kwargs) -> ir.Lambda:
        return ir.Lambda(
            params=[self.visit(param, **kwargs) for param in node.params],
            expr=self.visit(node.expr, **kwargs),
            type=None,
        )


def _concat_where_condition_from_domain(domain_expr: ir.Expr) -> ir.Expr:
    if cpm.is_call_to(domain_expr, "cartesian_domain") and len(domain_expr.args) > 1:
        axis_domains = []
        for range_expr in domain_expr.args:
            if cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3:
                if _get_axis_name(range_expr.args[0]) == "Kolor":
                    continue
                axis_domains.append(im.call("cartesian_domain")(copy.deepcopy(range_expr)))
        if not axis_domains:
            return domain_expr
        cond = axis_domains[0]
        for axis_domain in axis_domains[1:]:
            cond = im.and_(cond, axis_domain)
        return cond
    return domain_expr


# =====================================================================
# Pass 2: SetAtRemapper — General dispatcher + entity-specialized methods
# =====================================================================

@dataclasses.dataclass
class SetAtRemapper(NodeTranslator):
    """General dispatcher: remaps SetAt domains for all entity types.

    Entity-specialized methods:
      _remap_edge_setat()    — Edge (3 kolors), per-kolor split unless E2C2E/E2C2EO
      _remap_cell_setat()    — Cell (2 kolors), structured domain with 2-kolor range
      _remap_vertex_setat()  — Vertex (1 kolor), structured domain
    """

    # ── visit_Program: per-kolor split for eligible edge SetAts ──────────────

    @staticmethod
    def _mapping_enables_kolor_split(symbolic_domain_sizes: dict[str, Any]) -> bool:
        """Return True when edge_to_ijk mapping is present AND horizontal_start > 0."""
        if not _horizontal_start_mapping_enabled(symbolic_domain_sizes):
            return False
        mapping_rows = _coerce_mapping_rows(symbolic_domain_sizes.get("edge_to_ijk"), 3)
        if mapping_rows is None:
            return False
        hs = _pick_symbolic_int(symbolic_domain_sizes,
                                "horizontal_start_edge", "horizontal_start_e", "horizontal_start")
        return hs is not None and hs > 0

    def visit_Program(self, node: ir.Program, **kwargs) -> ir.Program:
        program_param_ids = {str(param.id) for param in node.params}
        child_kwargs = dict(kwargs)
        child_kwargs["program_param_ids"] = program_param_ids
        symbolic_domain_sizes: dict[str, Any] = kwargs.get("symbolic_domain_sizes") or {}

        new_body: list[ir.SetAt] = []
        for stmt in node.body:
            if (
                isinstance(stmt, ir.SetAt)
                and _detect_setat_entity(stmt.domain) == "Edge"
                and not _e2c2e_on_local_intermediate(stmt.expr)
                and self._mapping_enables_kolor_split(symbolic_domain_sizes)
            ):
                for k in range(3):
                    new_body.append(self.visit(stmt, current_kolor=k, **child_kwargs))
            else:
                new_body.append(self.visit(stmt, **child_kwargs))

        return ir.Program(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params,
            declarations=node.declarations,
            body=new_body,
        )

    # ── visit_SetAt: entity dispatch ─────────────────────────────────────────

    def visit_SetAt(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        entity = _detect_setat_entity(node.domain)
        if entity == "Edge":
            return self._remap_edge_setat(node, **kwargs)
        if entity == "Cell":
            return self._remap_cell_setat(node, **kwargs)
        if entity == "Vertex":
            return self._remap_vertex_setat(node, **kwargs)
        return self.generic_visit(node, **kwargs)

    # ── Edge-specialized (3 kolors) ──────────────────────────────────────────

    def _remap_edge_setat(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        """Edge-specialized: build structured domain, optionally split into per-kolor SetAt."""
        symbolic_domain_sizes: dict[str, Any] = kwargs.get("symbolic_domain_sizes") or {}
        program_param_ids: set[str] = kwargs.get("program_param_ids") or set()
        current_kolor: int | None = kwargs.get("current_kolor")

        full_domain = self._build_entity_structured_domain(
            "Edge", node.domain, symbolic_domain_sizes, program_param_ids
        )
        new_expr = self.visit(node.expr, current_domain=full_domain, **kwargs)
        new_target = self.visit(node.target, **kwargs)

        if _is_fully_structured_field(new_target):
            new_expr = _retarget_self_refs(new_expr, new_target)

        if current_kolor is not None:
            kolor_domain = self._build_edge_kolor_domain(
                current_kolor, full_domain, symbolic_domain_sizes
            )
            if kolor_domain is not None:
                return ir.SetAt(expr=new_expr, domain=kolor_domain, target=new_target)
        else:
            masked_expr = self._build_edge_validity_masked_expr(
                new_expr, new_target, full_domain, symbolic_domain_sizes
            )
            if masked_expr is not None:
                new_expr = masked_expr

        return ir.SetAt(expr=new_expr, domain=full_domain, target=new_target)

    def _build_edge_kolor_domain(
        self,
        kolor: int,
        full_structured_domain: ir.Expr,
        symbolic_domain_sizes: dict[str, Any],
    ) -> ir.Expr | None:
        """Build a kolor-specific SetAt domain from the full structured edge domain."""
        cart_domain = None
        if cpm.is_call_to(full_structured_domain, "make_tuple"):
            for a in full_structured_domain.args:
                if cpm.is_call_to(a, "cartesian_domain"):
                    cart_domain = a
                    break
        elif cpm.is_call_to(full_structured_domain, "cartesian_domain"):
            cart_domain = full_structured_domain

        if cart_domain is None:
            return None

        id_axis = j_axis = k_axis = None
        other_ranges: list[ir.Expr] = []
        i_lo = i_hi = j_lo = j_hi = None

        for nr in cart_domain.args:
            if _named_range_args(nr) is None:
                continue
            name = _get_axis_name(nr.args[0])
            if name == "IDim":
                id_axis, i_lo, i_hi = nr.args[0], nr.args[1], nr.args[2]
            elif name == "JDim":
                j_axis, j_lo, j_hi = nr.args[0], nr.args[1], nr.args[2]
            elif name == "Kolor":
                k_axis = nr.args[0]
            else:
                other_ranges.append(copy.deepcopy(nr))

        if id_axis is None or j_axis is None or k_axis is None:
            return None

        mapping_rows = _coerce_mapping_rows(symbolic_domain_sizes.get("edge_to_ijk"), 3)
        horizontal_start = _pick_symbolic_int(
            symbolic_domain_sizes, "horizontal_start_edge", "horizontal_start_e", "horizontal_start"
        )
        max_i_int = _pick_symbolic_int(symbolic_domain_sizes, "i_max", "domain_i_max", "max_i", "domain_max_i", "nx")
        max_j_int = _pick_symbolic_int(symbolic_domain_sizes, "j_max", "domain_j_max", "max_j", "domain_max_j", "ny")

        ilo = jlo = ihi = jhi = None

        if (
            _horizontal_start_mapping_enabled(symbolic_domain_sizes)
            and mapping_rows is not None
            and horizontal_start is not None
            and horizontal_start >= 0
            and max_i_int is not None and max_j_int is not None
        ):
            by_kolor = _derive_entity_start_bounds_from_mapping(
                "Edge", mapping_rows=mapping_rows, horizontal_start=horizontal_start,
                max_i=max_i_int, max_j=max_j_int,
            )
            if by_kolor is not None and kolor in by_kolor:
                k_ilo, k_jlo, k_ihi, k_jhi = by_kolor[kolor]
                ilo = ir.OffsetLiteral(value=k_ilo)
                jlo = ir.OffsetLiteral(value=k_jlo)
                ihi = ir.OffsetLiteral(value=k_ihi + 1)
                jhi = ir.OffsetLiteral(value=k_jhi + 1)

        if ilo is None:
            return None

        new_ranges = [
            im.named_range(cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(id_axis)), ilo, ihi),
            im.named_range(cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(j_axis)), jlo, jhi),
            im.named_range(
                cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(k_axis)),
                ir.OffsetLiteral(value=kolor),
                ir.OffsetLiteral(value=kolor + 1),
            ),
        ]
        new_ranges.extend(other_ranges)
        result_domain = im.call("cartesian_domain")(*new_ranges)
        if cpm.is_call_to(full_structured_domain, "make_tuple"):
            n = len(full_structured_domain.args)
            return im.make_tuple(*[copy.deepcopy(result_domain) for _ in range(n)])
        return result_domain

    def _build_edge_validity_masked_expr(
        self,
        expr: ir.Expr,
        target: ir.Expr,
        domain_expr: ir.Expr,
        symbolic_domain_sizes: dict[str, Any],
    ) -> ir.Expr | None:
        """Build 3-kolor concat_where mask for non-split edge SetAts.

        Returns a concat_where expression that restricts each kolor to its valid
        per-kolor (I, J) extent derived from the edge_to_ijk mapping.  Returns
        None when mapping data is absent or when the domain is not a 3-kolor edge
        domain — in both cases the caller leaves expr unchanged.
        """
        cart_domain = None
        for dom in _iter_domain_nodes(domain_expr):
            if cpm.is_call_to(dom, "cartesian_domain"):
                cart_domain = dom
                break
        if cart_domain is None:
            return None

        id_axis = j_axis = k_axis = None
        i_lo = i_hi = j_lo = j_hi = k_hi = None

        for nr in cart_domain.args:
            if _named_range_args(nr) is None:
                continue
            axis_name = _get_axis_name(nr.args[0])
            if axis_name == "IDim":
                id_axis, i_lo, i_hi = nr.args[0], nr.args[1], nr.args[2]
            elif axis_name == "JDim":
                j_axis, j_lo, j_hi = nr.args[0], nr.args[1], nr.args[2]
            elif axis_name == "Kolor":
                k_axis, _, k_hi = nr.args[0], nr.args[1], nr.args[2]

        if any(v is None for v in (id_axis, j_axis, k_axis, i_lo, i_hi, j_lo, j_hi, k_hi)):
            return None

        if isinstance(k_hi, ir.OffsetLiteral):
            if k_hi.value != 3:
                return None
        elif isinstance(k_hi, ir.Literal):
            if str(k_hi.value) != "3":
                return None
        else:
            return None

        def _dom(axis: ir.Expr, lo: ir.Expr, hi: ir.Expr) -> ir.Expr:
            return im.call("cartesian_domain")(
                im.named_range(cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(axis)),
                               copy.deepcopy(lo), copy.deepcopy(hi))
            )

        def _and(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
            return im.call("and_")(lhs, rhs)

        mapping_rows = _coerce_mapping_rows(symbolic_domain_sizes.get("edge_to_ijk"), 3)
        horizontal_start = _pick_symbolic_int(
            symbolic_domain_sizes, "horizontal_start_edge", "horizontal_start_e", "horizontal_start"
        )
        max_i_int = _pick_symbolic_int(symbolic_domain_sizes, "i_max", "domain_i_max", "max_i", "domain_max_i", "nx")
        max_j_int = _pick_symbolic_int(symbolic_domain_sizes, "j_max", "domain_j_max", "max_j", "domain_max_j", "ny")

        if (
            _horizontal_start_mapping_enabled(symbolic_domain_sizes)
            and mapping_rows is not None
            and horizontal_start is not None
            and horizontal_start >= 0
            and max_i_int is not None and max_j_int is not None
        ):
            by_kolor = _derive_entity_start_bounds_from_mapping(
                "Edge", mapping_rows=mapping_rows, horizontal_start=horizontal_start,
                max_i=max_i_int, max_j=max_j_int,
            )
            if by_kolor is not None and all(k in by_kolor for k in (0, 1, 2)):
                i0_lo, j0_lo, i0_hi, j0_hi = by_kolor[0]
                i1_lo, j1_lo, i1_hi, j1_hi = by_kolor[1]
                i2_lo, j2_lo, i2_hi, j2_hi = by_kolor[2]

                cond_k0 = _and(
                    _dom(copy.deepcopy(k_axis), ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1)),
                    _and(
                        _dom(copy.deepcopy(id_axis), ir.OffsetLiteral(value=i0_lo), ir.OffsetLiteral(value=i0_hi + 1)),
                        _dom(copy.deepcopy(j_axis), ir.OffsetLiteral(value=j0_lo), ir.OffsetLiteral(value=j0_hi + 1)),
                    ),
                )
                cond_k1 = _and(
                    _dom(copy.deepcopy(k_axis), ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2)),
                    _and(
                        _dom(copy.deepcopy(id_axis), ir.OffsetLiteral(value=i1_lo), ir.OffsetLiteral(value=i1_hi + 1)),
                        _dom(copy.deepcopy(j_axis), ir.OffsetLiteral(value=j1_lo), ir.OffsetLiteral(value=j1_hi + 1)),
                    ),
                )
                cond_k2 = _and(
                    _dom(copy.deepcopy(k_axis), ir.OffsetLiteral(value=2), ir.OffsetLiteral(value=3)),
                    _and(
                        _dom(copy.deepcopy(id_axis), ir.OffsetLiteral(value=i2_lo), ir.OffsetLiteral(value=i2_hi + 1)),
                        _dom(copy.deepcopy(j_axis), ir.OffsetLiteral(value=j2_lo), ir.OffsetLiteral(value=j2_hi + 1)),
                    ),
                )
                return im.concat_where(
                    cond_k0, copy.deepcopy(expr),
                    im.concat_where(
                        cond_k1, copy.deepcopy(expr),
                        im.concat_where(cond_k2, copy.deepcopy(expr), copy.deepcopy(target)),
                    ),
                )

        # Fallback: no mapping available, return None (domain unchanged)
        return None

    # ── Cell-specialized (2 kolors) ───────────────────────────────────────────

    def _remap_cell_setat(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        """Cell-specialized: build structured 2-kolor domain."""
        symbolic_domain_sizes: dict[str, Any] = kwargs.get("symbolic_domain_sizes") or {}
        program_param_ids: set[str] = kwargs.get("program_param_ids") or set()

        full_domain = self._build_entity_structured_domain(
            "Cell", node.domain, symbolic_domain_sizes, program_param_ids
        )
        new_expr = self.visit(node.expr, current_domain=full_domain, **kwargs)
        new_target = self.visit(node.target, **kwargs)
        if _is_fully_structured_field(new_target):
            new_expr = _retarget_self_refs(new_expr, new_target)
        return ir.SetAt(expr=new_expr, domain=full_domain, target=new_target)

    # ── Vertex-specialized (1 kolor) ─────────────────────────────────────────

    def _remap_vertex_setat(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        """Vertex-specialized: build structured 1-kolor domain."""
        symbolic_domain_sizes: dict[str, Any] = kwargs.get("symbolic_domain_sizes") or {}
        program_param_ids: set[str] = kwargs.get("program_param_ids") or set()

        full_domain = self._build_entity_structured_domain(
            "Vertex", node.domain, symbolic_domain_sizes, program_param_ids
        )
        new_expr = self.visit(node.expr, current_domain=full_domain, **kwargs)
        new_target = self.visit(node.target, **kwargs)
        if _is_fully_structured_field(new_target):
            new_expr = _retarget_self_refs(new_expr, new_target)
        return ir.SetAt(expr=new_expr, domain=full_domain, target=new_target)

    # ── Shared: unstructured_domain → cartesian_domain ───────────────────────

    def _build_entity_structured_domain(
        self,
        entity_name: str,
        orig_domain: ir.Expr,
        symbolic_domain_sizes: dict[str, Any],
        program_param_ids: set[str],
    ) -> ir.Expr:
        """Convert unstructured_domain(entity, K) to cartesian_domain(IDim, JDim, Kolor, K)."""
        def _convert_single(dom: ir.Expr) -> ir.Expr:
            if not cpm.is_call_to(dom, "unstructured_domain"):
                return copy.deepcopy(dom)
            new_ranges: list[ir.Expr] = []
            for nr in dom.args:
                if _named_range_args(nr) is None:
                    new_ranges.append(copy.deepcopy(nr))
                    continue
                axis_name = _get_axis_name(nr.args[0])
                if axis_name == entity_name:
                    idim_b = _entity_cartesian_bounds(entity_name, "IDim", program_param_ids, symbolic_domain_sizes)
                    jdim_b = _entity_cartesian_bounds(entity_name, "JDim", program_param_ids, symbolic_domain_sizes)
                    kolor_b = _entity_kolor_bounds(entity_name)
                    if idim_b and jdim_b and kolor_b:
                        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
                        JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
                        Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
                        new_ranges.extend([
                            im.named_range(IDim, *idim_b),
                            im.named_range(JDim, *jdim_b),
                            im.named_range(Kolor, *kolor_b),
                        ])
                    else:
                        new_ranges.append(copy.deepcopy(nr))
                else:
                    new_ranges.append(copy.deepcopy(nr))
            return im.call("cartesian_domain")(*new_ranges)

        if cpm.is_call_to(orig_domain, "make_tuple"):
            converted = [_convert_single(sub) for sub in orig_domain.args]
            return im.make_tuple(*converted)
        return _convert_single(orig_domain)

    # ── visit_FunCall: convert unstructured_domain in stencil bodies ──────────

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        """Minimal FunCall handler: converts unstructured_domain in as_fieldop bodies."""
        symbolic_domain_sizes: dict[str, Any] = kwargs.get("symbolic_domain_sizes") or {}
        program_param_ids: set[str] = kwargs.get("program_param_ids") or set()

        if cpm.is_call_to(node, "unstructured_domain") or (
            cpm.is_call_to(node, "cartesian_domain")
            and any(_get_axis_name(nr.args[0]) in {"Edge", "Cell", "Vertex"}
                    for nr in node.args
                    if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3)
        ):
            new_ranges: list[ir.Expr] = []
            needs_remap = False
            for nr in node.args:
                if _named_range_args(nr) is not None:
                    axis_name = _get_axis_name(nr.args[0])
                    if axis_name in _UNSTRUCTURED_AXES:
                        idim_b = _entity_cartesian_bounds(axis_name, "IDim", program_param_ids, symbolic_domain_sizes)
                        jdim_b = _entity_cartesian_bounds(axis_name, "JDim", program_param_ids, symbolic_domain_sizes)
                        kolor_b = _entity_kolor_bounds(axis_name)
                        if idim_b and jdim_b and kolor_b:
                            IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
                            JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
                            Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
                            new_ranges.extend([
                                im.named_range(IDim, *idim_b),
                                im.named_range(JDim, *jdim_b),
                                im.named_range(Kolor, *kolor_b),
                            ])
                            needs_remap = True
                            continue
                new_ranges.append(self.generic_visit(nr, **kwargs))
            if needs_remap:
                return im.call("cartesian_domain")(*new_ranges)

        return self.generic_visit(node, **kwargs)

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        symbolic_domain_sizes: dict[str, Any] | None = None,
        **kwargs,
    ) -> ir.Node:
        return cls().visit(node, symbolic_domain_sizes=symbolic_domain_sizes or {}, **kwargs)


# =====================================================================
# Pass 3: BroadcastAxisExpander — General
# =====================================================================

@dataclasses.dataclass
class BroadcastAxisExpander(NodeTranslator):
    """General: Expands unstructured broadcast axes to structured Cartesian axes.

    broadcast(v, [Edge, K]) → broadcast(v, [IDim, JDim, Kolor, K])
    Works for Edge, Cell, and Vertex broadcast axes.
    """

    @staticmethod
    def _structured_axis_literals() -> list[ir.Expr]:
        return [
            ir.AxisLiteral(value="IDim", kind=common.DimensionKind.HORIZONTAL),
            ir.AxisLiteral(value="JDim", kind=common.DimensionKind.HORIZONTAL),
            ir.AxisLiteral(value="Kolor", kind=common.DimensionKind.HORIZONTAL),
        ]

    @staticmethod
    def _expand_unstructured_axis(axis_expr: ir.Expr) -> list[ir.Expr]:
        if _get_axis_name(axis_expr) in _UNSTRUCTURED_AXES:
            return BroadcastAxisExpander._structured_axis_literals()
        return [copy.deepcopy(axis_expr)]

    @staticmethod
    def _remap_broadcast_axes(axes_expr: ir.Expr) -> ir.Expr:
        if cpm.is_call_to(axes_expr, "make_tuple"):
            remapped: list[ir.Expr] = []
            for arg in axes_expr.args:
                remapped.extend(BroadcastAxisExpander._expand_unstructured_axis(arg))
            return im.call("make_tuple")(*remapped)
        expanded = BroadcastAxisExpander._expand_unstructured_axis(axes_expr)
        if len(expanded) == 1:
            return expanded[0]
        return im.call("make_tuple")(*expanded)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        new_node = self.generic_visit(node, **kwargs)
        if cpm.is_call_to(new_node, "broadcast") and len(new_node.args) == 2:
            remapped_axes = self._remap_broadcast_axes(new_node.args[1])
            return ir.FunCall(
                fun=new_node.fun, args=[new_node.args[0], remapped_axes], type=new_node.type
            )
        return new_node

    @classmethod
    def apply(cls, node: ir.Node, **kwargs) -> ir.Node:
        return cls().visit(node, **kwargs)


# =====================================================================
# Pass 4: ThresholdConditionRewriter — General dispatcher + entity-specialized
# =====================================================================

@dataclasses.dataclass
class ThresholdConditionRewriter(NodeTranslator):
    """General dispatcher: rewrites threshold comparisons to per-kolor structured conditions.

    Entity-specialized methods:
      _rewrite_edge_threshold()        — edge start-only thresholds
      _rewrite_edge_range_threshold()  — edge and_(start, end) ranges
      _rewrite_cell_threshold()        — cell lateral_boundary thresholds
    """

    @staticmethod
    def _mapping_based_threshold_condition(
        tid: str, n_kolors: int, symbolic_domain_sizes: dict[str, Any],
    ) -> ir.Expr | None:
        IDim_d = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        JDim_d = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        Kolor_d = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

        def _dom(axis, lo_val: int, hi_val: int) -> ir.Expr:
            return im.call("cartesian_domain")(
                im.named_range(copy.deepcopy(axis),
                               ir.OffsetLiteral(value=lo_val), ir.OffsetLiteral(value=hi_val))
            )

        conds = []
        for k in range(n_kolors):
            ilo = symbolic_domain_sizes.get(f"{tid}_k{k}_ilo")
            jlo = symbolic_domain_sizes.get(f"{tid}_k{k}_jlo")
            ihi = symbolic_domain_sizes.get(f"{tid}_k{k}_ihi")
            jhi = symbolic_domain_sizes.get(f"{tid}_k{k}_jhi")
            if None in (ilo, jlo, ihi, jhi):
                return None
            conds.append(im.and_(
                _dom(Kolor_d, k, k + 1),
                im.and_(_dom(IDim_d, int(ilo), int(ihi)), _dom(JDim_d, int(jlo), int(jhi))),
            ))
        if len(conds) == 2:
            return im.or_(conds[0], conds[1])
        if len(conds) == 3:
            return im.or_(conds[0], im.or_(conds[1], conds[2]))
        return None

    @staticmethod
    def _structured_entity_condition(
        entity_name: str,
        program_param_ids: set[str],
        symbolic_domain_sizes: dict[str, Any],
        extra_halo: int = 0,
    ) -> ir.Expr | None:
        idim_bounds = _entity_cartesian_bounds(entity_name, "IDim", program_param_ids, symbolic_domain_sizes)
        jdim_bounds = _entity_cartesian_bounds(entity_name, "JDim", program_param_ids, symbolic_domain_sizes)
        kolor_bounds = _entity_kolor_bounds(entity_name)
        if idim_bounds is None or jdim_bounds is None or kolor_bounds is None:
            return None

        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

        def _axis_domain(axis: common.Dimension, lo: ir.Expr, hi: ir.Expr) -> ir.Expr:
            return im.call("cartesian_domain")(
                im.named_range(copy.deepcopy(axis), copy.deepcopy(lo), copy.deepcopy(hi))
            )

        # Edge with extra_halo: try per-kolor mapping bounds first
        if entity_name == "Edge" and extra_halo > 0:
            mapping_k_bounds = {}
            all_present = True
            for k in range(3):
                ib = _entity_cartesian_bounds(entity_name, "IDim", program_param_ids, symbolic_domain_sizes, kolor=k)
                jb = _entity_cartesian_bounds(entity_name, "JDim", program_param_ids, symbolic_domain_sizes, kolor=k)
                if ib is None or jb is None:
                    all_present = False
                    break
                mapping_k_bounds[k] = (ib, jb)
            if all_present:
                conds = []
                for k in range(3):
                    ib, jb = mapping_k_bounds[k]
                    conds.append(im.and_(
                        _axis_domain(Kolor, ir.OffsetLiteral(value=k), ir.OffsetLiteral(value=k + 1)),
                        im.and_(_axis_domain(IDim, *ib), _axis_domain(JDim, *jb)),
                    ))
                return im.or_(conds[0], im.or_(conds[1], conds[2]))

        i_lo, i_hi = idim_bounds
        j_lo, j_hi = jdim_bounds
        if extra_halo > 0:
            extra = ir.OffsetLiteral(value=extra_halo)
            i_lo = _offset_add(i_lo, extra)
            i_hi = _offset_sub(i_hi, extra)
            j_lo = _offset_add(j_lo, extra)
            j_hi = _offset_sub(j_hi, extra)

        domain_expr = im.call("cartesian_domain")(
            im.named_range(IDim, i_lo, i_hi),
            im.named_range(JDim, j_lo, j_hi),
            im.named_range(Kolor, *kolor_bounds),
        )
        return _concat_where_condition_from_domain(domain_expr)

    # ── Edge-specialized ─────────────────────────────────────────────────────

    def _rewrite_edge_threshold(
        self, threshold_id: str, is_positive: bool,
        symbolic_domain_sizes: dict[str, Any], program_param_ids: set[str],
    ) -> ir.Expr | None:
        if f"{threshold_id}_k0_ilo" in symbolic_domain_sizes:
            interior_cond = self._mapping_based_threshold_condition(threshold_id, 3, symbolic_domain_sizes)
        elif threshold_id == _EDGE_NUDGE_THRESHOLD_ID:
            interior_cond = self._structured_entity_condition("Edge", program_param_ids, symbolic_domain_sizes, extra_halo=2)
        else:
            interior_cond = self._structured_entity_condition("Edge", program_param_ids, symbolic_domain_sizes)
        if interior_cond is None:
            return None
        return interior_cond if is_positive else im.not_(interior_cond)

    def _rewrite_edge_range_threshold(
        self, start_id: str, end_id: str, symbolic_domain_sizes: dict[str, Any],
    ) -> ir.Expr | None:
        """Edge-specialized: rewrite and_(Edge >= start, Edge < end) range pattern."""
        range_key = f"{start_id}|{end_id}"
        if f"{range_key}_k0_ilo" not in symbolic_domain_sizes:
            return None
        IDim_d = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        JDim_d = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        Kolor_d = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

        def _rdom(axis, lo: int, hi: int) -> ir.Expr:
            return im.call("cartesian_domain")(
                im.named_range(copy.deepcopy(axis),
                               ir.OffsetLiteral(value=lo), ir.OffsetLiteral(value=hi))
            )

        conds = []
        for k in range(3):
            ilo = symbolic_domain_sizes.get(f"{range_key}_k{k}_ilo")
            jlo = symbolic_domain_sizes.get(f"{range_key}_k{k}_jlo")
            ihi = symbolic_domain_sizes.get(f"{range_key}_k{k}_ihi")
            jhi = symbolic_domain_sizes.get(f"{range_key}_k{k}_jhi")
            if None in (ilo, jlo, ihi, jhi):
                break
            conds.append(im.and_(
                _rdom(Kolor_d, k, k + 1),
                im.and_(_rdom(IDim_d, int(ilo), int(ihi)), _rdom(JDim_d, int(jlo), int(jhi))),
            ))
        if len(conds) == 3:
            return im.or_(conds[0], im.or_(conds[1], conds[2]))
        if len(conds) == 2:
            return im.or_(conds[0], conds[1])
        if len(conds) == 1:
            return conds[0]
        return None

    # ── Cell-specialized ─────────────────────────────────────────────────────

    def _rewrite_cell_threshold(
        self, threshold_id: str, is_positive: bool, is_interior: bool, is_halo: bool,
        symbolic_domain_sizes: dict[str, Any], program_param_ids: set[str],
    ) -> ir.Expr | None:
        if f"{threshold_id}_k0_ilo" in symbolic_domain_sizes:
            interior_cond = self._mapping_based_threshold_condition(threshold_id, 2, symbolic_domain_sizes)
        elif is_interior:
            interior_cond = self._structured_entity_condition("Cell", program_param_ids, symbolic_domain_sizes, extra_halo=1)
        else:
            interior_cond = self._structured_entity_condition("Cell", program_param_ids, symbolic_domain_sizes)
        if interior_cond is None:
            return None
        return interior_cond if is_positive else im.not_(interior_cond)

    # ── visit_FunCall: detect entity and dispatch ─────────────────────────────

    @staticmethod
    def _extract_edge_cmp(expr: ir.Expr):
        if not (
            cpm.is_call_to(expr, ("greater_equal", "greater", "less_equal", "less"))
            and len(expr.args) == 2
        ):
            return None
        lhs, rhs = expr.args
        lhs_ax, rhs_ax = _get_axis_name(lhs), _get_axis_name(rhs)
        if lhs_ax == "Edge":
            side, thresh = "lhs", rhs
        elif rhs_ax == "Edge":
            side, thresh = "rhs", lhs
        else:
            return None
        tid = str(thresh.id) if isinstance(thresh, ir.SymRef) else None
        return (side, str(expr.fun.id), tid) if tid is not None else None

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        symbolic_domain_sizes: dict[str, Any] = kwargs.get("symbolic_domain_sizes") or {}
        program_param_ids: set[str] = kwargs.get("program_param_ids") or set()

        # Edge range: and_(Edge >= start, Edge < end) — must intercept before generic_visit
        if cpm.is_call_to(node, "and_") and len(node.args) == 2 and symbolic_domain_sizes:
            cmp_a = self._extract_edge_cmp(node.args[0])
            cmp_b = self._extract_edge_cmp(node.args[1])
            if cmp_a is not None and cmp_b is not None:
                for ca, cb in ((cmp_a, cmp_b), (cmp_b, cmp_a)):
                    _, _, start_id = ca
                    _, _, end_id = cb
                    result = self._rewrite_edge_range_threshold(start_id, end_id, symbolic_domain_sizes)
                    if result is not None:
                        return result

        new_node = self.generic_visit(node, **kwargs)

        if (
            cpm.is_call_to(new_node, ("greater_equal", "greater", "less_equal", "less"))
            and len(new_node.args) == 2
        ):
            lhs, rhs = new_node.args
            lhs_axis = _get_axis_name(lhs)
            rhs_axis = _get_axis_name(rhs)
            axis_side = entity_axis = None

            if lhs_axis in {"Edge", "Cell"}:
                axis_side, entity_axis = "lhs", lhs_axis
            elif rhs_axis in {"Edge", "Cell"}:
                axis_side, entity_axis = "rhs", rhs_axis

            if entity_axis is not None:
                op_name = str(new_node.fun.id)
                threshold_expr = rhs if axis_side == "lhs" else lhs
                threshold_id = str(threshold_expr.id) if isinstance(threshold_expr, ir.SymRef) else None
                if threshold_id is None:
                    return new_node

                def _has_sym(expr: ir.Expr, sym: str) -> bool:
                    """Return True if expr contains a SymRef with id==sym (early-exit via `or`/`any`)."""
                    if isinstance(expr, ir.SymRef):
                        return str(expr.id) == sym
                    if isinstance(expr, ir.FunCall):
                        return _has_sym(expr.fun, sym) or any(_has_sym(a, sym) for a in expr.args)
                    return False

                is_interior = threshold_id == "interior_idx" or _has_sym(threshold_expr, "interior_idx")
                is_halo = threshold_id == "halo_idx" or _has_sym(threshold_expr, "halo_idx")

                # is_positive: True when the comparison means "entity_index >= threshold"
                # (lhs side with >= / > operator, or rhs side with <= / < operator).
                # For halo thresholds the sense is inverted (entity_index <= halo_bound).
                _interior_positive = (
                    (axis_side == "lhs" and op_name in {"greater_equal", "greater"})
                    or (axis_side == "rhs" and op_name in {"less_equal", "less"})
                )

                if entity_axis == "Edge":
                    result = self._rewrite_edge_threshold(
                        threshold_id, _interior_positive, symbolic_domain_sizes, program_param_ids
                    )
                    if result is not None:
                        return result

                elif entity_axis == "Cell":
                    is_positive = not _interior_positive if is_halo else _interior_positive
                    result = self._rewrite_cell_threshold(
                        threshold_id, is_positive, is_interior, is_halo,
                        symbolic_domain_sizes, program_param_ids
                    )
                    if result is not None:
                        return result

        return new_node

    @classmethod
    def apply(
        cls, node: ir.Node, *, symbolic_domain_sizes: dict[str, Any] | None = None, **kwargs,
    ) -> ir.Node:
        return cls().visit(node, symbolic_domain_sizes=symbolic_domain_sizes or {}, **kwargs)


# =====================================================================
# Pass 5: SymbolicSizeInliner — General
# =====================================================================

@dataclasses.dataclass
class SymbolicSizeInliner(NodeTranslator):
    """General: Inlines symbolic domain size values into domain range expressions.

    Handles:
      get_domain_range(axis, field)            → (lo, hi) make_tuple
      tuple_get(idx, get_domain_range(...))    → individual bound
      tuple_get(idx, make_tuple(...))          → direct element
    """

    def visit_Program(self, node: ir.Program, **kwargs) -> ir.Program:
        program_param_ids = {str(param.id) for param in node.params}
        child_kwargs = dict(kwargs)
        child_kwargs["program_param_ids"] = program_param_ids
        new_body = [self.visit(stmt, **child_kwargs) for stmt in node.body]
        return ir.Program(
            id=node.id, function_definitions=node.function_definitions,
            params=node.params, declarations=node.declarations, body=new_body,
        )

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        symbolic_domain_sizes: dict[str, Any] = kwargs.get("symbolic_domain_sizes") or {}
        program_param_ids: set[str] = kwargs.get("program_param_ids") or set()

        new_node = self.generic_visit(node, **kwargs)

        # get_domain_range(axis, field) → (lo, hi)
        if cpm.is_call_to(new_node, "get_domain_range") and len(new_node.args) == 2:
            axis_name = _get_axis_name(new_node.args[1])
            if axis_name is not None:
                bounds = _cartesian_axis_bounds(axis_name, program_param_ids, symbolic_domain_sizes)
                if bounds is not None:
                    return im.make_tuple(*bounds)

        # tuple_get(idx, make_tuple(...)) or tuple_get(idx, get_domain_range(...))
        # Accept both ir.Literal (from im.tuple_get) and ir.OffsetLiteral as the index.
        if cpm.is_call_to(new_node, "tuple_get") and len(new_node.args) == 2:
            tuple_index, inner = new_node.args
            idx_val = _extract_index_value(tuple_index)
            # tuple_get(idx, make_tuple(...)) → direct element
            if cpm.is_call_to(inner, "make_tuple") and idx_val is not None:
                if 0 <= idx_val < len(inner.args):
                    return copy.deepcopy(inner.args[idx_val])
            # tuple_get(idx, get_domain_range(axis, field)) → individual bound
            if cpm.is_call_to(inner, "get_domain_range") and len(inner.args) == 2:
                axis_name = _get_axis_name(inner.args[1])
                if axis_name is not None and idx_val in {0, 1}:
                    bounds = _cartesian_axis_bounds(axis_name, program_param_ids, symbolic_domain_sizes)
                    if bounds is not None:
                        return copy.deepcopy(bounds[idx_val])

        return new_node

    @classmethod
    def apply(
        cls, node: ir.Node, *, symbolic_domain_sizes: dict[str, Any] | None = None, **kwargs,
    ) -> ir.Node:
        return cls().visit(node, symbolic_domain_sizes=symbolic_domain_sizes or {}, **kwargs)



# =====================================================================
# Pass 6: NeighborReductionUnroller — General + edge-specialized kolor peeling
# =====================================================================

@dataclasses.dataclass
class NeighborReductionUnroller(NodeTranslator):
    """General: Unrolls reduce(op)(neighbors(conn, it), ...) into explicit Cartesian shifts.

    Edge-specialized behavior in visit_SetAt:
      Peels outer kolor concat_where branches for non-split edge SetAts so inner
      shifts resolve per-kolor (only when E2C2EO is not on a local intermediate).
    """

    @staticmethod
    def _collect_neighbor_tags(expr: ir.Expr) -> set[str]:
        """Return all connectivity names used in neighbors() calls within expr."""
        tags: set[str] = set()
        for call in expr.pre_walk_values().if_isinstance(ir.FunCall):
            if cpm.is_call_to(call, "neighbors") and len(call.args) == 2:
                offset = call.args[0]
                if isinstance(offset, ir.OffsetLiteral) and isinstance(offset.value, str):
                    tags.add(offset.value)
        return tags

    @staticmethod
    def _mapped_connection_size(expr: ir.Expr) -> int | None:
        tags = NeighborReductionUnroller._collect_neighbor_tags(expr)
        if len(tags) != 1:
            return None
        conn = next(iter(tags))
        idx_values = sorted(
            key[1].value
            for key in map_dict
            if key[0].value == conn and isinstance(key[1].value, int)
        )
        # Verify slots are a contiguous range [0, N). `list(range(0)) == []` handles empty.
        if idx_values != list(range(len(idx_values))):
            return None
        return len(idx_values)

    @staticmethod
    def _extract_generic_reduce_inputs(expr: ir.Expr) -> tuple[ir.Expr, ir.Expr, ir.Expr, int] | None:
        if not cpm.is_applied_as_fieldop(expr) or len(expr.args) != 1:
            return None
        reduce_stencil = expr.fun.args[0]
        if not (
            isinstance(reduce_stencil, ir.Lambda)
            and len(reduce_stencil.params) == 1
            and cpm.is_applied_reduce(reduce_stencil.expr)
        ):
            return None
        if len(reduce_stencil.expr.fun.args) != 2:
            return None
        red_op, red_init = reduce_stencil.expr.fun.args
        if len(reduce_stencil.expr.args) != 1:
            return None
        red_arg = reduce_stencil.expr.args[0]
        if not (
            cpm.is_call_to(red_arg, "deref")
            and len(red_arg.args) == 1
            and isinstance(red_arg.args[0], ir.SymRef)
            and red_arg.args[0].id == reduce_stencil.params[0].id
        ):
            return None
        list_expr = expr.args[0]
        if (conn_size := NeighborReductionUnroller._mapped_connection_size(list_expr)) is None:
            return None
        return red_op, red_init, list_expr, conn_size

    @staticmethod
    def _local_list_element_expr(
        expr: ir.Expr,
        index: int,
        bindings: dict[str, dict[str, ir.Expr | str]],
        neutral_element: ir.Expr,
        domain_bounds: dict[str, tuple[ir.Expr, ir.Expr]] | None = None,
    ) -> ir.Expr | None:
        def _local_concat_where(base_ref: ir.Expr, branches: tuple) -> ir.Expr:
            cond, shift_spec = branches[0]
            shifted = _bounded_shifted_deref(base_ref, shift_spec, neutral_element, domain_bounds)
            if len(branches) == 1 or cond is None:
                return shifted
            return im.concat_where(
                copy.deepcopy(cond), shifted, _local_concat_where(base_ref, branches[1:])
            )

        if cpm.is_call_to(expr, "neighbors") and len(expr.args) == 2:
            conn, it = expr.args
            if (
                isinstance(conn, ir.OffsetLiteral) and isinstance(conn.value, str)
                and isinstance(it, ir.SymRef) and str(it.id) in bindings
            ):
                key: tuple[ir.OffsetLiteral, ir.OffsetLiteral] = (
                    ir.OffsetLiteral(value=conn.value), ir.OffsetLiteral(value=index),
                )
                if key in map_dict:
                    entry = cast(dict[str, object], map_dict[key])
                    ref = cast(ir.Expr, copy.deepcopy(bindings[str(it.id)]["ref"]))
                    if entry["kind"] == "shift":
                        return _bounded_shifted_deref(
                            ref, cast(tuple[ir.OffsetLiteral, ...], entry["shifts"]),
                            neutral_element, domain_bounds,
                        )
                    if entry["kind"] == "concat_where":
                        return _local_concat_where(ref, cast(tuple, entry["branches"]))
            return None

        if cpm.is_call_to(expr, "deref") and len(expr.args) == 1:
            it = expr.args[0]
            if isinstance(it, ir.SymRef) and str(it.id) in bindings:
                binding = bindings[str(it.id)]
                ref = cast(ir.Expr, copy.deepcopy(binding["ref"]))
                kind = binding["kind"]
                if kind == "neighbors":
                    bound_conn = binding["conn"]
                    assert isinstance(bound_conn, str)
                    key = (ir.OffsetLiteral(value=bound_conn), ir.OffsetLiteral(value=index))
                    if key in map_dict:
                        entry = cast(dict[str, object], map_dict[key])
                        if entry["kind"] == "shift":
                            return _bounded_shifted_deref(
                                ref, cast(tuple[ir.OffsetLiteral, ...], entry["shifts"]),
                                neutral_element, domain_bounds,
                            )
                        if entry["kind"] == "concat_where":
                            return _local_concat_where(ref, cast(tuple, entry["branches"]))
                    return None
                return im.list_get(im.literal(str(index), "int32"), im.deref(ref))
            return None

        if cpm.is_applied_map(expr):
            if len(expr.fun.args) != 1:
                return None
            mapped_op = copy.deepcopy(expr.fun.args[0])
            elem_args: list[ir.Expr] = []
            for arg in expr.args:
                elem = NeighborReductionUnroller._local_list_element_expr(
                    arg, index, bindings, neutral_element, domain_bounds
                )
                if elem is None:
                    return None
                elem_args.append(elem)
            return im.call(mapped_op)(*elem_args)

        if cpm.is_call_to(expr, "if_") and len(expr.args) == 3:
            cond, true_val, false_val = expr.args
            true_elem = NeighborReductionUnroller._local_list_element_expr(
                true_val, index, bindings, neutral_element, domain_bounds
            )
            false_elem = NeighborReductionUnroller._local_list_element_expr(
                false_val, index, bindings, neutral_element, domain_bounds
            )
            if true_elem is None or false_elem is None:
                return None
            return im.if_(copy.deepcopy(cond), true_elem, false_elem)

        return None

    @classmethod
    def _eval_list_field_at_idx(
        cls,
        expr: ir.Expr,
        idx: int,
        domain: ir.Expr | None,
        neutral_element: ir.Expr,
        current_kolor: int | None = None,
    ) -> ir.Expr | None:
        if cpm.is_applied_as_fieldop(expr):
            stencil = expr.fun.args[0]
            if isinstance(stencil, ir.Lambda):
                if cpm.is_call_to(stencil.expr, "neighbors") and len(stencil.expr.args) == 2:
                    conn_expr = stencil.expr.args[0]
                    if isinstance(conn_expr, ir.OffsetLiteral) and isinstance(conn_expr.value, str):
                        conn = conn_expr.value
                        field_expr = expr.args[0]
                        key: tuple[ir.OffsetLiteral, ir.OffsetLiteral] = (
                            ir.OffsetLiteral(value=conn), ir.OffsetLiteral(value=idx),
                        )
                        if key not in map_dict:
                            return None
                        entry = cast(dict[str, object], map_dict[key])
                        if entry["kind"] == "shift":
                            return _make_lifted_deref_shift(
                                field_expr,
                                cast(tuple[ir.OffsetLiteral, ...], entry["shifts"]),
                                domain,
                            )
                        if entry["kind"] == "concat_where":
                            _, is_ene = _conn_name_and_is_edge_to_non_edge(
                                (ir.OffsetLiteral(value=conn), ir.OffsetLiteral(value=0))
                            )
                            return _build_field_concat_where_from_branches(
                                field_expr,
                                cast(tuple, entry["branches"]),
                                domain,
                                apply_edge_shape_bounds=_needs_edge_shape_bounds(conn),
                                current_kolor=current_kolor if is_ene else None,
                            )

                if isinstance(stencil.expr, ir.FunCall) and cpm.is_call_to(stencil.expr.fun, "map_"):
                    mapped_op = stencil.expr.fun.args[0]
                    new_args = []
                    for arg in expr.args:
                        inner_elem = cls._eval_list_field_at_idx(
                            arg, idx, domain, neutral_element, current_kolor=current_kolor
                        )
                        if inner_elem is None:
                            return None
                        new_args.append(inner_elem)
                    param_names = [f"__cart_eval_arg{i}" for i in range(len(new_args))]
                    deref_args = [im.deref(p) for p in param_names]
                    scalar_op_call = im.call(copy.deepcopy(mapped_op))(*deref_args)
                    return im.as_fieldop(im.lambda_(*param_names)(scalar_op_call), domain)(*new_args)

        return im.as_fieldop(
            im.lambda_("__cart_lst")(
                im.list_get(im.literal(str(idx), "int32"), im.deref("__cart_lst"))
            ),
            domain,
        )(copy.deepcopy(expr))

    @staticmethod
    def _build_generic_unrolled_reduce_expr(
        red_op: ir.Expr,
        red_init: ir.Expr,
        list_expr: ir.Expr,
        conn_size: int,
        domain: ir.Expr | None,
        current_kolor: int | None = None,
    ) -> ir.Expr:
        domain_bounds: dict[str, tuple[ir.Expr, ir.Expr]] = {}
        if cpm.is_call_to(domain, "cartesian_domain"):
            for range_expr in domain.args:
                nra = _named_range_args(range_expr)
                if nra is not None:
                    axis_name = _get_axis_name(nra[0])
                    if axis_name in {"IDim", "JDim"}:
                        domain_bounds[axis_name] = (
                            copy.deepcopy(nra[1]),
                            copy.deepcopy(nra[2]),
                        )

        widened_init = copy.deepcopy(red_init)
        if isinstance(red_init, ir.Literal) and isinstance(red_init.type, ts.ScalarType):
            if red_init.type.kind in {
                ts.ScalarKind.INT8, ts.ScalarKind.INT16,
                ts.ScalarKind.INT32, ts.ScalarKind.INT64,
            }:
                widened_init = im.literal(red_init.value, "int64")
        elif isinstance(red_init, ir.OffsetLiteral) and isinstance(red_init.value, int):
            widened_init = im.literal(str(red_init.value), "int64")

        neutral_element = _neutral_element_for_reduce_op(red_op, widened_init)
        # If the reduction init is an integer literal but the neutral element (from
        # _neutral_element_for_reduce_op) is float64 — e.g. for plus/max/min — replace
        # widened_init with the neutral element so the accumulator starts at the correct
        # float64 identity rather than an int.  This prevents type mismatches in the
        # as_fieldop accumulation lambda.
        if (
            isinstance(widened_init, ir.Literal)
            and isinstance(widened_init.type, ts.ScalarType)
            and widened_init.type.kind in {
                ts.ScalarKind.INT8, ts.ScalarKind.INT16,
                ts.ScalarKind.INT32, ts.ScalarKind.INT64,
            }
            and isinstance(neutral_element, ir.Literal)
            and isinstance(neutral_element.type, ts.ScalarType)
            and neutral_element.type.kind == ts.ScalarKind.FLOAT64
        ):
            widened_init = copy.deepcopy(neutral_element)

        # Fast-path: direct neighbors reduction
        if cpm.is_call_to(list_expr, "neighbors") and len(list_expr.args) == 2:
            conn_expr = list_expr.args[0]
            if isinstance(conn_expr, ir.OffsetLiteral) and isinstance(conn_expr.value, str):
                conn = conn_expr.value
                input_field = copy.deepcopy(list_expr.args[1])
                acc_field = im.as_fieldop(
                    im.lambda_("__acc_init")(im.deref("__acc_init")), domain,
                )(im.as_fieldop(im.lambda_("__x")(copy.deepcopy(widened_init)), domain)(input_field))
                for idx in range(conn_size):
                    key: tuple[ir.OffsetLiteral, ir.OffsetLiteral] = (
                        ir.OffsetLiteral(value=conn), ir.OffsetLiteral(value=idx),
                    )
                    if key not in map_dict:
                        break
                    entry = cast(dict[str, object], map_dict[key])
                    if entry["kind"] != "shift":
                        break
                    elem_field = _make_lifted_deref_shift(
                        input_field, cast(tuple[ir.OffsetLiteral, ...], entry["shifts"]), domain
                    )
                    acc_field = im.as_fieldop(
                        im.lambda_("__a", "__b")(
                            im.call(copy.deepcopy(red_op))(im.deref("__a"), im.deref("__b"))
                        ),
                        domain,
                    )(acc_field, elem_field)
                else:
                    return acc_field

        # Fast-path: fieldop list via map_dict
        if cpm.is_applied_as_fieldop(list_expr):
            def _extract_base_field(node: ir.Expr) -> ir.Expr:
                if cpm.is_applied_as_fieldop(node):
                    return _extract_base_field(node.args[0])
                return node

            base_field = copy.deepcopy(_extract_base_field(list_expr))
            acc_field = im.as_fieldop(
                im.lambda_("__acc_init")(im.deref("__acc_init")), domain,
            )(im.as_fieldop(im.lambda_("__x")(copy.deepcopy(widened_init)), domain)(base_field))
            for idx in range(conn_size):
                elem_field = NeighborReductionUnroller._eval_list_field_at_idx(
                    list_expr, idx, domain, neutral_element, current_kolor=current_kolor
                )
                if elem_field is None:
                    break
                acc_field = im.as_fieldop(
                    im.lambda_("__a", "__b")(
                        im.call(copy.deepcopy(red_op))(im.deref("__a"), im.deref("__b"))
                    ),
                    domain,
                )(acc_field, elem_field)
            else:
                return acc_field

            # Fallback: pointwise strategy via lambda bindings
            stencil = list_expr.fun.args[0]
            if isinstance(stencil, ir.Lambda):
                param_names: list[str] = []
                call_args: list[ir.Expr] = []
                bindings: dict[str, dict[str, ir.Expr | str]] = {}
                for i, (param, arg_expr) in enumerate(zip(stencil.params, list_expr.args, strict=True)):
                    local_name = f"__cart_reduce_arg{i}"
                    local_ref = im.ref(local_name)
                    bound_arg = copy.deepcopy(arg_expr)
                    if cpm.is_applied_as_fieldop(bound_arg):
                        bound_stencil = bound_arg.fun.args[0]
                        if (
                            isinstance(bound_stencil, ir.Lambda)
                            and len(bound_stencil.params) == 1
                            and cpm.is_call_to(bound_stencil.expr, "neighbors")
                            and len(bound_stencil.expr.args) == 2
                            and isinstance(bound_stencil.expr.args[0], ir.OffsetLiteral)
                            and isinstance(bound_stencil.expr.args[1], ir.SymRef)
                            and bound_stencil.expr.args[1].id == bound_stencil.params[0].id
                            and len(bound_arg.args) == 1
                        ):
                            stencil_conn = bound_stencil.expr.args[0].value
                            if isinstance(stencil_conn, str):
                                bindings[str(param.id)] = {
                                    "kind": "neighbors", "conn": stencil_conn, "ref": local_ref,
                                }
                                param_names.append(local_name)
                                call_args.append(copy.deepcopy(bound_arg.args[0]))
                                continue
                    bindings[str(param.id)] = {"kind": "list", "ref": local_ref}
                    param_names.append(local_name)
                    call_args.append(bound_arg)

                acc = copy.deepcopy(widened_init)
                for idx in range(conn_size):
                    elem = NeighborReductionUnroller._local_list_element_expr(
                        stencil.expr, idx, bindings,
                        _neutral_element_for_reduce_op(red_op, widened_init),
                        domain_bounds,
                    )
                    if elem is None:
                        break
                    acc = im.call(copy.deepcopy(red_op))(acc, elem)
                else:
                    return im.as_fieldop(im.lambda_(*param_names)(acc), domain)(*call_args)

        # Last resort: list_get fallback
        lst_name = "__cart_reduce_lst"
        acc = copy.deepcopy(widened_init)
        for idx in range(conn_size):
            acc = im.call(copy.deepcopy(red_op))(
                acc, im.list_get(im.literal(str(idx), "int32"), im.deref(im.ref(lst_name)))
            )
        return im.as_fieldop(im.lambda_(lst_name)(acc), domain)(copy.deepcopy(list_expr))

    @staticmethod
    def _is_structured_domain(domain: ir.Expr) -> bool:
        if cpm.is_call_to(domain, "cartesian_domain"):
            return True
        if cpm.is_call_to(domain, "make_tuple"):
            return any(cpm.is_call_to(sub, "cartesian_domain") for sub in domain.args)
        return False

    # ── visit_SetAt: edge-specialized kolor branch peeling ───────────────────

    def visit_SetAt(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        new_domain = self.visit(node.domain, **kwargs)
        if not self._is_structured_domain(new_domain):
            new_expr = self.visit(node.expr, **kwargs)
            return ir.SetAt(
                expr=new_expr, domain=new_domain, target=self.visit(node.target, **kwargs)
            )

        current_kolor = _kolor_from_domain(new_domain)
        if current_kolor is None and cpm.is_call_to(new_domain, "make_tuple"):
            for sub in new_domain.args:
                current_kolor = _kolor_from_domain(sub)
                if current_kolor is not None:
                    break

        if current_kolor is None and not _e2c2e_on_local_intermediate(node.expr):
            new_expr = self._visit_expr_with_kolor_branches(node.expr, new_domain, **kwargs)
        else:
            new_expr = self.visit(
                node.expr, current_domain=new_domain, current_kolor=current_kolor, **kwargs
            )
        new_target = self.visit(node.target, **kwargs)
        return ir.SetAt(expr=new_expr, domain=new_domain, target=new_target)

    def _visit_expr_with_kolor_branches(
        self, expr: ir.Expr, domain: ir.Expr, **kwargs
    ) -> ir.Expr:
        """Edge-specialized: visit 3-kolor concat_where branch-by-branch with per-kolor context."""
        branches: list[tuple[int, ir.Expr, ir.Expr]] = []
        cur = expr

        while cpm.is_call_to(cur, "concat_where") and len(cur.args) == 3:
            cond, branch_expr, rest = cur.args
            k = _kolor_from_cw_condition(cond)
            if k is None:
                break
            branches.append((k, cond, branch_expr))
            cur = rest

        if not branches:
            return self.visit(expr, current_domain=domain, current_kolor=None, **kwargs)

        tail = cur
        visited: list[tuple[int, ir.Expr, ir.Expr]] = []
        for k, cond, branch_expr in branches:
            per_kolor_domain = _kolor_cw_condition_to_domain(cond, domain)
            visited_expr = self.visit(
                branch_expr,
                current_domain=per_kolor_domain if per_kolor_domain is not None else domain,
                current_kolor=k,
                **kwargs,
            )
            visited.append((k, cond, visited_expr))

        visited_tail = self.visit(tail, current_domain=domain, current_kolor=None, **kwargs)
        result = visited_tail
        for k, cond, visited_expr in reversed(visited):
            result = im.concat_where(copy.deepcopy(cond), visited_expr, result)
        return result

    # ── visit_FunCall: general reduction detection and unrolling ─────────────

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        current_domain = kwargs.get("current_domain")
        current_kolor: int | None = kwargs.get("current_kolor")

        # Path 1 — as_fieldop wrapping a single deref-shift:
        #   as_fieldop(λit. deref(shift(conn, slot)(it)), domain)(field)
        #   Matches stencils that are just lifted field accesses (already fieldop-wrapped shifts).
        #   Mutually exclusive with Path 2 (reduce) and Path 3 (raw shift): an as_fieldop with a
        #   pure deref-shift stencil cannot simultaneously be a reduce or a bare shift node.
        if cpm.is_applied_as_fieldop(node) and len(node.args) == 1:
            stencil = node.fun.args[0]
            if (
                isinstance(stencil, ir.Lambda)
                and len(stencil.params) == 1
                and cpm.is_call_to(stencil.expr, "deref")
                and len(stencil.expr.args) == 1
                and cpm.is_applied_shift(stencil.expr.args[0])
                and isinstance(stencil.expr.args[0].args[0], ir.SymRef)
                and stencil.expr.args[0].args[0].id == stencil.params[0].id
            ):
                shift_call = stencil.expr.args[0]
                key = cast(tuple[ir.OffsetLiteral, ir.OffsetLiteral], tuple(shift_call.fun.args))
                if key in map_dict:
                    entry = cast(dict[str, object], map_dict[key])
                    rewritten_arg = self.visit(node.args[0], **kwargs)
                    conn_name, is_ene = _conn_name_and_is_edge_to_non_edge(key)
                    if entry["kind"] == "concat_where":
                        return _build_field_concat_where_from_branches(
                            rewritten_arg,
                            cast(tuple, entry["branches"]),
                            current_domain,
                            apply_edge_shape_bounds=_needs_edge_shape_bounds(conn_name or None),
                            current_kolor=current_kolor if is_ene else None,
                        )
                    elif entry["kind"] == "shift":
                        return _make_lifted_deref_shift(
                            rewritten_arg,
                            cast(tuple[ir.OffsetLiteral, ...], entry["shifts"]),
                            current_domain,
                        )

            # Path 2 — generic reduce: as_fieldop(λit. reduce(op)(init)(deref(it)), domain)(list)
            #   Matches reduce(plus)(neighbors(conn, field), ...) patterns.
            #   Only fires when Path 1 did not match (stencil is not a pure deref-shift).
            if (reduce_inputs := self._extract_generic_reduce_inputs(node)) is not None:
                red_op, red_init, list_expr, conn_size = reduce_inputs
                rewritten_list_expr = self.visit(list_expr, **kwargs)
                return self._build_generic_unrolled_reduce_expr(
                    red_op, red_init, rewritten_list_expr, conn_size, current_domain,
                    current_kolor=current_kolor,
                )

        new_node = self.generic_visit(node, **kwargs)

        # Path 3 — raw shift: shift(conn, slot)(field)
        #   Matches bare (non-fieldop-wrapped) shifts that appear inside stencil lambdas.
        #   Only reached after generic_visit because the node must first be visited to resolve
        #   nested structure.  Mutually exclusive with Paths 1/2 (those return early).
        if cpm.is_applied_shift(new_node):
            key = cast(tuple[ir.OffsetLiteral, ir.OffsetLiteral], tuple(new_node.fun.args))
            if key in map_dict:
                entry = cast(dict[str, object], map_dict[key])
                arg = new_node.args[0]
                if entry["kind"] == "shift":
                    return _apply_shift_chain(
                        copy.deepcopy(arg), cast(tuple[ir.OffsetLiteral, ...], entry["shifts"])
                    )
                if entry["kind"] == "concat_where":
                    conn_name, is_ene = _conn_name_and_is_edge_to_non_edge(key)
                    if current_domain is not None:
                        return _build_field_concat_where_from_branches(
                            arg, entry["branches"], current_domain,
                            apply_edge_shape_bounds=_needs_edge_shape_bounds(conn_name or None),
                            current_kolor=current_kolor if is_ene else None,
                        )
                    return _build_concat_where_from_branches(arg, entry["branches"])

        return new_node

    @classmethod
    def apply(cls, node: ir.Node, **kwargs) -> ir.Node:
        return cls().visit(node, **kwargs)





# =====================================================================
# Pass 7: KolorConstantPropagation — General (optional, currently disabled)
# =====================================================================

@dataclasses.dataclass
class KolorConstantPropagation(NodeTranslator):
    """General (optional): Eliminates dead kolor branches via constant propagation.

    After NeighborReductionUnroller each expanded concat_where emits per-kolor
    sub-branches.  When the outer SetAt already constrains Kolor to a single value
    (per-kolor split), the inner branches for other kolors are statically unreachable.
    This pass removes them, reducing IR size ≈3x for edge stencils.

    Currently NOT wired into StructuredBackend.apply() — enable if needed.
    """

    @staticmethod
    def _kolor_range_from_cond(cond: ir.Expr) -> tuple[int, int] | None:
        """Return (lo, hi) if cond contains exactly one Kolor named_range."""
        # GT4Py IR always represents and_() as a binary operator (exactly 2 args).
        if cpm.is_call_to(cond, "and_") and len(cond.args) == 2:
            for arg in cond.args:
                result = KolorConstantPropagation._kolor_range_from_cond(arg)
                if result is not None:
                    return result
            return None
        if not cpm.is_call_to(cond, "cartesian_domain"):
            return None
        for nr in cond.args:
            if _named_range_args(nr) is None:
                continue
            if _get_axis_name(nr.args[0]) != "Kolor":
                continue
            lo, hi = nr.args[1], nr.args[2]
            if isinstance(lo, ir.OffsetLiteral) and isinstance(hi, ir.OffsetLiteral):
                return int(lo.value), int(hi.value)
        return None

    def visit_FunCall(
        self,
        node: ir.FunCall,
        *,
        known_kolor: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> ir.Expr:
        if not cpm.is_call_to(node, "concat_where") or len(node.args) != 3:
            return self.generic_visit(node, known_kolor=known_kolor, **kwargs)

        cond, true_branch, false_branch = node.args
        inner_range = self._kolor_range_from_cond(cond)

        if inner_range is not None and known_kolor is not None:
            klo, khi = known_kolor
            ilo, ihi = inner_range
            if ihi <= klo or ilo >= khi:
                return self.visit(false_branch, known_kolor=known_kolor, **kwargs)
            if ilo >= klo and ihi <= khi:
                return self.visit(true_branch, known_kolor=inner_range, **kwargs)

        if inner_range is not None:
            new_true = self.visit(true_branch, known_kolor=inner_range, **kwargs)
            new_false = self.visit(false_branch, known_kolor=known_kolor, **kwargs)
            if new_true is true_branch and new_false is false_branch:
                return node
            return ir.FunCall(fun=copy.deepcopy(node.fun), args=[cond, new_true, new_false])

        return self.generic_visit(node, known_kolor=known_kolor, **kwargs)

    @classmethod
    def apply(cls, node: ir.Node, **kwargs) -> ir.Node:
        return cls().visit(node, known_kolor=None, **kwargs)


# =====================================================================
# Pass 8: CanDerefRewriter — General
# =====================================================================

class CanDerefRewriter(NodeTranslator):
    """General: Replaces can_deref(...) with True.

    After concat_where-based validity masking, bounds are enforced structurally
    so can_deref calls are always True within the valid domain.
    """

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        new_node = self.generic_visit(node, **kwargs)
        if cpm.is_call_to(new_node, "can_deref") and len(new_node.args) == 1:
            return im.literal("True", "bool")
        return new_node

    @classmethod
    def apply(cls, node: ir.Node, **kwargs) -> ir.Node:
        return cls().visit(node, **kwargs)


# =====================================================================
# Orchestration: StructuredBackend
# =====================================================================

@dataclasses.dataclass
class StructuredBackend:
    """Orchestrates all structured backend passes in order.

    Replaces CartUnroll as the main entry point.
    Passes (in order):
      1. StructuredTypeRemapper   — type-level remapping + validation gate
      2. SetAtRemapper            — domain remapping + per-kolor split
      3. BroadcastAxisExpander    — broadcast axis expansion
      4. ThresholdConditionRewriter — threshold condition rewriting
      5. SymbolicSizeInliner      — symbolic size inlining
      6. NeighborReductionUnroller — reduction unrolling
      7. CanDerefRewriter         — can_deref → True
      (KolorConstantPropagation is optional, not wired in by default)
    """

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        symbolic_domain_sizes: dict[str, Any] | None = None,
        offset_provider: common.OffsetProvider | None = None,
    ) -> ir.Node:
        effective = dict(symbolic_domain_sizes or {})

        # Infer sizes from offset_provider if needed (E2C2E case)
        has_structured_bounds = any(
            key in effective
            for key in ("i_min", "i_max", "j_min", "j_max", "max_i", "max_j",
                        "domain_max_i", "domain_max_j", "nx", "ny")
        )
        if not has_structured_bounds and offset_provider is not None:
            for key, value in StructuredTypeRemapper._infer_symbolic_sizes_from_offset_provider(
                offset_provider
            ).items():
                effective.setdefault(key, value)

        # Validation gate: skip silently if requirements not met
        if not StructuredTypeRemapper._validate_structured_remap_requirements(node, effective):
            return node

        node = StructuredTypeRemapper.apply(node, symbolic_domain_sizes=effective)
        node = SetAtRemapper.apply(node, symbolic_domain_sizes=effective)
        node = BroadcastAxisExpander.apply(node)
        node = ThresholdConditionRewriter.apply(node, symbolic_domain_sizes=effective)
        node = SymbolicSizeInliner.apply(node, symbolic_domain_sizes=effective)
        node = NeighborReductionUnroller.apply(node)
        node = CanDerefRewriter.apply(node)
        return node


# =====================================================================
# Backward-compat aliases (used by cart_unroll.py shim)
# =====================================================================

# CartesianDomainAndTypeRemapper combined the first 5 passes into one class.
# For the shim we expose a thin wrapper that runs them in sequence.
class CartesianDomainAndTypeRemapper:
    """Backward-compat wrapper: runs passes 1–5 (type + domain + broadcast + threshold + inliner)."""

    _cartesian_remapped_type = staticmethod(StructuredTypeRemapper._cartesian_remapped_type)
    _validate_structured_remap_requirements = staticmethod(
        StructuredTypeRemapper._validate_structured_remap_requirements
    )

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        symbolic_domain_sizes: dict[str, Any] | None = None,
        offset_provider: common.OffsetProvider | None = None,
    ) -> ir.Node:
        effective = dict(symbolic_domain_sizes or {})
        has_structured_bounds = any(
            key in effective
            for key in ("i_min", "i_max", "j_min", "j_max", "max_i", "max_j",
                        "domain_max_i", "domain_max_j", "nx", "ny")
        )
        if not has_structured_bounds and offset_provider is not None:
            for key, value in StructuredTypeRemapper._infer_symbolic_sizes_from_offset_provider(
                offset_provider
            ).items():
                effective.setdefault(key, value)
        # For non-Program nodes (raw expressions): skip Program-level passes but still run
        # passes that work on any node (broadcast expansion, symbolic size inlining, shifts).
        if not isinstance(node, ir.Program):
            node = BroadcastAxisExpander.apply(node)
            node = SymbolicSizeInliner.apply(node, symbolic_domain_sizes=effective)
            return node
        if not StructuredTypeRemapper._validate_structured_remap_requirements(node, effective):
            return node
        node = StructuredTypeRemapper.apply(node, symbolic_domain_sizes=effective)
        node = SetAtRemapper.apply(node, symbolic_domain_sizes=effective)
        node = BroadcastAxisExpander.apply(node)
        node = ThresholdConditionRewriter.apply(node, symbolic_domain_sizes=effective)
        node = SymbolicSizeInliner.apply(node, symbolic_domain_sizes=effective)
        return node


CartesianReductionUnroller = NeighborReductionUnroller
RewriteCartesianCanDeref = CanDerefRewriter


class CartUnroll:
    """Backward-compat: runs CartesianDomainAndTypeRemapper then CartesianReductionUnroller."""

    _cartesian_remapped_type = staticmethod(StructuredTypeRemapper._cartesian_remapped_type)

    @classmethod
    def apply(
        cls, node: ir.Node, *, symbolic_domain_sizes: dict[str, Any] | None = None
    ) -> ir.Node:
        transformed = CartesianDomainAndTypeRemapper.apply(
            node, symbolic_domain_sizes=symbolic_domain_sizes
        )
        return NeighborReductionUnroller.apply(transformed)