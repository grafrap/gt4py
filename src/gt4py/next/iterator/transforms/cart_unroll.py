# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
import math
import numbers
from typing import Any, cast

import gt4py.next.iterator.transforms.map_dict as map_dict_module
from gt4py.eve import NodeTranslator
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_specifications as ts


# You can keep map_dict as a local alias so you don't have to change the rest of your code
map_dict = map_dict_module.map_dict


# Cache entity start bounds derived from structured index mappings.
# Key: (entity_name, id(mapping_rows), horizontal_start, max_i, max_j)
_ENTITY_START_BOUNDS_CACHE: dict[
    tuple[str, int, int, int, int], dict[int, tuple[int, int, int, int]]
] = {}


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

    if entity_name == "Edge":
        by_kolor: dict[int, tuple[int, int, int, int]] = {}
        for i_val, j_val, kolor in mapping_rows[horizontal_start:]:
            if i_val < 0 or j_val < 0 or kolor < 0:
                continue
            if kolor not in {0, 1, 2}:
                continue
            prev = by_kolor.get(kolor)
            if prev is None:
                by_kolor[kolor] = (i_val, j_val, i_val, j_val)
            else:
                by_kolor[kolor] = (
                    min(prev[0], i_val),
                    min(prev[1], j_val),
                    max(prev[2], i_val),
                    max(prev[3], j_val),
                )

        if not by_kolor:
            return None

        _ENTITY_START_BOUNDS_CACHE[cache_key] = by_kolor
        return by_kolor

    # Vertex/Cell: scalar horizontal_start maps to a single (i,j) lower shell.
    min_i: int | None = None
    min_j: int | None = None
    max_i_seen: int | None = None
    max_j_seen: int | None = None
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

# =====================================================================
# Shift and Concat-Where Helpers
# =====================================================================


def _apply_shift_chain(arg: ir.Expr, shift_spec: tuple[ir.OffsetLiteral, ...]) -> ir.Expr:
    normalized: list[ir.OffsetLiteral] = []
    for idx, off in enumerate(shift_spec):
        if idx % 2 == 0 and isinstance(off.value, str) and off.value in {"IDim", "JDim", "Kolor"}:
            normalized.append(
                ir.OffsetLiteral(value=common.dimension_to_implicit_offset(off.value))
            )
        else:
            normalized.append(off)
    return im.call(im.call("shift")(*tuple(normalized)))(arg)


def _build_concat_where_from_branches(arg: ir.Expr, branches):
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
    # Any sparse connectivity that dereferences edge neighbors can hit
    # padded edge slots in structured layout and therefore needs edge-shape
    # clipping for shifted dereferences.
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
):
    def _extract_kolor_interval(cond_expr: ir.Expr | None) -> tuple[int, int] | None:
        if cond_expr is None or not cpm.is_call_to(cond_expr, "cartesian_domain"):
            return None
        for range_expr in cond_expr.args:
            if not (cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3):
                continue
            if _get_axis_name(range_expr.args[0]) != "Kolor":
                continue
            lo = range_expr.args[1]
            hi = range_expr.args[2]
            if (
                isinstance(lo, ir.OffsetLiteral)
                and isinstance(lo.value, int)
                and isinstance(hi, ir.OffsetLiteral)
                and isinstance(hi.value, int)
            ):
                return lo.value, hi.value
        return None

    def _kolor_shift(shift_spec: tuple[ir.OffsetLiteral, ...]) -> int:
        kolor_axis_tags = {
            common.dimension_to_implicit_offset("Kolor"),
            "Kolor",
        }
        for idx in range(0, len(shift_spec), 2):
            if idx + 1 >= len(shift_spec):
                break
            axis_lit = shift_spec[idx]
            off_lit = shift_spec[idx + 1]
            if (
                isinstance(axis_lit, ir.OffsetLiteral)
                and axis_lit.value in kolor_axis_tags
                and isinstance(off_lit, ir.OffsetLiteral)
                and isinstance(off_lit.value, int)
            ):
                return int(off_lit.value)
        return 0

    def _minus_one(expr: ir.Expr) -> ir.Expr:
        if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
            return ir.OffsetLiteral(value=expr.value - 1)
        return im.minus(copy.deepcopy(expr), ir.OffsetLiteral(value=1))

    def _edge_shape_domain(
        source_kolor: int,
        shift_spec: tuple[ir.OffsetLiteral, ...],
        current_kolor: int | None = None,
    ) -> ir.Expr | None:
        # With per-kolor split, the SetAt domain is already restricted to valid
        # interior source edges, so all neighbor accesses are within the valid
        # grid range. Clipping the already-narrow per-kolor domain would
        # over-clip and exclude valid boundary source positions.
        if current_kolor is not None:
            return None
        if domain is None or not cpm.is_call_to(domain, "cartesian_domain"):
            return None

        target_kolor = source_kolor + _kolor_shift(shift_spec)
        if target_kolor not in {0, 1, 2}:
            return None

        id_range = None
        jd_range = None
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

        # mypy can't always narrow types after the loop checks above; assert to help the type checker
        assert id_range is not None and jd_range is not None

        i_lo, i_hi = copy.deepcopy(id_range.args[1]), copy.deepcopy(id_range.args[2])
        j_lo, j_hi = copy.deepcopy(jd_range.args[1]), copy.deepcopy(jd_range.args[2])

        if target_kolor in {1, 2}:
            i_hi = _minus_one(i_hi)
        if target_kolor in {0, 2}:
            j_hi = _minus_one(j_hi)

        new_ranges: list[ir.Expr] = []
        for range_expr in domain.args:
            if not (cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3):
                continue
            axis_name = _get_axis_name(range_expr.args[0])
            if axis_name == "IDim":
                new_ranges.append(
                    im.named_range(
                        cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(range_expr.args[0])),
                        i_lo,
                        i_hi,
                    )
                )
            elif axis_name == "JDim":
                new_ranges.append(
                    im.named_range(
                        cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(range_expr.args[0])),
                        j_lo,
                        j_hi,
                    )
                )
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

    # When a specific kolor is known (per-kolor SetAt split), select the matching branch
    # directly without generating a concat_where — but only for edge-center connectivities.
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
        # Fallback: no explicit match found (e.g., else-branch for the last kolor)
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

    expr = _make_lifted_deref_shift(arg, shift_spec, branch_domain)
    if len(branches) == 1:
        return expr
    return im.concat_where(
        cond,
        expr,
        _build_field_concat_where_from_branches(
            arg,
            branches[1:],
            domain,
            inferred_kolor_start=next_inferred_kolor,
            apply_edge_shape_bounds=apply_edge_shape_bounds,
            current_kolor=current_kolor,
        ),
    )


def _is_unstructured_edge_domain_stmt(domain_expr: ir.Expr) -> bool:
    """Return True if any domain node in domain_expr names the Edge axis."""
    def _iter(expr: ir.Expr):
        if cpm.is_call_to(expr, "make_tuple"):
            for a in expr.args:
                yield from _iter(a)
            return
        yield expr

    for dom in _iter(domain_expr):
        if not cpm.is_call_to(dom, "unstructured_domain"):
            continue
        for nr in dom.args:
            if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                if _get_axis_name(nr.args[0]) == "Edge":
                    return True
    return False


_EDGE_TO_EDGE_CONNECTIVITIES: frozenset[str] = frozenset({"E2C2EO", "E2C2E"})


def _kolor_from_cw_condition(cond: ir.Expr) -> int | None:
    """Extract the kolor integer from a concat_where condition built by _build_edge_validity_masked_expr.

    The condition has the shape and_(kolor_domain, and_(idim_domain, jdim_domain)) where
    kolor_domain = cartesian_domain(named_range(Kolor, k, k+1)).
    Returns k if found, else None.
    """
    # Walk into and_ chains to find a single-kolor cartesian_domain.
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
    """Convert a concat_where condition back to a cartesian_domain for use as current_domain.

    The condition is and_(kolor_domain, and_(idim_domain, jdim_domain)).
    We reconstruct a cartesian_domain with axes in the canonical order:
    IDim, JDim, Kolor, then any remaining axes (e.g. K) from fallback_domain.
    """
    by_axis: dict[str, ir.Expr] = {}

    def _collect(expr: ir.Expr) -> None:
        if cpm.is_call_to(expr, "and_"):
            for arg in expr.args:
                _collect(arg)
        elif cpm.is_call_to(expr, "cartesian_domain"):
            for nr in expr.args:
                if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                    name = _get_axis_name(nr.args[0])
                    if name is not None:
                        by_axis[name] = copy.deepcopy(nr)

    _collect(cond)
    if not by_axis:
        return None

    # Collect non-horizontal axes (e.g. K) from fallback_domain.
    def _collect_fallback(domain: ir.Expr) -> None:
        if cpm.is_call_to(domain, "cartesian_domain"):
            for nr in domain.args:
                if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                    name = _get_axis_name(nr.args[0])
                    if name is not None and name not in {"IDim", "JDim", "Kolor"}:
                        by_axis.setdefault(name, copy.deepcopy(nr))
        elif cpm.is_call_to(domain, "make_tuple"):
            for sub in domain.args:
                _collect_fallback(sub)
                break  # only need the first sub-domain for K ranges

    _collect_fallback(fallback_domain)

    # Emit ranges in canonical order: IDim, JDim, Kolor, then everything else.
    canonical = ["IDim", "JDim", "Kolor"]
    ordered = [by_axis[n] for n in canonical if n in by_axis]
    for name, nr in by_axis.items():
        if name not in canonical:
            ordered.append(nr)

    return im.call("cartesian_domain")(*ordered)


def _expr_uses_edge_to_edge_connectivity(node: ir.Node) -> bool:
    """Return True if node contains any edge-to-edge connectivity offset (e.g. E2C2EO)."""
    for lit in node.pre_walk_values().if_isinstance(ir.OffsetLiteral):
        if isinstance(lit.value, str) and lit.value.rstrip("ₒ") in _EDGE_TO_EDGE_CONNECTIVITIES:
            return True
    return False


def _e2c2e_on_local_intermediate(node: ir.Node) -> bool:
    """Return True if node has an edge-to-edge neighbor access on a lambda-bound (local) field.

    When an E2C2EO/E2C2E reduction operates on a field that is a lambda parameter (local
    intermediate computed within the same SetAt), per-kolor branch visiting is unsafe: the
    intermediate is only materialized for the current kolor's domain, so cross-kolor shifts
    in E2C2EO would access uninitialized data.

    Heuristic: if the argument to `neighbors(E2C2EO, ...)` inside an applied reduce is NOT
    a direct SymRef into the top-level program params — i.e. it is itself the result of an
    as_fieldop (a computed field) rather than a bare program-parameter reference — then it
    is a local intermediate. We detect this by checking whether any outer Lambda in the tree
    binds the SymRef used as the E2C2EO source.
    """
    # Collect all lambda parameter names at any nesting level within node.
    lambda_params: set[str] = set()
    for lam in node.pre_walk_values().if_isinstance(ir.Lambda):
        for p in lam.params:
            lambda_params.add(str(p.id))

    def _unwrap_to_symref(expr: ir.Expr) -> ir.SymRef | None:
        """Unwrap cast/identity as_fieldop wrappers to find the underlying SymRef, if any."""
        if isinstance(expr, ir.SymRef):
            return expr
        # Unwrap single-arg as_fieldop (e.g. cast_, identity) applied to a field.
        if cpm.is_applied_as_fieldop(expr) and len(expr.args) == 1:
            return _unwrap_to_symref(expr.args[0])
        return None

    # Now look for neighbors(E2C2EO/E2C2E, it) where `it` resolves to a lambda-bound field.
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
        # Unwrap cast/identity wrappers on the field argument, then check if it is a
        # lambda-bound parameter (local intermediate) rather than a program param.
        sym = _unwrap_to_symref(fun_call.args[0])
        if sym is not None and str(sym.id) in lambda_params:
            return True
    return False


def _kolor_from_domain(domain: ir.Expr | None) -> int | None:
    """Return k if *domain* has a single-kolor Kolor:[k, k+1) range, else None."""
    if domain is None or not cpm.is_call_to(domain, "cartesian_domain"):
        return None
    for nr in domain.args:
        if not (cpm.is_call_to(nr, "named_range") and len(nr.args) == 3):
            continue
        if _get_axis_name(nr.args[0]) != "Kolor":
            continue
        lo, hi = nr.args[1], nr.args[2]
        if isinstance(lo, ir.OffsetLiteral) and isinstance(hi, ir.OffsetLiteral):
            if hi.value - lo.value == 1:
                return int(lo.value)
    return None


def _get_axis_name(axis_expr: ir.Expr) -> str | None:
    """Safely extracts the axis string name from either an AxisLiteral or a Dimension object."""
    if isinstance(axis_expr, ir.AxisLiteral) and isinstance(axis_expr.value, str):
        return axis_expr.value
    if hasattr(axis_expr, "value") and isinstance(axis_expr.value, str):
        return axis_expr.value
    return None

def _concat_where_condition_from_domain(domain_expr: ir.Expr) -> ir.Expr:
    if cpm.is_call_to(domain_expr, "cartesian_domain") and len(domain_expr.args) > 1:
        axis_domains = []
        for range_expr in domain_expr.args:
            if cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3:
                # Explicitly drop Kolor using the robust extractor
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
# Pass 1: Domain and Type Remapping
# =====================================================================


@dataclasses.dataclass
class CartesianDomainAndTypeRemapper(NodeTranslator):
    @staticmethod
    def _program_uses_horizontal_unstructured_axis(node: ir.Node) -> bool:
        for fun_call in node.pre_walk_values().if_isinstance(ir.FunCall):
            if not cpm.is_call_to(fun_call, "named_range") or len(fun_call.args) != 3:
                continue
            axis_name = _get_axis_name(fun_call.args[0])
            if axis_name in {"Edge", "Vertex", "Cell"}:
                return True
        return False

    @staticmethod
    def _validate_structured_remap_requirements(
        node: ir.Node, symbolic_domain_sizes: dict[str, Any]
    ) -> None:
        if not isinstance(node, ir.Program):
            return
        if not CartesianDomainAndTypeRemapper._program_uses_horizontal_unstructured_axis(node):
            return

        program_param_ids = {str(param.id) for param in node.params}

        def _has_any(*names: str) -> bool:
            return any(name in symbolic_domain_sizes or name in program_param_ids for name in names)

        has_symbolic_structured_sizes = any(
            name in symbolic_domain_sizes
            for name in ("max_i", "max_j", "domain_max_i", "domain_max_j", "nx", "ny")
        )
        has_symbolic_signal = has_symbolic_structured_sizes or any(
            name in symbolic_domain_sizes for name in ("lateral", "lateral_bounds", "lateral_edge")
        )

        if not has_symbolic_signal:
            return

        missing_groups: list[str] = []
        use_mapping = symbolic_domain_sizes.get("use_horizontal_start_mapping") in {
            True, 1, "1", "true", "on", "yes"
        } and "edge_to_ijk" in symbolic_domain_sizes
        if has_symbolic_structured_sizes and not use_mapping and not _has_any(
            "lateral", "lateral_bounds", "lateral_edge"
        ):
            missing_groups.append("lateral")
        if not _has_any("i_max", "domain_i_max", "max_i", "domain_max_i", "nx", "num_i", "ni"):
            missing_groups.append("IDim upper bound")
        if not _has_any("j_max", "domain_j_max", "max_j", "domain_max_j", "ny", "num_j", "nj"):
            missing_groups.append("JDim upper bound")

        if missing_groups:
            missing = ", ".join(missing_groups)
            raise ValueError(
                "Cartesian unstructured→structured remap requires explicit symbolic domain "
                f"sizes. Missing: {missing}. "
                "Provide these values via backend option "
                "'otf_workflow__bare_translation__symbolic_domain_sizes'."
            )

    @staticmethod
    def _cartesian_remapped_type(type_: ts.TypeSpec | None) -> ts.TypeSpec | None:
        if not isinstance(type_, ts.FieldType):
            return type_
        if not any(dim.value in {"Edge", "Vertex", "Cell"} for dim in type_.dims):
            return type_
        idim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        jdim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
        new_dims: list[common.Dimension] = []
        for dim in type_.dims:
            if dim.value in {"Edge", "Vertex", "Cell"}:
                new_dims.extend([idim, jdim, kolor])
            else:
                new_dims.append(dim)
        return ts.FieldType(dims=new_dims, dtype=type_.dtype)

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        symbolic_domain_sizes: dict[str, Any] | None = None,
        offset_provider: common.OffsetProvider | None = None,
    ) -> ir.Node:
        effective_symbolic_sizes = dict(symbolic_domain_sizes or {})

        has_structured_bounds = any(
            key in effective_symbolic_sizes
            for key in (
                "i_min",
                "i_max",
                "j_min",
                "j_max",
                "max_i",
                "max_j",
                "domain_max_i",
                "domain_max_j",
                "nx",
                "ny",
            )
        )

        if (
            not has_structured_bounds
            and offset_provider is not None
            and cls._program_uses_offsets(node, {"E2C2E"})
        ):
            inferred_sizes = cls._infer_symbolic_sizes_from_offset_provider(offset_provider)
            for key, value in inferred_sizes.items():
                effective_symbolic_sizes.setdefault(key, value)

        cls._validate_structured_remap_requirements(node, effective_symbolic_sizes)

        return cls().visit(node, symbolic_domain_sizes=effective_symbolic_sizes)

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
                if (
                    isinstance(dim_name, str)
                    and isinstance(size, numbers.Integral)
                    and int(size) > 0
                ):
                    axis_sizes.setdefault(dim_name, int(size))

        i_extent = axis_sizes.get("IDim")
        j_extent = axis_sizes.get("JDim")
        if not (
            isinstance(i_extent, int)
            and isinstance(j_extent, int)
            and i_extent > 0
            and j_extent > 0
        ):
            return {}

        return {
            "i_min": 0,
            "i_max": i_extent,
            "j_min": 0,
            "j_max": j_extent,
            "max_i": i_extent,
            "max_j": j_extent,
        }

    def visit_Program(self, node: ir.Program, **kwargs) -> ir.Program:
        program_param_ids = {str(param.id) for param in node.params}
        new_params = [
            ir.Sym(id=param.id, type=self._cartesian_remapped_type(param.type))
            for param in node.params
        ]
        child_kwargs = dict(kwargs)
        child_kwargs["program_param_ids"] = program_param_ids

        new_body: list[ir.SetAt] = []
        for stmt in node.body:
            # Split edge-domain SetAts into 3 per-kolor SetAts so that the
            # CartesianReductionUnroller can use the fixed kolor from the domain
            # to select shift branches directly (avoiding inner kolor concat_where).
            # Skip splitting when the SetAt uses edge-to-edge connectivity (E2C2EO, E2C2E)
            # because those accesses span multiple kolors on a local intermediate field
            # that is only defined for the current kolor — splitting would produce wrong results.
            if (
                isinstance(stmt, ir.SetAt)
                and _is_unstructured_edge_domain_stmt(stmt.domain)
                and not _expr_uses_edge_to_edge_connectivity(stmt.expr)
            ):
                for k in range(3):
                    new_body.append(self.visit(stmt, current_kolor=k, **child_kwargs))
            else:
                new_body.append(self.visit(stmt, **child_kwargs))

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

    def visit_SetAt(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        symbolic_domain_sizes: dict[str, Any] = kwargs.get("symbolic_domain_sizes") or {}

        def _pick_symbolic_int(*names: str) -> int | None:
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

        def _horizontal_start_mapping_enabled() -> bool:
            raw = symbolic_domain_sizes.get("use_horizontal_start_mapping")
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, numbers.Integral):
                return int(raw) != 0
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "yes", "on"}
            return False

        def _edge_phase_size_for_setat() -> int:
            phase = _pick_symbolic_int("edge_phase_size", "lateral_edge_phase")
            if phase is not None:
                return max(0, phase)
            lateral_edge = _pick_symbolic_int("lateral_edge")
            if lateral_edge is not None:
                return 1 if lateral_edge > 0 and lateral_edge % 2 == 1 else 0
            return 0

        def _has_unstructured_axis(type_: ts.TypeSpec | None) -> bool:
            return isinstance(type_, ts.FieldType) and any(
                dim.value in {"Edge", "Vertex", "Cell"} for dim in type_.dims
            )

        def _retarget_self_refs(expr: ir.Expr, target: ir.Expr) -> ir.Expr:
            if not isinstance(target, ir.SymRef) or not isinstance(target.type, ts.FieldType):
                return expr

            target_id = target.id
            target_type = target.type

            class _RetargetSelfRef(NodeTranslator):
                def visit_SymRef(self, n: ir.SymRef, **kws) -> ir.SymRef:
                    if n.id == target_id and _has_unstructured_axis(n.type):
                        return ir.SymRef(id=n.id, type=target_type)
                    return n

            return _RetargetSelfRef().visit(expr)

        def _is_fully_structured_field(expr: ir.Expr) -> bool:
            type_ = getattr(expr, "type", None)
            if not isinstance(type_, ts.FieldType):
                return False
            dim_names = {dim.value for dim in type_.dims}
            if {"IDim", "JDim", "Kolor"}.isdisjoint(dim_names):
                return False
            return not any(name in {"Edge", "Vertex", "Cell"} for name in dim_names)

        def _is_unstructured_edge_domain(domain_expr: ir.Expr) -> bool:
            def _iter_domain_nodes(expr: ir.Expr):
                if cpm.is_call_to(expr, "make_tuple"):
                    for arg in expr.args:
                        yield from _iter_domain_nodes(arg)
                    return
                yield expr

            for dom in _iter_domain_nodes(domain_expr):
                if not cpm.is_call_to(dom, "unstructured_domain"):
                    continue
                for nr in dom.args:
                    if not (cpm.is_call_to(nr, "named_range") and len(nr.args) == 3):
                        continue
                    axis_name = _get_axis_name(nr.args[0])
                    if axis_name == "Edge":
                        return True
            return False

        def _minus_one(expr: ir.Expr) -> ir.Expr:
            if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
                return ir.OffsetLiteral(value=expr.value - 1)
            if isinstance(expr, ir.Literal) and str(expr.value).lstrip("-").isdigit():
                return ir.Literal(value=str(int(expr.value) - 1), type=expr.type)
            return im.minus(copy.deepcopy(expr), ir.OffsetLiteral(value=1))

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

        def _build_edge_validity_masked_expr(
            expr: ir.Expr, target: ir.Expr, domain_expr: ir.Expr
        ) -> ir.Expr | None:
            def _iter_domain_nodes(expr_: ir.Expr):
                if cpm.is_call_to(expr_, "make_tuple"):
                    for arg_ in expr_.args:
                        yield from _iter_domain_nodes(arg_)
                    return
                yield expr_

            cart_domain = None
            for dom in _iter_domain_nodes(domain_expr):
                if cpm.is_call_to(dom, "cartesian_domain"):
                    cart_domain = dom
                    break
            if cart_domain is None:
                return None

            id_axis = j_axis = k_axis = None
            i_lo = i_hi = j_lo = j_hi = k_lo = k_hi = None
            for nr in cart_domain.args:
                if not (cpm.is_call_to(nr, "named_range") and len(nr.args) == 3):
                    continue
                axis_name = _get_axis_name(nr.args[0])
                if axis_name == "IDim":
                    id_axis, i_lo, i_hi = nr.args[0], nr.args[1], nr.args[2]
                elif axis_name == "JDim":
                    j_axis, j_lo, j_hi = nr.args[0], nr.args[1], nr.args[2]
                elif axis_name == "Kolor":
                    k_axis, k_lo, k_hi = nr.args[0], nr.args[1], nr.args[2]

            if any(
                v is None for v in (id_axis, j_axis, k_axis, i_lo, i_hi, j_lo, j_hi, k_lo, k_hi)
            ):
                return None

            # Help the type checker: all these are now not None
            assert (
                id_axis is not None
                and j_axis is not None
                and k_axis is not None
                and i_lo is not None
                and i_hi is not None
                and j_lo is not None
                and j_hi is not None
                and k_lo is not None
                and k_hi is not None
            )

            # Only apply this mask for 3-kolor edge cartesian domains.
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
                    im.named_range(
                        cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(axis)),
                        copy.deepcopy(lo),
                        copy.deepcopy(hi),
                    )
                )

            def _and(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
                return im.call("and_")(lhs, rhs)

            mapping_rows = _coerce_mapping_rows(symbolic_domain_sizes.get("edge_to_ijk"), 3)
            horizontal_start = _pick_symbolic_int(
                "horizontal_start_edge",
                "horizontal_start_e",
                "horizontal_start",
            )
            max_i_int = _pick_symbolic_int("i_max", "domain_i_max", "max_i", "domain_max_i", "nx")
            max_j_int = _pick_symbolic_int("j_max", "domain_j_max", "max_j", "domain_max_j", "ny")

            if (
                _horizontal_start_mapping_enabled()
                and mapping_rows is not None
                and horizontal_start is not None
                and horizontal_start > 0
                and max_i_int is not None
                and max_j_int is not None
            ):
                by_kolor = _derive_entity_start_bounds_from_mapping(
                    "Edge",
                    mapping_rows=mapping_rows,
                    horizontal_start=horizontal_start,
                    max_i=max_i_int,
                    max_j=max_j_int,
                )
                if by_kolor is not None and all(k in by_kolor for k in (0, 1, 2)):
                    i0_lo, j0_lo, i0_hi, j0_hi = by_kolor[0]
                    i1_lo, j1_lo, i1_hi, j1_hi = by_kolor[1]
                    i2_lo, j2_lo, i2_hi, j2_hi = by_kolor[2]

                    cond_k0 = _and(
                        _dom(
                            copy.deepcopy(k_axis),
                            ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value=1),
                        ),
                        _and(
                            _dom(
                                copy.deepcopy(id_axis),
                                ir.OffsetLiteral(value=i0_lo),
                                ir.OffsetLiteral(value=i0_hi + 1),
                            ),
                            _dom(
                                copy.deepcopy(j_axis),
                                ir.OffsetLiteral(value=j0_lo),
                                ir.OffsetLiteral(value=j0_hi + 1),
                            ),
                        ),
                    )
                    cond_k1 = _and(
                        _dom(
                            copy.deepcopy(k_axis),
                            ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value=2),
                        ),
                        _and(
                            _dom(
                                copy.deepcopy(id_axis),
                                ir.OffsetLiteral(value=i1_lo),
                                ir.OffsetLiteral(value=i1_hi + 1),
                            ),
                            _dom(
                                copy.deepcopy(j_axis),
                                ir.OffsetLiteral(value=j1_lo),
                                ir.OffsetLiteral(value=j1_hi + 1),
                            ),
                        ),
                    )
                    cond_k2 = _and(
                        _dom(
                            copy.deepcopy(k_axis),
                            ir.OffsetLiteral(value=2),
                            ir.OffsetLiteral(value=3),
                        ),
                        _and(
                            _dom(
                                copy.deepcopy(id_axis),
                                ir.OffsetLiteral(value=i2_lo),
                                ir.OffsetLiteral(value=i2_hi + 1),
                            ),
                            _dom(
                                copy.deepcopy(j_axis),
                                ir.OffsetLiteral(value=j2_lo),
                                ir.OffsetLiteral(value=j2_hi + 1),
                            ),
                        ),
                    )
                    # print(im.concat_where(
                    #     cond_k0,
                    #     copy.deepcopy(expr),
                    #     im.concat_where(
                    #         cond_k1,
                    #         copy.deepcopy(expr),
                    #         im.concat_where(
                    #             cond_k2,
                    #             copy.deepcopy(expr),
                    #             copy.deepcopy(target),
                    #         ),
                    #     ),
                    # ))

                    return im.concat_where(
                        cond_k0,
                        copy.deepcopy(expr),
                        im.concat_where(
                            cond_k1,
                            copy.deepcopy(expr),
                            im.concat_where(
                                cond_k2,
                                copy.deepcopy(expr),
                                copy.deepcopy(target),
                            ),
                        ),
                    )

            edge_phase = _edge_phase_size_for_setat()
            i_lo_k0 = _plus_n((i_lo), edge_phase)
            i_hi_k0 = _minus_n((i_hi), edge_phase)
            j_lo_k1 = _plus_n((j_lo), edge_phase)
            j_hi_k1 = _minus_n((j_hi), edge_phase)

            cond_k0 = _and(
                _dom(copy.deepcopy(k_axis), ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1)),
                _and(
                    _dom(copy.deepcopy(id_axis), i_lo_k0, i_hi_k0),
                    _dom(
                        copy.deepcopy(j_axis), copy.deepcopy(j_lo), _minus_one(copy.deepcopy(j_hi))
                    ),
                ),
            )
            cond_k1 = _and(
                _dom(copy.deepcopy(k_axis), ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2)),
                _and(
                    _dom(
                        copy.deepcopy(id_axis), copy.deepcopy(i_lo), _minus_one(copy.deepcopy(i_hi))
                    ),
                    _dom(copy.deepcopy(j_axis), j_lo_k1, j_hi_k1),
                ),
            )
            cond_k2 = _and(
                _dom(copy.deepcopy(k_axis), ir.OffsetLiteral(value=2), ir.OffsetLiteral(value=3)),
                _and(
                    _dom(
                        copy.deepcopy(id_axis), copy.deepcopy(i_lo), _minus_one(copy.deepcopy(i_hi))
                    ),
                    _dom(
                        copy.deepcopy(j_axis), copy.deepcopy(j_lo), _minus_one(copy.deepcopy(j_hi))
                    ),
                ),
            )

            return im.concat_where(
                cond_k0,
                copy.deepcopy(expr),
                im.concat_where(
                    cond_k1,
                    copy.deepcopy(expr),
                    im.concat_where(
                        cond_k2,
                        copy.deepcopy(expr),
                        copy.deepcopy(target),
                    ),
                ),
            )

        def _per_kolor_domain(kolor: int, full_structured_domain: ir.Expr) -> ir.Expr | None:
            """Build a kolor-specific SetAt domain from the full structured domain."""
            cart_domain = None
            for dom in (full_structured_domain,):
                if cpm.is_call_to(dom, "make_tuple"):
                    for a in dom.args:
                        if cpm.is_call_to(a, "cartesian_domain"):
                            cart_domain = a
                            break
                elif cpm.is_call_to(dom, "cartesian_domain"):
                    cart_domain = dom
                    break
            if cart_domain is None:
                return None

            id_axis = j_axis = k_axis = None
            other_ranges: list[ir.Expr] = []
            i_lo = i_hi = j_lo = j_hi = None

            for nr in cart_domain.args:
                if not (cpm.is_call_to(nr, "named_range") and len(nr.args) == 3):
                    continue
                name = _get_axis_name(nr.args[0])
                if name == "IDim":
                    id_axis = nr.args[0]
                    i_lo, i_hi = nr.args[1], nr.args[2]
                elif name == "JDim":
                    j_axis = nr.args[0]
                    j_lo, j_hi = nr.args[1], nr.args[2]
                elif name == "Kolor":
                    k_axis = nr.args[0]
                else:
                    other_ranges.append(copy.deepcopy(nr))

            if id_axis is None or j_axis is None or k_axis is None:
                return None

            # Compute per-kolor I/J bounds using the same logic as _build_edge_validity_masked_expr.
            mapping_rows = _coerce_mapping_rows(symbolic_domain_sizes.get("edge_to_ijk"), 3)
            horizontal_start = _pick_symbolic_int(
                "horizontal_start_edge", "horizontal_start_e", "horizontal_start"
            )
            max_i_int = _pick_symbolic_int("i_max", "domain_i_max", "max_i", "domain_max_i", "nx")
            max_j_int = _pick_symbolic_int("j_max", "domain_j_max", "max_j", "domain_max_j", "ny")

            ilo: ir.Expr | None = None
            jlo: ir.Expr | None = None
            ihi: ir.Expr | None = None
            jhi: ir.Expr | None = None

            if (
                _horizontal_start_mapping_enabled()
                and mapping_rows is not None
                and horizontal_start is not None
                and horizontal_start > 0
                and max_i_int is not None
                and max_j_int is not None
            ):
                by_kolor = _derive_entity_start_bounds_from_mapping(
                    "Edge",
                    mapping_rows=mapping_rows,
                    horizontal_start=horizontal_start,
                    max_i=max_i_int,
                    max_j=max_j_int,
                )
                if by_kolor is not None and kolor in by_kolor:
                    k_ilo, k_jlo, k_ihi, k_jhi = by_kolor[kolor]
                    ilo = ir.OffsetLiteral(value=k_ilo)
                    jlo = ir.OffsetLiteral(value=k_jlo)
                    ihi = ir.OffsetLiteral(value=k_ihi + 1)
                    jhi = ir.OffsetLiteral(value=k_jhi + 1)

            if ilo is None:
                # Phase-based fallback bounds per kolor
                edge_phase = _edge_phase_size_for_setat()
                if kolor == 0:
                    ilo = _plus_n(copy.deepcopy(i_lo), edge_phase)
                    jlo = copy.deepcopy(j_lo)
                    ihi = _minus_n(copy.deepcopy(i_hi), edge_phase)
                    jhi = _minus_one(copy.deepcopy(j_hi))
                elif kolor == 1:
                    ilo = copy.deepcopy(i_lo)
                    jlo = _plus_n(copy.deepcopy(j_lo), edge_phase)
                    ihi = _minus_one(copy.deepcopy(i_hi))
                    jhi = _minus_n(copy.deepcopy(j_hi), edge_phase)
                else:  # kolor == 2
                    ilo = copy.deepcopy(i_lo)
                    jlo = copy.deepcopy(j_lo)
                    ihi = _minus_one(copy.deepcopy(i_hi))
                    jhi = _minus_one(copy.deepcopy(j_hi))

            if ilo is None or jlo is None or ihi is None or jhi is None:
                return None

            new_ranges = [
                im.named_range(
                    cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(id_axis)), ilo, ihi
                ),
                im.named_range(
                    cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(j_axis)), jlo, jhi
                ),
                im.named_range(
                    cast(ir.AxisLiteral | common.Dimension, copy.deepcopy(k_axis)),
                    ir.OffsetLiteral(value=kolor),
                    ir.OffsetLiteral(value=kolor + 1),
                ),
            ]
            new_ranges.extend(other_ranges)

            result_domain = im.call("cartesian_domain")(*new_ranges)
            # Wrap in make_tuple if the original was a make_tuple (multi-output SetAt)
            if cpm.is_call_to(full_structured_domain, "make_tuple"):
                n_outputs = len(full_structured_domain.args)
                return im.make_tuple(*[copy.deepcopy(result_domain) for _ in range(n_outputs)])
            return result_domain

        current_kolor: int | None = kwargs.get("current_kolor")

        new_domain = self.visit(node.domain, **kwargs)
        new_expr = self.visit(node.expr, current_domain=new_domain, **kwargs)
        new_target = self.visit(node.target, **kwargs)

        # Some concat_where expansions keep a stale unstructured self-reference in the
        # SetAt expression (e.g. `next_vn` in the false branch). Retype those refs to
        # the structured target so expr/target dimensions stay aligned.
        if _is_fully_structured_field(new_target):
            new_expr = _retarget_self_refs(new_expr, new_target)

        if _is_unstructured_edge_domain(node.domain):
            if current_kolor is not None:
                # Per-kolor split: build kolor-specific domain, skip concat_where wrapper.
                kolor_domain = _per_kolor_domain(current_kolor, new_domain)
                if kolor_domain is not None:
                    return ir.SetAt(expr=new_expr, domain=kolor_domain, target=new_target)
            # Original behaviour: wrap expression in 3-kolor concat_where mask.
            masked_expr = _build_edge_validity_masked_expr(new_expr, new_target, new_domain)
            if masked_expr is not None:
                new_expr = masked_expr

        return ir.SetAt(expr=new_expr, domain=new_domain, target=new_target)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        program_param_ids: set[str] = kwargs.get("program_param_ids", set())
        symbolic_domain_sizes: dict[str, Any] | None = kwargs.get("symbolic_domain_sizes")
        new_node = copy.deepcopy(self.generic_visit(node, **kwargs))

        def _structured_axis_literals() -> list[ir.Expr]:
            return [
                ir.AxisLiteral(value="IDim", kind=common.DimensionKind.HORIZONTAL),
                ir.AxisLiteral(value="JDim", kind=common.DimensionKind.HORIZONTAL),
                ir.AxisLiteral(value="Kolor", kind=common.DimensionKind.HORIZONTAL),
            ]

        def _expand_unstructured_axis(axis_expr: ir.Expr) -> list[ir.Expr]:
            axis_name = _get_axis_name(axis_expr)
            if axis_name in {"Edge", "Vertex", "Cell"}:
                return _structured_axis_literals()
            return [copy.deepcopy(axis_expr)]

        def _remap_broadcast_axes(axes_expr: ir.Expr) -> ir.Expr:
            if cpm.is_call_to(axes_expr, "make_tuple"):
                remapped_args: list[ir.Expr] = []
                for arg in axes_expr.args:
                    remapped_args.extend(_expand_unstructured_axis(arg))
                return im.call("make_tuple")(*remapped_args)

            expanded = _expand_unstructured_axis(axes_expr)
            if len(expanded) == 1:
                return expanded[0]
            return im.call("make_tuple")(*expanded)

        if cpm.is_call_to(new_node, "broadcast") and len(new_node.args) == 2:
            remapped_axes = _remap_broadcast_axes(new_node.args[1])
            new_node = ir.FunCall(
                fun=new_node.fun, args=[new_node.args[0], remapped_axes], type=new_node.type
            )

        def _pick_size_param(*candidates: str) -> ir.Expr | None:
            for candidate in candidates:
                if candidate in program_param_ids:
                    return im.ref(candidate)
                if symbolic_domain_sizes is not None and candidate in symbolic_domain_sizes:
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

        def _pick_symbolic_int(*candidates: str) -> int | None:
            if symbolic_domain_sizes is None:
                return None
            for candidate in candidates:
                if candidate not in symbolic_domain_sizes:
                    continue
                parsed = _coerce_int(symbolic_domain_sizes[candidate])
                if parsed is not None:
                    return parsed
            return None

        def _horizontal_start_mapping_enabled() -> bool:
            if symbolic_domain_sizes is None:
                return False
            raw = symbolic_domain_sizes.get("use_horizontal_start_mapping")
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, numbers.Integral):
                return int(raw) != 0
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "yes", "on"}
            return False

        def _entity_start_bounds_from_horizontal_start(
            entity_name: str,
        ) -> dict[int, tuple[int, int, int, int]] | None:
            if symbolic_domain_sizes is None:
                return None
            # print(f"Entered _entity_start_bounds_from_horizontal_start for {entity_name}")
            if not _horizontal_start_mapping_enabled():
                return None
            # print("Horizontal start mapping is enabled.")
            max_i_int = _pick_symbolic_int("i_max", "domain_i_max", "max_i", "domain_max_i", "nx")
            max_j_int = _pick_symbolic_int("j_max", "domain_j_max", "max_j", "domain_max_j", "ny")
            if max_i_int is None or max_j_int is None:
                return None

            if entity_name == "Edge":
                mapping_value = symbolic_domain_sizes.get("edge_to_ijk")
                mapping_rows = _coerce_mapping_rows(mapping_value, expected_len=3)
                horizontal_start = _pick_symbolic_int(
                    "horizontal_start_edge",
                    "horizontal_start_e",
                    "horizontal_start",
                )
            elif entity_name == "Vertex":
                mapping_value = symbolic_domain_sizes.get("vertex_to_ij")
                mapping_rows = _coerce_mapping_rows(mapping_value, expected_len=2)
                horizontal_start = _pick_symbolic_int(
                    "horizontal_start_vertex",
                    "horizontal_start_v",
                    "horizontal_start",
                )
            elif entity_name == "Cell":
                mapping_value = symbolic_domain_sizes.get("cell_to_ijk")
                mapping_rows = _coerce_mapping_rows(mapping_value, expected_len=3)
                horizontal_start = _pick_symbolic_int(
                    "horizontal_start_cell",
                    "horizontal_start_c",
                    "horizontal_start",
                )
            else:
                return None
            # print(f"Mapping value for {entity_name}:", mapping_value, " with horizontal start:", horizontal_start)
            if mapping_rows is None or horizontal_start is None:
                return None
            if horizontal_start < 0:
                return None
            
            # print(f"Deriving {entity_name} start bounds from horizontal start mapping:\n", _derive_entity_start_bounds_from_mapping(
            #     entity_name,
            #     mapping_rows=mapping_rows,
            #     horizontal_start=horizontal_start,
            #     max_i=max_i_int,
            #     max_j=max_j_int,
            # ))

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
            *,
            kolor: int | None = None,
        ) -> tuple[ir.Expr, ir.Expr] | None:
            if axis_name not in {"IDim", "JDim"}:
                return None
            # print(f"Attempting to get mapping-based axis bounds for {entity_name} along {axis_name} with kolor={kolor}")
            bounds_by_kolor = _entity_start_bounds_from_horizontal_start(entity_name)
            if not bounds_by_kolor:
                return None

            # axis_index = 0 if axis_name == "IDim" else 1
            # axis_hi_index = 2 if axis_name == "IDim" else 3

            # if entity_name == "Edge" and kolor is not None:
            #     if kolor not in bounds_by_kolor:
            #         return None
            #     axis_lo = bounds_by_kolor[kolor][axis_index]
            #     axis_hi = bounds_by_kolor[kolor][axis_hi_index] + 1
            # else:
            #     axis_lo = min(pair[axis_index] for pair in bounds_by_kolor.values())
            #     axis_hi = max(pair[axis_hi_index] for pair in bounds_by_kolor.values()) + 1
            # return ir.OffsetLiteral(value=axis_lo), ir.OffsetLiteral(value=axis_hi)
            axis_index = 0 if axis_name == "IDim" else 1
            axis_hi_index = 2 if axis_name == "IDim" else 3

            if entity_name == "Edge":
            # Per-kolor exact bounds when caller requests a specific kolor
                if kolor is not None:
                    if kolor not in bounds_by_kolor:
                        return None
                    axis_lo = bounds_by_kolor[kolor][axis_index]
                    axis_hi = bounds_by_kolor[kolor][axis_hi_index] + 1
                    return ir.OffsetLiteral(value=axis_lo), ir.OffsetLiteral(value=axis_hi)
            # No single-kolor requested: use the absolute outer bounds across all kolors.
            axis_los = [pair[axis_index] for pair in bounds_by_kolor.values()]
            axis_his = [pair[axis_hi_index] for pair in bounds_by_kolor.values()]

            axis_lo_val = min(axis_los)
            axis_hi_val = max(axis_his) + 1

            return ir.OffsetLiteral(value=axis_lo_val), ir.OffsetLiteral(value=axis_hi_val)

        def _lateral_size() -> ir.Expr:
            lateral = _pick_size_param("lateral", "lateral_bounds")
            if lateral is not None:
                return lateral

            if symbolic_domain_sizes is not None and "lateral_edge" in symbolic_domain_sizes:
                lateral_edge = symbolic_domain_sizes["lateral_edge"]
                if isinstance(lateral_edge, numbers.Integral):
                    return ir.OffsetLiteral(value=max(0, int(lateral_edge) // 2))
                if isinstance(lateral_edge, str):
                    try:
                        return ir.OffsetLiteral(value=max(0, int(lateral_edge) // 2))
                    except ValueError:
                        pass

            return ir.OffsetLiteral(value=0)

        def _offset_int_value(expr: ir.Expr) -> int | None:
            if isinstance(expr, ir.OffsetLiteral) and isinstance(expr.value, int):
                return expr.value
            return None

        def _offset_add(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
            lhs_val = _offset_int_value(lhs)
            rhs_val = _offset_int_value(rhs)
            if lhs_val is not None and rhs_val is not None:
                return ir.OffsetLiteral(value=lhs_val + rhs_val)
            if rhs_val == 0:
                return copy.deepcopy(lhs)
            if lhs_val == 0:
                return copy.deepcopy(rhs)
            return im.plus(copy.deepcopy(lhs), copy.deepcopy(rhs))

        def _offset_sub(lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
            lhs_val = _offset_int_value(lhs)
            rhs_val = _offset_int_value(rhs)
            if lhs_val is not None and rhs_val is not None:
                return ir.OffsetLiteral(value=lhs_val - rhs_val)
            if rhs_val == 0:
                return copy.deepcopy(lhs)
            return im.minus(copy.deepcopy(lhs), copy.deepcopy(rhs))

        def _cartesian_axis_bounds(
            axis_name: str, *, use_lateral: bool = True
        ) -> tuple[ir.Expr, ir.Expr] | None:
            if axis_name == "IDim":
                lower_base = _pick_size_param("i_min", "domain_i_min", "imin")
                upper_base = _pick_size_param(
                    "i_max", "domain_i_max", "max_i", "domain_max_i", "nx", "num_i", "ni"
                )
                if upper_base is None:
                    return None
                lower = ir.OffsetLiteral(value=0) if lower_base is None else copy.deepcopy(lower_base)
                upper = copy.deepcopy(upper_base)
                if use_lateral:
                    lateral = _lateral_size()
                    lower = _offset_add(lower, copy.deepcopy(lateral))
                    upper = _offset_sub(upper, lateral)
                return lower, upper
            if axis_name == "JDim":
                lower_base = _pick_size_param("j_min", "domain_j_min", "jmin")
                upper_base = _pick_size_param(
                    "j_max", "domain_j_max", "max_j", "domain_max_j", "ny", "num_j", "nj"
                )
                if upper_base is None:
                    return None
                lower = ir.OffsetLiteral(value=0) if lower_base is None else copy.deepcopy(lower_base)
                upper = copy.deepcopy(upper_base)
                if use_lateral:
                    lateral = _lateral_size()
                    lower = _offset_add(lower, copy.deepcopy(lateral))
                    upper = _offset_sub(upper, lateral)
                return lower, upper
            if axis_name == "Kolor":
                return ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=3)
            return None

        def _entity_kolor_bounds(axis_name: str) -> tuple[ir.Expr, ir.Expr] | None:
            kolor_stops = {"Vertex": 1, "Cell": 2, "Edge": 3}
            if axis_name not in kolor_stops:
                return None
            return ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=kolor_stops[axis_name])

        def _entity_cartesian_bounds(
            entity_name: str,
            axis_name: str,
            *,
            extra_halo: int = 0,
            use_lateral: bool = True,
            kolor: int | None = None,
        ) -> tuple[ir.Expr, ir.Expr] | None:
            bounds = None
            if use_lateral:
                bounds = _mapping_based_axis_bounds(entity_name, axis_name, kolor=kolor)
            if bounds is None:
                bounds = _cartesian_axis_bounds(axis_name, use_lateral=use_lateral)
            if bounds is None:
                return None
            lo, hi = bounds
            # Cell-centered fields live on nx*ny interior cells, not on the full
            # vertex-like (nx+1)*(ny+1) extent. Clip one extra layer on both
            # horizontal axes when remapping Cell domains.
            if entity_name == "Cell" and not (
                use_lateral and _mapping_based_axis_bounds(entity_name, axis_name, kolor=kolor) is not None
            ):
                hi = _offset_sub(hi, ir.OffsetLiteral(value=1))
            if axis_name in {"IDim", "JDim"} and extra_halo > 0:
                extra = ir.OffsetLiteral(value=extra_halo)
                lo = _offset_add(lo, extra)
                hi = _offset_sub(hi, extra)
            return lo, hi

        def _axis_name(axis_expr: ir.Expr) -> str | None:
            if isinstance(axis_expr, ir.AxisLiteral) and isinstance(axis_expr.value, str):
                return axis_expr.value
            if isinstance(axis_expr, common.Dimension):
                return axis_expr.value
            return (
                getattr(axis_expr, "value", None)
                if isinstance(getattr(axis_expr, "value", None), str)
                else None
            )

        def _structured_entity_condition(
            entity_name: str, *, extra_halo: int = 0, use_lateral: bool = True
        ) -> ir.Expr | None:
            idim_bounds = _entity_cartesian_bounds(
                entity_name, "IDim", extra_halo=extra_halo, use_lateral=use_lateral
            )
            jdim_bounds = _entity_cartesian_bounds(
                entity_name, "JDim", extra_halo=extra_halo, use_lateral=use_lateral
            )
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

            # Edge storage is stacked as three kolors with different valid extents:
            # k0: nx*(ny+1), k1: (nx+1)*ny, k2: nx*ny.
            # Keep comparator predicates kolor-aware to avoid carving out wrong
            # interior stripes when one global IDim/JDim mask is applied.
            if entity_name == "Edge" and extra_halo > 0:
                mapping_k0_idim = _entity_cartesian_bounds(
                    entity_name,
                    "IDim",
                    extra_halo=extra_halo,
                    use_lateral=use_lateral,
                    kolor=0,
                )
                mapping_k0_jdim = _entity_cartesian_bounds(
                    entity_name,
                    "JDim",
                    extra_halo=extra_halo,
                    use_lateral=use_lateral,
                    kolor=0,
                )
                mapping_k1_idim = _entity_cartesian_bounds(
                    entity_name,
                    "IDim",
                    extra_halo=extra_halo,
                    use_lateral=use_lateral,
                    kolor=1,
                )
                mapping_k1_jdim = _entity_cartesian_bounds(
                    entity_name,
                    "JDim",
                    extra_halo=extra_halo,
                    use_lateral=use_lateral,
                    kolor=1,
                )
                mapping_k2_idim = _entity_cartesian_bounds(
                    entity_name,
                    "IDim",
                    extra_halo=extra_halo,
                    use_lateral=use_lateral,
                    kolor=2,
                )
                mapping_k2_jdim = _entity_cartesian_bounds(
                    entity_name,
                    "JDim",
                    extra_halo=extra_halo,
                    use_lateral=use_lateral,
                    kolor=2,
                )
                if all(
                    bound is not None
                    for bound in (
                        mapping_k0_idim,
                        mapping_k0_jdim,
                        mapping_k1_idim,
                        mapping_k1_jdim,
                        mapping_k2_idim,
                        mapping_k2_jdim,
                    )
                ):
                    assert mapping_k0_idim is not None
                    assert mapping_k0_jdim is not None
                    assert mapping_k1_idim is not None
                    assert mapping_k1_jdim is not None
                    assert mapping_k2_idim is not None
                    assert mapping_k2_jdim is not None

                    cond_k0 = im.and_(
                        _axis_domain(Kolor, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1)),
                        im.and_(
                            _axis_domain(IDim, *mapping_k0_idim),
                            _axis_domain(JDim, *mapping_k0_jdim),
                        ),
                    )
                    cond_k1 = im.and_(
                        _axis_domain(Kolor, ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2)),
                        im.and_(
                            _axis_domain(IDim, *mapping_k1_idim),
                            _axis_domain(JDim, *mapping_k1_jdim),
                        ),
                    )
                    cond_k2 = im.and_(
                        _axis_domain(Kolor, ir.OffsetLiteral(value=2), ir.OffsetLiteral(value=3)),
                        im.and_(
                            _axis_domain(IDim, *mapping_k2_idim),
                            _axis_domain(JDim, *mapping_k2_jdim),
                        ),
                    )
                    return im.or_(cond_k0, im.or_(cond_k1, cond_k2))

                i_lo, i_hi = idim_bounds
                j_lo, j_hi = jdim_bounds
                i_lo_k1k2 = _offset_sub(i_lo, ir.OffsetLiteral(value=1))
                j_lo_k0k2 = _offset_sub(j_lo, ir.OffsetLiteral(value=1))
                # i_hi_k1k2 = _offset_sub(i_hi, ir.OffsetLiteral(value=1))
                # j_hi_k0k2 = _offset_sub(j_hi, ir.OffsetLiteral(value=1))

                cond_k0 = im.and_(
                    _axis_domain(Kolor, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1)),
                    im.and_(
                        _axis_domain(IDim, i_lo, i_hi),
                        _axis_domain(JDim, j_lo_k0k2, j_hi),
                    ),
                )
                cond_k1 = im.and_(
                    _axis_domain(Kolor, ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2)),
                    im.and_(
                        _axis_domain(IDim, i_lo_k1k2, i_hi),
                        _axis_domain(JDim, j_lo, j_hi),
                    ),
                )
                cond_k2 = im.and_(
                    _axis_domain(Kolor, ir.OffsetLiteral(value=2), ir.OffsetLiteral(value=3)),
                    im.and_(
                        _axis_domain(IDim, i_lo_k1k2, i_hi),
                        _axis_domain(JDim, j_lo_k0k2, j_hi),
                    ),
                )
                return im.or_(cond_k0, im.or_(cond_k1, cond_k2))

            domain_expr = im.call("cartesian_domain")(
                im.named_range(IDim, *idim_bounds),
                im.named_range(JDim, *jdim_bounds),
                im.named_range(Kolor, *kolor_bounds),
            )
            return _concat_where_condition_from_domain(domain_expr)

        def _expr_symbol_id(expr: ir.Expr) -> str | None:
            if isinstance(expr, ir.SymRef):
                return str(expr.id)
            return None

        def _expr_contains_symbol(expr: ir.Expr, symbol_ids: set[str]) -> bool:
            if isinstance(expr, ir.SymRef):
                return str(expr.id) in symbol_ids
            if isinstance(expr, ir.FunCall):
                if _expr_contains_symbol(expr.fun, symbol_ids):
                    return True
                return any(_expr_contains_symbol(arg, symbol_ids) for arg in expr.args)
            return False

        # Remap scalar comparisons against unstructured horizontal axes (Edge/Vertex/Cell)
        # to structured IDim/JDim conditions so concat_where predicates stay structured.
        if (
            cpm.is_call_to(new_node, ("greater_equal", "greater", "less_equal", "less"))
            and len(new_node.args) == 2
        ):
            lhs, rhs = new_node.args
            lhs_axis = _axis_name(lhs)
            rhs_axis = _axis_name(rhs)
            axis_side = None
            entity_axis = None

            if lhs_axis in {"Edge", "Vertex", "Cell"}:
                axis_side = "lhs"
                entity_axis = lhs_axis
            elif rhs_axis in {"Edge", "Vertex", "Cell"}:
                axis_side = "rhs"
                entity_axis = rhs_axis

            if entity_axis is not None:
                op_name = str(new_node.fun.id)
                threshold_expr = rhs if axis_side == "lhs" else lhs
                threshold_id = _expr_symbol_id(threshold_expr)
                is_interior_threshold = threshold_id == "interior_idx" or _expr_contains_symbol(
                    threshold_expr, {"interior_idx"}
                )
                is_halo_threshold = threshold_id == "halo_idx" or _expr_contains_symbol(
                    threshold_expr, {"halo_idx"}
                )
                is_start_2nd_nudge_threshold = (
                    threshold_id == "start_2nd_nudge_line_idx_e"
                    or _expr_contains_symbol(threshold_expr, {"start_2nd_nudge_line_idx_e"})
                )

                def _mapping_based_threshold_condition(tid: str) -> ir.Expr | None:
                    """Build per-kolor condition from precomputed bounds stored in symbolic_domain_sizes."""
                    if symbolic_domain_sizes is None:
                        return None
                    IDim_d = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
                    JDim_d = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
                    Kolor_d = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

                    def _dom(axis, lo_val: int, hi_val: int) -> ir.Expr:
                        return im.call("cartesian_domain")(
                            im.named_range(
                                copy.deepcopy(axis),
                                ir.OffsetLiteral(value=lo_val),
                                ir.OffsetLiteral(value=hi_val),
                            )
                        )

                    conds = []
                    for k in range(3):
                        ilo = symbolic_domain_sizes.get(f"{tid}_k{k}_ilo")
                        jlo = symbolic_domain_sizes.get(f"{tid}_k{k}_jlo")
                        ihi = symbolic_domain_sizes.get(f"{tid}_k{k}_ihi")
                        jhi = symbolic_domain_sizes.get(f"{tid}_k{k}_jhi")
                        if None in (ilo, jlo, ihi, jhi):
                            return None
                        conds.append(im.and_(
                            _dom(Kolor_d, k, k + 1),
                            im.and_(
                                _dom(IDim_d, int(ilo), int(ihi)),
                                _dom(JDim_d, int(jlo), int(jhi)),
                            ),
                        ))
                    return im.or_(conds[0], im.or_(conds[1], conds[2])) if len(conds) == 3 else None

                # Prefer exact mapping-based bounds (injected at compile time by the wrapper).
                has_threshold_mapping = (
                    entity_axis == "Edge"
                    and threshold_id is not None
                    and symbolic_domain_sizes is not None
                    and f"{threshold_id}_k0_ilo" in symbolic_domain_sizes
                )

                extra_halo = 0
                use_lateral = True
                if has_threshold_mapping:
                    interior_cond = _mapping_based_threshold_condition(threshold_id)
                elif entity_axis == "Edge" and is_start_2nd_nudge_threshold:
                    # Legacy fallback: symmetric clip when mapping bounds not available.
                    extra_halo = 2
                    interior_cond = _structured_entity_condition(
                        entity_axis, extra_halo=extra_halo, use_lateral=use_lateral
                    )
                elif entity_axis == "Cell" and is_interior_threshold:
                    extra_halo = 1
                    interior_cond = _structured_entity_condition(
                        entity_axis, extra_halo=extra_halo, use_lateral=use_lateral
                    )
                elif entity_axis == "Cell" and is_halo_threshold:
                    use_lateral = False
                    interior_cond = _structured_entity_condition(
                        entity_axis, extra_halo=extra_halo, use_lateral=use_lateral
                    )
                else:
                    interior_cond = _structured_entity_condition(
                        entity_axis, extra_halo=extra_halo, use_lateral=use_lateral
                    )

                if interior_cond is not None:
                    if is_halo_threshold:
                        is_positive = (
                            axis_side == "lhs" and op_name in {"less_equal", "less"}
                        ) or (axis_side == "rhs" and op_name in {"greater_equal", "greater"})
                    elif is_interior_threshold or is_start_2nd_nudge_threshold or has_threshold_mapping:
                        is_positive = (
                            axis_side == "lhs" and op_name in {"greater_equal", "greater"}
                        ) or (axis_side == "rhs" and op_name in {"less_equal", "less"})
                    else:
                        return new_node
                    return interior_cond if is_positive else im.not_(interior_cond)

        if cpm.is_call_to(new_node, "get_domain_range") and len(new_node.args) == 2:
            axis_name = _axis_name(new_node.args[1])
            if axis_name is not None and (bounds := _cartesian_axis_bounds(axis_name)) is not None:
                return im.make_tuple(*bounds)

        def _extract_field_from_get_domain_range(
            expr: ir.Expr, expected_axis: str
        ) -> ir.Expr | None:
            if not cpm.is_call_to(expr, "tuple_get") or len(expr.args) != 2:
                return None
            gdr = expr.args[1]
            if not cpm.is_call_to(gdr, "get_domain_range") or len(gdr.args) != 2:
                return None
            if _axis_name(gdr.args[1]) != expected_axis:
                return None
            return gdr.args[0]

        if cpm.is_call_to(new_node, "tuple_get") and len(new_node.args) == 2:
            tuple_index, gdr = new_node.args
            if cpm.is_call_to(gdr, "make_tuple") and isinstance(tuple_index, ir.Literal):
                idx = int(tuple_index.value)
                if 0 <= idx < len(gdr.args):
                    return copy.deepcopy(gdr.args[idx])
            if cpm.is_call_to(gdr, "get_domain_range") and len(gdr.args) == 2:
                axis_name = _axis_name(gdr.args[1])
                if (
                    axis_name is not None
                    and isinstance(tuple_index, ir.OffsetLiteral)
                    and tuple_index.value in {0, 1}
                ):
                    if (bounds := _cartesian_axis_bounds(axis_name)) is not None:
                        idx = int(tuple_index.value)  # mypy: ensure an int index
                        return copy.deepcopy(bounds[idx])

        # if cpm.is_call_to(new_node, "cartesian_domain") and len(new_node.args) == 1:
        #     nr = new_node.args[0]
        #     if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
        #         axis_name = _axis_name(nr.args[0])
        #         if axis_name in {"Edge", "Vertex", "Cell"}:
        #             idim_bounds = _entity_cartesian_bounds(axis_name, "IDim")
        #             jdim_bounds = _entity_cartesian_bounds(axis_name, "JDim")
        #             kolor_bounds = _entity_kolor_bounds(axis_name)
        #             if (
        #                 idim_bounds is not None
        #                 and jdim_bounds is not None
        #                 and kolor_bounds is not None
        #             ):
        #                 IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        #                 JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        #                 Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
        #                 return im.call("cartesian_domain")(
        #                     im.named_range(IDim, *idim_bounds),
        #                     im.named_range(JDim, *jdim_bounds),
        #                     im.named_range(Kolor, *kolor_bounds),
        #                 )

        if cpm.is_call_to(new_node, "unstructured_domain") or cpm.is_call_to(
            new_node, "cartesian_domain"
        ):
            new_ranges: list[ir.Expr] = []
            needs_remap = False
            for nr in new_node.args:
                if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                    axis_name = _axis_name(nr.args[0])
                    # If it's a horizontal unstructured dimension, split it into 3 Cartesian dimensions
                    if axis_name in {"Edge", "Vertex", "Cell"}:
                        idim_bounds = _entity_cartesian_bounds(axis_name, "IDim")
                        jdim_bounds = _entity_cartesian_bounds(axis_name, "JDim")
                        kolor_bounds = _entity_kolor_bounds(axis_name)
                        if (
                            idim_bounds is not None
                            and jdim_bounds is not None
                            and kolor_bounds is not None
                        ):
                            IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
                            JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
                            Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
                            new_ranges.extend(
                                [
                                    im.named_range(IDim, *idim_bounds),
                                    im.named_range(JDim, *jdim_bounds),
                                    im.named_range(Kolor, *kolor_bounds),
                                ]
                            )
                            needs_remap = True
                            continue

                # If it's a vertical dimension (K), or we couldn't remap it, keep it exactly as it is
                new_ranges.append(nr)

            if needs_remap:
                return im.call("cartesian_domain")(*new_ranges)

        return new_node


# =====================================================================
# Pass 2: Unrolling Reductions and Shifts
# =====================================================================


@dataclasses.dataclass
class CartesianReductionUnroller(NodeTranslator):
    @staticmethod
    def _collect_neighbor_tags(expr: ir.Expr) -> set[str]:
        tags: set[str] = set()

        def _walk(node: ir.Node) -> None:
            if cpm.is_call_to(node, "neighbors") and len(node.args) == 2:
                offset = node.args[0]
                if isinstance(offset, ir.OffsetLiteral) and isinstance(offset.value, str):
                    tags.add(offset.value)

            if isinstance(node, ir.FunCall):
                _walk(node.fun)
                for arg in node.args:
                    _walk(arg)
            elif isinstance(node, ir.Lambda):
                _walk(node.expr)

        _walk(expr)
        return tags

    @staticmethod
    def _mapped_connection_size(expr: ir.Expr) -> int | None:
        tags = CartesianReductionUnroller._collect_neighbor_tags(expr)
        if len(tags) != 1:
            return None

        conn = next(iter(tags))
        idx_values = sorted(
            key[1].value
            for key in map_dict
            if key[0].value == conn and isinstance(key[1].value, int)
        )
        if not idx_values or idx_values != list(range(len(idx_values))):
            return None
        return len(idx_values)

    @staticmethod
    def _extract_generic_reduce_inputs(
        expr: ir.Expr,
    ) -> tuple[ir.Expr, ir.Expr, ir.Expr, int] | None:
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
        if (conn_size := CartesianReductionUnroller._mapped_connection_size(list_expr)) is None:
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
                isinstance(conn, ir.OffsetLiteral)
                and isinstance(conn.value, str)
                and isinstance(it, ir.SymRef)
                and str(it.id) in bindings
            ):
                key: tuple[ir.OffsetLiteral, ir.OffsetLiteral] = (
                    ir.OffsetLiteral(value=conn.value),
                    ir.OffsetLiteral(value=index),
                )
                if key in map_dict:
                    entry = cast(dict[str, object], map_dict[key])
                    ref = cast(ir.Expr, copy.deepcopy(bindings[str(it.id)]["ref"]))
                    if entry["kind"] == "shift":
                        return _bounded_shifted_deref(
                            ref,
                            cast(tuple[ir.OffsetLiteral, ...], entry["shifts"]),
                            neutral_element,
                            domain_bounds,
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
                                ref,
                                cast(tuple[ir.OffsetLiteral, ...], entry["shifts"]),
                                neutral_element,
                                domain_bounds,
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
                elem = CartesianReductionUnroller._local_list_element_expr(
                    arg, index, bindings, neutral_element, domain_bounds
                )
                if elem is None:
                    return None
                elem_args.append(elem)
            return im.call(mapped_op)(*elem_args)

        if cpm.is_call_to(expr, "if_") and len(expr.args) == 3:
            cond, true_val, false_val = expr.args
            true_elem = CartesianReductionUnroller._local_list_element_expr(
                true_val, index, bindings, neutral_element, domain_bounds
            )
            false_elem = CartesianReductionUnroller._local_list_element_expr(
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
                # Is it neighbors?
                if cpm.is_call_to(stencil.expr, "neighbors") and len(stencil.expr.args) == 2:
                    conn_expr = stencil.expr.args[0]
                    if isinstance(conn_expr, ir.OffsetLiteral) and isinstance(conn_expr.value, str):
                        conn = conn_expr.value
                        field_expr = expr.args[0]
                        key: tuple[ir.OffsetLiteral, ir.OffsetLiteral] = (
                            ir.OffsetLiteral(value=conn),
                            ir.OffsetLiteral(value=idx),
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
                            normalized = conn.rstrip("ₒ")
                            is_edge_to_non_edge = (
                                normalized.startswith("E")
                                and normalized not in _EDGE_TO_EDGE_CONNECTIVITIES
                            )
                            return _build_field_concat_where_from_branches(
                                field_expr,
                                cast(tuple, entry["branches"]),
                                domain,
                                apply_edge_shape_bounds=_needs_edge_shape_bounds(conn),
                                current_kolor=current_kolor if is_edge_to_non_edge else None,
                            )

                # Is it map_ ?
                if isinstance(stencil.expr, ir.FunCall) and cpm.is_call_to(
                    stencil.expr.fun, "map_"
                ):
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
                    return im.as_fieldop(im.lambda_(*param_names)(scalar_op_call), domain)(
                        *new_args
                    )

        # Fallback
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
                if (
                    cpm.is_call_to(range_expr, "named_range")
                    and len(range_expr.args) == 3
                    and isinstance(range_expr.args[0], ir.AxisLiteral)
                ):
                    axis_name = range_expr.args[0].value
                    if axis_name in {"IDim", "JDim"}:
                        domain_bounds[axis_name] = (
                            copy.deepcopy(range_expr.args[1]),
                            copy.deepcopy(range_expr.args[2]),
                        )

        widened_init = copy.deepcopy(red_init)
        if isinstance(red_init, ir.Literal) and isinstance(red_init.type, ts.ScalarType):
            if red_init.type.kind in {
                ts.ScalarKind.INT8,
                ts.ScalarKind.INT16,
                ts.ScalarKind.INT32,
                ts.ScalarKind.INT64,
            }:
                widened_init = im.literal(red_init.value, "int64")
        elif isinstance(red_init, ir.OffsetLiteral) and isinstance(red_init.value, int):
            widened_init = im.literal(str(red_init.value), "int64")

        neutral_element = _neutral_element_for_reduce_op(red_op, widened_init)
        if (
            isinstance(widened_init, ir.Literal)
            and isinstance(widened_init.type, ts.ScalarType)
            and widened_init.type.kind
            in {ts.ScalarKind.INT8, ts.ScalarKind.INT16, ts.ScalarKind.INT32, ts.ScalarKind.INT64}
            and isinstance(neutral_element, ir.Literal)
            and isinstance(neutral_element.type, ts.ScalarType)
            and neutral_element.type.kind == ts.ScalarKind.FLOAT64
        ):
            widened_init = copy.deepcopy(neutral_element)

        # Fast-path for direct neighbor reduction
        if cpm.is_call_to(list_expr, "neighbors") and len(list_expr.args) == 2:
            conn_expr = list_expr.args[0]
            if isinstance(conn_expr, ir.OffsetLiteral) and isinstance(conn_expr.value, str):
                conn = conn_expr.value
                input_field = copy.deepcopy(list_expr.args[1])

                acc_field = im.as_fieldop(
                    im.lambda_("__acc_init")(im.deref("__acc_init")),
                    domain,
                )(
                    im.as_fieldop(im.lambda_("__x")(copy.deepcopy(widened_init)), domain)(
                        input_field
                    )
                )

                elem_field: ir.Expr | None = None
                for idx in range(conn_size):
                    key: tuple[ir.OffsetLiteral, ir.OffsetLiteral] = (
                        ir.OffsetLiteral(value=conn),
                        ir.OffsetLiteral(value=idx),
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

        # Fast-path for fieldops via map_dict (e.g. weighted sum via map)
        if cpm.is_applied_as_fieldop(list_expr):

            def _extract_base_field(node: ir.Expr) -> ir.Expr:
                if cpm.is_applied_as_fieldop(node):
                    return _extract_base_field(node.args[0])
                return node

            base_field = copy.deepcopy(_extract_base_field(list_expr))
            acc_field = im.as_fieldop(
                im.lambda_("__acc_init")(im.deref("__acc_init")),
                domain,
            )(im.as_fieldop(im.lambda_("__x")(copy.deepcopy(widened_init)), domain)(base_field))
            for idx in range(conn_size):
                elem_field = CartesianReductionUnroller._eval_list_field_at_idx(
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

            # Fallback to older pointwise strategy if the field-level approach fails
            stencil = list_expr.fun.args[0]
            if isinstance(stencil, ir.Lambda):
                param_names: list[str] = []
                call_args: list[ir.Expr] = []
                bindings: dict[str, dict[str, ir.Expr | str]] = {}

                for i, (param, arg_expr) in enumerate(
                    zip(stencil.params, list_expr.args, strict=True)
                ):
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
                                    "kind": "neighbors",
                                    "conn": stencil_conn,
                                    "ref": local_ref,
                                }
                                param_names.append(local_name)
                                call_args.append(copy.deepcopy(bound_arg.args[0]))
                                continue

                    bindings[str(param.id)] = {"kind": "list", "ref": local_ref}
                    param_names.append(local_name)
                    call_args.append(bound_arg)

                acc = copy.deepcopy(widened_init)
                for idx in range(conn_size):
                    elem = CartesianReductionUnroller._local_list_element_expr(
                        stencil.expr,
                        idx,
                        bindings,
                        _neutral_element_for_reduce_op(red_op, widened_init),
                        domain_bounds,
                    )
                    if elem is None:
                        break
                    acc = im.call(copy.deepcopy(red_op))(acc, elem)
                else:
                    return im.as_fieldop(im.lambda_(*param_names)(acc), domain)(*call_args)

        lst_name = "__cart_reduce_lst"
        lst_ref = im.ref(lst_name)
        lst_val = im.deref(lst_ref)

        acc = copy.deepcopy(widened_init)
        for idx in range(conn_size):
            elem = im.list_get(im.literal(str(idx), "int32"), lst_val)
            acc = im.call(copy.deepcopy(red_op))(acc, elem)

        return im.as_fieldop(im.lambda_(lst_name)(acc), domain)(copy.deepcopy(list_expr))

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)

    def visit_SetAt(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        new_domain = self.visit(node.domain, **kwargs)
        # Extract a fixed kolor from the domain (set by CartesianDomainAndTypeRemapper's
        # per-kolor split) and propagate it so shift expansion uses kolor-specific branches.
        current_kolor = _kolor_from_domain(new_domain)
        if current_kolor is None and cpm.is_call_to(new_domain, "make_tuple"):
            for sub in new_domain.args:
                current_kolor = _kolor_from_domain(sub)
                if current_kolor is not None:
                    break

        if current_kolor is None and not _e2c2e_on_local_intermediate(node.expr):
            # Non-split SetAt: the expression may be wrapped in the 3-kolor domain concat_where
            # by _build_edge_validity_masked_expr. Peel each kolor branch and visit with the
            # correct current_kolor so inner shift concat_wheres are resolved per-kolor.
            # Excluded: stencils where E2C2EO/E2C2E operates on a LOCAL intermediate edge field
            # (lambda-bound variable). Those intermediates are only defined for the current kolor,
            # so cross-kolor E2C2EO shifts on them would read garbage.
            new_expr = self._visit_expr_with_kolor_branches(
                node.expr, new_domain, **kwargs
            )
        else:
            new_expr = self.visit(
                node.expr, current_domain=new_domain, current_kolor=current_kolor, **kwargs
            )
        new_target = self.visit(node.target, **kwargs)
        return ir.SetAt(expr=new_expr, domain=new_domain, target=new_target)

    def _visit_expr_with_kolor_branches(
        self, expr: ir.Expr, domain: ir.Expr, **kwargs
    ) -> ir.Expr:
        """Visit a 3-kolor domain concat_where expression branch-by-branch with per-kolor context.

        When CartesianDomainAndTypeRemapper wraps an expression in:
            concat_where(cond_k0 ∧ domain_k0, expr,
              concat_where(cond_k1 ∧ domain_k1, expr,
                concat_where(cond_k2 ∧ domain_k2, expr, target)))
        each branch contains identical copies of the full expression, but the inner shift
        concat_wheres still branch on kolor. Since we know which kolor each outer branch
        corresponds to, we can resolve the inner shift concat_where immediately by visiting
        each branch with the matching current_kolor.
        """
        branches: list[tuple[int, ir.Expr, ir.Expr]] = []  # (kolor, condition, branch_expr)
        tail: ir.Expr | None = None
        cur = expr

        while cpm.is_call_to(cur, "concat_where") and len(cur.args) == 3:
            cond, branch_expr, rest = cur.args
            k = _kolor_from_cw_condition(cond)
            if k is None:
                break
            branches.append((k, cond, branch_expr))
            cur = rest

        if not branches:
            # No peelable kolor structure — visit normally without kolor context.
            return self.visit(expr, current_domain=domain, current_kolor=None, **kwargs)

        tail = cur  # remaining expression after all kolor branches (usually the target field)

        # Visit each kolor branch with the known current_kolor.
        visited: list[tuple[int, ir.Expr, ir.Expr]] = []
        for k, cond, branch_expr in branches:
            # Build a per-kolor domain from the condition for use as current_domain.
            per_kolor_domain = _kolor_cw_condition_to_domain(cond, domain)
            visited_expr = self.visit(
                branch_expr,
                current_domain=per_kolor_domain if per_kolor_domain is not None else domain,
                current_kolor=k,
                **kwargs,
            )
            visited.append((k, cond, visited_expr))

        visited_tail = self.visit(tail, current_domain=domain, current_kolor=None, **kwargs)

        # Reassemble the concat_where chain.
        result = visited_tail
        for k, cond, visited_expr in reversed(visited):
            result = im.concat_where(copy.deepcopy(cond), visited_expr, result)
        return result

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        current_domain = kwargs.get("current_domain")
        current_kolor: int | None = kwargs.get("current_kolor")

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
                # mypy: cast the runtime tuple of OffsetLiterals to the expected key type
                key = cast(tuple[ir.OffsetLiteral, ir.OffsetLiteral], tuple(shift_call.fun.args))
                if key in map_dict:
                    entry = cast(dict[str, object], map_dict[key])
                    rewritten_arg = self.visit(node.args[0], **kwargs)

                    # Provide a neutral element (0.0) as fallback for out-of-bounds accesses
                    _neutral_element = im.literal("0.0", "float64")

                    if entry["kind"] == "concat_where":
                        conn_name_str = (
                            key[0].value
                            if isinstance(key[0], ir.OffsetLiteral) and isinstance(key[0].value, str)
                            else ""
                        )
                        normalized_conn = conn_name_str.rstrip("ₒ")
                        # Pass current_kolor only for edge-to-cell/vertex connectivities (E2C, E2V).
                        # Edge-to-edge (E2C2EO, E2C2E) accesses a DIFFERENT kolor on an edge
                        # field — when that field is a local intermediate (not a program param),
                        # it is only defined for the current kolor, so cross-kolor reads are wrong.
                        is_edge_to_non_edge_conn = (
                            normalized_conn.startswith("E")
                            and normalized_conn not in _EDGE_TO_EDGE_CONNECTIVITIES
                        )
                        return _build_field_concat_where_from_branches(
                            rewritten_arg,
                            cast(tuple, entry["branches"]),
                            current_domain,
                            apply_edge_shape_bounds=_needs_edge_shape_bounds(conn_name_str or None),
                            current_kolor=current_kolor if is_edge_to_non_edge_conn else None,
                        )
                    elif entry["kind"] == "shift":
                        return _make_lifted_deref_shift(
                            rewritten_arg,
                            cast(tuple[ir.OffsetLiteral, ...], entry["shifts"]),
                            current_domain,
                        )

            if (reduce_inputs := self._extract_generic_reduce_inputs(node)) is not None:
                red_op, red_init, list_expr, conn_size = reduce_inputs
                rewritten_list_expr = self.visit(list_expr, **kwargs)
                return self._build_generic_unrolled_reduce_expr(
                    red_op, red_init, rewritten_list_expr, conn_size, current_domain,
                    current_kolor=current_kolor,
                )

        new_node = copy.deepcopy(self.generic_visit(node, **kwargs))

        if cpm.is_applied_shift(new_node):
            # mypy: these fun.args are expected to be OffsetLiteral tuples at runtime
            key = cast(tuple[ir.OffsetLiteral, ir.OffsetLiteral], tuple(new_node.fun.args))
            if key in map_dict:
                entry = cast(dict[str, object], map_dict[key])
                arg = new_node.args[0]

                if entry["kind"] == "shift":
                    return _apply_shift_chain(
                        copy.deepcopy(arg), cast(tuple[ir.OffsetLiteral, ...], entry["shifts"])
                    )

                if entry["kind"] == "concat_where":
                    conn_name = (
                        key[0].value
                        if key and isinstance(key[0], ir.OffsetLiteral) and isinstance(key[0].value, str)
                        else None
                    )
                    normalized_conn2 = conn_name.rstrip("ₒ") if conn_name else ""
                    is_edge_to_non_edge_conn2 = (
                        normalized_conn2.startswith("E")
                        and normalized_conn2 not in _EDGE_TO_EDGE_CONNECTIVITIES
                    )
                    if current_domain is not None:
                        return _build_field_concat_where_from_branches(
                            arg,
                            entry["branches"],
                            current_domain,
                            apply_edge_shape_bounds=_needs_edge_shape_bounds(conn_name),
                            current_kolor=current_kolor if is_edge_to_non_edge_conn2 else None,
                        )
                    return _build_concat_where_from_branches(arg, entry["branches"])

        return new_node


# =====================================================================
# Pass 3: Resolve Can Deref
# =====================================================================


@dataclasses.dataclass
class KolorConstantPropagation(NodeTranslator):
    """Eliminate dead kolor branches from the expanded structured IR.

    After CartesianReductionUnroller, each `neighbors()` expansion emits per-kolor
    shifted-field expressions wrapped in concat_where conditions, e.g.::

        concat_where(Kolor:[0,1), shift_k0(vn), concat_where(Kolor:[1,2), shift_k1(vn), shift_k2(vn)))

    These inner conditions appear INSIDE the outer kolor-specific branches added by
    `_build_edge_validity_masked_expr`, e.g. the kolor-0 branch already guarantees
    that the current element has Kolor=0.  The inner kolor-1 and kolor-2 branches are
    therefore dead code and inflate the IR 3x per neighbor slot.

    This pass propagates the outer kolor context inward and removes statically
    unreachable branches, reducing the expanded IR by ≈3x for edge stencils and
    similarly for two-kolor cell stencils.
    """

    @staticmethod
    def _kolor_range_from_cond(cond: ir.Expr) -> tuple[int, int] | None:
        """Return (lo, hi) if *cond* contains exactly one Kolor named_range."""
        if cpm.is_call_to(cond, "and_") and len(cond.args) == 2:
            for arg in cond.args:
                result = KolorConstantPropagation._kolor_range_from_cond(arg)
                if result is not None:
                    return result
            return None
        if not cpm.is_call_to(cond, "cartesian_domain"):
            return None
        for nr in cond.args:
            if not (cpm.is_call_to(nr, "named_range") and len(nr.args) == 3):
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
                # Inner condition is always FALSE given the outer kolor → dead branch.
                return self.visit(false_branch, known_kolor=known_kolor, **kwargs)
            if ilo >= klo and ihi <= khi:
                # Inner condition is always TRUE given the outer kolor → skip wrapper.
                return self.visit(true_branch, known_kolor=inner_range, **kwargs)

        if inner_range is not None:
            # Enter the true branch with narrowed kolor context.
            new_true = self.visit(true_branch, known_kolor=inner_range, **kwargs)
            # False branch: kolor is outside inner_range but we only know the parent range.
            new_false = self.visit(false_branch, known_kolor=known_kolor, **kwargs)
            if new_true is true_branch and new_false is false_branch:
                return node
            return ir.FunCall(fun=copy.deepcopy(node.fun), args=[cond, new_true, new_false])

        # Non-kolor concat_where: recurse with same kolor context.
        return self.generic_visit(node, known_kolor=known_kolor, **kwargs)

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node, known_kolor=None)


class RewriteCartesianCanDeref(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        new_node = self.generic_visit(node, **kwargs)

        # Replace any remaining can_deref with True since bounds checking
        # is now correctly handled via field-level concat_where wrappers.
        if cpm.is_call_to(new_node, "can_deref") and len(new_node.args) == 1:
            return im.literal("True", "bool")

        return new_node


@dataclasses.dataclass
class CartUnroll:
    _cartesian_remapped_type = staticmethod(CartesianDomainAndTypeRemapper._cartesian_remapped_type)

    @classmethod
    def apply(
        cls, node: ir.Node, *, symbolic_domain_sizes: dict[str, Any] | None = None
    ) -> ir.Node:
        transformed = CartesianDomainAndTypeRemapper.apply(
            node, symbolic_domain_sizes=symbolic_domain_sizes
        )
        return CartesianReductionUnroller.apply(transformed)
