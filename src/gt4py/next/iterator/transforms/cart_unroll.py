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
from typing import cast

import gt4py.next.iterator.transforms.map_dict as map_dict_module
from gt4py.eve import NodeTranslator
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_specifications as ts


# You can keep map_dict as a local alias so you don't have to change the rest of your code
map_dict = map_dict_module.map_dict

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
    ) -> ir.Expr | None:
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
        ),
    )


def _get_axis_name(axis_expr: ir.Expr) -> str | None:
    """Safely extracts the axis string name from either an AxisLiteral or a Dimension object."""
    if isinstance(axis_expr, ir.AxisLiteral) and isinstance(axis_expr.value, str):
        return axis_expr.value
    if hasattr(axis_expr, "value") and isinstance(axis_expr.value, str):
        return axis_expr.value
    return None


def _to_offset_literal(node: ir.Expr) -> ir.Expr:
    """Forces integer bounds to be formatted as OffsetLiterals (with 'ₒ')."""
    if isinstance(node, ir.OffsetLiteral):
        return copy.deepcopy(node)
    if hasattr(node, "value") and str(node.value).lstrip("-").isdigit():
        return ir.OffsetLiteral(value=int(str(node.value)))
    return copy.deepcopy(node)


def _compute_shift_guard_domain(
    shift_spec: tuple[ir.OffsetLiteral, ...],
    full_domain: ir.Expr,
) -> ir.Expr | None:
    if not cpm.is_call_to(full_domain, "cartesian_domain"):
        return None

    shifts_by_dim: dict[str, int] = {}
    for k in range(0, len(shift_spec), 2):
        if k + 1 >= len(shift_spec):
            break
        dim_tag = shift_spec[k]
        offset_tag = shift_spec[k + 1]
        if (
            isinstance(dim_tag, ir.OffsetLiteral)
            and isinstance(dim_tag.value, str)
            and isinstance(offset_tag, ir.OffsetLiteral)
            and isinstance(offset_tag.value, int)
        ):
            shifts_by_dim[dim_tag.value] = offset_tag.value

    domain_ranges: dict[str, tuple[ir.Expr, ir.Expr, ir.Expr]] = {}
    for range_expr in full_domain.args:
        if cpm.is_call_to(range_expr, "named_range") and len(range_expr.args) == 3:
            axis_name = _get_axis_name(range_expr.args[0])
            if axis_name is not None:
                domain_ranges[axis_name] = (
                    range_expr.args[0],
                    range_expr.args[1],
                    range_expr.args[2],
                )

    new_ranges: dict[str, tuple[ir.Expr, ir.Expr, ir.Expr]] = {}
    needs_restriction = False

    for axis_name, (axis_lit, lo, hi) in domain_ranges.items():
        lo_off = _to_offset_literal(lo)
        hi_off = _to_offset_literal(hi)

        # Hard-skip Kolor: Never shrink it.
        if axis_name == "Kolor":
            new_ranges[axis_name] = (axis_lit, lo_off, hi_off)
            continue

        offset = shifts_by_dim.get(axis_name, 0)

        if offset < 0:
            if isinstance(lo_off, ir.OffsetLiteral) and isinstance(lo_off.value, int):
                new_lo = ir.OffsetLiteral(value=max(lo_off.value, -offset))
            else:
                new_lo = ir.OffsetLiteral(value=-offset)
            new_ranges[axis_name] = (axis_lit, new_lo, hi_off)
            needs_restriction = True

        elif offset > 0:
            if isinstance(hi_off, ir.OffsetLiteral) and isinstance(hi_off.value, int):
                new_hi = ir.OffsetLiteral(value=hi_off.value - offset)
            else:
                new_hi = im.minus(hi_off, ir.OffsetLiteral(value=offset))
            new_ranges[axis_name] = (axis_lit, lo_off, new_hi)
            needs_restriction = True

        else:
            new_ranges[axis_name] = (axis_lit, lo_off, hi_off)

    if not needs_restriction:
        return None

    dim_kind_map = {
        "IDim": common.DimensionKind.HORIZONTAL,
        "JDim": common.DimensionKind.HORIZONTAL,
        "Kolor": common.DimensionKind.HORIZONTAL,
        "K": common.DimensionKind.VERTICAL,
    }
    dim_ranges: dict[common.Dimension, tuple[ir.Expr, ir.Expr]] = {}
    for axis_name, (axis_lit, lo, hi) in new_ranges.items():
        kind = (
            axis_lit.kind
            if isinstance(axis_lit, ir.AxisLiteral)
            else dim_kind_map.get(axis_name, common.DimensionKind.HORIZONTAL)
        )
        dim = common.Dimension(axis_name, kind=kind)
        dim_ranges[dim] = (lo, hi)

    return im.domain(common.GridType.CARTESIAN, dim_ranges)


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
        node: ir.Node, symbolic_domain_sizes: dict[str, str | int]
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
        if has_symbolic_structured_sizes and not _has_any(
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
        symbolic_domain_sizes: dict[str, str | int] | None = None,
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
        new_body = [self.visit(stmt, **child_kwargs) for stmt in node.body]
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
        symbolic_domain_sizes: dict[str, str | int] = kwargs.get("symbolic_domain_sizes") or {}

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

        new_domain = self.visit(node.domain, **kwargs)
        new_expr = self.visit(node.expr, current_domain=new_domain, **kwargs)
        new_target = self.visit(node.target, **kwargs)

        # Some concat_where expansions keep a stale unstructured self-reference in the
        # SetAt expression (e.g. `next_vn` in the false branch). Retype those refs to
        # the structured target so expr/target dimensions stay aligned.
        if _is_fully_structured_field(new_target):
            new_expr = _retarget_self_refs(new_expr, new_target)

        if _is_unstructured_edge_domain(node.domain):
            masked_expr = _build_edge_validity_masked_expr(new_expr, new_target, new_domain)
            if masked_expr is not None:
                new_expr = masked_expr

        return ir.SetAt(expr=new_expr, domain=new_domain, target=new_target)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        program_param_ids: set[str] = kwargs.get("program_param_ids", set())
        symbolic_domain_sizes: dict[str, str | int] | None = kwargs.get("symbolic_domain_sizes")
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
        ) -> tuple[ir.Expr, ir.Expr] | None:
            bounds = _cartesian_axis_bounds(axis_name, use_lateral=use_lateral)
            if bounds is None:
                return None
            lo, hi = bounds
            # Cell-centered fields live on nx*ny interior cells, not on the full
            # vertex-like (nx+1)*(ny+1) extent. Clip one extra layer on both
            # horizontal axes when remapping Cell domains.
            if entity_name == "Cell":
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

                extra_halo = 0
                use_lateral = True
                if entity_axis == "Edge" and is_start_2nd_nudge_threshold:
                    # Equivalent to translator edge remap with boundary_level=10.
                    extra_halo = 3
                elif entity_axis == "Cell" and is_interior_threshold:
                    # Cell interior starts one shell deeper than edge/vertex mapping.
                    extra_halo = 1
                elif entity_axis == "Cell" and is_halo_threshold:
                    # `halo_idx` is the upper Cell bound; keep full Cell coverage.
                    use_lateral = False

                interior_cond = _structured_entity_condition(
                    entity_axis, extra_halo=extra_halo, use_lateral=use_lateral
                )
                if interior_cond is not None:
                    if is_halo_threshold:
                        is_positive = (
                            axis_side == "lhs" and op_name in {"less_equal", "less"}
                        ) or (axis_side == "rhs" and op_name in {"greater_equal", "greater"})
                    elif is_interior_threshold or is_start_2nd_nudge_threshold:
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

        if cpm.is_call_to(new_node, "cartesian_domain") and len(new_node.args) == 1:
            nr = new_node.args[0]
            if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                axis_name = _axis_name(nr.args[0])
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
                        return im.call("cartesian_domain")(
                            im.named_range(IDim, *idim_bounds),
                            im.named_range(JDim, *jdim_bounds),
                            im.named_range(Kolor, *kolor_bounds),
                        )

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
        cls, expr: ir.Expr, idx: int, domain: ir.Expr | None, neutral_element: ir.Expr
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
                            return _build_field_concat_where_from_branches(
                                field_expr,
                                cast(tuple, entry["branches"]),
                                domain,
                                apply_edge_shape_bounds=_needs_edge_shape_bounds(conn),
                            )

                # Is it map_ ?
                if isinstance(stencil.expr, ir.FunCall) and cpm.is_call_to(
                    stencil.expr.fun, "map_"
                ):
                    mapped_op = stencil.expr.fun.args[0]
                    new_args = []
                    for arg in expr.args:
                        inner_elem = cls._eval_list_field_at_idx(arg, idx, domain, neutral_element)
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
                    list_expr, idx, domain, neutral_element
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
        new_expr = self.visit(node.expr, current_domain=new_domain, **kwargs)
        new_target = self.visit(node.target, **kwargs)
        return ir.SetAt(expr=new_expr, domain=new_domain, target=new_target)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        current_domain = kwargs.get("current_domain")

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
                        return _build_field_concat_where_from_branches(
                            rewritten_arg,
                            cast(tuple, entry["branches"]),
                            current_domain,
                            apply_edge_shape_bounds=_needs_edge_shape_bounds(
                                key[0].value
                                if isinstance(key[0], ir.OffsetLiteral)
                                and isinstance(key[0].value, str)
                                else None
                            ),
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
                    red_op, red_init, rewritten_list_expr, conn_size, current_domain
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
                    conn_name = None
                    if (
                        key
                        and isinstance(key[0], ir.OffsetLiteral)
                        and isinstance(key[0].value, str)
                    ):
                        conn_name = key[0].value
                    if current_domain is not None:
                        return _build_field_concat_where_from_branches(
                            arg,
                            entry["branches"],
                            current_domain,
                            apply_edge_shape_bounds=_needs_edge_shape_bounds(conn_name),
                        )
                    return _build_concat_where_from_branches(arg, entry["branches"])

        return new_node


# =====================================================================
# Pass 3: Resolve Can Deref
# =====================================================================


@dataclasses.dataclass
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
        cls, node: ir.Node, *, symbolic_domain_sizes: dict[str, str | int] | None = None
    ) -> ir.Node:
        transformed = CartesianDomainAndTypeRemapper.apply(
            node, symbolic_domain_sizes=symbolic_domain_sizes
        )
        return CartesianReductionUnroller.apply(transformed)
