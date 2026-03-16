# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import math
import copy
import os

from gt4py.next import common
from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.transforms.map_dict import map_dict as map_dict


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
    # branches: ((cond_or_none, shift_spec), ...)
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
    can_deref_shifted = im.can_deref(copy.deepcopy(shifted_ref))
    deref_shifted = im.deref(shifted_ref)
    return im.if_(can_deref_shifted, deref_shifted, copy.deepcopy(neutral_element))


def _build_field_concat_where_from_branches(arg: ir.Expr, branches, domain: ir.Expr | None = None):
    # branches: ((cond_or_none, shift_spec), ...)
    cond, shift_spec = branches[0]
    expr = _make_lifted_deref_shift(arg, shift_spec, domain)
    if len(branches) == 1:
        return expr
    return im.concat_where(
        cond, expr, _build_field_concat_where_from_branches(arg, branches[1:], domain)
    )


def _compute_shift_guard_domain(
    shift_spec: tuple[ir.OffsetLiteral, ...],
    full_domain: ir.Expr,
) -> ir.Expr | None:
    """Return the subdomain of full_domain where the shift is valid, or None if always valid."""
    if not cpm.is_call_to(full_domain, "cartesian_domain"):
        return None

    # Parse shift offsets by dimension name
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

    # Collect named ranges from the full domain
    domain_ranges: dict[str, tuple[ir.Expr, ir.Expr, ir.Expr]] = {}  # axis_name -> (axis_lit, lo, hi)
    for range_expr in full_domain.args:
        if (
            cpm.is_call_to(range_expr, "named_range")
            and len(range_expr.args) == 3
            and isinstance(range_expr.args[0], ir.AxisLiteral)
        ):
            axis_name = range_expr.args[0].value
            domain_ranges[axis_name] = (range_expr.args[0], range_expr.args[1], range_expr.args[2])

    # Compute restricted bounds
    new_ranges: dict[str, tuple[ir.Expr, ir.Expr, ir.Expr]] = {}
    needs_restriction = False
    for axis_name, (axis_lit, lo, hi) in domain_ranges.items():
        offset = shifts_by_dim.get(axis_name, 0)
        if offset < 0:
            # Need index >= -offset to keep shifted index in bounds (>= 0)
            new_lo = ir.OffsetLiteral(value=-offset)
            new_ranges[axis_name] = (axis_lit, new_lo, copy.deepcopy(hi))
            needs_restriction = True
        elif offset > 0:
            # Need index + offset < hi so index < hi - offset
            new_hi = im.minus(copy.deepcopy(hi), ir.OffsetLiteral(value=offset))
            new_ranges[axis_name] = (axis_lit, copy.deepcopy(lo), new_hi)
            needs_restriction = True
        else:
            new_ranges[axis_name] = (axis_lit, copy.deepcopy(lo), copy.deepcopy(hi))

    if not needs_restriction:
        return None

    # Build restricted cartesian domain using Dimension objects for im.domain
    dim_kind_map = {
        "IDim": common.DimensionKind.HORIZONTAL,
        "JDim": common.DimensionKind.HORIZONTAL,
        "Kolor": common.DimensionKind.HORIZONTAL,
    }
    dim_ranges: dict[common.Dimension, tuple[ir.Expr, ir.Expr]] = {}
    for axis_name, (_, lo, hi) in new_ranges.items():
        kind = dim_kind_map.get(axis_name, common.DimensionKind.HORIZONTAL)
        dim = common.Dimension(axis_name, kind=kind)
        dim_ranges[dim] = (lo, hi)

    return im.domain(common.GridType.CARTESIAN, dim_ranges)


@dataclasses.dataclass
class CartUnroll(NodeTranslator):

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

    @staticmethod
    def _is_zero_literal(expr: ir.Expr) -> bool:
        if isinstance(expr, ir.OffsetLiteral):
            return expr.value == 0
        if isinstance(expr, ir.Literal):
            return expr.value in {"0", "0.0"}
        return False

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
        tags = CartUnroll._collect_neighbor_tags(expr)
        if len(tags) != 1:
            return None

        conn = next(iter(tags))
        idx_values = sorted(
            key[1].value
            for key in map_dict
            if key[0].value == conn and isinstance(key[1].value, int)
        )
        if not idx_values:
            return None
        if idx_values != list(range(len(idx_values))):
            return None
        return len(idx_values)

    @staticmethod
    def _extract_generic_reduce_inputs(expr: ir.Expr) -> tuple[ir.Expr, ir.Expr, ir.Expr, int] | None:
        # Match generic list reduction shape:
        # as_fieldop(λ(lst) → reduce(op, init)(deref(lst)))(list_expr)
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
        if (conn_size := CartUnroll._mapped_connection_size(list_expr)) is None:
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
        def _local_concat_where(ref: ir.Expr, branches: tuple) -> ir.Expr:
            cond, shift_spec = branches[0]
            shifted = _bounded_shifted_deref(ref, shift_spec, neutral_element, domain_bounds)
            if len(branches) == 1 or cond is None:
                return shifted
            return im.concat_where(copy.deepcopy(cond), shifted, _local_concat_where(ref, branches[1:]))

        if cpm.is_call_to(expr, "neighbors") and len(expr.args) == 2:
            conn, it = expr.args
            if (
                isinstance(conn, ir.OffsetLiteral)
                and isinstance(conn.value, str)
                and isinstance(it, ir.SymRef)
                and str(it.id) in bindings
            ):
                key = (ir.OffsetLiteral(value=conn.value), ir.OffsetLiteral(value=index))
                if key in map_dict:
                    entry = map_dict[key]
                    ref = copy.deepcopy(bindings[str(it.id)]["ref"])
                    if entry["kind"] == "shift":
                        return _bounded_shifted_deref(
                            ref, entry["shifts"], neutral_element, domain_bounds
                        )
                    if entry["kind"] == "concat_where":
                        return _local_concat_where(ref, entry["branches"])
            return None

        if cpm.is_call_to(expr, "deref") and len(expr.args) == 1:
            it = expr.args[0]
            if isinstance(it, ir.SymRef) and str(it.id) in bindings:
                binding = bindings[str(it.id)]
                ref = copy.deepcopy(binding["ref"])
                kind = binding["kind"]
                if kind == "neighbors":
                    conn = binding["conn"]
                    assert isinstance(conn, str)
                    key = (ir.OffsetLiteral(value=conn), ir.OffsetLiteral(value=index))
                    if key in map_dict:
                        entry = map_dict[key]
                        if entry["kind"] == "shift":
                            return _bounded_shifted_deref(
                                ref, entry["shifts"], neutral_element, domain_bounds
                            )
                        if entry["kind"] == "concat_where":
                            return _local_concat_where(ref, entry["branches"])
                    return None

                return im.list_get(im.literal(str(index), "int32"), im.deref(ref))
            return None

        if cpm.is_applied_map(expr):
            if len(expr.fun.args) != 1:
                return None
            mapped_op = copy.deepcopy(expr.fun.args[0])
            elem_args: list[ir.Expr] = []
            for arg in expr.args:
                elem = CartUnroll._local_list_element_expr(
                    arg, index, bindings, neutral_element, domain_bounds
                )
                if elem is None:
                    return None
                elem_args.append(elem)
            return im.call(mapped_op)(*elem_args)

        if cpm.is_call_to(expr, "if_") and len(expr.args) == 3:
            cond, true_val, false_val = expr.args
            true_elem = CartUnroll._local_list_element_expr(
                true_val, index, bindings, neutral_element, domain_bounds
            )
            false_elem = CartUnroll._local_list_element_expr(
                false_val, index, bindings, neutral_element, domain_bounds
            )
            if true_elem is None or false_elem is None:
                return None
            return im.if_(copy.deepcopy(cond), true_elem, false_elem)

        return None

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
            in {
                ts.ScalarKind.INT8,
                ts.ScalarKind.INT16,
                ts.ScalarKind.INT32,
                ts.ScalarKind.INT64,
            }
            and isinstance(neutral_element, ir.Literal)
            and isinstance(neutral_element.type, ts.ScalarType)
            and neutral_element.type.kind == ts.ScalarKind.FLOAT64
        ):
            widened_init = copy.deepcopy(neutral_element)

        if cpm.is_applied_as_fieldop(list_expr):
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
                            conn = bound_stencil.expr.args[0].value
                            if isinstance(conn, str):
                                bindings[str(param.id)] = {
                                    "kind": "neighbors",
                                    "conn": conn,
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
                    elem = CartUnroll._local_list_element_expr(
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

                # Fallback for neighbor-only reductions: build field-level guarded sum using concat_where.
                if len(bindings) == 1:
                    only_binding = next(iter(bindings.values()))
                    if (
                        only_binding.get("kind") == "neighbors"
                        and isinstance(only_binding.get("conn"), str)
                        and len(call_args) == 1
                    ):
                        conn = only_binding["conn"]
                        input_field = copy.deepcopy(call_args[0])
                        acc_field = im.as_fieldop(
                            im.lambda_("__acc_init")(im.deref("__acc_init")),
                            domain,
                        )(im.as_fieldop(im.lambda_("__x")(copy.deepcopy(widened_init)), domain)(input_field))

                        neutral_field = im.as_fieldop(
                            im.lambda_("__x")(copy.deepcopy(neutral_element)),
                            domain,
                        )(input_field)

                        for idx in range(conn_size):
                            key = (ir.OffsetLiteral(value=conn), ir.OffsetLiteral(value=idx))
                            if key not in map_dict:
                                break
                            entry = map_dict[key]
                            if entry["kind"] != "shift":
                                break

                            shift_spec = entry["shifts"]
                            shifted_field = _make_lifted_deref_shift(input_field, shift_spec, domain)

                            guard_domain = (
                                _compute_shift_guard_domain(shift_spec, domain)
                                if domain is not None
                                else None
                            )
                            if guard_domain is not None:
                                elem_field = im.concat_where(guard_domain, shifted_field, neutral_field)
                            else:
                                elem_field = shifted_field

                            acc_field = im.as_fieldop(
                                im.lambda_("__a", "__b")(im.call(copy.deepcopy(red_op))(im.deref("__a"), im.deref("__b"))),
                                domain,
                            )(acc_field, elem_field)
                        else:
                            return acc_field

        lst_name = "__cart_reduce_lst"
        lst_ref = im.ref(lst_name)
        lst_val = im.deref(lst_ref)

        acc = copy.deepcopy(widened_init)
        for idx in range(conn_size):
            elem = im.list_get(im.literal(str(idx), "int32"), lst_val)
            acc = im.call(copy.deepcopy(red_op))(acc, elem)

        return im.as_fieldop(im.lambda_(lst_name)(acc), domain)(copy.deepcopy(list_expr))


    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        symbolic_domain_sizes: dict[str, str] | None = None,
    ) -> ir.Node:  # add dict for mapping later
        return cls().visit(node, symbolic_domain_sizes=symbolic_domain_sizes)

    def visit_Program(self, node: ir.Program, **kwargs) -> ir.Program:
        program_param_ids = {str(param.id) for param in node.params}
        new_params = [
            ir.Sym(id=param.id, type=self._cartesian_remapped_type(param.type)) for param in node.params
        ]
        child_kwargs = dict(kwargs)
        child_kwargs["program_param_ids"] = program_param_ids
        new_body = [
            self.visit(stmt, **child_kwargs)
            for stmt in node.body
        ]
        return ir.Program(
            id=node.id,
            function_definitions=node.function_definitions,
            params=new_params,
            declarations=node.declarations,
            body=new_body,
        )

    def visit_SymRef(self, node: ir.SymRef, **kwargs) -> ir.SymRef:
        if isinstance(node.type, it_ts.IteratorType):
            return ir.SymRef(id=node.id, type=None)

        if str(node.id).startswith("__"):
            if node.type is None:
                return node
            return ir.SymRef(id=node.id, type=None)

        mapped_type = self._cartesian_remapped_type(node.type)
        if mapped_type is node.type:
            return node
        return ir.SymRef(id=node.id, type=mapped_type)

    def visit_Sym(self, node: ir.Sym, **kwargs) -> ir.Sym:
        if isinstance(node.type, it_ts.IteratorType):
            return ir.Sym(id=node.id, type=None)

        if str(node.id).startswith("__"):
            if node.type is None:
                return node
            return ir.Sym(id=node.id, type=None)

        mapped_type = self._cartesian_remapped_type(node.type)
        if mapped_type is node.type:
            return node
        return ir.Sym(id=node.id, type=mapped_type)

    def visit_Lambda(self, node: ir.Lambda, **kwargs) -> ir.Lambda:
        new_params = [self.visit(param, **kwargs) for param in node.params]
        new_expr = self.visit(node.expr, **kwargs)
        return ir.Lambda(params=new_params, expr=new_expr, type=None)

    def visit_SetAt(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        new_domain = self.visit(node.domain, **kwargs)
        new_expr = self.visit(node.expr, current_domain=new_domain, **kwargs)
        new_target = self.visit(node.target, **kwargs)
        return ir.SetAt(expr=new_expr, domain=new_domain, target=new_target)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        debug = os.environ.get("GT4PY_DEBUG_CART_UNROLL", "0") == "1"
        current_domain = kwargs.get("current_domain")
        program_param_ids: set[str] = kwargs.get("program_param_ids", set())
        symbolic_domain_sizes: dict[str, str] | None = kwargs.get("symbolic_domain_sizes")

        def _debug(msg: str, value: ir.Expr | None = None) -> None:
            if debug:
                print(f"[CartUnroll] {msg}")
                if value is not None:
                    print(f"[CartUnroll]   {value}")

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
                key = tuple(shift_call.fun.args)
                if key in map_dict and map_dict[key]["kind"] == "concat_where":
                    rewritten_arg = self.visit(node.args[0], **kwargs)
                    out = _build_field_concat_where_from_branches(
                        rewritten_arg, map_dict[key]["branches"], current_domain
                    )
                    _debug("applied lifted concat_where rewrite", out)
                    return out

            if (reduce_inputs := self._extract_generic_reduce_inputs(node)) is not None:
                red_op, red_init, list_expr, conn_size = reduce_inputs
                rewritten_list_expr = self.visit(list_expr, **kwargs)
                out = self._build_generic_unrolled_reduce_expr(
                    red_op,
                    red_init,
                    rewritten_list_expr,
                    conn_size,
                    current_domain,
                )
                _debug("applied generic neighbors reduce rewrite", out)
                return out

        new_node = copy.deepcopy(self.generic_visit(node, **kwargs))

        if cpm.is_call_to(new_node, "can_deref") and len(new_node.args) == 1:
            can_deref_arg = new_node.args[0]
            if isinstance(can_deref_arg, ir.SymRef):
                out = im.literal("True", "bool")
                _debug("rewrote can_deref(symref) to True", out)
                return out

        if cpm.is_applied_shift(new_node):
            key = tuple(new_node.fun.args)
            if key in map_dict:
                entry = map_dict[key]
                arg = new_node.args[0]

                if entry["kind"] == "shift":
                    out = _apply_shift_chain(copy.deepcopy(arg), entry["shifts"])
                    _debug("applied shift rewrite", out)
                    return out

                if entry["kind"] == "concat_where":
                    out = _build_concat_where_from_branches(arg, entry["branches"])
                    _debug("applied concat_where rewrite", out)
                    return out
                    # (cond0, s0), (cond1, s1), (_, s2) = entry["branches"]
                    # b0 = _apply_shift_chain(arg, s0)
                    # b1 = _apply_shift_chain(arg, s1)
                    # b2 = _apply_shift_chain(arg, s2)
                    # return im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
        
        # replace Field[Edge] with Field[IDim, JDim, Kolor]
        # i.e. replace get_domain_range(out, Edgeₕ) by the equivalent in IDim, JDim, Kolor
        _debug("visit_FunCall new_node", new_node)

        def _pick_size_param(*candidates: str) -> ir.Expr | None:
            for candidate in candidates:
                if candidate in program_param_ids:
                    return im.ref(candidate)
                if symbolic_domain_sizes is not None and candidate in symbolic_domain_sizes:
                    symbolic_size = symbolic_domain_sizes[candidate]
                    if isinstance(symbolic_size, int):
                        return ir.OffsetLiteral(value=symbolic_size)
                    if isinstance(symbolic_size, str):
                        try:
                            return ir.OffsetLiteral(value=int(symbolic_size))
                        except ValueError:
                            return ir.OffsetLiteral(value=symbolic_size)
                    return im.ensure_expr(symbolic_size)
            return None

        def _cartesian_axis_bounds(axis_name: str) -> tuple[ir.Expr, ir.Expr] | None:
            if axis_name == "IDim":
                upper = _pick_size_param("max_i", "domain_max_i", "nx")
                return (ir.OffsetLiteral(value=0), upper) if upper is not None else None
            if axis_name == "JDim":
                upper = _pick_size_param("max_j", "domain_max_j", "ny")
                return (ir.OffsetLiteral(value=0), upper) if upper is not None else None
            if axis_name == "Kolor":
                return ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=3)
            return None

        def _entity_kolor_bounds(axis_name: str) -> tuple[ir.Expr, ir.Expr] | None:
            kolor_stops = {"Vertex": 1, "Cell": 2, "Edge": 3}
            if axis_name not in kolor_stops:
                return None
            return ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=kolor_stops[axis_name])

        def _axis_name(axis_expr: ir.Expr) -> str | None:
            if isinstance(axis_expr, ir.AxisLiteral) and isinstance(axis_expr.value, str):
                return axis_expr.value
            if isinstance(axis_expr, common.Dimension):
                return axis_expr.value
            return getattr(axis_expr, "value", None) if isinstance(getattr(axis_expr, "value", None), str) else None

        if cpm.is_call_to(new_node, "get_domain_range") and len(new_node.args) == 2:
            field_expr, axis_expr = new_node.args
            axis_name = _axis_name(axis_expr)
            if axis_name is not None:
                if (bounds := _cartesian_axis_bounds(axis_name)) is not None:
                    out = im.make_tuple(*bounds)
                    _debug("inlined get_domain_range", out)
                    return out

        # if cpm.is_call_to(new_node, "get_domain_range") and new_node.args[1] == ir.OffsetLiteral(value="Edgeₕ"):
        def _extract_field_from_get_domain_range(expr: ir.Expr, expected_axis: str) -> ir.Expr | None:
        # Match tuple_get(k, get_domain_range(field, Axis))
            if not cpm.is_call_to(expr, "tuple_get") or len(expr.args) != 2:
                return None

            gdr = expr.args[1]
            if not cpm.is_call_to(gdr, "get_domain_range") or len(gdr.args) != 2:
                return None

            field_expr, axis_expr = gdr.args
            axis_name = _axis_name(axis_expr)
            if axis_name != expected_axis:
                return None
            return field_expr

        if cpm.is_call_to(new_node, "tuple_get") and len(new_node.args) == 2:
            tuple_index, gdr = new_node.args
            if cpm.is_call_to(gdr, "make_tuple") and isinstance(tuple_index, ir.Literal):
                idx = int(tuple_index.value)
                if 0 <= idx < len(gdr.args):
                    out = copy.deepcopy(gdr.args[idx])
                    _debug("collapsed tuple_get(make_tuple(...))", out)
                    return out
            if cpm.is_call_to(gdr, "get_domain_range") and len(gdr.args) == 2:
                axis_name = _axis_name(gdr.args[1])
                if axis_name is not None and isinstance(tuple_index, ir.OffsetLiteral) and tuple_index.value in {0, 1}:
                    if (bounds := _cartesian_axis_bounds(axis_name)) is not None:
                        out = copy.deepcopy(bounds[tuple_index.value])
                        _debug("inlined tuple_get(get_domain_range(...))", out)
                        return out

        if cpm.is_call_to(new_node, "unstructured_domain") and len(new_node.args) == 1:
            nr = new_node.args[0]
            if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                axis, start, stop = nr.args
                if isinstance(axis, ir.AxisLiteral) and axis.value in {"Edge", "Vertex", "Cell"}:
                    field_from_start = _extract_field_from_get_domain_range(start, axis.value)
                    field_from_stop = _extract_field_from_get_domain_range(stop, axis.value)

                    if field_from_start is not None and field_from_start == field_from_stop:
                        _debug("domain rewrite matched field", field_from_start)
                        out_field = field_from_start

                        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
                        JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
                        Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

                        idim_bounds = _cartesian_axis_bounds("IDim")
                        jdim_bounds = _cartesian_axis_bounds("JDim")
                        if idim_bounds is None or jdim_bounds is None:
                            _debug("domain rewrite skipped: missing IDim/JDim bounds")
                            return new_node

                        def _bounds(dim: common.Dimension) -> tuple[ir.Expr, ir.Expr]:
                            if dim.value in {"IDim", "JDim"}:
                                if dim.value == "IDim":
                                    return idim_bounds
                                return jdim_bounds
                            if dim.value == "Kolor":
                                if (bounds := _entity_kolor_bounds(axis.value)) is not None:
                                    return bounds
                            gdr = im.call("get_domain_range")(out_field, dim)
                            return im.tuple_get(0, gdr), im.tuple_get(1, gdr)

                        i0, i1 = _bounds(IDim)
                        j0, j1 = _bounds(JDim)
                        k0, k1 = _bounds(Kolor)
                        replacement_domain = im.call("cartesian_domain")(
                            im.named_range(IDim, i0, i1),
                            im.named_range(JDim, j0, j1),
                            im.named_range(Kolor, k0, k1),
                        )
                        _debug("domain rewrite replacement", replacement_domain)
                        return replacement_domain
                    _debug("domain rewrite skipped: start/stop fields mismatch")
                else:
                    _debug("domain rewrite skipped: axis is not Edge/Vertex/Cell")
            else:
                _debug("domain rewrite skipped: not a named_range with 3 args")
        else:
            _debug("domain rewrite skipped: not a single-range unstructured_domain")
        return new_node


@dataclasses.dataclass
class RewriteCartesianCanDeref(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        new_node = self.generic_visit(node, **kwargs)

        if cpm.is_call_to(new_node, "can_deref") and len(new_node.args) == 1:
            arg = new_node.args[0]

            if isinstance(arg, ir.SymRef):
                return im.literal("True", "bool")

        return new_node

# dict V2V[0] -> cart, ...