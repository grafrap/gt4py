# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gt4py import eve
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


def _collect_associative_args(node: ir.Expr, op: str) -> list[ir.Expr]:
    if cpm.is_call_to(node, op):
        assert isinstance(node, ir.FunCall)
        args: list[ir.Expr] = []
        for arg in node.args:
            args.extend(_collect_associative_args(arg, op))
        return args
    return [node]


def _contains_in_associative_tree(node: ir.Expr, op: str, candidate: ir.Expr) -> bool:
    return any(arg == candidate for arg in _collect_associative_args(node, op))


def _rebuild_binary(op: str, args: list[ir.Expr]) -> ir.Expr:
    assert len(args) >= 1
    expr = args[0]
    for arg in args[1:]:
        expr = im.call(op)(expr, arg)
    return expr


def _simplify_bound_expr(expr: ir.Expr) -> ir.Expr:
    if not isinstance(expr, ir.FunCall):
        return expr

    simplified_args = [_simplify_bound_expr(arg) for arg in expr.args]
    node = ir.FunCall(fun=expr.fun, args=simplified_args, type=expr.type)

    if cpm.is_call_to(node, ("minimum", "maximum")):
        assert isinstance(node.fun, ir.SymRef)
        op = node.fun.id
        opposite = "maximum" if op == "minimum" else "minimum"
        lhs, rhs = node.args

        if lhs == rhs:
            return lhs

        # Idempotence over associative trees:
        # minimum(minimum(a,b),a) -> minimum(a,b)
        # maximum(maximum(a,b),a) -> maximum(a,b)
        flat = _collect_associative_args(node, op)
        dedup: list[ir.Expr] = []
        for arg in flat:
            if any(existing == arg for existing in dedup):
                continue
            dedup.append(arg)
        if len(dedup) == 1:
            return dedup[0]

        rebuilt = _rebuild_binary(op, dedup)

        # Absorption (safe for integer bound algebra):
        # minimum(maximum(x, y), x) -> x
        # maximum(minimum(x, y), x) -> x
        if cpm.is_call_to(lhs, opposite) and _contains_in_associative_tree(lhs, opposite, rhs):
            return rhs
        if cpm.is_call_to(rhs, opposite) and _contains_in_associative_tree(rhs, opposite, lhs):
            return lhs

        return rebuilt

    return node


class SimplifyDomainBounds(eve.NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)
        if cpm.is_call_to(node, "named_range") and len(node.args) == 3:
            axis, start, stop = node.args
            return ir.FunCall(
                fun=node.fun,
                args=[axis, _simplify_bound_expr(start), _simplify_bound_expr(stop)],
                type=node.type,
            )
        return node
