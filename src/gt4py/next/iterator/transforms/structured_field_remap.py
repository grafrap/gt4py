# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from gt4py import eve
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils, ir_makers as im
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension("IDim")
JDim = common.Dimension("JDim")
Kolor = common.Dimension("Kolor")
STRUCTURED_DIMS = (IDim, JDim, Kolor)
LOCATION_DIM_NAMES = {"Vertex", "Cell", "Edge"}


def _is_location_dim(dim: common.Dimension) -> bool:
    return dim.kind == common.DimensionKind.HORIZONTAL and dim.value in LOCATION_DIM_NAMES


def _remap_dims(dims: Iterable[common.Dimension]) -> list[common.Dimension]:
    dims_list = list(dims)
    if sum(1 for dim in dims_list if _is_location_dim(dim)) != 1:
        return dims_list

    remapped: list[common.Dimension] = []
    for dim in dims_list:
        if _is_location_dim(dim):
            remapped.extend(STRUCTURED_DIMS)
        else:
            remapped.append(dim)
    return remapped


def _remap_type(type_: ts.TypeSpec) -> ts.TypeSpec:
    match type_:
        case ts.FieldType(dims=dims, dtype=dtype):
            return ts.FieldType(dims=_remap_dims(dims), dtype=dtype)
        case ts.DomainType(dims=dims):
            return ts.DomainType(dims=_remap_dims(dims))
        case ts.TupleType(types=types):
            return ts.TupleType(
                types=[
                    _remap_type(type_) if isinstance(type_, ts.TypeSpec) else type_ for type_ in types
                ]
            )
        case ts.NamedCollectionType(types=types, keys=keys, original_python_type=original_python_type):
            return ts.NamedCollectionType(
                types=[
                    _remap_type(type_) if isinstance(type_, ts.TypeSpec) else type_ for type_ in types
                ],
                keys=keys,
                original_python_type=original_python_type,
            )
        case ts.FunctionType(
            pos_only_args=pos_only_args,
            pos_or_kw_args=pos_or_kw_args,
            kw_only_args=kw_only_args,
            returns=returns,
        ):
            return ts.FunctionType(
                pos_only_args=[
                    _remap_type(type_) if isinstance(type_, ts.TypeSpec) else type_
                    for type_ in pos_only_args
                ],
                pos_or_kw_args={
                    name: _remap_type(type_) if isinstance(type_, ts.TypeSpec) else type_
                    for name, type_ in pos_or_kw_args.items()
                },
                kw_only_args={
                    name: _remap_type(type_) if isinstance(type_, ts.TypeSpec) else type_
                    for name, type_ in kw_only_args.items()
                },
                returns=_remap_type(returns) if isinstance(returns, ts.TypeSpec) else returns,
            )
        case ts.ConstructorType(definition=definition):
            remapped_definition = _remap_type(definition)
            assert isinstance(remapped_definition, ts.FunctionType)
            return ts.ConstructorType(definition=remapped_definition)
        case it_ts.IteratorType(position_dims=position_dims, defined_dims=defined_dims, element_type=element_type):
            remapped_position_dims = (
                position_dims if position_dims == "unknown" else _remap_dims(position_dims)
            )
            remapped_element_type = _remap_type(element_type)
            assert isinstance(remapped_element_type, ts.DataType)
            return it_ts.IteratorType(
                position_dims=remapped_position_dims,
                defined_dims=_remap_dims(defined_dims),
                element_type=remapped_element_type,
            )
        case it_ts.ProgramType(params=params):
            return it_ts.ProgramType(
                params={
                    name: _remap_type(type_) if isinstance(type_, ts.DataType) else type_
                    for name, type_ in params.items()
                }
            )
        case _:
            return type_


def _extract_axis_literal(node: itir.Expr) -> Optional[itir.AxisLiteral]:
    if isinstance(node, itir.AxisLiteral):
        return node
    return None


def _extract_get_domain_range_field(
    bound: itir.Expr, *, axis_name: str, tuple_index: int
) -> Optional[itir.Expr]:
    if not cpm.is_call_to(bound, "tuple_get"):
        return None
    index_arg, range_arg = bound.args
    if not (
        isinstance(index_arg, itir.Literal)
        and index_arg.value == str(tuple_index)
        and cpm.is_call_to(range_arg, "get_domain_range")
    ):
        return None

    field_arg, axis_arg = range_arg.args
    axis_literal = _extract_axis_literal(axis_arg)
    if axis_literal is None or axis_literal.value != axis_name:
        return None
    return field_arg


def _extract_domain_field_from_named_range(node: itir.FunCall) -> Optional[itir.Expr]:
    assert cpm.is_call_to(node, "named_range")
    axis, start, stop = node.args
    axis_literal = _extract_axis_literal(axis)
    if axis_literal is None or axis_literal.value not in LOCATION_DIM_NAMES:
        return None

    start_field = _extract_get_domain_range_field(start, axis_name=axis_literal.value, tuple_index=0)
    stop_field = _extract_get_domain_range_field(stop, axis_name=axis_literal.value, tuple_index=1)
    if start_field is None or stop_field is None or start_field != stop_field:
        return None
    return start_field


def _make_named_range_from_field_domain(field: itir.Expr, dim: common.Dimension) -> itir.FunCall:
    field_domain_range = im.call("get_domain_range")(field, dim)
    return im.named_range(dim, im.tuple_get(0, field_domain_range), im.tuple_get(1, field_domain_range))


def _remap_symbolic_domain(
    remapper: StructuredFieldRemap,
    domain: domain_utils.SymbolicDomain | tuple[Any, ...],
) -> domain_utils.SymbolicDomain | tuple[Any, ...]:
    if isinstance(domain, tuple):
        return tuple(_remap_symbolic_domain(remapper, item) for item in domain)

    remapped_expr = remapper.visit(domain.as_expr())
    assert cpm.is_call_to(remapped_expr, ("cartesian_domain", "unstructured_domain"))
    return domain_utils.SymbolicDomain.from_expr(remapped_expr)


class StructuredFieldRemap(eve.PreserveLocationVisitor, eve.NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls, node: itir.Node) -> itir.Node:
        return cls().visit(node)

    def visit(self, node: eve.concepts.RootNode, **kwargs: Any) -> Any:
        new_node = super().visit(node, **kwargs)

        if isinstance(new_node, itir.Node) and new_node.type is not None:
            object.__setattr__(new_node, "type", _remap_type(new_node.type))

        if isinstance(new_node, itir.Node) and getattr(new_node, "__node_annex__", None):
            if hasattr(new_node.annex, "domain"):
                new_node.annex.domain = _remap_symbolic_domain(self, new_node.annex.domain)

        return new_node

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        new_node = self.generic_visit(node)

        if not cpm.is_call_to(new_node, ("unstructured_domain", "cartesian_domain")):
            return new_node

        rewritten_args: list[itir.Expr] = []
        rewritten_dims: list[common.Dimension] = []
        changed = False

        for arg in new_node.args:
            if not cpm.is_call_to(arg, "named_range"):
                return new_node

            axis = _extract_axis_literal(arg.args[0])
            if axis is None:
                return new_node

            if axis.value in LOCATION_DIM_NAMES:
                field_arg = _extract_domain_field_from_named_range(arg)
                if field_arg is None:
                    return new_node
                rewritten_args.extend(
                    _make_named_range_from_field_domain(field_arg, dim) for dim in STRUCTURED_DIMS
                )
                rewritten_dims.extend(STRUCTURED_DIMS)
                changed = True
            else:
                rewritten_args.append(arg)
                rewritten_dims.append(common.Dimension(axis.value, axis.kind))

        if not changed:
            return new_node

        rewritten_domain = im.call("cartesian_domain")(*rewritten_args)
        rewritten_domain.type = ts.DomainType(dims=rewritten_dims)
        return rewritten_domain