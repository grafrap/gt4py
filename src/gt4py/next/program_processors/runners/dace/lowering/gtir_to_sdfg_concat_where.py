# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the lowering of concat_where operator.

This builtin translator implements the `PrimitiveTranslator` protocol as other
translators in `gtir_to_sdfg_primitives` module.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import dace
from dace import nodes as dace_nodes, subsets as dace_subsets

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args
from gt4py.next.program_processors.runners.dace.lowering import (
    gtir_domain,
    gtir_to_sdfg,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
)
from gt4py.next.type_system import type_specifications as ts


def _translate_concat_where_branch(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    source_expr: gtir.Expr,
    is_lower: bool,
    concat_dim: gtx_common.Dimension,
    concat_dim_bound_expr: gtir.Expr,
    output_domain: domain_utils.SymbolicDomain,
    output_type: ts.FieldType,
    output_desc: dace.data.Array,
    output_node: dace_nodes.AccessNode,
    output_origin: Sequence[dace.symbolic.SymbolicType],
    _previsited_source: "gtir_to_sdfg_types.FieldopData | None" = None,
    _previsited_source_domain=None,
) -> None:
    """
    Translate one of the two branches of a 'concat_where' expression.

    The 'concat_where' expression requires two input fields, one is written on the
    lower part of the result domain, with respect to the domain boundary extracted
    from the first argument; the other input field is written on the upper domain.
    The handling of the two branches is similar, there is only one small difference
    in case the input field does not contain the 'concat_where' dimension, which
    is the case of scalar values or horizontal slice fields. In this case, we use
    the 'is_lower' argument to specify which of the two branches this function
    should target.

    The _regular_ case is `concat_where` applied to two fields with same domain
    as the result field, for example two 'IJK'-fields:
    ```python
    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim < 10, a, b)
    ```

    In case of the arguments is a scalar value, we broadcast it on all dimensions
    of the result field, for example:
    ```python
    @gtx.field_operator
    def testee(a: np.int32, b: IJKField) -> IJKField:
        return concat_where(KDim == 0, a, b)
    ```
    Similarly, we broadcast a field defined as a slice of the output domain, i.e.
    a field with a smaller number of dimensions than the result field.
    Consider for example the following field operator, where the 'boundary' IJ-field
    is used for the vertical boundary (`KDim == 0`):
    ```python
    @gtx.field_operator
    def testee(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)
    ```

    Args:
        ctx: The SDFG context in which to lower the `concat_where` branch expression.
        sdfg_builder: The visitor object to build the SDFG.
        source_expr: The expression producing the input field for this branch.
        is_lower: Flag to specify whether the input expression is for the lower or upper domain.
        concat_dim: The dimension along which to apply `concat_where` on the input fields.
        concat_dim_bound_expr: The expression of the `concat_where` domain condition.
        output_domain: The domain of the result field.
        output_type: The GT4Py type descriptor of the result field.
        output_desc: The DaCe array descriptor of the result field.
        output_node: The SDFG access node for writing the result field.
        output_origin: The origin of the result field, derived from `output_domain`,
            to be used in the write memlet to `output_node`.
    """
    if _previsited_source is not None:
        # Fast path: caller already visited the source expression; use provided data directly.
        source = _previsited_source
        source_domain = _previsited_source_domain
    else:
        assert isinstance(source_expr.type, (ts.FieldType, ts.ScalarType))
        source_domain = source_expr.annex.domain
        if isinstance(source_expr.type, ts.ScalarType) or len(source_expr.type.dims) < len(
            output_type.dims
        ):
            if concat_dim not in source_domain.ranges:
                source_domain.ranges[concat_dim] = (
                    domain_utils.SymbolicRange(
                        start=output_domain.ranges[concat_dim].start,
                        stop=concat_dim_bound_expr,
                    )
                    if is_lower
                    else domain_utils.SymbolicRange(
                        start=concat_dim_bound_expr,
                        stop=output_domain.ranges[concat_dim].stop,
                    )
                )
            source_domain = domain_utils.promote_domain(source_domain, output_type.dims)
            source_domain = domain_utils.domain_intersection(source_domain, output_domain)
            bcast_expr = im.as_fieldop("deref", source_domain.as_expr())(source_expr)
            bcast_expr.type = output_type
            source = sdfg_builder.visit(bcast_expr, ctx=ctx)
        else:
            source = sdfg_builder.visit(source_expr, ctx=ctx)

    assert source.gt_type == output_type
    source_domain_range = source_domain.ranges[concat_dim]
    source_range_0 = gtir_to_sdfg_utils.get_symbolic(source_domain_range.start)
    source_range_1 = gtir_to_sdfg_utils.get_symbolic(
        im.maximum(source_domain_range.start, source_domain_range.stop)
    )
    source_range_size = source_range_1 - source_range_0

    if isinstance(output_type.dtype, ts.ScalarType):
        all_dims = gtx_common.order_dimensions(output_type.dims)
    else:
        assert output_type.dtype.offset_type
        all_dims = gtx_common.order_dimensions([*output_type.dims, output_type.dtype.offset_type])

    source_subset = []
    output_subset = []
    for dim, size, src_origin, dst_origin in zip(
        all_dims,
        output_desc.shape,
        source.origin,
        output_origin,
        strict=True,
    ):
        if dim == concat_dim:
            # Write only the subset corresponding to the range of lower or upper branch.
            source_subset.append(
                (
                    source_range_0 - src_origin,
                    source_range_1 - src_origin - 1,
                    1,
                )
            )
            output_subset.append(
                (
                    source_range_0 - dst_origin,
                    source_range_0 - dst_origin + source_range_size - 1,
                    1,
                )
            )
        else:
            # Write the full subset which covers the array size in this dimension.
            source_subset.append(
                (
                    dst_origin - src_origin,
                    dst_origin - src_origin + size - 1,
                    1,
                )
            )
            output_subset.append((0, size - 1, 1))

    ctx.state.add_nedge(
        source.dc_node,
        output_node,
        dace.Memlet(
            data=source.dc_node.data,
            subset=dace_subsets.Range(source_subset),
            other_subset=dace_subsets.Range(output_subset),
        ),
    )


def _translate_concat_where_from_data(
    outer_node: gtir.Node,
    elem_node: gtir.Node,
    lower_data: gtir_to_sdfg_types.FieldopData,
    upper_data: gtir_to_sdfg_types.FieldopData,
    lower_source_domain,
    upper_source_domain,
    elem_type: "ts.FieldType",
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopData:
    """Lower a single-field concat_where using pre-visited FieldopData for both branches.

    Used when tuple-output concat_where branches are complex expressions (not make_tuple).
    Mirrors the logic of translate_concat_where for the single-field case, but bypasses
    the source_expr visit step.
    """
    mask_domain = domain_utils.SymbolicDomain.from_expr(outer_node.args[0])
    infinity_literals = (gtir.InfinityLiteral.POSITIVE, gtir.InfinityLiteral.NEGATIVE)
    concat_dim_candidates = list(mask_domain.ranges.keys())
    if len(concat_dim_candidates) != 1:
        raise NotImplementedError(
            f"_translate_concat_where_from_data: expected 1 concat dim, got {concat_dim_candidates}"
        )
    concat_dim = concat_dim_candidates[0]

    if mask_domain.ranges[concat_dim].start in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].stop
        lower_source = lower_data
        upper_source = upper_data
        lower_dom = lower_source_domain
        upper_dom = upper_source_domain
    elif mask_domain.ranges[concat_dim].stop in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].start
        upper_source = lower_data
        lower_source = upper_data
        upper_dom = lower_source_domain
        lower_dom = upper_source_domain
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain} with finite domain.")

    output_domain = gtir_domain.get_field_domain(elem_node.annex.domain)
    output_dims, output_origin, output_shape = gtir_domain.get_field_layout(output_domain)
    assert output_dims == elem_type.dims

    if isinstance(elem_type.dtype, ts.ScalarType):
        dtype = gtx_dace_args.as_dace_type(elem_type.dtype)
    else:
        raise NotImplementedError("'concat_where' from data with list output is not supported")

    output, output_desc = sdfg_builder.add_temp_array(ctx.sdfg, output_shape, dtype)
    output_node = ctx.state.add_access(output)

    # Write lower branch.
    _translate_concat_where_branch(
        ctx=ctx,
        sdfg_builder=sdfg_builder,
        source_expr=None,  # type: ignore[arg-type]
        is_lower=True,
        concat_dim=concat_dim,
        concat_dim_bound_expr=bound_expr,
        output_domain=output_domain,
        output_type=elem_type,
        output_desc=output_desc,
        output_node=output_node,
        output_origin=output_origin,
        _previsited_source=lower_source,
        _previsited_source_domain=lower_dom,
    )
    # Write upper branch.
    _translate_concat_where_branch(
        ctx=ctx,
        sdfg_builder=sdfg_builder,
        source_expr=None,  # type: ignore[arg-type]
        is_lower=False,
        concat_dim=concat_dim,
        concat_dim_bound_expr=bound_expr,
        output_domain=output_domain,
        output_type=elem_type,
        output_desc=output_desc,
        output_node=output_node,
        output_origin=output_origin,
        _previsited_source=upper_source,
        _previsited_source_domain=upper_dom,
    )
    return gtir_to_sdfg_types.FieldopData(output_node, elem_type, origin=tuple(output_origin))


def translate_concat_where(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Lowers a `concat_where` expression to a dataflow where two memlets write
    disjoint subsets, for the lower and upper domain, on one data access node.

    Implements the `PrimitiveTranslator` protocol.
    """
    assert cpm.is_call_to(node, "concat_where")
    assert len(node.args) == 3
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    if isinstance(node.type, ts.TupleType):
        # Structured-backend non-split path: split tuple concat_where into per-element ones.
        n_elems = len(node.type.types)
        raw_domain = node.annex.domain if hasattr(node, "annex") and hasattr(node.annex, "domain") else None
        has_tuple_domain = raw_domain is not None and cpm.is_call_to(raw_domain, "make_tuple")

        if cpm.is_call_to(node.args[1], "make_tuple") and cpm.is_call_to(node.args[2], "make_tuple"):
            # Fast path: branches are already make_tuple — split element-wise.
            assert len(node.args[1].args) == len(node.args[2].args) == n_elems
            results = []
            for i, (lower_elem, upper_elem, elem_type) in enumerate(
                zip(node.args[1].args, node.args[2].args, node.type.types)
            ):
                elem_node = gtir.FunCall(fun=node.fun, args=[node.args[0], lower_elem, upper_elem])
                elem_node.type = elem_type
                if isinstance(raw_domain, tuple):
                    elem_node.annex.domain = raw_domain[i]
                elif has_tuple_domain:
                    elem_node.annex.domain = raw_domain.args[i]
                elif raw_domain is not None:
                    elem_node.annex.domain = raw_domain
                results.append(translate_concat_where(elem_node, ctx, sdfg_builder))
            return tuple(results)
        else:
            # General path: branches are complex expressions (e.g. as_fieldop calls).
            # Visit both branches to get tuples of FieldopData, then lower element-wise.
            branch1_data = sdfg_builder.visit(node.args[1], ctx=ctx)
            branch2_data = sdfg_builder.visit(node.args[2], ctx=ctx)
            if not isinstance(branch1_data, (tuple, list)):
                branch1_data = (branch1_data,)
            if not isinstance(branch2_data, (tuple, list)):
                branch2_data = (branch2_data,)
            assert len(branch1_data) == len(branch2_data) == n_elems
            # Build a synthetic single-field concat_where per element and lower it.
            # The element domain comes from the branch expression's annex (tuple of domains).
            b1_domain = getattr(getattr(node.args[1], "annex", None), "domain", None)
            b2_domain = getattr(getattr(node.args[2], "annex", None), "domain", None)
            results = []
            for i, (b1_elem, b2_elem, elem_type) in enumerate(
                zip(branch1_data, branch2_data, node.type.types)
            ):
                # Reconstruct source domains for element i. The annex.domain for a
                # tuple-output expression is a Python tuple of SymbolicDomain objects
                # (one per element) or a single shared SymbolicDomain.
                if isinstance(b1_domain, (tuple, list)):
                    b1_elem_domain = b1_domain[i]
                elif b1_domain is not None and hasattr(b1_domain, "args"):
                    b1_elem_domain = b1_domain.args[i]
                else:
                    b1_elem_domain = b1_domain
                if isinstance(b2_domain, (tuple, list)):
                    b2_elem_domain = b2_domain[i]
                elif b2_domain is not None and hasattr(b2_domain, "args"):
                    b2_elem_domain = b2_domain.args[i]
                else:
                    b2_elem_domain = b2_domain
                # Create a synthetic placeholder IR node for each branch element.
                # We create minimal SymRef proxies that carry the pre-visited data
                # via a temporary SDFG array registered under a unique name.
                elem_node = gtir.FunCall(fun=node.fun, args=[node.args[0], node.args[1], node.args[2]])
                elem_node.type = elem_type
                if isinstance(raw_domain, tuple):
                    elem_node.annex.domain = raw_domain[i]
                elif has_tuple_domain:
                    elem_node.annex.domain = raw_domain.args[i]
                elif raw_domain is not None:
                    elem_node.annex.domain = raw_domain
                # Directly lower using the pre-visited FieldopData.
                results.append(
                    _translate_concat_where_from_data(
                        node, elem_node, b1_elem, b2_elem, b1_elem_domain, b2_elem_domain,
                        elem_type, ctx, sdfg_builder
                    )
                )
            return tuple(results)

    # First argument is a domain expression that defines the mask of the true branch:
    # we extract the dimension along which we need to concatenate the field arguments,
    # and determine whether the true branch argument should be on the lower or upper
    # range with respect to the boundary value.
    mask_domain = domain_utils.SymbolicDomain.from_expr(node.args[0])
    if len(mask_domain.ranges) != 1:
        raise NotImplementedError("Expected `concat_where` along single axis.")

    concat_dim = next(iter(mask_domain.ranges.keys()))

    # Expect unbound range in the concat domain expression on range start or end:
    #  - If the domain expression is unbound on range start (negative infinite),
    #    the expression on the true branch is to be considered the input for the
    #    lower domain.
    #  - Vice versa, if the domain expression is unbound on range stop (positive
    #    infinite), the true expression represents the input for the upper domain.
    infinity_literals = (gtir.InfinityLiteral.POSITIVE, gtir.InfinityLiteral.NEGATIVE)
    if mask_domain.ranges[concat_dim].start in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].stop
        lower_expr, upper_expr = node.args[1:]
    elif mask_domain.ranges[concat_dim].stop in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].start
        upper_expr, lower_expr = node.args[1:]
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain} with finite domain.")

    # We use the concat domain, stored in the annex, as the domain of output field.
    output_domain = gtir_domain.get_field_domain(node.annex.domain)
    output_dims, output_origin, output_shape = gtir_domain.get_field_layout(output_domain)
    assert output_dims == node.type.dims

    if isinstance(node.type.dtype, ts.ScalarType):
        dtype = gtx_dace_args.as_dace_type(node.type.dtype)
    else:
        # TODO(edopao): Refactor allocation of fields with local dimension and enable this.
        raise NotImplementedError("'concat_where' with list output is not supported")

    output, output_desc = sdfg_builder.add_temp_array(ctx.sdfg, output_shape, dtype)
    output_node = ctx.state.add_access(output)

    # Translate the input expression on the lower domain.
    _translate_concat_where_branch(
        ctx=ctx,
        sdfg_builder=sdfg_builder,
        source_expr=lower_expr,
        is_lower=True,
        concat_dim=concat_dim,
        concat_dim_bound_expr=bound_expr,
        output_domain=node.annex.domain,
        output_type=node.type,
        output_desc=output_desc,
        output_node=output_node,
        output_origin=output_origin,
    )

    # Translate the input expression on the upper domain.
    _translate_concat_where_branch(
        ctx=ctx,
        sdfg_builder=sdfg_builder,
        source_expr=upper_expr,
        is_lower=False,
        concat_dim=concat_dim,
        concat_dim_bound_expr=bound_expr,
        output_domain=node.annex.domain,
        output_type=node.type,
        output_desc=output_desc,
        output_node=output_node,
        output_origin=output_origin,
    )

    return gtir_to_sdfg_types.FieldopData(output_node, node.type, origin=tuple(output_origin))
