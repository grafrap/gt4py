# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
from typing import Optional

from gt4py.eve import PreserveLocationVisitor
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.ir_utils.domain_utils import SymbolicDomain
from gt4py.next.iterator.transforms import fixed_point_transformation


def _range_complement(
    range_: domain_utils.SymbolicRange,
) -> tuple[domain_utils.SymbolicRange, domain_utils.SymbolicRange]:
    # `[a, b[` -> `[-inf, a[` ∪ `[b, inf[`  # noqa: RUF003
    assert not any(isinstance(b, itir.InfinityLiteral) for b in [range_.start, range_.stop])
    return (
        domain_utils.SymbolicRange(itir.InfinityLiteral.NEGATIVE, range_.start),
        domain_utils.SymbolicRange(range_.stop, itir.InfinityLiteral.POSITIVE),
    )


class _CanonicalizeDomainArgument(
    PreserveLocationVisitor, fixed_point_transformation.FixedPointTransformation
):
    """
    Transform `concat_where` expressions into their canonical form.

    The canonical form of a `concat_where(domain, tb, fb)` expression is an expression where
    `domain` is a simple domain expression, i.e. no union or intersection, which is unbounded in
    one and only one side, e.g. something like [-inf, 1] or [1, inf], but not [1, 2] or
    d1 | d2.  This choice of a canonical form ensures that the domain inference can infer a
    contiguous domain for `tb` and `fb` as the one-sided domain simply splits the contiguous
    domain on which the entire expression is accessed into two contiguous parts. Or more
    formally expressed:

    The domain `tb` is inferred by intersecting the domain of the entire `concat_where` expression
    with the domain argument. Intersection with a single bounded domain arg preserves the domain
    contiguity. The domain of `fb` is inferred by intersection of the entire domain with the
    complement of the domain argument. The complement of a single sided domain is another single
    sided domain, so then following the same argument as before the domain of `fb` is contiguous.
    To make this more concrete consider the `concat_where` expr is accessed on the domain [a, b]
    and its domain argument is [-inf, c] then the domain of `tb` is inferred to be [a, min(b, c)]
    and the domain of `fb` is [min(b, c), b].

    Description of the transformation:

    If the expression is not simple, but a union or intersection, e.g., [1, 2] | [3, 4], then this
    transformation first expands into a nested `concat_where` of simple domain expressions.
    In our example `concat_where([1, 2] | [3, 4], tb, fv)` is rewritten to
    `concat_where([1, 2], tb, concat_where([3, 4], tb, fb))`.
    If the expression is simple and bounded on both sides e.g. something like
    `concat_where([1, 2], tb, fb)` then the expression is rewritten into a union of simple
    domain expressions which are bounded on one side and unbounded in the other, namely
    `concat_where([-inf, 1] | [2, inf], fb, tb)`. Both transformations are applied until a fixed
    point is reached, ensuring first, a simple domain and second domain bounded on one side, in
    other words the desired canonical form.
    """

    @classmethod
    def apply(cls, node: itir.Node):
        return cls().visit(node)

    def transform(self, node: itir.Node) -> Optional[itir.Node]:  # type: ignore[override] # ignore kwargs for simplicity
        if cpm.is_call_to(node, "concat_where"):
            cond_expr, field_a, field_b = node.args
            # `concat_where(d1 & d2, a, b)` -> concat_where(d1, concat_where(d2, a, b), b)
            if cpm.is_call_to(cond_expr, "and_"):
                conds = cond_expr.args
                if os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1":
                    merged_named_ranges: list[itir.Expr] = []
                    seen_axes: set[str] = set()
                    merged_grid_fun: str | None = None
                    can_merge = True
                    for cond in conds:
                        if not cpm.is_call_to(cond, ("cartesian_domain", "unstructured_domain")):
                            can_merge = False
                            break

                        grid_fun = str(cond.fun.id)
                        if merged_grid_fun is None:
                            merged_grid_fun = grid_fun
                        elif merged_grid_fun != grid_fun:
                            can_merge = False
                            break

                        for named_range in cond.args:
                            if not (
                                cpm.is_call_to(named_range, "named_range")
                                and len(named_range.args) == 3
                            ):
                                can_merge = False
                                break
                            axis_name = getattr(named_range.args[0], "value", None)
                            if not isinstance(axis_name, str) or axis_name in seen_axes:
                                can_merge = False
                                break
                            seen_axes.add(axis_name)
                            merged_named_ranges.append(copy.deepcopy(named_range))
                        if not can_merge:
                            break

                    # Structured edge remap emits conjunctions of axis-specific finite domains.
                    # Merge them directly back into one multi-axis domain to avoid nested
                    # concat_where expansion blow-up.
                    if can_merge and merged_grid_fun and len(merged_named_ranges) >= 2:
                        axis_order = {"IDim": 0, "JDim": 1, "K": 2, "Kolor": 3}
                        merged_named_ranges.sort(
                            key=lambda nr: axis_order.get(getattr(nr.args[0], "value", ""), 100)
                        )
                        merged_domain = im.call(merged_grid_fun)(*merged_named_ranges)
                        return self.fp_transform(im.concat_where(merged_domain, field_a, field_b))
                return im.let(("__cwcda_field_a", field_a), ("__cwcda_field_b", field_b))(
                    self.fp_transform(
                        im.concat_where(
                            conds[0],
                            self.fp_transform(
                                im.concat_where(conds[1], "__cwcda_field_a", "__cwcda_field_b")
                            ),
                            "__cwcda_field_b",
                        )
                    )
                )
            # `concat_where(d1 | d2, a, b)` -> concat_where(d1, a, concat_where(d2, a, b))
            if cpm.is_call_to(cond_expr, "or_"):
                conds = cond_expr.args
                return im.let(("__cwcda_field_a", field_a), ("__cwcda_field_b", field_b))(
                    self.fp_transform(
                        im.concat_where(
                            conds[0],
                            "__cwcda_field_a",
                            self.fp_transform(
                                im.concat_where(conds[1], "__cwcda_field_a", "__cwcda_field_b")
                            ),
                        )
                    )
                )

            # concat_where([1, 2[, a, b) -> concat_where([-inf, 1] | [2, inf[, b, a)
            if cpm.is_call_to(cond_expr, ("cartesian_domain", "unstructured_domain")):
                domain = SymbolicDomain.from_expr(cond_expr)
                if len(domain.ranges) == 1:
                    dim, range_ = next(iter(domain.ranges.items()))
                    if domain_utils.is_finite(range_):
                        complement = _range_complement(range_)
                        new_domains = [
                            im.domain(domain.grid_type, {dim: (cr.start, cr.stop)})
                            for cr in complement
                        ]
                        return self.fp_transform(
                            im.concat_where(im.call("or_")(*new_domains), field_b, field_a)
                        )
                else:
                    # Keep multi-axis domains unchanged. Expanding them into nested boolean
                    # domain expressions can cause severe IR/compile-time growth in structured
                    # remap workloads.
                    return None

        return None


canonicalize_domain_argument = _CanonicalizeDomainArgument.apply
