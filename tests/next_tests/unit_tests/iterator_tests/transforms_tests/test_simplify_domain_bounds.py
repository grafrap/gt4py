# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms.simplify_domain_bounds import SimplifyDomainBounds
from gt4py.next.type_system import type_specifications as ts


def _nr(start, stop):
    return im.named_range(common.Dimension("K", kind=common.DimensionKind.VERTICAL), start, stop)


def _program_with_expr(expr):
    kdim = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
    dtype = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    ftype = ts.FieldType(dims=[kdim], dtype=dtype)
    return itir.Program(
        id="f",
        function_definitions=[],
        params=[
            im.sym("out", ftype),
            im.sym("vertical_start", ts.ScalarType(kind=ts.ScalarKind.INT32)),
            im.sym("vertical_end", ts.ScalarType(kind=ts.ScalarKind.INT32)),
            im.sym("a", ts.ScalarType(kind=ts.ScalarKind.INT32)),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=expr,
                domain=im.call("cartesian_domain")(_nr(im.ref("vertical_start"), im.ref("vertical_end"))),
                target=im.ref("out"),
            )
        ],
    )


def test_simplify_domain_bound_absorption_pattern():
    vertical_start = im.ref("vertical_start")
    vertical_end = im.ref("vertical_end")

    # minimum(maximum(maximum(minimum(vertical_end, vertical_start), vertical_end), vertical_end), vertical_end)
    stop_expr = im.minimum(
        im.maximum(
            im.maximum(im.minimum(vertical_end, vertical_start), vertical_end),
            vertical_end,
        ),
        vertical_end,
    )

    program = _program_with_expr(im.ref("out"))
    set_at = program.body[0]
    assert isinstance(set_at, itir.SetAt)
    program = itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=program.declarations,
        body=[
            itir.SetAt(
                expr=set_at.expr,
                target=set_at.target,
                domain=im.call("cartesian_domain")(_nr(vertical_start, stop_expr)),
            )
        ],
    )

    actual = SimplifyDomainBounds.apply(program)

    named_range = next(
        node
        for node in actual.pre_walk_values().if_isinstance(itir.FunCall).to_list()
        if cpm.is_call_to(node, "named_range")
    )
    assert named_range.args[2] == vertical_end


def test_simplify_domain_bound_does_not_touch_non_domain_math():
    expr = im.maximum(im.minimum(im.ref("a"), 1), im.ref("a"))
    program = _program_with_expr(expr)

    actual = SimplifyDomainBounds.apply(program)

    set_at = next(node for node in actual.pre_walk_values().if_isinstance(itir.SetAt).to_list())
    assert set_at.expr == expr
