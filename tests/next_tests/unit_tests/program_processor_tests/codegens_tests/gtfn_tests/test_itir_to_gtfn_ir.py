# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.codegens.gtfn import gtfn_ir, itir_to_gtfn_ir as it2gtfn
from gt4py.next.type_system import type_translation


def test_funcall_to_op():
    testee = itir.FunCall(
        fun=itir.SymRef(id="plus"), args=[itir.SymRef(id="foo"), itir.SymRef(id="bar")]
    )
    expected = gtfn_ir.BinaryExpr(
        op="+", lhs=gtfn_ir.SymRef(id="foo"), rhs=gtfn_ir.SymRef(id="bar")
    )

    actual = it2gtfn.GTFN_lowering(
        grid_type=gtx.GridType.CARTESIAN, offset_provider_type={}, column_axis=None
    ).visit(testee)

    assert expected == actual


def test_unapplied_funcall_to_function_object():
    testee = itir.SymRef(id="plus")
    expected = gtfn_ir.SymRef(id="plus")

    actual = it2gtfn.GTFN_lowering(
        grid_type=gtx.GridType.CARTESIAN, offset_provider_type={}, column_axis=None
    ).visit(testee)

    assert expected == actual


def test_get_domains():
    domain = im.call("cartesian_domain")(im.named_range(itir.AxisLiteral(value="D"), 1, 2))
    testee = itir.Program(
        id="foo",
        function_definitions=[],
        params=[itir.Sym(id="bar")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop("deref")(),
                domain=domain,
                target=itir.SymRef(id="bar"),
            )
        ],
    )

    result = list(it2gtfn._get_domains(testee.body))
    assert result == [domain]


def test_collect_offset_definitions_keeps_cartesian_connectivity_tags():
    vertex = common.Dimension("Vertex")
    edge = common.Dimension("Edge")
    v2e = common.Dimension("V2E", kind=common.DimensionKind.LOCAL)
    connectivity_type = common.NeighborConnectivityType(
        domain=(vertex, v2e),
        codomain=edge,
        skip_value=None,
        dtype=np.dtype(np.int32),
        max_neighbors=6,
    )

    used_connectivity = itir.OffsetLiteral(value="V2E")

    actual = it2gtfn._collect_offset_definitions(
        used_connectivity,
        gtx.GridType.CARTESIAN,
        {"V2E": connectivity_type},
    )

    assert {name for name in actual} >= {"V2E", "Vertex", "Edge"}


def test_collect_local_dimensions_from_parameter_types():
    IDim = common.Dimension("IDim")
    JDim = common.Dimension("JDim")
    Kolor = common.Dimension("Kolor")
    V2E = common.Dimension("V2E", kind=common.DimensionKind.LOCAL)
    sign = gtx.as_field([IDim, JDim, Kolor, V2E], np.zeros((2, 3, 1, 6), dtype=np.float64))

    actual = it2gtfn._collect_local_dimensions_from_types(
        [im.sym("sign", type_translation.from_value(sign))],
        [],
    )

    assert set(actual) == {"V2E"}
