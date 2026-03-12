# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils, ir_makers as im
from gt4py.next.iterator.transforms.structured_field_remap import StructuredFieldRemap
from gt4py.next.iterator.type_system import type_specifications as it_ts, type_synthesizer
from gt4py.next.type_system import type_specifications as ts


FLOAT64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
Vertex = common.Dimension("Vertex")
Edge = common.Dimension("Edge")
KDim = common.Dimension("KDim", kind=common.DimensionKind.VERTICAL)
IDim = common.Dimension("IDim")
JDim = common.Dimension("JDim")
Kolor = common.Dimension("Kolor")


def test_remaps_location_field_types_and_domains_to_structured_dims():
    vertex_type = ts.FieldType(dims=[Vertex, KDim], dtype=FLOAT64)
    edge_type = ts.FieldType(dims=[Edge, KDim], dtype=FLOAT64)

    out_ref = im.ref("out", edge_type)
    domain = im.get_field_domain(common.GridType.UNSTRUCTURED, out_ref, dims=[Edge, KDim])
    program = itir.Program(
        id="structured_remap_test",
        function_definitions=[],
        params=[im.sym("pp", vertex_type), im.sym("out", edge_type)],
        declarations=[],
        body=[itir.SetAt(expr=im.ref("pp", vertex_type), domain=domain, target=out_ref)],
    )

    remapped = StructuredFieldRemap.apply(program)
    assert isinstance(remapped, itir.Program)

    assert remapped.params[0].type == ts.FieldType(dims=[IDim, JDim, Kolor, KDim], dtype=FLOAT64)
    assert remapped.params[1].type == ts.FieldType(dims=[IDim, JDim, Kolor, KDim], dtype=FLOAT64)

    remapped_domain = remapped.body[0].domain
    assert cpm.is_call_to(remapped_domain, "cartesian_domain")
    assert remapped_domain.type == ts.DomainType(dims=[IDim, JDim, Kolor, KDim])
    assert [arg.args[0].value for arg in remapped_domain.args] == ["IDim", "JDim", "Kolor", "KDim"]


def test_remaps_cached_symbolic_domains_alongside_ir_domains():
    vertex_type = ts.FieldType(dims=[Vertex], dtype=FLOAT64)
    edge_type = ts.FieldType(dims=[Edge], dtype=FLOAT64)

    domain = im.get_field_domain(common.GridType.UNSTRUCTURED, im.ref("out", edge_type), dims=[Edge])
    expr = im.as_fieldop("deref", domain)(im.ref("pp", vertex_type))

    assert hasattr(expr.annex, "domain")
    original_domain = expr.annex.domain
    assert isinstance(original_domain, domain_utils.SymbolicDomain)
    assert list(original_domain.ranges) == [Edge]

    remapped_expr = StructuredFieldRemap.apply(expr)
    assert isinstance(remapped_expr, itir.FunCall)
    assert hasattr(remapped_expr.annex, "domain")
    remapped_domain = remapped_expr.annex.domain
    assert isinstance(remapped_domain, domain_utils.SymbolicDomain)
    assert list(remapped_domain.ranges) == [IDim, JDim, Kolor]
    assert cpm.is_call_to(remapped_expr.fun.args[1], "cartesian_domain")


def test_concat_where_accepts_iterator_branches_on_structured_path():
    iterator_type = it_ts.IteratorType(
        position_dims=[IDim, JDim, Kolor],
        defined_dims=[IDim, JDim, Kolor],
        element_type=FLOAT64,
    )

    result = type_synthesizer.concat_where(
        ts.DomainType(dims=[Kolor]), iterator_type, iterator_type
    )

    assert result == iterator_type


def test_symbolic_domain_translate_accepts_explicit_cartesian_axes_without_offset_provider():
    domain = domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.CARTESIAN, {IDim: (0, 2), JDim: (1, 3)})
    )

    translated = domain.translate(
        (itir.OffsetLiteral(value="IDim"), itir.OffsetLiteral(value=1)),
        offset_provider={},
    )

    assert list(translated.ranges) == [IDim, JDim]