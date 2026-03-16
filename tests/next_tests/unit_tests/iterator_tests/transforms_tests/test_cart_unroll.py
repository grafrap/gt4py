# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next import common
from gt4py.eve import SymbolRef
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms.cart_unroll import CartUnroll, _bounded_shifted_deref, map_dict
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.type_system import type_specifications as ts


def test_normalize_shifts_removes_zero_offsets():
    testee = im.shift("_OffIDim", 0)(im.shift("_OffJDim", 0)("iter"))
    actual = NormalizeShifts().visit(testee)
    assert actual == im.ref("iter")

    testee = im.shift("_OffIDim", 0)(im.shift("_OffJDim", -1)("iter"))
    actual = NormalizeShifts().visit(testee)
    expected = im.shift("_OffJDim", -1)("iter")
    assert actual == expected


def test_normalize_shifts_keeps_zero_offsets_on_connectivity_axes():
    testee = im.shift("E2V", 0)("iter")
    actual = NormalizeShifts().visit(testee)
    assert actual == testee


def test_bounded_shifted_deref_adds_idim_jdim_bounds_guard():
    shifted = _bounded_shifted_deref(
        im.ref("inp"),
        (
            ir.OffsetLiteral(value="IDim"),
            ir.OffsetLiteral(value=-1),
            ir.OffsetLiteral(value="JDim"),
            ir.OffsetLiteral(value=1),
        ),
        im.literal("0.0", "float64"),
    )
    print(shifted)

    assert cpm.is_call_to(shifted, "if_")


def test_V2E():
    # pytest.xfail(
    #     "Not implementeds we don't have an easy way to determine the type of the one literal (type inference is to expensive)."
    # )
    testee = im.shift("V2E", 0)("iter")
    expected = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 1)("iter")
    expected = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 2)("iter")
    expected = im.shift("_OffKolor", 2)(im.shift("_OffJDim", -1)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 3)("iter")
    expected = im.shift("_OffKolor", 0)(im.shift("_OffJDim", -1)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 4)("iter")
    expected = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 5)("iter")
    expected = im.shift("_OffKolor", 2)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

def test_V2C():
    testee = im.shift("V2C", 0)("iter")
    expected = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 1)("iter")
    expected = im.shift("_OffKolor", 1)(im.shift("_OffJDim", -1)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 2)("iter")
    expected = im.shift("_OffKolor", 0)(im.shift("_OffJDim", -1)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 3)("iter")
    expected = im.shift("_OffKolor", 1)(im.shift("_OffJDim", -1)((im.shift("_OffIDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 4)("iter")
    expected = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 5)("iter")
    expected = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected


def test_E2V():
    
    testee = im.shift("E2V", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )
    cond1 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2))},
    )

    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2V", 1)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

def test_E2C():
    testee = im.shift("E2C", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )
    cond1 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2))},
    )

    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C", 1)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", -1)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

def test_C2V():
    testee = im.shift("C2V", 0)("iter")
    kolor = common.Dimension("Kolor")
    
    cond = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )

    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2V", 1)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2V", 2)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

def test_C2E():
    testee = im.shift("C2E", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )

    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E", 1)("iter")
    b0 = im.shift("_OffKolor", 2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E", 2)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

def test_E2C2V():
    testee = im.shift("E2C2V", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )
    cond1 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2))},
    )

    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)

    assert actual == expected

    testee = im.shift("E2C2V", 1)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2V", 2)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2V", 3)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", -1)(im.shift("_OffIDim", 1)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

def test_C2E2C():
    testee = im.shift("C2E2C", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )

    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C", 1)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C", 2)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", -1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

def test_C2E2CO():
    testee = im.shift("C2E2CO", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )

    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2CO", 1)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2CO", 2)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", -1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2CO", 3)("iter")
    expected = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

def test_E2C2E():
    testee = im.shift("E2C2E", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )
    cond1 = im.domain(
        common.GridType.CARTESIAN,
         {kolor: (ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2))},
    )

    b0 = im.shift("_OffKolor", 2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2E", 1)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2E", 2)("iter")
    b0 = im.shift("_OffKolor", 2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", -1)(im.shift("_OffIDim", 1)("iter")))
    b2 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2E", 3)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", -1)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

def test_E2C2EO():
    testee = im.shift("E2C2EO", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )
    cond1 = im.domain(
        common.GridType.CARTESIAN,
         {kolor: (ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2))},
    )

    b0 = im.shift("_OffKolor", 2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2EO", 1)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2EO", 2)("iter")
    b0 = im.shift("_OffKolor", 2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", -1)(im.shift("_OffIDim", 1)("iter")))
    b2 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2EO", 3)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", -1)(im.shift("_OffIDim", 0)("iter")))
    b2 = im.shift("_OffKolor", -2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2EO", 4)("iter")
    expected = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)((im.shift("_OffIDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

def test_C2E2C2E():
    testee = im.shift("C2E2C2E", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )

    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 1)("iter")
    b0 = im.shift("_OffKolor", 2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 2)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 3)("iter")
    b0 = im.shift("_OffKolor", 2)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 4)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 5)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 6)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 7)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim",-1)(im.shift("_OffIDim", 1)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 8)("iter")
    b0 = im.shift("_OffKolor", 2)(im.shift("_OffJDim",-1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

def test_C2E2C2E2C():
    testee = im.shift("C2E2C2E2C", 0)("iter")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )

    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 1)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", -1)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 2)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 3)("iter")
    b0 = im.shift("_OffKolor", 1)(im.shift("_OffJDim",-1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", -1)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 4)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim",-1)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim",-1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 5)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim",-1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 6)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim",-1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 7)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim",-1)(im.shift("_OffIDim", 1)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim",-1)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 8)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim",-1)(im.shift("_OffIDim", 0)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 9)("iter")
    b0 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim",-1)("iter")))
    b1 = im.shift("_OffKolor", 0)(im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected


@pytest.mark.parametrize("axis", ["Edge", "Vertex", "Cell"])
def test_unstructured_domain_is_not_rewritten_to_cartesian_domain_without_bounds(axis):
    out = im.ref("out")
    start = im.tuple_get(0, im.call("get_domain_range")(out, ir.AxisLiteral(value=axis)))
    stop = im.tuple_get(1, im.call("get_domain_range")(out, ir.AxisLiteral(value=axis)))
    testee = im.call("unstructured_domain")(im.call("named_range")(ir.AxisLiteral(value=axis), start, stop))

    actual = CartUnroll.apply(testee)
    assert actual == testee


def test_lifted_applied_shift_is_rewritten_to_lifted_concat_where():
    IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
    JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
    Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

    domain = im.call("cartesian_domain")(
        im.named_range(IDim, 0, 4),
        im.named_range(JDim, 0, 5),
        im.named_range(Kolor, 0, 2),
    )

    testee = im.as_fieldop(
        im.lambda_("it")(im.deref(im.shift("E2V", 0)("it"))),
        domain,
    )("arg")

    cond0 = im.call("cartesian_domain")(im.named_range(Kolor, 0, 1))
    cond1 = im.call("cartesian_domain")(im.named_range(Kolor, 1, 2))

    b0 = im.as_fieldop(
        im.lambda_("__cart_unroll_it")(
            im.deref(
                im.shift("_OffKolor", 0)(
                    im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("__cart_unroll_it"))
                )
            )
        ),
        domain,
    )("arg")
    b1 = im.as_fieldop(
        im.lambda_("__cart_unroll_it")(
            im.deref(
                im.shift("_OffKolor", -1)(
                    im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("__cart_unroll_it"))
                )
            )
        ),
        domain,
    )("arg")
    b2 = im.as_fieldop(
        im.lambda_("__cart_unroll_it")(
            im.deref(
                im.shift("_OffKolor", -2)(
                    im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("__cart_unroll_it"))
                )
            )
        ),
        domain,
    )("arg")
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)

    assert cpm.is_call_to(actual, "concat_where")
    _, branch0, tail = actual.args
    assert cpm.is_call_to(tail, "concat_where")
    _, branch1, branch2 = tail.args

    def _assert_lifted_branch_shift(branch, expected_shift):
        assert cpm.is_applied_as_fieldop(branch)
        assert isinstance(branch.fun, ir.FunCall)
        assert isinstance(branch.fun.args[0], ir.Lambda)
        stencil = branch.fun.args[0]
        assert cpm.is_call_to(stencil.expr, "deref")
        assert len(stencil.expr.args) == 1
        shifted = NormalizeShifts().visit(stencil.expr.args[0])
        assert shifted == NormalizeShifts().visit(expected_shift)

    _assert_lifted_branch_shift(
        branch0,
        im.shift("_OffKolor", 0)(im.shift("_OffJDim", 0)(im.shift("_OffIDim", 0)("__cart_unroll_it"))),
    )
    _assert_lifted_branch_shift(
        branch1,
        im.shift("_OffKolor", -1)(
            im.shift("_OffJDim", 0)(im.shift("_OffIDim", 1)("__cart_unroll_it"))
        ),
    )
    _assert_lifted_branch_shift(
        branch2,
        im.shift("_OffKolor", -2)(
            im.shift("_OffJDim", 1)(im.shift("_OffIDim", 0)("__cart_unroll_it"))
        ),
    )


def test_cartesian_remapped_type_includes_cell_fields():
    Cell = common.Dimension("Cell", kind=common.DimensionKind.HORIZONTAL)
    testee = ts.FieldType(dims=[Cell], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))

    actual = CartUnroll._cartesian_remapped_type(testee)

    assert isinstance(actual, ts.FieldType)
    assert [d.value for d in actual.dims] == ["IDim", "JDim", "Kolor"]


def test_cartesian_remapped_type_preserves_local_dims():
    Vertex = common.Dimension("Vertex", kind=common.DimensionKind.HORIZONTAL)
    V2EDim = common.Dimension("V2E", kind=common.DimensionKind.LOCAL)
    testee = ts.FieldType(dims=[Vertex, V2EDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))

    actual = CartUnroll._cartesian_remapped_type(testee)

    assert isinstance(actual, ts.FieldType)
    assert [d.value for d in actual.dims] == ["IDim", "JDim", "Kolor", "V2E"]


def test_v2e_neighbors_reduce_plus_is_generically_unrolled():
    def _contains_call(node: ir.Node, name: str) -> bool:
        if cpm.is_call_to(node, name):
            return True
        if isinstance(node, ir.FunCall):
            if _contains_call(node.fun, name):
                return True
            return any(_contains_call(arg, name) for arg in node.args)
        if isinstance(node, ir.Lambda):
            return _contains_call(node.expr, name)
        return False

    neighbors_applied = im.as_fieldop(im.lambda_("it")(im.neighbors("V2E", "it")))("zavgS")
    mapped = im.as_fieldop(
        im.lambda_("a", "b")(im.map_("multiplies")(im.deref("a"), im.deref("b")))
    )(neighbors_applied, "sign")
    testee = im.as_fieldop(im.lambda_("lst")(im.reduce("plus", 0)(im.deref("lst"))))(mapped)

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)

    assert not _contains_call(actual, "neighbors")
    assert not _contains_call(actual, "reduce")


def test_neighbors_reduce_rewritten_for_non_v2e_connection_and_maximum():
    def _contains_neighbors(node: ir.Node) -> bool:
        if cpm.is_call_to(node, "neighbors"):
            return True
        if isinstance(node, ir.FunCall):
            if _contains_neighbors(node.fun):
                return True
            return any(_contains_neighbors(arg) for arg in node.args)
        if isinstance(node, ir.Lambda):
            return _contains_neighbors(node.expr)
        return False

    neighbors_applied = im.as_fieldop(im.lambda_("it")(im.neighbors("E2C", "it")))("zavgS")
    mapped = im.as_fieldop(
        im.lambda_("a", "b")(im.map_("plus")(im.deref("a"), im.deref("b")))
    )(neighbors_applied, "sign")
    init = im.literal("0.0", "float64")
    testee = im.as_fieldop(im.lambda_("lst")(im.reduce("maximum", init)(im.deref("lst"))))(mapped)

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)

    assert not _contains_neighbors(actual)


def test_neighbors_reduce_rewritten_for_minimum_without_map_pattern():
    def _contains_call(node: ir.Node, name: str) -> bool:
        if cpm.is_call_to(node, name):
            return True
        if isinstance(node, ir.FunCall):
            if _contains_call(node.fun, name):
                return True
            return any(_contains_call(arg, name) for arg in node.args)
        if isinstance(node, ir.Lambda):
            return _contains_call(node.expr, name)
        return False

    neighbors_applied = im.as_fieldop(im.lambda_("it")(im.neighbors("V2E", "it")))("zavgS")
    init = im.literal("1.0", "float64")
    testee = im.as_fieldop(im.lambda_("lst")(im.reduce("minimum", init)(im.deref("lst"))))(
        neighbors_applied
    )

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)

    assert not _contains_call(actual, "neighbors")
    assert not _contains_call(actual, "reduce")


def test_neighbors_reduce_edge_to_vertex_writes_only_kolor0():
    def _contains_call(node: ir.Node, name: str) -> bool:
        if cpm.is_call_to(node, name):
            return True
        if isinstance(node, ir.FunCall):
            if _contains_call(node.fun, name):
                return True
            return any(_contains_call(arg, name) for arg in node.args)
        if isinstance(node, ir.Lambda):
            return _contains_call(node.expr, name)
        return False

    vertex_axis = ir.AxisLiteral(value="Vertex")
    domain_expr = im.call("unstructured_domain")(
        im.call("named_range")(
            vertex_axis,
            im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), vertex_axis)),
            im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), vertex_axis)),
        )
    )

    neighbors_applied = im.as_fieldop(im.lambda_("it")(im.neighbors("V2E", "it")))("zavgS")
    mapped = im.as_fieldop(
        im.lambda_("a", "b")(im.map_("multiplies")(im.deref("a"), im.deref("b")))
    )(neighbors_applied, "sign")
    reduce_expr = im.as_fieldop(im.lambda_("lst")(im.reduce("plus", 0)(im.deref("lst"))))(mapped)

    testee = ir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("zavgS"), im.sym("sign"), im.sym("out"), im.sym("max_i"), im.sym("max_j")],
        declarations=[],
        body=[ir.SetAt(expr=reduce_expr, domain=domain_expr, target=im.ref("out"))],
    )

    IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
    JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
    Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)

    expected_domain = im.call("cartesian_domain")(
        im.named_range(IDim, ir.OffsetLiteral(value=0), im.ref("max_i")),
        im.named_range(JDim, ir.OffsetLiteral(value=0), im.ref("max_j")),
        im.named_range(Kolor, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1)),
    )

    assert isinstance(actual, ir.Program)
    assert len(actual.body) == 1
    assert isinstance(actual.body[0], ir.SetAt)
    assert actual.body[0].domain == expected_domain
    assert not _contains_call(actual.body[0].expr, "neighbors")
    assert not _contains_call(actual.body[0].expr, "reduce")


@pytest.mark.parametrize(
    ("axis", "kolor_stop"),
    [("Vertex", 1), ("Cell", 2), ("Edge", 3)],
)
def test_unstructured_domain_inlines_nx_ny_and_kolor_bounds_when_available(axis, kolor_stop):
    domain_expr = im.call("unstructured_domain")(
        im.call("named_range")(
            ir.AxisLiteral(value=axis),
            im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), ir.AxisLiteral(value=axis))),
            im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), ir.AxisLiteral(value=axis))),
        )
    )

    testee = ir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out"), im.sym("max_i"), im.sym("max_j")],
        declarations=[],
        body=[ir.SetAt(expr=im.ref("inp"), domain=domain_expr, target=im.ref("out"))],
    )

    IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
    JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
    Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

    expected = ir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out"), im.sym("max_i"), im.sym("max_j")],
        declarations=[],
        body=[
            ir.SetAt(
                expr=im.ref("inp"),
                domain=im.call("cartesian_domain")(
                    im.named_range(IDim, ir.OffsetLiteral(value=0), im.ref("max_i")),
                    im.named_range(JDim, ir.OffsetLiteral(value=0), im.ref("max_j")),
                    im.named_range(
                        Kolor,
                        ir.OffsetLiteral(value=0),
                        ir.OffsetLiteral(value=kolor_stop),
                    ),
                ),
                target=im.ref("out"),
            )
        ],
    )

    actual = CartUnroll.apply(testee)
    assert actual == expected


def test_unstructured_domain_is_not_rewritten_without_max_i_max_j():
    axis = "Edge"
    domain_expr = im.call("unstructured_domain")(
        im.call("named_range")(
            ir.AxisLiteral(value=axis),
            im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), ir.AxisLiteral(value=axis))),
            im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), ir.AxisLiteral(value=axis))),
        )
    )

    testee = ir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[ir.SetAt(expr=im.ref("inp"), domain=domain_expr, target=im.ref("out"))],
    )

    actual = CartUnroll.apply(testee)
    assert actual == testee


def test_tuple_get_get_domain_range_inlines_max_i_max_j():
    IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
    JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
    Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

    domain_expr = im.call("cartesian_domain")(
        im.named_range(
            IDim,
            im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), IDim)),
            im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), IDim)),
        ),
        im.named_range(
            JDim,
            im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), JDim)),
            im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), JDim)),
        ),
        im.named_range(
            Kolor,
            im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), Kolor)),
            im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), Kolor)),
        ),
    )

    testee = ir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out"), im.sym("max_i"), im.sym("max_j")],
        declarations=[],
        body=[ir.SetAt(expr=im.ref("inp"), domain=domain_expr, target=im.ref("out"))],
    )

    expected = ir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out"), im.sym("max_i"), im.sym("max_j")],
        declarations=[],
        body=[
            ir.SetAt(
                expr=im.ref("inp"),
                domain=im.call("cartesian_domain")(
                    im.named_range(IDim, ir.OffsetLiteral(value=0), im.ref("max_i")),
                    im.named_range(JDim, ir.OffsetLiteral(value=0), im.ref("max_j")),
                    im.named_range(Kolor, ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=3)),
                ),
                target=im.ref("out"),
            )
        ],
    )

    actual = CartUnroll.apply(testee)
    assert actual == expected


def test_tuple_get_get_domain_range_inlines_symbolic_domain_sizes():
    IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
    JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)

    domain_expr = im.call("cartesian_domain")(
        im.named_range(
            IDim,
            im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), IDim)),
            im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), IDim)),
        ),
        im.named_range(
            JDim,
            im.tuple_get(0, im.call("get_domain_range")(im.ref("out"), JDim)),
            im.tuple_get(1, im.call("get_domain_range")(im.ref("out"), JDim)),
        ),
    )

    testee = ir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out"), im.sym("size_i"), im.sym("size_j")],
        declarations=[],
        body=[ir.SetAt(expr=im.ref("inp"), domain=domain_expr, target=im.ref("out"))],
    )

    expected = ir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out"), im.sym("size_i"), im.sym("size_j")],
        declarations=[],
        body=[
            ir.SetAt(
                expr=im.ref("inp"),
                domain=im.call("cartesian_domain")(
                    im.named_range(IDim, ir.OffsetLiteral(value=0), im.ref("size_i")),
                    im.named_range(JDim, ir.OffsetLiteral(value=0), im.ref("size_j")),
                ),
                target=im.ref("out"),
            )
        ],
    )

    actual = CartUnroll.apply(
        testee,
        symbolic_domain_sizes={"max_i": "size_i", "max_j": "size_j"},
    )
    assert actual == expected


