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
from gt4py.next.iterator.transforms.cart_unroll import CartUnroll
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts


def test_V2E():
    # pytest.xfail(
    #     "Not implementeds we don't have an easy way to determine the type of the one literal (type inference is to expensive)."
    # )
    testee = im.shift("V2E", 0)("iter")
    expected = im.shift("Kolor", 0)(im.shift("JDim", 0)((im.shift("IDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 1)("iter")
    expected = im.shift("Kolor", 1)(im.shift("JDim", 0)((im.shift("IDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 2)("iter")
    expected = im.shift("Kolor", 2)(im.shift("JDim", -1)((im.shift("IDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 3)("iter")
    expected = im.shift("Kolor", 0)(im.shift("JDim", -1)((im.shift("IDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 4)("iter")
    expected = im.shift("Kolor", 1)(im.shift("JDim", 0)((im.shift("IDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 5)("iter")
    expected = im.shift("Kolor", 2)(im.shift("JDim", 0)((im.shift("IDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

def test_V2C():
    testee = im.shift("V2C", 0)("iter")
    expected = im.shift("Kolor", 0)(im.shift("JDim", 0)((im.shift("IDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 1)("iter")
    expected = im.shift("Kolor", 1)(im.shift("JDim", -1)((im.shift("IDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 2)("iter")
    expected = im.shift("Kolor", 0)(im.shift("JDim", -1)((im.shift("IDim", 0)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 3)("iter")
    expected = im.shift("Kolor", 1)(im.shift("JDim", -1)((im.shift("IDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 4)("iter")
    expected = im.shift("Kolor", 0)(im.shift("JDim", 0)((im.shift("IDim", -1)("iter"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 5)("iter")
    expected = im.shift("Kolor", 1)(im.shift("JDim", 0)((im.shift("IDim", -1)("iter"))))
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

    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)

    assert actual == expected

    testee = im.shift("E2V", 1)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

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

    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C", 1)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", -1)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

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

    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2V", 1)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2V", 2)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

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

    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E", 1)("iter")
    b0 = im.shift("Kolor", 2)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E", 2)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

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

    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)

    assert actual == expected

    testee = im.shift("E2C2V", 1)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2V", 2)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2V", 3)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", -1)(im.shift("IDim", 1)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 1)(im.shift("IDim", 1)("iter")))

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

    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C", 1)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C", 2)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", -1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

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

    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2CO", 1)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2CO", 2)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", -1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2CO", 3)("iter")
    expected = im.shift("Kolor", 0)(im.shift("JDim", 0)((im.shift("IDim", 0)("iter"))))
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

    b0 = im.shift("Kolor", 2)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2E", 1)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2E", 2)("iter")
    b0 = im.shift("Kolor", 2)(im.shift("JDim", 0)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", -1)(im.shift("IDim", 1)("iter")))
    b2 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2E", 3)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 1)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", 1)(im.shift("JDim", -1)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

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

    b0 = im.shift("Kolor", 2)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2EO", 1)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2EO", 2)("iter")
    b0 = im.shift("Kolor", 2)(im.shift("JDim", 0)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", -1)(im.shift("IDim", 1)("iter")))
    b2 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2EO", 3)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 1)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", 1)(im.shift("JDim", -1)(im.shift("IDim", 0)("iter")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("E2C2EO", 4)("iter")
    expected = im.shift("Kolor", 0)(im.shift("JDim", 0)((im.shift("IDim", 0)("iter"))))
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

    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 1)("iter")
    b0 = im.shift("Kolor", 2)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 2)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 3)("iter")
    b0 = im.shift("Kolor", 2)(im.shift("JDim", 0)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", 1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 4)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 1)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 5)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 6)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 7)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim",-1)(im.shift("IDim", 1)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E", 8)("iter")
    b0 = im.shift("Kolor", 2)(im.shift("JDim",-1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

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

    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 1)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", -1)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 2)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 3)("iter")
    b0 = im.shift("Kolor", 1)(im.shift("JDim",-1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 4)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim",-1)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim",-1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 5)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim",-1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 6)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim",-1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 7)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim",-1)(im.shift("IDim", 1)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim",-1)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 8)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim",-1)(im.shift("IDim", 0)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 1)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected

    testee = im.shift("C2E2C2E2C", 9)("iter")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim",-1)("iter")))
    b1 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", 0)("iter")))

    expected = im.concat_where(cond0, b0, b1)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected
