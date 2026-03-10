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
    testee = im.shift("V2E", 0)("i_s")
    expected = im.shift("Kolor", 2)(im.shift("JDim", -1)((im.shift("IDim", 0)("i_s"))))
    expected = NormalizeShifts().visit(expected)

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 1)("i_s")
    expected = im.shift("Kolor", 0)(im.shift("JDim", -1)((im.shift("IDim", 0)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 2)("i_s")
    expected = im.shift("Kolor", 1)(im.shift("JDim", 0)((im.shift("IDim", -1)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 3)("i_s")
    expected = im.shift("Kolor", 2)(im.shift("JDim", 0)((im.shift("IDim", -1)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 4)("i_s")
    expected = im.shift("Kolor", 0)(im.shift("JDim", 0)((im.shift("IDim", 0)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2E", 5)("i_s")
    expected = im.shift("Kolor", 1)(im.shift("JDim", 0)((im.shift("IDim", 0)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

def test_V2C():
    testee = im.shift("V2C", 0)("i_s")
    expected = im.shift("Kolor", 1)(im.shift("JDim", -1)((im.shift("IDim", 0)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 1)("i_s")
    expected = im.shift("Kolor", 0)(im.shift("JDim", -1)((im.shift("IDim", 0)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 2)("i_s")
    expected = im.shift("Kolor", 1)(im.shift("JDim", -1)((im.shift("IDim", -1)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 3)("i_s")
    expected = im.shift("Kolor", 0)(im.shift("JDim", 0)((im.shift("IDim", -1)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 4)("i_s")
    expected = im.shift("Kolor", 1)(im.shift("JDim", 0)((im.shift("IDim", -1)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected

    testee = im.shift("V2C", 5)("i_s")
    expected = im.shift("Kolor", 0)(im.shift("JDim", 0)((im.shift("IDim", 0)("i_s"))))
    expected = NormalizeShifts().visit(expected)
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    assert actual == expected


def test_E2V():
    
    testee = im.shift("E2V", 0)("i_s")
    kolor = common.Dimension("Kolor")
    cond0 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=0), ir.OffsetLiteral(value=1))},
    )
    cond1 = im.domain(
        common.GridType.CARTESIAN,
        {kolor: (ir.OffsetLiteral(value=1), ir.OffsetLiteral(value=2))},
    )

    b0 = im.shift("Kolor", 0)(im.shift("JDim", 0)(im.shift("IDim", 0)("i_s")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 0)("i_s")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 0)(im.shift("IDim", 1)("i_s")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))

    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)

    assert actual == expected

    testee = im.shift("E2V", 1)("i_s")
    b0 = im.shift("Kolor", 0)(im.shift("JDim", 1)(im.shift("IDim", 0)("i_s")))
    b1 = im.shift("Kolor", -1)(im.shift("JDim", 0)(im.shift("IDim", 1)("i_s")))
    b2 = im.shift("Kolor", -2)(im.shift("JDim", 1)(im.shift("IDim", 0)("i_s")))

    expected = im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
    actual = CartUnroll.apply(testee)
    actual = NormalizeShifts().visit(actual)
    expected = NormalizeShifts().visit(expected)
    assert actual == expected