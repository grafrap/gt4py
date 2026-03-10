# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import math

from gt4py.next import common
from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm


def _apply_shift_chain(arg: ir.Expr, shift_spec: tuple[ir.OffsetLiteral, ...]) -> ir.Expr:
    return im.call(im.call("shift")(*shift_spec))(arg)


def _kolor_slice(start: int, stop: int) -> ir.Expr:
    # Equivalent to Kolor in [start, stop)
    return im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension("Kolor"): (ir.OffsetLiteral(value=start), ir.OffsetLiteral(value=stop))},
    )

map_dict = {# V2E:
            (ir.OffsetLiteral(value="V2E"), ir.OffsetLiteral(value=0)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                    ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                    ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                ),
            },
            (ir.OffsetLiteral(value="V2E"), ir.OffsetLiteral(value=1)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value="V2E"), ir.OffsetLiteral(value=2)): {
                "kind": "shift",
                "shifts":(
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 3)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 2)),
            },
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 4)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 5)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            # V2C:
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 0)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 1)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 2)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 3)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 4)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 5)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },

            # From now on, we need concat_where to sort out the cases,
            #  as we have more than one Kolor mappint to the current index
            # E2V: 
            (ir.OffsetLiteral(value="E2V"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,  # else branch for Kolor==2
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },

            (ir.OffsetLiteral(value="E2V"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },
        }
            
needs_concat_where = {
    ir.OffsetLiteral(value="E2V"),
    ir.OffsetLiteral(value="E2C"),
    ir.OffsetLiteral(value="C2V"),
    ir.OffsetLiteral(value="C2E"),
}



@dataclasses.dataclass
class CartUnroll(NodeTranslator):

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node: # add dict for mapping later
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.Expr:
        new_node = self.generic_visit(node)

        if cpm.is_applied_shift(new_node):
            key = tuple(new_node.fun.args)
            if key in map_dict:
                entry = map_dict[key]
                arg = new_node.args[0]

                if entry["kind"] == "shift":
                    return _apply_shift_chain(arg, entry["shifts"])

                if entry["kind"] == "concat_where":
                    (cond0, s0), (cond1, s1), (_, s2) = entry["branches"]
                    b0 = _apply_shift_chain(arg, s0)
                    b1 = _apply_shift_chain(arg, s1)
                    b2 = _apply_shift_chain(arg, s2)
                    return im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
        return new_node

# dict V2V[0] -> cart, ...