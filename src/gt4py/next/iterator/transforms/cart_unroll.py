# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import math
import copy
import os

from gt4py.next import common
from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.type_system import type_specifications as ts


def _apply_shift_chain(arg: ir.Expr, shift_spec: tuple[ir.OffsetLiteral, ...]) -> ir.Expr:
    normalized: list[ir.OffsetLiteral] = []
    for idx, off in enumerate(shift_spec):
        if idx % 2 == 0 and isinstance(off.value, str) and off.value in {"IDim", "JDim", "Kolor"}:
            normalized.append(
                ir.OffsetLiteral(value=common.dimension_to_implicit_offset(off.value))
            )
        else:
            normalized.append(off)
    return im.call(im.call("shift")(*tuple(normalized)))(arg)


def _kolor_slice(start: int, stop: int) -> ir.Expr:
    # Equivalent to Kolor in [start, stop)
    return im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension("Kolor"): (ir.OffsetLiteral(value=start), ir.OffsetLiteral(value=stop))},
    )

map_dict = {# V2E:
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 0)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 1)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 2)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                    ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                    ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                ),
            },
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 3)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 4)): {
                "kind": "shift",
                "shifts":(
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            (ir.OffsetLiteral(value = "V2E"), ir.OffsetLiteral(value = 5)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 2)),
            },
            
            # V2C:
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 0)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 1)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 2)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 3)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 4)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 0)),
            },
            (ir.OffsetLiteral(value = "V2C"), ir.OffsetLiteral(value = 5)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value = "IDim"), ir.OffsetLiteral(value = -1),
                    ir.OffsetLiteral(value = "JDim"), ir.OffsetLiteral(value = 0),
                    ir.OffsetLiteral(value = "Kolor"), ir.OffsetLiteral(value = 1)),
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
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,  # else branch for Kolor==2
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
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
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },

            # E2C:
            (ir.OffsetLiteral(value="E2C"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> cell (i-1, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> cell (i, j-1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> cell (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },

            # C2V:
            (ir.OffsetLiteral(value="C2V"), ir.OffsetLiteral(value=0)): {
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
                        None, # else branch for Kolor==1
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2V"), ir.OffsetLiteral(value=1)): {
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
                        None, # else branch for Kolor==1
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2V"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },

            # C2E: 
            (ir.OffsetLiteral(value="C2E"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge kolor 0
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> edge (i, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge kolor 2
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> edge (i, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge kolor 1
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> edge (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },

            # E2C2V
            (ir.OffsetLiteral(value="E2C2V"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # edge kolor 0 -> vertex (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2), # edge kolor 1 -> vertex (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None, # edge kolor 2 -> vertex (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2V"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # edge kolor 0 -> vertex (i, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2), # edge kolor 1 -> vertex (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None, # edge kolor 2 -> vertex (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2V"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # edge kolor 0 -> vertex (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2), # edge kolor 1 -> vertex (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None, # edge kolor 2 -> vertex (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2V"), ir.OffsetLiteral(value=3)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # edge kolor 0 -> vertex (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2), # edge kolor 1 -> vertex (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None, # edge kolor 2 -> vertex (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },

            # C2E2C:
            (ir.OffsetLiteral(value="C2E2C"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i-1, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> cell (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j-1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> cell (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            
            # C2E2CO:
            (ir.OffsetLiteral(value="C2E2CO"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i-1, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> cell (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2CO"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2CO"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j-1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # else branch for Kolor==1, cell (i, j, 1) -> cell (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2CO"), ir.OffsetLiteral(value=3)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                    ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                    ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                ),
            },

            # E2C2E:
            (ir.OffsetLiteral(value="E2C2E"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> edge (i, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> edge (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> edge (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2E"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> edge (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> edge (i, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> edge (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2E"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> edge (i-1, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> edge (i+1, j-1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> edge (i, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2E"), ir.OffsetLiteral(value=3)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> edge (i-1, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> edge (i, j-1, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> edge (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },

            # E2C2EO:
            (ir.OffsetLiteral(value="E2C2EO"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> edge (i, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> edge (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> edge (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2EO"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> edge (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> edge (i, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> edge (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2EO"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> edge (i-1, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> edge (i+1, j-1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> edge (i, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2EO"), ir.OffsetLiteral(value=3)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> edge (i-1, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        _kolor_slice(1, 2),  # edge kolor 1 -> edge (i, j-1, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None,  # edge kolor 2 -> edge (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="E2C2EO"), ir.OffsetLiteral(value=4)): {
                "kind": "shift",
                "shifts": (
                    ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                    ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                    ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                ),
            },

            # C2E2C2E:
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),                    
            },
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                ),                    
            },
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),                    
            },
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=3)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i-1, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i, j+1, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                ),                    
            },
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=4)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i-1, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),                    
            },
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=5)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),                    
            },
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=6)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),                    
            },
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=7)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i+1, j-1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i+1, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),                    
            },
            (ir.OffsetLiteral(value="C2E2C2E"), ir.OffsetLiteral(value=8)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> edge (i, j-1, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=2),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> edge (i+1, j, 2)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                ),                    
            },

            # C2E2C2E2C:
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=1)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i-1, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=2)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=3)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j-1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=4)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i-1, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i-1, j+1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=5)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i-1, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=6)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i+1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i, j-1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=7)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i+1, j-1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i+1, j-1, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=8)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i, j-1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i+1, j, 1)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),
            },
            (ir.OffsetLiteral(value="C2E2C2E2C"), ir.OffsetLiteral(value=9)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1), # cell (i, j, 0) -> cell (i-1, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                    (
                        None, # cell (i, j, 1) -> cell (i, j+1, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=1),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
                        ),
                    ),
                ),
            },
        }


def _build_concat_where_from_branches(arg: ir.Expr, branches):
    # branches: ((cond_or_none, shift_spec), ...)
    cond, shift_spec = branches[0]
    expr = _apply_shift_chain(copy.deepcopy(arg), shift_spec)
    if len(branches) == 1:
        return expr
    return im.concat_where(cond, expr, _build_concat_where_from_branches(arg, branches[1:]))


def _make_lifted_deref_shift(
    arg: ir.Expr, shift_spec: tuple[ir.OffsetLiteral, ...], domain: ir.Expr | None = None
) -> ir.Expr:
    it_name = "__cart_unroll_it"
    return im.as_fieldop(
        im.lambda_(it_name)(im.deref(_apply_shift_chain(im.ref(it_name), shift_spec))),
        domain,
    )(copy.deepcopy(arg))


def _build_field_concat_where_from_branches(arg: ir.Expr, branches, domain: ir.Expr | None = None):
    # branches: ((cond_or_none, shift_spec), ...)
    cond, shift_spec = branches[0]
    expr = _make_lifted_deref_shift(arg, shift_spec, domain)
    if len(branches) == 1:
        return expr
    return im.concat_where(
        cond, expr, _build_field_concat_where_from_branches(arg, branches[1:], domain)
    )

@dataclasses.dataclass
class CartUnroll(NodeTranslator):

    @staticmethod
    def _cartesian_remapped_type(type_: ts.TypeSpec | None) -> ts.TypeSpec | None:
        if not isinstance(type_, ts.FieldType):
            return type_

        if not any(dim.value in {"Edge", "Vertex"} for dim in type_.dims):
            return type_

        idim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
        jdim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
        kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
        return ts.FieldType(dims=[idim, jdim, kolor], dtype=type_.dtype)

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node: # add dict for mapping later
        return cls().visit(node)

    def visit_Program(self, node: ir.Program, **kwargs) -> ir.Program:
        new_params = [
            ir.Sym(id=param.id, type=self._cartesian_remapped_type(param.type)) for param in node.params
        ]
        new_body = [self.visit(stmt, **kwargs) for stmt in node.body]
        return ir.Program(
            id=node.id,
            function_definitions=node.function_definitions,
            params=new_params,
            declarations=node.declarations,
            body=new_body,
        )

    def visit_SymRef(self, node: ir.SymRef, **kwargs) -> ir.SymRef:
        if str(node.id).startswith("__"):
            if node.type is None:
                return node
            return ir.SymRef(id=node.id, type=None)

        mapped_type = self._cartesian_remapped_type(node.type)
        if mapped_type is node.type:
            return node
        return ir.SymRef(id=node.id, type=mapped_type)

    def visit_Sym(self, node: ir.Sym, **kwargs) -> ir.Sym:
        if str(node.id).startswith("__"):
            if node.type is None:
                return node
            return ir.Sym(id=node.id, type=None)

        mapped_type = self._cartesian_remapped_type(node.type)
        if mapped_type is node.type:
            return node
        return ir.Sym(id=node.id, type=mapped_type)

    def visit_Lambda(self, node: ir.Lambda, **kwargs) -> ir.Lambda:
        new_params = [self.visit(param, **kwargs) for param in node.params]
        new_expr = self.visit(node.expr, **kwargs)
        return ir.Lambda(params=new_params, expr=new_expr, type=None)

    def visit_SetAt(self, node: ir.SetAt, **kwargs) -> ir.SetAt:
        new_domain = self.visit(node.domain, **kwargs)
        new_expr = self.visit(node.expr, current_domain=new_domain, **kwargs)
        new_target = self.visit(node.target, **kwargs)
        return ir.SetAt(expr=new_expr, domain=new_domain, target=new_target)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Expr:
        debug = os.environ.get("GT4PY_DEBUG_CART_UNROLL", "0") == "1"
        current_domain = kwargs.get("current_domain")

        def _debug(msg: str, value: ir.Expr | None = None) -> None:
            if debug:
                print(f"[CartUnroll] {msg}")
                if value is not None:
                    print(f"[CartUnroll]   {value}")

        if cpm.is_applied_as_fieldop(node) and len(node.args) == 1:
            stencil = node.fun.args[0]
            if (
                isinstance(stencil, ir.Lambda)
                and len(stencil.params) == 1
                and cpm.is_call_to(stencil.expr, "deref")
                and len(stencil.expr.args) == 1
                and cpm.is_applied_shift(stencil.expr.args[0])
                and isinstance(stencil.expr.args[0].args[0], ir.SymRef)
                and stencil.expr.args[0].args[0].id == stencil.params[0].id
            ):
                shift_call = stencil.expr.args[0]
                key = tuple(shift_call.fun.args)
                if key in map_dict and map_dict[key]["kind"] == "concat_where":
                    rewritten_arg = self.visit(node.args[0], **kwargs)
                    out = _build_field_concat_where_from_branches(
                        rewritten_arg, map_dict[key]["branches"], current_domain
                    )
                    _debug("applied lifted concat_where rewrite", out)
                    return out

        new_node = copy.deepcopy(self.generic_visit(node, **kwargs))

        if cpm.is_applied_shift(new_node):
            key = tuple(new_node.fun.args)
            if key in map_dict:
                entry = map_dict[key]
                arg = new_node.args[0]

                if entry["kind"] == "shift":
                    out = _apply_shift_chain(copy.deepcopy(arg), entry["shifts"])
                    _debug("applied shift rewrite", out)
                    return out

                if entry["kind"] == "concat_where":
                    out = _build_concat_where_from_branches(arg, entry["branches"])
                    _debug("applied concat_where rewrite", out)
                    return out
                    # (cond0, s0), (cond1, s1), (_, s2) = entry["branches"]
                    # b0 = _apply_shift_chain(arg, s0)
                    # b1 = _apply_shift_chain(arg, s1)
                    # b2 = _apply_shift_chain(arg, s2)
                    # return im.concat_where(cond0, b0, im.concat_where(cond1, b1, b2))
        
        # replace Field[Edge] with Field[IDim, JDim, Kolor]
        # i.e. replace get_domain_range(out, Edgeₕ) by the equivalent in IDim, JDim, Kolor
        _debug("visit_FunCall new_node", new_node)

        # if cpm.is_call_to(new_node, "get_domain_range") and new_node.args[1] == ir.OffsetLiteral(value="Edgeₕ"):
        def _extract_field_from_get_domain_range(expr: ir.Expr, expected_axis: str) -> ir.Expr | None:
        # Match tuple_get(k, get_domain_range(field, Axis))
            if not cpm.is_call_to(expr, "tuple_get") or len(expr.args) != 2:
                return None

            gdr = expr.args[1]
            if not cpm.is_call_to(gdr, "get_domain_range") or len(gdr.args) != 2:
                return None

            field_expr, axis_expr = gdr.args
            if not isinstance(axis_expr, ir.AxisLiteral) or axis_expr.value != expected_axis:
                return None
            return field_expr

        if cpm.is_call_to(new_node, "unstructured_domain") and len(new_node.args) == 1:
            nr = new_node.args[0]
            if cpm.is_call_to(nr, "named_range") and len(nr.args) == 3:
                axis, start, stop = nr.args
                if isinstance(axis, ir.AxisLiteral) and axis.value == "Edge":
                    field_from_start = _extract_field_from_get_domain_range(start, "Edge")
                    field_from_stop = _extract_field_from_get_domain_range(stop, "Edge")

                    if field_from_start is not None and field_from_start == field_from_stop:
                        _debug("domain rewrite matched field", field_from_start)
                        out_field = field_from_start

                        IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
                        JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
                        Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)

                        def _bounds(dim: common.Dimension) -> tuple[ir.Expr, ir.Expr]:
                            gdr = im.call("get_domain_range")(out_field, dim)
                            return im.tuple_get(0, gdr), im.tuple_get(1, gdr)

                        i0, i1 = _bounds(IDim)
                        j0, j1 = _bounds(JDim)
                        k0, k1 = _bounds(Kolor)
                        replacement_domain = im.call("cartesian_domain")(
                            im.named_range(IDim, i0, i1),
                            im.named_range(JDim, j0, j1),
                            im.named_range(Kolor, k0, k1),
                        )
                        _debug("domain rewrite replacement", replacement_domain)
                        return replacement_domain
                    _debug("domain rewrite skipped: start/stop fields mismatch")
                else:
                    _debug("domain rewrite skipped: axis is not Edge")
            else:
                _debug("domain rewrite skipped: not a named_range with 3 args")
        else:
            _debug("domain rewrite skipped: not a single-range unstructured_domain")
        return new_node

# dict V2V[0] -> cart, ...