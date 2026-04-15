from gt4py.next import common
from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_specifications as ts

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
            # E2V: # TODO: maybe change shifts for some edge kolors, since order matters.
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
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=1),
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

            # E2C: # TODO: maybe change shifts for edge kolor 0 and 2, to reproduce old behavior.
            (ir.OffsetLiteral(value="E2C"), ir.OffsetLiteral(value=0)): {
                "kind": "concat_where",
                "branches": (
                    (
                        _kolor_slice(0, 1),  # edge kolor 0 -> cell (i, j, 0)
                        (
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=-1),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=1),
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
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-1),
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
                            ir.OffsetLiteral(value="IDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="JDim"), ir.OffsetLiteral(value=0),
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=0),
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
                            ir.OffsetLiteral(value="Kolor"), ir.OffsetLiteral(value=-2),
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
