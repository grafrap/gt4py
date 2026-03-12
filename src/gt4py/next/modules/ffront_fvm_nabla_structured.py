import gt4py.next as gtx
from gt4py.next.experimental import concat_where

# from gt4py.next import where as concat_where
from gt4py.next import neighbor_sum

# Define Dimensions
IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
Kolor = gtx.Dimension("Kolor")


@gtx.field_operator
def compute_zavgS_cartesian_0(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(JDim == domain_max_j - 1, pp, pp + pp(JDim + 1))
    # zavg = 0.5 * (pp + pp(JDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_1(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        IDim == domain_max_i - 1, pp(Kolor - 1), pp(Kolor - 1) + pp(IDim + 1)(Kolor - 1)
    )
    # zavg = 0.5 * (pp + pp(IDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_2(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        IDim == domain_max_i - 1,
        concat_where(JDim == domain_max_j - 1, 0.0, pp(Kolor - 2) + pp(JDim + 1)(Kolor - 2)),
        concat_where(
            JDim == domain_max_j - 1,
            pp(IDim + 1)(Kolor - 2),
            pp(IDim + 1)(Kolor - 2) + pp(JDim + 1)(Kolor - 2),
        ),
    )
    # zavg = 0.5 * (pp(IDim + 1) + pp(JDim + 1))
    return S_M * zavg


@gtx.field_operator
def on_edges(
    f0: gtx.Field[[IDim, JDim, Kolor], float],
    f1: gtx.Field[[IDim, JDim, Kolor], float],
    f2: gtx.Field[[IDim, JDim, Kolor], float],
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    return concat_where(
        Kolor == 0,
        f0,
        concat_where(Kolor == 1, f1, f2),
    )


@gtx.field_operator
def compute_zavgS_cartesian(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    return on_edges(
        compute_zavgS_cartesian_0(pp, S_M, domain_max_j),
        compute_zavgS_cartesian_1(pp, S_M, domain_max_i),
        compute_zavgS_cartesian_2(pp, S_M, domain_max_i, domain_max_j),
    )


@gtx.program
def zavg(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    out: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,
):
    compute_zavgS_cartesian(
        pp,
        S_M,
        domain_max_i,
        domain_max_j,
        out=out,
        domain={IDim: (0, domain_max_i), JDim: (0, domain_max_j)},
    )


@gtx.field_operator
def compute_pnabla_cartesian(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    sign: tuple[
        gtx.Field[[IDim, JDim, Kolor], float], # [0] East
        gtx.Field[[IDim, JDim, Kolor], float], # [1] West
        gtx.Field[[IDim, JDim, Kolor], float], # [2] NE
        gtx.Field[[IDim, JDim, Kolor], float], # [3] SW
        gtx.Field[[IDim, JDim, Kolor], float], # [4] NW
        gtx.Field[[IDim, JDim, Kolor], float], # [5] SE
    ],
    vol: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    
    zavgS = compute_zavgS_cartesian(pp, S_M, domain_max_i, domain_max_j)

    # Kolor 0: Forward edge is (i, j, 0) [East], Backward is (i, j-1, 0) [West]
    term0 = concat_where(
        JDim == 0,
        zavgS * sign[0],
        zavgS * sign[0] + zavgS(JDim - 1) * sign[1]
    )

    # Kolor 1: Forward edge is (i, j, 1) [NE], Backward is (i-1, j, 1) [SW]
    term1 = concat_where(
        IDim == 0,
        zavgS * sign[2](Kolor - 1),
        zavgS * sign[2](Kolor - 1) + zavgS(IDim - 1) * sign[3](Kolor - 1)
    )

    # Kolor 2: Forward edge is (i, j-1, 2) [NW], Backward is (i-1, j, 2) [SE]
    term2 = concat_where(
        IDim == 0,
        concat_where(JDim == 0, 0.0, zavgS(JDim - 1) * sign[4](Kolor - 2)),
        concat_where(
            JDim == 0,
            zavgS(IDim - 1) * sign[5](Kolor - 2),
            zavgS(JDim - 1) * sign[4](Kolor - 2) + zavgS(IDim - 1) * sign[5](Kolor - 2)
        ),
    )

    pnabla_M = concat_where(
        Kolor == 0,
        term0,
        concat_where(Kolor == 1, term1, term2),
    )
    
    return neighbor_sum(pnabla_M, axis=Kolor) / vol

@gtx.program
def pnabla_cartesian(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    sign: tuple[
        gtx.Field[[IDim, JDim, Kolor], float],
        gtx.Field[[IDim, JDim, Kolor], float],
        gtx.Field[[IDim, JDim, Kolor], float],
        gtx.Field[[IDim, JDim, Kolor], float],
        gtx.Field[[IDim, JDim, Kolor], float],
        gtx.Field[[IDim, JDim, Kolor], float],
    ],
    vol: gtx.Field[[IDim, JDim, Kolor], float],
    out: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,
):
    compute_pnabla_cartesian(
        pp,
        S_M,
        sign,
        vol,
        domain_max_i,
        domain_max_j,
        out=out,
        domain={IDim: (0, domain_max_i), JDim: (0, domain_max_j), Kolor: (0, domain_max_kolor)},
    )


# Observations from this structured stencil vs the unstructured version of pnabla and zavgS:
# - If i need E2V, i need to use concat_where to handle all three edge types. Easiest is to just split it 
#   into three different field operators, one for each edge type.
# - i need forward and back shifts for V2E accesses, so that i can compute the contribution of a vertex field 
#   (has only Kolor 0) to all edge types.
# - When a field is [Globaldim, Localdim], e.g. [Vertex, V2EDim], i need to use two separate fields to 
#   represent this field, because all edges contribute from two edges to the same vertex, i.e. 6 contributions 
#   for in total, while Kolor is only 3.
# - E2V accesses must be written out explicitly with IDim and JDim, but produce the same result in the end.
# - neighbor_sum goes over six edges for each vertex, hence 2 per Kolor, that's why we add every edge in the same
#   Kolor contribution together before doing the neighbor_sum, which then just goes over the three Kolors.
# - Remember reshifts for e.g. the sign or pp field, if we use concat_where on a Vertex field, we need to reshift back 