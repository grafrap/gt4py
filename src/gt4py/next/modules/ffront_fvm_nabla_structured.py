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
    sign: gtx.Field[[IDim, JDim, Kolor], float],
    vol: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavgS = compute_zavgS_cartesian(pp, S_M, domain_max_i, domain_max_j)

    pnabla_M = concat_where(
        Kolor == 0,
        concat_where(JDim == 0, zavgS, zavgS + zavgS(JDim - 1)),
        concat_where(
            Kolor == 1,
            concat_where(IDim == 0, zavgS, zavgS + zavgS(IDim - 1)),
            concat_where(
                IDim == 0,
                concat_where(JDim == 0, zavgS - zavgS, zavgS(JDim - 1)),
                concat_where(JDim == 0, zavgS(IDim - 1), zavgS(IDim - 1) + zavgS(JDim - 1)),
            ),
        ),
    )
    pnabla_M = pnabla_M * sign
    return neighbor_sum(pnabla_M, axis=Kolor) / vol


@gtx.program
def pnabla_cartesian(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    sign: gtx.Field[[IDim, JDim, Kolor], float],
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
