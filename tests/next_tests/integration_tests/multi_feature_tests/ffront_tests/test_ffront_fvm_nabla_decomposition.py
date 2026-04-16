# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

pytest.importorskip("atlas4py")

from gt4py import next as gtx

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)
from next_tests.integration_tests.multi_feature_tests.ffront_tests.test_ffront_fvm_nabla import (
    IDim,
    JDim,
    Kolor,
    compute_divide_volume,
    compute_neighbor_sum_unweighted,
    compute_neighbor_sum_weighted,
    compute_zavgS,
    _prepare_parallelogram_structured_case,
)
from gt4py.next.program_processors.program_setup_utils import setup_program


def _require_gtfn_run_gtfn_backend(request):
    param_id = str(getattr(getattr(request.node, "callspec", None), "id", ""))
    if param_id != "gtfn.run_gtfn":
        pytest.skip("decomposition debug test is restricted to gtfn.run_gtfn")


@pytest.mark.requires_atlas
def test_ffront_nabla_decompose_part_zavgS(exec_alloc_descriptor, request):
    _require_gtfn_run_gtfn_backend(request)
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)

    zavgS_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 3,
        },
        allocator=exec_alloc_descriptor.allocator,
    )

    program = setup_program(
        compute_zavgS,
        backend=case["backend"],
        offset_provider={"E2V": case["setup"].edges2node_connectivity},
    )

    program(pp=case["pp_struct"], S_M=case["s_m_struct"], out=zavgS_struct)
    assert np.isfinite(zavgS_struct.asnumpy()).all()


@pytest.mark.requires_atlas
def test_ffront_nabla_decompose_part_neighbor_sum_unweighted(exec_alloc_descriptor, request):
    _require_gtfn_run_gtfn_backend(request)
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)

    zavgS_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 3,
        },
        allocator=exec_alloc_descriptor.allocator,
    )
    pnabla_m_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 1,
        },
        allocator=exec_alloc_descriptor.allocator,
    )

    zavg_program = setup_program(
        compute_zavgS,
        backend=case["backend"],
        offset_provider={"E2V": case["setup"].edges2node_connectivity},
    )
    unweighted_program = setup_program(
        compute_neighbor_sum_unweighted,
        backend=case["backend"],
        offset_provider={"V2E": case["setup"].nodes2edge_connectivity},
    )

    zavg_program(pp=case["pp_struct"], S_M=case["s_m_struct"], out=zavgS_struct)
    unweighted_program(zavgS=zavgS_struct, out=pnabla_m_struct)

    print(f"zavgS: ", zavgS_struct.asnumpy()[:, :, 0])
    print(pnabla_m_struct.asnumpy()[:, :, 0])

    # numpy reference implementation of the unweighted neighbor sum
    # with zero-padded boundaries (no negative-index wraparound).
    zavg_np = zavgS_struct.asnumpy()
    pnabla_numpy = np.zeros_like(pnabla_m_struct.asnumpy())

    def _zavg_or_zero(i: int, j: int, k: int) -> float:
        if i < 0 or j < 0 or i >= zavg_np.shape[0] or j >= zavg_np.shape[1]:
            return 0.0
        return float(zavg_np[i, j, k])

    for i in range(0, case["remap_sizes"].max_i):
        for j in range(0, case["remap_sizes"].max_j):
            pnabla_numpy[i, j, 0] = (
                _zavg_or_zero(i, j, 0)
                + _zavg_or_zero(i, j - 1, 0)
                + _zavg_or_zero(i, j, 1)
                + _zavg_or_zero(i - 1, j, 1)
                + _zavg_or_zero(i, j - 1, 2)
                + _zavg_or_zero(i - 1, j, 2)
            )

    print(pnabla_numpy[:, :, 0])
    assert np.isfinite(pnabla_m_struct.asnumpy()).all()
    assert np.allclose(pnabla_m_struct.asnumpy(), pnabla_numpy, rtol=1e-10, atol=0)


@pytest.mark.requires_atlas
def test_ffront_nabla_decompose_part_neighbor_sum_weighted(exec_alloc_descriptor, request):
    _require_gtfn_run_gtfn_backend(request)
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)

    zavgS_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 3,
        },
        allocator=exec_alloc_descriptor.allocator,
    )
    pnabla_m_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 1,
        },
        allocator=exec_alloc_descriptor.allocator,
    )

    zavg_program = setup_program(
        compute_zavgS,
        backend=case["backend"],
        offset_provider={"E2V": case["setup"].edges2node_connectivity},
    )
    weighted_program = setup_program(
        compute_neighbor_sum_weighted,
        backend=case["backend"],
        offset_provider={"V2E": case["setup"].nodes2edge_connectivity},
    )

    zavg_program(pp=case["pp_struct"], S_M=case["s_m_struct"], out=zavgS_struct)
    weighted_program(zavgS=zavgS_struct, sign=case["sign_struct"], out=pnabla_m_struct)

    # Numpy reference implementation:
    zavg_np = zavgS_struct.asnumpy()
    sign_np = case["sign_struct"].asnumpy()
    pnabla_numpy = np.zeros_like(pnabla_m_struct.asnumpy())

    # print("sign: ", sign_np)
    def _zavg_or_zero(i: int, j: int, k: int) -> float:
        if i < 0 or j < 0 or i >= zavg_np.shape[0] or j >= zavg_np.shape[1]:
            return 0.0
        return float(zavg_np[i, j, k])

    # print("zavgS k=0: ", zavg_np[:,:,0])
    # print("zavgS k=1: ", zavg_np[:,:,1])
    # print("zavgS k=2: ", zavg_np[:,:,2])

    for i in range(0, case["remap_sizes"].max_i):
        for j in range(0, case["remap_sizes"].max_j):
            pnabla_numpy[i, j, 0] = (
                _zavg_or_zero(i, j, 0) * sign_np[i, j, 0, 0]
                + _zavg_or_zero(i, j - 1, 0) * sign_np[i, j, 0, 3]
                + _zavg_or_zero(i, j, 1) * sign_np[i, j, 0, 1]
                + _zavg_or_zero(i - 1, j, 1) * sign_np[i, j, 0, 4]
                + _zavg_or_zero(i, j - 1, 2) * sign_np[i, j, 0, 2]
                + _zavg_or_zero(i - 1, j, 2) * sign_np[i, j, 0, 5]
            )

    # print(f"\npnabla from calculation: \n", pnabla_m_struct.asnumpy()[:, :, 0])
    # print(f"pnabla from numpy reference: \n", pnabla_numpy[:,:,0])
    assert np.isfinite(pnabla_m_struct.asnumpy()).all()
    assert np.allclose(pnabla_m_struct.asnumpy(), pnabla_numpy, rtol=1e-10, atol=0)


@pytest.mark.requires_atlas
def test_ffront_nabla_decompose_part_divide(exec_alloc_descriptor, request):
    _require_gtfn_run_gtfn_backend(request)
    case = _prepare_parallelogram_structured_case(exec_alloc_descriptor)

    zavgS_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 3,
        },
        allocator=exec_alloc_descriptor.allocator,
    )
    pnabla_m_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 1,
        },
        allocator=exec_alloc_descriptor.allocator,
    )
    out_struct = gtx.zeros(
        {
            IDim: int(case["remap_sizes"].max_i),
            JDim: int(case["remap_sizes"].max_j),
            Kolor: 1,
        },
        allocator=exec_alloc_descriptor.allocator,
    )

    zavg_program = setup_program(
        compute_zavgS,
        backend=case["backend"],
        offset_provider={"E2V": case["setup"].edges2node_connectivity},
    )
    weighted_program = setup_program(
        compute_neighbor_sum_weighted,
        backend=case["backend"],
        offset_provider={"V2E": case["setup"].nodes2edge_connectivity},
    )
    divide_program = setup_program(
        compute_divide_volume,
        backend=case["backend"],
    )

    sign_np = case["sign_struct"].asnumpy()
    # print(sign_np.shape)
    zavg_program(pp=case["pp_struct"], S_M=case["s_m_struct"], out=zavgS_struct)
    weighted_program(zavgS=zavgS_struct, sign=case["sign_struct"], out=pnabla_m_struct)
    divide_program(pnabla_M=pnabla_m_struct, vol=case["vol_struct"], out=out_struct)

    # Numpy reference implementation:
    zavg_np = zavgS_struct.asnumpy()

    vol_np = case["vol_struct"].asnumpy()
    pnabla_numpy = np.zeros_like(pnabla_m_struct.asnumpy())
    out_numpy = np.zeros_like(out_struct.asnumpy())

    def _zavg_or_zero(i: int, j: int, k: int) -> float:
        if i < 0 or j < 0 or i >= zavg_np.shape[0] or j >= zavg_np.shape[1]:
            return 0.0
        return float(zavg_np[i, j, k])

    for i in range(0, case["remap_sizes"].max_i):
        for j in range(0, case["remap_sizes"].max_j):
            pnabla_numpy[i, j, 0] = (
                _zavg_or_zero(i, j, 0) * sign_np[i, j, 0, 0]
                + _zavg_or_zero(i, j - 1, 0) * sign_np[i, j, 0, 3]
                + _zavg_or_zero(i, j, 1) * sign_np[i, j, 0, 1]
                + _zavg_or_zero(i - 1, j, 1) * sign_np[i, j, 0, 4]
                + _zavg_or_zero(i, j - 1, 2) * sign_np[i, j, 0, 2]
                + _zavg_or_zero(i - 1, j, 2) * sign_np[i, j, 0, 5]
            )
            out_numpy[i, j, 0] = pnabla_numpy[i, j, 0] / vol_np[i, j, 0]

    assert np.isfinite(out_struct.asnumpy()).all()
    # print(out_struct.asnumpy()[:,:,0])
    # print(out_numpy[:,:,0])
    assert np.allclose(out_struct.asnumpy(), out_numpy, rtol=1e-10, atol=0)
