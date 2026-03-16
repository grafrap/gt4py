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

    print(pnabla_m_struct.asnumpy())

    assert np.isfinite(pnabla_m_struct.asnumpy()).all()


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
    print(pnabla_m_struct.asnumpy())
    assert np.isfinite(pnabla_m_struct.asnumpy()).all()


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

    zavg_program(pp=case["pp_struct"], S_M=case["s_m_struct"], out=zavgS_struct)
    weighted_program(zavgS=zavgS_struct, sign=case["sign_struct"], out=pnabla_m_struct)
    divide_program(pnabla_M=pnabla_m_struct, vol=case["vol_struct"], out=out_struct)

    assert np.isfinite(out_struct.asnumpy()).all()
