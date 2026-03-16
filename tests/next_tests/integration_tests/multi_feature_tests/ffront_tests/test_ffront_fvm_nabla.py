# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple
import os

import numpy as np
import pytest

pytest.importorskip("atlas4py")

from gt4py import next as gtx
from gt4py.next import neighbor_sum
from gt4py.next.program_processors.program_setup_utils import setup_program
from gt4py.next.program_processors.runners import gtfn as gtfn_runner


from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)
from next_tests.integration_tests.multi_feature_tests.fvm_nabla_setup import (
    E2V,
    V2E,
    E2VDim,
    Edge,
    V2EDim,
    Vertex,
    assert_close,
    nabla_setup,
)
from gt4py.next.modules.translator import load_structured_remap_sizes_from_netcdf
from gt4py.next.modules.translator import (
    IDim,
    JDim,
    Kolor,
    build_index_map_from_lonlat_e2v,
    pack_edge_field_to_structured,
    pack_vertex_field,
    pack_vertex_field_to_structured,
    unpack_edge_field,
    unpack_vertex_field_to_unstructured,
)


def _first_present(ds, names, required=True):
    for name in names:
        if name in ds:
            return ds[name].values
    if required:
        raise KeyError(f"None of the variables {names} found in dataset.")
    return None


def _read_e2v(ds):
    raw = _first_present(ds, ["E2V", "edge_vertices", "edges2nodes", "edge_node_connectivity"])
    arr = np.asarray(raw, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError("e2v dataset must be 2-D")
    if arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"e2v must have shape (n_edge, 2), got {arr.shape}")
    return np.where(arr > 0, arr - 1, -1)


def _read_v2e(ds):
    raw = _first_present(
        ds,
        ["V2E", "vertex_edges", "nodes2edges", "node_edge_connectivity", "edges_of_vertex"],
        required=False,
    )
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError("v2e dataset must be 2-D")
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return np.where(arr > 0, arr - 1, -1)


def _read_lonlat(ds):
    if "longitude_vertices" in ds and "latitude_vertices" in ds:
        lon = ds["longitude_vertices"].values.astype(np.float64)
        lat = ds["latitude_vertices"].values.astype(np.float64)
        return np.stack([lon, lat], axis=1)
    return _first_present(ds, ["lonlat", "vertex_lonlat", "node_lonlat"], required=False)


@gtx.field_operator
def compute_zavgS(
    pp: gtx.Field[[Vertex], float], S_M: gtx.Field[[Edge], float]
) -> gtx.Field[[Edge], float]:
    zavg = 0.5 * (pp(E2V[0]) + pp(E2V[1]))
    return S_M * zavg


@gtx.field_operator
def compute_pnabla(
    pp: gtx.Field[[Vertex], float],
    S_M: gtx.Field[[Edge], float],
    sign: gtx.Field[[Vertex, V2EDim], float],
    vol: gtx.Field[[Vertex], float],
) -> gtx.Field[[Vertex], float]:
    zavgS = compute_zavgS(pp, S_M)
    pnabla_M = neighbor_sum(zavgS(V2E) * sign, axis=V2EDim)
    return pnabla_M / vol


@gtx.field_operator
def compute_neighbor_sum_unweighted(
    zavgS: gtx.Field[[Edge], float],
) -> gtx.Field[[Vertex], float]:
    return neighbor_sum(zavgS(V2E), axis=V2EDim)


@gtx.field_operator
def compute_neighbor_sum_weighted(
    zavgS: gtx.Field[[Edge], float],
    sign: gtx.Field[[Vertex, V2EDim], float],
) -> gtx.Field[[Vertex], float]:
    return neighbor_sum(zavgS(V2E) * sign, axis=V2EDim)


@gtx.field_operator
def compute_divide_volume(
    pnabla_M: gtx.Field[[Vertex], float],
    vol: gtx.Field[[Vertex], float],
) -> gtx.Field[[Vertex], float]:
    return pnabla_M / vol


@gtx.field_operator
def pnabla(
    pp: gtx.Field[[Vertex], float],
    S_M: Tuple[gtx.Field[[Edge], float], gtx.Field[[Edge], float]],
    sign: gtx.Field[[Vertex, V2EDim], float],
    vol: gtx.Field[[Vertex], float],
) -> Tuple[gtx.Field[[Vertex], float], gtx.Field[[Vertex], float]]:
    return compute_pnabla(pp, S_M[0], sign, vol), compute_pnabla(pp, S_M[1], sign, vol)


@pytest.mark.requires_atlas
def test_ffront_compute_zavgS(exec_alloc_descriptor):
    setup = nabla_setup(allocator=exec_alloc_descriptor.allocator)

    zavgS = gtx.zeros({Edge: setup.edges_size}, allocator=exec_alloc_descriptor.allocator)

    compute_zavgS.with_backend(
        None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor
    )(
        setup.input_field,
        setup.S_fields[0],
        out=zavgS,
        offset_provider={"E2V": setup.edges2node_connectivity},
    )

    assert_close(-199755464.25741270, np.min(zavgS.asnumpy()))
    assert_close(388241977.58389181, np.max(zavgS.asnumpy()))


@pytest.mark.requires_atlas
def test_ffront_compute_zavgS_parallelogram_grid(exec_alloc_descriptor):
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH",
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc",
    )
    xr = pytest.importorskip("xarray")

    with xr.open_dataset(mesh_nc) as ds:
        e2v = _read_e2v(ds)
        v2e = _read_v2e(ds)
        lonlat = _read_lonlat(ds)
        remap_sizes = load_structured_remap_sizes_from_netcdf(mesh_nc)

        print(f"remap sies: ",remap_sizes)

        setup = nabla_setup.from_connectivity(
            allocator=exec_alloc_descriptor.allocator,
            e2v=e2v,
            v2e=v2e,
            lonlat_deg=lonlat,
        )

        assert setup.nodes_size == remap_sizes.vertex_size
        assert setup.edges_size <= remap_sizes.edge_size_padded
        assert int(ds.sizes["cell"]) == remap_sizes.cell_size

    index_map = build_index_map_from_lonlat_e2v(
        lonlat,
        e2v,
        nodes_size=setup.nodes_size,
        edges_size=setup.edges_size,
    )

    pp_struct_np = pack_vertex_field_to_structured(setup.input_field.asnumpy(), index_map)
    s_m_struct_np = pack_edge_field_to_structured(setup.S_fields[0].asnumpy(), index_map)
    zavgS_struct_np = np.zeros_like(s_m_struct_np)

    assert pp_struct_np.shape[0] == remap_sizes.max_i
    assert pp_struct_np.shape[1] == remap_sizes.max_j
    assert s_m_struct_np.shape[2] == 3

    pp_struct = gtx.as_field([IDim, JDim, Kolor], pp_struct_np, allocator=exec_alloc_descriptor.allocator)
    s_m_struct = gtx.as_field(
        [IDim, JDim, Kolor], s_m_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    zavgS_struct = gtx.as_field(
        [IDim, JDim, Kolor], zavgS_struct_np, allocator=exec_alloc_descriptor.allocator
    )

    selected_backend = gtfn_runner.GTFNBackendFactory(
        cached=True,
        otf_workflow__cached_translation=True,
        otf_workflow__bare_translation__symbolic_domain_sizes={
            "max_i": int(remap_sizes.max_i),
            "max_j": int(remap_sizes.max_j),
        },
    )
    compute_zavgS_program = setup_program(
        compute_zavgS,
        backend=selected_backend,
        offset_provider={"E2V": setup.edges2node_connectivity},
    )

    compute_zavgS_program(
        pp=pp_struct,
        S_M=s_m_struct,
        out=zavgS_struct,
    )

    zavgS_np = unpack_edge_field(zavgS_struct.asnumpy(), index_map, setup.edges_size)

    e2v_conn = setup.edges2node_connectivity.asnumpy()
    valid = np.all(e2v_conn >= 0, axis=1)
    pp = setup.input_field.asnumpy()
    s_m = setup.S_fields[0].asnumpy()
    ref = np.zeros_like(s_m)
    ref[valid] = s_m[valid] * 0.5 * (pp[e2v_conn[valid, 0]] + pp[e2v_conn[valid, 1]])

    np.testing.assert_allclose(zavgS_np, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.requires_atlas
def test_ffront_nabla(exec_alloc_descriptor):
    setup = nabla_setup(allocator=exec_alloc_descriptor.allocator)

    pnabla_MXX = gtx.zeros({Vertex: setup.nodes_size}, allocator=exec_alloc_descriptor.allocator)
    pnabla_MYY = gtx.zeros({Vertex: setup.nodes_size}, allocator=exec_alloc_descriptor.allocator)

    pnabla.with_backend(None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor)(
        setup.input_field,
        setup.S_fields,
        setup.sign_field,
        setup.vol_field,
        out=(pnabla_MXX, pnabla_MYY),
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    # TODO this check is not sensitive enough, need to implement a proper numpy reference!
    assert_close(-3.5455427772566003e-003, np.min(pnabla_MXX.asnumpy()))
    assert_close(3.5455427772565435e-003, np.max(pnabla_MXX.asnumpy()))
    assert_close(-3.3540113705465301e-003, np.min(pnabla_MYY.asnumpy()))
    assert_close(3.3540113705465301e-003, np.max(pnabla_MYY.asnumpy()))

def test_ffront_nabla_parallelogram_grid(exec_alloc_descriptor):
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH",
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc",
    )
    xr = pytest.importorskip("xarray")

    with xr.open_dataset(mesh_nc) as ds:
        e2v = _read_e2v(ds)
        v2e = _read_v2e(ds)
        lonlat = _read_lonlat(ds)
        remap_sizes = load_structured_remap_sizes_from_netcdf(mesh_nc)

        print(f"remap sies: ",remap_sizes)

        setup = nabla_setup.from_connectivity(
            allocator=exec_alloc_descriptor.allocator,
            e2v=e2v,
            v2e=v2e,
            lonlat_deg=lonlat,
        )

        assert setup.nodes_size == remap_sizes.vertex_size
        assert setup.edges_size <= remap_sizes.edge_size_padded
        assert int(ds.sizes["cell"]) == remap_sizes.cell_size
    
    index_map = build_index_map_from_lonlat_e2v(
        lonlat,
        e2v,
        nodes_size=setup.nodes_size,
        edges_size=setup.edges_size,
    )

    pp_struct_np = pack_vertex_field_to_structured(setup.input_field.asnumpy(), index_map)
    s_m_struct_np = pack_edge_field_to_structured(setup.S_fields[0].asnumpy(), index_map)
    sign_struct_np = pack_vertex_field(setup.sign_field.asnumpy(), index_map)
    assert sign_struct_np.ndim == 4
    assert sign_struct_np.shape[2] == 1
    assert sign_struct_np.shape[3] == 6
    vol_struct_np = pack_vertex_field_to_structured(setup.vol_field.asnumpy(), index_map)
    pnabla_mxx_struct_np = np.zeros_like(vol_struct_np)
    pnabla_myy_struct_np = np.zeros_like(vol_struct_np)

    pp_struct = gtx.as_field([IDim, JDim, Kolor], pp_struct_np, allocator=exec_alloc_descriptor.allocator)
    s_m_struct = gtx.as_field(
        [IDim, JDim, Kolor], s_m_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    sign_struct = gtx.as_field(
        [IDim, JDim, Kolor, V2EDim], sign_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    vol_struct = gtx.as_field(
        [IDim, JDim, Kolor], vol_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    pnabla_mxx_struct = gtx.as_field(
        [IDim, JDim, Kolor], pnabla_mxx_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    pnabla_myy_struct = gtx.as_field(
        [IDim, JDim, Kolor], pnabla_myy_struct_np, allocator=exec_alloc_descriptor.allocator
    )

    selected_backend = gtfn_runner.GTFNBackendFactory(
        cached=True,
        otf_workflow__cached_translation=True,
        otf_workflow__bare_translation__symbolic_domain_sizes={
            "max_i": int(remap_sizes.max_i),
            "max_j": int(remap_sizes.max_j),
        },
    )
    pnabla_mxx_program = setup_program(
        compute_pnabla,
        backend=selected_backend,
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )
    pnabla_myy_program = setup_program(
        compute_pnabla,
        backend=selected_backend,
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    pnabla_mxx_program(
        pp=pp_struct,
        S_M=s_m_struct,
        sign=sign_struct,
        vol=vol_struct,
        out=pnabla_mxx_struct,
    )
    pnabla_myy_program(
        pp=pp_struct,
        S_M=s_m_struct,
        sign=sign_struct,
        vol=vol_struct,
        out=pnabla_myy_struct,
    )

    pnabla_mxx_np = unpack_vertex_field_to_unstructured(pnabla_mxx_struct.asnumpy(), index_map)
    pnabla_myy_np = unpack_vertex_field_to_unstructured(pnabla_myy_struct.asnumpy(), index_map)
    print(pnabla_mxx_np)
    print(pnabla_myy_np)
    # TODO this check is not sensitive enough, need to implement a proper numpy reference!
    assert_close(-3.5455427772566003e-003, np.min(pnabla_mxx_np))
    assert_close(3.5455427772565435e-003, np.max(pnabla_mxx_np))
    assert_close(-3.3540113705465301e-003, np.min(pnabla_myy_np))
    assert_close(3.3540113705465301e-003, np.max(pnabla_myy_np))


def _prepare_parallelogram_structured_case(exec_alloc_descriptor):
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH",
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc",
    )
    xr = pytest.importorskip("xarray")

    with xr.open_dataset(mesh_nc) as ds:
        e2v = _read_e2v(ds)
        v2e = _read_v2e(ds)
        lonlat = _read_lonlat(ds)
        remap_sizes = load_structured_remap_sizes_from_netcdf(mesh_nc)
        setup = nabla_setup.from_connectivity(
            allocator=exec_alloc_descriptor.allocator,
            e2v=e2v,
            v2e=v2e,
            lonlat_deg=lonlat,
        )

    index_map = build_index_map_from_lonlat_e2v(
        lonlat,
        e2v,
        nodes_size=setup.nodes_size,
        edges_size=setup.edges_size,
    )

    pp_struct_np = pack_vertex_field_to_structured(setup.input_field.asnumpy(), index_map)
    s_m_struct_np = pack_edge_field_to_structured(setup.S_fields[0].asnumpy(), index_map)
    sign_struct_np = pack_vertex_field(setup.sign_field.asnumpy(), index_map)
    vol_struct_np = pack_vertex_field_to_structured(setup.vol_field.asnumpy(), index_map)

    pp_struct = gtx.as_field([IDim, JDim, Kolor], pp_struct_np, allocator=exec_alloc_descriptor.allocator)
    s_m_struct = gtx.as_field(
        [IDim, JDim, Kolor], s_m_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    sign_struct = gtx.as_field(
        [IDim, JDim, Kolor, V2EDim], sign_struct_np, allocator=exec_alloc_descriptor.allocator
    )
    vol_struct = gtx.as_field(
        [IDim, JDim, Kolor], vol_struct_np, allocator=exec_alloc_descriptor.allocator
    )

    selected_backend = gtfn_runner.GTFNBackendFactory(
        cached=True,
        otf_workflow__cached_translation=True,
        otf_workflow__bare_translation__symbolic_domain_sizes={
            "max_i": int(remap_sizes.max_i),
            "max_j": int(remap_sizes.max_j),
        },
    )

    return {
        "setup": setup,
        "index_map": index_map,
        "backend": selected_backend,
        "pp_struct": pp_struct,
        "s_m_struct": s_m_struct,
        "sign_struct": sign_struct,
        "vol_struct": vol_struct,
        "remap_sizes": remap_sizes,
    }


def test_ffront_nabla_parallelogram_part_zavgS(exec_alloc_descriptor):
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

    program(
        pp=case["pp_struct"],
        S_M=case["s_m_struct"],
        out=zavgS_struct,
    )

    assert np.isfinite(zavgS_struct.asnumpy()).all()


def test_ffront_nabla_parallelogram_part_neighbor_sum_unweighted(exec_alloc_descriptor):
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

    assert np.isfinite(pnabla_m_struct.asnumpy()).all()


def test_ffront_nabla_parallelogram_part_neighbor_sum_weighted(exec_alloc_descriptor):
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

    assert np.isfinite(pnabla_m_struct.asnumpy()).all()


def test_ffront_nabla_parallelogram_part_divide(exec_alloc_descriptor):
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