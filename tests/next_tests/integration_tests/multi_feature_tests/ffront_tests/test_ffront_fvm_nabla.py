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

        setup = nabla_setup.from_connectivity(
            allocator=exec_alloc_descriptor.allocator,
            e2v=e2v,
            v2e=v2e,
            lonlat_deg=lonlat,
        )

        assert setup.nodes_size == remap_sizes.vertex_size
        assert setup.edges_size <= remap_sizes.edge_size_padded
        assert int(ds.sizes["cell"]) == remap_sizes.cell_size

    zavgS = gtx.zeros({Edge: setup.edges_size}, allocator=exec_alloc_descriptor.allocator)

    compute_zavgS.with_backend(
        None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor
    )(
        setup.input_field,
        setup.S_fields[0],
        out=zavgS,
        offset_provider={"E2V": setup.edges2node_connectivity},
    )

    e2v_conn = setup.edges2node_connectivity.asnumpy()
    valid = np.all(e2v_conn >= 0, axis=1)
    pp = setup.input_field.asnumpy()
    s_m = setup.S_fields[0].asnumpy()
    ref = np.zeros_like(s_m)
    ref[valid] = s_m[valid] * 0.5 * (pp[e2v_conn[valid, 0]] + pp[e2v_conn[valid, 1]])

    np.testing.assert_allclose(zavgS.asnumpy(), ref, rtol=1e-12, atol=1e-12)


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
