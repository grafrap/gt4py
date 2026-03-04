import os

import numpy as np
import pytest

pytest.importorskip("atlas4py")

from gt4py import next as gtx
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import exec_alloc_descriptor
from next_tests.integration_tests.multi_feature_tests.fvm_nabla_setup import nabla_setup
from next_tests.integration_tests.multi_feature_tests.ffront_tests.test_ffront_fvm_nabla import pnabla
from gt4py.next.iterator import atlas_utils

from gt4py.next.modules.translator import (
    build_index_map_from_atlas_setup,
    build_index_map_from_lonlat_e2v,
    build_structured_sign_from_unstructured,
    run_structured_pnabla_from_unstructured,
    build_index_map_from_ds_regular,
    build_index_map_for_ragged_lonlat_e2v,
)


def _first_present(ds, names, required=True):
    for name in names:
        if name in ds:
            return ds[name].values
    if required:
        raise KeyError(f"None of the variables {names} found in dataset.")
    return None

def _get_lonlat(ds):
    # ICON files store lon/lat separately
    if "longitude_vertices" in ds and "latitude_vertices" in ds:
        lat = ds["latitude_vertices"].values.astype(np.float64)
        lon = ds["longitude_vertices"].values.astype(np.float64)
        return np.stack([lon, lat], axis=1)
    # fall back to any single combined array
    return _first_present(ds, ["lonlat", "vertex_lonlat", "node_lonlat"], required=False)

def _get_dual_normals(ds):
    # pick whichever pair exists
    if "zonal_normal_dual_edge" in ds and "meridional_normal_dual_edge" in ds:
        zx = ds["zonal_normal_dual_edge"].values.astype(np.float64)
        zy = ds["meridional_normal_dual_edge"].values.astype(np.float64)
        return np.stack([zx, zy], axis=1)
    # fallback: look for a single two‑column variable
    normals = _first_present(
        ds,
        ["dual_normals", "edge_dual_normals", "edge_normals"],
        required=False,
    )
    return normals

def _read_e2v(ds):
    raw = _first_present(
        ds,
        ["E2V", "edge_vertices", "edges2nodes", "edge_node_connectivity"],
    )
    arr = np.asarray(raw, dtype=np.int32)

    # make sure edges are first axis and there are exactly two endpoints
    if arr.ndim != 2:
        raise ValueError("e2v dataset must be 2‑D")
    if arr.shape[1] != 2:
        # try transpose – common case is (2, n_edge)
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError("e2v must have shape (n_edge, 2), got %s" % (arr.shape,))
    # convert from 1‑based (+0 for missing) to 0‑based with –1 padding
    arr = np.where(arr > 0, arr - 1, -1)
    return arr


'''
Connectivities by MPI mesh generator:
- C2V: "vertex_of_cell"
- C2E: "edge_of_cell"
- V2C: "cells_of_vertex"
- V2E: "edges_of_vertex"
- E2C: "adjacent_cell_of_edge"
- E2V: "edge_vertices"
- V2C2V / V2E2V: "vertices_of_vertex"
Domain size: 
  nx = 10
  ny = 12
'''

def _read_v2e(ds):
    raw = _first_present(
        ds,
        ["V2E", "vertex_edges", "nodes2edges", "node_edge_connectivity", "edges_of_vertex"],
        required=False,
    )
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.int32)
    # input may already be (n_vertex, max_deg); if not, transpose
    if arr.ndim != 2:
        raise ValueError("v2e dataset must be 2‑D")
    if arr.shape[1] != 6:
        # try transpose – common case is (max_deg, n_vertex)
        arr = arr.T
    if arr.shape[1] != 6:
        raise ValueError("v2e must have shape (n_vertex, max_deg), got %s" % (arr.shape,))
    # convert 1‑based → 0‑based, keep –1 for padding
    arr = np.where(arr > 0, arr - 1, -1)
    return arr

def print_structured_vertex_edges(v, m, S_field):
    i, j = m.vertex_to_ij[v]
    ni, nj, _ = m.ijk_to_edge.shape
    print(f"Structured: Vertex {v} maps to (i={i}, j={j})")
    # Kolor 0: edge at (i, j, 0) and (i, j-1, 0) if j > 0
    if j < nj - 1:
        edge_idx = m.ijk_to_edge[i, j, 0]
        if edge_idx >= 0:
            print(f"  Kolor 0 (east): edge_idx={edge_idx}, value={S_field[edge_idx]}, connects ({i},{j}) <-> ({i},{j+1})")
    if j > 0:
        edge_idx = m.ijk_to_edge[i, j-1, 0]
        if edge_idx >= 0:
            print(f"  Kolor 0 (west): edge_idx={edge_idx}, value={S_field[edge_idx]}, connects ({i},{j-1}) <-> ({i},{j})")
    # Kolor 1: edge at (i, j, 1) and (i-1, j, 1) if i > 0
    if i < ni - 1:
        edge_idx = m.ijk_to_edge[i, j, 1]
        if edge_idx >= 0:
            print(f"  Kolor 1 (NE): edge_idx={edge_idx}, value={S_field[edge_idx]}, connects ({i},{j}) <-> ({i+1},{j})")
    if i > 0:
        edge_idx = m.ijk_to_edge[i-1, j, 1]
        if edge_idx >= 0:
            print(f"  Kolor 1 (SW): edge_idx={edge_idx}, value={S_field[edge_idx]}, connects ({i-1},{j}) <-> ({i},{j})")
    # Kolor 2: edge at (i, j, 2) and (i-1, j+1, 2) if i > 0 and j < nj-1
    if i < ni - 1 and j > 0:
        edge_idx = m.ijk_to_edge[i, j-1, 2]
        if edge_idx >= 0:
            print(f"  Kolor 2 (NW): edge_idx={edge_idx}, value={S_field[edge_idx]}, connects ({i},{j}) <-> ({i+1},{j-1})")
    if i > 0 and j < nj - 1:
        edge_idx = m.ijk_to_edge[i-1, j, 2]
        if edge_idx >= 0:
            print(f"  Kolor 2 (SE): edge_idx={edge_idx}, value={S_field[edge_idx]}, connects ({i-1},{j+1}) <-> ({i},{j})")



@pytest.mark.requires_atlas
def test_structured_bridge_matches_unstructured(exec_alloc_descriptor):
    mesh_nc = os.environ.get("GT4PY_TRANSLATOR_MESH") or "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"
    # print(f"Testing structured/unstructured pnabla comparison with mesh_nc={mesh_nc}")
    if mesh_nc:
        xr = pytest.importorskip("xarray")
        with xr.open_dataset(mesh_nc) as ds:
            # print(ds)
            # print(ds.data_vars)
            # print(ds.attrs)
            # print(ds.variables)
            e2v = _read_e2v(ds)
            # print(f"Read e2v with shape {e2v.shape} from dataset")
            v2e = _read_v2e(ds)
            # print(f"Read v2e with shape {v2e.shape if v2e is not None else None} from dataset")
            lonlat = _get_lonlat(ds)
            # print(f"Read lonlat with shape {lonlat.shape if lonlat is not None else None} from dataset")
            # print(f"Lonlat: {lonlat}")
            dual_volumes = _first_present(
                ds,
                ["dual_volumes", "dual_area", "node_volume", "vertex_area"],
                required=False,
            )
            dual_normals = _get_dual_normals(ds)

            # try regular parallelogram mapping first, fallback to ragged
            try:
                m = build_index_map_from_ds_regular(ds, e2v)
            except ValueError:
                m = build_index_map_for_ragged_lonlat_e2v(lonlat, e2v)
    else:
        setup = nabla_setup(allocator=exec_alloc_descriptor.allocator)
        
        e2v = atlas_utils.AtlasTable(
            setup.mesh.edges.node_connectivity
        ).asnumpy()
        v2e = atlas_utils.AtlasTable(
            setup.mesh.nodes.edge_connectivity
        ).asnumpy()
        lonlat = None
        dual_volumes = None
        dual_normals = None

    setup = nabla_setup.from_connectivity(
        allocator=exec_alloc_descriptor.allocator,
        e2v=e2v,
        v2e=v2e,
        lonlat_deg=lonlat,
        dual_normals=dual_normals,
        dual_volumes=dual_volumes,
    )
    # Fill all fields with ones for debugging
    setup.input_field.asnumpy()[:] = 1.0
    for S in setup.S_fields:
        S.asnumpy()[:] = 1.0
    setup.sign_field.asnumpy()[:] = 1.0
    setup.vol_field.asnumpy()[:] = 1.0

    # Build map
    if setup.mesh is not None:
        # atlas‑generated mesh case
        m = build_index_map_from_atlas_setup(setup)
    else:
        # file‑based mesh: use the lon/lat+e2v arrays we already have
        if lonlat is None:
            raise RuntimeError("need lonlat info to build structured index map")
        m = build_index_map_from_lonlat_e2v(lonlat, e2v)
    
    # Unstructured reference
    ref0 = gtx.zeros({setup.input_field.domain.dims[0]: setup.nodes_size}, allocator=exec_alloc_descriptor.allocator)
    ref1 = gtx.zeros({setup.input_field.domain.dims[0]: setup.nodes_size}, allocator=exec_alloc_descriptor.allocator)
    # print(f"backend: ", exec_alloc_descriptor)
    print(f"input field: ", setup.input_field)
    pnabla.with_backend(None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor)(
        setup.input_field,
        setup.S_fields,
        setup.sign_field,
        setup.vol_field,
        out=(ref0, ref1),
        offset_provider={"E2V": setup.edges2node_connectivity, "V2E": setup.nodes2edge_connectivity},
    )

    # 3) Build structured sign from unstructured sign + V2E
    sign_np = setup.sign_field.asnumpy()
    if setup.mesh is not None:
        v2e_np = atlas_utils.AtlasTable(setup.mesh.nodes.edge_connectivity).asnumpy()
    else:
        v2e_np = setup.nodes2edge_connectivity.asnumpy()
    sign_struct = build_structured_sign_from_unstructured(sign_np, v2e_np, m)

    print("input field:", setup.input_field.asnumpy())
    # 4) Run structured kernel through translator
    out0_u, out1_u = run_structured_pnabla_from_unstructured(
        pp_vertex=setup.input_field.asnumpy(),
        S_M_edges_3=(setup.S_fields[0].asnumpy(), setup.S_fields[1].asnumpy()),
        sign_struct=sign_struct,
        vol_vertex=setup.vol_field.asnumpy(),
        m=m,
        backend=None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor,
    )
    for v in range(setup.nodes_size):

        # --- Unstructured approach ---
        # Get the edges connected to vertex v via V2E
        v2e = setup.nodes2edge_connectivity.asnumpy()  # shape: (n_vertex, max_deg)
        connected_edges = v2e[v]
        # Filter out invalid entries (usually -1)
        connected_edges = connected_edges[connected_edges >= 0]
        print(f"Unstructured: Vertex {v} is connected to edges {connected_edges}")

        # For each connected edge, print its endpoints and value
        e2v = setup.edges2node_connectivity.asnumpy()  # shape: (n_edge, 2)
        for e in connected_edges:
            verts = e2v[e]
            print(f"  Edge {e}: connects vertices {verts}, value = {setup.S_fields[0].asnumpy()[e]}")

        # --- Structured approach ---
        # Get (i, j) for this vertex
        print_structured_vertex_edges(v, m, setup.S_fields[0].asnumpy())
        # i, j = m.vertex_to_ij[v]
        # print(f"Structured: Vertex {v} maps to (i={i}, j={j})")

        # # For each Kolor (edge orientation), print the corresponding edge index and value
        # for k in range(3):  # assuming 3 edge orientations
        #     edge_idx = m.ijk_to_edge[i, j, k]
        #     if edge_idx >= 0:
        #         print(f"  Kolor {k}: edge_idx={edge_idx}, value={setup.S_fields[0].asnumpy()[edge_idx]}")
        #     else:
        #         print(f"  Kolor {k}: no edge (padding or boundary)")

    print("Structured output:", out0_u, out1_u)
    print("Unstructured reference:", ref0.asnumpy(), ref1.asnumpy())
    np.testing.assert_allclose(out0_u, ref0.asnumpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(out1_u, ref1.asnumpy(), rtol=1e-10, atol=1e-10)