import os
import numpy as np
import pytest
import xarray as xr

from gt4py.next.modules import translator as tr

def _read_e2v(ds):
    raw = None
    for name in ["E2V", "edge_vertices", "edges2nodes", "edge_node_connectivity"]:
        if name in ds:
            raw = ds[name].values
            break
    if raw is None:
        raise KeyError("No E2V found")
    arr = np.asarray(raw, dtype=np.int32)
    if arr.ndim == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("e2v must have shape (n_edge,2)")
    # convert 1-based -> 0-based, keep 0 -> -1
    arr = np.where(arr > 0, arr - 1, -1)
    return arr

def _get_lonlat(ds):
    if "longitude_vertices" in ds and "latitude_vertices" in ds:
        lon = ds["longitude_vertices"].values.astype(np.float64)
        lat = ds["latitude_vertices"].values.astype(np.float64)
        return np.stack([lon, lat], axis=1)
    for name in ["lonlat", "vertex_lonlat", "node_lonlat"]:
        if name in ds:
            return ds[name].values
    return None

@pytest.mark.skipif(not os.path.exists("/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"),
                    reason="mesh not found")
def test_translator_pack_unpack_roundtrip():
    mesh_nc = "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"
    with xr.open_dataset(mesh_nc) as ds:
        e2v = _read_e2v(ds)
        lonlat = _get_lonlat(ds)

    # build index map: try regular then ragged
    try:
        m = tr.build_index_map_from_lonlat_e2v(lonlat, e2v)
    except Exception:
        m = tr.build_index_map_for_ragged_lonlat_e2v(lonlat, e2v)

    n_vertex = int(m.vertex_to_ij.shape[0])
    n_edge = int(m.edge_to_ijk.shape[0])

    # make deterministic test data
    v_values = np.arange(n_vertex, dtype=np.float64)
    e_values = np.arange(n_edge, dtype=np.float64)

    # vertex roundtrip
    v_struct = tr.pack_vertex_field_to_structured(v_values, m)
    print("Structured vertex field (packed):", v_struct[...,0])
    v_un = tr.unpack_vertex_field_to_unstructured(v_struct, m)
    print("Unstructured vertex field (unpacked):", v_un)
    assert v_un.shape[0] == n_vertex
    assert np.allclose(v_un, v_values)

    # edge roundtrip: pack then reconstruct edges from m.ijk_to_edge
    e_struct = tr.pack_edge_field_to_structured(e_values, m)  # shape (ni,max_nj,3)
    # edge roundtrip: for each edge, check pack/unpack consistency
    for e in range(n_edge):
        i, j, k = m.edge_to_ijk[e]
        if i < 0 or j < 0 or k < 0:
            continue  # skip unmapped
        assert e_struct[i, j, k] == e_values[e], f"Edge {e}: got {e_struct[i, j, k]}, expected {e_values[e]}"

    print("translator pack/unpack roundtrip passed")