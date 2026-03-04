from dataclasses import dataclass
import numpy as np
import gt4py.next as gtx
from gt4py.next.iterator import atlas_utils

from .ffront_fvm_nabla_structured import IDim, JDim, Kolor, pnabla_cartesian
from gt4py.next.program_processors.program_setup_utils import setup_program


from typing import List

@dataclass(frozen=True)
class IndexMap:
    vertex_to_ij: np.ndarray      # (n_vertex, 2) -> (row_i, local_j)
    row_lengths: np.ndarray       # (ni,) lengths per row
    row_offsets: np.ndarray       # (ni,) cumulative offsets
    ij_to_vertex: np.ndarray      # (ni, max_nj) ragged padded with -1
    edge_to_ijk: np.ndarray       # (n_edge, 3) -> (i, j, k) j is local_j
    ijk_to_edge: np.ndarray       # (ni, max_nj, 3) padded with -1

def build_index_map_for_ragged_lonlat_e2v(
    lonlat_deg: np.ndarray,
    e2v: np.ndarray,
    decimals: int = 10,
) -> IndexMap:
    lonlat = np.asarray(lonlat_deg, dtype=np.float64)
    e2v_np = np.asarray(e2v, dtype=np.int32)
    if lonlat.ndim != 2 or lonlat.shape[1] < 2:
        raise ValueError("lonlat_deg must have shape (n_vertex, 2).")
    if e2v_np.ndim != 2 or e2v_np.shape[1] != 2:
        raise ValueError("e2v must have shape (n_edge, 2).")

    n_vertex = lonlat.shape[0]
    lon = lonlat[:, 0]
    lat = lonlat[:, 1]

    lat_r = np.round(lat.astype(np.float64), decimals=decimals)
    uniq_lat = np.unique(lat_r)
    ni = uniq_lat.size

    row_indices: List[np.ndarray] = []
    row_lengths = np.zeros((ni,), dtype=np.int32)
    for i, latv in enumerate(uniq_lat):
        mask = lat_r == latv
        idxs = np.nonzero(mask)[0]
        local_lon = np.round(lon[idxs].astype(np.float64), decimals=decimals)
        order = np.argsort(local_lon)
        row_idxs_sorted = idxs[order]
        row_indices.append(row_idxs_sorted)
        row_lengths[i] = row_idxs_sorted.size

    max_nj = int(row_lengths.max())
    row_offsets = np.zeros((ni,), dtype=np.int32)
    cum = 0
    for i in range(ni):
        row_offsets[i] = cum
        cum += row_lengths[i]

    vertex_to_ij = np.full((n_vertex, 2), -1, dtype=np.int32)
    ij_to_vertex = np.full((ni, max_nj), -1, dtype=np.int32)
    for i in range(ni):
        row = row_indices[i]
        for local_j, v in enumerate(row):
            vertex_to_ij[int(v), 0] = i
            vertex_to_ij[int(v), 1] = local_j
            ij_to_vertex[i, local_j] = int(v)

    n_edge = e2v_np.shape[0]
    edge_to_ijk = np.full((n_edge, 3), -1, dtype=np.int32)
    ijk_to_edge = np.full((ni, max_nj, 3), -1, dtype=np.int32)

    for e in range(n_edge):
        v0, v1 = int(e2v_np[e, 0]), int(e2v_np[e, 1])
        i0, j0 = int(vertex_to_ij[v0, 0]), int(vertex_to_ij[v0, 1])
        i1, j1 = int(vertex_to_ij[v1, 0]), int(vertex_to_ij[v1, 1])
        if i0 < 0 or i1 < 0:
            continue
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        if di == 0 and dj == 1:
            k = 0
            i = i0
            j = min(j0, j1)
        elif di == 1 and dj == 0:
            k = 1
            i = min(i0, i1)
            j = j0 if j0 >= 0 else j1
        elif di == 1 and dj == 1:
            k = 2
            i = min(i0, i1)
            j = min(j0, j1)
        else:
            continue
        if j < 0 or j >= max_nj:
            continue
        edge_to_ijk[e] = (i, j, k)
        if ijk_to_edge[i, j, k] == -1:
            ijk_to_edge[i, j, k] = e

    return IndexMap(
        vertex_to_ij=vertex_to_ij,
        row_lengths=row_lengths,
        row_offsets=row_offsets,
        ij_to_vertex=ij_to_vertex,
        edge_to_ijk=edge_to_ijk,
        ijk_to_edge=ijk_to_edge,
    )

def pack_vertex_field_to_structured(vertex_values: np.ndarray, m: IndexMap) -> np.ndarray:
    ni, max_nj = m.ij_to_vertex.shape
    out = np.zeros((ni, max_nj, 1), dtype=vertex_values.dtype)
    for v in range(vertex_values.shape[0]):
        i, local_j = int(m.vertex_to_ij[v, 0]), int(m.vertex_to_ij[v, 1])
        if i >= 0:
            out[i, local_j, 0] = vertex_values[v]
    return out

# def pack_edge_field_to_structured(edge_values: np.ndarray, m: IndexMap) -> np.ndarray:
#     ni, max_nj, _ = m.ijk_to_edge.shape
#     out = np.zeros((ni, max_nj, 3), dtype=edge_values.dtype)
#     valid = m.ijk_to_edge >= 0
#     out[valid] = edge_values[m.ijk_to_edge[valid]]
#     return out
def pack_edge_field_to_structured(edge_values: np.ndarray, m: IndexMap) -> np.ndarray:
    ni, max_nj, n_kolor = m.ijk_to_edge.shape
    out = np.zeros((ni, max_nj, n_kolor), dtype=edge_values.dtype)
    for i in range(ni):
        for j in range(max_nj):
            for k in range(n_kolor):
                e = m.ijk_to_edge[i, j, k]
                if e >= 0:
                    out[i, j, k] = edge_values[e]
    print("m.ijk_to_edge: ", m.ijk_to_edge)
    return out

def unpack_vertex_field_to_unstructured(struct_values: np.ndarray, m: IndexMap) -> np.ndarray:
    n_vertex = m.vertex_to_ij.shape[0]
    out = np.zeros((n_vertex,), dtype=struct_values.dtype)
    for v in range(n_vertex):
        i, local_j = int(m.vertex_to_ij[v, 0]), int(m.vertex_to_ij[v, 1])
        out[v] = struct_values[i, local_j, 0]
    return out


def run_structured_pnabla_from_unstructured(
    pp_vertex: np.ndarray,         # (n_vertex,)
    S_M_edges_3: tuple[np.ndarray, np.ndarray],  # each (n_edge,)
    sign_struct: np.ndarray,       # (ni, nj, 3)
    vol_vertex: np.ndarray,        # (n_vertex,)
    m: IndexMap,
    backend,
) -> tuple[np.ndarray, np.ndarray]:
    pp_s = pack_vertex_field_to_structured(pp_vertex, m)
    sm0_s = pack_edge_field_to_structured(S_M_edges_3[0], m)
    sm1_s = pack_edge_field_to_structured(S_M_edges_3[1], m)
    vol_s = pack_vertex_field_to_structured(vol_vertex, m)

    print("Structured pp:", pp_s[...,0])
    print("Structured S_M0:", sm0_s)
    print("Structured S_M1:", sm1_s)
    print("Structured sign:", sign_struct)
    print("Structured vol:", vol_s[...,0])

    pp_f = gtx.as_field([IDim, JDim, Kolor], pp_s)
    sm0_f = gtx.as_field([IDim, JDim, Kolor], sm0_s)
    sm1_f = gtx.as_field([IDim, JDim, Kolor], sm1_s)
    sign_f = gtx.as_field([IDim, JDim, Kolor], sign_struct)
    vol_f = gtx.as_field([IDim, JDim, Kolor], vol_s)

    out0 = gtx.as_field([IDim, JDim, Kolor], np.zeros_like(pp_s))
    out1 = gtx.as_field([IDim, JDim, Kolor], np.zeros_like(pp_s))

    ni, nj = m.ij_to_vertex.shape

    prog = setup_program(
        pnabla_cartesian,
        backend=backend,
        horizontal_sizes={
            "domain_max_i": gtx.int32(ni),
            "domain_max_j": gtx.int32(nj),
            "domain_max_kolor": gtx.int32(1),
        },
    )

    prog(pp=pp_f, S_M=sm0_f, sign=sign_f, vol=vol_f, out=out0, offset_provider={})
    prog(pp=pp_f, S_M=sm1_f, sign=sign_f, vol=vol_f, out=out1, offset_provider={})

    u0 = unpack_vertex_field_to_unstructured(out0.asnumpy(), m)
    u1 = unpack_vertex_field_to_unstructured(out1.asnumpy(), m)
    print("Unstructured output:", u0, u1)
    return u0, u1

from typing import Any


def _rounded_unique(vals: np.ndarray, decimals: int = 10) -> np.ndarray:
    return np.unique(np.round(vals.astype(np.float64), decimals=decimals))


def build_index_map_from_lonlat_e2v(
    lonlat_deg: np.ndarray,
    e2v: np.ndarray,
    nodes_size: int | None = None,
    edges_size: int | None = None,
    decimals: int = 10,
) -> IndexMap:
    """
    Build IndexMap from lon/lat coordinates and edge-to-vertex connectivity.

    Robustly handle grids where longitudes are row-shifted (parallelogram): group by
    rounded latitude to determine rows and use max row length as nj.
    """
    lonlat = np.asarray(lonlat_deg)
    if lonlat.ndim != 2 or lonlat.shape[1] < 2:
        raise ValueError("lonlat_deg must have shape (n_vertex, 2).")

    e2v_np = np.asarray(e2v)
    if e2v_np.ndim != 2 or e2v_np.shape[1] != 2:
        raise ValueError("e2v must have shape (n_edge, 2).")

    n_vertex = int(nodes_size if nodes_size is not None else lonlat.shape[0])
    n_edge = int(edges_size if edges_size is not None else e2v_np.shape[0])

    lon = lonlat[:n_vertex, 0].astype(np.float64)
    lat = lonlat[:n_vertex, 1].astype(np.float64)

    # print(lon)

    # Group vertices by rounded latitude to form rows (handles shifted longitudes)
    lat_r = np.round(lat, decimals=decimals)
    uniq_lat = np.unique(lat_r)
    # print(f"Unique rounded latitudes (decimals={decimals}): {uniq_lat}")

    ni = uniq_lat.size

    row_indices = []
    row_lengths = np.zeros((ni,), dtype=np.int32)
    for i, latv in enumerate(uniq_lat):
        mask = lat_r == latv
        idxs = np.nonzero(mask)[0]
        # sort by longitude within the row
        local_lon = np.round(lon[idxs], decimals=decimals)
        order = np.argsort(local_lon)
        row_sorted = idxs[order]
        row_indices.append(row_sorted)
        row_lengths[i] = row_sorted.size

    max_nj = int(row_lengths.max())

    # sanity check: if fully regular parallelogram, ni * max_nj should equal vertex count
    if ni * max_nj != n_vertex:
        raise ValueError(
            f"Cannot build regular (i,j) map from lonlat+e2v by row grouping: "
            f"ni*max_nj={ni * max_nj} != nodes_size={n_vertex}. Grid may be irregular."
        )

    # row offsets (cumulative)
    row_offsets = np.zeros((ni,), dtype=np.int32)
    cum = 0
    for i in range(ni):
        row_offsets[i] = cum
        cum += int(row_lengths[i])

    # Build mappings (vertex -> (row, local_j)) and padded ij->vertex
    vertex_to_ij = np.full((n_vertex, 2), -1, dtype=np.int32)
    ij_to_vertex = np.full((ni, max_nj), -1, dtype=np.int32)
    for i in range(ni):
        row = row_indices[i]
        for local_j, v in enumerate(row):
            vertex_to_ij[int(v), 0] = i
            vertex_to_ij[int(v), 1] = local_j
            ij_to_vertex[i, local_j] = int(v)

    # Edge mapping similar to ragged builder but with local_j indices
    edge_to_ijk = np.full((n_edge, 3), -1, dtype=np.int32)
    ijk_to_edge = np.full((ni, max_nj, 3), -1, dtype=np.int32)

    for e in range(n_edge):
        v0, v1 = int(e2v_np[e, 0]), int(e2v_np[e, 1])
        i0, j0 = int(vertex_to_ij[v0, 0]), int(vertex_to_ij[v0, 1])
        i1, j1 = int(vertex_to_ij[v1, 0]), int(vertex_to_ij[v1, 1])
        print(f"Edge {e}: vertices {v0}-{v1} -> (i0,j0)=({i0},{j0}), (i1,j1)=({i1},{j1})")
        if i0 < 0 or i1 < 0:
            continue
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        if di == 0 and dj == 1:
            k = 0
            i = i0
            j = min(j0, j1)
        elif di == 1 and dj == 0:
            k = 1
            i = min(i0, i1)
            j = j0 if j0 >= 0 else j1
        elif di == 1 and dj == 1:
            k = 2
            i = min(i0, i1)
            j = min(j0, j1)
        else:
            continue
        if j < 0 or j >= max_nj:
            continue
        edge_to_ijk[e] = (i, j, k)
        if ijk_to_edge[i, j, k] == -1:
            ijk_to_edge[i, j, k] = e
        print(f"ijk_to_edge: ", ijk_to_edge[i, j, k], " at (i,j,k)=", (i, j, k))

    return IndexMap(
        vertex_to_ij=vertex_to_ij,
        row_lengths=row_lengths,
        row_offsets=row_offsets,
        ij_to_vertex=ij_to_vertex,
        edge_to_ijk=edge_to_ijk,
        ijk_to_edge=ijk_to_edge,
    )


def build_index_map_from_ds_regular(ds, e2v):
    """
    If dataset ds encodes a regular parallelogram grid (same as test_simple_structured),
    compute nx, ny and build the structured index map via lonlat + e2v.
    Returns IndexMap or raises ValueError if not regular.
    """
    import numpy as np

    nx = int(np.int32(ds.attrs["domain_length"] / ds.attrs["mean_edge_length"]))
    # test_simple_structured uses (ds.sizes["cell"]) / (2 * nx)
    ny = int(np.int32((ds.sizes["cell"]) / (2 * nx)))
    max_j = nx + 1
    max_i = ny + 1
    expected_nodes = int(max_i * max_j)

    # read lonlat from ICON-style names used in your file
    lon = ds["longitude_vertices"].values.astype(np.float64)
    lat = ds["latitude_vertices"].values.astype(np.float64)
    lonlat = np.stack([lon, lat], axis=1)

    if lonlat.shape[0] != expected_nodes:
        raise ValueError(f"Dataset is not regular: expected {expected_nodes} nodes, got {lonlat.shape[0]}")

    # e2v must be (n_edge,2) already converted to 0-based
    return build_index_map_from_lonlat_e2v(lonlat, e2v, nodes_size=expected_nodes)



def build_index_map_from_atlas_setup(setup: Any, decimals: int = 10) -> IndexMap:
    """
    Build IndexMap from Atlas mesh.
    Assumes nodes lie on a regular tensor-product lon/lat grid (no reduced rows).
    """
    lonlat = np.array(setup.mesh.nodes.field("lonlat"), copy=False)[: setup.nodes_size, :2]
    e2v = atlas_utils.AtlasTable(setup.mesh.edges.node_connectivity).asnumpy()
    return build_index_map_from_lonlat_e2v(
        lonlat_deg=lonlat,
        e2v=e2v,
        nodes_size=setup.nodes_size,
        edges_size=setup.edges_size,
        decimals=decimals,
    )


def build_structured_sign_from_unstructured(
    sign_vertex_v2e: np.ndarray,  # [n_vertex, edges_per_vertex]
    nodes2edge: np.ndarray,        # [n_vertex, edges_per_vertex], -1 padded
    m: IndexMap,
) -> np.ndarray:
    ni, nj = m.ij_to_vertex.shape
    out = np.zeros((ni, nj, 3), dtype=sign_vertex_v2e.dtype)

    for v in range(m.vertex_to_ij.shape[0]):
        i, j = m.vertex_to_ij[v]
        for l in range(nodes2edge.shape[1]):
            e = int(nodes2edge[v, l])
            if e < 0:
                continue
            ie, je, ke = m.edge_to_ijk[e]
            if ke < 0:
                continue
            # assign sign for the edge anchored at this node
            if ie == i and je == j:
                out[i, j, ke] = sign_vertex_v2e[v, l]

    return out
