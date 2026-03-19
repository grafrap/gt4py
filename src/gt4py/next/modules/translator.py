from dataclasses import dataclass
import numpy as np
import gt4py.next as gtx
from gt4py.next.iterator import atlas_utils

from .ffront_fvm_nabla_structured import IDim, JDim, Kolor#, pnabla_cartesian
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


@dataclass(frozen=True)
class StructuredRemapSizes:
    nx: int
    ny: int
    max_i: int
    max_j: int
    vertex_size: int
    edge_size_padded: int
    cell_size: int

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

def infer_structured_remap_sizes(
    *,
    domain_length: float,
    mean_edge_length: float,
    n_cells: int,
) -> StructuredRemapSizes:
    if mean_edge_length <= 0:
        raise ValueError("mean_edge_length must be > 0.")

    nx = int(np.int32(domain_length / mean_edge_length))
    if nx <= 0:
        raise ValueError(
            f"Invalid nx={nx} inferred from domain_length={domain_length} and mean_edge_length={mean_edge_length}."
        )

    ny = int(np.int32(n_cells / (2 * nx)))
    if ny <= 0:
        raise ValueError(f"Invalid ny={ny} inferred from n_cells={n_cells} and nx={nx}.")

    max_j = nx + 1
    max_i = ny + 1

    vertex_size = max_i * max_j
    edge_size_padded = 3 * max_i * max_j
    cell_size = 2 * nx * ny

    return StructuredRemapSizes(
        nx=nx,
        ny=ny,
        max_i=max_i,
        max_j=max_j,
        vertex_size=vertex_size,
        edge_size_padded=edge_size_padded,
        cell_size=cell_size,
    )


def load_structured_remap_sizes_from_netcdf(nc_path: str) -> StructuredRemapSizes:
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        if "domain_length" not in ds.attrs or "mean_edge_length" not in ds.attrs:
            raise KeyError(
                "Dataset must contain attributes 'domain_length' and 'mean_edge_length'."
            )
        if "cell" not in ds.sizes:
            raise KeyError("Dataset must contain dimension 'cell'.")

        sizes = infer_structured_remap_sizes(
            domain_length=float(ds.attrs["domain_length"]),
            mean_edge_length=float(ds.attrs["mean_edge_length"]),
            n_cells=int(ds.sizes["cell"]),
        )

    return sizes

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
    # print("m.ijk_to_edge: ", m.ijk_to_edge)
    return out

def unpack_vertex_field_to_unstructured(struct_values: np.ndarray, m: IndexMap) -> np.ndarray:
    n_vertex = m.vertex_to_ij.shape[0]
    out = np.zeros((n_vertex,), dtype=struct_values.dtype)
    for v in range(n_vertex):
        i, local_j = int(m.vertex_to_ij[v, 0]), int(m.vertex_to_ij[v, 1])
        out[v] = struct_values[i, local_j, 0]
    return out

import numpy as np
import gt4py.next as gtx
# from icon4py.model.common import dimension as dims

# Structured Dimensions
IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
Kolor = gtx.Dimension("Kolor")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)

def pack_edge_field(edge_values: np.ndarray, m: 'IndexMap') -> np.ndarray:
    """Packs 1D or 2D unstructured edge fields into structured [I, J, Kolor, (K)]."""
    ni, max_nj, n_kolor = m.ijk_to_edge.shape
    has_k = edge_values.ndim == 2
    
    if has_k:
        nk = edge_values.shape[1]
        out = np.zeros((ni, max_nj, n_kolor, nk), dtype=edge_values.dtype)
    else:
        out = np.zeros((ni, max_nj, n_kolor), dtype=edge_values.dtype)
        
    for i in range(ni):
        for j in range(max_nj):
            for k in range(n_kolor):
                e = m.ijk_to_edge[i, j, k]
                if e >= 0:
                    if has_k:
                        out[i, j, k, :] = edge_values[e, :]
                    else:
                        out[i, j, k] = edge_values[e]
    return out

def unpack_edge_field(struct_values: np.ndarray, m: 'IndexMap', n_edge: int) -> np.ndarray:
    """Unpacks structured [I, J, Kolor, (K)] fields back to unstructured."""
    has_k = struct_values.ndim == 4
    
    if has_k:
        nk = struct_values.shape[3]
        out = np.zeros((n_edge, nk), dtype=struct_values.dtype)
    else:
        out = np.zeros((n_edge,), dtype=struct_values.dtype)
        
    ni, max_nj, n_kolor = m.ijk_to_edge.shape
    for i in range(ni):
        for j in range(max_nj):
            for k in range(n_kolor):
                e = m.ijk_to_edge[i, j, k]
                if e >= 0:
                    if has_k:
                        out[e, :] = struct_values[i, j, k, :]
                    else:
                        out[e] = struct_values[i, j, k]
    return out

def pack_vertex_field(vertex_values: np.ndarray, m) -> np.ndarray:
    """Packs an unstructured vertex field into [IDim, JDim, Kolor=1, (KDim)]."""
    has_k = vertex_values.ndim == 2
    ni, nj = m.ij_to_vertex.shape
    
    # Allocate with Kolor dimension of size 1
    out = np.zeros((ni, nj, 1, vertex_values.shape[1] if has_k else 1), dtype=vertex_values.dtype)
    for i in range(ni):
        for j in range(nj):
            v = m.ij_to_vertex[i, j]
            if v >= 0:
                # Place data exactly at Kolor index 0
                out[i, j, 0, :] = vertex_values[v, :] if has_k else vertex_values[v]
    return out if has_k else out[:, :, :, 0]

# --- Cartesian Cell Helpers ---

def build_cell_to_ijk(m: IndexMap, ds) -> np.ndarray:
    """Maps unstructured 1D cell index from netcdf into Cartesian [I, J, Kolor] layout."""
    import numpy as np
    c2v = np.where(
        ds["vertex_of_cell"].transpose("cell", "nv").values.astype(np.int32) > 0,
        ds["vertex_of_cell"].transpose("cell", "nv").values.astype(np.int32) - 1, -1
    )
    n_cells = c2v.shape[0]
    ni, nj = m.ij_to_vertex.shape
    
    ijk_to_cell = np.full((ni, nj, 2), -1, dtype=np.int32)
    
    for c in range(n_cells):
        v = c2v[c]
        if np.any(v < 0): continue
        
        i_coords = [m.vertex_to_ij[v[0], 0], m.vertex_to_ij[v[1], 0], m.vertex_to_ij[v[2], 0]]
        j_coords = [m.vertex_to_ij[v[0], 1], m.vertex_to_ij[v[1], 1], m.vertex_to_ij[v[2], 1]]
        
        if any(i < 0 for i in i_coords): continue
        
        i_min = min(i_coords)
        j_min = min(j_coords)
        
        # Kolor 0 has 2 vertices at i_min, Kolor 1 has 1 vertex at i_min
        kolor = 0 if i_coords.count(i_min) == 2 else 1
        
        if 0 <= i_min < ni and 0 <= j_min < nj:
            ijk_to_cell[i_min, j_min, kolor] = c
            
    return ijk_to_cell

def pack_cell_field(cell_values: np.ndarray, ijk_to_cell: np.ndarray) -> np.ndarray:
    """Packs 1D/2D unstructured Cell arrays into [IDim, JDim, Kolor, (KDim)]."""
    import numpy as np
    ni, nj, n_kolor = ijk_to_cell.shape
    has_k = cell_values.ndim == 2
    out = np.zeros((ni, nj, n_kolor, cell_values.shape[1] if has_k else 1), dtype=cell_values.dtype)
        
    for i in range(ni):
        for j in range(nj):
            for k in range(n_kolor):
                c = ijk_to_cell[i, j, k]
                if c >= 0:
                    out[i, j, k, :] = cell_values[c, :] if has_k else cell_values[c]
    return out if has_k else out[:, :, :, 0]

def unpack_cell_field(struct_values: np.ndarray, ijk_to_cell: np.ndarray, n_cells: int) -> np.ndarray:
    """Unpacks [IDim, JDim, Kolor, (KDim)] Cell arrays back to unstructured."""
    import numpy as np
    has_k = struct_values.ndim == 4
    out = np.zeros((n_cells, struct_values.shape[3] if has_k else 1), dtype=struct_values.dtype)
        
    ni, nj, n_kolor = ijk_to_cell.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(n_kolor):
                c = ijk_to_cell[i, j, k]
                if c >= 0:
                    out[c, :] = struct_values[i, j, k, :] if has_k else struct_values[i, j, k]
    return out if has_k else out[:, 0]

def build_c2e2co_unstructured(ijk_to_cell: np.ndarray, n_cells: int) -> np.ndarray:
    """Uses Cartesian topology to dynamically build the exact C2E2CO connectivity map."""
    import numpy as np
    ni, nj, _ = ijk_to_cell.shape
    c2e2co = np.full((n_cells, 3), -1, dtype=np.int32)
    for i in range(ni):
        for j in range(nj):
            c0, c1 = ijk_to_cell[i, j, 0], ijk_to_cell[i, j, 1]
            
            if c0 >= 0:
                n0_0 = ijk_to_cell[i, j, 1] if j>=0 else -1
                n0_1 = ijk_to_cell[i, j-1, 1] if j-1>=0 else -1
                n0_2 = ijk_to_cell[i-1, j, 1] if i-1>=0 else -1
                c2e2co[c0] = [n0_0, n0_1, n0_2]
                    
            if c1 >= 0:
                n1_0 = ijk_to_cell[i, j, 0] if j>=0 else -1
                n1_1 = ijk_to_cell[i, j+1, 0] if j+1<nj else -1
                n1_2 = ijk_to_cell[i+1, j, 0] if i+1<ni else -1
                c2e2co[c1] = [n1_0, n1_1, n1_2]
    return c2e2co

def pack_c2e2co_field(field_np: np.ndarray, ijk_to_cell: np.ndarray) -> tuple[np.ndarray, ...]:
    """Packs C2E2CO neighbour lookup tables into a tuple of 3 [IDim, JDim, Kolor] fields."""
    import numpy as np
    ni, nj, _ = ijk_to_cell.shape
    n_neighbors = field_np.shape[1]
    out_s = tuple(np.zeros((ni, nj, 2), dtype=field_np.dtype) for _ in range(n_neighbors))
    
    for i in range(ni):
        for j in range(nj):
            c0, c1 = ijk_to_cell[i, j, 0], ijk_to_cell[i, j, 1]
            
            if c0 >= 0:
                n0_0 = ijk_to_cell[i, j, 1]
                n0_1 = ijk_to_cell[i, j-1, 1] if j > 0 else -1
                n0_2 = ijk_to_cell[i-1, j, 1] if i > 0 else -1
                neighbors0 = [n0_0, n0_1, n0_2]
                for idx in range(n_neighbors): 
                    if neighbors0[idx] != -1:  # <-- THE FIX: Force 0.0 for out-of-bounds!
                        out_s[idx][i, j, 0] = field_np[c0, idx]
            
            if c1 >= 0:
                n1_0 = ijk_to_cell[i, j, 0]
                n1_1 = ijk_to_cell[i, j+1, 0] if j+1 < nj else -1
                n1_2 = ijk_to_cell[i+1, j, 0] if i+1 < ni else -1
                neighbors1 = [n1_0, n1_1, n1_2]
                for idx in range(n_neighbors): 
                    if neighbors1[idx] != -1:  # <-- THE FIX: Force 0.0 for out-of-bounds!
                        out_s[idx][i, j, 1] = field_np[c1, idx]
    return out_s

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
        # print(f"Edge {e}: vertices {v0}-{v1} -> (i0,j0)=({i0},{j0}), (i1,j1)=({i1},{j1})")
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
        # print(f"ijk_to_edge: ", ijk_to_edge[i, j, k], " at (i,j,k)=", (i, j, k))

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
    sizes = infer_structured_remap_sizes(
        domain_length=float(ds.attrs["domain_length"]),
        mean_edge_length=float(ds.attrs["mean_edge_length"]),
        n_cells=int(ds.sizes["cell"]),
    )
    expected_nodes = sizes.vertex_size

    # read lonlat from ICON-style names used in your file
    lon = ds["longitude_vertices"].values.astype(np.float64)
    lat = ds["latitude_vertices"].values.astype(np.float64)
    lonlat = np.stack([lon, lat], axis=1)

    if lonlat.shape[0] != expected_nodes:
        raise ValueError(
            f"Dataset is not regular: expected {expected_nodes} nodes, got {lonlat.shape[0]}"
        )

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
    sign_vertex_v2e: np.ndarray,
    nodes2edge: np.ndarray,
    m: IndexMap,
) -> tuple[np.ndarray, ...]:
    ni, nj = m.ij_to_vertex.shape
    
    # Dynamically initialize a tuple of 6 fields
    signs = tuple(np.zeros((ni, nj, 1), dtype=sign_vertex_v2e.dtype) for _ in range(6))

    for v in range(m.vertex_to_ij.shape[0]):
        i, j = m.vertex_to_ij[v]
        if i < 0 or j < 0:
            continue
        for l in range(nodes2edge.shape[1]):
            e = int(nodes2edge[v, l])
            if e < 0:
                continue
            ie, je, ke = m.edge_to_ijk[e]
            if ke < 0:
                continue
            
            sign_val = sign_vertex_v2e[v, l]
            
            if ke == 0:
                if ie == i and je == j:
                    signs[0][i, j, 0] = sign_val # East
                elif ie == i and je == j - 1:
                    signs[3][i, j, 0] = sign_val # West
            elif ke == 1:
                if ie == i and je == j:
                    signs[1][i, j, 0] = sign_val # NE
                elif ie == i - 1 and je == j:
                    signs[4][i, j, 0] = sign_val # SW
            elif ke == 2:
                if ie == i and je == j - 1:
                    signs[2][i, j, 0] = sign_val # NW
                elif ie == i - 1 and je == j:
                    signs[5][i, j, 0] = sign_val # SE

    return signs

# def run_structured_pnabla_from_unstructured(
#     pp_vertex: np.ndarray,
#     S_M_edges_3: tuple[np.ndarray, np.ndarray],
#     sign_struct: tuple[np.ndarray, ...], 
#     vol_vertex: np.ndarray,
#     m: IndexMap,
#     backend,
# ) -> tuple[np.ndarray, np.ndarray]:
#     pp_s = pack_vertex_field_to_structured(pp_vertex, m)
#     sm0_s = pack_edge_field_to_structured(S_M_edges_3[0], m)
#     sm1_s = pack_edge_field_to_structured(S_M_edges_3[1], m)
#     vol_s = pack_vertex_field_to_structured(vol_vertex, m)
    
#     # Cast the entire tuple of numpy arrays into gt4py fields in one go!
#     sign_f = tuple(gtx.as_field([IDim, JDim, Kolor], s) for s in sign_struct)

#     pp_f = gtx.as_field([IDim, JDim, Kolor], pp_s)
#     sm0_f = gtx.as_field([IDim, JDim, Kolor], sm0_s)
#     sm1_f = gtx.as_field([IDim, JDim, Kolor], sm1_s)
#     vol_f = gtx.as_field([IDim, JDim, Kolor], vol_s)

#     out0 = gtx.as_field([IDim, JDim, Kolor], np.zeros_like(pp_s))
#     out1 = gtx.as_field([IDim, JDim, Kolor], np.zeros_like(pp_s))

#     ni, nj = m.ij_to_vertex.shape

#     prog = setup_program(
#         pnabla_cartesian,
#         backend=backend,
#         horizontal_sizes={
#             "domain_max_i": gtx.int32(ni),
#             "domain_max_j": gtx.int32(nj),
#             "domain_max_kolor": gtx.int32(1),
#         },
#     )

#     prog(pp=pp_f, S_M=sm0_f, sign=sign_f, vol=vol_f, out=out0, offset_provider={})
#     prog(pp=pp_f, S_M=sm1_f, sign=sign_f, vol=vol_f, out=out1, offset_provider={})

#     u0 = unpack_vertex_field_to_unstructured(out0.asnumpy(), m)
#     u1 = unpack_vertex_field_to_unstructured(out1.asnumpy(), m)
#     return u0, u1