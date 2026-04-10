from dataclasses import dataclass
import numpy as np
import gt4py.next as gtx
from gt4py.next.iterator import atlas_utils
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.map_dict import map_dict as _MAP_DICT

from .ffront_fvm_nabla_structured import IDim, JDim, Kolor#, pnabla_cartesian
from gt4py.next.program_processors.program_setup_utils import setup_program


from typing import List


def _parse_sparse_remap_table() -> dict[str, dict[int, dict[int, tuple[int, int, int]]]]:
    """Parse map_dict remaps into conn -> center_kolor -> slot -> (di, dj, neighbor_kolor)."""
    max_kolor = 3

    def _extract_shift(shifts_tuple: tuple) -> tuple[int, int, int]:
        vals: dict[str, int] = {}
        it = iter(shifts_tuple)
        for axis_lit, offset_lit in zip(it, it):
            vals[axis_lit.value] = int(offset_lit.value)
        return vals.get("IDim", 0), vals.get("JDim", 0), vals.get("Kolor", 0)

    def _kolor_range_from_domain(domain) -> tuple[int, int] | None:
        if domain is None:
            return None
        try:
            named_ranges = domain.args
            for nr in named_ranges:
                dim_arg = nr.args[0]
                if isinstance(dim_arg, ir.AxisLiteral) and dim_arg.value == "Kolor":
                    return int(nr.args[1].value), int(nr.args[2].value)
        except (AttributeError, IndexError, TypeError):
            return None
        return None

    table: dict[str, dict[int, dict[int, tuple[int, int, int]]]] = {}
    for (conn_lit, slot_lit), entry in _MAP_DICT.items():
        conn_name = conn_lit.value
        slot = int(slot_lit.value)
        table.setdefault(conn_name, {})

        if entry["kind"] == "shift":
            di, dj, dk = _extract_shift(entry["shifts"])
            for ck in range(max_kolor):
                table[conn_name].setdefault(ck, {})[slot] = (di, dj, ck + dk)
            continue

        if entry["kind"] == "concat_where":
            covered: set[int] = set()
            else_shift: tuple[int, int, int] | None = None
            resolved: list[tuple[tuple[int, int] | None, tuple[int, int, int]]] = []
            for domain, shifts in entry["branches"]:
                shift = _extract_shift(shifts)
                if domain is None:
                    else_shift = shift
                else:
                    resolved.append((_kolor_range_from_domain(domain), shift))

            for krange, shift in resolved:
                if krange is None:
                    continue
                for ck in range(krange[0], krange[1]):
                    covered.add(ck)
                    table[conn_name].setdefault(ck, {})[slot] = (shift[0], shift[1], ck + shift[2])

            if else_shift is not None:
                for ck in range(max_kolor):
                    if ck not in covered:
                        table[conn_name].setdefault(ck, {})[slot] = (
                            else_shift[0],
                            else_shift[1],
                            ck + else_shift[2],
                        )

    return table


_SPARSE_REMAP_TABLE: dict[str, dict[int, dict[int, tuple[int, int, int]]]] = _parse_sparse_remap_table()
_CENTER_ELEMENT_BY_PREFIX: dict[str, str] = {"E": "Edge", "C": "Cell", "V": "Vertex"}


def pack_sparse_local_field_to_structured(
    coeff: np.ndarray,
    connectivity: np.ndarray,
    index_map: "IndexMap",
    local_dim_name: str,
    *,
    cell_to_ijk: np.ndarray | None = None,
) -> np.ndarray:
    """Pack sparse-local coefficients into structured [I, J, Kolor, Local, ...]."""
    if coeff.ndim < 2:
        raise ValueError("Sparse-local coefficients must be at least 2-D [center, local, ...].")

    remap = _SPARSE_REMAP_TABLE.get(local_dim_name)
    if remap is None:
        raise KeyError(f"Unsupported sparse-local dimension '{local_dim_name}'.")

    conn = np.asarray(connectivity, dtype=np.int32)
    if conn.ndim != 2:
        raise ValueError("Sparse-local connectivity must be 2-D [center, local].")

    ni, nj, n_kolor = index_map.ijk_to_edge.shape
    max_i, max_j = index_map.ij_to_vertex.shape
    n_local = coeff.shape[1]
    tail_shape = coeff.shape[2:]

    if np.issubdtype(coeff.dtype, np.integer):
        out = np.full((ni, nj, n_kolor, n_local, *tail_shape), -1, dtype=coeff.dtype)
    else:
        out = np.zeros((ni, nj, n_kolor, n_local, *tail_shape), dtype=coeff.dtype)

    if local_dim_name == "E2C":
        n_elem = min(coeff.shape[0], index_map.edge_to_ijk.shape[0])
        n_local_eff = min(n_local, conn.shape[1])
        for elem in range(n_elem):
            ci, cj, ck = (int(v) for v in index_map.edge_to_ijk[elem])
            if ci < 0 or cj < 0 or ck < 0:
                continue
            for local in range(n_local_eff):
                out[ci, cj, ck, local, ...] = coeff[elem, local, ...]
        return out

    center_type = _CENTER_ELEMENT_BY_PREFIX.get(local_dim_name[0], "Edge")

    rel_to_slots: dict[int, dict[tuple[int, int, int], list[int]]] = {}
    for ck, slot_map in remap.items():
        r2s: dict[tuple[int, int, int], list[int]] = {}
        for slot, (di, dj, nk) in slot_map.items():
            r2s.setdefault((di, dj, nk), []).append(slot)
        rel_to_slots[int(ck)] = r2s

    def _normalize(delta: int, period: int) -> int:
        if period <= 0:
            return delta
        half = period // 2
        if delta > half:
            return delta - period
        if delta < -half:
            return delta + period
        return delta

    def _center_ijk(element_idx: int) -> tuple[int, int, int] | None:
        if center_type == "Edge":
            ijk = index_map.edge_to_ijk[element_idx]
            return int(ijk[0]), int(ijk[1]), int(ijk[2])
        if center_type == "Vertex":
            ij = index_map.vertex_to_ij[element_idx]
            return int(ij[0]), int(ij[1]), 0
        if center_type == "Cell":
            if cell_to_ijk is None or element_idx >= cell_to_ijk.shape[0]:
                return None
            ijk = cell_to_ijk[element_idx]
            return int(ijk[0]), int(ijk[1]), int(ijk[2])
        return None

    def _neighbor_ijk(neighbor_idx: int) -> tuple[int, int, int] | None:
        ntype = _CENTER_ELEMENT_BY_PREFIX.get(local_dim_name[-1], "Edge")
        if ntype == "Edge":
            ijk = index_map.edge_to_ijk[neighbor_idx]
            return int(ijk[0]), int(ijk[1]), int(ijk[2])
        if ntype == "Vertex":
            ij = index_map.vertex_to_ij[neighbor_idx]
            return int(ij[0]), int(ij[1]), 0
        if ntype == "Cell":
            if cell_to_ijk is None or neighbor_idx >= cell_to_ijk.shape[0]:
                return None
            ijk = cell_to_ijk[neighbor_idx]
            return int(ijk[0]), int(ijk[1]), int(ijk[2])
        return None

    n_elem = min(coeff.shape[0], conn.shape[0])
    for elem in range(n_elem):
        center = _center_ijk(elem)
        if center is None:
            continue
        ci, cj, ck = center
        if ck < 0:
            continue
        r2s = rel_to_slots.get(ck, {})
        for local in range(min(n_local, conn.shape[1])):
            neighbor_idx = int(conn[elem, local])
            if neighbor_idx < 0:
                continue
            nbr = _neighbor_ijk(neighbor_idx)
            if nbr is None:
                continue
            ni_, nj_, nk_ = nbr
            rel = (_normalize(ni_ - ci, max_i), _normalize(nj_ - cj, max_j), nk_)
            slots = r2s.get(rel)
            if slots is None:
                continue
            for slot in slots:
                if slot < n_local:
                    out[ci, cj, ck, slot, ...] = coeff[elem, local, ...]
    return out

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
    end_i: int
    end_j: int
    lateral: int = 0
    start_i: int = 0
    start_j: int = 0


def _first_present(ds, names: list[str], required: bool = True):
    for name in names:
        if name in ds:
            return ds[name]
    if required:
        raise KeyError(f"None of the dataset variables are present: {names}")
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

def infer_structured_remap_sizes(
    *,
    domain_length: float,
    mean_edge_length: float,
    n_cells: int,
    lateral: int = 0,
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
    start_i = lateral
    start_j = lateral
    end_i = max_i - lateral
    end_j = max_j - lateral

    return StructuredRemapSizes(
        nx=nx,
        ny=ny,
        max_i=max_i,
        max_j=max_j,
        vertex_size=vertex_size,
        edge_size_padded=edge_size_padded,
        cell_size=cell_size,
        lateral=lateral,
        start_i=start_i,
        start_j=start_j,
        end_i=end_i,
        end_j=end_j,
    )


def load_structured_remap_sizes_from_netcdf(nc_path: str, lateral=0) -> StructuredRemapSizes:
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
            lateral=lateral,
        )
        print(f"Inferred structured remap sizes from {nc_path}: {sizes}")

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
    trailing_shape = vertex_values.shape[1:]
    out = np.zeros((ni, max_nj, 1, *trailing_shape), dtype=vertex_values.dtype)
    for v in range(vertex_values.shape[0]):
        i, local_j = int(m.vertex_to_ij[v, 0]), int(m.vertex_to_ij[v, 1])
        if i >= 0:
            out[i, local_j, 0, ...] = vertex_values[v, ...]
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
    n_edge = edge_values.shape[0]
    for i in range(ni):
        for j in range(max_nj):
            for k in range(n_kolor):
                e = m.ijk_to_edge[i, j, k]
                if e < 0:
                    continue
                if e >= n_edge:
                    raise IndexError(
                        f"IndexMap edge id {e} at (i={i}, j={j}, kolor={k}) exceeds available edge axis {n_edge}. "
                        "Use an index map generated for the current grid."
                    )
                if 0 <= e < n_edge:
                    out[i, j, k] = edge_values[e]
    return out

def unpack_vertex_field_to_unstructured(struct_values: np.ndarray, m: IndexMap) -> np.ndarray:
    n_vertex = m.vertex_to_ij.shape[0]
    trailing_shape = struct_values.shape[3:]
    out = np.zeros((n_vertex, *trailing_shape), dtype=struct_values.dtype)
    for v in range(n_vertex):
        i, local_j = int(m.vertex_to_ij[v, 0]), int(m.vertex_to_ij[v, 1])
        out[v, ...] = struct_values[i, local_j, 0, ...]
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
    n_edge = edge_values.shape[0]
    
    if has_k:
        nk = edge_values.shape[1]
        out = np.zeros((ni, max_nj, n_kolor, nk), dtype=edge_values.dtype)
    else:
        out = np.zeros((ni, max_nj, n_kolor), dtype=edge_values.dtype)
        
    for i in range(ni):
        for j in range(max_nj):
            for k in range(n_kolor):
                e = m.ijk_to_edge[i, j, k]
                if e < 0:
                    continue
                if e >= n_edge:
                    raise IndexError(
                        f"IndexMap edge id {e} at (i={i}, j={j}, kolor={k}) exceeds available edge axis {n_edge}. "
                        "Use an index map generated for the current grid."
                    )
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
                if e < 0:
                    continue
                if e >= n_edge:
                    raise IndexError(
                        f"IndexMap edge id {e} at (i={i}, j={j}, kolor={k}) exceeds output edge axis {n_edge}. "
                        "Use an index map generated for the current grid."
                    )
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


def build_cell_ijk_maps(c2v: np.ndarray, m: IndexMap) -> tuple[np.ndarray, np.ndarray]:
    """Build both cell->(i,j,kolor) and (i,j,kolor)->cell lookup maps from C2V."""
    c2v_np = np.asarray(c2v, dtype=np.int32)
    n_cells = c2v_np.shape[0]
    ni, nj = m.ij_to_vertex.shape

    cell_to_ijk = np.full((n_cells, 3), -1, dtype=np.int32)
    ijk_to_cell = np.full((ni, nj, 2), -1, dtype=np.int32)

    for c in range(n_cells):
        v = c2v_np[c]
        if np.any(v < 0):
            continue

        i_coords = [int(m.vertex_to_ij[v[k], 0]) for k in range(3)]
        j_coords = [int(m.vertex_to_ij[v[k], 1]) for k in range(3)]
        if any(i < 0 for i in i_coords):
            continue

        i_min = min(i_coords)
        j_min = min(j_coords)
        kolor = 0 if i_coords.count(i_min) == 2 else 1

        if 0 <= i_min < ni and 0 <= j_min < nj:
            cell_to_ijk[c] = (i_min, j_min, kolor)
            ijk_to_cell[i_min, j_min, kolor] = c

    return cell_to_ijk, ijk_to_cell

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


def pack_cell_field_to_structured(
    cell_values: np.ndarray,
    cell_to_ijk: np.ndarray,
    ijk_to_cell: np.ndarray,
) -> np.ndarray:
    """Compatibility wrapper expected by cartesian_interceptor.

    Uses ijk_to_cell as authoritative mapping; cell_to_ijk is accepted for API compatibility.
    TODO: Fill this with the correct values
    """
    _ = cell_to_ijk
    return pack_cell_field(cell_values, ijk_to_cell)

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


def unpack_cell_field_from_structured(
    struct_values: np.ndarray,
    cell_to_ijk: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Compatibility API to unpack structured cell fields via cell_to_ijk map."""
    has_k = struct_values.ndim == 4
    out = np.zeros((n_cells, struct_values.shape[3] if has_k else 1), dtype=struct_values.dtype)

    for c in range(min(n_cells, cell_to_ijk.shape[0])):
        i, j, k = int(cell_to_ijk[c, 0]), int(cell_to_ijk[c, 1]), int(cell_to_ijk[c, 2])
        if i < 0 or j < 0 or k < 0:
            continue
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

def transform_to_unstructured(field: np.ndarray, nx: int, grid_obj: str = "Edge", boundary_level: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
        Transforms a field that is defined on a structured grid with a given boundary level into the classical icon format. 

        Parameters

        - field: the field to be transformed, defined on a structured grid with dimensions [Edge/Vertex/Cell, (KDim)]
        - grid_obj: the type of grid object the field is defined on ("Edge", "Vertex", "Cell")
        - boundary_level: the level of the boundary to be transformed (0 for the outermost boundary, 1 for the next one, etc.), this might differ for Edge and other grid objects, 
        - nx: the number of cells in the x direction, needed to determine the size of the structured grid and the mapping to the unstructured grid.        
        I.e. first loop around domain with all boundary levels, then classical structured layout in the interior. 

        Returns
        three arrays and an int: 
        - the transformation array (structured index -> unstructured index),  
        - the array for the back-transformation (unstructured index -> structured index),
        - the array with the values of the field in the unstructured layout
        - and the start index for the asked boundary level.
    """
    # length of the field in the structured layout
    N = field.shape[0]
    # create the transformation array (structured index -> unstructured index)
    transform_array = np.zeros(N, dtype=int)
    back_transform_array = np.zeros(N, dtype=int)
    # create the array with the values of the field in the unstructured layout
    unstructured_field = np.zeros(N, dtype=field.dtype)
    # fill the transformation array and the unstructured field
    start_at_b_level = 0
    idx = 0
    if grid_obj not in ["Edge", "Vertex", "Cell"]:
        raise ValueError(f"grid_obj must be one of 'Edge', 'Vertex', 'Cell', got {grid_obj}")
    if grid_obj == "Edge":
        # for edges, we have 3 boundary levels (0, 1, 2) corresponding to the 3 edge types in the structured grid
        n_levels = 5 # lateral 1 to 8 and two nudging levels, only write 5, because we need to fill two at the time to get the correct mapping for the interior edges
        ny = int((N - nx) / (3 * nx + 1))
        idx = 0
        kolor_1_start = nx * (ny + 1)
        kolor_2_start = (2 * nx * ny) + nx + ny
        complete_levels = 10
        needs_completion = False
        if min(nx, ny) < 2 * n_levels:
            n_levels = (np.ceil(min(nx, ny) / 2)).astype(int)
            complete_levels = min(complete_levels, min(nx, ny))
            print(f"Warning: Reduced number of boundary levels to {n_levels} due to small grid size (nx={nx}, ny={ny}).")
        for level in range(n_levels):
            # odd levels are single edge boundaries
            # cycle through the edges in the current boundary level, then fill the interior edges
            transform_array[idx:idx+nx - 2*level] = np.arange(0+level*(nx+1), nx+level*(nx-1), 1)  # south boundary edges
            idx += nx-2*level
            transform_array[idx:idx+ny - 2*level] = np.arange(kolor_1_start + nx + level * nx, kolor_2_start - level*(nx), nx+1) # east boundary edges
            idx += ny - 2*level
            transform_array[idx:idx+nx - 2*level] = np.arange(kolor_1_start-1 - level * (nx+1), kolor_1_start - 1 - nx - level * (nx - 1), -1)  # north boundary edges
            idx += nx - 2*level
            transform_array[idx:idx+ny - 2*level] = np.arange(kolor_2_start - 1 - nx - level * nx, kolor_1_start - 1 + level * (nx+2), -(nx+1)) # west boundary edges
            idx += ny - 2*level
            if boundary_level == 2 * (level + 1):
                start_at_b_level = idx
            if 2 * level + 1 == complete_levels:
                print(f"Reached complete level at {2*level+1}, filling remaining edges with interior mapping.")
                needs_completion = True
                break
            # even levels are more interior edges
            transform_array[idx:idx+nx-1- 2*level] = np.arange(kolor_1_start+1+level*(nx+2), kolor_1_start + nx + level*(nx), 1)
            idx += nx-1- 2*level
            transform_array[idx:idx+ny-1- 2*level] = np.arange(2*nx - 1 + level*(nx-1), kolor_1_start - 1 - level * (nx+1),  nx)
            idx += ny-1- 2*level
            transform_array[idx:idx+nx-1- 2*level] = np.arange(kolor_2_start - 2 - level * (nx+2), kolor_2_start - 2 - (nx-1) - level * nx, -1)
            idx += nx-1- 2*level
            transform_array[idx:idx+ny-1- 2*level] = np.arange(kolor_1_start - 2*nx - level * (nx-1), 0 + level * (nx+1), -nx)
            idx += ny-1- 2*level
            # kolor 2 edges
            transform_array[idx:idx+nx- 2*level] = np.arange(kolor_2_start + level * (nx+1), kolor_2_start + nx + level * (nx-1), 1)
            idx += nx- 2*level
            transform_array[idx:idx+ny-1- 2*level] = np.arange(kolor_2_start+2*nx-1 + level * (nx-1), N - level*(nx), nx)
            idx += ny-1- 2*level
            transform_array[idx:idx+nx-1- 2*level] = np.arange(N-2-level*(nx+1), N - nx - 1 - level * (nx-1), -1)
            idx += nx-1- 2*level
            transform_array[idx:idx+ny-2- 2*level] = np.arange(N - 2 * nx - level * (nx-1), kolor_2_start + level * (nx+1), -nx)
            idx += ny-2- 2*level
            if boundary_level == 2 * (level + 1) + 1:
                start_at_b_level = idx

    elif grid_obj == "Cell":
        n_levels = 5 # lateral 1 to 4 and one nudging levels
        ny = int(N/(2*nx))
        idx = 0
        kolor_1_start = nx * ny
        complete_levels = 5
        needs_completion = False
        if min(nx, ny) < 2 * n_levels:
            n_levels = (np.ceil(min(nx, ny) / 2)).astype(int)
            complete_levels = min(complete_levels, min(nx, ny)) // 2
            print(f"Warning: Reduced number of full boundary levels to {n_levels} due to small grid size (nx={nx}, ny={ny}).")
        for level in range(n_levels):
            if boundary_level == level + 1:
                start_at_b_level = idx
            # south up cells:
            transform_array[idx:idx+nx-1 - 2 * level] = np.arange(0+level * (nx+1), nx - 1 + level * (nx-1), 1)
            idx += nx - 1 - 2 * level
            # south down cells:
            transform_array[idx:idx+nx-1 - 2 * level] = np.arange(kolor_1_start + level * (nx+1), kolor_1_start + nx - 1 + level * (nx-1), 1)
            idx += nx - 1 - 2 * level
            # east up cells:
            transform_array[idx:idx+ny-1 - 2 * level] = np.arange(nx - 1 + level * (nx - 1), kolor_1_start - nx - level * (nx + 1), nx)
            idx += ny - 1 - 2 * level
            # east down cells:
            transform_array[idx:idx+ny-1 - 2 * level] = np.arange(kolor_1_start + nx - 1 + level * (nx - 1), N - nx - level * (nx + 1), nx)
            idx += ny - 1 - 2 * level
            # north up cells: 
            transform_array[idx:idx+nx-1 - 2 * level] = np.arange(kolor_1_start - 1 - level * (nx + 1), kolor_1_start - nx - level * (nx - 1), -1)
            idx += nx - 1 - 2 * level
            # north down cells:
            transform_array[idx:idx+nx-1 - 2 * level] = np.arange(N - 1 - level * (nx + 1), N - nx - level * (nx - 1), -1)
            idx += nx - 1 - 2 * level
            # west up cells:
            transform_array[idx:idx+ny-1 - 2 * level] = np.arange(kolor_1_start - nx - level * (nx - 1), 0 + level * (nx + 1), -nx)
            idx += ny - 1 - 2 * level
            # west down cells:
            transform_array[idx:idx+ny-1 - 2 * level] = np.arange(N - nx - level * (nx - 1), kolor_1_start + level * (nx + 1), -nx)
            idx += ny - 1 - 2 * level
            if level + 1 == complete_levels:
                print(f"Reached complete level at {level+1}, filling remaining cells ascending.")
                needs_completion = True
                break
        if boundary_level == n_levels + 1:
            start_at_b_level = idx


    elif grid_obj == "Vertex":
        # for vertices, we have the same boundary levels as for the cells, but only one kolor type. 
        # it can be filled the same way as the cell mapping, but with a total size of nx +1 for nx and ny+1 for ny
        n_levels = 5 # lateral 1 to 4 and one nudging levels
        ny = int((N) / (nx + 1) - 1)
        idx = 0
        complete_levels = 5
        needs_completion = False
        if min(nx, ny) < 2 * n_levels:
            n_levels = (np.ceil(min(nx, ny) / 2)).astype(int)
            complete_levels = min(complete_levels, min(nx+1, ny+1)) // 2
            print(f"Warning: Reduced number of full boundary levels to {n_levels} due to small grid size (nx={nx}, ny={ny}).")
        for level in range(n_levels):
            if boundary_level == level + 1:
                start_at_b_level = idx
            # south boundary vertices:
            transform_array[idx:idx+nx - 2 * level] = np.arange(0 + level * (nx + 2), nx + level * nx, 1)
            idx += nx - 2 * level
            # east boundary vertices:
            transform_array[idx:idx+ny - 2 * level] = np.arange(nx + level * nx, N - nx+1 - level * (nx + 2), nx + 1)
            idx += ny - 2 * level
            # north boundary vertices:
            transform_array[idx:idx+nx - 2 * level] = np.arange(N - 1 - level * (nx + 2), N - nx-1 - level * nx, -1)
            idx += nx - 2 * level
            # west boundary vertices:
            transform_array[idx:idx+ny - 2 * level] = np.arange(N - nx - 1 - level * nx, 1 + level * (nx + 2), -nx - 1)
            idx += ny - 2 * level
            if level + 1 == complete_levels:
                print(f"Reached complete level at {level+1}, filling remaining vertices ascending.")
                needs_completion = True
                break
        if boundary_level == n_levels + 1:
            start_at_b_level = idx

    # fill backtransform array:
    back_transform_array[transform_array[:idx]] = np.arange(idx)

    # fill remaining boundary levels and interior ascending.
    for i in range(1, N):
        if back_transform_array[i] == 0 and transform_array[0] != i:
            transform_array[idx] = i
            back_transform_array[i] = idx
            idx += 1
    
    # fill unstructured field
    unstructured_field[back_transform_array[:N]] = field[:N]
        

    if idx != N:
        print(f"Warning: Transformation array filled with {idx} entries, expected {N}. Check if mapping is correct.")

                

    return (transform_array, back_transform_array, unstructured_field, start_at_b_level)

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
    
    # print(f"int(nodes_size)={nodes_size}, ")

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
    # print(f"vertex_to_ij:\n{vertex_to_ij}\nij_to_vertex:\n{ij_to_vertex}")

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