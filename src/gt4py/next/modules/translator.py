# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, List, cast

import numpy as np
import xarray as xr

import gt4py.next as gtx
from gt4py.next.iterator import atlas_utils, ir
from gt4py.next.iterator.transforms.map_dict import map_dict as _MAP_DICT


# Define Dimensions
IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
Kolor = gtx.Dimension("Kolor")


def _parse_sparse_remap_table() -> dict[str, dict[int, dict[int, tuple[int, int, int]]]]:
    """Parse map_dict remaps into conn -> center_kolor -> slot -> (di, dj, neighbor_kolor)."""
    max_kolor = 3

    def _extract_shift(shifts_tuple: tuple) -> tuple[int, int, int]:
        vals: dict[str, int] = {}
        it = iter(shifts_tuple)
        for axis_lit, offset_lit in zip(it, it):
            vals[axis_lit.value] = int(offset_lit.value)
        return vals.get("IDim", 0), vals.get("JDim", 0), vals.get("Kolor", 0)

    def _kolor_range_from_domain(domain: Any) -> tuple[int, int] | None:
        # print(f"Extracting kolor range from domain: {domain}")
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
    for (conn_lit, slot_lit), raw_entry in _MAP_DICT.items():
        entry = cast(dict[str, object], raw_entry)
        conn_name = cast(str, conn_lit.value)
        slot = int(slot_lit.value)
        table.setdefault(conn_name, {})

        if entry["kind"] == "shift":
            di, dj, dk = _extract_shift(cast(tuple, entry["shifts"]))
            for ck in range(max_kolor):
                table[conn_name].setdefault(ck, {})[slot] = (di, dj, ck + dk)
            continue

        if entry["kind"] == "concat_where":
            covered: set[int] = set()
            else_shift: tuple[int, int, int] | None = None
            resolved: list[tuple[tuple[int, int] | None, tuple[int, int, int]]] = []
            for domain, shifts in cast(tuple, entry["branches"]):
                shift = _extract_shift(cast(tuple, shifts))
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


_SPARSE_REMAP_TABLE: dict[str, dict[int, dict[int, tuple[int, int, int]]]] = (
    _parse_sparse_remap_table()
)


_STRUCTURED_REMAP_SIZES_CACHE: dict[tuple[str, int, int, int], "StructuredRemapSizes"] = {}
_STRUCTURED_REMAP_SIZES_CACHE_LOCK = threading.Lock()
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
        out = np.full((ni, nj, n_kolor, n_local, *tail_shape), -1, dtype=coeff.dtype, order='F')
    else:
        out = np.zeros((ni, nj, n_kolor, n_local, *tail_shape), dtype=coeff.dtype, order='F')

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
        # Use the character after the last "2" to determine neighbor element type.
        # E.g. "E2C2EO" → last "2" at idx 3 → neighbor char = "E" (Edge), not "O".
        _last2 = local_dim_name.rfind("2")
        _nb_char = local_dim_name[_last2 + 1] if _last2 >= 0 and _last2 + 1 < len(local_dim_name) else local_dim_name[-1]
        ntype = _CENTER_ELEMENT_BY_PREFIX.get(_nb_char, "Edge")
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

    # The index mapping is fixed for a given grid+connectivity.
    # Build it once here; callers can cache the result for subsequent fast packing.
    idx = _build_sparse_pack_index_arrays(
        conn, index_map, local_dim_name, rel_to_slots, n_elem, n_local,
        max_i, max_j, _center_ijk, _neighbor_ijk, _normalize
    )
    if idx is not None:
        ea, la, cia, cja, cka, sa = idx
        out[cia, cja, cka, sa] = coeff[ea, la]
    return out


def precompute_sparse_pack_mapping(
    conn: np.ndarray,
    index_map: "IndexMap",
    local_dim_name: str,
    *,
    cell_to_ijk: np.ndarray | None = None,
) -> tuple | None:
    """Precompute and return sparse pack index arrays for caching.

    Returns (elem_arr, local_arr, ci_arr, cj_arr, ck_arr, slot_arr) or None.
    Cache this result and pass it to apply_sparse_pack_mapping for fast O(1) packing.
    """
    conn = np.asarray(conn, dtype=np.int32)
    n_local = conn.shape[1] if conn.ndim == 2 else 0

    # E2C special case: weights stored directly at the edge's structured position (no remap).
    # Mirrors the special case in pack_sparse_local_field_to_structured.
    if local_dim_name == "E2C":
        n_elem = min(conn.shape[0], index_map.edge_to_ijk.shape[0])
        n_local_eff = min(n_local, conn.shape[1])
        ijk = index_map.edge_to_ijk[:n_elem]
        valid = (ijk[:, 0] >= 0) & (ijk[:, 1] >= 0) & (ijk[:, 2] >= 0)
        vidx = np.where(valid)[0]
        if vidx.size == 0:
            return None
        ea = np.repeat(vidx, n_local_eff).astype(np.intp)
        la = np.tile(np.arange(n_local_eff, dtype=np.intp), vidx.size)
        cia = np.repeat(ijk[vidx, 0], n_local_eff).astype(np.intp)
        cja = np.repeat(ijk[vidx, 1], n_local_eff).astype(np.intp)
        cka = np.repeat(ijk[vidx, 2], n_local_eff).astype(np.intp)
        sa = la  # slot index = local index for E2C
        return (ea, la, cia, cja, cka, sa)

    remap = _SPARSE_REMAP_TABLE.get(local_dim_name)
    if remap is None:
        return None
    ni, nj, n_kolor = index_map.ijk_to_edge.shape
    max_i, max_j = index_map.ij_to_vertex.shape
    rel_to_slots: dict = {}
    for ck, slot_map in remap.items():
        r2s: dict = {}
        for slot, (di, dj, nk) in slot_map.items():
            r2s.setdefault((di, dj, nk), []).append(slot)
        rel_to_slots[int(ck)] = r2s
    center_type = _CENTER_ELEMENT_BY_PREFIX.get(local_dim_name[0], "Edge")

    def _normalize(delta, period):
        if period <= 0: return delta
        half = period // 2
        if delta > half: return delta - period
        if delta < -half: return delta + period
        return delta

    def _center_ijk(eidx):
        if center_type == "Edge":
            ijk = index_map.edge_to_ijk[eidx]
            return int(ijk[0]), int(ijk[1]), int(ijk[2])
        if center_type == "Vertex":
            ij = index_map.vertex_to_ij[eidx]
            return int(ij[0]), int(ij[1]), 0
        if center_type == "Cell":
            if cell_to_ijk is None or eidx >= cell_to_ijk.shape[0]: return None
            ijk = cell_to_ijk[eidx]
            return int(ijk[0]), int(ijk[1]), int(ijk[2])
        return None

    def _neighbor_ijk(nidx):
        _last2 = local_dim_name.rfind("2")
        _nb_char = local_dim_name[_last2 + 1] if _last2 >= 0 and _last2 + 1 < len(local_dim_name) else local_dim_name[-1]
        ntype = _CENTER_ELEMENT_BY_PREFIX.get(_nb_char, "Edge")
        if ntype == "Edge":
            ijk = index_map.edge_to_ijk[nidx]
            return int(ijk[0]), int(ijk[1]), int(ijk[2])
        if ntype == "Vertex":
            ij = index_map.vertex_to_ij[nidx]
            return int(ij[0]), int(ij[1]), 0
        if ntype == "Cell":
            if cell_to_ijk is None or nidx >= cell_to_ijk.shape[0]: return None
            ijk = cell_to_ijk[nidx]
            return int(ijk[0]), int(ijk[1]), int(ijk[2])
        return None

    n_elem = conn.shape[0]
    return _build_sparse_pack_index_arrays(
        conn, index_map, local_dim_name, rel_to_slots, n_elem, n_local,
        max_i, max_j, _center_ijk, _neighbor_ijk, _normalize
    )


def apply_sparse_pack_mapping(
    coeff: np.ndarray,
    mapping: tuple,
    out_shape: tuple,
) -> np.ndarray:
    """Apply precomputed sparse pack index arrays to pack coeff into structured layout."""
    ea, la, cia, cja, cka, sa = mapping
    out = np.zeros(out_shape, dtype=coeff.dtype, order='F')
    if ea.size > 0:
        out[cia, cja, cka, sa] = coeff[ea, la]
    return out


def _build_sparse_pack_index_arrays(
    conn, index_map, local_dim_name, rel_to_slots, n_elem, n_local,
    max_i, max_j, _center_ijk, _neighbor_ijk, _normalize
):
    """Build (elem, local, ci, cj, ck, slot) index arrays for sparse field packing.

    Returns a tuple of 6 numpy arrays, or None if no valid mappings exist.
    These arrays can be cached to make subsequent pack calls O(1) numpy ops.
    """
    elem_list, local_list, ci_list, cj_list, ck_list, slot_list = [], [], [], [], [], []
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
                    elem_list.append(elem)
                    local_list.append(local)
                    ci_list.append(ci)
                    cj_list.append(cj)
                    ck_list.append(ck)
                    slot_list.append(slot)
    if not elem_list:
        return None
    return (
        np.array(elem_list, dtype=np.intp),
        np.array(local_list, dtype=np.intp),
        np.array(ci_list, dtype=np.intp),
        np.array(cj_list, dtype=np.intp),
        np.array(ck_list, dtype=np.intp),
        np.array(slot_list, dtype=np.intp),
    )


@dataclass(frozen=True)
class IndexMap:
    vertex_to_ij: np.ndarray  # (n_vertex, 2) -> (row_i, local_j)
    row_lengths: np.ndarray  # (ni,) lengths per row
    row_offsets: np.ndarray  # (ni,) cumulative offsets
    ij_to_vertex: np.ndarray  # (ni, max_nj) ragged padded with -1
    edge_to_ijk: np.ndarray  # (n_edge, 3) -> (i, j, k) j is local_j
    ijk_to_edge: np.ndarray  # (ni, max_nj, 3) padded with -1


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


def _first_present(ds: xr.Dataset, names: list[str], required: bool = True) -> Any:
    for name in names:
        if name in ds:
            return ds[name]
    if required:
        raise KeyError(f"None of the dataset variables are present: {names}")
    return None


def _read_e2v(ds: xr.Dataset) -> np.ndarray:
    raw = _first_present(ds, ["E2V", "edge_vertices", "edges2nodes", "edge_node_connectivity"])
    arr = np.asarray(raw, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError("e2v dataset must be 2-D")
    if arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"e2v must have shape (n_edge, 2), got {arr.shape}")
    return np.where(arr > 0, arr - 1, -1)


def _read_v2e(ds: xr.Dataset) -> np.ndarray | None:
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


def _read_lonlat(ds: xr.Dataset) -> np.ndarray | None:
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


def _load_structured_remap_sizes_uncached(nc_path: str, lateral: int) -> StructuredRemapSizes:
    with xr.open_dataset(nc_path) as ds:
        if "domain_length" not in ds.attrs or "mean_edge_length" not in ds.attrs:
            raise KeyError(
                "Dataset must contain attributes 'domain_length' and 'mean_edge_length'."
            )
        if "cell" not in ds.sizes:
            raise KeyError("Dataset must contain dimension 'cell'.")

        return infer_structured_remap_sizes(
            domain_length=float(ds.attrs["domain_length"]),
            mean_edge_length=float(ds.attrs["mean_edge_length"]),
            n_cells=int(ds.sizes["cell"]),
            lateral=lateral,
        )


def load_structured_remap_sizes_from_netcdf(nc_path: str, lateral: int = 0) -> StructuredRemapSizes:
    # Cache by resolved path + file identity + lateral so repeated compilations
    # do not repeatedly open/read the same NetCDF metadata.
    resolved_path = Path(nc_path).expanduser().resolve()
    stat = resolved_path.stat()
    cache_key = (str(resolved_path), int(stat.st_mtime_ns), int(stat.st_size), int(lateral))

    with _STRUCTURED_REMAP_SIZES_CACHE_LOCK:
        cached = _STRUCTURED_REMAP_SIZES_CACHE.get(cache_key)
        if cached is not None:
            return cached

        sizes = _load_structured_remap_sizes_uncached(str(resolved_path), int(lateral))
        _STRUCTURED_REMAP_SIZES_CACHE[cache_key] = sizes
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
    out = np.zeros((ni, max_nj, 1, *trailing_shape), dtype=vertex_values.dtype, order='F')
    i_arr = m.vertex_to_ij[:, 0]
    j_arr = m.vertex_to_ij[:, 1]
    valid = i_arr >= 0
    out[i_arr[valid], j_arr[valid], 0] = vertex_values[valid]
    return out


def pack_edge_field_to_structured(edge_values: np.ndarray, m: IndexMap) -> np.ndarray:
    ni, max_nj, n_kolor = m.ijk_to_edge.shape
    valid = m.ijk_to_edge >= 0
    edge_indices = m.ijk_to_edge[valid]
    n_edge = edge_values.shape[0]
    if edge_indices.size > 0 and int(edge_indices.max()) >= n_edge:
        raise IndexError(
            f"IndexMap edge id {edge_indices.max()} exceeds available edge axis {n_edge}. "
            "Use an index map generated for the current grid."
        )
    out = np.zeros((ni, max_nj, n_kolor), dtype=edge_values.dtype, order='F')
    out[valid] = edge_values[edge_indices]
    return out


def unpack_vertex_field_to_unstructured(struct_values: np.ndarray, m: IndexMap) -> np.ndarray:
    n_vertex = m.vertex_to_ij.shape[0]
    trailing_shape = struct_values.shape[3:]
    out = np.zeros((n_vertex, *trailing_shape), dtype=struct_values.dtype)
    i_arr = m.vertex_to_ij[:, 0]
    j_arr = m.vertex_to_ij[:, 1]
    valid = i_arr >= 0
    out[valid] = struct_values[i_arr[valid], j_arr[valid], 0]
    return out


# from icon4py.model.common import dimension as dims

# Structured Dimensions
# IDim = gtx.Dimension("IDim")
# JDim = gtx.Dimension("JDim")
# Kolor = gtx.Dimension("Kolor")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)


def pack_edge_field(edge_values: np.ndarray, m: "IndexMap") -> np.ndarray:
    """Packs 1D or 2D unstructured edge fields into structured [I, J, Kolor, (K)]."""
    ni, max_nj, n_kolor = m.ijk_to_edge.shape
    valid = m.ijk_to_edge >= 0
    edge_indices = m.ijk_to_edge[valid]
    n_edge = edge_values.shape[0]
    if edge_indices.size > 0 and int(edge_indices.max()) >= n_edge:
        raise IndexError(
            f"IndexMap edge id {edge_indices.max()} exceeds available edge axis {n_edge}. "
            "Use an index map generated for the current grid."
        )
    has_k = edge_values.ndim == 2
    if has_k:
        nk = edge_values.shape[1]
        out = np.zeros((ni, max_nj, n_kolor, nk), dtype=edge_values.dtype, order='F')
        out[valid] = edge_values[edge_indices]
    else:
        out = np.zeros((ni, max_nj, n_kolor), dtype=edge_values.dtype, order='F')
        out[valid] = edge_values[edge_indices]
    return out


def unpack_edge_field(struct_values: np.ndarray, m: "IndexMap", n_edge: int) -> np.ndarray:
    """Unpacks structured [I, J, Kolor, (K)] fields back to unstructured."""
    valid = m.ijk_to_edge >= 0
    edge_indices = m.ijk_to_edge[valid]
    has_k = struct_values.ndim == 4
    if has_k:
        nk = struct_values.shape[3]
        out = np.zeros((n_edge, nk), dtype=struct_values.dtype)
        out[edge_indices] = struct_values[valid]
    else:
        out = np.zeros((n_edge,), dtype=struct_values.dtype)
        out[edge_indices] = struct_values[valid]
    return out


_STRIDE_PAD = 32  # pad IDim to this multiple so each JDim row starts cache-line aligned


def _pad_to_stride(n: int) -> int:
    """Round n up to the next multiple of _STRIDE_PAD."""
    return int(np.ceil(n / _STRIDE_PAD)) * _STRIDE_PAD


def pack_edge_field_compact(edge_values: np.ndarray, m: "IndexMap", shift_i: int, shift_j: int) -> np.ndarray:
    """Pack edge field to compact Fortran array with IDim padded to a multiple of 32.

    IDim=shift_i lands at ptr[0] (cache-aligned write start). IDim dimension is
    padded to ceil(ni_raw/_STRIDE_PAD)*_STRIDE_PAD so each JDim row begins at a
    cache-line boundary — DaCe strides are runtime parameters so no recompilation needed.
    """
    ijk = m.ijk_to_edge[shift_i:, shift_j:]
    ni_raw, nj, n_kolor = ijk.shape
    ni = _pad_to_stride(ni_raw)
    valid = ijk >= 0
    has_k = edge_values.ndim == 2
    if has_k:
        nk = edge_values.shape[1]
        out = np.zeros((ni, nj, n_kolor, nk), dtype=edge_values.dtype, order='F')
        out[:ni_raw][valid] = edge_values[ijk[valid]]
    else:
        out = np.zeros((ni, nj, n_kolor), dtype=edge_values.dtype, order='F')
        out[:ni_raw][valid] = edge_values[ijk[valid]]
    return out


def pack_vertex_field_padded(vertex_values: np.ndarray, m: IndexMap, shift_i: int, shift_j: int, pad: int) -> np.ndarray:
    """Pack vertex field to padded array for safe negative-offset reads on GPU.

    Shape: (padded(pad+ni_v-shift_i), pad+nj_v-shift_j, 1, nk) where the IDim
    is rounded up to a multiple of _STRIDE_PAD for cache-line row alignment.
    Pass with origin={IDim: pad, JDim: pad} so DaCe range_0=-pad.
    """
    ni_v, nj_v = m.ij_to_vertex.shape
    ni_raw = pad + ni_v - shift_i
    ni_out = _pad_to_stride(ni_raw)
    nj_out = pad + nj_v - shift_j
    has_k = vertex_values.ndim == 2
    nk = vertex_values.shape[1] if has_k else 1
    out = np.zeros((ni_out, nj_out, 1, nk), dtype=vertex_values.dtype, order='F')
    i_arr = m.vertex_to_ij[:, 0]
    j_arr = m.vertex_to_ij[:, 1]
    ci = i_arr - shift_i + pad
    cj = j_arr - shift_j + pad
    valid = (ci >= 0) & (ci < ni_raw) & (cj >= 0) & (cj < nj_out)
    if has_k:
        out[ci[valid], cj[valid], 0, :] = vertex_values[valid, :]
    else:
        out[ci[valid], cj[valid], 0, 0] = vertex_values[valid]
    return out if has_k else out[:, :, :, 0]


def pack_vertex_field(vertex_values: np.ndarray, m: IndexMap) -> np.ndarray:
    """Packs an unstructured vertex field into [IDim, JDim, Kolor=1, (KDim)]."""
    has_k = vertex_values.ndim == 2
    ni, nj = m.ij_to_vertex.shape
    nk = vertex_values.shape[1] if has_k else 1
    out = np.zeros((ni, nj, 1, nk), dtype=vertex_values.dtype, order='F')
    i_arr = m.vertex_to_ij[:, 0]
    j_arr = m.vertex_to_ij[:, 1]
    valid = i_arr >= 0
    if has_k:
        out[i_arr[valid], j_arr[valid], 0, :] = vertex_values[valid, :]
    else:
        out[i_arr[valid], j_arr[valid], 0, 0] = vertex_values[valid]
    return out if has_k else out[:, :, :, 0]


# --- Cartesian Cell Helpers ---


def build_cell_to_ijk(m: IndexMap, ds: xr.Dataset) -> np.ndarray:
    """Maps unstructured 1D cell index from netcdf into Cartesian [I, J, Kolor] layout."""
    import numpy as np

    c2v = np.where(
        ds["vertex_of_cell"].transpose("cell", "nv").values.astype(np.int32) > 0,
        ds["vertex_of_cell"].transpose("cell", "nv").values.astype(np.int32) - 1,
        -1,
    )
    n_cells = c2v.shape[0]
    ni, nj = m.ij_to_vertex.shape

    ijk_to_cell = np.full((ni, nj, 2), -1, dtype=np.int32)

    for c in range(n_cells):
        v = c2v[c]
        if np.any(v < 0):
            continue

        i_coords = [m.vertex_to_ij[v[0], 0], m.vertex_to_ij[v[1], 0], m.vertex_to_ij[v[2], 0]]
        j_coords = [m.vertex_to_ij[v[0], 1], m.vertex_to_ij[v[1], 1], m.vertex_to_ij[v[2], 1]]

        if any(i < 0 for i in i_coords):
            continue

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
    ni, nj, n_kolor = ijk_to_cell.shape
    has_k = cell_values.ndim == 2
    nk = cell_values.shape[1] if has_k else 1
    valid = ijk_to_cell >= 0
    cell_indices = ijk_to_cell[valid]
    out = np.zeros((ni, nj, n_kolor, nk), dtype=cell_values.dtype, order='F')
    if has_k:
        out[valid] = cell_values[cell_indices]
    else:
        out[valid] = cell_values[cell_indices, np.newaxis]
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


def unpack_cell_field(
    struct_values: np.ndarray, ijk_to_cell: np.ndarray, n_cells: int
) -> np.ndarray:
    """Unpacks [IDim, JDim, Kolor, (KDim)] Cell arrays back to unstructured."""
    has_k = struct_values.ndim == 4
    valid = ijk_to_cell >= 0
    cell_indices = ijk_to_cell[valid]
    if has_k:
        nk = struct_values.shape[3]
        out = np.zeros((n_cells, nk), dtype=struct_values.dtype)
        out[cell_indices] = struct_values[valid]
    else:
        out = np.zeros((n_cells, 1), dtype=struct_values.dtype)
        out[cell_indices, 0] = struct_values[valid]
    return out if has_k else out[:, 0]


def unpack_cell_field_from_structured(
    struct_values: np.ndarray,
    cell_to_ijk: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Compatibility API to unpack structured cell fields via cell_to_ijk map."""
    has_k = struct_values.ndim == 4
    m = min(n_cells, cell_to_ijk.shape[0])
    ijk = cell_to_ijk[:m]
    valid_mask = ijk[:, 0] >= 0
    c_valid = np.where(valid_mask)[0]
    i_arr = ijk[c_valid, 0]
    j_arr = ijk[c_valid, 1]
    k_arr = ijk[c_valid, 2]
    if has_k:
        nk = struct_values.shape[3]
        out = np.zeros((n_cells, nk), dtype=struct_values.dtype)
        out[c_valid] = struct_values[i_arr, j_arr, k_arr]
    else:
        out = np.zeros((n_cells, 1), dtype=struct_values.dtype)
        out[c_valid, 0] = struct_values[i_arr, j_arr, k_arr]
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
                n0_0 = ijk_to_cell[i, j, 1] if j >= 0 else -1
                n0_1 = ijk_to_cell[i, j - 1, 1] if j - 1 >= 0 else -1
                n0_2 = ijk_to_cell[i - 1, j, 1] if i - 1 >= 0 else -1
                c2e2co[c0] = [n0_0, n0_1, n0_2]

            if c1 >= 0:
                n1_0 = ijk_to_cell[i, j, 0] if j >= 0 else -1
                n1_1 = ijk_to_cell[i, j + 1, 0] if j + 1 < nj else -1
                n1_2 = ijk_to_cell[i + 1, j, 0] if i + 1 < ni else -1
                c2e2co[c1] = [n1_0, n1_1, n1_2]
    return c2e2co


def pack_c2e2co_field(field_np: np.ndarray, ijk_to_cell: np.ndarray) -> tuple[np.ndarray, ...]:
    """Packs C2E2CO neighbour lookup tables into a tuple of 3 [IDim, JDim, Kolor] fields."""
    import numpy as np

    ni, nj, _ = ijk_to_cell.shape
    n_neighbors = field_np.shape[1]
    out_s = tuple(np.zeros((ni, nj, 2), dtype=field_np.dtype, order='F') for _ in range(n_neighbors))

    for i in range(ni):
        for j in range(nj):
            c0, c1 = ijk_to_cell[i, j, 0], ijk_to_cell[i, j, 1]

            if c0 >= 0:
                n0_0 = ijk_to_cell[i, j, 1]
                n0_1 = ijk_to_cell[i, j - 1, 1] if j > 0 else -1
                n0_2 = ijk_to_cell[i - 1, j, 1] if i > 0 else -1
                neighbors0 = [n0_0, n0_1, n0_2]
                for idx in range(n_neighbors):
                    if neighbors0[idx] != -1:  # <-- THE FIX: Force 0.0 for out-of-bounds!
                        out_s[idx][i, j, 0] = field_np[c0, idx]

            if c1 >= 0:
                n1_0 = ijk_to_cell[i, j, 0]
                n1_1 = ijk_to_cell[i, j + 1, 0] if j + 1 < nj else -1
                n1_2 = ijk_to_cell[i + 1, j, 0] if i + 1 < ni else -1
                neighbors1 = [n1_0, n1_1, n1_2]
                for idx in range(n_neighbors):
                    if neighbors1[idx] != -1:  # <-- THE FIX: Force 0.0 for out-of-bounds!
                        out_s[idx][i, j, 1] = field_np[c1, idx]
    return out_s


def _rounded_unique(vals: np.ndarray, decimals: int = 10) -> np.ndarray:
    return np.unique(np.round(vals.astype(np.float64), decimals=decimals))


def transform_to_unstructured(
    field: np.ndarray, nx: int, grid_obj: str = "Edge", boundary_level: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
        n_levels = 5  # lateral 1 to 8 and two nudging levels, only write 5, because we need to fill two at the time to get the correct mapping for the interior edges
        ny = int((N - nx) / (3 * nx + 1))
        idx = 0
        kolor_1_start = nx * (ny + 1)
        kolor_2_start = (2 * nx * ny) + nx + ny
        complete_levels = 10
        if min(nx, ny) < 2 * n_levels:
            n_levels = (np.ceil(min(nx, ny) / 2)).astype(int)
            complete_levels = min(complete_levels, min(nx, ny))
            print(
                f"Warning: Reduced number of boundary levels to {n_levels} due to small grid size (nx={nx}, ny={ny})."
            )
        for level in range(n_levels):
            # odd levels are single edge boundaries
            # cycle through the edges in the current boundary level, then fill the interior edges
            transform_array[idx : idx + nx - 2 * level] = np.arange(
                0 + level * (nx + 1), nx + level * (nx - 1), 1
            )  # south boundary edges
            idx += nx - 2 * level
            transform_array[idx : idx + ny - 2 * level] = np.arange(
                kolor_1_start + nx + level * nx, kolor_2_start - level * (nx), nx + 1
            )  # east boundary edges
            idx += ny - 2 * level
            transform_array[idx : idx + nx - 2 * level] = np.arange(
                kolor_1_start - 1 - level * (nx + 1), kolor_1_start - 1 - nx - level * (nx - 1), -1
            )  # north boundary edges
            idx += nx - 2 * level
            transform_array[idx : idx + ny - 2 * level] = np.arange(
                kolor_2_start - 1 - nx - level * nx, kolor_1_start - 1 + level * (nx + 2), -(nx + 1)
            )  # west boundary edges
            idx += ny - 2 * level
            if boundary_level == 2 * (level + 1):
                start_at_b_level = idx
            if 2 * level + 1 == complete_levels:
                print(
                    f"Reached complete level at {2 * level + 1}, filling remaining edges with interior mapping."
                )
                break
            # even levels are more interior edges
            transform_array[idx : idx + nx - 1 - 2 * level] = np.arange(
                kolor_1_start + 1 + level * (nx + 2), kolor_1_start + nx + level * (nx), 1
            )
            idx += nx - 1 - 2 * level
            transform_array[idx : idx + ny - 1 - 2 * level] = np.arange(
                2 * nx - 1 + level * (nx - 1), kolor_1_start - 1 - level * (nx + 1), nx
            )
            idx += ny - 1 - 2 * level
            transform_array[idx : idx + nx - 1 - 2 * level] = np.arange(
                kolor_2_start - 2 - level * (nx + 2), kolor_2_start - 2 - (nx - 1) - level * nx, -1
            )
            idx += nx - 1 - 2 * level
            transform_array[idx : idx + ny - 1 - 2 * level] = np.arange(
                kolor_1_start - 2 * nx - level * (nx - 1), 0 + level * (nx + 1), -nx
            )
            idx += ny - 1 - 2 * level
            # kolor 2 edges
            transform_array[idx : idx + nx - 2 * level] = np.arange(
                kolor_2_start + level * (nx + 1), kolor_2_start + nx + level * (nx - 1), 1
            )
            idx += nx - 2 * level
            transform_array[idx : idx + ny - 1 - 2 * level] = np.arange(
                kolor_2_start + 2 * nx - 1 + level * (nx - 1), N - level * (nx), nx
            )
            idx += ny - 1 - 2 * level
            transform_array[idx : idx + nx - 1 - 2 * level] = np.arange(
                N - 2 - level * (nx + 1), N - nx - 1 - level * (nx - 1), -1
            )
            idx += nx - 1 - 2 * level
            transform_array[idx : idx + ny - 2 - 2 * level] = np.arange(
                N - 2 * nx - level * (nx - 1), kolor_2_start + level * (nx + 1), -nx
            )
            idx += ny - 2 - 2 * level
            if boundary_level == 2 * (level + 1) + 1:
                start_at_b_level = idx

    elif grid_obj == "Cell":
        n_levels = 5  # lateral 1 to 4 and one nudging levels
        ny = int(N / (2 * nx))
        idx = 0
        kolor_1_start = nx * ny
        complete_levels = 5
        if min(nx, ny) < 2 * n_levels:
            n_levels = (np.ceil(min(nx, ny) / 2)).astype(int)
            complete_levels = min(complete_levels, min(nx, ny)) // 2
            print(
                f"Warning: Reduced number of full boundary levels to {n_levels} due to small grid size (nx={nx}, ny={ny})."
            )
        for level in range(n_levels):
            if boundary_level == level + 1:
                start_at_b_level = idx
            # south up cells:
            transform_array[idx : idx + nx - 1 - 2 * level] = np.arange(
                0 + level * (nx + 1), nx - 1 + level * (nx - 1), 1
            )
            idx += nx - 1 - 2 * level
            # south down cells:
            transform_array[idx : idx + nx - 1 - 2 * level] = np.arange(
                kolor_1_start + level * (nx + 1), kolor_1_start + nx - 1 + level * (nx - 1), 1
            )
            idx += nx - 1 - 2 * level
            # east up cells:
            transform_array[idx : idx + ny - 1 - 2 * level] = np.arange(
                nx - 1 + level * (nx - 1), kolor_1_start - nx - level * (nx + 1), nx
            )
            idx += ny - 1 - 2 * level
            # east down cells:
            transform_array[idx : idx + ny - 1 - 2 * level] = np.arange(
                kolor_1_start + nx - 1 + level * (nx - 1), N - nx - level * (nx + 1), nx
            )
            idx += ny - 1 - 2 * level
            # north up cells:
            transform_array[idx : idx + nx - 1 - 2 * level] = np.arange(
                kolor_1_start - 1 - level * (nx + 1), kolor_1_start - nx - level * (nx - 1), -1
            )
            idx += nx - 1 - 2 * level
            # north down cells:
            transform_array[idx : idx + nx - 1 - 2 * level] = np.arange(
                N - 1 - level * (nx + 1), N - nx - level * (nx - 1), -1
            )
            idx += nx - 1 - 2 * level
            # west up cells:
            transform_array[idx : idx + ny - 1 - 2 * level] = np.arange(
                kolor_1_start - nx - level * (nx - 1), 0 + level * (nx + 1), -nx
            )
            idx += ny - 1 - 2 * level
            # west down cells:
            transform_array[idx : idx + ny - 1 - 2 * level] = np.arange(
                N - nx - level * (nx - 1), kolor_1_start + level * (nx + 1), -nx
            )
            idx += ny - 1 - 2 * level
            if level + 1 == complete_levels:
                print(f"Reached complete level at {level + 1}, filling remaining cells ascending.")
                break
        if boundary_level == n_levels + 1:
            start_at_b_level = idx

    elif grid_obj == "Vertex":
        # for vertices, we have the same boundary levels as for the cells, but only one kolor type.
        # it can be filled the same way as the cell mapping, but with a total size of nx +1 for nx and ny+1 for ny
        n_levels = 5  # lateral 1 to 4 and one nudging levels
        ny = int((N) / (nx + 1) - 1)
        idx = 0
        complete_levels = 5
        if min(nx, ny) < 2 * n_levels:
            n_levels = (np.ceil(min(nx, ny) / 2)).astype(int)
            complete_levels = min(complete_levels, min(nx + 1, ny + 1)) // 2
            print(
                f"Warning: Reduced number of full boundary levels to {n_levels} due to small grid size (nx={nx}, ny={ny})."
            )
        for level in range(n_levels):
            if boundary_level == level + 1:
                start_at_b_level = idx
            # south boundary vertices:
            transform_array[idx : idx + nx - 2 * level] = np.arange(
                0 + level * (nx + 2), nx + level * nx, 1
            )
            idx += nx - 2 * level
            # east boundary vertices:
            transform_array[idx : idx + ny - 2 * level] = np.arange(
                nx + level * nx, N - nx + 1 - level * (nx + 2), nx + 1
            )
            idx += ny - 2 * level
            # north boundary vertices:
            transform_array[idx : idx + nx - 2 * level] = np.arange(
                N - 1 - level * (nx + 2), N - nx - 1 - level * nx, -1
            )
            idx += nx - 2 * level
            # west boundary vertices:
            transform_array[idx : idx + ny - 2 * level] = np.arange(
                N - nx - 1 - level * nx, 1 + level * (nx + 2), -nx - 1
            )
            idx += ny - 2 * level
            if level + 1 == complete_levels:
                print(
                    f"Reached complete level at {level + 1}, filling remaining vertices ascending."
                )
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
        print(
            f"Warning: Transformation array filled with {idx} entries, expected {N}. Check if mapping is correct."
        )

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


def build_index_map_from_ds_regular(ds: xr.Dataset, e2v: np.ndarray) -> IndexMap:
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
