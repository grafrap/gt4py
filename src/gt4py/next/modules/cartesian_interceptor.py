import os
import functools
import time
import xarray as xr
import gt4py.next as gtx
import numpy as np
from icon4py.model.common.dimension import IDim, JDim, Kolor

from gt4py.next.modules.translator import (
    IndexMap,
    StructuredRemapSizes,
    load_structured_remap_sizes_from_netcdf,
    build_index_map_from_lonlat_e2v,
    build_cell_ijk_maps,
    pack_vertex_field_to_structured,
    pack_edge_field_to_structured,
    pack_edge_field,
    pack_cell_field_to_structured,
    unpack_vertex_field_to_unstructured,
    unpack_edge_field,
    unpack_cell_field_from_structured,
    precompute_sparse_pack_mapping,
    apply_sparse_pack_mapping,
    _read_e2v,
    _read_lonlat
)
from gt4py.next.iterator.transforms.map_dict import map_dict as _MAP_DICT
from gt4py.next.iterator import ir

# Module-level global used to communicate symbolic_domain_sizes to the DaCe backend's
# apply_fieldview_transforms at compile time. This works across threads because:
# - The main thread sets it before calling _compiled_programs() (which blocks until done).
# - The DaCe thread-pool compilation reads it while the main thread is blocked.
# - It is cleared only after the blocking call returns, so no race condition exists.
_CURRENT_COMPILE_SDS: dict | None = None


def get_compile_sds() -> dict | None:
    """Return the symbolic_domain_sizes currently being compiled, or None."""
    return _CURRENT_COMPILE_SDS


def _parse_map_dict_remap_table() -> dict[str, dict[int, dict[int, tuple[int, int, int]]]]:
    """
    Parse the map_dict into a nested lookup table:
      conn_name -> center_kolor -> slot -> (di, dj, neighbor_kolor_abs)

    For "shift" entries the same relative shift applies to all center kolors.
    For "concat_where" entries each branch covers a range of center kolors.
    The Kolor offset in the shifts is *relative*, so neighbor_kolor_abs = center_kolor + dk.
    """
    # Maximum number of kolors (0, 1, 2) per element type used in map_dict
    _MAX_KOLOR = 3

    def _extract_shift(shifts_tuple: tuple) -> tuple[int, int, int]:
        """Extract (di, dj, dk) from a flat shifts tuple."""
        vals: dict[str, int] = {}
        it = iter(shifts_tuple)
        for axis_lit, offset_lit in zip(it, it):
            axis = axis_lit.value
            offset = offset_lit.value
            vals[axis] = int(offset)
        return vals.get("IDim", 0), vals.get("JDim", 0), vals.get("Kolor", 0)

    def _kolor_range_from_domain(domain) -> tuple[int, int]:
        """Return (start, stop) kolor range from a _kolor_slice domain node, or None for else branch."""
        if domain is None:
            return None
        # domain is an ir.FunCall to im.domain; navigate its structure to find the Kolor range
        # The domain is built as ir.FunCall(fun=..., args=[ir.FunCall(fun=named_range, args=[Kolor, start, stop])])
        # We just need to check if it has the right form; extract start/stop offset literals.
        try:
            named_ranges = domain.args  # list of named_range calls
            for nr in named_ranges:
                dim_arg = nr.args[0]
                if isinstance(dim_arg, ir.AxisLiteral) and dim_arg.value == "Kolor":
                    start = nr.args[1].value
                    stop = nr.args[2].value
                    return int(start), int(stop)
        except (AttributeError, IndexError, TypeError):
            pass
        return None

    table: dict[str, dict[int, dict[int, tuple[int, int, int]]]] = {}

    for (conn_lit, slot_lit), entry in _MAP_DICT.items():
        conn_name: str = conn_lit.value
        slot: int = int(slot_lit.value)

        if conn_name not in table:
            table[conn_name] = {}

        kind = entry["kind"]

        if kind == "shift":
            di, dj, dk = _extract_shift(entry["shifts"])
            # Same mapping for all center kolors
            for ck in range(_MAX_KOLOR):
                nk = ck + dk
                table[conn_name].setdefault(ck, {})[slot] = (di, dj, nk)

        elif kind == "concat_where":
            branches = entry["branches"]
            # Each branch is (domain_or_None, shifts_tuple)
            # Collect (kolor_range, shift) pairs, then resolve the "else" branch
            resolved: list[tuple[tuple[int, int] | None, tuple[int, int, int]]] = []
            else_shift: tuple[int, int, int] | None = None

            for domain, shifts in branches:
                di, dj, dk = _extract_shift(shifts)
                if domain is None:
                    else_shift = (di, dj, dk)
                else:
                    krange = _kolor_range_from_domain(domain)
                    resolved.append((krange, (di, dj, dk)))

            # Determine which kolors are covered by explicit branches
            covered = set()
            for krange, shift in resolved:
                if krange is not None:
                    for ck in range(krange[0], krange[1]):
                        covered.add(ck)
                        nk = ck + shift[2]
                        table[conn_name].setdefault(ck, {})[slot] = (shift[0], shift[1], nk)

            # The else branch covers all remaining kolors in [0, MAX_KOLOR)
            if else_shift is not None:
                for ck in range(_MAX_KOLOR):
                    if ck not in covered:
                        nk = ck + else_shift[2]
                        table[conn_name].setdefault(ck, {})[slot] = (else_shift[0], else_shift[1], nk)

    return table


# Built once at module import time; keyed by connectivity name (e.g. "E2C2E", "C2E", "V2E").
_SPARSE_REMAP_TABLE: dict[str, dict[int, dict[int, tuple[int, int, int]]]] = _parse_map_dict_remap_table()

# Maps local Dim suffix (as it appears in field.domain.dims[1].value) to the connectivity name
# used in map_dict and in the offset_provider key.
# e.g. "E2C2E" -> "E2C2E", "E2C2EO" -> "E2C2EO", "E2C" -> "E2C", etc.
# We detect automatically by matching the dim value to a key in _SPARSE_REMAP_TABLE.

# Prefix of the connectivity name determines the center element type.
_CENTER_ELEMENT_BY_PREFIX: dict[str, str] = {
    "E": "Edge",
    "C": "Cell",
    "V": "Vertex",
}

# gtx.int32 stencil parameters that represent flat edge-index zone boundaries.
# For each such param found in call kwargs the wrapper computes per-kolor (i,j) bounds
# from the edge_to_ijk mapping and bakes them into symbolic_domain_sizes so that
# cart_unroll.py can use them instead of the symmetric extra_halo heuristic.
_KNOWN_EDGE_THRESHOLD_PARAMS: frozenset[str] = frozenset({
    "start_2nd_nudge_line_idx_e",
    "start_nudging_line_idx_e",
    "start_halo_level_2_idx_e",
    "start_edge_lateral_boundary",
    "start_edge_lateral_boundary_level_7",
    "start_edge_nudging_level_2",
    "end_edge_nudging",
    "end_edge_halo",
    "horizontal_start_distance",
    "horizontal_end_distance",
})

_KNOWN_CELL_THRESHOLD_PARAMS: frozenset[str] = frozenset({
    "lateral_boundary_level_2",
})


def _inject_edge_range_bounds(
    edge_mapping_rows: tuple[tuple[int, ...], ...],
    thresh_dict: dict[str, int],
    sds: dict,
    max_i: int,
    max_j: int,
) -> None:
    """Detect paired (horizontal_start_X / horizontal_end_X) thresholds and inject per-kolor
    range bounds for [start, end) so that `and_(EdgeDim>=start, EdgeDim<end)` can be expressed
    as a single structured domain condition rather than `and_(cond, not_(cond))`."""
    from gt4py.next.iterator.transforms.cart_unroll import _derive_entity_range_bounds_from_mapping

    for start_name, start_val in thresh_dict.items():
        for prefix in ("horizontal_start_", "start_"):
            if not start_name.startswith(prefix):
                continue
            suffix = start_name[len(prefix):]
            end_prefix = prefix.replace("start", "end")
            end_name = f"{end_prefix}{suffix}"
            if end_name not in thresh_dict:
                continue
            end_val = thresh_dict[end_name]
            if end_val <= start_val:
                continue
            range_bounds = _derive_entity_range_bounds_from_mapping(
                "Edge",
                mapping_rows=edge_mapping_rows,
                horizontal_start=start_val,
                horizontal_end=end_val,
                max_i=max_i,
                max_j=max_j,
            )
            if range_bounds is None:
                continue
            range_key = f"{start_name}|{end_name}"
            for kolor, (ilo, jlo, ihi, jhi) in range_bounds.items():
                sds[f"{range_key}_k{kolor}_ilo"] = ilo
                sds[f"{range_key}_k{kolor}_jlo"] = jlo
                sds[f"{range_key}_k{kolor}_ihi"] = ihi + 1  # exclusive
                sds[f"{range_key}_k{kolor}_jhi"] = jhi + 1  # exclusive


def pack_sparse_local_field_to_structured(
    coeff: np.ndarray,
    connectivity: np.ndarray,
    index_map: IndexMap,
    local_dim_name: str,
    *,
    cell_to_ijk: np.ndarray | None = None,
) -> np.ndarray:
    """Pack sparse-local coefficients into structured [I, J, Kolor, Local, ...].

    Args:
        coeff: Unstructured sparse-local coefficients, shape [center, local, ...].
        connectivity: Unstructured local connectivity, shape [center, local].
        index_map: Structured index map.
        local_dim_name: Connectivity/local dimension name (e.g. "V2E", "E2C2E").
        cell_to_ijk: Optional cell mapping for cell-centered connectivities.
    """
    if coeff.ndim < 2:
        raise ValueError("Sparse-local coefficients must have at least 2 dimensions [center, local, ...].")

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
            key = (di, dj, nk)
            r2s.setdefault(key, []).append(slot)
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
            rel = (
                _normalize(ni_ - ci, max_i),
                _normalize(nj_ - cj, max_j),
                nk_,
            )
            slots = r2s.get(rel)
            if slots is None:
                continue
            for slot in slots:
                if slot < n_local:
                    out[ci, cj, ck, slot, ...] = coeff[elem, local, ...]
    return out

_CACHED_INDEX_MAP = None
_CACHED_REMAP_SIZES = None
_CACHED_EDGE_COUNT = None


def _swap_index_map_edge_colors(index_map: IndexMap, color_a: int = 0, color_b: int = 2) -> IndexMap:
    ijk_to_edge = np.array(index_map.ijk_to_edge, copy=True)
    edge_to_ijk = np.array(index_map.edge_to_ijk, copy=True)

    tmp = np.array(ijk_to_edge[:, :, color_a], copy=True)
    ijk_to_edge[:, :, color_a] = ijk_to_edge[:, :, color_b]
    ijk_to_edge[:, :, color_b] = tmp

    mask_a = edge_to_ijk[:, 2] == color_a
    mask_b = edge_to_ijk[:, 2] == color_b
    edge_to_ijk[mask_a, 2] = color_b
    edge_to_ijk[mask_b, 2] = color_a

    return IndexMap(
        vertex_to_ij=index_map.vertex_to_ij,
        row_lengths=index_map.row_lengths,
        row_offsets=index_map.row_offsets,
        ij_to_vertex=index_map.ij_to_vertex,
        edge_to_ijk=edge_to_ijk,
        ijk_to_edge=ijk_to_edge,
    )


def _build_periodic_square_index_map(e2v: np.ndarray) -> tuple[IndexMap, StructuredRemapSizes] | None:
    n_edge = int(e2v.shape[0])
    n_vertex = int(e2v.max()) + 1 if e2v.size else 0
    side = int(round(np.sqrt(n_vertex)))
    if side * side != n_vertex or n_edge != 3 * n_vertex:
        return None

    vertex_to_ij = np.zeros((n_vertex, 2), dtype=np.int32)
    ij_to_vertex = np.zeros((side, side), dtype=np.int32)
    for vertex in range(n_vertex):
        i, j = divmod(vertex, side)
        vertex_to_ij[vertex] = (i, j)
        ij_to_vertex[i, j] = vertex

    row_lengths = np.full((side,), side, dtype=np.int32)
    row_offsets = (np.arange(side, dtype=np.int32) * side).astype(np.int32)

    ijk_to_edge = np.full((side, side, 3), -1, dtype=np.int32)
    edge_to_ijk = np.full((n_edge, 3), -1, dtype=np.int32)

    def _assign(i: int, j: int, kolor: int, edge_id: int) -> None:
        if ijk_to_edge[i, j, kolor] != -1:
            raise ValueError(
                f"Ambiguous periodic mapping at (i={i}, j={j}, kolor={kolor}) for edge {edge_id}."
            )
        ijk_to_edge[i, j, kolor] = edge_id
        edge_to_ijk[edge_id] = (i, j, kolor)

    for edge_id, (vertex_a, vertex_b) in enumerate(e2v.astype(np.int32, copy=False)):
        i_a, j_a = vertex_to_ij[int(vertex_a)]
        i_b, j_b = vertex_to_ij[int(vertex_b)]
        di = (int(i_b) - int(i_a)) % side
        dj = (int(j_b) - int(j_a)) % side

        if (di, dj) == (1 % side, 0):
            _assign(int(i_a), int(j_a), 0, edge_id)
        elif (di, dj) == (0, 1 % side):
            _assign(int(i_a), int(j_a), 1, edge_id)
        elif (di, dj) == (1 % side, 1 % side):
            _assign(int(i_a), int(j_a), 2, edge_id)
        elif (di, dj) == ((side - 1) % side, 0):
            _assign(int(i_b), int(j_b), 0, edge_id)
        elif (di, dj) == (0, (side - 1) % side):
            _assign(int(i_b), int(j_b), 1, edge_id)
        elif (di, dj) == ((side - 1) % side, (side - 1) % side):
            _assign(int(i_b), int(j_b), 2, edge_id)
        else:
            return None

    if np.any(ijk_to_edge < 0):
        return None

    index_map = IndexMap(
        vertex_to_ij=vertex_to_ij,
        row_lengths=row_lengths,
        row_offsets=row_offsets,
        ij_to_vertex=ij_to_vertex,
        edge_to_ijk=edge_to_ijk,
        ijk_to_edge=ijk_to_edge,
    )
    remap_sizes = StructuredRemapSizes(
        nx=max(side - 1, 1),
        ny=max(side - 1, 1),
        max_i=side,
        max_j=side,
        vertex_size=n_vertex,
        edge_size_padded=3 * side * side,
        cell_size=0,
        lateral=0,
    )
    return index_map, remap_sizes

def get_global_grid_mapping(e2v_override=None):
    """Builds or returns the cached index_map and remap_sizes for the current run."""
    global _CACHED_INDEX_MAP, _CACHED_REMAP_SIZES, _CACHED_EDGE_COUNT

    override_edge_count = None
    normalized_e2v_override = None
    if e2v_override is not None:
        normalized_e2v_override = np.asarray(e2v_override, dtype=np.int32)
        if normalized_e2v_override.ndim != 2:
            raise ValueError(f"Expected 2D E2V connectivity, got shape {normalized_e2v_override.shape}")
        if normalized_e2v_override.shape[1] != 2 and normalized_e2v_override.shape[0] == 2:
            normalized_e2v_override = normalized_e2v_override.T
        if normalized_e2v_override.shape[1] != 2:
            raise ValueError(
                f"Expected E2V connectivity with shape (n_edge, 2), got {normalized_e2v_override.shape}"
            )
        override_edge_count = int(normalized_e2v_override.shape[0])
    
    if _CACHED_INDEX_MAP is not None and (
        override_edge_count is None or override_edge_count == _CACHED_EDGE_COUNT
    ):
        return _CACHED_INDEX_MAP, _CACHED_REMAP_SIZES

    # Read the grid file specified in the environment (or default)
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH",
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"
    )
    
    # Put your standard reading logic here...
    with xr.open_dataset(mesh_nc) as ds:
        e2v = _read_e2v(ds)
        lonlat = _read_lonlat(ds)
        lateral = int(os.environ.get("GT4PY_TRANSLATOR_LATERAL", "1"))
        remap_sizes = load_structured_remap_sizes_from_netcdf(mesh_nc, lateral=lateral)

    # print(f"lateral={remap_sizes.lateral}, max_i={remap_sizes.max_i}, max_j={remap_sizes.max_j}")        
    index_map = build_index_map_from_lonlat_e2v(lonlat, e2v) # Add your exact sizes here
    
    _CACHED_INDEX_MAP = index_map
    _CACHED_REMAP_SIZES = remap_sizes
    _CACHED_EDGE_COUNT = int(e2v.shape[0])
    
    return _CACHED_INDEX_MAP, _CACHED_REMAP_SIZES

class GenericStructuredWrapper:
    def __init__(self, operator, backend_factory, index_map, remap_sizes, allocator, offset_provider):
        self.index_map = index_map
        self.allocator = allocator
        self.operator_name = getattr(operator, "id", None) or getattr(operator, "__name__", "")
        self.debug_enabled = os.environ.get("GT4PY_STRUCTURED_DEBUG", "0") == "1"
        self.remap_sizes = remap_sizes
        self.max_i = int(self.remap_sizes.max_i)
        self.max_j = int(remap_sizes.max_j)
        
        # 1. Dynamically extract connectivities from the offset_provider.
        # Store all raw (unsanitized) connectivity arrays keyed by name.
        self.v2e_conn = offset_provider.get("V2E").asnumpy() if "V2E" in offset_provider else None
        self.e2v_conn = offset_provider.get("E2V").asnumpy() if "E2V" in offset_provider else None
        self.c2v_conn = offset_provider.get("C2V").asnumpy() if "C2V" in offset_provider else None
        self.cell_to_ijk = None
        self.ijk_to_cell = None
        if self.c2v_conn is not None:
            self.cell_to_ijk, self.ijk_to_cell = build_cell_ijk_maps(self.c2v_conn, self.index_map)

        # Raw connectivity arrays for all sparse local-connectivity types in the offset_provider.
        self._raw_conn: dict[str, np.ndarray] = {}
        self._sanitized_conn: dict[str, np.ndarray] = {}
        if offset_provider:
            for key, value in offset_provider.items():
                key_name = getattr(key, "value", str(key))
                if hasattr(value, "asnumpy"):
                    arr = value.asnumpy()
                    if arr.ndim == 2:
                        if key_name == "E2C":
                            if self.index_map is not None:
                                edge_to_ijk = getattr(self.index_map, "edge_to_ijk", None)
                                if edge_to_ijk is not None:
                                    normalized = np.array(arr, copy=True)
                                    edge_kolor = np.asarray(edge_to_ijk[:, 2], dtype=np.int32)
                                    n_edge = min(normalized.shape[0], edge_kolor.shape[0])
                                    if n_edge > 0 and normalized.shape[1] >= 2:
                                        swap_mask = (edge_kolor[:n_edge] == 0) | (edge_kolor[:n_edge] == 2)
                                        if np.any(swap_mask):
                                            tmp = np.array(normalized[:n_edge, 0], copy=True)
                                            normalized[:n_edge, 0] = np.where(
                                                swap_mask, normalized[:n_edge, 1], normalized[:n_edge, 0]
                                            )
                                            normalized[:n_edge, 1] = np.where(
                                                swap_mask, tmp, normalized[:n_edge, 1]
                                            )
                                    arr = normalized
                        self._raw_conn[key_name] = arr
                        self._sanitized_conn[key_name] = self._sanitize_sparse_connectivity(arr)

        # Legacy aliases kept for debug methods that still reference them directly.
        self.e2c2e_conn_raw = self._raw_conn.get("E2C2E")
        self.e2c2e_conn = self._sanitized_conn.get("E2C2E")
        self.structured_offset_provider = self._build_structured_offset_provider(offset_provider)

        # Build base symbolic_domain_sizes (without horizontal_start — injected lazily at call time).
        symbolic_domain_sizes = {
            "max_i": int(remap_sizes.max_i),
            "max_j": int(remap_sizes.max_j),
            # Always enable mapping-based bounds when index_map is available.
            "use_horizontal_start_mapping": index_map is not None,
        }
        if self.index_map is not None:
            edge_to_ijk = getattr(self.index_map, "edge_to_ijk", None)
            if edge_to_ijk is not None:
                symbolic_domain_sizes["edge_to_ijk"] = [
                    (int(i), int(j), int(k)) for i, j, k in np.asarray(edge_to_ijk)
                ]

            vertex_to_ij = getattr(self.index_map, "vertex_to_ij", None)
            if vertex_to_ij is not None:
                symbolic_domain_sizes["vertex_to_ij"] = [
                    (int(i), int(j)) for i, j in np.asarray(vertex_to_ij)
                ]

        if self.cell_to_ijk is not None:
            symbolic_domain_sizes["cell_to_ijk"] = [
                (int(i), int(j), int(k)) for i, j, k in np.asarray(self.cell_to_ijk)
            ]

        # Store for lazy per-horizontal_start compilation.
        self._operator = operator
        self._backend_factory = backend_factory
        self._symbolic_domain_sizes_base = symbolic_domain_sizes
        # Caches keyed by (horizontal_start, extra_thresholds) tuple.
        self._compiled_cache: dict[tuple, object] = {}
        # Stores the full symbolic_domain_sizes for each cache_key so that the
        # thread-local can be re-set at DaCe JIT-compilation time (which fires
        # on first execution, not at _setup() time).
        self._sds_cache: dict[tuple, dict] = {}
        # Cache for precomputed sparse pack index arrays (built on first use).
        # Key: local_dim_name str; Value: tuple of 6 numpy arrays or None.
        self._sparse_pack_mappings: dict[str, tuple | None] = {}

    def _get_or_compile(
        self,
        horizontal_start: int,
        extra_thresholds: tuple[tuple[str, int], ...] = (),
    ):
        """Return (and cache) a compiled program for the given horizontal_start + thresholds."""
        cache_key = (horizontal_start, extra_thresholds)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        sds = dict(self._symbolic_domain_sizes_base)
        sds["horizontal_start"] = horizontal_start
        sds["horizontal_start_edge"] = horizontal_start
        sds["horizontal_start_cell"] = horizontal_start
        sds["horizontal_start_vertex"] = horizontal_start

        # Inject per-kolor bounds for each known threshold parameter (edge and cell).
        if extra_thresholds:
            from gt4py.next.iterator.transforms.cart_unroll import (
                _derive_entity_start_bounds_from_mapping,
            )
            edge_to_ijk = getattr(self.index_map, "edge_to_ijk", None) if self.index_map is not None else None
            if edge_to_ijk is not None:
                edge_mapping_rows = tuple(
                    tuple(int(x) for x in row) for row in edge_to_ijk
                )
                for param_name, threshold_value in extra_thresholds:
                    if param_name not in _KNOWN_EDGE_THRESHOLD_PARAMS or threshold_value <= 0:
                        continue
                    bounds = _derive_entity_start_bounds_from_mapping(
                        "Edge",
                        mapping_rows=edge_mapping_rows,
                        horizontal_start=threshold_value,
                        max_i=self.max_i,
                        max_j=self.max_j,
                    )
                    if bounds is not None:
                        for kolor, (ilo, jlo, ihi, jhi) in bounds.items():
                            sds[f"{param_name}_k{kolor}_ilo"] = ilo
                            sds[f"{param_name}_k{kolor}_jlo"] = jlo
                            sds[f"{param_name}_k{kolor}_ihi"] = ihi + 1  # exclusive
                            sds[f"{param_name}_k{kolor}_jhi"] = jhi + 1  # exclusive
                # Detect paired (start_X, end_X) edge threshold params and inject range bounds.
                thresh_dict = dict(extra_thresholds)
                _inject_edge_range_bounds(edge_mapping_rows, thresh_dict, sds, self.max_i, self.max_j)
            if self.cell_to_ijk is not None:
                cell_mapping_rows = tuple(
                    tuple(int(x) for x in row) for row in self.cell_to_ijk
                )
                for param_name, threshold_value in extra_thresholds:
                    if param_name not in _KNOWN_CELL_THRESHOLD_PARAMS or threshold_value <= 0:
                        continue
                    bounds = _derive_entity_start_bounds_from_mapping(
                        "Cell",
                        mapping_rows=cell_mapping_rows,
                        horizontal_start=threshold_value,
                        max_i=self.max_i,
                        max_j=self.max_j,
                    )
                    if bounds is not None:
                        for kolor, (ilo, jlo, ihi, jhi) in bounds.items():
                            sds[f"{param_name}_k{kolor}_ilo"] = ilo
                            sds[f"{param_name}_k{kolor}_jlo"] = jlo
                            sds[f"{param_name}_k{kolor}_ihi"] = ihi + 1  # exclusive
                            sds[f"{param_name}_k{kolor}_jhi"] = jhi + 1  # exclusive

        print(
            f"[structured] compiling '{self.operator_name}' for "
            f"horizontal_start={horizontal_start}, thresholds={extra_thresholds}"
        )
        # Build the backend instance. DaCe requires auto_optimize to be passed explicitly
        # (factory.Trait is not accessible via SelfAttribute unless set). GTfn passes
        # symbolic_domain_sizes through the factory hierarchy; DaCe reads it from the
        # thread-local set below.
        try:
            from gt4py.next.program_processors.runners.dace.workflow.backend import (
                DaCeBackendFactory as _DaCeBackendFactory,
            )
            _factory_is_dace = (
                isinstance(self._backend_factory, type)
                and issubclass(self._backend_factory, _DaCeBackendFactory)
            )
        except ImportError:
            _factory_is_dace = False

        _compile_start = time.perf_counter()
        if _factory_is_dace:
            # We need cached=True (outer CachedStep, keyed by SDFG content hash) for C++
            # compilation caching, but otf_workflow__cached_translation=False because the
            # translation cache is keyed by ITIR hash — which does NOT include
            # symbolic_domain_sizes / horizontal_start, causing wrong cached SDFGs to be
            # reused across different horizontal_start values.
            # _compiled_cache provides in-process caching per (horizontal_start, thresholds).
            from gt4py.next.program_processors.runners.dace.workflow.backend import (
                DaCeBackendFactory as _DaCeBackendFactory2,
            )
            from gt4py.next import config as _gt4py_config
            from gt4py.next import common as _common
            structured_backend = _DaCeBackendFactory2(  # type: ignore[return-value]
                gpu=False,
                cached=True,
                auto_optimize=True,
                otf_workflow__cached_translation=False,
                otf_workflow__bare_translation__async_sdfg_call=False,
                otf_workflow__bare_translation__auto_optimize_args={
                    "unit_strides_kind": _common.DimensionKind.HORIZONTAL
                    if _gt4py_config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                    else None,
                },
                otf_workflow__bare_translation__unstructured_horizontal_has_unit_stride=(
                    _gt4py_config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                ),
                otf_workflow__bare_translation__use_metrics=False,
                otf_workflow__bare_translation__disable_field_origin_on_program_arguments=False,
                otf_workflow__bare_translation__use_max_domain_range_on_unstructured_shift=None,
            )
        else:
            structured_backend = self._backend_factory(
                cached=True,
                otf_workflow__cached_translation=True,
                otf_workflow__bare_translation__symbolic_domain_sizes=sds,
            )
        # Store sds so __call__ can re-set the ContextVar at DaCe pool-thread compile time.
        self._sds_cache[cache_key] = sds
        from gt4py.next.program_processors.program_setup_utils import setup_program as _setup
        global _CURRENT_COMPILE_SDS
        _CURRENT_COMPILE_SDS = sds
        try:
            compiled = _setup(
                self._operator,
                backend=structured_backend,
                offset_provider=self.structured_offset_provider,
            )
        finally:
            _CURRENT_COMPILE_SDS = None
        _compile_end = time.perf_counter()
        _compile_elapsed = _compile_end - _compile_start
        print(f"[timing] {self.operator_name} compilation: {_compile_elapsed:.8f}s")
        self._compiled_cache[cache_key] = compiled
        return compiled

    def _get_connectivity(self, offset_provider, name: str):
        if not offset_provider:
            return None
        for key, value in offset_provider.items():
            key_name = getattr(key, "value", str(key))
            if key_name == name:
                return value.asnumpy()
        return None

    def _sanitize_sparse_connectivity(
        self,
        conn: np.ndarray | None,
    ) -> np.ndarray | None:
        if conn is None:
            return None
        sanitized = np.array(conn, copy=True)
        invalid_mask = sanitized < 0
        if not invalid_mask.any():
            return sanitized

        row_max = sanitized.max(axis=1, keepdims=True)
        # Keep fully-invalid rows untouched; fill partial invalid entries with a
        # valid placeholder for connectivities that do not encode masking via -1.
        has_valid = row_max >= 0
        fill_values = np.where(has_valid, 0, sanitized)
        # print(f"[structured-debug] fill values: {fill_values}")
        sanitized = np.where(invalid_mask, fill_values, sanitized)
        return sanitized

    def _normalize_e2c_slot_order(self, conn: np.ndarray | None) -> np.ndarray | None:
        if conn is None:
            return None
        if conn.ndim != 2 or conn.shape[1] < 2:
            return conn
        if self.index_map is None or self.cell_to_ijk is None:
            return conn

        remap = _SPARSE_REMAP_TABLE.get("E2C")
        if remap is None:
            return conn

        normalized = np.array(conn, copy=True)
        edge_to_ijk = self.index_map.edge_to_ijk
        n_edge = min(normalized.shape[0], edge_to_ijk.shape[0])

        for edge in range(n_edge):
            ei, ej, ek = (int(v) for v in edge_to_ijk[edge])
            if ei < 0 or ej < 0 or ek < 0:
                continue

            slot_map = remap.get(ek)
            if slot_map is None or 0 not in slot_map or 1 not in slot_map:
                continue

            c0 = int(normalized[edge, 0])
            c1 = int(normalized[edge, 1])
            if c0 < 0 or c1 < 0:
                continue
            if c0 >= self.cell_to_ijk.shape[0] or c1 >= self.cell_to_ijk.shape[0]:
                continue

            ci0, cj0, ck0 = (int(v) for v in self.cell_to_ijk[c0])
            ci1, cj1, ck1 = (int(v) for v in self.cell_to_ijk[c1])
            if min(ci0, cj0, ck0, ci1, cj1, ck1) < 0:
                continue

            rel0 = (ci0 - ei, cj0 - ej, ck0)
            rel1 = (ci1 - ei, cj1 - ej, ck1)
            exp0 = tuple(int(v) for v in slot_map[0])
            exp1 = tuple(int(v) for v in slot_map[1])

            if rel0 == exp1 and rel1 == exp0:
                normalized[edge, 0], normalized[edge, 1] = c1, c0

        return normalized

    def _build_structured_offset_provider(self, offset_provider):
        if not offset_provider:
            return offset_provider

        structured_offset_provider = dict(offset_provider)
        for key, value in offset_provider.items():
            key_name = getattr(key, "value", str(key))
            # Only replace sparse local-connectivity arrays that appear in the remap table,
            # to avoid corrupting point-to-point connectivities (E2V, V2E, E2C, C2E, etc.)
            # which the structured backend does not need replaced.
            if key_name not in _SPARSE_REMAP_TABLE:
                continue
            sanitized = self._sanitized_conn.get(key_name)
            if sanitized is None:
                continue
            structured_offset_provider[key] = gtx.as_connectivity(
                value.domain.dims,
                value.codomain,
                data=sanitized,
                dtype=gtx.int32,
                skip_value=None,
                allocator=self.allocator,
            )
        return structured_offset_provider

    def _reference_tangential_wind(self, vn: np.ndarray, coeff: np.ndarray, e2c2e: np.ndarray) -> np.ndarray:
        vt_ref = np.zeros((vn.shape[0], vn.shape[1]), dtype=vn.dtype)
        for edge in range(e2c2e.shape[0]):
            for local in range(e2c2e.shape[1]):
                neighbor = int(e2c2e[edge, local])
                if neighbor >= 0:
                    vt_ref[edge, :] += vn[neighbor, :] * coeff[edge, local]
        return vt_ref

    def _print_tangential_wind_debug(
        self,
        vn: np.ndarray,
        coeff: np.ndarray,
        vt_out: np.ndarray,
        e2c2e_raw: np.ndarray,
        e2c2e_effective: np.ndarray | None,
    ) -> None:
        compare_sanitized = os.environ.get("GT4PY_STRUCTURED_DEBUG_COMPARE_SANITIZED", "0") == "1"
        vt_ref_raw = self._reference_tangential_wind(vn, coeff, e2c2e_raw)
        vt_ref_eff = (
            self._reference_tangential_wind(vn, coeff, e2c2e_effective)
            if compare_sanitized and e2c2e_effective is not None
            else None
        )

        print(f"[structured-debug] vt_out: {vt_out}")
        print(f"[structured-debug] vt_ref_raw: {vt_ref_raw}")
        if vt_ref_eff is not None:
            print(f"[structured-debug] vt_ref_effective: {vt_ref_eff}")

        abs_diff_raw = np.abs(vt_out - vt_ref_raw)
        max_abs_raw = float(np.nanmax(abs_diff_raw)) if abs_diff_raw.size else 0.0
        mismatched_raw = int(np.count_nonzero((abs_diff_raw > 1e-9) | np.isnan(abs_diff_raw)))

        if vt_ref_eff is not None:
            abs_diff_eff = np.abs(vt_out - vt_ref_eff)
            max_abs_eff = float(np.nanmax(abs_diff_eff)) if abs_diff_eff.size else 0.0
            mismatched_eff = int(np.count_nonzero((abs_diff_eff > 1e-9) | np.isnan(abs_diff_eff)))
        else:
            abs_diff_eff = None
            max_abs_eff = float("nan")
            mismatched_eff = -1

        total = int(abs_diff_raw.size)
        if vt_ref_eff is not None:
            print(
                f"[structured-debug] operator={self.operator_name} "
                f"raw(max_abs={max_abs_raw:.6e}, mismatched={mismatched_raw}/{total}) "
                f"effective(max_abs={max_abs_eff:.6e}, mismatched={mismatched_eff}/{total})"
            )
        else:
            print(
                f"[structured-debug] operator={self.operator_name} "
                f"raw(max_abs={max_abs_raw:.6e}, mismatched={mismatched_raw}/{total})"
            )

        # Raw connectivity is the canonical reference for this debug path; sanitized
        # comparison is optional and only reported when explicitly enabled.
        abs_diff = abs_diff_raw
        vt_ref = vt_ref_raw
        e2c2e = e2c2e_raw
        print("[structured-debug] detail_reference=raw")

        edge_scores = abs_diff.max(axis=1)
        color_mismatch = {0: 0, 1: 0, 2: 0}
        for edge in range(edge_scores.shape[0]):
            if edge_scores[edge] > 1e-9:
                color = int(self.index_map.edge_to_ijk[edge, 2])
                color_mismatch[color] = color_mismatch.get(color, 0) + 1
        # print(
        #     "[structured-debug] mismatch_by_kolor="
        #     f"{color_mismatch}"
        # )

        top_edges = np.argsort(edge_scores)[-5:][::-1]
        for edge in top_edges:
            if edge_scores[edge] <= 1e-9:
                continue
            center_ijk = tuple(int(v) for v in self.index_map.edge_to_ijk[edge])
            neighbors = [int(v) for v in e2c2e[edge]]
            neighbor_ijk = [tuple(int(v) for v in self.index_map.edge_to_ijk[n]) if n >= 0 else (-1, -1, -1) for n in neighbors]
            print(
                f"[structured-debug] edge={int(edge)} center_ijk={center_ijk} "
                f"out_k0={float(vt_out[edge, 0]):+.6e} ref_k0={float(vt_ref[edge, 0]):+.6e} "
                f"coeff={coeff[edge].tolist()} neighbors={neighbors} neighbor_ijk={neighbor_ijk}"
            )

    def _is_unstructured(self, field, axis_name):
        if not getattr(field, "domain", None):
            return False
        return any(d.value == axis_name for d in field.domain.dims)

    def _get_local_dim_name(self, field) -> str | None:
        """Return the local-connectivity dimension name (e.g. 'E2C2E') from a sparse-local field, or None."""
        dims = list(getattr(field.domain, "dims", ()))
        if len(dims) < 2:
            return None
        return getattr(dims[1], "value", None)

    def _is_sparse_local_field(self, field, np_data: np.ndarray) -> bool:
        """Return True if the field is a sparse local-connectivity field whose connectivity is
        known in the map_dict remap table and for which we have the raw connectivity array."""
        if np_data.ndim < 2:
            return False
        local_dim = self._get_local_dim_name(field)
        if local_dim is None:
            return False
        if local_dim not in _SPARSE_REMAP_TABLE:
            return False
        # Must have a connectivity array available
        return local_dim in self._raw_conn

    def _pack_sparse_local_field(self, field, coeff: np.ndarray) -> np.ndarray:
        """Generic packing of a sparse local-connectivity field into a structured
        (ni, nj, n_kolor, n_local, ...) array, derived from the map_dict remap table.

        Works for any connectivity whose name appears as a key in _SPARSE_REMAP_TABLE
        (e.g. E2C2E, E2C2EO, E2C, C2E, V2E, E2C2V, …).
        """
        local_dim = self._get_local_dim_name(field)
        conn = self._raw_conn.get(local_dim)
        if conn is None:
            ni, nj, n_kolor = self.index_map.ijk_to_edge.shape
            n_local = coeff.shape[1]
            tail_shape = coeff.shape[2:]
            if np.issubdtype(coeff.dtype, np.integer):
                return np.full((ni, nj, n_kolor, n_local, *tail_shape), -1, dtype=coeff.dtype)
            return np.zeros((ni, nj, n_kolor, n_local, *tail_shape), dtype=coeff.dtype)

        # Use precomputed index arrays if available — avoids Python loops on every call.
        # Build once on first use (slow), then all subsequent calls are pure numpy.
        if local_dim not in self._sparse_pack_mappings:
            self._sparse_pack_mappings[local_dim] = precompute_sparse_pack_mapping(
                conn=conn,
                index_map=self.index_map,
                local_dim_name=local_dim,
                cell_to_ijk=self.cell_to_ijk,
            )

        mapping = self._sparse_pack_mappings.get(local_dim)
        ni, nj, n_kolor = self.index_map.ijk_to_edge.shape
        n_local = coeff.shape[1]
        tail_shape = coeff.shape[2:]
        out_shape = (ni, nj, n_kolor, n_local, *tail_shape)
        if mapping is not None and tail_shape == ():
            # Fast path: pure numpy indexing using precomputed index arrays.
            packed = apply_sparse_pack_mapping(coeff, mapping, out_shape)
        else:
            # Slow fallback for unusual shapes or first-call precompute failure
            packed = pack_sparse_local_field_to_structured(
                coeff=coeff,
                connectivity=conn,
                index_map=self.index_map,
                local_dim_name=local_dim,
                cell_to_ijk=self.cell_to_ijk,
            )

        # Default OFF: zeroing clipped sparse coefficients was causing false mismatches
        # for divergence-damping stencils near edge-shape boundaries.
        if local_dim in {"E2C2E", "E2C2EO"} and os.environ.get("GT4PY_TRANSLATOR_ZERO_CLIPPED_SPARSE", "0") == "1":
            # Keep sparse coefficients zero on clipped edge-shape center lines.
            ni, nj, nk = packed.shape[:3]
            i_idx = np.arange(ni, dtype=np.int32)[:, None, None]
            j_idx = np.arange(nj, dtype=np.int32)[None, :, None]
            k_idx = np.arange(nk, dtype=np.int32)[None, None, :]
            invalid_center = np.zeros((ni, nj, nk), dtype=bool)
            invalid_center |= np.isin(k_idx, (1, 2)) & (i_idx >= (self.remap_sizes.end_i - 1))
            invalid_center |= np.isin(k_idx, (0, 2)) & (j_idx >= (self.remap_sizes.end_j - 1))
            packed[invalid_center, ...] = 0

        return packed

    def _pack_argument(self, field):
        if not getattr(field, "domain", None):
            return field

        np_data = field.asnumpy()

        if self._is_sparse_local_field(field, np_data):
            local_dim = field.domain.dims[1]
            struct_np = self._pack_sparse_local_field(field, np_data)
            trailing_dims = list(field.domain.dims[2:]) if np_data.ndim > 2 else []
            return gtx.as_field(
                [IDim, JDim, Kolor, local_dim, *trailing_dims],
                struct_np,
                allocator=self.allocator,
            )

        if self._is_unstructured(field, "Vertex"):
            struct_np = pack_vertex_field_to_structured(np_data, self.index_map)
            trailing_dims = list(field.domain.dims[1:]) if np_data.ndim > 1 else []
            return gtx.as_field([IDim, JDim, Kolor, *trailing_dims], struct_np, allocator=self.allocator)

        elif self._is_unstructured(field, "Edge"):
            struct_np = pack_edge_field(np_data, self.index_map)
            trailing_dims = list(field.domain.dims[1:]) if np_data.ndim > 1 else []
            return gtx.as_field([IDim, JDim, Kolor, *trailing_dims], struct_np, allocator=self.allocator)

        elif self._is_unstructured(field, "Cell"):
            if self.cell_to_ijk is None or self.ijk_to_cell is None:
                return field
            struct_np = pack_cell_field_to_structured(np_data, self.cell_to_ijk, self.ijk_to_cell)
            trailing_dims = list(field.domain.dims[1:]) if np_data.ndim > 1 else []
            return gtx.as_field([IDim, JDim, Kolor, *trailing_dims], struct_np, allocator=self.allocator)

        return field

    def _unpack_to_buffer(self, structured_field, original_unstructured_field):
        # print(f"unpacking field:", structured_field, "to", original_unstructured_field)
        if not getattr(original_unstructured_field, "domain", None):
            return

        struct_np = structured_field.asnumpy()
        orig_np = original_unstructured_field.asnumpy()

        # Sparse local-connectivity inputs (e.g. [Edge, E2C2E]) are read-only coefficients
        # and must not be overwritten during unpack. Doing so mutates benchmark inputs.
        if self._is_sparse_local_field(original_unstructured_field, orig_np):
            return

        # Vertex-local sparse coefficient inputs (e.g. [Vertex, V2E]) are read-only.
        # Do not skip regular vertex outputs like [Vertex, K].
        local_dim_name = self._get_local_dim_name(original_unstructured_field)
        if (
            self._is_unstructured(original_unstructured_field, "Vertex")
            and orig_np.ndim > 1
            and local_dim_name in _SPARSE_REMAP_TABLE
        ):
            return
        
        if self._is_unstructured(original_unstructured_field, "Vertex"):
            unstruct_np = unpack_vertex_field_to_unstructured(struct_np, self.index_map)
        elif self._is_unstructured(original_unstructured_field, "Edge"):
            unstruct_np = unpack_edge_field(struct_np, self.index_map, orig_np.shape[0])
        elif self._is_unstructured(original_unstructured_field, "Cell"):
            if self.cell_to_ijk is None:
                return
            unstruct_np = unpack_cell_field_from_structured(struct_np, self.cell_to_ijk, orig_np.shape[0])
        else:
            return 

        np.copyto(orig_np, unstruct_np)
    
    def __call__(self, *args, **kwargs):
        # Map positional args to kwargs using the program's declared param order.
        # This handles callers (like diffusion init_run) that pass some args positionally.
        if args:
            try:
                program_params = self._operator.past_stage.past_node.params
                param_names = [str(p.id) for p in program_params]
                for i, arg in enumerate(args):
                    if i < len(param_names) and param_names[i] not in kwargs:
                        kwargs[param_names[i]] = arg
            except AttributeError:
                pass

        structured_kwargs = {}
        packed_fields: list[tuple[object, object]] = []

        _pack_start = time.perf_counter()
        for arg_name, arg_val in kwargs.items():
            if arg_name == "offset_provider":
                continue
            if isinstance(arg_val, tuple):
                packed_tuple = tuple(self._pack_argument(f) for f in arg_val)
                structured_kwargs[arg_name] = packed_tuple
                for original_field, packed_field in zip(arg_val, packed_tuple, strict=False):
                    if getattr(original_field, "domain", None) is not None:
                        packed_fields.append((original_field, packed_field))
            else:
                # print(f"Packing argument '{arg_name}' for operator '{self.operator_name}'")
                packed_arg = self._pack_argument(arg_val)
                structured_kwargs[arg_name] = packed_arg
                if getattr(arg_val, "domain", None) is not None:
                    packed_fields.append((arg_val, packed_arg))
        _pack_end = time.perf_counter()
        _pack_elapsed = _pack_end - _pack_start

        # Determine horizontal_start for lazy compilation (use 0 if not present).
        hs_raw = (
            kwargs.get("horizontal_start_edge")
            or kwargs.get("horizontal_start_e")
            or kwargs.get("horizontal_start")
            or 0
        )
        horizontal_start = int(hs_raw) if hs_raw is not None else 0

        # Collect known threshold parameters (edge and cell) so per-kolor bounds can be computed.
        _all_threshold_params = _KNOWN_EDGE_THRESHOLD_PARAMS | _KNOWN_CELL_THRESHOLD_PARAMS
        extra_thresholds = tuple(sorted(
            (name, int(val))
            for name, val in kwargs.items()
            if name in _all_threshold_params
            and isinstance(val, (int, np.integer))
            and int(val) > 0
        ))
        compiled = self._get_or_compile(horizontal_start, extra_thresholds)

        # Re-set the ContextVar so DaCe pool-thread compilation sees the correct
        # symbolic_domain_sizes. ContextVar is inherited by ThreadPoolExecutor threads
        # (via contextvars.copy_context), unlike threading.local.
        # Set global so the DaCe pool-thread compilation can read it.
        # The main thread is blocked inside _compiled_programs() while the pool thread
        # compiles, so this global is stable for the duration — no race condition.
        global _CURRENT_COMPILE_SDS
        _call_sds = self._sds_cache.get((horizontal_start, extra_thresholds))
        if _call_sds is not None:
            _CURRENT_COMPILE_SDS = _call_sds
        try:
            _exec_start = time.perf_counter()
            if isinstance(compiled, functools.partial) and hasattr(compiled.func, "_compiled_programs"):
                bound_kwargs = dict(compiled.keywords or {})
                offset_provider = bound_kwargs.pop("offset_provider", self.structured_offset_provider)
                enable_jit = bound_kwargs.pop("enable_jit", None)

                merged_kwargs = {**bound_kwargs, **structured_kwargs}

                params = getattr(compiled.func.past_stage.past_node, "params", ())
                ordered_args = []
                for param in params:
                    name = str(param.id)
                    if name in merged_kwargs:
                        ordered_args.append(merged_kwargs.pop(name))

                compiled.func._compiled_programs(
                    *ordered_args,
                    offset_provider=offset_provider,
                    enable_jit=enable_jit,
                )
            else:
                # Fallback for non-partial wrappers.
                compiled(**structured_kwargs)
            _exec_end = time.perf_counter()
            _exec_elapsed = _exec_end - _exec_start
        finally:
            if _call_sds is not None:
                _CURRENT_COMPILE_SDS = None

        _unpack_start = time.perf_counter()
        for original_field, packed_field in packed_fields:
            self._unpack_to_buffer(packed_field, original_field)
        _unpack_end = time.perf_counter()
        _unpack_elapsed = _unpack_end - _unpack_start

        # Print timing summary
        print(f"[timing] {self.operator_name} pack={_pack_elapsed:.8f}s exec={_exec_elapsed:.8f}s unpack={_unpack_elapsed:.8f}s total={_pack_elapsed + _exec_elapsed + _unpack_elapsed:.8f}s")

        # if (
        #     debug_tangential
        #     and debug_vn is not None
        #     and debug_coeff is not None
        #     and debug_e2c2e_raw is not None
        # ):
        #         vt_out = kwargs["vt"].asnumpy()
        #         self._print_tangential_wind_debug(
        #             vn=debug_vn,
        #             coeff=debug_coeff,
        #             vt_out=vt_out,
        #             e2c2e_raw=debug_e2c2e_raw,
        #             e2c2e_effective=debug_e2c2e_effective,
        #         )