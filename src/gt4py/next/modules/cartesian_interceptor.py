import os
import functools
import xarray as xr
import gt4py.next as gtx
import numpy as np
from icon4py.model.common.dimension import IDim, JDim, Kolor

from gt4py.next.modules.translator import (
    IndexMap,
    StructuredRemapSizes,
    load_structured_remap_sizes_from_netcdf,
    build_index_map_from_lonlat_e2v,
    build_structured_sign_from_unstructured,
    pack_vertex_field_to_structured,
    pack_edge_field_to_structured,
    pack_edge_field,
    unpack_vertex_field_to_unstructured,
    unpack_edge_field,
    _read_e2v,
    _read_lonlat
)

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

    print(f"lateral={remap_sizes.lateral}, max_i={remap_sizes.max_i}, max_j={remap_sizes.max_j}")        
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
        self.max_i = int(remap_sizes.max_i)
        self.max_j = int(remap_sizes.max_j)
        
        # 1. Dynamically extract connectivities from the offset_provider!
        # Tests will pass standard offset_providers like {"V2E": v2e_field, "E2V": e2v_field}
        self.v2e_conn = offset_provider.get("V2E").asnumpy() if "V2E" in offset_provider else None
        self.e2v_conn = offset_provider.get("E2V").asnumpy() if "E2V" in offset_provider else None
        self.e2c2e_conn_raw = self._get_connectivity(offset_provider, "E2C2E")
        self.e2c2e_conn = self._sanitize_sparse_connectivity(self.e2c2e_conn_raw)
        self.structured_offset_provider = self._build_structured_offset_provider(offset_provider)

        # 2. Instantiate the structured backend dynamically using the remap_sizes
        structured_backend = backend_factory(
            cached=True,
            otf_workflow__cached_translation=True,
            otf_workflow__bare_translation__symbolic_domain_sizes={
                "max_i": int(remap_sizes.max_i),
                "max_j": int(remap_sizes.max_j),
                "lateral": int(remap_sizes.lateral)
            },
        )

        # 3. Compile the actual program
        from gt4py.next.program_processors.program_setup_utils import setup_program as original_setup
        self._compiled_program = original_setup(
            operator,
            backend=structured_backend,
            offset_provider=self.structured_offset_provider
        )

    def _get_connectivity(self, offset_provider, name: str):
        if not offset_provider:
            return None
        for key, value in offset_provider.items():
            key_name = getattr(key, "value", str(key))
            if key_name == name:
                return value.asnumpy()
        return None

    def _sanitize_sparse_connectivity(self, conn: np.ndarray | None) -> np.ndarray | None:
        if conn is None:
            return None
        sanitized = np.array(conn, copy=True)
        invalid_mask = sanitized < 0
        if not invalid_mask.any():
            return sanitized

        row_max = sanitized.max(axis=1, keepdims=True)
        # Keep fully-invalid rows untouched; fill partial invalid entries with last valid neighbor.
        has_valid = row_max >= 0
        fill_values = np.where(has_valid, 0, sanitized)
        # print(f"[structured-debug] fill values: {fill_values}")
        sanitized = np.where(invalid_mask, fill_values, sanitized)
        return sanitized

    def _build_structured_offset_provider(self, offset_provider):
        if not offset_provider or self.e2c2e_conn is None:
            return offset_provider

        structured_offset_provider = dict(offset_provider)
        for key, value in offset_provider.items():
            key_name = getattr(key, "value", str(key))
            if key_name != "E2C2E":
                continue
            structured_offset_provider[key] = gtx.as_connectivity(
                value.domain.dims,
                value.codomain,
                data=self.e2c2e_conn,
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

    def _is_edge_sparse_e2c2e(self, field, np_data: np.ndarray) -> bool:
        if not self._is_unstructured(field, "Edge") or np_data.ndim != 2:
            return False
        dims = list(getattr(field.domain, "dims", ()))
        return len(dims) >= 2 and getattr(dims[1], "value", None) == "E2C2E"

    def _pack_edge_sparse_e2c2e(self, coeff: np.ndarray) -> np.ndarray:
        ni, nj, n_kolor = self.index_map.ijk_to_edge.shape
        n_local = coeff.shape[1]
        out = np.zeros((ni, nj, n_kolor, n_local), dtype=coeff.dtype)
        conn = self.e2c2e_conn_raw if self.e2c2e_conn_raw is not None else self.e2c2e_conn
        if conn is None:
            return out

        # Canonical E2C2E mapping derived from iterator/transforms/map_dict.py.
        # Relation tuple is (delta_i, delta_j, neighbor_kolor).
        expected_rel: dict[int, dict[int, tuple[tuple[int, int, int], ...]]] = {
            0: {
                0: ((0, 0, 2),),
                1: ((0, 0, 1),),
                2: ((-1, 0, 2),),
                3: ((-1, 1, 1),),
            },
            1: {
                0: ((0, 0, 0),),
                1: ((0, 0, 2),),
                2: ((1, -1, 0),),
                3: ((0, -1, 2),),
            },
            2: {
                0: ((0, 0, 1),),
                1: ((0, 0, 0),),
                2: ((0, 1, 1),),
                3: ((1, 0, 0),),
            },
        }
        
        slot_by_rel = {
            kolor: {
                rel: tuple(
                    slot
                    for slot, rels in rel_map.items()
                    if rel in rels
                )
                for rel in {r for rels in rel_map.values() for r in rels}
            }
            for kolor, rel_map in expected_rel.items()
        }

        def _normalize_shift(delta: int, period: int) -> int:
            if period <= 0:
                return delta
            half = period // 2
            if delta > half:
                return delta - period
            if delta < -half:
                return delta + period
            return delta

        mapped = 0
        unmatched = 0
        n_edge = min(coeff.shape[0], conn.shape[0])
        for edge in range(n_edge):
            i, j, kolor = (int(v) for v in self.index_map.edge_to_ijk[edge])
            if kolor < 0:
                continue
            rel_to_slot = slot_by_rel.get(kolor, {})
            for local in range(min(n_local, conn.shape[1])):
                neighbor = int(conn[edge, local])
                if neighbor < 0:
                    continue
                ni_, nj_, nk_ = (int(v) for v in self.index_map.edge_to_ijk[neighbor])
                rel = (
                    _normalize_shift(ni_ - i, self.max_i),
                    _normalize_shift(nj_ - j, self.max_j),
                    nk_,
                )
                slots = rel_to_slot.get(rel)
                if slots is None:
                    unmatched += 1
                    continue
                assigned = False
                for slot in slots:
                    if slot >= n_local:
                        continue
                    out[i, j, kolor, slot] = coeff[edge, local]
                    assigned = True
                if not assigned:
                    unmatched += 1
                    continue
                mapped += 1

        # if self.debug_enabled:
        #     print(
        #         f"[structured-debug] E2C2E remap mapped={mapped} unmatched={unmatched} "
        #         f"edges={n_edge} local={n_local}"
        #     )
        return out

    def _pack_argument(self, field):
        if not getattr(field, "domain", None):
            return field 

        np_data = field.asnumpy()
        
        # 1. Sparse fields (e.g., Sign: [Vertex, V2EDim])
        if self._is_unstructured(field, "Vertex") and np_data.ndim == 2:
            local_dim = field.domain.dims[1] 
            struct_np = np.stack(
                build_structured_sign_from_unstructured(np_data, self.v2e_conn, self.index_map),
                axis=-1
            )
            return gtx.as_field([IDim, JDim, Kolor, local_dim], struct_np, allocator=self.allocator)

        if self._is_edge_sparse_e2c2e(field, np_data):
            # print(f"np_data: ",np_data)
            local_dim = field.domain.dims[1]
            struct_np = self._pack_edge_sparse_e2c2e(np_data)
            # print(f"[structured-debug] e2c2e packed full-field: ", struct_np)
            return gtx.as_field([IDim, JDim, Kolor, local_dim], struct_np, allocator=self.allocator)
            
        # 2. Standard unstructured fields
        if self._is_unstructured(field, "Vertex"):
            struct_np = pack_vertex_field_to_structured(np_data, self.index_map)
            return gtx.as_field([IDim, JDim, Kolor], struct_np, allocator=self.allocator)
            
        elif self._is_unstructured(field, "Edge"):
            struct_np = pack_edge_field(np_data, self.index_map)
            trailing_dims = list(field.domain.dims[1:]) if np_data.ndim > 1 else []
            return gtx.as_field([IDim, JDim, Kolor, *trailing_dims], struct_np, allocator=self.allocator)
        
        elif self._is_unstructured(field, "Cell"): # TODO: check if this is actually correct.
            struct_np = pack_cell_field_to_structured(np_data, self.index_map)
            return gtx.as_field([IDim, JDim, Kolor], struct_np, allocator=self.allocator)

        return field 

    def _unpack_to_buffer(self, structured_field, original_unstructured_field):
        if not getattr(original_unstructured_field, "domain", None):
            return

        struct_np = structured_field.asnumpy()
        orig_np = original_unstructured_field.asnumpy()

        # Sparse local-connectivity inputs (e.g. [Edge, E2C2E]) are read-only coefficients
        # and must not be overwritten during unpack. Doing so mutates benchmark inputs.
        if self._is_edge_sparse_e2c2e(original_unstructured_field, orig_np):
            return
        
        if self._is_unstructured(original_unstructured_field, "Vertex"):
            unstruct_np = unpack_vertex_field_to_unstructured(struct_np, self.index_map)
        elif self._is_unstructured(original_unstructured_field, "Edge"):
            unstruct_np = unpack_edge_field(struct_np, self.index_map, orig_np.shape[0])
        else:
            return 

        np.copyto(orig_np, unstruct_np)
    
    def __call__(self, **kwargs):
        structured_kwargs = {}
        packed_fields: list[tuple[object, object]] = []
        runtime_offset_provider = kwargs.get("offset_provider")
        debug_tangential = self.debug_enabled and "compute_tangential_wind" in str(self.operator_name)

        debug_vn: np.ndarray | None = None
        debug_coeff: np.ndarray | None = None
        debug_e2c2e_raw: np.ndarray | None = None
        debug_e2c2e_effective: np.ndarray | None = None
        if debug_tangential and {"vn", "rbf_vec_coeff_e", "vt"}.issubset(kwargs):
            debug_vn = np.array(kwargs["vn"].asnumpy(), copy=True)
            debug_coeff = np.array(kwargs["rbf_vec_coeff_e"].asnumpy(), copy=True)
            runtime_e2c2e = self._get_connectivity(runtime_offset_provider, "E2C2E")
            if runtime_e2c2e is not None:
                debug_e2c2e_raw = np.array(runtime_e2c2e, copy=True)
                debug_e2c2e_effective = self._sanitize_sparse_connectivity(runtime_e2c2e)

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

        compiled = self._compiled_program
        if isinstance(compiled, functools.partial) and hasattr(compiled.func, "_compiled_programs"):
            bound_kwargs = dict(compiled.keywords or {})
            offset_provider = bound_kwargs.pop(
                "offset_provider", self.structured_offset_provider
            )
            enable_jit = bound_kwargs.pop("enable_jit", None)
            bound_kwargs.pop("offset_provider", None)
            call_kwargs = {**bound_kwargs, **structured_kwargs}
            compiled.func._compiled_programs(
                **call_kwargs,
                offset_provider=offset_provider,
                enable_jit=enable_jit,
            )
        else:
            self._compiled_program(**structured_kwargs)

        for original_field, packed_field in packed_fields:
            self._unpack_to_buffer(packed_field, original_field)

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