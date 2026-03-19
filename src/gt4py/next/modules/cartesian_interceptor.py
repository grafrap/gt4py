import os
import xarray as xr
import gt4py as gtx
import numpy as np
from icon4py.model.common.dimension import IDim, JDim, Kolor

from gt4py.next.modules.translator import (
    load_structured_remap_sizes_from_netcdf,
    build_index_map_from_lonlat_e2v,
    build_structured_sign_from_unstructured,
    pack_vertex_field_to_structured,
    pack_edge_field_to_structured,
    unpack_vertex_field_to_unstructured,
    unpack_edge_field,
    _read_e2v,
    _read_lonlat
)

_CACHED_INDEX_MAP = None
_CACHED_REMAP_SIZES = None

def get_global_grid_mapping():
    """Builds or returns the cached index_map and remap_sizes for the current run."""
    global _CACHED_INDEX_MAP, _CACHED_REMAP_SIZES
    
    if _CACHED_INDEX_MAP is not None:
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
        remap_sizes = load_structured_remap_sizes_from_netcdf(mesh_nc)
        
    index_map = build_index_map_from_lonlat_e2v(lonlat, e2v, ...) # Add your exact sizes here
    
    _CACHED_INDEX_MAP = index_map
    _CACHED_REMAP_SIZES = remap_sizes
    
    return _CACHED_INDEX_MAP, _CACHED_REMAP_SIZES

class GenericStructuredWrapper:
    def __init__(self, operator, backend_factory, index_map, remap_sizes, allocator, offset_provider):
        self.index_map = index_map
        self.allocator = allocator
        
        # 1. Dynamically extract connectivities from the offset_provider!
        # Tests will pass standard offset_providers like {"V2E": v2e_field, "E2V": e2v_field}
        self.v2e_conn = offset_provider.get("V2E").asnumpy() if "V2E" in offset_provider else None
        self.e2v_conn = offset_provider.get("E2V").asnumpy() if "E2V" in offset_provider else None
        
        # 2. Instantiate the structured backend dynamically using the remap_sizes
        structured_backend = backend_factory(
            cached=True,
            otf_workflow__cached_translation=True,
            otf_workflow__bare_translation__symbolic_domain_sizes={
                "max_i": int(remap_sizes.max_i),
                "max_j": int(remap_sizes.max_j),
            },
        )

        # 3. Compile the actual program
        from gt4py.next.program_processors.program_setup_utils import setup_program as original_setup
        self._compiled_program = original_setup(
            operator,
            backend=structured_backend,
            offset_provider=offset_provider
        )

    def _is_unstructured(self, field, axis_name):
        if not getattr(field, "domain", None):
            return False
        return any(d.value == axis_name for d in field.domain.dims)

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
            
        # 2. Standard unstructured fields
        if self._is_unstructured(field, "Vertex"):
            struct_np = pack_vertex_field_to_structured(np_data, self.index_map)
            return gtx.as_field([IDim, JDim, Kolor], struct_np, allocator=self.allocator)
            
        elif self._is_unstructured(field, "Edge"):
            struct_np = pack_edge_field_to_structured(np_data, self.index_map)
            return gtx.as_field([IDim, JDim, Kolor], struct_np, allocator=self.allocator)

        return field 

    def _unpack_to_buffer(self, structured_field, original_unstructured_field):
        if not getattr(original_unstructured_field, "domain", None):
            return

        struct_np = structured_field.asnumpy()
        orig_np = original_unstructured_field.asnumpy()
        
        if self._is_unstructured(original_unstructured_field, "Vertex"):
            unstruct_np = unpack_vertex_field_to_unstructured(struct_np, self.index_map)
        elif self._is_unstructured(original_unstructured_field, "Edge"):
            unstruct_np = unpack_edge_field(struct_np, self.index_map, orig_np.shape[0])
        else:
            return 

        np.copyto(orig_np, unstruct_np)
    
    def __call__(self, **kwargs):
        structured_kwargs = {}
        out_fields = []

        for arg_name, arg_val in kwargs.items():
            if arg_name == "out":
                if isinstance(arg_val, tuple):
                    out_fields = list(arg_val)
                    structured_kwargs[arg_name] = tuple(self._pack_argument(f) for f in arg_val)
                else:
                    out_fields = [arg_val]
                    structured_kwargs[arg_name] = self._pack_argument(arg_val)
            else:
                structured_kwargs[arg_name] = self._pack_argument(arg_val)

        self._compiled_program(**structured_kwargs)

        if isinstance(kwargs["out"], tuple):
            for orig_f, struct_f in zip(out_fields, structured_kwargs["out"]):
                self._unpack_to_buffer(struct_f, orig_f)
        else:
            self._unpack_to_buffer(structured_kwargs["out"], out_fields[0])